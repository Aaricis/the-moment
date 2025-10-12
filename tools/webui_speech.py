import json
import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])  # 添加包搜索路径
import textwrap

import requests
import torch
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.configs.tts_config import audio_path, slicer_list
from src.utils.loader import load_text_audio_mappings

from generate import async_chat
from src.configs.base_config import model_path
from src.configs.rag_config import prompt_template
from src.rag.pipeline import EmoLLMRAG
import gradio as gr
import base64
from pathlib import Path

from src.asr.asr import whisper_recognize

load_dotenv()  # 自动把 .env 读入环境变量

LANGSEARCH_API_URL = "https://api.langsearch.com/v1/web-search"
LANGSEARCH_API_KEY = os.getenv('LANGSEARCH_API_KEY')


def lang_search(query, max_results=5):
    """
    联网搜索
    :param query: 用户问题
    :param max_results: 返回结果数量
    :return: 搜索结果
    """
    payload = json.dumps({
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": max_results
    })

    headers = {
        "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(LANGSEARCH_API_URL, headers=headers, data=payload)
    if response.status_code == 200:
        logger.info("Response Success: 200")
        results = json.loads(response.text).get("data").get("webPages").get("value")
        search_results = []
        for result in results:
            title = result.get("name", "")
            snippet = result.get("snippet", "")
            url = result.get("url", "")
            search_results.append(f"标题：{title}\n摘要：{snippet}\n链接：{url}\n")
        return "\n".join(search_results)
    else:
        logger.error(f"Response Error: {response.status_code}")
        return ""


@torch.inference_mode()
async def generate_response_and_tts(
        history,
        temperature,
        top_p,
        max_new_tokens,
        repetition_penalty,
        active_gen,
        selected_text,
        ref_text,
        prompt_language,
        text_language,
        how_to_cut
):
    print("history len =", len(history), "history =", history)
    print(f"type = {type(history)}")

    # 获取用户消息（倒数第二条）
    user_message = history[-2]["content"]
    print(f"user_message = {user_message}")

    conversation = []
    # 遍历除最后两条之外的所有消息
    for msg in history[:-2]:
        if msg.get("content") and msg.get("content").strip():  # 确保内容不为空
            conversation.append({
                "role": msg.get("role", "user"),  # 默认角色为user
                "content": msg.get("content", "")
            })

    # 联网搜索
    searched_results = lang_search(user_message)
    if searched_results:
        logger.info(f"联网搜索结果：{searched_results}")
    else:
        logger.info("联网未搜索到准确信息")

    # 知识库搜索
    rag = EmoLLMRAG()
    retrieved_context = rag.get_retrieval_content(user_message)
    if retrieved_context:
        logger.info(f"知识库搜索结果：{retrieved_context}")
    else:
        logger.info("知识库未搜索到准确信息")

    conversation.append(
        {
            "role": "user",
            "content": textwrap.dedent(prompt_template).format(
                user_input=user_message,
                searched_results=searched_results,
                retrieved_context=retrieved_context
            )
        }
    )

    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,  # 直接输出 token
        add_generation_prompt=True,  # 在结尾加上“Assistant 开始回答”的提示，让模型进入生成模式
        return_tensors="pt"  # 直接返回 PyTorch tensor
    ).to(model.device)

    streamer = TextIteratorStreamer(  # 流式输出
        tokenizer,
        timeout=20.0,
        skip_prompt=True,  # 不返回 prompt 文本（只输出模型生成的部分）
        skip_special_tokens=True  # 滤掉特殊符号<endoftext>, <pad>
    )

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
    )

    async for history, audio, conversion_time in async_chat(
            active_gen,
            history,
            tokenizer,
            model,
            generate_kwargs,
            selected_text,
            ref_text,
            prompt_language,
            text_language,
            how_to_cut
    ):
        yield history, audio, conversion_time


# 创建一个包装函数来处理流式输出
async def generate_wrapper(
        chatbot,
        temperature,
        top_p,
        max_new_tokens,
        repetition_penalty,
        active_gen,
        selected_text,
        ref_text,
        prompt_language,
        text_language,
        how_to_cut
):
    async for chatbot, audio, tts_time in generate_response_and_tts(
            chatbot,
            temperature,
            top_p,
            max_new_tokens,
            repetition_penalty,
            active_gen,
            selected_text,
            ref_text,
            prompt_language,
            text_language,
            how_to_cut
    ):

        # 如果是中间结果，只更新chatbot
        if audio is None:
            # 文本流
            yield chatbot, None, None
        else:
            # 音频流
            yield chatbot, audio, tts_time


def build_app():
    assets_dir = Path(__file__).parent.parent / "assets"
    bg_path = str(assets_dir / "bg.jpg")

    with open(bg_path, "rb") as f:
        bg_base64 = base64.b64encode(f.read()).decode()

    css = f"""
    html, body {{
        height: 100%;
        margin: 0;
        background: linear-gradient(rgba(18,18,18,0.6), rgba(18,18,18,0.6)),
                    url("data:image/jpg;base64,{bg_base64}") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Inter', sans-serif;
        color: #e5e7eb;
    }}
    .container {{
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        background: rgba(255,255,255,0.07);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 40px rgba(0,0,0,0.4);
        padding: 28px;
        margin: 40px auto;
        max-width: 950px;
    }}
    .app-title {{
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #facc15;
        margin-bottom: 6px;
        text-shadow: 0 0 10px rgba(250,204,21,0.6);
    }}
    .subtitle {{
        text-align: center;
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 20px;
    }}
    #chatbot {{
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 18px;
        height: 520px !important;
        overflow-y: auto;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: inset 0 0 20px rgba(0,0,0,0.25);
    }}
    .section {{
        margin-top: 20px;
        margin-bottom: 12px;
    }}
    """

    def user(message, history):
        """将用户消息加入聊天历史"""
        if not message:
            return "", history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        return "", history

    with gr.Blocks(css=css, title="The Moment") as demo:
        with gr.Column(elem_classes="container"):
            # ===== 标题 =====
            gr.HTML("<div class='app-title'>✨ The Moment ✨</div>")
            gr.HTML("<div class='subtitle'>AI 情感陪伴 · 中文语音识别 + 智能对话</div>")

            # ===== 聊天窗口 =====
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=520,
                show_label=False,
                render_markdown=True,
                type="messages",
                show_copy_button=True,
            )

            # ===== 输入区：文字 + 语音 并列 =====
            gr.HTML("<div class='section'><b>💬 输入文字或语音进行交流</b></div>")
            with gr.Row(equal_height=True):
                # 左侧：文字输入框 + 发送按钮
                with gr.Column(scale=3):
                    msg = gr.Textbox(
                        placeholder="📝 输入文字内容...",
                        lines=2,
                        scale=3
                    )
                    submit_btn = gr.Button("🚀 发送文字", variant='primary')

                # 右侧：语音录制 + 识别按钮
                with gr.Column(scale=2):
                    mic = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 语音录制",
                    )
                    voice_to_text_btn = gr.Button("🎧 转文字", variant='secondary')

            # ===== 示例话题 =====
            gr.HTML("<div class='section'><b>💡 示例话题</b></div>")
            gr.Examples(
                examples=[
                    ["最近压力很大，总是睡不好，该怎么办？"],
                    ["我和父母总是沟通不顺，他们总觉得我不懂事。"],
                    ["我害怕失败，总觉得自己不够好。"],
                    ["我喜欢一个人，但不敢表白。"],
                    ["怎么才能让自己更有自信？"]
                ],
                inputs=msg,
            )

            # ===== 参数设置 =====
            gr.HTML("<div class='section'><b>⚙️ 模型参数设置</b></div>")
            with gr.Accordion("生成参数调整", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.1, 1.5, 0.6, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, 0.95, label="Top-p")
                with gr.Row():
                    max_new_tokens = gr.Slider(2048, 32768, 4096, step=64, label="Max Tokens")
                    repetition_penalty = gr.Slider(1, 1.5, 1.2, step=0.01, label="Repetition Penalty")

            # ===== 语音输出 =====
            gr.HTML("<div class='section'><b>🔊 语音输出</b></div>")
            with gr.Row():
                output_audio = gr.Audio(label="模型语音回应", streaming=True, autoplay=True)
                tts_time_display = gr.Textbox(label="TTS 用时", value="0s", interactive=False)

            # ===== 控制按钮 =====
            with gr.Row():
                clear_btn = gr.Button("🧹 清空对话", variant='secondary')
                stop_btn = gr.Button("⏹ 停止生成", variant='stop')

            # ===== 模型逻辑 =====
            active_gen = gr.State([False])

            text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)
            default_audio_select = list(text_to_audio_mappings.keys())[0] if text_to_audio_mappings else ""
            default_ref_text = default_audio_select
            default_prompt_language = "zh"
            default_text_language = "zh"
            default_how_to_cut = "按标点符号切"

            # === 文字输入事件 ===
            submit_event = submit_btn.click(
                user, [msg, chatbot], [msg, chatbot], queue=False
            ).then(
                lambda: [True], outputs=active_gen
            ).then(
                generate_wrapper,
                [
                    chatbot, temperature, top_p, max_new_tokens, repetition_penalty, active_gen,
                    gr.State(default_audio_select), gr.State(default_ref_text),
                    gr.State(default_prompt_language), gr.State(default_text_language),
                    gr.State(default_how_to_cut)
                ],
                [chatbot, output_audio, tts_time_display]
            )

            # === 语音识别事件 ===
            voice_to_text_btn.click(
                fn=whisper_recognize,  # 语音转文字
                inputs=mic,
                outputs=msg,
                show_progress=True
            )

            # === 控制按钮 ===
            stop_btn.click(lambda: [False], None, active_gen, cancels=[submit_event])
            clear_btn.click(lambda: (None, None, "0s"), None,
                            [chatbot, output_audio, tts_time_display],
                            queue=False).then(lambda: [False], None, active_gen,
                                              cancels=[submit_event])

    return demo


if __name__ == "__main__":
    logger.info("Loading Deepseek-R1 model...")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.use_default_system_prompt = False  # 关闭 tokenizer 自动在对话最前面追加「系统默认提示词」的行为

    app = build_app()
    app.queue(api_open=True, max_size=20, default_concurrency_limit=20).launch(server_name="0.0.0.0", server_port=7860,
                                                                               max_threads=40)
