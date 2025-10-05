import json
import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])  # 添加包搜索路径
import textwrap

import gradio as gr
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
import base64
from pathlib import Path

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

    # 转 base64
    with open(bg_path, "rb") as f:
        bg_base64 = base64.b64encode(f.read()).decode()

    css = f"""
    /* ===== 背景与全局 ===== */
    html, body {{
        height: 100%;
        margin: 0 !important;
        padding: 0 !important;
        background: linear-gradient(rgba(18,18,18,0.65), rgba(18,18,18,0.65)),
                    url("data:image/jpg;base64,{bg_base64}") no-repeat center center fixed !important;
        background-size: cover !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #e5e7eb;
        overflow-x: hidden;
        animation: fadeIn 1s ease-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ===== 容器玻璃化层 ===== */
    .container {{
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        background: rgba(255,255,255,0.08);
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 8px 40px rgba(0,0,0,0.45);
        padding: 32px;
        margin: 50px auto;
        max-width: 900px;
        transition: transform 0.3s ease;
    }}
    .container:hover {{
        transform: scale(1.01);
    }}

    /* ===== 标题 ===== */
    .app-title {{
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        color: #facc15;
        text-shadow: 0 0 12px rgba(250,204,21,0.6);
        margin-bottom: 20px;
        animation: glow 3s ease-in-out infinite alternate;
    }}
    @keyframes glow {{
        from {{ text-shadow: 0 0 6px rgba(250,204,21,0.4); }}
        to {{ text-shadow: 0 0 16px rgba(250,204,21,0.9); }}
    }}

    /* ===== 聊天区 ===== */
    #chatbot {{
        background: rgba(255,255,255,0.05);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.18);
        padding: 16px;
        height: 500px !important;
        overflow-y: auto !important;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.3);
        animation: fadeIn 1s ease-out;
    }}
    #chatbot::-webkit-scrollbar {{
        width: 6px;
    }}
    #chatbot::-webkit-scrollbar-thumb {{
        background: #facc15;
        border-radius: 3px;
    }}

    /* ===== 气泡动画 ===== */
    .user, .assistant {{
        opacity: 0;
        animation: bubbleIn 0.4s ease forwards;
    }}
    @keyframes bubbleIn {{
        from {{ opacity: 0; transform: translateY(10px) scale(0.95); }}
        to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}

    .user {{
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: #fff !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 12px 16px !important;
        margin: 8px 0 8px auto !important;
        max-width: 70%;
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }}

    .assistant {{
        background: linear-gradient(135deg, #10b981, #059669);
        color: #fff !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 12px 16px !important;
        margin: 8px auto 8px 0 !important;
        max-width: 70%;
        box-shadow: 0 4px 12px rgba(16,185,129,0.4);
    }}

    /* ===== 输入框 ===== */
    .gr-text-input input, .gr-textarea textarea {{
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease;
    }}
    .gr-text-input input:focus, .gr-textarea textarea:focus {{
        border-color: #facc15 !important;
        box-shadow: 0 0 0 2px rgba(250,204,21,0.4) !important;
    }}

    /* ===== 按钮 ===== */
    .gr-button {{
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        box-shadow: 0 0 0 rgba(250,204,21,0);
    }}
    .gr-button.primary {{
        background: linear-gradient(135deg, #facc15, #eab308) !important;
        color: #000 !important;
    }}
    .gr-button.secondary {{
        background: rgba(255,255,255,0.1) !important;
        color: #e5e7eb !important;
    }}
    .gr-button:hover {{
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 0 12px rgba(250,204,21,0.5);
    }}

    /* ===== 折叠面板 ===== */
    .gr-accordion {{
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 14px !important;
        margin-top: 12px !important;
    }}

    /* ===== 渐显动效 ===== */
    .fade-in {{
        animation: fadeIn 1s ease-in-out;
    }}
    """

    def user(message, history):
        if not message:
            return "", history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        return "", history

    with gr.Blocks(css=css, title="The Moment") as demo:
        with gr.Column(elem_classes="container"):
            gr.HTML("<div class='app-title'>✨ The Moment ✨</div>")

            active_gen = gr.State([False])
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=500,
                show_label=False,
                render_markdown=True,
                type="messages",
                show_copy_button=True
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message...",
                    container=False,
                    scale=4,
                    max_lines=3
                )
                submit_btn = gr.Button("Send", variant='primary', scale=1)

            with gr.Row():
                clear_btn = gr.Button("Clear", variant='secondary')
                stop_btn = gr.Button("Stop", variant='stop')

            with gr.Accordion("Parameters", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.1, 1.5, 0.6, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, 0.95, label="Top-p")
                with gr.Row():
                    max_new_tokens = gr.Slider(2048, 32768, 4096, step=64, label="Max Tokens")
                    repetition_penalty = gr.Slider(1, 1.5, 1.2, step=0.01, label="Repetition Penalty")

            gr.Examples(
                examples=[
                    ["最近压力很大，总是睡不好，该怎么办？"],
                    ["我和父母总是沟通不顺，他们总觉得我不懂事。"],
                    ["我害怕失败，总觉得自己不够好。"],
                    ["我喜欢一个人，但不敢表白。"],
                    ["怎么才能让自己更有自信？"]
                ],
                inputs=msg,
                label="💬 咨询示例（点击可快速开始对话）"
            )

            with gr.Row():
                output_audio = gr.Audio(label="Converted Voice", streaming=True, autoplay=True)
                tts_time_display = gr.Textbox(label="TTS Conversion Time", value="0s", interactive=False)

            # --- 示例逻辑（保持你的 generate_wrapper 与事件绑定）
            text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)
            default_audio_select = list(text_to_audio_mappings.keys())[0] if text_to_audio_mappings else ""
            default_ref_text = default_audio_select
            default_prompt_language = "zh"
            default_text_language = "zh"
            default_how_to_cut = "按标点符号切"

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
