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


# def build_app():
#     css = """
#     /* ===== 全局玻璃深色主题 ===== */
#     :root {
#         --bg-main: #121212;
#         --bg-glass: rgba(255, 255, 255, 0.08);
#         --bg-glass-hover: rgba(255, 255, 255, 0.12);
#         --accent: #facc15;   /* 金色点缀 */
#         --user-bubble: #3b82f6;
#         --assistant-bubble: #10b981;
#         --text-primary: #e5e7eb;
#         --text-secondary: #9ca3af;
#         --border: rgba(255, 255, 255, 0.15);
#         --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
#     }
#
#     body {
#         background: var(--bg-main);
#         color: var(--text-primary);
#         font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
#     }
#
#     /* ===== 玻璃卡片 ===== */
#     .gradio-container {
#         backdrop-filter: blur(12px);
#         -webkit-backdrop-filter: blur(12px);
#         border-radius: 24px;
#         border: 1px solid var(--border);
#         box-shadow: var(--shadow);
#         padding: 24px;
#         margin: 24px auto;
#         max-width: 900px;
#     }
#
#     /* ===== 聊天区域 ===== */
#     #chatbot {
#         background: var(--bg-glass);
#         border-radius: 16px;
#         border: 1px solid var(--border);
#         padding: 16px;
#         height: 500px !important;
#     }
#
#     .user {
#         background: var(--user-bubble) !important;
#         color: #fff !important;
#         border-radius: 18px 18px 4px 18px !important;
#         padding: 10px 14px !important;
#         margin: 8px 0 !important;
#         max-width: 70%;
#         box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
#     }
#
#     .assistant {
#         background: var(--assistant-bubble) !important;
#         color: #fff !important;
#         border-radius: 18px 18px 18px 4px !important;
#         padding: 10px 14px !important;
#         margin: 8px 0 !important;
#         max-width: 70%;
#         box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
#     }
#
#     /* ===== 思考折叠框 ===== */
#     details {
#         border: 1px solid var(--border);
#         border-radius: 12px;
#         background: var(--bg-glass);
#         padding: 12px;
#         margin: 8px 0;
#         transition: all 0.3s ease;
#     }
#
#     details[open] {
#         background: var(--bg-glass-hover);
#         border-color: var(--accent);
#     }
#
#     summary {
#         cursor: pointer;
#         font-weight: 600;
#         color: var(--accent);
#     }
#
#     /* ===== 输入框 & 按钮 ===== */
#     input[type="text"], textarea, select {
#         background: var(--bg-glass) !important;
#         border: 1px solid var(--border) !important;
#         border-radius: 12px !important;
#         color: var(--text-primary) !important;
#         padding: 12px 16px !important;
#         transition: all 0.2s ease;
#     }
#
#     input[type="text"]:focus, textarea:focus, select:focus {
#         border-color: var(--accent) !important;
#         box-shadow: 0 0 0 2px rgba(250, 204, 21, 0.4) !important;
#     }
#
#     button {
#         border-radius: 12px !important;
#         font-weight: 600 !important;
#         transition: all 0.2s ease !important;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
#     }
#
#     button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
#     }
#
#     button:active {
#         transform: translateY(0);
#     }
#
#     /* ===== 参数玻璃卡片 ===== */
#     .gr-accordion .gr-form {
#         background: var(--bg-glass) !important;
#         border-radius: 16px !important;
#         border: 1px solid var(--border) !important;
#         padding: 16px !important;
#         margin-top: 8px !important;
#     }
#
#     /* ===== 示例卡片 ===== */
#     .gr-examples {
#         background: var(--bg-glass) !important;
#         border: 1px solid var(--border) !important;
#         border-radius: 12px !important;
#         padding: 12px !important;
#     }
#
#     .gr-examples .gr-button {
#         background: rgba(255, 255, 255, 0.1) !important;
#         border: 1px solid var(--border) !important;
#         color: var(--text-primary) !important;
#         border-radius: 8px !important;
#         margin: 4px !important;
#         transition: all 0.2s ease;
#     }
#
#     .gr-examples .gr-button:hover {
#         background: var(--accent) !important;
#         color: #000 !important;
#         border-color: var(--accent) !important;
#     }
#
#     /* ===== 音频播放器 ===== */
#     #output_audio {
#         border-radius: 16px !important;
#         border: 1px solid var(--border) !important;
#         background: var(--bg-glass) !important;
#         padding: 12px !important;
#     }
#
#     /* ===== 加载动画保留 ===== */
#     .spinner {
#         animation: spin 1s linear infinite;
#         display: inline-block;
#         margin-right: 8px;
#     }
#     @keyframes spin {
#         from { transform: rotate(0deg); }
#         to { transform: rotate(360deg); }
#     }
#
#     /* ===== 思考内容保留 ===== */
#     .thinking-container {
#         border-left: 3px solid var(--accent);
#         padding-left: 12px;
#         margin: 8px 0;
#         background: var(--bg-glass);
#         border-radius: 8px;
#     }
#
#     /* ===== 输入框文字 → 深灰色 ===== */
#     input[type="text"], textarea, select,
#     .gr-text-input input, .gr-textbox input {
#         color: #4b5563 !important;   /* 深灰 #4b5563，对比度足够但不刺眼 */
#         caret-color: var(--accent) !important;
#     }
#     """
#
#     def user(message, history):
#         if not message:
#             return "", history
#         history.append({"role": "user", "content": message})
#         history.append({"role": "assistant", "content": ""})  # 空字符串占位
#         return "", history
#
#     # 计算 assets 绝对路径
#     assets_dir = Path(__file__).parent.parent / "assets"
#     bg_path = str(assets_dir / "bg.jpg")
#
#     with gr.Blocks(css=css, title="The Moment") as demo:
#         # ① 先渲染图片（不显示，只为了拿到 URL）
#         bg_img = gr.Image(value=str(assets_dir / "bg.jpg"), visible=False, elem_id="bg_source")
#
#         active_gen = gr.State([False])
#
#         chatbot = gr.Chatbot(
#             elem_id="chatbot",
#             height=500,
#             show_label=False,
#             render_markdown=True,
#             type="messages"
#         )
#
#         with gr.Row():
#             msg = gr.Textbox(
#                 label="Message",
#                 placeholder="Type your message...",
#                 container=False,
#                 scale=4
#             )
#             submit_btn = gr.Button("Send", variant='primary', scale=1)
#
#         with gr.Row():
#             clear_btn = gr.Button("Clear", variant='secondary')
#             stop_btn = gr.Button("Stop", variant='stop')
#         with gr.Accordion("Parameters", open=False):
#             temperature = gr.Slider(
#                 minimum=0.1,
#                 maximum=1.5,
#                 value=0.6,
#                 label="Temperature"
#             )
#
#             top_p = gr.Slider(
#                 minimum=0.1,
#                 maximum=1.0,
#                 value=0.95,
#                 label="Top-p"
#             )
#
#             max_new_tokens = gr.Slider(
#                 minimum=2048,
#                 maximum=32768,
#                 value=4096,
#                 step=64,
#                 label="Max Tokens"
#             )
#
#             repetition_penalty = gr.Slider(
#                 minimum=1,
#                 maximum=1.5,
#                 value=1.2,
#                 step=0.01,
#                 label="Repetition Penalty"
#             )
#
#         gr.Examples(
#             examples=[
#                 ["你是谁呀"],
#                 ["我很难过，爸妈不爱我"],
#                 ["爸妈老是说我笨"]
#             ],
#             inputs=msg,  # 点击示例后，自动填充到前面的 msg 文本框
#             label="咨询例子"
#         )
#
#         output_audio = gr.Audio(label="converted voice", streaming=True, autoplay=True)
#         tts_time_display = gr.Textbox(label="TTS Conversion Time", value="0s", interactive=False)  # 只读，用户不能修改
#
#         text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)
#         default_audio_select = list(text_to_audio_mappings.keys())[0] if text_to_audio_mappings else ""
#         default_ref_text = default_audio_select
#         default_prompt_language = "zh"
#         default_text_language = "zh"
#         default_how_to_cut = "按标点符号切"
#
#         # 使用包装函数
#         submit_event = submit_btn.click(
#             user, [msg, chatbot], [msg, chatbot], queue=False
#         ).then(
#             lambda: [True], outputs=active_gen
#         ).then(
#             generate_wrapper,
#             [
#                 chatbot,
#                 temperature,
#                 top_p,
#                 max_new_tokens,
#                 repetition_penalty,
#                 active_gen,
#                 gr.State(default_audio_select),
#                 gr.State(default_ref_text),
#                 gr.State(default_prompt_language),
#                 gr.State(default_text_language),
#                 gr.State(default_how_to_cut)
#             ],
#             [
#                 chatbot,
#                 output_audio,
#                 tts_time_display
#             ]
#         )
#
#         stop_btn.click(
#             lambda: [False], None, active_gen, cancels=[submit_event]
#         )
#
#         clear_btn.click(
#             lambda: (None, None, "0s"), None, [chatbot, output_audio, tts_time_display], queue=False
#         ).then(
#             lambda: [False], None, active_gen, cancels=[submit_event]
#         )
#
#         stop_btn.click(
#             lambda: [False], None, active_gen, cancels=[submit_event]
#         )
#
#         clear_btn.click(
#             lambda: (None, None, "0s"), None, [chatbot, output_audio, tts_time_display], queue=False
#         ).then(
#             lambda: [False], None, active_gen, cancels=[submit_event]
#         )
#
#         # ③ 页面加载后把 URL 写进 CSS 变量
#         gr.HTML(
#             """
#             <script>
#             window.addEventListener('load', function () {
#                 const img = document.getElementById('bg_source');
#                 if (img && img.src) {
#                     document.documentElement.style.setProperty('--bg-url', 'url(' + img.src + ')');
#                 }
#             });
#             </script>
#             """,
#             visible=False
#         )
#
#     return demo

import gradio as gr
from pathlib import Path
import base64


def build_app():
    # 计算 assets 绝对路径
    assets_dir = Path(__file__).parent.parent / "assets"
    bg_path = assets_dir / "bg.jpg"

    # 基础CSS
    base_css = """
    /* ===== 全局玻璃深色主题 ===== */
    :root {
        --bg-main: #121212;
        --bg-glass: rgba(255, 255, 255, 0.08);
        --bg-glass-hover: rgba(255, 255, 255, 0.12);
        --accent: #facc15;
        --user-bubble: #3b82f6;
        --assistant-bubble: #10b981;
        --text-primary: #e5e7eb;
        --text-secondary: #9ca3af;
        --border: rgba(255, 255, 255, 0.15);
        --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    body {
        background: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        margin: 0;
        padding: 0;
        min-height: 100vh;
    }

    .gradio-container {
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        padding: 24px;
        margin: 24px auto;
        max-width: 900px;
    }

    /* ===== 聊天区域 ===== */
    #chatbot {
        border-radius: 16px;
        border: 1px solid var(--border);
        padding: 16px;
        height: 500px !important;
        position: relative;
        overflow: hidden;
    }

    /* 其他CSS样式保持不变... */
    """

    # 添加聊天框背景图片
    if bg_path.exists():
        try:
            with open(bg_path, "rb") as image_file:
                bg_base64 = base64.b64encode(image_file.read()).decode()

            bg_css = f"""
            /* ===== 聊天框背景图片 - 自适应显示 ===== */
            #chatbot {{
                background: 
                    linear-gradient(rgba(18, 18, 18, 0.6), rgba(18, 18, 18, 0.6)),
                    url("data:image/jpeg;base64,{bg_base64}") !important;
                background-size: contain !important;      /* 完整显示图片 */
                background-position: center center !important;
                background-repeat: no-repeat !important;
                background-attachment: local !important;
            }}

            /* 确保聊天内容可读性 */
            #chatbot .gr-panel {{
                background: transparent !important;
            }}

            .user, .assistant {{
                background: rgba(255, 255, 255, 0.15) !important;
                backdrop-filter: blur(12px) !important;
                -webkit-backdrop-filter: blur(12px) !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
            }}

            .user {{
                background: rgba(59, 130, 246, 0.8) !important;
            }}

            .assistant {{
                background: rgba(16, 185, 129, 0.8) !important;
            }}
            """
            css = base_css + bg_css
            print("✅ 聊天框背景图片加载成功 - 自适应模式")
        except Exception as e:
            css = base_css
            print(f"❌ 聊天框背景图片加载失败: {e}")
    else:
        fallback_bg = """
        #chatbot {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        """
        css = base_css + fallback_bg
        print(f"❌ 背景图片不存在，使用备用背景")

    # 其余代码保持不变...
    def user(message, history):
        if not message:
            return "", history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        return "", history

    with gr.Blocks(css=css, title="The Moment") as demo:
        active_gen = gr.State([False])

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            height=500,
            show_label=False,
            render_markdown=True,
            type="messages"
        )

        # 其余组件定义...
        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message...",
                container=False,
                scale=4
            )
            submit_btn = gr.Button("Send", variant='primary', scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear", variant='secondary')
            stop_btn = gr.Button("Stop", variant='stop')

        with gr.Accordion("Parameters", open=False):
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.6,
                label="Temperature"
            )

            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.95,
                label="Top-p"
            )

            max_new_tokens = gr.Slider(
                minimum=2048,
                maximum=32768,
                value=4096,
                step=64,
                label="Max Tokens"
            )

            repetition_penalty = gr.Slider(
                minimum=1,
                maximum=1.5,
                value=1.2,
                step=0.01,
                label="Repetition Penalty"
            )

        gr.Examples(
            examples=[
                ["你是谁呀"],
                ["我很难过，爸妈不爱我"],
                ["爸妈老是说我笨"]
            ],
            inputs=msg,
            label="咨询例子"
        )

        output_audio = gr.Audio(label="converted voice", streaming=True, autoplay=True)
        tts_time_display = gr.Textbox(label="TTS Conversion Time", value="0s", interactive=False)

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
                chatbot,
                temperature,
                top_p,
                max_new_tokens,
                repetition_penalty,
                active_gen,
                gr.State(default_audio_select),
                gr.State(default_ref_text),
                gr.State(default_prompt_language),
                gr.State(default_text_language),
                gr.State(default_how_to_cut)
            ],
            [
                chatbot,
                output_audio,
                tts_time_display
            ]
        )

        stop_btn.click(
            lambda: [False], None, active_gen, cancels=[submit_event]
        )

        clear_btn.click(
            lambda: (None, None, "0s"), None, [chatbot, output_audio, tts_time_display], queue=False
        ).then(
            lambda: [False], None, active_gen, cancels=[submit_event]
        )

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
