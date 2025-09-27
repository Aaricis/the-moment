import json
import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])  # 添加包搜索路径
import textwrap

import gradio as gr
import requests
import torch
from dotenv import load_dotenv
from gradio import ChatMessage
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.configs.tts_config import audio_path, slicer_list
from src.utils.loader import load_text_audio_mappings

from generate import async_chat
from src.configs.base_config import model_path
from src.configs.rag_config import prompt_template
from src.rag.pipeline import EmoLLMRAG

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
    final_chatbot = None
    final_audio = None
    final_tts_time = None

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
        final_chatbot = chatbot
        final_audio = audio
        final_tts_time = tts_time
        # 如果是中间结果，只更新chatbot
        if audio is None:
            yield chatbot, None, None

    # 返回最终结果
    yield final_chatbot, final_audio, final_tts_time


def build_app():
    css = """
    .spinner {
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 8px;
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .thinking-summary {
        cursor: pointer;
        padding: 8px;
        background: #f5f5f5;
        border-radius: 4px;
        margin: 4px 0;
    }
    .thought-content {
        padding: 10px;
        background: #f8f9fa;
        border-radius: 4px;
        margin: 5px 0;
    }
    .thinking-container {
        border-left: 3px solid #facc15;
        padding-left: 10px;
        margin: 8px 0;
        background: #ffffff;
    }
    details:not([open]) .thinking-container {
        border-left-color: #290c15;
    }
    details {
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin: 8px 0 !important;
        transition: border-color 0.2s;
    }
    """

    description = '''
    # 🧠 An AI assistant with extensive knowledge in psychology, and my name is Care.

    ## 🚀 Overview
    This model is finetuned on deepseek-r1. If this repo helps you, star and share it ❤️. This repo will be continuously merged into EmoLLM.

    ## ✨ Functions
    ✅Provide an interactive chat interface for psychological consultation seekers.

    ✅Integrate knowledge retrieval

    ✅Integrate web searching

    ✅Two customized tts (ISSUE: more voice models)

    ✅Display the consuming time of generating voice with the streaming way

    ❌Virtual mental companion

    ## ⚠️ issue status
    - 2025.4.29 fix bug of clearing and stopping op.
    - 2025.5.3 web search supports.
    - 2025.5.5 rag supports. (demo code, needs to be checked)
    - 2025.5.7 fix bug of rag.
    - 2025.5.9 tts supports.
    - 2025.5.10 two voice models.
    - 2025.5.16 merge into EmoLLM.
    - 2025.8.22 display the time of streaming voice.

    ## 🙏 Acknowledgments
    We are grateful to Modelscope for supporting this project with resources.

    The rag codes are based on [EmoLLM](https://github.com/SmartFlowAI/EmoLLM )

    ## 🤝 Contributing
    Feel free to contribute to this project via our [github repo](https://github.com/HaiyangPeng/careyou ). Grow together!
    '''

    def user(message, history):
        if not message:
            return "", history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})  # 空字符串占位
        return "", history

    with gr.Blocks(css=css) as demo:
        gr.Markdown(description)
        active_gen = gr.State([False])

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            height=500,
            show_label=False,
            render_markdown=True,
            type="messages"
        )

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
            inputs=msg,  # 点击示例后，自动填充到前面的 msg 文本框
            label="咨询例子"
        )

        output_audio = gr.Audio(label="converted voice", streaming=True, autoplay=True)
        tts_time_display = gr.Textbox(label="TTS Conversion Time", value="0s", interactive=False)  # 只读，用户不能修改

        text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)
        default_audio_select = list(text_to_audio_mappings.keys())[0] if text_to_audio_mappings else ""
        default_ref_text = default_audio_select
        default_prompt_language = "zh"
        default_text_language = "zh"
        default_how_to_cut = "不切"

        # 使用包装函数
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
