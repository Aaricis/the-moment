import json
import os
import textwrap

import requests
import torch
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

import sys

sys.path.append(os.path.split(sys.path[0])[0])  # 添加包搜索路径

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
    user_message = history[-1].content

    conversation = []
    for user, assistant in history[:-1]:
        conversation.extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])

    # 联网搜索
    searched_results = lang_search(user_message)
    if searched_results:
        logger.info("联网搜索结果：", searched_results)
    else:
        logger.info("联网未搜索到准确信息")

    # 知识库搜索
    rag = EmoLLMRAG()
    retrieved_context = rag.get_retrieval_content(user_message)
    if retrieved_context:
        logger.info("知识库搜索结果：", retrieved_context)
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


if __name__ == "__main__":
    logger.info("Loading Deepseek-R1 model...")

    # 把模型权重量化为 4-bit NF4（Normal Float 4）
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.use_default_system_prompt = False  # 关闭 tokenizer 自动在对话最前面追加「系统默认提示词」的行为

    from gradio import ChatMessage

    chat_his = [
        ChatMessage(role="user", content="我很难过"),
    ]

    from src.utils.loader import load_text_audio_mappings
    from src.configs.tts_config import audio_path, slicer_list

    text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)

    DEFAULT_AUDIO_SELECT = list(text_to_audio_mappings.keys())[0] if text_to_audio_mappings else ""
    DEFAULT_REF_TEXT = DEFAULT_AUDIO_SELECT
    DEFAULT_PROMPT_LANGUAGE = "中文"
    DEFAULT_TEXT_LANGUAGE = "中文"
    DEFAULT_HOW_TO_CUT = "不切"
    # his, audio, conversion_time = generate_response_and_tts(
    #     chat_his,
    #     0.7,
    #     0.7,
    #     0,
    #     1.2,
    #     True,
    #     DEFAULT_AUDIO_SELECT,
    #     DEFAULT_REF_TEXT,
    #     DEFAULT_PROMPT_LANGUAGE,
    #     DEFAULT_TEXT_LANGUAGE,
    #     DEFAULT_HOW_TO_CUT
    # )
    #
    # print(his)
    # print(audio)
    # print(conversion_time)

    import asyncio


    async def debug():
        async for his, audio, conversion_time in generate_response_and_tts(
                chat_his,
                0.7,
                0.7,
                0,
                1.2,
                [True],
                DEFAULT_AUDIO_SELECT,
                DEFAULT_REF_TEXT,
                DEFAULT_PROMPT_LANGUAGE,
                DEFAULT_TEXT_LANGUAGE,
                DEFAULT_HOW_TO_CUT
        ):
            print("his:", his)
            print("audio:", audio)
            print("time:", conversion_time)


    asyncio.run(debug())

