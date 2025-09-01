import asyncio

from transformers import TextIteratorStreamer
from src.utils.parser import ParserState, parse_response
from src.utils.response import format_response
from src.utils.loader import load_text_audio_mappings
from src.configs.tts_config import audio_path, slicer_list

from loguru import logger


def get_tts_wav(
        selected_text,
        prompt_text,
        prompt_language,
        text,
        text_language,
        how_to_cut
):
    """
    [参考语音 + 提示文本 + 目标文本]合成用参考音频模式讲的目标文本语音
    :param selected_text: 参考文本
    :param prompt_text: 提示文本
    :param prompt_language: 提示语言
    :param text: 目标文本
    :param text_language: 目标语言
    :param how_to_cut: 音频切割方式
    :return: 采样率，音频字节流
    """
    text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)
    ref_wav_path = text_to_audio_mappings.get(selected_text, "")
    if not ref_wav_path:
        logger.error("Audio file not found for the selected text.")

    prompt_text = prompt_text.strip("\n")
    text = text.strip("\n")


async def async_generate(model, tokenizer, generate_kwargs):
    """
    异步包装器，边生成边yield chunk
    """
    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=20.0,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    kwargs = generate_kwargs.copy()
    kwargs["streamer"] = streamer

    # 在后台线程启动 generate
    gen_task = asyncio.create_task(asyncio.to_thread(model.generate, **kwargs))

    try:
        # 异步地消费同步迭代器
        while True:
            chunk = await asyncio.to_thread(next, streamer, None)
            if chunk is None:
                break
            yield chunk
    finally:
        await gen_task


async def async_chat(
        active_gen, history,
        tokenizer, model, generate_kwargs,
        selected_text, ref_text, prompt_language, text_language, how_to_cut
):
    if not active_gen[0]:
        return

    generate_kwargs["tokenizer"] = tokenizer  # 注入 tokenizer
    full_response = "<think>"
    state = ParserState()

    try:
        async for chunk in async_generate(model, tokenizer, generate_kwargs):
            if not active_gen[0]:
                break
            full_response += chunk
            state, elapsed = parse_response(full_response, state)
            collapsible, answer_part = format_response(state, elapsed)
            history[-1][1] = "\n\n".join(collapsible + [answer_part])
            yield history, None

        # 最终整理
        state, elapsed = parse_response(full_response, state)
        collapsible, answer_part = format_response(state, elapsed)
        history[-1][1] = "\n\n".join(collapsible + [answer_part])

        answer_text = answer_part.replace('<think>', '').replace('</think>', '').strip()
        audio_data = get_tts_wav(
            selected_text,
            ref_text,
            prompt_language,
            answer_text,
            text_language,
            how_to_cut
        )
        yield history, audio_data

    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
        yield history, None
    finally:
        active_gen[0] = False
