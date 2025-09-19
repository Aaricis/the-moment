import asyncio
import time

import librosa
import numpy as np
import torch
from loguru import logger
from transformers import TextIteratorStreamer

from src.configs.base_config import is_half, device
from src.configs.tts_config import audio_path, slicer_list, sovits_path, cnhubert_base_path, gpt_path
from src.tts.feature_extractor.cnhubert import CNHubert
from src.tts.module.mel_processing import spectrogram_torch
from src.utils.loader import load_text_audio_mappings, load_sovits_weights, load_gpt_weights, load_audio
from src.utils.parser import ParserState, parse_response
from src.utils.response import format_response, format_time
from src.utils.tts_utils import clean_text_inf, nonen_clean_text_inf, cut1, cut2, cut3, cut4, cut5, splits, \
    get_bert_inf, nonen_get_bert_inf


def get_ssl_model():
    ssl_model = CNHubert(cnhubert_base_path)
    ssl_model.eval()
    ssl_model = ssl_model.half().to(device) if is_half else ssl_model.to(device)
    return ssl_model


def get_spepc(hps, filename):
    """
    把任意音频文件变成梅尔谱图
    :param hps:超参数
    :param filename:音频文件
    :return:梅尔谱图
    """
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


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
    :return: （采样率，音频字节流），音频转换时间
    """
    text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)
    ref_wav_path = text_to_audio_mappings.get(selected_text, "")
    if not ref_wav_path:
        logger.error("Audio file not found for the selected text.")

    prompt_text = prompt_text.strip("\n")
    text = text.strip("\n")

    hps, vq_model = load_sovits_weights(sovits_path)

    # 生成一条 0.3 秒的静音音频，格式（float16/float32）随硬件自动选择，用于后续拼接或对齐
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half else np.float32,
    )

    with torch.no_grad():
        # librosa 把 任何采样率的音频文件 统一读成 16 kHz 的单声道波形
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)

        dtype = torch.float16 if is_half else torch.float32
        wav16k = wav16k.to(device=device, dtype=dtype)
        zero_wav_torch = zero_wav_torch.to(device=device, dtype=dtype)

        # wav16k + zero_wav_torch，总时长 = 原音频 + 0.3 s
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_model = get_ssl_model()
        # 把 16 kHz 波形 喂给 SSL 模型（如 CN-Hubert），提取帧级语义特征，并整理成下游任务需要的形状
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        # 向量量化模型 (VQ model) 里，把输入特征 ssl_content 转换成 离散化的 latent code（码本索引）
        codes = vq_model.extract_latent(ssl_content)
        # 取 第 1 条音频、第 1 个码本通道、第 1 帧 的 token 值
        prompt_semantic = codes[0, 0]

        if prompt_language == "en":
            phones1, word2ph1, norm_text1 = clean_text_inf(prompt_text, prompt_language)
        else:
            phones1, word2ph1, norm_text1 = nonen_clean_text_inf(prompt_text, prompt_language)

        if how_to_cut == "凑四句一切":
            text = cut1(text)
        elif how_to_cut == "凑50字一切":
            text = cut2(text)
        elif how_to_cut == "按中文句号。切":
            text = cut3(text)
        elif how_to_cut == "按英文句号.切":
            text = cut4(text)
        elif how_to_cut == "按标点符号切":
            text = cut5(text)

        if text and text[-1] not in splits:
            text += "。" if text_language != "en" else "."

        texts = text.split("\n")
        total_conversion_time = 0.0

        if prompt_language == "en":
            bert1 = get_bert_inf(phones1, word2ph1, norm_text1, prompt_language)
        else:
            bert1 = nonen_get_bert_inf(prompt_text, prompt_language)

        for text_chunk in texts:
            if not text_chunk.strip():
                continue

            start_time = time.time()

            if text_language == "en":
                phones2, word2ph2, norm_text2 = clean_text_inf(text_chunk, text_language)
                bert2 = get_bert_inf(phones2, word2ph2, norm_text2, text_language)
            else:
                phones2, word2ph2, norm_text2 = nonen_clean_text_inf(text_chunk, text_language)
                bert2 = nonen_get_bert_inf(text_chunk, text_language)

            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).unsqueeze(0)
            bert = bert.to(device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
            prompt = prompt_semantic.unsqueeze(0).to(device)

            hz, max_sec, t2s_model, config = load_gpt_weights(gpt_path)
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    top_k=config["inference"]["top_k"],
                    early_stop_num=hz * max_sec,
                )
            pred_semantic = pred_semantic[:, -idx].unsqueeze(0)  # 只保留 最后 idx 个 生成的 token（去掉 prompt 部分）
            refer = get_spepc(hps, ref_wav_path)
            refer = refer.half().to(device) if is_half else refer.to(device)
            audio_chunk = vq_model.decode(
                pred_semantic,
                torch.LongTensor(phones2).to(device).unsqueeze(0),
                refer
            ).detach().cpu().numpy()[0, 0]

            end_time = time.time()
            conversion_duration = end_time - start_time
            total_conversion_time += conversion_duration

            audio_with_pause = np.concatenate([audio_chunk, zero_wav])
            yield (hps.data.sampling_rate, (audio_with_pause * 32768).astype(np.int16)), format_time(
                total_conversion_time)


async def async_generate(model, tokenizer, generate_kwargs):
    """
    异步包装器，边生成边yield chunk
    """
    streamer = TextIteratorStreamer( # 边生成边输出
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
        active_gen, history: list,
        tokenizer, model, generate_kwargs,
        selected_text, ref_text, prompt_language, text_language, how_to_cut
):
    if not active_gen[0]:
        return

    # generate_kwargs["tokenizer"] = tokenizer  # 注入 tokenizer
    full_response = "<think>"
    state = ParserState()

    try:
        # === 1. 流式文本生成 ===
        async for chunk in async_generate(model, tokenizer, generate_kwargs):
            if not active_gen[0]:
                break

            if chunk:
                full_response += chunk
                state, elapsed = parse_response(full_response, state)
                collapsible, answer_part = format_response(state, elapsed)

                # ✅ 更新最后一个 ChatMessage 的 content
                history[-1].content = "\n\n".join(collapsible + [answer_part])

                # 流式推送文本，音频还没生成
                yield history, None, "0s"

        # === 2. 文本生成完成，最终整理 ===
        state, elapsed = parse_response(full_response, state)
        collapsible, answer_part = format_response(state, elapsed)
        history[-1].content = "\n\n".join(collapsible + [answer_part])

        # 去掉 <think> 标签
        answer_text = answer_part.replace("<think>", "").replace("</think>", "").strip()

        # === 3. 流式返回音频 ===
        # 如果 get_tts_wav 是异步生成器
        for audio_chunk, conversion_time in get_tts_wav(
            selected_text,
            ref_text,
            prompt_language,
            answer_text,
            text_language,
            how_to_cut
        ):
            yield history, audio_chunk, conversion_time

    except Exception as e:
        history[-1].content = f"Error: {str(e)}"
        yield history, None, "Error"

    finally:
        active_gen[0] = False

