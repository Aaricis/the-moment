import asyncio
import os
import sys
import time

sys.path.append(os.path.split(sys.path[0])[0])  # 添加包搜索路径

import librosa
import torch
from transformers import TextIteratorStreamer

from src.configs.base_config import is_half, device
from src.configs.tts_config import audio_path, slicer_list, sovits_path, cnhubert_base_path, gpt_path
from src.tts.module.mel_processing import spectrogram_torch
from src.utils.loader import load_text_audio_mappings, load_audio
from src.utils.parser import ParserState, parse_response
from src.utils.response import format_response, format_time
from src.utils.tts_utils import clean_text_inf, nonen_clean_text_inf, cut1, cut2, cut3, cut4, cut5, splits, \
    get_bert_inf, nonen_get_bert_inf, process_text, merge_short_text_in_array
from src.tts.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from src.tts.module.models import SynthesizerTrn
from src.utils.parser import DictToAttrRecursive
from src.tts.feature_extractor import cnhubert


def get_spepc(hps, filename):
    """
    把任意音频文件变成梅尔谱图
    :param hps:超参数
    :param filename:音频文件
    :return:梅尔谱图
    """
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)


def change_sovits_weights(sovits_path):
    """
    加载SoVITS模型权重，根据配置初始化模型，修改部分配置项，并将模型移动到指定设备，最后加载权重并设置为评估模式。
    :param sovits_path: SoVITS模型权重路径
    :return: 超参数，SoVITS 声学模型（vq_model）
    """

    global vq_model, hps
    print(os.path.isfile(sovits_path), 2222222)
    dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))


change_sovits_weights(sovits_path)

ssl_model = cnhubert.get_model(cnhubert_base_path)
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_gpt_weights(gpt_path):
    """
        加载GPT文本→语义 token模型
        :param gpt_path: 模型路径
        :return: 固定语义帧率，最大时长限制，模型，配置参数
    """
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


change_gpt_weights(gpt_path)


async def get_tts_wav(selected_text, prompt_text, prompt_language, text, text_language, how_to_cut="不切"):
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
    ref_wav_path = text_to_audio_mappings.get(selected_text, "")
    if not ref_wav_path:
        print("Audio file not found for the selected text.")
        return

    prompt_text = prompt_text.strip("\n")
    text = text.strip("\n")

    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half else np.float32,
    )

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
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
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
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
        else:
            phones2, word2ph2, norm_text2 = nonen_clean_text_inf(text_chunk, text_language)

        if text_language == "en":
            bert2 = get_bert_inf(phones2, word2ph2, norm_text2, text_language)
        else:
            bert2 = nonen_get_bert_inf(text_chunk, text_language)

        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=config["inference"]["top_k"],
                early_stop_num=hz * max_sec,
            )

        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        refer = get_spepc(hps, ref_wav_path)
        if is_half:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)

        audio_chunk = vq_model.decode(
            pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
        ).detach().cpu().numpy()[0, 0]

        end_time = time.time()
        conversion_duration = end_time - start_time
        total_conversion_time += conversion_duration

        audio_with_pause = np.concatenate([audio_chunk, zero_wav])
        yield (hps.data.sampling_rate, (audio_with_pause * 32768).astype(np.int16)), format_time(total_conversion_time)


async def async_generate(model, tokenizer, generate_kwargs):
    """
    异步包装器，边生成边yield chunk
    """
    streamer = TextIteratorStreamer(  # 边生成边输出
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
                history[-1]["content"] = "\n\n".join(collapsible + [answer_part])

                # 流式推送文本，音频还没生成
                yield history, None, "0s"

        # === 2. 文本生成完成，最终整理 ===
        state, elapsed = parse_response(full_response, state)
        collapsible, answer_part = format_response(state, elapsed)
        history[-1]["content"] = "\n\n".join(collapsible + [answer_part])

        # 去掉 <think> 标签
        answer_text = answer_part.replace("<think>", "").replace("</think>", "").strip()
        print("len(answer text) =", len(answer_text), "answer text =", answer_text)

        lines = [line.strip() for line in answer_text.split('\n') if line.strip()]
        cleaned_answer_text = ''.join(lines)
        print("cleaned answer text =", len(cleaned_answer_text), "text =", cleaned_answer_text)

        # === 3. 流式返回音频 ===
        async for audio_chunk, conversion_time in get_tts_wav(
                selected_text,
                ref_text,
                prompt_language,
                # answer_text,
                cleaned_answer_text,  # 使用清理后的文本
                text_language,
                how_to_cut
        ):
            yield history, audio_chunk, conversion_time


    except Exception as e:
        import traceback
        # 1. 打印完整堆栈到终端
        traceback.print_exc()
        history[-1]["content"] = f"Error: {str(e)}"
        yield history, None, "Error"

    finally:
        active_gen[0] = False


import asyncio
import numpy as np
import soundfile as sf


async def test_tts():
    text_to_audio_mappings = load_text_audio_mappings(audio_path, slicer_list)

    DEFAULT_AUDIO_SELECT = list(text_to_audio_mappings.keys())[0] if text_to_audio_mappings else ""
    DEFAULT_REF_TEXT = DEFAULT_AUDIO_SELECT
    DEFAULT_PROMPT_LANGUAGE = "zh"
    DEFAULT_TEXT_LANGUAGE = "zh"
    DEFAULT_HOW_TO_CUT = "按标点符号切"

    # 1. 累计所有 yield 段
    sr, pcm_total = None, []
    async for (sr, audio_bytes), t in get_tts_wav(
            DEFAULT_AUDIO_SELECT,
            DEFAULT_REF_TEXT,
            DEFAULT_PROMPT_LANGUAGE,
            "哎呀，看来你也挺有意思的嘛！为什么会突然问我“你是谁呀”呢？是不是最近有点好奇啊，或者是想到了某些特别的事情？你可以多聊聊哦～比如说，你在什么时候会有这样的疑问？或者说，有没有什么特定的原因让你这么问？",
            DEFAULT_TEXT_LANGUAGE,
            DEFAULT_HOW_TO_CUT
    ):
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        pcm_total.append(pcm)
        print(f"【test】收到段 {len(pcm)} 点，累计时间 {t}")

    # 2. 拼接
    full_pcm = np.concatenate(pcm_total) if pcm_total else np.zeros(0, dtype=np.int16)

    # 3. 写 .wav
    out_file = "test_cut.wav"
    sf.write(out_file, full_pcm, sr)
    print(f"✅ 已生成 {out_file}  {len(full_pcm) / sr:.2f}s  请用播放器打开！")


# 运行
if __name__ == "__main__":
    asyncio.run(test_tts())
