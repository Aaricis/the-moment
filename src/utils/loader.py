import os

import ffmpeg
import numpy as np
import torch

from src.configs.base_config import device, is_half
from src.tts.module.models import SynthesizerTrn
from src.utils.parser import DictToAttrRecursive


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def load_text_audio_mappings(folder_path, list_file_name):
    """
    读取folder_path目录下的 list_file_name 文件（格式：音频文件名|list_file_name|语言|文本内容），构造音频文件->文本映射字典
    :param folder_path:文件路径
    :param list_file_name:文件名称
    :return:文本->音频字典
    """
    text_to_audio_mappings = {}  # text_to_audio_mappings[文本]=完整音频路径

    with open(os.path.join(folder_path, list_file_name), 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                audio_file_name = parts[0]
                text = parts[3]
                audio_file_path = os.path.join(folder_path, audio_file_name)
                text_to_audio_mappings[text] = audio_file_path
    return text_to_audio_mappings


def load_sovits_weights(sovits_path):
    """
    加载SoVITS模型权重，根据配置初始化模型，修改部分配置项，并将模型移动到指定设备，最后加载权重并设置为评估模式。
    :param sovits_path: SoVITS模型权重路径
    :return: 超参数，SoVITS 声学模型（vq_model）
    """

    dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
    hps = dict_s2['config']
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    vq_model = vq_model.half().to(device) if is_half else vq_model.to(device)
    vq_model.eval()
    return hps, vq_model
