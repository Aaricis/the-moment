import os
import soundfile as sf
import whisper

# ========== 加载 Whisper 模型（只加载一次） ==========
whisper_model = whisper.load_model("small")


def whisper_recognize(audio):
    """将麦克风录音转换为中文文本"""
    if audio is None:
        return ""
    sr, data = audio
    tmp_path = "temp_record.wav"
    sf.write(tmp_path, data, sr)
    result = whisper_model.transcribe(
        tmp_path,
        language="zh",
        initial_prompt="生于忧患，死于欢乐。不亦快哉！"
    )
    os.remove(tmp_path)
    return result["text"]
