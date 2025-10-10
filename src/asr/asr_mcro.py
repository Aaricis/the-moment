import gradio as gr
import whisper
import os

# ========== 1. 预加载 Whisper 模型 ==========
# small 速度快、准确度高；如需更精准可改为 "medium" 或 "large-v3"
model = whisper.load_model("small")

# ========== 2. 推理函数 ==========
def transcribe_audio(audio):
    """
    audio: 来自 Gradio 的麦克风录音 (sample_rate, numpy.ndarray)
    """
    if audio is None:
        return "请录制或上传一段语音后再试。"

    # Gradio 麦克风输入返回一个 tuple (sr, data)
    sr, audio_data = audio

    # 临时保存音频为 WAV 文件
    tmp_path = "temp_record.wav"
    import soundfile as sf
    sf.write(tmp_path, audio_data, sr)

    # Whisper 识别
    result = model.transcribe(
        tmp_path,
        language="zh",
        initial_prompt="以下是普通话中文语音，请保留合适的标点符号。"
    )

    # 删除临时文件
    os.remove(tmp_path)

    return result["text"]

# ========== 3. 构建 Gradio 前端 ==========
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(
        sources=["microphone"],     # ✅ 只启用麦克风
        type="numpy",               # 返回 (sr, numpy array)
        label="🎤 录制你的中文语音"
    ),
    outputs=gr.Textbox(
        label="📝 识别结果（自动加标点）",
        lines=6
    ),
    title="Whisper 中文语音识别 🎧",
    description="点击录音按钮，说一段中文（≤30 秒），松开后自动识别并生成带标点的文本。",
    cache_examples=False,
    theme="soft"   # 可选主题：default / soft / glass
)

# ========== 4. 启动服务 ==========
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
