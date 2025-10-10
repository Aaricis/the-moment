import gradio as gr
import whisper

# 1. 全局加载一次模型，避免每次推理都 reload
model = whisper.load_model("small")   # 可换 medium / large-v3

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path, language="zh",
                              initial_prompt="生于忧患，死于欢乐。不亦快哉！")
    return result["text"]

def speech_to_text(audio_file):
    if audio_file is None:
        return ""
    return transcribe_audio(audio_file)

# 2. 构建 Gradio 界面
demo = gr.Interface(
    fn=speech_to_text,                      # 推理函数
    inputs=gr.Audio(type="filepath", label="上传音频或麦克风录制"),  # 支持文件/麦克风
    outputs=gr.Textbox(label="识别结果", lines=6),
    title="Whisper 中文语音识别",
    description="上传一段 ≤30 s 的中文音频（wav/mp3/m4a 均可），点击 Submit 即可看到带标点的文字。",
    examples=[["test_cut.wav"]],            # 可放默认示例文件
    cache_examples=False
)

# 3. 启动
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

