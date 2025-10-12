# The Moment
>_什么是人生里关键性的一刻，_
_是一个决定；_  
_是一次选择；_  
_是向左，还是向右；_  
_是继续，或者放弃；_  
_是跟过去告别的一刻；_  
_是勇敢擦拭伤口的那一刻；_  
_是抉择未来的那一刻。_  
......

## 项目预览 | Preview



## 技术架构 | Architecture

![工程架构图](/assets/the_moment_workflow.png)

The moment as a typical conversational AI application uses three subsystems to do the steps of processing and transcribing the audio, understanding (deriving meaning) of the question asked, generating the response (text) and speaking the response back to the human. These steps are achieved by multiple deep learning solutions working together. First, automatic speech recognition (ASR) is used to process the raw audio signal and transcribing text from it. Second, natural language processing (NLP) is used to derive meaning from the transcribed text (ASR output). Last, speech synthesis or text-to-speech (TTS) is used for the artificial production of human speech from text. Optimizing this multi-step process is complicated, as each of these steps requires building and using one or more deep learning models. When developing a deep learning model to achieve the highest performance and accuracy for each of these areas, a developer will encounter several approaches and experiments that can vary by domain application.

## 快速开始 | Quick Start

1. 安装环境 | Install Environment

**Install Directly**

```text
pip install -r requirements.txt
```

**Install from conda**

```text
conda create -n <env name> python=3.11
conda activate <env name>
pip install -r requirements.txt
```

2. 下载模型权重 | Download Model Weights
```bash
./scripts/download_weights.sh
```
3. 启动 Web UI | Run Web Interface
```bash
cd the_moment/tools
python webui_speech.py
```

## 项目优化 ｜ TODO
- [ ] 更新问答模型，使用更加专业的心理学模型；
- [ ] 使用Acoustic Model，而不是传统的ASR->LLM->TTS架构；

## 免责声明 | Disclaimer

本项目仅供学习和研究使用。使用者须遵守当地的法律法规，包括但不限于 DMCA 相关法律。我们不对任何非法使用承担责任。

This project is for research and learning purposes only. Users must comply with local laws and regulations, including but not limited to DMCA-related laws. We do not take any responsibility for illegal usage.

## 技术鸣谢 | Credits

本项目基于以下开源项目构建：

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Whisper](https://github.com/openai/whisper)