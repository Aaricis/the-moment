# The Moment

## 快速开始 | Quick Start

1. 安装环境 | Install Environment

**Install Directly**

```text
pip install -r requirements.txt
```

**Install from conda**

```text
conda create -n chattts python=3.11
conda activate chattts
pip install -r requirements.txt
```

2. 下载模型权重 | Download Model Weights
```bash
./scripts/download_weights.sh
```
3. 启动 Web UI | Run Web Interface
```bash
cd the_moment/tools
python webui.py
```

## 项目优化 ｜ TODO
- [ ] 增加ASR语音识别；
- [ ] 更新问答模型，使用更加专业的心理学模型；
- [ ] 使用语音语言模型，直接语音进语音出，不要语音 vs 文字转换；

## 免责声明 | Disclaimer

本项目仅供学习和研究使用。使用者须遵守当地的法律法规，包括但不限于 DMCA 相关法律。我们不对任何非法使用承担责任。

This project is for research and learning purposes only. Users must comply with local laws and regulations, including but not limited to DMCA-related laws. We do not take any responsibility for illegal usage.

## 技术鸣谢 | Credits

本项目基于以下开源项目构建：

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)