#!/bin/bash
set -e   # 出错立即退出

# 定义下载 URL 和目标文件名
# shellcheck disable=SC2054
URL=(
"https://github.com/Aaricis/the-moment/releases/download/v0.1.0/EmoLLMRAGTXT.tar.gz", # vectorDB
"https://github.com/Aaricis/the-moment/releases/download/v0.1.0/pretrained_models.tar.gz" # TTS模型权重
)
# shellcheck disable=SC2054
FILE=(
"EmoLLMRAGTXT", "pretrained_models"
)

# ========== 逐个处理 ==========
for i in "${!URL[@]}"; do
  url="${URL[$i]}"
  file="${FILE[$i]}"

  # 下载文件
  echo "Downloading weights from $url..."
  wget -O "$file" "$url"

  # 检查是否下载成功
  # shellcheck disable=SC2181
  if [ $? -ne 0 ]; then
    echo "Download failed. Please check the URL or your network connection."
    exit 1
  fi

  # 解压文件
  echo "Extracting $file"
  tar -xzvf "$file"

  # 检查是否解压成功
  # shellcheck disable=SC2181
  if [ $? -ne 0 ]; then
    echo "Extraction failed. Please check the downloaded file."
    exit 1
  fi

  # 删除下载的压缩文件
  rm "$file"
  echo "Download and extraction completed successfully!"
done

DIR=${1:-./model}
echo "Downloading model to $DIR ..."
mkdir -p "$DIR"
modelscope download \
  --model haiyangpengai/careyou_7b_16bit_v3_2_qwen14_4bit \
  --local_dir "$DIR"
echo "✅ Download completed!"

echo "✅ All downloads and extractions completed!"