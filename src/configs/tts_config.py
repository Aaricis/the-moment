import os
from .rag_config import base_dir

# gpt path
gpt_path = os.path.join(base_dir, 'pretrained_models/heart_girl/heartful_sister.ckpt')
sovits_path = os.path.join(base_dir, 'pretrained_models/heart_girl/heartful_sister.pth')
cnhubert_base_path = os.path.join(base_dir, "pretrained_models/chinese-hubert-base")
bert_path = os.path.join(base_dir, "pretrained_models/chinese-roberta-wwm-ext-large")

# audio path
audio_path = os.path.join(base_dir, "data/audio")
slicer_list = 'slicer_opt.list'