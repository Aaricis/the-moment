import os

import torch

from rag_config import base_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_half = True if torch.cuda.is_available() else False

# Deepseek-R1 path
model_path = os.path.join(base_dir, 'model')
