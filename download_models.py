"""
Pre-download HuggingFace models to cache to avoid git issues during training.
Run this ONCE before training.
"""

import os

# Set cache location outside git repo
HF_CACHE_DIR = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

# Disable all git operations
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/bin/false'  # Disable git entirely

os.makedirs(HF_CACHE_DIR, exist_ok=True)

print("Downloading models to cache...")
print(f"Cache directory: {HF_CACHE_DIR}")

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

print("\n1. Downloading BERT (bert-base-uncased)...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
print("   ✓ BERT downloaded successfully")

print("\n2. Downloading DeBERTa (microsoft/deberta-v3-small)...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
model = AutoModel.from_pretrained('microsoft/deberta-v3-small')
print("   ✓ DeBERTa downloaded successfully")

print("\n" + "="*70)
print("All models downloaded successfully!")
print("You can now run training scripts.")
print("="*70)
