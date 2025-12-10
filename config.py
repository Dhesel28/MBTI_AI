"""
Configuration file for MBTI personality prediction models.
Contains all hyperparameters, model architectures, and shared utilities.
"""

import os

# Move HuggingFace cache outside git repo to prevent mutex lock issues
HF_CACHE_DIR = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

# Completely disable git integration
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/bin/false'  # Block git entirely
os.environ['GIT_TERMINAL_PROMPT'] = '0'
os.environ['GCM_INTERACTIVE'] = 'never'

# Create cache directory if it doesn't exist
os.makedirs(HF_CACHE_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel
import numpy as np
import warnings

# --- 1. Global Configuration ---

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Model Names ---
BERT_MODEL_NAME = 'bert-base-uncased'
DEBERTA_MODEL_NAME = 'microsoft/deberta-v3-small'

# --- Data & General Hyperparameters ---
DATASET_PATH = 'mbti_1.csv'
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 100

# --- Hyperparameters for each model ---
# Model 1: Basic BERT
BASIC_LR = 2e-5

# Model 2: BERT Deep Head (3-layer head)
BERT_DEEP_HEAD_LR = 1e-4
BERT_DEEP_BERT_LR = 1e-5
BERT_DEEP_DROPOUT = 0.5

# Model 3: DeBERTa Deep Head (2-layer head, CLS pooling)
DEBERTA_DEEP_HEAD_LR = 1e-4
DEBERTA_DEEP_BERT_LR = 1e-5
DEBERTA_DEEP_HIDDEN = 256
DEBERTA_DEEP_DROPOUT = 0.4

# Model 4: DeBERTa + Attn Pool + Deep Head
ABLATION_HEAD_LR = 1e-4
ABLATION_BERT_LR = 1e-5
ABLATION_HIDDEN = 256
ABLATION_DROPOUT = 0.4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()

# Print device info
if USE_GPU:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected. Training will run on CPU (this will be slower).")

# Trait mapping
TRAIT_CONFIG = {
    'EI': {
        'label': 'is_I',
        'name': 'Mind (I/E)',
        'position': 0,
        'models_dir': 'models/EI'
    },
    'NS': {
        'label': 'is_N',
        'name': 'Energy (N/S)',
        'position': 1,
        'models_dir': 'models/NS'
    },
    'TF': {
        'label': 'is_T',
        'name': 'Nature (T/F)',
        'position': 2,
        'models_dir': 'models/TF'
    },
    'JP': {
        'label': 'is_P',
        'name': 'Tactics (P/J)',
        'position': 3,
        'models_dir': 'models/JP'
    }
}

# --- 2. Model Architectures ---

class BaselineBERTClassifier(nn.Module):
    """Model 1: Basic BERT with single linear layer"""
    def __init__(self, n_out=1, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_out)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BERTDeepHeadClassifier(nn.Module):
    """Model 2: BERT with deeper 3-layer head"""
    def __init__(self, n_out=1, dropout_rate=BERT_DEEP_DROPOUT, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size_1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_2, n_out)
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)
        return logits


class DeBERTaClassifier(nn.Module):
    """Model 3: DeBERTa with 2-layer head (CLS pooling)"""
    def __init__(self, n_out=1, dropout_rate=DEBERTA_DEEP_DROPOUT, hidden_size=DEBERTA_DEEP_HIDDEN):
        super().__init__()
        self.bert = AutoModel.from_pretrained(DEBERTA_MODEL_NAME)
        bert_hidden_size = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, n_out)
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)
        return logits


class AttentionPooling(nn.Module):
    """Simple Attention Pooling layer"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_net = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        # Calculate attention scores
        scores = self.attention_net(hidden_states)  # (batch_size, seq_len, 1)

        # Apply mask - set attention scores to -infinity for padding tokens
        scores.masked_fill_(attention_mask.unsqueeze(-1) == 0, -float('inf'))

        # Calculate attention weights (softmax over seq_len dimension)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)

        # Calculate weighted average (context vector)
        context = torch.sum(attn_weights * hidden_states, dim=1)  # (batch_size, hidden_size)

        return context


class DeBERTaAblationModel(nn.Module):
    """Model 4: DeBERTa + Attention Pooling + Deep Head"""
    def __init__(self, n_out=1, dropout_rate=ABLATION_DROPOUT, hidden_size=ABLATION_HIDDEN):
        super().__init__()
        self.bert = AutoModel.from_pretrained(DEBERTA_MODEL_NAME)
        bert_hidden_size = self.bert.config.hidden_size

        # Attention Pooling Layer
        self.attention_pooling = AttentionPooling(bert_hidden_size)

        # Deep Head
        self.head = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, n_out)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get all hidden states from the last layer
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Apply Attention Pooling
        pooled_output = self.attention_pooling(last_hidden_state, attention_mask)

        # Pass pooled output through the head
        logits = self.head(pooled_output)
        return logits
