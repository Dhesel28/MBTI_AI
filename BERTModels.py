import re
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F # Needed for Attention Pooling
from torch.utils.data import Dataset, DataLoader
# Import all necessary transformer components
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    precision_score, 
    recall_score
)
from tqdm import tqdm
import warnings
import math # For Attention Pooling

# --- 1. Configuration ---

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Model Names ---
BERT_MODEL_NAME = 'bert-base-uncased'
DEBERTA_MODEL_NAME = 'microsoft/deberta-v3-small'

# --- Data & General Hyperparameters ---
DATASET_PATH = 'mbti_1.csv' # Or your augmented file
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3 # Consider increasing epochs slightly (e.g., 4) for deeper models

# --- Hyperparameters for each model ---
# Model 1: Basic BERT
BASIC_LR = 2e-5

# Model 2: BERT Very Deep Head (6-layer head)
BERT_DEEP_HEAD_LR = 1e-4
BERT_DEEP_BERT_LR = 1e-5
BERT_DEEP_DROPOUT = 0.55 # Increased dropout for deeper head

# Model 3: DeBERTa Very Deep Head (6-layer head, CLS pooling)
DEBERTA_DEEP_HEAD_LR = 1e-4
DEBERTA_DEEP_BERT_LR = 1e-5
DEBERTA_DEEP_DROPOUT = 0.55 # Increased dropout for deeper head

# Model 4: DeBERTa + Attn Pool + Very Deep Head (Ablation Target, 6-layer head)
ABLATION_HEAD_LR = 1e-4 
ABLATION_BERT_LR = 1e-5
ABLATION_DROPOUT = 0.55 # Increased dropout for deeper head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Data Loading and Preprocessing (Unchanged) ---

def clean_text(text):
    text = text.replace('|||', ' '); text = re.sub(r'http\S+', '', text); text = re.sub(r'@\w+', '', text); text = text.lower(); text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(path):
    df = pd.read_csv(path); df['cleaned_posts'] = df['posts'].apply(clean_text)
    df['is_I'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0); df['is_N'] = df['type'].apply(lambda x: 1 if x[1] == 'N' else 0)
    df['is_T'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0); df['is_P'] = df['type'].apply(lambda x: 1 if x[3] == 'P' else 0)
    return df

# --- 3. PyTorch Dataset Class (Unchanged) ---

class MBTITraitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts; self.labels = labels; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx]); label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label, dtype=torch.float)}

# --- 4. Model Architectures ---

class BaselineBERTClassifier(nn.Module):
    """Model 1: Basic BERT"""
    def __init__(self, n_out=1, dropout_rate=0.3):
        super().__init__(); self.bert = BertModel.from_pretrained(BERT_MODEL_NAME); self.dropout = nn.Dropout(p=dropout_rate); self.classifier = nn.Linear(self.bert.config.hidden_size, n_out)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask); pooled_output = outputs.pooler_output; pooled_output = self.dropout(pooled_output); logits = self.classifier(pooled_output); return logits

# --- NEW: Helper function to create very deep heads ---
def create_very_deep_head(input_size, output_size, hidden_sizes, dropout_rate):
    """Creates a sequential head with multiple linear layers."""
    layers = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend([
            nn.Linear(current_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        ])
        current_size = hidden_size
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)

class BERTDeepHeadClassifier(nn.Module):
    """Model 2: BERT with very deep head (6 linear layers)"""
    def __init__(self, n_out=1, dropout_rate=BERT_DEEP_DROPOUT): 
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        bert_hidden_size = self.bert.config.hidden_size # 768
        # Define hidden sizes for 6 linear layers total (input -> h1 -> h2 -> h3 -> h4 -> h5 -> output)
        hidden_sizes = [512, 256, 128, 64, 32] 
        self.head = create_very_deep_head(bert_hidden_size, n_out, hidden_sizes, dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate) # Dropout on BERT output
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)
        return logits

class DeBERTaClassifier(nn.Module):
    """Model 3: DeBERTa with very deep head (6 linear layers, CLS pooling)"""
    def __init__(self, n_out=1, dropout_rate=DEBERTA_DEEP_DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(DEBERTA_MODEL_NAME)
        bert_hidden_size = self.bert.config.hidden_size # 768 for small
        hidden_sizes = [512, 256, 128, 64, 32]
        self.head = create_very_deep_head(bert_hidden_size, n_out, hidden_sizes, dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate) # Dropout on CLS token output

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0] # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)
        return logits

# --- Attention Pooling Layer (Unchanged) ---
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__(); self.attention_net = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states, attention_mask):
        scores = self.attention_net(hidden_states); scores.masked_fill_(attention_mask.unsqueeze(-1) == 0, -float('inf'))
        attn_weights = F.softmax(scores, dim=1); context = torch.sum(attn_weights * hidden_states, dim=1)
        return context

class DeBERTaAblationModel(nn.Module):
    """Model 4: DeBERTa + Attention Pooling + Very Deep Head (6 linear layers)"""
    def __init__(self, n_out=1, dropout_rate=ABLATION_DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(DEBERTA_MODEL_NAME)
        bert_hidden_size = self.bert.config.hidden_size
        
        self.attention_pooling = AttentionPooling(bert_hidden_size)
        
        # Use the very deep head
        hidden_sizes = [512, 256, 128, 64, 32]
        self.head = create_very_deep_head(bert_hidden_size, n_out, hidden_sizes, dropout_rate)
        # No extra dropout usually needed between pooling and head

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.attention_pooling(last_hidden_state, attention_mask)
        logits = self.head(pooled_output)
        return logits

# --- 5. Training and Evaluation Functions (Unchanged) ---

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    model = model.train(); total_loss = 0
    pbar = tqdm(data_loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask); loss = loss_fn(outputs.squeeze(), labels); total_loss += loss.item()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
        if scheduler: scheduler.step()
        optimizer.zero_grad(); pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval(); all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device); labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask); probs = torch.sigmoid(outputs).cpu().numpy(); preds = (probs > 0.5).astype(int)
            all_labels.extend(labels.cpu().numpy()); all_probs.extend(probs.flatten()); all_preds.extend(preds.flatten())
    accuracy = accuracy_score(all_labels, all_preds); precision = precision_score(all_labels, all_preds, average='binary', zero_division=0); recall = recall_score(all_labels, all_preds, average='binary', zero_division=0); f1 = f1_score(all_labels, all_preds, average='binary')
    try: auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError: auc_roc = 0.5
    return accuracy, precision, recall, f1, auc_roc

# --- 6. Main Execution Pipeline ---

def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(DATASET_PATH)
    print(f"Loaded {len(df)} data samples.")
    
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
    
    traits = ['is_I', 'is_N', 'is_T', 'is_P']
    trait_names = {'is_I': 'Mind (I/E)', 'is_N': 'Energy (N/S)', 'is_T': 'Nature (T/F)', 'is_P': 'Tactics (P/J)'}
    
    final_results = {}

    for trait in traits:
        trait_name = trait_names[trait]
        print(f"\n{'='*25} PROCESSING TRAIT: {trait_name} {'='*25}")
        
        df_train, df_val = train_test_split(df, test_size=0.1, random_state=42, stratify=df[trait])
        
        # --- BERT DataLoaders ---
        bert_train_loader = DataLoader(MBTITraitDataset(df_train['cleaned_posts'].values, df_train[trait].values, bert_tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        bert_val_loader = DataLoader(MBTITraitDataset(df_val['cleaned_posts'].values, df_val[trait].values, bert_tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        # --- DeBERTa DataLoaders ---
        deberta_train_loader = DataLoader(MBTITraitDataset(df_train['cleaned_posts'].values, df_train[trait].values, deberta_tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        deberta_val_loader = DataLoader(MBTITraitDataset(df_val['cleaned_posts'].values, df_val[trait].values, deberta_tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        # --- Model 1: Basic BERT ---
        print(f"\n--- Training Basic BERT for: {trait_name} ---")
        model = BaselineBERTClassifier().to(device); loss_fn = nn.BCEWithLogitsLoss().to(device)
        optimizer = AdamW(model.parameters(), lr=BASIC_LR, correct_bias=False); total_steps = len(bert_train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        best_f1 = 0; best_metrics = {}
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}"); train_loss = train_epoch(model, bert_train_loader, loss_fn, optimizer, device, scheduler)
            acc, prc, rec, f1, auc = eval_model(model, bert_val_loader, loss_fn, device); print(f"Val F1: {f1:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")
            if f1 > best_f1: best_f1, best_metrics = f1, {'Accuracy': acc, 'Precision': prc, 'Recall': rec, 'F1': f1, 'AUC-ROC': auc}
        final_results[(trait_name, 'Basic BERT')] = best_metrics
        del model, optimizer, scheduler; torch.cuda.empty_cache()

        # --- Model 2: BERT Very Deep Head ---
        print(f"\n--- Training BERT Very Deep Head for: {trait_name} ---")
        model = BERTDeepHeadClassifier().to(device); loss_fn = nn.BCEWithLogitsLoss().to(device)
        optimizer_grouped_parameters = [{"params": model.bert.parameters(), "lr": BERT_DEEP_BERT_LR}, {"params": model.head.parameters(), "lr": BERT_DEEP_HEAD_LR}]
        optimizer = AdamW(optimizer_grouped_parameters); scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS)
        best_f1 = 0; best_metrics = {}
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}"); train_loss = train_epoch(model, bert_train_loader, loss_fn, optimizer, device, scheduler=None)
            acc, prc, rec, f1, auc = eval_model(model, bert_val_loader, loss_fn, device); scheduler.step()
            print(f"Val F1: {f1:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")
            if f1 > best_f1: best_f1, best_metrics = f1, {'Accuracy': acc, 'Precision': prc, 'Recall': rec, 'F1': f1, 'AUC-ROC': auc}
        final_results[(trait_name, 'BERT Very Deep Head')] = best_metrics # Renamed for clarity
        del model, optimizer, scheduler; torch.cuda.empty_cache()

        # --- Model 3: DeBERTa Very Deep Head (CLS Pooling) ---
        print(f"\n--- Training DeBERTa Very Deep Head for: {trait_name} ---")
        model = DeBERTaClassifier().to(device); loss_fn = nn.BCEWithLogitsLoss().to(device)
        optimizer_grouped_parameters = [{"params": model.bert.parameters(), "lr": DEBERTA_DEEP_BERT_LR}, {"params": model.head.parameters(), "lr": DEBERTA_DEEP_HEAD_LR}]
        optimizer = AdamW(optimizer_grouped_parameters); scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS)
        best_f1 = 0; best_metrics = {}
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}"); train_loss = train_epoch(model, deberta_train_loader, loss_fn, optimizer, device, scheduler=None)
            acc, prc, rec, f1, auc = eval_model(model, deberta_val_loader, loss_fn, device); scheduler.step()
            print(f"Val F1: {f1:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")
            if f1 > best_f1: best_f1, best_metrics = f1, {'Accuracy': acc, 'Precision': prc, 'Recall': rec, 'F1': f1, 'AUC-ROC': auc}
        final_results[(trait_name, 'DeBERTa Very Deep Head (CLS)')] = best_metrics # Renamed for clarity
        del model, optimizer, scheduler; torch.cuda.empty_cache()

        # --- Model 4: DeBERTa + Attn Pool + Very Deep Head ---
        print(f"\n--- Training DeBERTa + Attn Pool + Very Deep Head for: {trait_name} ---")
        model = DeBERTaAblationModel().to(device); loss_fn = nn.BCEWithLogitsLoss().to(device)
        optimizer_grouped_parameters = [
            {"params": model.bert.parameters(), "lr": ABLATION_BERT_LR},
            {"params": model.attention_pooling.parameters(), "lr": ABLATION_HEAD_LR}, 
            {"params": model.head.parameters(), "lr": ABLATION_HEAD_LR}
        ]
        optimizer = AdamW(optimizer_grouped_parameters); scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS)
        best_f1 = 0; best_metrics = {}
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}"); train_loss = train_epoch(model, deberta_train_loader, loss_fn, optimizer, device, scheduler=None)
            acc, prc, rec, f1, auc = eval_model(model, deberta_val_loader, loss_fn, device); scheduler.step()
            print(f"Val F1: {f1:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")
            if f1 > best_f1: best_f1, best_metrics = f1, {'Accuracy': acc, 'Precision': prc, 'Recall': rec, 'F1': f1, 'AUC-ROC': auc}
        final_results[(trait_name, 'DeBERTa AttnPool Very Deep')] = best_metrics # Renamed for clarity
        del model, optimizer, scheduler; torch.cuda.empty_cache()


    # --- 3. Display Final Comparison ---
    print("\n" + "="*70)
    print("--- Project Complete: Final Comparison Report ---")
    print("="*70 + "\n")
    
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    results_df = pd.DataFrame.from_dict(final_results, orient='index')
    results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['Trait', 'Model'])
    results_df = results_df.sort_index()
    
    print(results_df)

if __name__ == "__main__":
    main()
