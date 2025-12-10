import re
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Configuration ---
BERT_MODEL_NAME = 'bert-base-uncased'
DATASET_PATH = 'mbti_1.csv'
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 1  # Just 1 epoch for quick demo
BASIC_LR = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("="*70)
print("QUICK DEMO: Training 1 Model (Baseline BERT) on 1 Trait (Mind: I/E)")
print("This will take approximately 30-45 minutes on CPU")
print("="*70 + "\n")

# --- Data Loading ---
def clean_text(text):
    text = text.replace('|||', ' ')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples")

    print("Cleaning text...")
    df['cleaned_posts'] = df['posts'].apply(clean_text)

    print("Extracting MBTI traits...")
    df['is_I'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)
    df['is_N'] = df['type'].apply(lambda x: 1 if x[1] == 'N' else 0)
    df['is_T'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
    df['is_P'] = df['type'].apply(lambda x: 1 if x[3] == 'P' else 0)

    return df

# --- Dataset Class ---
class MBTITraitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# --- Model Architecture ---
class BaselineBERTClassifier(nn.Module):
    """Model 1: Basic BERT with single linear layer"""
    def __init__(self, n_out=1, dropout_rate=0.3):
        super().__init__()
        print("Loading BERT model (this may take a minute)...")
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_out)
        print("BERT model loaded successfully!")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# --- Training Function ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    model = model.train()
    total_loss = 0

    pbar = tqdm(data_loader, desc="Training", ncols=100)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.squeeze(), labels)
        total_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        optimizer.zero_grad()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(data_loader)

# --- Evaluation Function ---
def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary')

    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.5

    return accuracy, precision, recall, f1, auc_roc

# --- Main Execution ---
def main():
    # Load data
    df = load_and_preprocess_data(DATASET_PATH)

    # We'll train on the Mind (I/E) trait
    trait = 'is_I'
    trait_name = 'Mind (Introversion/Extroversion)'

    print(f"\n{'='*70}")
    print(f"Training on trait: {trait_name}")
    print(f"{'='*70}\n")

    # Show trait distribution
    trait_counts = df[trait].value_counts()
    print(f"Trait distribution:")
    print(f"  I (Introvert): {trait_counts[1]} samples ({trait_counts[1]/len(df)*100:.1f}%)")
    print(f"  E (Extrovert): {trait_counts[0]} samples ({trait_counts[0]/len(df)*100:.1f}%)")
    print()

    # Split data
    print("Splitting data (90% train, 10% validation)...")
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42, stratify=df[trait])
    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print()

    # Create tokenizer and data loaders
    print("Creating tokenizer and data loaders...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    train_dataset = MBTITraitDataset(
        df_train['cleaned_posts'].values,
        df_train[trait].values,
        tokenizer,
        MAX_LEN
    )

    val_dataset = MBTITraitDataset(
        df_val['cleaned_posts'].values,
        df_val[trait].values,
        tokenizer,
        MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print()

    # Create model
    print("Initializing Baseline BERT model...")
    model = BaselineBERTClassifier().to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} total parameters")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print()

    # Create optimizer and scheduler
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=BASIC_LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"{'='*70}")
    print(f"Starting training for {EPOCHS} epoch(s)...")
    print(f"{'='*70}\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 70)

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        print(f"\nTraining Loss: {train_loss:.4f}")

        print("\nRunning validation...")
        acc, prc, rec, f1, auc = eval_model(model, val_loader, loss_fn, device)

        print(f"\nValidation Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prc:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print()

    # Final summary
    print(f"{'='*70}")
    print("DEMO COMPLETE!")
    print(f"{'='*70}")
    print(f"\nFinal Results for {trait_name}:")
    print(f"{'─'*70}")
    print(f"  Model: Baseline BERT")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prc:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"{'─'*70}")
    print("\nThis was a quick demo with 1 model and 1 trait.")
    print("The full pipeline (BERTModels.py) trains 4 models on 4 traits (16 models total).")
    print("Each model is trained for 3 epochs for better performance.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
