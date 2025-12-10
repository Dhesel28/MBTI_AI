"""
Training and evaluation utilities for MBTI personality prediction models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)
from tqdm import tqdm
import os


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    """Train model for one epoch"""
    model = model.train()
    total_loss = 0

    pbar = tqdm(data_loader, desc="Training")
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
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)


def eval_model(model, data_loader, loss_fn, device):
    """Evaluate model on validation set"""
    model = model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
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


def train_model(model, train_loader, val_loader, model_name, trait_name, save_dir,
                epochs, optimizer, scheduler=None, device='cpu'):
    """Complete training loop for a model"""
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    best_f1 = 0
    best_metrics = {}
    best_model_state = None

    print(f"\n--- Training {model_name} for: {trait_name} ---")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler=None)

        acc, prc, rec, f1, auc = eval_model(model, val_loader, loss_fn, device)

        if scheduler and not isinstance(scheduler, type(None)):
            scheduler.step()

        print(f"Val F1: {f1:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'Accuracy': acc,
                'Precision': prc,
                'Recall': rec,
                'F1': f1,
                'AUC-ROC': auc
            }
            best_model_state = model.state_dict().copy()

    # Save best model
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'metrics': best_metrics,
        'model_name': model_name,
        'trait_name': trait_name
    }, save_path)
    print(f"Saved best model to: {save_path}")

    return best_metrics


def create_optimizer_and_scheduler(model, model_type, total_steps, epochs):
    """Create optimizer and scheduler based on model type"""
    from config import (
        BASIC_LR, BERT_DEEP_HEAD_LR, BERT_DEEP_BERT_LR,
        DEBERTA_DEEP_HEAD_LR, DEBERTA_DEEP_BERT_LR,
        ABLATION_HEAD_LR, ABLATION_BERT_LR
    )

    if model_type == 'Basic BERT':
        optimizer = AdamW(model.parameters(), lr=BASIC_LR)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    elif model_type == 'BERT Deep Head':
        optimizer_grouped_parameters = [
            {"params": model.bert.parameters(), "lr": BERT_DEEP_BERT_LR},
            {"params": model.head.parameters(), "lr": BERT_DEEP_HEAD_LR}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs)
    elif model_type == 'DeBERTa Deep Head':
        optimizer_grouped_parameters = [
            {"params": model.bert.parameters(), "lr": DEBERTA_DEEP_BERT_LR},
            {"params": model.head.parameters(), "lr": DEBERTA_DEEP_HEAD_LR}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs)
    elif model_type == 'DeBERTa AttnPool Deep':
        optimizer_grouped_parameters = [
            {"params": model.bert.parameters(), "lr": ABLATION_BERT_LR},
            {"params": model.attention_pooling.parameters(), "lr": ABLATION_HEAD_LR},
            {"params": model.head.parameters(), "lr": ABLATION_HEAD_LR}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return optimizer, scheduler
