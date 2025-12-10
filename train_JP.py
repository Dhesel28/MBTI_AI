"""
Training script for MBTI Tactics dimension (J/P trait).
Trains all 4 model architectures and saves the best performing models.
Can be run independently or in parallel with other trait trainers.
"""

import os
import sys

# Move HuggingFace cache outside git repo to prevent mutex lock issues
HF_CACHE_DIR = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

# Disable git integration and telemetry
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "/bin/false"  # Block git entirely
os.environ["GIT_TERMINAL_PROMPT"] = "0"
os.environ["GCM_INTERACTIVE"] = "never"

# Create cache directory if it doesn't exist
os.makedirs(HF_CACHE_DIR, exist_ok=True)

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoTokenizer

# Import shared modules
from config import (
    DATASET_PATH, MAX_LEN, BATCH_SIZE, EPOCHS, TRAIT_CONFIG, device, USE_GPU,
    BaselineBERTClassifier, BERTDeepHeadClassifier, DeBERTaClassifier, DeBERTaAblationModel,
    BERT_MODEL_NAME, DEBERTA_MODEL_NAME
)
from data_utils import load_and_preprocess_data, create_train_val_split, MBTITraitDataset
from train_utils import train_model, create_optimizer_and_scheduler


def main():
    """Main training pipeline for J/P trait"""
    trait_key = 'JP'
    trait_config = TRAIT_CONFIG[trait_key]
    trait_label = trait_config['label']
    trait_name = trait_config['name']
    save_dir = trait_config['models_dir']

    print(f"\n{'='*70}")
    print(f"TRAINING MODELS FOR TRAIT: {trait_name}")
    print(f"{'='*70}\n")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(DATASET_PATH)
    print(f"Loaded {len(df)} data samples.")

    # Create train/validation split
    df_train, df_val = create_train_val_split(df, trait_label)
    print(f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}")

    # Initialize tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)

    # Create BERT DataLoaders
    bert_train_loader = DataLoader(
        MBTITraitDataset(df_train['cleaned_posts'].values, df_train[trait_label].values,
                        bert_tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 if USE_GPU else 0,
        pin_memory=USE_GPU
    )
    bert_val_loader = DataLoader(
        MBTITraitDataset(df_val['cleaned_posts'].values, df_val[trait_label].values,
                        bert_tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if USE_GPU else 0,
        pin_memory=USE_GPU
    )

    # Create DeBERTa DataLoaders
    deberta_train_loader = DataLoader(
        MBTITraitDataset(df_train['cleaned_posts'].values, df_train[trait_label].values,
                        deberta_tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 if USE_GPU else 0,
        pin_memory=USE_GPU
    )
    deberta_val_loader = DataLoader(
        MBTITraitDataset(df_val['cleaned_posts'].values, df_val[trait_label].values,
                        deberta_tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if USE_GPU else 0,
        pin_memory=USE_GPU
    )

    results = {}

    # --- Model 1: Basic BERT ---
    model = BaselineBERTClassifier().to(device)
    total_steps = len(bert_train_loader) * EPOCHS
    optimizer, scheduler = create_optimizer_and_scheduler(model, 'Basic BERT', total_steps, EPOCHS)

    metrics = train_model(
        model, bert_train_loader, bert_val_loader,
        'Basic_BERT', trait_name, save_dir,
        EPOCHS, optimizer, scheduler, device
    )
    results['Basic BERT'] = metrics
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    # --- Model 2: BERT Deep Head ---
    model = BERTDeepHeadClassifier().to(device)
    total_steps = len(bert_train_loader) * EPOCHS
    optimizer, scheduler = create_optimizer_and_scheduler(model, 'BERT Deep Head', total_steps, EPOCHS)

    metrics = train_model(
        model, bert_train_loader, bert_val_loader,
        'BERT_Deep_Head', trait_name, save_dir,
        EPOCHS, optimizer, scheduler, device
    )
    results['BERT Deep Head'] = metrics
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    # --- Model 3: DeBERTa Deep Head ---
    model = DeBERTaClassifier().to(device)
    total_steps = len(deberta_train_loader) * EPOCHS
    optimizer, scheduler = create_optimizer_and_scheduler(model, 'DeBERTa Deep Head', total_steps, EPOCHS)

    metrics = train_model(
        model, deberta_train_loader, deberta_val_loader,
        'DeBERTa_Deep_Head', trait_name, save_dir,
        EPOCHS, optimizer, scheduler, device
    )
    results['DeBERTa Deep Head'] = metrics
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    # --- Model 4: DeBERTa AttnPool Deep ---
    model = DeBERTaAblationModel().to(device)
    total_steps = len(deberta_train_loader) * EPOCHS
    optimizer, scheduler = create_optimizer_and_scheduler(model, 'DeBERTa AttnPool Deep', total_steps, EPOCHS)

    metrics = train_model(
        model, deberta_train_loader, deberta_val_loader,
        'DeBERTa_AttnPool_Deep', trait_name, save_dir,
        EPOCHS, optimizer, scheduler, device
    )
    results['DeBERTa AttnPool Deep'] = metrics
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    # --- Display Results ---
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE FOR TRAIT: {trait_name}")
    print(f"{'='*70}\n")

    results_df = pd.DataFrame.from_dict(results, orient='index')
    print(results_df.to_string())
    print(f"\nModels saved to: {save_dir}")

    # Save results summary
    os.makedirs(save_dir, exist_ok=True)
    results_df.to_csv(os.path.join(save_dir, 'results_summary.csv'))


if __name__ == "__main__":
    main()
