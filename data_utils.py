"""
Data loading and preprocessing utilities for MBTI personality prediction.
"""

import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Clean and preprocess text data"""
    # Handle None or non-string values
    if pd.isna(text) or text is None:
        return ""

    # Convert to string if not already
    text = str(text)

    # Clean the text
    text = text.replace('|||', ' ')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_data(path):
    """Load MBTI dataset and create binary labels for each trait"""
    # Read CSV with robust error handling for malformed rows
    try:
        df = pd.read_csv(path, on_bad_lines='skip', engine='python', encoding='utf-8')
    except Exception as e:
        print(f"Error with python engine, trying c engine with error handling...")
        df = pd.read_csv(path, on_bad_lines='warn', encoding='utf-8')

    print(f"Loaded {len(df)} rows from dataset")

    # Handle missing values
    df = df.dropna(subset=['type', 'posts'])
    print(f"After removing missing values: {len(df)} rows")

    df['cleaned_posts'] = df['posts'].apply(clean_text)

    # Remove rows with empty cleaned posts
    df = df[df['cleaned_posts'].str.len() > 0]
    print(f"After removing empty posts: {len(df)} rows")

    # Create binary labels for each MBTI dimension
    df['is_I'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)  # Mind: I vs E
    df['is_N'] = df['type'].apply(lambda x: 1 if x[1] == 'N' else 0)  # Energy: N vs S
    df['is_T'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)  # Nature: T vs F
    df['is_P'] = df['type'].apply(lambda x: 1 if x[3] == 'P' else 0)  # Tactics: P vs J

    print(f"Final dataset size: {len(df)} rows")
    return df


def create_train_val_split(df, trait_label, test_size=0.1, random_state=42):
    """Create stratified train/validation split for a specific trait"""
    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[trait_label]
    )
    return df_train, df_val


class MBTITraitDataset(Dataset):
    """PyTorch Dataset for MBTI trait classification"""
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
