# Neural Meets Social Network - MBTI Personality Prediction

## Overview
This project uses transformer-based deep learning models to predict MBTI personality types from social media posts. It features a comprehensive model comparison framework, an elegant 3-column web interface with celebrity matching, and dual-mode prediction capabilities (batch training + real-time inference).

## Features

- **3-Column Responsive Layout**: Elegant dark theme (#1c1b37) web interface
- **Celebrity Matching System**: Integration with 50,879+ celebrity MBTI database across 6 categories (Pop Culture, Internet, Sports, Musicians, The Arts, Historical)
- **Custom Branding**: Artistic header with "AI MBTI Analyzer" design
- **Interactive UI**: Animated progress bars, hover effects, and category-colored celebrity cards
- **Presentation Materials**: Comprehensive `presentation.md` with project documentation

## High-Level Workflow

### 1. Data Preparation
- **Input**: MBTI dataset (`mbti_1.csv`) containing 8,675 users with social media posts
- **Preprocessing**: Clean text (remove URLs, mentions, special characters), normalize to lowercase
- **Feature Engineering**: Extract 4 binary traits from MBTI types:
  - I/E (Introversion/Extroversion)
  - N/S (Intuition/Sensing)
  - T/F (Thinking/Feeling)
  - P/J (Perceiving/Judging)
- **Split**: 90% training, 10% validation (stratified)

### 2. Model Training (`BERTModels.py`)
The project trains and compares **4 different model architectures**:

| Model | Base Transformer | Pooling Strategy | Classification Head |
|-------|------------------|------------------|---------------------|
| Model 1 | BERT-base | Pooler output | 1 linear layer |
| Model 2 | BERT-base | Pooler output | 6-layer deep head |
| Model 3 | DeBERTa v3 Small | CLS token | 6-layer deep head |
| Model 4 | DeBERTa v3 Small | Attention pooling | 6-layer deep head |

**Training Process**:
- Each model is trained for 3 epochs on each of the 4 MBTI traits
- Batch size: 16, Learning rate: 1e-5 to 2e-5
- Optimization: AdamW with gradient clipping
- Differential learning rates: transformer base (1e-5) vs classification head (1e-4)

### 3. Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Output**: Comparison report across all models and traits
- **Best Model Selection**: Tracked by F1 score per trait

### 4. Real-Time Prediction (`Predictor.html`)
- **3-Column Web Interface**:
  - Column 1: Personality analyzer input
  - Column 2: MBTI type results with trait breakdown
  - Column 3: Celebrity personality matches
- **Backend**: Google Gemini 2.5 Flash API for inference
- **Celebrity Matching**: Shows one celebrity per category matching user's predicted MBTI
- **Output**:
  - Predicted MBTI type (4 letters)
  - Confidence scores for each dimension (0-100%)
  - Reasoning explanation for each trait
  - 6 celebrity matches with color-coded category badges

## Data Flow

```
Social Media Text
      ↓
Text Cleaning & Tokenization
      ↓
Transformer Encoding (BERT/DeBERTa)
      ↓
Classification Head (Deep/Simple)
      ↓
MBTI Trait Prediction (4 binary outputs)
      ↓
Evaluation & Comparison
      ↓
Celebrity Matching (50K+ database)
```

## Project Structure

| File | Purpose | Details |
|------|---------|---------|
| `BERTModels.py` | Training pipeline with 4 model variants | ~309 lines |
| `Predictor.html` | 3-column web UI with celebrity matching | ~500 lines |
| `config.py` | Hyperparameters and model definitions | ~232 lines |
| `presentation.md` | Project presentation slides | ~550 lines |
| `mbti_1.csv` | Training dataset (8,675 users) | ~60MB |
| `mbti_celebrities.csv` | Celebrity MBTI database (50,879 entries) | ~3MB |
| `Header.png` | Custom branded header for web interface | 139KB |

## Technologies Used
- **Deep Learning**: PyTorch 2.0+, Hugging Face Transformers 4.30+
- **Models**: BERT (bert-base-uncased, 110M params), DeBERTa v3 Small (44M params)
- **Frontend**: HTML5, Tailwind CSS, JavaScript, PapaParse
- **API**: Google Gemini 2.5 Flash
- **Design**: Dark theme (#1c1b37), Poppins typography, responsive grid

## How to Use

### Training Models
```bash
# Full training (all 4 models × 4 traits)
python BERTModels.py
```

### Real-Time Prediction
```bash
# Start local server
python3 -m http.server 8000

# Open browser to http://localhost:8000/Predictor.html
# Enter 2-3 social media posts → Click "Analyze"
# Get MBTI type + confidence scores + celebrity matches
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU acceleration)
- Google Gemini API key (for web predictor)

## Project Goals
1. Benchmark multiple transformer architectures for personality classification
2. Compare different pooling strategies (CLS token vs attention pooling)
3. Evaluate classifier head depths (1-layer vs 6-layer deep head)
4. Provide dual-mode prediction: batch evaluation + real-time web interface
5. Enable celebrity personality matching from extensive database

## Notes
- Multi-trait approach trains 4 separate binary classifiers per model (16 total models)
- Memory-optimized with GPU cache clearing between training runs
- Web predictor uses API inference—no local model deployment required
- Celebrity matching filters to show one match per category for variety
- Responsive design works on desktop and mobile devices
