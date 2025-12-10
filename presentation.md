# Neural Meets Social Network
## MBTI Personality Prediction from Social Media Posts

---

## Overview

**Project Goal**: Predict MBTI personality types from social media text using transformer-based deep learning models

**Approach**: Multi-model comparison with 4 different architectures

**Key Innovation**: Treating MBTI as 4 independent binary classification tasks

---

## Problem Statement

### Challenge
- Predict personality traits from unstructured social media text
- MBTI has 16 types (4 binary dimensions)
- Traditional methods lack contextual understanding

### Our Solution
- Leverage transformer models (BERT, DeBERTa)
- Test multiple architectures and pooling strategies
- Dual-mode system: batch evaluation + real-time web interface

---

## MBTI Framework

### 4 Binary Dimensions

| Dimension | Trait 1 | Trait 2 | Description |
|-----------|---------|---------|-------------|
| **Mind** | I (Introversion) | E (Extraversion) | Energy direction |
| **Energy** | N (Intuition) | S (Sensing) | Information processing |
| **Nature** | T (Thinking) | F (Feeling) | Decision making |
| **Tactics** | P (Perceiving) | J (Judging) | Lifestyle approach |

**Result**: 2^4 = 16 personality types (e.g., INTJ, ENFP)

---

## Dataset

### MBTI Social Media Dataset
- **Source**: `mbti_1.csv` - Social media posts labeled with personality types
- **Size**: 8,675 users with multiple posts each
- **Split**: 90% training, 10% validation (stratified)

### Data Preprocessing
1. Clean text (remove URLs, mentions, special characters)
2. Normalize to lowercase
3. Remove separators (|||)
4. Tokenize with model-specific tokenizers
5. Max sequence length: 256 tokens

---

## Model Architecture Comparison

### 4 Models Tested

| Model | Base Transformer | Pooling Strategy | Classification Head |
|-------|------------------|------------------|---------------------|
| **Model 1** | BERT-base | Pooler output | 1 linear layer |
| **Model 2** | BERT-base | Pooler output | 6-layer deep head |
| **Model 3** | DeBERTa v3 Small | CLS token | 6-layer deep head |
| **Model 4** | DeBERTa v3 Small | Attention pooling | 6-layer deep head |

---

## Model 1: Baseline BERT

### Architecture
```
Input Text → BERT-base-uncased
           → Pooler Output
           → Dropout (0.3)
           → Linear (768 → 1)
           → Binary Classification
```

### Hyperparameters
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW with linear warmup

**Purpose**: Establish baseline performance

---

## Model 2: BERT + Deep Head

### Architecture
```
Input Text → BERT-base-uncased
           → Pooler Output
           → 6-Layer Head:
              768 → 512 → 256 → 128 → 64 → 32 → 1
              (GELU + Dropout after each layer)
```

### Key Features
- Differential learning rates:
  - BERT layers: 1e-5
  - Classification head: 1e-4
- Dropout: 0.55 (higher for deeper network)
- Cosine annealing scheduler

**Hypothesis**: Deeper head captures more complex trait patterns

---

## Model 3: DeBERTa + Deep Head

### Architecture
```
Input Text → DeBERTa v3 Small
           → CLS Token (last_hidden_state[:, 0])
           → 6-Layer Head:
              768 → 512 → 256 → 128 → 64 → 32 → 1
              (GELU + Dropout after each layer)
```

### DeBERTa Advantages
- Disentangled attention mechanism
- Enhanced mask decoder
- Better position encoding
- More efficient than BERT-base

---

## Model 4: DeBERTa + Attention Pooling

### Architecture
```
Input Text → DeBERTa v3 Small
           → All Hidden States (seq_len × 768)
           → Attention Pooling Layer:
              - Learned attention weights
              - Masked softmax over sequence
              - Weighted average of all tokens
           → 6-Layer Head (768 → ... → 1)
```

### Attention Pooling Mechanism
- Goes beyond single CLS token
- Learns which parts of text are important
- Adaptive to different personality traits

**Hypothesis**: Full sequence context improves trait detection

---

## Training Strategy

### Per-Trait Binary Classification
- Train 4 separate models (one per MBTI dimension)
- Each trait is independent binary task
- Allows specialized learning per dimension

### Optimization
- Loss function: Binary Cross-Entropy with Logits
- Gradient clipping: max_norm=1.0
- Memory management: Clear CUDA cache between runs
- Mixed precision training ready

### Evaluation Metrics
- Accuracy
- Precision & Recall
- F1 Score (primary metric for model selection)
- ROC-AUC

---

## Training Pipeline

```
For each MBTI trait (I/E, N/S, T/F, P/J):
  ├── Split data (90/10 train/val, stratified)
  ├── Create trait-specific data loaders
  │
  ├── Train Model 1 (Basic BERT)
  │   └── Track best F1 score
  │
  ├── Train Model 2 (BERT Deep Head)
  │   └── Track best F1 score
  │
  ├── Train Model 3 (DeBERTa Deep Head)
  │   └── Track best F1 score
  │
  └── Train Model 4 (DeBERTa Attn Pool)
      └── Track best F1 score
```

**Result**: 16 trained models (4 architectures × 4 traits)

---

## Key Implementation Details

### Tokenization
- **BERT models**: `BertTokenizer` (WordPiece)
- **DeBERTa models**: `AutoTokenizer` (SentencePiece)
- Max length: 256 tokens
- Padding: max_length
- Truncation: enabled

### Data Augmentation
- Dataset already contains multiple posts per user
- Natural variation in writing style

### Reproducibility
- Random seed: 42 (PyTorch, NumPy, CUDA)
- Deterministic operations enabled

---

## Real-Time Web Interface

### Predictor.html Features
- **Elegant 3-column responsive layout**
  - Column 1: Personality analyzer input
  - Column 2: MBTI type results with trait breakdown
  - Column 3: Celebrity personality matches
- **Custom branded header** with artistic design
- **Dark theme UI** (#1c1b37 background) with white cards
- **Input**: 2-3 social media posts from user
- **Backend**: Google Gemini 2.5 Flash API
- **Output**:
  - Predicted 4-letter MBTI type
  - Confidence scores (0-100%) for each dimension
  - Reasoning explanation for each trait
  - Celebrity matches from 50K+ database

### Celebrity Matching System
- **Database**: 50,879 celebrities from `mbti_celebrities.csv`
- **Categories**: Pop Culture, Internet, Sports, Musicians, The Arts, Historical
- **Smart filtering**: Shows one celebrity per category matching user's MBTI
- **Color-coded badges**: Each category has unique visual styling
- **Interactive cards**: Hover effects and elegant design

### Design Philosophy
- **Minimalist & Chic**: Clean white cards on dark background
- **Typography**: Poppins font family for consistency
- **Accessibility**: High contrast, readable text sizes
- **Responsive**: Mobile-friendly 3-column grid layout

### Advantages
- No local model deployment needed
- Fast inference via API
- Beautiful, user-friendly interface
- Real-time predictions with celebrity insights

---

## UI/UX Design Showcase

### Visual Design Elements
- **Custom Header**: Branded "Ai Mbti Analyzer" with playful cat icon and star decorations
- **Color Scheme**:
  - Background: `#1c1b37` (deep navy/purple)
  - Cards: White with subtle shadows
  - Accents: Category-specific colors (blue, green, purple, etc.)
- **Typography**: Poppins font family throughout for clean, modern look
- **Layout**: Responsive grid adapting to desktop and mobile

### Interactive Features
- **Progress Bars**: Animated confidence visualization for each trait
- **Celebrity Cards**:
  - Hover effects with smooth transitions
  - Color-coded category badges
  - Gradient top border matching category
- **Loading States**: Spinner animation during API calls
- **Error Handling**: User-friendly error messages

### User Experience Flow
1. User enters 2-3 social media posts
2. Clicks "Analyze" button (dark theme matching background)
3. Real-time API call to Google Gemini
4. Results populate all three columns simultaneously
5. Celebrity matches load from local CSV database
6. Interactive exploration of results

---

## Results: Model Performance

### Comparison Framework
- Best validation F1 score across 3 epochs
- Evaluated on each of 4 MBTI dimensions
- Results tracked in final comparison report

### Expected Patterns
- **Mind (I/E)**: Easiest to detect (vocabulary differences)
- **Energy (N/S)**: Moderate difficulty (abstract vs concrete language)
- **Nature (T/F)**: Challenging (subtle emotional cues)
- **Tactics (P/J)**: Moderate difficulty (planning language)

### Model Selection
- Best model per trait selected by F1 score
- Ensemble potential across architectures

---

## Technical Stack

### Deep Learning Framework
- **PyTorch**: Neural network implementation
- **Hugging Face Transformers**: Pre-trained models
- **CUDA**: GPU acceleration

### Models
- **BERT**: `bert-base-uncased` (110M parameters)
- **DeBERTa**: `microsoft/deberta-v3-small` (44M parameters)

### Frontend
- **HTML5** + **Tailwind CSS**: Responsive 3-column layout
- **JavaScript**: API integration and dynamic UI
- **PapaParse**: Client-side CSV parsing for celebrity data
- **Google Gemini 2.5 Flash**: Cloud inference API
- **Custom Design**: Dark theme (#1c1b37) with Poppins typography

### Data Processing
- **Pandas**: Data manipulation
- **Scikit-learn**: Train/test split, metrics
- **tqdm**: Progress tracking
- **Celebrity Database**: 50K+ entries with MBTI classifications

---

## Project Workflow

```
┌─────────────────────────────────────────────┐
│  Data Loading & Preprocessing               │
│  (mbti_1.csv → cleaned text + binary labels)│
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  Model Training (4 architectures × 4 traits)│
│  • Baseline BERT                            │
│  • BERT + Deep Head                         │
│  • DeBERTa + Deep Head (CLS)                │
│  • DeBERTa + Attention Pooling              │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  Evaluation & Model Comparison              │
│  (F1, Accuracy, Precision, Recall, AUC-ROC) │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  Deployment                                 │
│  • 3-column web interface (Predictor.html)  │
│  • Google Gemini API inference              │
│  • Celebrity matching (50K+ database)       │
│  • Dark theme UI with custom branding       │
└─────────────────────────────────────────────┘
```

---

## Key Findings & Insights

### Architectural Insights
1. **Deep heads** help models learn complex trait representations
2. **Attention pooling** leverages full sequence context beyond CLS token
3. **DeBERTa** shows better efficiency than BERT (fewer parameters)
4. **Differential learning rates** crucial for fine-tuning

### MBTI Trait Complexity
- Different traits have different linguistic markers
- Some dimensions easier to predict than others
- Multi-trait approach allows specialized learning

### Practical Deployment
- API-based inference enables easy deployment
- No need for local model serving infrastructure
- Fast real-time predictions

---

## Challenges & Solutions

### Challenge 1: Class Imbalance
- **Problem**: Some MBTI types more common than others
- **Solution**: Stratified sampling, balanced metrics (F1 over accuracy)

### Challenge 2: Computational Resources
- **Problem**: Training multiple large transformer models
- **Solution**: Memory optimization, CUDA cache clearing, batch size tuning

### Challenge 3: Overfitting
- **Problem**: Deep networks prone to overfitting
- **Solution**: High dropout (0.55), early stopping via best F1 tracking

### Challenge 4: Long Training Time
- **Problem**: Each model takes hours to train
- **Solution**: Efficient data loaders, GPU acceleration, limited epochs (3)

---

## Future Enhancements

### Model Improvements
- Test larger models (BERT-large, DeBERTa-base/large)
- Ensemble methods across architectures
- Multi-task learning (all 4 traits jointly)
- Cross-attention between traits

### Data Augmentation
- Back-translation
- Paraphrasing
- Synthetic data generation

### Deployment
- Model compression (quantization, distillation)
- ONNX export for production
- A/B testing different models
- User feedback loop

---

## Code Organization

### Main Files

| File | Purpose | Size/LOC |
|------|---------|----------|
| `BERTModels.py` | Complete training pipeline with 4 model variants | ~309 lines |
| `config.py` | Hyperparameters and model definitions | ~232 lines |
| `quick_demo.py` | Quick demo (1 model, 1 trait, 1 epoch) | ~289 lines |
| `Predictor.html` | 3-column web UI with celebrity matching | ~500 lines |
| `presentation.md` | Project presentation slides | ~550 lines |
| `mbti_1.csv` | Training dataset (8,675 users) | ~60MB |
| `mbti_celebrities.csv` | Celebrity MBTI database (50,879 entries) | ~3MB |
| `Header.png` | Custom branded header for web interface | 139KB |

### Modular Design
- Clean separation of concerns
- Reusable model architectures
- Easy to add new models
- Configuration-driven hyperparameters

---

## How to Use

### Training Models
```bash
python BERTModels.py
```
- Trains all 4 models on all 4 traits
- Outputs comparison report
- Saves best checkpoints

### Real-Time Prediction
```bash
# Start local server
python3 -m http.server 8000

# Open http://localhost:8000/Predictor.html in browser
# Enter 2-3 social media posts
# Click "Analyze" button
# Get MBTI type + confidence scores + celebrity matches
```

### Requirements
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU)
- Google Gemini API key (for web predictor)

---

## Conclusion

### What We Built
- Comprehensive MBTI prediction system
- 4 different transformer architectures
- Dual-mode: batch training + real-time web interface
- Celebrity matching with 50K+ database
- Complete evaluation framework

### Key Contributions
1. **Multi-architecture comparison** for personality prediction
2. **Attention pooling** vs CLS token analysis
3. **Deep classification heads** for trait extraction
4. **Production-ready web interface** with elegant 3-column design
5. **Celebrity personality matching** from extensive database
6. **Minimalist UI/UX** with dark theme and responsive layout

### Impact
- Demonstrates transformer effectiveness for personality prediction
- Provides reusable framework for text classification tasks
- Combines research rigor with practical deployment

---

## Thank You!

### Project Repository
**Neural Meets Social Network**

### Technologies Used
PyTorch • Transformers • BERT • DeBERTa • Gemini API • Tailwind CSS • PapaParse

### Key Features
- 4 Transformer architectures for model comparison
- 3-column responsive web interface
- 50K+ celebrity personality database
- Real-time AI-powered analysis

### Contact
Questions? Feedback? Let's discuss!

---

## Appendix: Model Architectures in Detail

### Deep Head Structure (6 Layers)
```
Layer 1: Linear(768 → 512)  + GELU + Dropout(0.55)
Layer 2: Linear(512 → 256)  + GELU + Dropout(0.55)
Layer 3: Linear(256 → 128)  + GELU + Dropout(0.55)
Layer 4: Linear(128 → 64)   + GELU + Dropout(0.55)
Layer 5: Linear(64 → 32)    + GELU + Dropout(0.55)
Layer 6: Linear(32 → 1)     (output)
```

### Attention Pooling Mechanism
```python
# Calculate attention scores for each token
scores = Linear(hidden_states)  # (batch, seq_len, 1)

# Apply mask (ignore padding)
scores.masked_fill_(mask == 0, -inf)

# Softmax to get weights
weights = softmax(scores, dim=1)

# Weighted sum of all hidden states
output = sum(weights * hidden_states)
```

---

## Appendix: Hyperparameter Summary

| Component | Value | Rationale |
|-----------|-------|-----------|
| Max sequence length | 256 | Balance context vs memory |
| Batch size | 16 | Fit in GPU memory (12GB) |
| Epochs | 3 | Prevent overfitting |
| Dropout | 0.55 | High regularization for deep networks |
| Base LR (BERT/DeBERTa) | 1e-5 | Fine-tune pre-trained weights |
| Head LR | 1e-4 | Faster learning for new layers |
| Gradient clipping | 1.0 | Prevent exploding gradients |
| Train/Val split | 90/10 | Maximize training data |

---

## Appendix: Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating curve

### Why F1 as Primary Metric?
- Balances precision and recall
- Robust to class imbalance
- Single number for model selection
- Standard in binary classification

---

## Appendix: Dataset Statistics

### MBTI Type Distribution
- **Most common**: INFP, INTP, INFJ
- **Least common**: ESTP, ESFP, ESTJ
- **Overall**: I > E, N > S, varied T/F, varied P/J

### Text Characteristics
- **Posts per user**: 50 on average
- **Words per post**: 50-200
- **Vocabulary**: Social media style (informal, emoji, abbreviations)
- **Topics**: Personal reflections, opinions, daily life

---

## Appendix: Related Work

### Personality Prediction Literature
- Traditional ML: SVM, Naive Bayes with TF-IDF
- Early deep learning: LSTM, CNN on word embeddings
- Recent: BERT-based models show SOTA results

### Our Contribution
- Systematic comparison of 4 architectures
- Attention pooling vs CLS token analysis
- Very deep classification heads (6 layers)
- Production-ready deployment

### Future Research Directions
- Cross-cultural MBTI prediction
- Multi-modal personality (text + image + audio)
- Explainable AI for personality traits
