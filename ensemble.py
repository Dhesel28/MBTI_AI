"""
Ensemble model for MBTI personality prediction.
Combines predictions from all four trait dimensions (E/I, N/S, T/F, J/P)
to generate complete MBTI personality types.
"""

import os

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
from transformers import BertTokenizer, AutoTokenizer

from config import (
    TRAIT_CONFIG, device, MAX_LEN,
    BaselineBERTClassifier, BERTDeepHeadClassifier,
    DeBERTaClassifier, DeBERTaAblationModel,
    BERT_MODEL_NAME, DEBERTA_MODEL_NAME
)


class MBTIEnsemble:
    """
    Ensemble model that loads trained models for each MBTI dimension
    and combines their predictions to determine full personality type.
    """

    def __init__(self, model_type='DeBERTa_AttnPool_Deep'):
        """
        Initialize ensemble with specified model type.

        Args:
            model_type: Which model architecture to use
                       ('Basic_BERT', 'BERT_Deep_Head', 'DeBERTa_Deep_Head', 'DeBERTa_AttnPool_Deep')
        """
        self.model_type = model_type
        self.models = {}
        self.tokenizers = {}

        # Determine if we need BERT or DeBERTa tokenizer
        if 'BERT' in model_type and 'DeBERTa' not in model_type:
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.tokenizer_type = 'BERT'
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
            self.tokenizer_type = 'DeBERTa'

        # Load models for each trait
        self._load_models()

    def _get_model_class(self):
        """Get the appropriate model class based on model type"""
        if self.model_type == 'Basic_BERT':
            return BaselineBERTClassifier
        elif self.model_type == 'BERT_Deep_Head':
            return BERTDeepHeadClassifier
        elif self.model_type == 'DeBERTa_Deep_Head':
            return DeBERTaClassifier
        elif self.model_type == 'DeBERTa_AttnPool_Deep':
            return DeBERTaAblationModel
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_models(self):
        """Load trained models for each MBTI dimension"""
        model_class = self._get_model_class()

        for trait_key, config in TRAIT_CONFIG.items():
            model_path = os.path.join(config['models_dir'], f"{self.model_type}.pth")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Please train the model first using train_{trait_key}.py"
                )

            # Load model
            model = model_class().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.models[trait_key] = model
            print(f"Loaded {trait_key} model from {model_path}")
            print(f"  Metrics: {checkpoint['metrics']}")

    def predict_traits(self, text):
        """
        Predict all four MBTI traits for a given text.

        Args:
            text: Input text to analyze

        Returns:
            dict: Predictions for each trait with probabilities
        """
        # Tokenize input
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        predictions = {}

        with torch.no_grad():
            for trait_key, model in self.models.items():
                # Get model prediction
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                prob = torch.sigmoid(outputs).item()

                # Determine trait label
                config = TRAIT_CONFIG[trait_key]
                label = config['label']

                predictions[trait_key] = {
                    'probability': prob,
                    'prediction': int(prob > 0.5),
                    'label': label
                }

        return predictions

    def predict_personality(self, text):
        """
        Predict complete MBTI personality type for a given text.

        Args:
            text: Input text to analyze

        Returns:
            dict: Complete MBTI prediction with type and confidence scores
        """
        traits = self.predict_traits(text)

        # Map predictions to MBTI letters
        personality = ""
        confidence_scores = {}

        # E/I (Mind)
        ei_prob = traits['EI']['probability']
        personality += 'I' if ei_prob > 0.5 else 'E'
        confidence_scores['EI'] = ei_prob if ei_prob > 0.5 else (1 - ei_prob)

        # N/S (Energy)
        ns_prob = traits['NS']['probability']
        personality += 'N' if ns_prob > 0.5 else 'S'
        confidence_scores['NS'] = ns_prob if ns_prob > 0.5 else (1 - ns_prob)

        # T/F (Nature)
        tf_prob = traits['TF']['probability']
        personality += 'T' if tf_prob > 0.5 else 'F'
        confidence_scores['TF'] = tf_prob if tf_prob > 0.5 else (1 - tf_prob)

        # J/P (Tactics)
        jp_prob = traits['JP']['probability']
        personality += 'P' if jp_prob > 0.5 else 'J'
        confidence_scores['JP'] = jp_prob if jp_prob > 0.5 else (1 - jp_prob)

        # Calculate overall confidence (average of individual confidences)
        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

        return {
            'personality_type': personality,
            'overall_confidence': overall_confidence,
            'trait_confidences': confidence_scores,
            'raw_predictions': traits
        }

    def predict_batch(self, texts):
        """
        Predict MBTI personality types for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            list: List of predictions for each text
        """
        predictions = []
        for text in texts:
            pred = self.predict_personality(text)
            predictions.append(pred)
        return predictions


def main():
    """Demo of ensemble prediction"""
    import argparse

    parser = argparse.ArgumentParser(description='MBTI Ensemble Prediction')
    parser.add_argument('--model', type=str, default='DeBERTa_AttnPool_Deep',
                       choices=['Basic_BERT', 'BERT_Deep_Head', 'DeBERTa_Deep_Head', 'DeBERTa_AttnPool_Deep'],
                       help='Model type to use for ensemble')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze (optional, will use demo text if not provided)')

    args = parser.parse_args()

    print(f"\nInitializing MBTI Ensemble with {args.model}...")
    ensemble = MBTIEnsemble(model_type=args.model)

    # Demo text
    if args.text:
        demo_text = args.text
    else:
        demo_text = """
        I love spending time alone reading and thinking about deep philosophical questions.
        I prefer to plan everything in advance and stick to schedules. When making decisions,
        I rely heavily on logic and objective analysis rather than emotions. I'm fascinated
        by abstract theories and future possibilities rather than practical details.
        """

    print("\n" + "="*70)
    print("MBTI Personality Prediction")
    print("="*70)
    print(f"\nInput text: {demo_text[:200]}...")

    prediction = ensemble.predict_personality(demo_text)

    print(f"\nPredicted Personality Type: {prediction['personality_type']}")
    print(f"Overall Confidence: {prediction['overall_confidence']:.4f}")
    print("\nTrait Confidences:")
    for trait, conf in prediction['trait_confidences'].items():
        print(f"  {trait}: {conf:.4f}")

    print("\nDetailed Predictions:")
    for trait_key, trait_data in prediction['raw_predictions'].items():
        config = TRAIT_CONFIG[trait_key]
        print(f"  {config['name']}: {trait_data['probability']:.4f}")


if __name__ == "__main__":
    main()
