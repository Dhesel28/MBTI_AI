"""
Main pipeline for MBTI personality prediction system.
Orchestrates training of all trait models and ensemble prediction.

Usage:
    # Train all models sequentially
    python main_pipeline.py --mode train

    # Train specific trait only
    python main_pipeline.py --mode train --trait EI

    # Test ensemble on sample data
    python main_pipeline.py --mode predict

    # Train in parallel (manually run these commands in separate terminals):
    python train_EI.py &
    python train_NS.py &
    python train_TF.py &
    python train_JP.py &
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
from datetime import datetime


def train_all_traits_sequential():
    """Train all trait models sequentially"""
    traits = ['EI', 'NS', 'TF', 'JP']
    training_scripts = {
        'EI': 'train_EI.py',
        'NS': 'train_NS.py',
        'TF': 'train_TF.py',
        'JP': 'train_JP.py'
    }

    print("\n" + "="*70)
    print("MBTI TRAINING PIPELINE - SEQUENTIAL MODE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will train all 4 traits sequentially.")
    print("Each trait trains 4 different model architectures.")
    print("Total: 16 models will be trained.\n")

    for trait in traits:
        print(f"\n{'='*70}")
        print(f"Training trait: {trait}")
        print(f"{'='*70}\n")

        script = training_scripts[trait]
        result = subprocess.run([sys.executable, script], capture_output=False)

        if result.returncode != 0:
            print(f"\nERROR: Training failed for trait {trait}")
            sys.exit(1)

    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_training_summary()


def train_single_trait(trait):
    """Train models for a single trait"""
    training_scripts = {
        'EI': 'train_EI.py',
        'NS': 'train_NS.py',
        'TF': 'train_TF.py',
        'JP': 'train_JP.py'
    }

    if trait not in training_scripts:
        print(f"ERROR: Unknown trait '{trait}'. Must be one of: {list(training_scripts.keys())}")
        sys.exit(1)

    print(f"\nTraining trait: {trait}")
    script = training_scripts[trait]
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\nERROR: Training failed for trait {trait}")
        sys.exit(1)


def print_parallel_instructions():
    """Print instructions for parallel training"""
    print("\n" + "="*70)
    print("PARALLEL TRAINING INSTRUCTIONS")
    print("="*70)
    print("\nTo train all traits in parallel, open 4 separate terminal windows")
    print("and run one of these commands in each:\n")
    print("  Terminal 1: python train_EI.py")
    print("  Terminal 2: python train_NS.py")
    print("  Terminal 3: python train_TF.py")
    print("  Terminal 4: python train_JP.py")
    print("\nThis will train all 4 traits simultaneously, significantly reducing")
    print("total training time.")
    print("="*70 + "\n")


def print_training_summary():
    """Print summary of all trained models"""
    from config import TRAIT_CONFIG

    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70 + "\n")

    all_results = {}

    for trait_key, config in TRAIT_CONFIG.items():
        models_dir = config['models_dir']
        trait_name = config['name']
        results_file = os.path.join(models_dir, 'results_summary.csv')

        if os.path.exists(results_file):
            print(f"\n{trait_name}:")
            df = pd.read_csv(results_file, index_col=0)
            print(df.to_string())

            # Find best model
            best_model = df['F1'].idxmax()
            best_f1 = df['F1'].max()
            print(f"  → Best Model: {best_model} (F1: {best_f1:.4f})")

            all_results[trait_key] = {
                'trait': trait_name,
                'best_model': best_model,
                'best_f1': best_f1
            }
        else:
            print(f"\n{trait_name}: Not yet trained")

    if all_results:
        print("\n" + "="*70)
        print("OVERALL BEST MODELS")
        print("="*70)
        for trait_key, result in all_results.items():
            print(f"{result['trait']:20s} → {result['best_model']:25s} (F1: {result['best_f1']:.4f})")


def test_ensemble_prediction():
    """Test ensemble prediction on sample data"""
    from ensemble import MBTIEnsemble

    print("\n" + "="*70)
    print("TESTING ENSEMBLE PREDICTION")
    print("="*70 + "\n")

    # Check if models exist
    from config import TRAIT_CONFIG
    model_type = 'DeBERTa_AttnPool_Deep'  # Use the best model by default

    missing_models = []
    for trait_key, config in TRAIT_CONFIG.items():
        model_path = os.path.join(config['models_dir'], f"{model_type}.pth")
        if not os.path.exists(model_path):
            missing_models.append(trait_key)

    if missing_models:
        print(f"ERROR: Missing trained models for traits: {missing_models}")
        print("Please train these models first using --mode train")
        sys.exit(1)

    # Load ensemble
    print("Loading ensemble model...")
    ensemble = MBTIEnsemble(model_type=model_type)

    # Test cases
    test_texts = [
        {
            'name': 'Introverted Thinker',
            'text': """I prefer working alone on complex problems. I spend a lot of time
            analyzing systems and thinking about how things work. I make decisions based on
            logic rather than feelings, and I like to have everything planned out in advance."""
        },
        {
            'name': 'Extroverted Feeler',
            'text': """I love being around people and helping others. I make decisions based
            on how they will affect people's feelings. I'm very spontaneous and prefer to keep
            my options open. I focus on the here and now rather than future possibilities."""
        },
        {
            'name': 'Creative Intuitive',
            'text': """I'm always thinking about future possibilities and abstract concepts.
            I love brainstorming new ideas and exploring theoretical frameworks. I prefer
            flexibility over rigid schedules, and I trust my gut feelings when making decisions."""
        }
    ]

    print("\nRunning predictions on test cases...\n")
    for test_case in test_texts:
        print(f"\n{'-'*70}")
        print(f"Test Case: {test_case['name']}")
        print(f"{'-'*70}")
        print(f"Text: {test_case['text'][:150]}...")

        prediction = ensemble.predict_personality(test_case['text'])

        print(f"\nPredicted Type: {prediction['personality_type']}")
        print(f"Confidence: {prediction['overall_confidence']:.4f}")
        print("\nTrait Breakdown:")
        for trait, conf in prediction['trait_confidences'].items():
            print(f"  {trait}: {conf:.4f}")

    print("\n" + "="*70)
    print("ENSEMBLE TESTING COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='MBTI Personality Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all traits sequentially
  python main_pipeline.py --mode train

  # Train specific trait
  python main_pipeline.py --mode train --trait EI

  # View parallel training instructions
  python main_pipeline.py --mode parallel

  # Test ensemble predictions
  python main_pipeline.py --mode predict

  # View training summary
  python main_pipeline.py --mode summary
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'summary', 'parallel'],
                       help='Operation mode')
    parser.add_argument('--trait', type=str, default=None,
                       choices=['EI', 'NS', 'TF', 'JP'],
                       help='Specific trait to train (only with --mode train)')

    args = parser.parse_args()

    if args.mode == 'train':
        if args.trait:
            train_single_trait(args.trait)
        else:
            train_all_traits_sequential()

    elif args.mode == 'parallel':
        print_parallel_instructions()

    elif args.mode == 'predict':
        test_ensemble_prediction()

    elif args.mode == 'summary':
        print_training_summary()


if __name__ == "__main__":
    main()
