"""
Quick start script for training and evaluation
"""
import os
import sys
import argparse
import torch

from config import Config
from train import train, set_seed
from inference import detailed_evaluation


def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment...")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available. Training will be slow on CPU.")

    # Check data files
    config = Config()
    data_files = [
        config.train_data_path,
        config.train_label_path,
        config.test_data_path,
        config.test_label_path
    ]

    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Nested Privacy Entity Recognition")
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'both'],
                       help='Mode: train, eval, or both')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--use_fgm', action='store_true', default=True,
                       help='Use FGM adversarial training')
    parser.add_argument('--use_pgd', action='store_true', default=False,
                       help='Use PGD adversarial training')
    parser.add_argument('--use_rdrop', action='store_true', default=True,
                       help='Use R-Drop for model smoothing')

    args = parser.parse_args()

    # Check environment
    if not check_environment():
        print("\nEnvironment check failed. Please ensure all data files are present.")
        return

    print("\n" + "="*80)
    print("Nested Privacy Entity Recognition System")
    print("="*80)

    # Load config
    config = Config()

    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr

    config.use_fgm = args.use_fgm
    config.use_pgd = args.use_pgd
    config.use_rdrop = args.use_rdrop

    # Print configuration
    print("\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max span length: {config.max_span_length}")
    print(f"  Adversarial training: FGM={config.use_fgm}, PGD={config.use_pgd}")
    print(f"  Model smoothing: R-Drop={config.use_rdrop}")
    print()

    # Train
    if args.mode in ['train', 'both']:
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        train(config)

    # Evaluate
    if args.mode in ['eval', 'both']:
        if not os.path.exists(args.model_path):
            print(f"\nModel not found at {args.model_path}")
            if args.mode == 'eval':
                print("Please train the model first or specify a valid model path.")
                return
        else:
            print("\n" + "="*80)
            print("Starting Evaluation")
            print("="*80)
            f1_score = detailed_evaluation(args.model_path, config)

            print("\n" + "="*80)
            print(f"Final F1 Score: {f1_score:.4f}")
            if f1_score >= 0.95:
                print("✓ Target F1 score (≥0.95) achieved!")
            else:
                print(f"✗ Target F1 score not reached. Current: {f1_score:.4f}, Target: 0.95")
            print("="*80)


if __name__ == "__main__":
    main()
