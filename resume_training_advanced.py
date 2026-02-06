"""
Resume training from any checkpoint with flexible options
"""
import argparse
import os
from config import Config
from train import train

def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                        help='Path to checkpoint file (default: best_model.pt)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total epochs to train (default: use config value)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: use config value)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: use config value)')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file '{args.checkpoint}' not found!")
        print("\nAvailable checkpoints:")
        for f in os.listdir('.'):
            if f.endswith('.pt'):
                print(f"  - {f}")
        return

    # Load config
    config = Config()

    # Set resume checkpoint
    config.resume_from_checkpoint = args.checkpoint

    # Override config if specified
    if args.epochs is not None:
        config.num_epochs = args.epochs
        print(f"Overriding num_epochs to {args.epochs}")

    if args.lr is not None:
        config.learning_rate = args.lr
        print(f"Overriding learning_rate to {args.lr}")

    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"Overriding batch_size to {args.batch_size}")

    print("\n" + "="*80)
    print("RESUME TRAINING MODE")
    print("="*80)
    print(f"Checkpoint: {config.resume_from_checkpoint}")
    print(f"Total epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print("="*80 + "\n")

    # Start training
    train(config)

if __name__ == "__main__":
    main()
