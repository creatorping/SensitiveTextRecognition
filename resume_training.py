"""
Resume training from checkpoint
"""
from config import Config
from train import train

if __name__ == "__main__":
    config = Config()

    # Set checkpoint to resume from
    config.resume_from_checkpoint = 'best_model.pt'

    print("="*80)
    print("RESUME TRAINING MODE")
    print("="*80)
    print(f"Will resume from: {config.resume_from_checkpoint}")
    print(f"Total epochs: {config.num_epochs}")
    print("="*80 + "\n")

    # Start training (will automatically resume from checkpoint)
    train(config)
