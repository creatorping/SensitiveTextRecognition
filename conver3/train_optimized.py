"""
Quick Start Script - 使用所有优化功能
"""
from config import Config
from train import train

if __name__ == "__main__":
    config = Config()

    print("="*80)
    print("🚀 高级优化训练模式")
    print("="*80)

    # 启用所有优化功能
    config.use_focal_loss = True
    config.use_data_augmentation = False  # 第一阶段先不用，稳定后再开启
    config.use_crf_model = False  # 第一阶段先不用，基础模型训练好后再开启

    # 优化超参数
    config.learning_rate = 3e-5
    config.batch_size = 16
    config.num_epochs = 100

    print(f"✓ Focal Loss: {config.use_focal_loss}")
    print(f"✓ Data Augmentation: {config.use_data_augmentation}")
    print(f"✓ CRF Model: {config.use_crf_model}")
    print(f"✓ Learning Rate: {config.learning_rate}")
    print(f"✓ Batch Size: {config.batch_size}")
    print(f"✓ Epochs: {config.num_epochs}")
    print("="*80 + "\n")

    # 开始训练
    train(config)
