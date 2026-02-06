"""
诊断脚本 - 检查Docker和GPU环境配置
在容器内运行此脚本以验证环境是否正确配置
"""
import torch
import sys
import os

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_pytorch():
    print_section("PyTorch 配置检查")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"检测到的GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
            print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

            # 测试显存分配
            try:
                torch.cuda.set_device(i)
                x = torch.randn(1000, 1000).cuda()
                print(f"  显存分配测试: ✓ 成功")
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  显存分配测试: ✗ 失败 - {e}")
    else:
        print("⚠️  警告: CUDA不可用！")
        return False

    return True

def check_amp():
    print_section("混合精度训练 (AMP) 检查")

    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，无法测试AMP")
        return False

    try:
        from torch.cuda.amp import autocast, GradScaler

        # 测试AMP
        model = torch.nn.Linear(100, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()

        x = torch.randn(32, 100).cuda()
        y = torch.randn(32, 10).cuda()

        with autocast():
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print("✓ AMP功能正常")
        return True
    except Exception as e:
        print(f"✗ AMP测试失败: {e}")
        return False

def check_distributed():
    print_section("分布式训练环境检查")

    env_vars = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']

    print("环境变量:")
    for var in env_vars:
        value = os.environ.get(var, "未设置")
        print(f"  {var}: {value}")

    try:
        import torch.distributed as dist
        print("\n✓ torch.distributed 模块可用")

        # 检查NCCL后端
        if torch.cuda.is_available():
            print(f"✓ NCCL 后端可用")

        return True
    except Exception as e:
        print(f"✗ 分布式训练检查失败: {e}")
        return False

def check_model():
    print_section("模型加载检查")

    try:
        from config import Config
        from model import NestedPrivacyNER

        config = Config()
        print(f"配置加载: ✓")
        print(f"  模型名称: {config.model_name}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  使用AMP: {config.use_amp}")
        print(f"  最大长度: {config.max_length}")

        # 尝试创建模型
        if torch.cuda.is_available():
            model = NestedPrivacyNER(config).cuda()
            print(f"模型创建: ✓")

            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  总参数量: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")

            # 测试前向传播
            batch_size = 2
            seq_len = 128
            num_spans = 10

            input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
            attention_mask = torch.ones(batch_size, seq_len).cuda()
            span_positions = torch.randint(0, seq_len, (batch_size, num_spans, 2)).cuda()

            with torch.no_grad():
                output = model(input_ids, attention_mask, span_positions)

            print(f"前向传播测试: ✓")
            print(f"  输出形状: {output.shape}")

            del model
            torch.cuda.empty_cache()
        else:
            print("⚠️  CUDA不可用，跳过GPU模型测试")

        return True
    except Exception as e:
        print(f"✗ 模型检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data():
    print_section("数据文件检查")

    data_files = [
        'data/409_data_train.txt',
        'data/409_train_lable.txt',
        'data/409_data_test.txt',
        'data/409_test_lable.txt'
    ]

    all_exist = True
    for file_path in data_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")

        if exists:
            size = os.path.getsize(file_path)
            print(f"    大小: {size:,} bytes")
        else:
            all_exist = False

    return all_exist

def check_dependencies():
    print_section("依赖包检查")

    packages = [
        'torch',
        'transformers',
        'tqdm',
        'sklearn',
        'numpy',
        'torchcrf'
    ]

    all_installed = True
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', '未知')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: 未安装")
            all_installed = False

    return all_installed

def main():
    print("\n" + "="*80)
    print("  双RTX 5090D环境诊断脚本")
    print("="*80)

    results = {
        'PyTorch': check_pytorch(),
        'AMP': check_amp(),
        'Distributed': check_distributed(),
        'Dependencies': check_dependencies(),
        'Data': check_data(),
        'Model': check_model()
    }

    print_section("诊断总结")

    for name, status in results.items():
        status_str = "✓ 通过" if status else "✗ 失败"
        print(f"{name:20s}: {status_str}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✓ 所有检查通过！环境配置正确。")
        print("可以开始训练: bash start_training.sh")
    else:
        print("✗ 部分检查失败，请根据上述信息排查问题。")
        sys.exit(1)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
