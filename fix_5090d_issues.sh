#!/bin/bash
# RTX 5090D 问题修复脚本
# 解决CUDA兼容性和网络问题

set -e

echo "=========================================="
echo "RTX 5090D 环境修复脚本"
echo "=========================================="

# 1. 下载预编译模型（避免网络问题）
echo "步骤 1/3: 下载预训练模型"
echo "请从以下地址手动下载模型："
echo ""
echo "方案A - 使用modelscope（国内镜像）："
echo "  pip install modelscope"
echo "  modelscope download --model 'hfl/chinese-roberta-wwm-ext' --local_dir ./models/chinese-roberta-wwm-ext"
echo ""
echo "方案B - 使用镜像站："
echo "  wget -O chinese-roberta-wwm-ext.tar.gz https://hf-mirror.com/hfl/chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin"
echo ""
echo "方案C - 手动下载并放置："
echo "  1. 访问: https://hf-mirror.com/hfl/chinese-roberta-wwm-ext"
echo "  2. 下载所有文件到: ./models/chinese-roberta-wwm-ext/"
echo ""

read -p "是否已下载模型到 ./models/chinese-roberta-wwm-ext/? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "请先下载模型后再运行此脚本"
    exit 1
fi

# 2. 检查PyTorch版本和CUDA兼容性
echo ""
echo "步骤 2/3: 检查CUDA兼容性"
python3 << 'EOF'
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  计算能力: sm_{props.major}{props.minor}")
        print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
EOF

# 3. 更新config.py使用本地模型
echo ""
echo "步骤 3/3: 更新配置使用本地模型"
if [ -f "config.py" ]; then
    # 备份原文件
    cp config.py config.py.backup

    # 更新model_name路径
    sed -i 's|model_name = "hfl/chinese-roberta-wwm-ext"|model_name = "./models/chinese-roberta-wwm-ext"|g' config.py

    echo "✓ 配置已更新为使用本地模型"
    echo "  原配置已备份至: config.py.backup"
fi

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "后续步骤："
echo "1. 如果看到CUDA兼容性警告，需要使用PyTorch nightly版本："
echo "   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124"
echo ""
echo "2. 或者等待PyTorch官方发布支持sm_120的稳定版"
echo ""
echo "3. 当前可以继续训练，但可能无法充分利用5090D的新特性"
echo ""
