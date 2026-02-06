#!/bin/bash
# 离线训练环境设置脚本
# Offline training environment setup script

echo "=========================================="
echo "离线训练环境设置"
echo "Offline Training Environment Setup"
echo "=========================================="

# 步骤 1: 检查模型文件
echo ""
echo "步骤 1: 检查模型文件"
echo "----------------------------------------"
if [ -d "./models/chinese-roberta-wwm-ext" ]; then
    echo "✓ 找到本地模型: ./models/chinese-roberta-wwm-ext"

    # 检查必需文件
    required_files=("config.json" "pytorch_model.bin" "tokenizer_config.json" "vocab.txt")
    all_present=true

    for file in "${required_files[@]}"; do
        if [ -f "./models/chinese-roberta-wwm-ext/$file" ]; then
            echo "  ✓ $file"
        else
            echo "  ✗ $file (缺失)"
            all_present=false
        fi
    done

    if [ "$all_present" = true ]; then
        echo "✓ 所有必需文件都存在"
    else
        echo "✗ 缺少必需文件，请重新下载模型"
        echo ""
        echo "下载方法:"
        echo "1. 在有网络的机器上运行: python3 download_model.py"
        echo "2. 将 models 文件夹复制到此目录"
        exit 1
    fi
else
    echo "✗ 未找到本地模型"
    echo ""
    echo "请按以下步骤操作:"
    echo "1. 在有网络的机器上运行: python3 download_model.py"
    echo "2. 将生成的 models 文件夹复制到此目录"
    echo "3. 重新运行此脚本"
    exit 1
fi

# 步骤 2: 检查数据文件
echo ""
echo "步骤 2: 检查数据文件"
echo "----------------------------------------"
data_files=("data/409_data_train.txt" "data/409_train_lable.txt" "data/409_data_test.txt" "data/409_test_lable.txt")
data_ok=true

for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (缺失)"
        data_ok=false
    fi
done

if [ "$data_ok" = false ]; then
    echo "✗ 缺少数据文件"
    exit 1
fi

# 步骤 3: 设置环境变量
echo ""
echo "步骤 3: 设置环境变量"
echo "----------------------------------------"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "✓ TRANSFORMERS_OFFLINE=1"
echo "✓ HF_DATASETS_OFFLINE=1"
echo "✓ TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1"
echo "✓ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# 步骤 4: 检查 GPU
echo ""
echo "步骤 4: 检查 GPU"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "检测到 $gpu_count 个 GPU:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r idx name memory; do
        echo "  GPU $idx: $name ($memory)"
    done
else
    echo "✗ 未检测到 nvidia-smi"
    exit 1
fi

# 步骤 5: 生成训练脚本
echo ""
echo "步骤 5: 生成训练脚本"
echo "----------------------------------------"

cat > start_training.sh << 'EOF'
#!/bin/bash

# 设置离线模式
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "启动双RTX 5090D分布式训练"
echo "=========================================="

# 检测GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -eq 2 ]; then
    echo "使用 2 个GPU进行分布式训练"
    torchrun --nproc_per_node=2 \
        --master_port=29500 \
        train_5090d.py
elif [ $GPU_COUNT -eq 1 ]; then
    echo "使用 1 个GPU进行训练"
    torchrun --nproc_per_node=1 \
        --master_port=29500 \
        train_5090d.py
else
    echo "错误: 未检测到GPU"
    exit 1
fi
EOF

chmod +x start_training.sh
echo "✓ 已生成 start_training.sh"

echo ""
echo "=========================================="
echo "✓ 环境设置完成！"
echo "=========================================="
echo ""
echo "开始训练:"
echo "  ./start_training.sh"
echo ""
echo "或使用 Docker:"
echo "  docker run --rm --gpus all \\"
echo "    --ipc=host --shm-size=32g \\"
echo "    -v \$(pwd):/workspace \\"
echo "    -v \$(pwd)/data:/workspace/data \\"
echo "    -v \$(pwd)/models:/workspace/models \\"
echo "    -e NVIDIA_VISIBLE_DEVICES=0,1 \\"
echo "    -e CUDA_VISIBLE_DEVICES=0,1 \\"
echo "    -e TRANSFORMERS_OFFLINE=1 \\"
echo "    -e HF_DATASETS_OFFLINE=1 \\"
echo "    -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \\"
echo "    --name ner-training \\"
echo "    ner-5090d:latest \\"
echo "    bash start_training.sh"
echo ""
