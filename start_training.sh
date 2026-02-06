#!/bin/bash
# 双RTX 5090D训练启动脚本
# 使用DistributedDataParallel进行多GPU训练

set -e

echo "=========================================="
echo "启动双RTX 5090D分布式训练"
echo "=========================================="

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -lt 2 ]; then
    echo "警告: 检测到少于2个GPU，将使用单GPU训练"
    python train_5090d.py
else
    echo "使用 $GPU_COUNT 个GPU进行分布式训练"

    # 使用torchrun启动分布式训练
    # --nproc_per_node: 每个节点的进程数（GPU数量）
    # --master_port: 主节点端口
    torchrun \
        --nproc_per_node=$GPU_COUNT \
        --master_port=29500 \
        train_5090d.py
fi

echo "=========================================="
echo "训练完成！"
echo "=========================================="
