#!/bin/bash
# Launch script for 7x H800 GPU training
# 7个H800 GPU分布式训练启动脚本

# ========== Configuration ==========
NUM_GPUS=7
MASTER_PORT=29500

# ========== Environment Setup ==========
# 设置NCCL环境变量以优化通信性能
export NCCL_DEBUG=INFO  # 调试时使用，生产环境可以设置为WARN
export NCCL_IB_DISABLE=0  # 启用InfiniBand（如果有）
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口修改
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5  # 根据实际硬件修改

# 优化NCCL性能
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_NCHANNELS=4

# CUDA优化
export CUDA_LAUNCH_BLOCKING=0  # 异步执行，提升性能
export TORCH_DISTRIBUTED_DEBUG=OFF  # 生产环境关闭调试

# OMP线程数（根据CPU核心数调整）
export OMP_NUM_THREADS=8

# ========== Check GPU Status ==========
echo "=========================================="
echo "Checking GPU Status..."
echo "=========================================="
nvidia-smi

echo ""
echo "=========================================="
echo "Starting Distributed Training on $NUM_GPUS GPUs"
echo "=========================================="
echo "Master Port: $MASTER_PORT"
echo "Training Script: train_h800_optimized.py"
echo "Config: config_h800.py"
echo ""

# ========== Launch Training ==========
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    --use_env \
    train_h800_optimized.py

# Alternative: Use torchrun (PyTorch >= 1.10)
# torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=$MASTER_PORT \
#     train_h800_optimized.py

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
