#!/bin/bash

# Alternative launcher using torchrun (recommended for PyTorch 1.10+)
# torchrun provides better error handling and process management

echo "=========================================="
echo "Multi-GPU Training with torchrun"
echo "=========================================="

# Check available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Set environment variables
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Starting distributed training with torchrun..."
echo "=========================================="

# Launch with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_ddp.py

echo "=========================================="
echo "Training completed!"
echo "=========================================="
