#!/bin/bash

# Multi-GPU Training Launcher Script
# This script launches distributed training across all available GPUs

echo "=========================================="
echo "Multi-GPU Training Launcher"
echo "=========================================="

# Check available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Starting distributed training..."
echo "=========================================="

# Launch training with DDP
python train_ddp.py

echo "=========================================="
echo "Training completed!"
echo "=========================================="
