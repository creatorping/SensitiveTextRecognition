# Multi-GPU Training Guide

## Overview
This project now supports efficient multi-GPU training using PyTorch's DistributedDataParallel (DDP). DDP is significantly more efficient than DataParallel and scales better across multiple GPUs.

## Available Training Scripts

### 1. **train_ddp.py** (Recommended for Multi-GPU)
- Uses DistributedDataParallel (DDP)
- Optimized for 8x H800 GPUs
- Better performance and scaling than DataParallel
- Supports gradient accumulation, EMA, and R-Drop

### 2. **train_multi_gpu.py** (Legacy)
- Uses DataParallel
- Simpler but less efficient
- Good for 2-4 GPUs

### 3. **train.py** (Single GPU)
- Original single-GPU training script
- Use when only 1 GPU is available

## Quick Start

### Method 1: Using the launcher script (Easiest)
```bash
# Make script executable (first time only)
chmod +x launch_multi_gpu.sh

# Launch training
./launch_multi_gpu.sh
```

### Method 2: Using torchrun (Recommended)
```bash
# Make script executable (first time only)
chmod +x launch_torchrun.sh

# Launch training
./launch_torchrun.sh
```

### Method 3: Direct Python execution
```bash
# DDP will automatically detect and use all available GPUs
python train_ddp.py
```

### Method 4: Specify number of GPUs
```bash
# Use only 4 GPUs (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ddp.py

# Use only 2 GPUs (0,1)
CUDA_VISIBLE_DEVICES=0,1 python train_ddp.py
```

## Performance Comparison

| Method | GPUs | Effective Batch Size | Speed | Memory Efficiency |
|--------|------|---------------------|-------|-------------------|
| Single GPU | 1 | 32 | 1x | Baseline |
| DataParallel | 8 | 256 | ~5x | Poor (duplicates model) |
| **DDP** | 8 | 256 | **~7.5x** | **Excellent** |

## Key Features of DDP Training

### 1. **Automatic GPU Detection**
- Automatically detects all available GPUs
- No manual configuration needed

### 2. **Efficient Memory Usage**
- Each GPU holds only one model replica
- No redundant model copies like DataParallel

### 3. **Better Gradient Synchronization**
- Gradients are synchronized efficiently using NCCL
- Overlaps computation with communication

### 4. **Distributed Sampling**
- Each GPU processes different data batches
- No data duplication across GPUs

### 5. **Checkpoint Compatibility**
- Checkpoints saved from DDP can be loaded in single-GPU mode
- Easy to switch between training modes

## Configuration Tips

### Batch Size Tuning
With 8 GPUs, your effective batch size is:
```
Effective Batch Size = batch_size × num_gpus × gradient_accumulation_steps
```

Current config:
- `batch_size = 32` (per GPU)
- `num_gpus = 8`
- `gradient_accumulation_steps = 1`
- **Effective batch size = 256**

If you encounter OOM (Out of Memory) errors:
1. Reduce `batch_size` in `config.py`
2. Increase `gradient_accumulation_steps` to maintain effective batch size

### Learning Rate Scaling
When using larger batch sizes, consider scaling the learning rate:
```python
# Linear scaling rule
learning_rate = base_lr × (effective_batch_size / base_batch_size)

# Example: if base_lr=3e-5 for batch_size=32
# For effective_batch_size=256: lr = 3e-5 × (256/32) = 2.4e-4
```

However, the current config already uses a good learning rate for large batches.

## Monitoring Training

### Check GPU Utilization
```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi
```

You should see:
- All 8 GPUs at high utilization (>90%)
- Similar memory usage across all GPUs
- Similar GPU temperature

### Training Logs
- Only GPU 0 (rank 0) prints training progress
- All GPUs participate in training
- Metrics are aggregated across all GPUs

## Troubleshooting

### Issue: "Address already in use"
```bash
# Kill existing processes
pkill -9 python

# Or change the port in train_ddp.py
os.environ['MASTER_PORT'] = '12356'  # Change from 12355
```

### Issue: OOM (Out of Memory)
```python
# In config.py, reduce batch size
batch_size = 16  # Reduce from 32

# Or enable gradient checkpointing
use_gradient_checkpointing = True
```

### Issue: Slow training
```bash
# Check if all GPUs are being used
nvidia-smi

# Ensure NCCL is using the right network interface
export NCCL_SOCKET_IFNAME=eth0  # or your network interface
```

### Issue: NaN loss
- Already handled in the code with gradient clipping
- Check if learning rate is too high
- Ensure data is properly normalized

## Advanced Usage

### Resume Training
```python
# In config.py
resume_from_checkpoint = "best_model.pt"
```

### Mixed Precision Training (AMP)
Already enabled in config:
```python
use_amp = True  # 2-3x speedup with minimal accuracy loss
```

### Custom GPU Selection
```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,2,4,6 python train_ddp.py  # Use 4 GPUs

# Use all except GPU 0
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_ddp.py
```

## Expected Training Time

With 8x H800 GPUs:
- **~7.5x faster** than single GPU
- Epoch time depends on dataset size
- Monitor first epoch to estimate total time

## Best Practices

1. **Start with fewer epochs** to verify setup works
2. **Monitor GPU utilization** to ensure all GPUs are active
3. **Save checkpoints regularly** (already implemented)
4. **Use tensorboard** for detailed metrics (can be added)
5. **Test on small dataset first** before full training

## Comparison with Original Scripts

| Feature | train.py | train_multi_gpu.py | train_ddp.py |
|---------|----------|-------------------|--------------|
| Multi-GPU | ❌ | ✅ DataParallel | ✅ DDP |
| Efficiency | N/A | Medium | High |
| Scalability | N/A | Poor (>4 GPUs) | Excellent |
| Memory | Baseline | High overhead | Optimal |
| Speed (8 GPUs) | 1x | ~5x | ~7.5x |

## Next Steps

1. Run a test with 1-2 epochs to verify everything works
2. Monitor GPU utilization and adjust batch size if needed
3. Start full training with all epochs
4. Compare results with single-GPU training

## Support

If you encounter issues:
1. Check GPU availability: `nvidia-smi`
2. Verify PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Check NCCL: `python -c "import torch; print(torch.cuda.nccl.version())"`
4. Review error logs carefully

Happy training! 🚀
