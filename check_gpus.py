"""
Simple GPU detection and verification script
"""
import torch

print("="*60)
print("GPU Detection and Verification")
print("="*60)

# Check CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    exit(1)

# Get number of GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")

# List all GPUs
print(f"\nDetected GPUs:")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Get memory info
    props = torch.cuda.get_device_properties(i)
    total_memory = props.total_memory / (1024**3)  # Convert to GB
    print(f"    Memory: {total_memory:.2f} GB")
    print(f"    Compute Capability: {props.major}.{props.minor}")

# Test basic tensor operations on each GPU
print(f"\nTesting tensor operations on each GPU:")
for i in range(num_gpus):
    try:
        device = torch.device(f'cuda:{i}')
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.matmul(x, y)
        print(f"  ✓ GPU {i}: Operations successful")
    except Exception as e:
        print(f"  ✗ GPU {i}: Error - {e}")

# Check NCCL availability (required for DDP)
try:
    nccl_available = torch.cuda.nccl.version() is not None
    nccl_version = torch.cuda.nccl.version()
    print(f"\nNCCL Available: Yes (version {nccl_version})")
except:
    print(f"\nNCCL Available: Unknown")

# Check PyTorch version
print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")

print("\n" + "="*60)
print("✅ GPU verification complete!")
print("="*60)

if num_gpus >= 2:
    print(f"\n✓ Multi-GPU training is supported with {num_gpus} GPUs")
    print(f"\nTo start training:")
    print(f"  1. Quick test: python train_ddp.py")
    print(f"  2. Using launcher: ./launch_multi_gpu.sh")
    print(f"  3. Using torchrun: ./launch_torchrun.sh")
else:
    print(f"\n⚠ Only {num_gpus} GPU detected. Use train.py for single-GPU training.")

print("="*60 + "\n")
