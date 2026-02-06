"""
Quick test script to verify multi-GPU setup
Tests DDP initialization and basic forward pass
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def test_worker(rank, world_size):
    """Test function for each GPU"""
    setup(rank, world_size)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Multi-GPU Test")
        print(f"{'='*60}")
        print(f"World Size: {world_size}")
        print(f"Testing on {world_size} GPUs...")
        print(f"{'='*60}\n")

    # Create a simple model
    model = torch.nn.Linear(10, 5).to(rank)
    model = DDP(model, device_ids=[rank])

    # Create dummy data
    x = torch.randn(4, 10).to(rank)

    # Forward pass
    output = model(x)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Synchronize
    dist.barrier()

    if rank == 0:
        print(f"✓ GPU {rank}: Forward/Backward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Loss: {loss.item():.4f}")

    # Test all_reduce
    tensor = torch.tensor([rank], dtype=torch.float32).to(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        expected_sum = sum(range(world_size))
        print(f"\n✓ All-reduce test successful")
        print(f"  Sum of ranks: {tensor.item():.0f} (expected: {expected_sum})")

    dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"✅ All tests passed! Multi-GPU setup is working correctly.")
        print(f"{'='*60}\n")
        print(f"You can now run full training with:")
        print(f"  python train_ddp.py")
        print(f"  or")
        print(f"  ./launch_multi_gpu.sh")
        print(f"{'='*60}\n")

    cleanup()


def main():
    """Main test function"""
    world_size = torch.cuda.device_count()

    if world_size < 2:
        print(f"Warning: Only {world_size} GPU(s) detected.")
        print(f"Multi-GPU training requires at least 2 GPUs.")
        return

    print(f"Detected {world_size} GPUs:")
    for i in range(world_size):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Spawn processes
    mp.spawn(
        test_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
