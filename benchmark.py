"""
Speed benchmark script for inference performance testing
"""
import torch
import time
import numpy as np
from tqdm import tqdm

from config import Config
from model import NestedPrivacyNER
from data_loader import create_dataloader


def benchmark_inference_speed(model_path: str, config: Config, num_samples=100):
    """Benchmark inference speed"""
    print("="*80)
    print("Inference Speed Benchmark")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = NestedPrivacyNER(config).to(config.device)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    test_loader, _ = create_dataloader(
        config.test_data_path,
        config.test_label_path,
        config,
        is_train=False
    )

    # Warmup
    print("Warming up GPU...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 5:
                break
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            span_positions = batch['span_positions'].to(config.device)
            _ = model(input_ids, attention_mask, span_positions)

    # Benchmark
    print(f"\nBenchmarking on {num_samples} samples...")
    times = []
    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            if sample_count >= num_samples:
                break

            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            span_positions = batch['span_positions'].to(config.device)

            # Measure time
            torch.cuda.synchronize()
            start_time = time.time()

            _ = model(input_ids, attention_mask, span_positions)

            torch.cuda.synchronize()
            end_time = time.time()

            batch_time = end_time - start_time
            times.append(batch_time)
            sample_count += input_ids.size(0)

    # Calculate statistics
    times = np.array(times)
    total_time = np.sum(times)
    avg_time_per_batch = np.mean(times)
    avg_time_per_sample = total_time / sample_count

    print("\n" + "="*80)
    print("Benchmark Results")
    print("="*80)
    print(f"Total samples: {sample_count}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per batch: {avg_time_per_batch*1000:.2f} ms")
    print(f"Average time per sample: {avg_time_per_sample*1000:.2f} ms")
    print(f"Throughput: {sample_count/total_time:.2f} samples/second")
    print(f"Latency (p50): {np.percentile(times, 50)*1000:.2f} ms")
    print(f"Latency (p95): {np.percentile(times, 95)*1000:.2f} ms")
    print(f"Latency (p99): {np.percentile(times, 99)*1000:.2f} ms")
    print("="*80)

    # GPU memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


def profile_model_components(model_path: str, config: Config):
    """Profile different components of the model"""
    print("\n" + "="*80)
    print("Model Component Profiling")
    print("="*80)

    model = NestedPrivacyNER(config).to(config.device)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input
    batch_size = config.batch_size
    seq_len = 128
    num_spans = 100

    input_ids = torch.randint(0, 21128, (batch_size, seq_len)).to(config.device)
    attention_mask = torch.ones(batch_size, seq_len).to(config.device)
    span_positions = torch.randint(0, seq_len, (batch_size, num_spans, 2)).to(config.device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids, attention_mask, span_positions)

    # Profile BERT encoding
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
    torch.cuda.synchronize()
    bert_time = (time.time() - start) / 100

    # Profile CNN layers
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            cnn_input = sequence_output.transpose(1, 2)
            cnn_outputs = [torch.relu(conv(cnn_input)) for conv in model.cnn_layers]
            cnn_output = torch.cat(cnn_outputs, dim=1)
    torch.cuda.synchronize()
    cnn_time = (time.time() - start) / 100

    # Profile span representation
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model.span_repr(sequence_output, span_positions)
    torch.cuda.synchronize()
    span_time = (time.time() - start) / 100

    print(f"\nComponent timing (average over 100 runs):")
    print(f"  BERT encoding: {bert_time*1000:.2f} ms")
    print(f"  CNN layers: {cnn_time*1000:.2f} ms")
    print(f"  Span representation: {span_time*1000:.2f} ms")
    print(f"  Total: {(bert_time + cnn_time + span_time)*1000:.2f} ms")
    print("="*80)


if __name__ == "__main__":
    config = Config()
    model_path = 'best_model.pt'

    if torch.cuda.is_available():
        benchmark_inference_speed(model_path, config, num_samples=200)
        profile_model_components(model_path, config)
    else:
        print("CUDA not available. Benchmark requires GPU.")
