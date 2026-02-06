"""
Simple test to verify FGM can find and modify embeddings
"""
import torch
from model import NestedPrivacyNER
from config import Config
from adversarial import FGM

config = Config()
model = NestedPrivacyNER(config).to(config.device)

print("Step 1: Check embedding parameters")
print("-" * 50)
emb_params = []
for name, param in model.named_parameters():
    if 'word_embeddings' in name:
        print(f"Found: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Requires grad: {param.requires_grad}")
        emb_params.append((name, param))

if not emb_params:
    print("ERROR: No word_embeddings parameters found!")
    print("\nAll embedding-related parameters:")
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            print(f"  {name}")
else:
    print(f"\nTotal embedding parameters found: {len(emb_params)}")

print("\nStep 2: Test FGM attack")
print("-" * 50)

# Create dummy input
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(config.device)
attention_mask = torch.ones(batch_size, seq_len).to(config.device)
span_positions = torch.randint(0, seq_len, (batch_size, 5, 2)).to(config.device)
span_labels = torch.randint(0, len(config.entity_types), (batch_size, 5)).to(config.device)

# Forward pass
model.train()
logits = model(input_ids, attention_mask, span_positions)
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)),
    span_labels.view(-1),
    ignore_index=-1
)

print(f"Loss: {loss.item():.4f}")

# Backward to create gradients
loss.backward()

print("\nStep 3: Check gradients exist")
print("-" * 50)
for name, param in emb_params:
    if param.grad is not None:
        print(f"{name}: grad exists, norm={torch.norm(param.grad).item():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")

print("\nStep 4: Test FGM attack")
print("-" * 50)
fgm = FGM(model, epsilon=1.0)

# Save original values
original_values = {}
for name, param in emb_params:
    original_values[name] = param.data.clone()

# Attack
fgm.attack(emb_name='word_embeddings')

# Check if parameters changed
changed = False
for name, param in emb_params:
    if not torch.equal(param.data, original_values[name]):
        diff = torch.norm(param.data - original_values[name]).item()
        print(f"{name}: CHANGED (diff norm={diff:.6f})")
        changed = True
    else:
        print(f"{name}: NOT CHANGED")

if not changed:
    print("\nERROR: FGM attack did not modify any parameters!")
    print("This means the emb_name pattern doesn't match any parameters.")
else:
    print("\nSUCCESS: FGM attack modified parameters")

# Restore
fgm.restore(emb_name='word_embeddings')

# Verify restoration
print("\nStep 5: Verify restoration")
print("-" * 50)
for name, param in emb_params:
    if torch.equal(param.data, original_values[name]):
        print(f"{name}: RESTORED correctly")
    else:
        print(f"{name}: RESTORATION FAILED")

print("\nStep 6: Test adversarial forward pass")
print("-" * 50)
# Clear gradients
model.zero_grad()

# Normal forward
logits1 = model(input_ids, attention_mask, span_positions)
loss1 = torch.nn.functional.cross_entropy(
    logits1.view(-1, logits1.size(-1)),
    span_labels.view(-1),
    ignore_index=-1
)
loss1.backward()

# Attack
fgm.attack(emb_name='word_embeddings')

# Adversarial forward
try:
    logits2 = model(input_ids, attention_mask, span_positions)
    loss2 = torch.nn.functional.cross_entropy(
        logits2.view(-1, logits2.size(-1)),
        span_labels.view(-1),
        ignore_index=-1
    )
    print(f"Adversarial loss: {loss2.item():.4f}")
    print(f"Loss has grad_fn: {loss2.grad_fn is not None}")

    # Try backward
    loss2.backward()
    print("SUCCESS: Adversarial backward pass completed!")

except Exception as e:
    print(f"ERROR during adversarial pass: {e}")

fgm.restore(emb_name='word_embeddings')

print("\n" + "=" * 50)
print("Test completed")
print("=" * 50)
