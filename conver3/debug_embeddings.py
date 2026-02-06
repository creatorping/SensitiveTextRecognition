"""
Debug script to check embedding parameter names
"""
from model import NestedPrivacyNER
from config import Config

config = Config()
model = NestedPrivacyNER(config)

print("All parameters containing 'embedding':")
for name, param in model.named_parameters():
    if 'embedding' in name.lower():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

print("\nSearching for 'embeddings.word_embeddings':")
found = False
for name, param in model.named_parameters():
    if 'embeddings.word_embeddings' in name:
        print(f"  Found: {name}")
        found = True

if not found:
    print("  NOT FOUND!")
    print("\nTrying 'word_embeddings' only:")
    for name, param in model.named_parameters():
        if 'word_embeddings' in name:
            print(f"  Found: {name}")
