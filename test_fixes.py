"""
Test script to verify all fixes
"""
import torch
from config import Config
from model import NestedPrivacyNER
from data_loader import create_dataloader
from adversarial import FGM, PGD

def test_config():
    """Test config has all entity types"""
    config = Config()
    print("Entity types:", config.entity_types)
    assert len(config.entity_types) == 9, f"Expected 9 entity types, got {len(config.entity_types)}"
    assert 'PI' in config.entity_types, "PI should be in entity_types"
    assert 'CNU' in config.entity_types, "CNU should be in entity_types"
    assert 'MS' in config.entity_types, "MS should be in entity_types"
    assert 'PC' in config.entity_types, "PC should be in entity_types"
    print("✓ Config test passed")

def test_dataloader():
    """Test dataloader with custom collate function"""
    config = Config()
    try:
        train_loader, tokenizer = create_dataloader(
            config.train_data_path,
            config.train_label_path,
            config,
            is_train=True
        )
        print(f"✓ DataLoader created successfully with {len(train_loader)} batches")

        # Test one batch
        batch = next(iter(train_loader))
        print(f"  Batch keys: {batch.keys()}")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  span_positions shape: {batch['span_positions'].shape}")
        print(f"  span_labels shape: {batch['span_labels'].shape}")
        print("✓ DataLoader test passed")
        return train_loader, tokenizer
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        raise

def test_model():
    """Test model initialization"""
    config = Config()
    try:
        model = NestedPrivacyNER(config).to(config.device)
        print(f"✓ Model created successfully")

        # Check embedding parameters
        emb_params = [name for name, _ in model.named_parameters() if 'embeddings.word_embeddings' in name]
        print(f"  Embedding parameters found: {emb_params}")
        assert len(emb_params) > 0, "No embedding parameters found"
        print("✓ Model test passed")
        return model
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        raise

def test_adversarial():
    """Test adversarial training setup"""
    config = Config()
    model = NestedPrivacyNER(config).to(config.device)

    try:
        fgm = FGM(model, epsilon=config.adv_epsilon)
        pgd = PGD(model, epsilon=config.adv_epsilon, alpha=config.adv_alpha)
        print("✓ Adversarial training objects created")

        # Test FGM attack/restore
        original_params = {}
        for name, param in model.named_parameters():
            if 'embeddings.word_embeddings' in name:
                original_params[name] = param.data.clone()

        # Simulate gradient
        for name, param in model.named_parameters():
            if 'embeddings.word_embeddings' in name and param.requires_grad:
                param.grad = torch.randn_like(param.data) * 0.01

        fgm.attack(emb_name='embeddings.word_embeddings')
        print("  FGM attack applied")

        fgm.restore(emb_name='embeddings.word_embeddings')
        print("  FGM restore applied")

        # Verify restoration
        for name, param in model.named_parameters():
            if name in original_params:
                assert torch.allclose(param.data, original_params[name]), f"Parameter {name} not restored correctly"

        print("✓ Adversarial training test passed")
    except Exception as e:
        print(f"✗ Adversarial training test failed: {e}")
        raise

def test_forward_pass():
    """Test forward pass with real data"""
    config = Config()
    model = NestedPrivacyNER(config).to(config.device)
    train_loader, _ = create_dataloader(
        config.train_data_path,
        config.train_label_path,
        config,
        is_train=True
    )

    try:
        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        span_positions = batch['span_positions'].to(config.device)

        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask, span_positions)

        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Expected: [batch_size={config.batch_size}, num_spans=200, num_classes={len(config.entity_types)+1}]")
        print("✓ Forward pass test passed")
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        raise

if __name__ == "__main__":
    print("Running tests...\n")

    print("1. Testing Config...")
    test_config()
    print()

    print("2. Testing DataLoader...")
    test_dataloader()
    print()

    print("3. Testing Model...")
    test_model()
    print()

    print("4. Testing Adversarial Training...")
    test_adversarial()
    print()

    print("5. Testing Forward Pass...")
    test_forward_pass()
    print()

    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
