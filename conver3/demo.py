"""
Demo script for privacy entity extraction
"""
import torch
from transformers import AutoTokenizer
from model import NestedPrivacyNER
from config import Config


def demo():
    """Demo the privacy entity extraction system"""
    config = Config()

    # Load model
    print("Loading model...")
    model = NestedPrivacyNER(config).to(config.device)

    try:
        checkpoint = torch.load('best_model.pt', map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with F1: {checkpoint['f1']:.4f}")
    except FileNotFoundError:
        print("No trained model found. Please run train.py first.")
        return

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Test examples
    test_texts = [
        "方卫英女士，中国国籍，无境外居留权，出生于1964年8月，党校研究生学历，高级编辑职称。",
        "张杰点一首《逆战》送给徐梦桃这首歌的适配度太高了",
        "现任西藏药业董事会秘书。",
        "1962年出生，毕业于江苏大学，本科学历，工程师、高级经济师。",
        "商丘市梁园区前进小学教师王茜用沙画庆祝苏翊鸣夺冠。"
    ]

    print("\n" + "="*80)
    print("Privacy Entity Extraction Demo")
    print("="*80)

    for text in test_texts:
        print(f"\nText: {text}")

        # Tokenize
        encoding = tokenizer(
            text,
            max_length=config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(config.device)
        attention_mask = encoding['attention_mask'].to(config.device)

        # Generate spans
        span_positions = []
        char_to_token = []

        for i in range(len(text)):
            token_idx = encoding.char_to_token(0, i)
            char_to_token.append(token_idx if token_idx is not None else -1)

        max_span_len = min(config.max_span_length, len(text))
        for start in range(len(text)):
            for end in range(start, min(start + max_span_len, len(text))):
                start_token = char_to_token[start] if start < len(char_to_token) else -1
                end_token = char_to_token[end] if end < len(char_to_token) else -1

                if start_token != -1 and end_token != -1:
                    span_positions.append((start_token, end_token, start, end))

        if len(span_positions) == 0:
            print("  No entities found.")
            continue

        # Inference
        span_pos_tensor = torch.tensor(
            [(s[0], s[1]) for s in span_positions],
            dtype=torch.long
        ).unsqueeze(0).to(config.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, span_pos_tensor)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1)[0]

        # Extract entities
        entities = []
        for idx, (_, _, char_start, char_end) in enumerate(span_positions):
            pred = preds[0, idx].item()
            conf = confidences[0, idx].item()

            if pred > 0 and conf >= 0.5:
                entity_type = config.entity_types[pred - 1]
                entity_text = text[char_start:char_end+1]
                entities.append((char_start, char_end, entity_type, entity_text, conf))

        # Sort by position
        entities = sorted(entities, key=lambda x: x[0])

        if entities:
            print("  Entities:")
            for start, end, entity_type, entity_text, conf in entities:
                print(f"    [{entity_type}] {entity_text} (confidence: {conf:.3f})")
        else:
            print("  No entities found.")

    print("\n" + "="*80)


if __name__ == "__main__":
    demo()
