#!/usr/bin/env python3
"""
Tinker OMR Training Script
Train a Vision-Language Model for Optical Music Recognition (score image -> Paraff)
"""

import os
import sys
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime

# Use HF mirror for China network (avoid proxy issues with Tinker API)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import tinker
from tinker import types


# === Prompt Templates ===
SYSTEM_PROMPT = """You are a music score recognition system. You convert sheet music images into Paraff notation, a domain-specific language for representing musical scores."""

QUESTIONS = [
    "Recognize this music score as Paraff notation.",
    "Parse the sheet music into Paraff language.",
    "Convert this score image to Paraff format.",
    "Transcribe this music sheet using Paraff.",
    "Read this score and output Paraff representation.",
    "Interpret this sheet music in Paraff format.",
]


def format_prompt(question: str) -> str:
    """Format prompt for Qwen3-VL model."""
    return f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
<|vision_start|><|image_placeholder|><|vision_end|>
{question}
<|im_end|>
<|im_start|>assistant
"""


class OMRDataset:
    """Dataset for OMR training."""

    def __init__(self, csv_path: str, images_dir: str, limit: int = None):
        self.images_dir = Path(images_dir).expanduser()
        df = pd.read_csv(Path(csv_path).expanduser())

        if limit:
            df = df.head(limit)

        self.data = []
        print(f"Loading {len(df)} samples...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = self.images_dir / row['image']
            if img_path.exists():
                self.data.append({
                    'index': row['index'],
                    'image_path': str(img_path),
                    'paraff': row['sentence'],
                })
            else:
                print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_image_bytes(self, idx):
        """Load image bytes lazily."""
        with open(self.data[idx]['image_path'], 'rb') as f:
            return f.read()


def prepare_datum(sample: dict, image_bytes: bytes, tokenizer) -> types.Datum:
    """Convert sample to Tinker Datum format for vision-language training."""
    question = random.choice(QUESTIONS)
    response = sample['paraff']

    # Qwen3-VL special tokens for vision
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    # Build prompt parts
    system_part = f"{im_start}system\n{SYSTEM_PROMPT}\n{im_end}\n"
    user_start = f"{im_start}user\n{vision_start}"
    user_end = f"{vision_end}\n{question}\n{im_end}\n{im_start}assistant\n"
    response_part = f"{response}{im_end}\n"

    # Tokenize each part
    system_tokens = tokenizer.encode(system_part, add_special_tokens=False)
    user_start_tokens = tokenizer.encode(user_start, add_special_tokens=False)
    user_end_tokens = tokenizer.encode(user_end, add_special_tokens=False)
    response_tokens = tokenizer.encode(response_part, add_special_tokens=False)

    # Build chunks for model input (including response for training)
    chunks = [
        types.EncodedTextChunk(tokens=system_tokens),
        types.EncodedTextChunk(tokens=user_start_tokens),
        types.ImageChunk(data=image_bytes, format="png"),
        types.EncodedTextChunk(tokens=user_end_tokens),
        types.EncodedTextChunk(tokens=response_tokens),
    ]

    model_input = types.ModelInput(chunks=chunks)

    # Calculate token lengths for weight assignment
    # Note: ImageChunk contributes variable tokens depending on image size
    # For Qwen3-VL, we estimate ~1000 tokens per image as placeholder
    IMAGE_TOKENS_ESTIMATE = 1000

    prompt_length = len(system_tokens) + len(user_start_tokens) + IMAGE_TOKENS_ESTIMATE + len(user_end_tokens)
    total_length = prompt_length + len(response_tokens)

    # Create weights: 0 for prompt (no loss), 1 for response (compute loss)
    weights = [0.0] * prompt_length + [1.0] * len(response_tokens)

    # For target tokens, we need the actual token sequence
    # Since we can't get exact tokens with images, we'll use the text tokens
    # The model will handle the image tokens internally
    all_text_tokens = system_tokens + user_start_tokens + user_end_tokens + response_tokens
    target_tokens = all_text_tokens[1:] + [tokenizer.eos_token_id]  # Shifted

    # Adjust weights to match target length
    weights = weights[:len(target_tokens)]
    if len(weights) < len(target_tokens):
        weights = weights + [1.0] * (len(target_tokens) - len(weights))

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=target_tokens, dtype="int64"),
            "weights": types.TensorData(data=weights, dtype="float32"),
        }
    )


def train(args):
    """Main training function."""
    print("=" * 60)
    print("Tinker OMR Training")
    print("=" * 60)

    # === Initialize Tinker Client ===
    print("\nInitializing Tinker client...")
    service_client = tinker.ServiceClient()

    print(f"Creating LoRA training client for {args.model}...")
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
    )

    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer loaded, vocab size: {len(tokenizer)}")

    # === Load Dataset ===
    print(f"\nLoading dataset from {args.csv}...")
    dataset = OMRDataset(args.csv, args.images_dir, limit=args.limit)

    # Split train/val
    indices = list(range(len(dataset)))
    if args.shuffle:
        random.shuffle(indices)

    val_size = max(1, int(len(indices) * args.val_ratio))
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # === Training Setup ===
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = log_dir / "metrics.jsonl"
    config_file = log_dir / "config.json"

    # Save config
    config = vars(args)
    config['start_time'] = datetime.now().isoformat()
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining config:")
    print(f"  Model: {args.model}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps: {args.steps}")
    print(f"  Log dir: {log_dir}")

    # === Training Loop ===
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    best_val_loss = float('inf')

    for step in range(args.steps):
        # Sample batch
        batch_indices = random.choices(train_indices, k=args.batch_size)
        batch = []
        for idx in batch_indices:
            sample = dataset[idx]
            image_bytes = dataset.get_image_bytes(idx)
            datum = prepare_datum(sample, image_bytes, tokenizer)
            batch.append(datum)

        # Forward-backward
        fwdbwd_result = training_client.forward_backward(
            batch,
            loss_fn="cross_entropy"
        ).result()

        # Optimizer step
        optim_result = training_client.optim_step(
            types.AdamParams(
                learning_rate=args.lr,
                beta1=0.9,
                beta2=0.98,
                eps=1e-9,
                grad_clip_norm=1.0,
            )
        ).result()

        train_loss = fwdbwd_result.loss

        # Logging
        metrics = {
            'step': step,
            'train_loss': train_loss,
            'timestamp': datetime.now().isoformat(),
        }

        if step % args.log_every == 0:
            print(f"Step {step:4d} | train_loss: {train_loss:.4f}")

        # Validation
        if step % args.eval_every == 0 and val_indices:
            val_batch_indices = random.sample(val_indices, min(args.batch_size, len(val_indices)))
            val_batch = []
            for idx in val_batch_indices:
                sample = dataset[idx]
                image_bytes = dataset.get_image_bytes(idx)
                datum = prepare_datum(sample, image_bytes, tokenizer)
                val_batch.append(datum)

            val_result = training_client.forward_backward(
                val_batch,
                loss_fn="cross_entropy"
            ).result()

            val_loss = val_result.loss
            metrics['val_loss'] = val_loss
            print(f"Step {step:4d} | val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_client.save_state(name="best")
                print(f"  -> New best model saved!")

        # Save metrics
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

        # Checkpoint
        if step > 0 and step % args.save_every == 0:
            training_client.save_state(name=f"step_{step}")
            print(f"  -> Checkpoint saved: step_{step}")

    # === Save Final Model ===
    print("\nSaving final model...")
    final_path = training_client.save_state(name="final").result().path
    print(f"Final model saved to: {final_path}")

    # Save for sampling
    print("Creating sampling client...")
    sampler_path = training_client.save_weights_for_sampler(name="final_sampler").result().path
    print(f"Sampler weights saved to: {sampler_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return training_client


def test_connection(args):
    """Test Tinker API connection."""
    print("Testing Tinker API connection...")

    try:
        service_client = tinker.ServiceClient()
        print("✓ ServiceClient created successfully")

        # Get model info
        print(f"\nTrying to get info for model: {args.model}")
        # Just try creating a client to verify connection
        training_client = service_client.create_lora_training_client(
            base_model=args.model,
            rank=8,
        )
        print("✓ TrainingClient created successfully")

        info = training_client.get_info()
        print(f"✓ Model info: {info}")

        tokenizer = training_client.get_tokenizer()
        print(f"✓ Tokenizer loaded, vocab size: {len(tokenizer)}")

        # Test encode/decode
        test_text = "BOM K0 TN2 TD4 S1 Cg Mu a Osup D32 EOM"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"✓ Tokenization test passed")
        print(f"  Original: {test_text}")
        print(f"  Tokens: {len(tokens)} tokens")
        print(f"  Decoded: {decoded}")

        print("\n✓ All connection tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Tinker OMR Training")

    # Data arguments
    parser.add_argument("--csv", type=str, default="~/data/paraff/ly+cc-vl20250324.csv",
                        help="Path to CSV file with image,sentence columns")
    parser.add_argument("--images-dir", type=str, default="~/data/scores/render/20250324/_Page",
                        help="Directory containing PNG images")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for testing)")

    # Model arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")

    # Training arguments
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--val-ratio", type=float, default=0.02,
                        help="Validation set ratio")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle data")

    # Logging arguments
    parser.add_argument("--log-dir", type=str, default="./logs/tinker_omr",
                        help="Log directory")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N steps")

    # Mode
    parser.add_argument("--test", action="store_true",
                        help="Test API connection only")

    args = parser.parse_args()

    if args.test:
        success = test_connection(args)
        sys.exit(0 if success else 1)
    else:
        train(args)


if __name__ == "__main__":
    main()
