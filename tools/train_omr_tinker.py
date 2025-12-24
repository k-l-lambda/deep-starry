#!/usr/bin/env python3
"""
Tinker OMR Training Script
Train a Vision-Language Model for Optical Music Recognition (score image -> Paraff)
"""

import os
import sys
import random
import argparse
import io
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime

# Use HF mirror for China network (avoid proxy issues with Tinker API)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import tinker
from tinker import types
from PIL import Image


import torch


# Helper function to compute NLL from forward_backward output
def compute_mean_nll(logprobs_list, weights_list) -> float:
    """Compute weighted mean negative log likelihood."""
    total_weighted_logprobs = 0.0
    total_weights = 0.0

    for logprobs, weights in zip(logprobs_list, weights_list):
        logprobs_torch = torch.tensor(logprobs.data)
        weights_torch = torch.tensor(weights.data)
        total_weighted_logprobs += logprobs_torch.dot(weights_torch).item()
        total_weights += weights_torch.sum().item()

    if total_weights == 0:
        return float("nan")

    return float(-total_weighted_logprobs / total_weights)


# === Image Processor Utilities (from tinker-cookbook) ===
_image_processor_cache = {}

def get_image_processor(model_name: str):
    """Get image processor for a model (cached)."""
    if model_name not in _image_processor_cache:
        from transformers.models.auto.image_processing_auto import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        _image_processor_cache[model_name] = processor
    return _image_processor_cache[model_name]


def image_to_chunk(image_bytes: bytes, image_processor) -> types.ImageChunk:
    """Convert image bytes to tinker ImageChunk with correct expected_tokens."""
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed (JPEG doesn't support RGBA/LA/P modes)
    if pil_image.mode in ("RGBA", "LA", "P"):
        pil_image = pil_image.convert("RGB")

    # Save as JPEG
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    image_data = img_byte_arr.getvalue()

    width, height = pil_image.size

    # Calculate number of image tokens using image processor
    num_image_tokens = (
        image_processor.get_number_of_image_patches(height, width, images_kwargs={})
        // image_processor.merge_size**2
    )

    return types.ImageChunk(
        data=image_data,
        format="jpeg",
        expected_tokens=num_image_tokens,
    )


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


def prepare_datum(sample: dict, image_bytes: bytes, tokenizer, image_processor) -> types.Datum:
    """Convert sample to Tinker Datum format for vision-language training.

    Uses Qwen3-VL Instruct format (no thinking tokens).
    Following tinker-cookbook conventions for next-token prediction:
    - model_input has last token removed
    - target_tokens has first token removed
    - weights aligned to target_tokens
    """
    question = random.choice(QUESTIONS)
    response = sample['paraff']

    # Qwen3-VL special tokens
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"

    # Build prompt in Qwen3 Instruct format (no thinking tags)
    # System message
    system_str = f"{im_start}system\n{SYSTEM_PROMPT}{im_end}\n"

    # User message with image
    user_prefix = f"{im_start}user\n{vision_start}"
    user_suffix = f"{vision_end}\n{question}{im_end}\n"

    # Assistant response (no thinking for Instruct models)
    assistant_prefix = f"{im_start}assistant\n"
    assistant_content = f"{response}{im_end}"

    # Create image chunk with correct expected_tokens
    image_chunk = image_to_chunk(image_bytes, image_processor)
    num_image_tokens = image_chunk.expected_tokens

    # Tokenize all text parts
    system_tokens = tokenizer.encode(system_str, add_special_tokens=False)
    user_prefix_tokens = tokenizer.encode(user_prefix, add_special_tokens=False)
    user_suffix_tokens = tokenizer.encode(user_suffix, add_special_tokens=False)
    assistant_prefix_tokens = tokenizer.encode(assistant_prefix, add_special_tokens=False)
    assistant_content_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)

    # Build full sequence chunks (will truncate last token for input)
    full_chunks = [
        types.EncodedTextChunk(tokens=system_tokens),
        types.EncodedTextChunk(tokens=user_prefix_tokens),
        image_chunk,
        types.EncodedTextChunk(tokens=user_suffix_tokens),
        types.EncodedTextChunk(tokens=assistant_prefix_tokens),
        types.EncodedTextChunk(tokens=assistant_content_tokens),
    ]

    # Calculate full sequence length
    prompt_length = (
        len(system_tokens) +
        len(user_prefix_tokens) +
        num_image_tokens +
        len(user_suffix_tokens) +
        len(assistant_prefix_tokens)
    )
    response_length = len(assistant_content_tokens)
    total_length = prompt_length + response_length

    # Create weights for full sequence: 0 for prompt, 1 for response
    full_weights = [0.0] * prompt_length + [1.0] * response_length

    # For next-token prediction:
    # - Input: all tokens except last
    # - Target: all tokens except first (shifted left)
    # - Weights: aligned with target (remove first weight)

    # Build input chunks (remove last token from last chunk)
    if assistant_content_tokens:
        input_chunks = [
            types.EncodedTextChunk(tokens=system_tokens),
            types.EncodedTextChunk(tokens=user_prefix_tokens),
            image_chunk,
            types.EncodedTextChunk(tokens=user_suffix_tokens),
            types.EncodedTextChunk(tokens=assistant_prefix_tokens),
            types.EncodedTextChunk(tokens=assistant_content_tokens[:-1]) if len(assistant_content_tokens) > 1 else None,
        ]
        input_chunks = [c for c in input_chunks if c is not None]
    else:
        input_chunks = full_chunks[:-1]

    model_input = types.ModelInput(chunks=input_chunks)

    # Target tokens: collect all tokens, shift left (remove first)
    all_tokens = (
        system_tokens +
        user_prefix_tokens +
        [0] * num_image_tokens +  # Placeholder for image tokens
        user_suffix_tokens +
        assistant_prefix_tokens +
        assistant_content_tokens
    )
    target_tokens = all_tokens[1:]  # Left shift

    # Weights: align with target (remove first weight)
    weights = full_weights[1:]

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

    # Load image processor for calculating image token counts
    print(f"Loading image processor for {args.model}...")
    image_processor = get_image_processor(args.model)
    print(f"Image processor loaded (merge_size={image_processor.merge_size}, patch_size={image_processor.patch_size})")

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
            datum = prepare_datum(sample, image_bytes, tokenizer, image_processor)
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

        # Compute loss from logprobs and weights
        logprobs = [x["logprobs"] for x in fwdbwd_result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in batch]
        train_loss = compute_mean_nll(logprobs, weights)

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
                datum = prepare_datum(sample, image_bytes, tokenizer, image_processor)
                val_batch.append(datum)

            val_result = training_client.forward_backward(
                val_batch,
                loss_fn="cross_entropy"
            ).result()

            val_logprobs = [x["logprobs"] for x in val_result.loss_fn_outputs]
            val_weights = [datum.loss_fn_inputs["weights"] for datum in val_batch]
            val_loss = compute_mean_nll(val_logprobs, val_weights)
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
