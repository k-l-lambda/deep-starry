#!/usr/bin/env python3
"""
Tinker OMR Evaluation Script
Evaluate trained model by generating Paraff from test images.
"""

import os
import sys
import argparse
import io
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import difflib

# Use HF mirror for China network
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import tinker
from tinker import types
from PIL import Image


def get_image_processor(model_name: str):
    """Get image processor for a model."""
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(model_name, use_fast=True)


def image_to_chunk(image_bytes: bytes, image_processor) -> types.ImageChunk:
    """Convert image bytes to tinker ImageChunk with correct expected_tokens."""
    pil_image = Image.open(io.BytesIO(image_bytes))

    if pil_image.mode in ("RGBA", "LA", "P"):
        pil_image = pil_image.convert("RGB")

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    image_data = img_byte_arr.getvalue()

    width, height = pil_image.size
    num_image_tokens = (
        image_processor.get_number_of_image_patches(height, width, images_kwargs={})
        // image_processor.merge_size**2
    )

    return types.ImageChunk(
        data=image_data,
        format="jpeg",
        expected_tokens=num_image_tokens,
    )


# Prompt template
SYSTEM_PROMPT = """You are a music score recognition system. You convert sheet music images into Paraff notation, a domain-specific language for representing musical scores."""

QUESTION = "Recognize this music score as Paraff notation."


def build_inference_prompt(image_bytes: bytes, tokenizer, image_processor) -> types.ModelInput:
    """Build prompt for inference (without response)."""
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"

    system_str = f"{im_start}system\n{SYSTEM_PROMPT}{im_end}\n"
    user_prefix = f"{im_start}user\n{vision_start}"
    user_suffix = f"{vision_end}\n{QUESTION}{im_end}\n"
    assistant_prefix = f"{im_start}assistant\n"

    image_chunk = image_to_chunk(image_bytes, image_processor)

    system_tokens = tokenizer.encode(system_str, add_special_tokens=False)
    user_prefix_tokens = tokenizer.encode(user_prefix, add_special_tokens=False)
    user_suffix_tokens = tokenizer.encode(user_suffix, add_special_tokens=False)
    assistant_prefix_tokens = tokenizer.encode(assistant_prefix, add_special_tokens=False)

    chunks = [
        types.EncodedTextChunk(tokens=system_tokens),
        types.EncodedTextChunk(tokens=user_prefix_tokens),
        image_chunk,
        types.EncodedTextChunk(tokens=user_suffix_tokens),
        types.EncodedTextChunk(tokens=assistant_prefix_tokens),
    ]

    return types.ModelInput(chunks=chunks)


def compute_cer(pred: str, target: str) -> float:
    """Compute Character Error Rate."""
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0

    # Use difflib to compute edit operations
    matcher = difflib.SequenceMatcher(None, pred, target)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    errors = max(len(pred), len(target)) - matches
    return errors / len(target)


def compute_wer(pred: str, target: str) -> float:
    """Compute Word Error Rate (token-level for Paraff)."""
    pred_tokens = pred.split()
    target_tokens = target.split()

    if len(target_tokens) == 0:
        return 1.0 if len(pred_tokens) > 0 else 0.0

    # Dynamic programming for edit distance
    m, n = len(pred_tokens), len(target_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == target_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return dp[m][n] / len(target_tokens)


def evaluate(args):
    """Run evaluation."""
    print("=" * 60)
    print("Tinker OMR Evaluation")
    print("=" * 60)

    # Load test data
    print(f"\nLoading test data from {args.csv}...")
    images_dir = Path(args.images_dir).expanduser()
    df = pd.read_csv(Path(args.csv).expanduser())

    if args.limit:
        df = df.tail(args.limit)  # Use last N samples as test set

    test_data = []
    for _, row in df.iterrows():
        img_path = images_dir / row['image']
        if img_path.exists():
            test_data.append({
                'index': row['index'],
                'image_path': str(img_path),
                'paraff': row['sentence'],
            })

    print(f"Loaded {len(test_data)} test samples")

    # Initialize sampling client
    print(f"\nInitializing sampling client...")
    print(f"Loading weights from: {args.weights}")

    service_client = tinker.ServiceClient()

    if args.weights.startswith("tinker://"):
        # Load from tinker path
        sampling_client = service_client.create_sampling_client(model_path=args.weights)
    else:
        # Create base model client (no fine-tuning)
        sampling_client = service_client.create_sampling_client(base_model=args.model)

    # Load tokenizer separately (SamplingClient doesn't have get_tokenizer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Tokenizer loaded, vocab size: {len(tokenizer)}")

    print(f"Loading image processor...")
    image_processor = get_image_processor(args.model)

    # Get stop token
    im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    stop_token = im_end_tokens[0] if len(im_end_tokens) == 1 else None

    print(f"\nRunning inference on {len(test_data)} samples...")
    print("=" * 60)

    results = []
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0

    for i, sample in enumerate(tqdm(test_data)):
        # Load image
        with open(sample['image_path'], 'rb') as f:
            image_bytes = f.read()

        # Build prompt
        model_input = build_inference_prompt(image_bytes, tokenizer, image_processor)

        # Generate
        try:
            sampling_params = types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature if args.temperature > 0 else 0.0,
                stop_tokens=[stop_token] if stop_token else None,
            )
            response = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()

            # Decode response (response.sequences is a list of SampledSequence)
            if response.sequences:
                pred_text = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True).strip()
            else:
                pred_text = ""
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            pred_text = ""

        target_text = sample['paraff'].strip()

        # Compute metrics
        cer = compute_cer(pred_text, target_text)
        wer = compute_wer(pred_text, target_text)
        exact = pred_text == target_text

        total_cer += cer
        total_wer += wer
        if exact:
            exact_matches += 1

        results.append({
            'index': sample['index'],
            'target': target_text,
            'prediction': pred_text,
            'cer': cer,
            'wer': wer,
            'exact_match': exact,
        })

        # Print some examples
        if i < 3 or (args.verbose and i < 10):
            print(f"\n--- Sample {i} ---")
            print(f"Target:     {target_text[:100]}...")
            print(f"Prediction: {pred_text[:100]}...")
            print(f"CER: {cer:.4f}, WER: {wer:.4f}, Exact: {exact}")

    # Compute averages
    n = len(results)
    avg_cer = total_cer / n if n > 0 else 0
    avg_wer = total_wer / n if n > 0 else 0
    accuracy = exact_matches / n if n > 0 else 0

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Samples evaluated: {n}")
    print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Exact match accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Exact matches: {exact_matches}/{n}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            'model': args.model,
            'weights': args.weights,
            'num_samples': n,
            'avg_cer': avg_cer,
            'avg_wer': avg_wer,
            'exact_match_accuracy': accuracy,
            'exact_matches': exact_matches,
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results,
            }, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")

    return avg_cer, avg_wer, accuracy


def main():
    parser = argparse.ArgumentParser(description="Tinker OMR Evaluation")

    parser.add_argument("--csv", type=str, default="~/data/paraff/ly+cc-vl20250324.csv",
                        help="Path to CSV file")
    parser.add_argument("--images-dir", type=str, default="~/data/scores/render/20250324/_Page",
                        help="Directory containing images")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of test samples")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="Base model name")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to fine-tuned weights (tinker:// path)")

    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")

    parser.add_argument("--output", type=str, default="logs/eval_results.json",
                        help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print more examples")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
