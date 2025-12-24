#!/usr/bin/env python3
"""
Test chat capabilities of fine-tuned model vs base model.
Check for catastrophic forgetting.
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import argparse
import tinker
from tinker import types
from transformers import AutoTokenizer


def build_chat_prompt(messages: list[dict], tokenizer) -> types.ModelInput:
    """Build chat prompt in Qwen3 format."""
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt_str = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_str += f"{im_start}{role}\n{content}{im_end}\n"

    # Add assistant prefix for generation
    prompt_str += f"{im_start}assistant\n"

    tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
    return types.ModelInput.from_ints(tokens)


def test_model(sampling_client, tokenizer, model_name: str, questions: list[str]):
    """Test model with a list of questions."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Get stop token
    im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    stop_token = im_end_tokens[0] if len(im_end_tokens) == 1 else None

    results = []

    for i, question in enumerate(questions):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {question}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]

        prompt = build_chat_prompt(messages, tokenizer)

        sampling_params = types.SamplingParams(
            max_tokens=256,
            temperature=0.7,
            stop_tokens=[stop_token] if stop_token else None,
        )

        try:
            response = sampling_client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()

            if response.sequences:
                answer = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True).strip()
            else:
                answer = "[No response]"
        except Exception as e:
            answer = f"[Error: {e}]"

        print(f"A: {answer[:500]}{'...' if len(answer) > 500 else ''}")
        results.append({"question": question, "answer": answer})

    return results


def main():
    parser = argparse.ArgumentParser(description="Test chat capabilities")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--finetuned-weights", type=str,
                        default="tinker://61b37718-ee38-5575-9dee-e54a4bd73aba:train:0/sampler_weights/final_sampler")
    parser.add_argument("--test-base", action="store_true", help="Also test base model")
    args = parser.parse_args()

    # Test questions covering different capabilities
    questions = [
        # Basic reasoning
        "What is 15 + 27?",

        # Common knowledge
        "What is the capital of France?",

        # Explanation
        "Explain what machine learning is in 2 sentences.",

        # Creative
        "Write a haiku about programming.",

        # Instruction following
        "List 3 benefits of exercise.",

        # Music-related (domain adjacent to our fine-tuning)
        "What are the basic elements of music notation?",
    ]

    service_client = tinker.ServiceClient()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("="*60)
    print("Catastrophic Forgetting Test")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Fine-tuned weights: {args.finetuned_weights}")
    print(f"Questions: {len(questions)}")

    # Test fine-tuned model
    print("\n\nLoading fine-tuned model...")
    finetuned_client = service_client.create_sampling_client(model_path=args.finetuned_weights)
    finetuned_results = test_model(finetuned_client, tokenizer, "Fine-tuned Model", questions)

    # Optionally test base model for comparison
    if args.test_base:
        print("\n\nLoading base model...")
        base_client = service_client.create_sampling_client(base_model=args.model)
        base_results = test_model(base_client, tokenizer, "Base Model", questions)

        # Compare
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        for i, q in enumerate(questions):
            print(f"\nQ{i+1}: {q[:50]}...")
            print(f"  Base:      {base_results[i]['answer'][:80]}...")
            print(f"  Finetuned: {finetuned_results[i]['answer'][:80]}...")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
