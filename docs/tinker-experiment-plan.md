# Tinker OMR Fine-tuning Experiment Plan

Date: 2025-12-24

## Overview

This document outlines a plan to fine-tune a Vision-Language Model on Tinker for Optical Music Recognition (OMR), converting score images to Paraff representation.

---

## Phase 1: Data Preparation

### 1.1 Convert Dataset Format

Current format requires conversion for Tinker API:

```python
# Current: Pre-computed embeddings in ZIP
# Target: Raw PNG images for Qwen3-VL native processing

# Script: tools/prepare_tinker_dataset.py
import pandas as pd
from PIL import Image
import io

def prepare_tinker_data(csv_path, images_dir, output_path):
    """Convert Deep Starry dataset to Tinker format."""
    df = pd.read_csv(csv_path)

    records = []
    for _, row in df.iterrows():
        # Load original PNG image
        img_path = f"{images_dir}/{row['image']}"
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        records.append({
            'index': row['index'],
            'image_bytes': image_bytes,
            'paraff': row['sentence'],
        })

    return records
```

### 1.2 Dataset Statistics

| Dataset | Samples | Use |
|---------|---------|-----|
| ly-vl20250226 | 5,331 | Initial experiments |
| ly+cc-vl20250324 | 14,517 | Full training |

### 1.3 Train/Val Split

```python
# 98% train, 2% validation (matching current setup)
train_indices = [i for i in range(len(df)) if i % 100 < 98]
val_indices = [i for i in range(len(df)) if i % 100 >= 98]
```

---

## Phase 2: Tinker Training Setup

### 2.1 Model Selection

**Primary Choice: `Qwen/Qwen3-VL-30B-A3B-Instruct`**

Rationale:
- MoE architecture (30B total, 3B active) - efficient
- Native vision-language, no separate vision encoder needed
- Dynamic resolution support for score detail
- Available on Tinker API

**Alternative: `Qwen/Qwen3-VL-235B-A22B-Instruct`**
- For best quality if budget allows

### 2.2 Environment Setup

```bash
# Install Tinker
pip install tinker

# Clone cookbook for training utilities
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
pip install -e tinker-cookbook/

# Set API key
export TINKER_API_KEY=<your_key>
```

### 2.3 Prompt Design

Adapt from current templates:

```python
SYSTEM_PROMPT = """You are a music score recognition system. You convert sheet music images into Paraff notation, a domain-specific language for representing musical scores."""

def format_prompt(question: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
<|vision_start|><|image_placeholder|><|vision_end|>
{question}
<|im_end|>
<|im_start|>assistant
"""

# Randomized questions (from paraffUndPrompt.jinja)
QUESTIONS = [
    "Recognize this music score as Paraff notation.",
    "Parse the sheet music into Paraff language.",
    "Convert this score image to Paraff format.",
    "Transcribe this music sheet using Paraff.",
]
```

---

## Phase 3: Training Implementation

### 3.1 Core Training Script

```python
# train_omr_tinker.py
import tinker
from tinker import types
import random
from tqdm import tqdm

# === Configuration ===
BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
BATCH_SIZE = 128
LEARNING_RATE = 5e-4  # Base LR * 10 for LoRA
NUM_STEPS = 1000
SAVE_EVERY = 100
LOG_PATH = "./logs/omr_tinker"

# === Initialize Client ===
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    lora_config=types.LoraConfig(
        rank=16,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
    )
)
tokenizer = training_client.get_tokenizer()

# === Data Loading ===
def load_dataset(csv_path, images_dir):
    """Load and prepare dataset."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    data = []
    for _, row in df.iterrows():
        img_path = f"{images_dir}/{row['image']}"
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        data.append({
            'image': image_bytes,
            'paraff': row['sentence'],
        })
    return data

# === Datum Preparation ===
def prepare_datum(sample, tokenizer):
    """Convert sample to Tinker Datum format."""
    question = random.choice(QUESTIONS)
    prompt = format_prompt(question)
    response = sample['paraff']

    # Build multimodal input
    prompt_chunks = prompt.split("<|image_placeholder|>")

    model_input = tinker.ModelInput(chunks=[
        types.EncodedTextChunk(tokens=tokenizer.encode(prompt_chunks[0])),
        types.ImageChunk(data=sample['image'], format="png"),
        types.EncodedTextChunk(tokens=tokenizer.encode(prompt_chunks[1])),
    ])

    # Tokenize response
    response_tokens = tokenizer.encode(response + "<|im_end|>")

    # Combine for training
    input_tokens = model_input.to_ints()
    all_tokens = input_tokens + response_tokens

    # Loss weights: 0 for prompt, 1 for response
    weights = [0] * len(input_tokens) + [1] * len(response_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_list(all_tokens[1:]),
            "weights": types.TensorData.from_list(weights[1:]),
        }
    )

# === Training Loop ===
def train(train_data, val_data, num_steps):
    metrics = []

    for step in tqdm(range(num_steps)):
        # Sample batch
        batch_samples = random.choices(train_data, k=BATCH_SIZE)
        batch = [prepare_datum(s, tokenizer) for s in batch_samples]

        # Forward-backward
        fwdbwd_result = training_client.forward_backward(
            batch,
            loss_fn="cross_entropy"
        ).result()

        # Optimizer step
        optim_result = training_client.optim_step(
            types.AdamParams(
                learning_rate=LEARNING_RATE,
                beta1=0.9,
                beta2=0.98,
                eps=1e-9,
                grad_clip_norm=1.0,
            )
        ).result()

        train_loss = fwdbwd_result.loss

        # Logging
        if step % 10 == 0:
            print(f"Step {step}: train_loss = {train_loss:.4f}")
            metrics.append({'step': step, 'train_loss': train_loss})

        # Validation
        if step % 50 == 0 and val_data:
            val_loss = evaluate(val_data[:100])
            print(f"Step {step}: val_loss = {val_loss:.4f}")
            metrics[-1]['val_loss'] = val_loss

        # Checkpoint
        if step % SAVE_EVERY == 0 and step > 0:
            training_client.save_state(name=f"step_{step}")
            print(f"Saved checkpoint: step_{step}")

    return metrics

def evaluate(val_data):
    """Evaluate on validation set."""
    batch = [prepare_datum(s, tokenizer) for s in val_data]
    result = training_client.forward_backward(
        batch,
        loss_fn="cross_entropy"
    ).result()
    return result.loss

# === Main ===
if __name__ == "__main__":
    # Load data (使用完整数据集)
    train_data = load_dataset(
        "~/data/paraff/ly+cc-vl20250324.csv",
        "~/data/scores/render/20250324/_Page"
    )

    # Split
    train_set = [train_data[i] for i in range(len(train_data)) if i % 100 < 98]
    val_set = [train_data[i] for i in range(len(train_data)) if i % 100 >= 98]

    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # Train
    metrics = train(train_set, val_set, NUM_STEPS)

    # Save final model
    training_client.save_state(name="final")
    print("Training complete!")
```

### 3.2 Hyperparameter Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 16 | Balance efficiency/quality |
| Learning rate | 5e-4 | Base 5e-5 × 10 (LoRA multiplier) |
| Batch size | 128 | Tinker recommended |
| Gradient clip | 1.0 | Stability |
| Warmup | 100 steps | ~10% of training |
| Adam β | (0.9, 0.98) | Match current config |

---

## Phase 4: Evaluation & Inference

### 4.1 Create Sampling Client

```python
# After training
sampling_client = training_client.save_weights_and_get_sampling_client(
    name="omr_model_v1"
)
```

### 4.2 Inference Script

```python
def recognize_score(image_path: str, sampling_client) -> str:
    """Recognize music score and return Paraff notation."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    prompt = format_prompt("Recognize this music score as Paraff notation.")
    prompt_chunks = prompt.split("<|image_placeholder|>")

    model_input = tinker.ModelInput(chunks=[
        types.EncodedTextChunk(tokens=tokenizer.encode(prompt_chunks[0])),
        types.ImageChunk(data=image_bytes, format="png"),
        types.EncodedTextChunk(tokens=tokenizer.encode(prompt_chunks[1])),
    ])

    result = sampling_client.sample(
        prompt=model_input,
        sampling_params=types.SamplingParams(
            max_tokens=1024,
            temperature=0.0,  # Deterministic
            stop=["<|im_end|>", "EOM"],  # Stop at end
        ),
        num_samples=1
    ).result()

    output = tokenizer.decode(result.sequences[0].tokens)
    return output.strip()
```

### 4.3 Evaluation Metrics

```python
def evaluate_accuracy(predictions, targets):
    """Calculate token-level and sequence-level accuracy."""
    token_correct = 0
    token_total = 0
    seq_correct = 0

    for pred, target in zip(predictions, targets):
        pred_tokens = pred.split()
        target_tokens = target.split()

        # Token accuracy
        for p, t in zip(pred_tokens, target_tokens):
            if p == t:
                token_correct += 1
            token_total += 1

        # Sequence accuracy
        if pred.strip() == target.strip():
            seq_correct += 1

    return {
        'token_accuracy': token_correct / token_total,
        'sequence_accuracy': seq_correct / len(predictions),
    }
```

---

## Phase 5: Experiment Schedule

### Week 1: Setup & Pilot

| Day | Task |
|-----|------|
| 1 | Set up Tinker account, test API connection |
| 2 | Convert small dataset subset (500 samples) |
| 3-4 | Pilot training run (100 steps) |
| 5 | Debug data pipeline, validate outputs |

### Week 2: Initial Training

| Day | Task |
|-----|------|
| 1-2 | Full ly-vl dataset training (5K samples) |
| 3 | Evaluation on held-out set |
| 4-5 | Hyperparameter tuning (LR, rank) |

### Week 3: Full Scale

| Day | Task |
|-----|------|
| 1-3 | Train on ly+cc dataset (14K samples) |
| 4 | Compare with local Janus-Pro baseline |
| 5 | Document results, decide next steps |

---

## Phase 6: Cost Estimation

### Tinker Pricing (参考)

根据公开信息:
- **Qwen3-8B**: $0.40 / million tokens
- **新用户**: $150 免费额度
- **当前状态**: Private beta, 使用基础定价即将推出

### Token 估算

| 组件 | 估算 tokens |
|------|-------------|
| 系统提示 + 用户提示 | ~100 tokens |
| 图像 (VLM处理) | ~500-1000 tokens |
| Paraff 输出 (平均268字符) | ~100-150 tokens |
| **每样本总计** | ~700-1250 tokens |

### Training Costs (Estimated)

假设 Qwen3-VL-30B 定价约为 8B 的 3-4x:

| 实验 | Samples | Steps | Batch | Tokens | Est. Cost |
|------|---------|-------|-------|--------|-----------|
| Pilot | 500 | 100 | 128 | ~12.8M | ~$5-15 |
| Small | 5K | 1000 | 128 | ~128M | ~$50-150 |
| Full | 14K | 3000 | 128 | ~384M | ~$150-450 |

*实际价格以 Tinker console 为准，新用户有 $150 额度可覆盖初期实验*

### Comparison with Local Training

| Aspect | Local (Janus-Pro-7B) | Tinker (Qwen3-VL-30B) |
|--------|---------------------|----------------------|
| GPU | RTX 4090 (24GB) | Cloud-managed |
| 模型参数 | 7B (全参数) | 30B (3B active MoE) |
| Vision | 预计算 SigLIP 384×384 | 原生 VLM 动态分辨率 |
| 训练方式 | 部分微调 | LoRA 全模型 |
| 成本 | 电费+硬件折旧 | API 按量付费 |
| Quality | Baseline | Expected better |

---

## Phase 7: Integration Plan

### 7.1 Download Weights

```bash
# After training, download for local deployment
tinker download <model_path> --output ./models/omr_qwen3vl/
```

### 7.2 Integration with Deep Starry

Option A: **Replace Janus pipeline entirely**
- Use Qwen3-VL for both vision and language
- Requires updating inference code

Option B: **Hybrid approach**
- Keep current pipeline for fast local inference
- Use Tinker model for high-quality output

### 7.3 New Config Template

```yaml
# configs/paraff-tinker-qwen3vl.yaml
id: paraff/{date}-tinker-qwen3vl

data:
  type: VisionLanguageRaw  # New class for raw images
  root: ly+cc-vl20250324
  splits: '*1..98/100:99/100'

model:
  type: TinkerQwen3VL  # New wrapper class
  args:
    base_model: Qwen/Qwen3-VL-30B-A3B-Instruct
    checkpoint: tinker://<model_id>/final

trainer:
  # Tinker handles training, this is for inference config
  device: tinker_api
```

---

## Appendix: File Structure

```
deep-starry/
├── docs/
│   ├── v2l-task-analysis.md
│   ├── tinker-omr-guide.md
│   └── tinker-experiment-plan.md  (this file)
├── tools/
│   ├── prepare_tinker_dataset.py  (to create)
│   └── train_omr_tinker.py        (to create)
├── configs/
│   └── paraff-tinker-qwen3vl.yaml (to create)
└── starry/
    └── paraff/
        └── models/
            └── tinker_wrapper.py  (to create)
```

---

## Next Steps

1. [x] Get Tinker API access and test connection (已有 API key)
2. [x] Verify original PNG images are available
   - 路径: `~/data/scores/render/20250324/_Page/`
   - 数量: 14,517 张 PNG
   - 大小: ~4-5KB/张
3. [ ] Create `prepare_tinker_dataset.py` script
4. [ ] Run pilot training with 500 samples
5. [ ] Compare results with current Janus-Pro baseline
