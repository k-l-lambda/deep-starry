# Tinker API Analysis for OMR Task

Research date: 2025-12-24

## What is Tinker?

Tinker is a **Training API** from Thinking Machines Lab that handles distributed LLM fine-tuning infrastructure. You control the data, loss functions, and training logic while Tinker manages GPU distribution and hardware failures.

**Website:** https://tinker-docs.thinkingmachines.ai/

---

## Supported Vision-Language Models

Tinker supports these VLMs for training:

| Model | Type | Size |
|-------|------|------|
| Qwen/Qwen3-VL-235B-A22B-Instruct | MoE | Large |
| Qwen/Qwen3-VL-30B-A3B-Instruct | MoE | Medium |

These are the latest Qwen3-VL models with vision capabilities.

---

## Installation

```bash
# Core SDK
pip install tinker

# Optional cookbook (training code examples)
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
pip install -e .
```

**API Key:** Get from `tinker-console.thinkingmachines.ai`, set as `TINKER_API_KEY`

---

## How to Run OMR Task on Tinker

### Step 1: Create Training Client

```python
import tinker
from tinker import types

service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-VL-30B-A3B-Instruct"  # Vision model
)

tokenizer = training_client.get_tokenizer()
```

### Step 2: Prepare Vision Training Data

For OMR (Optical Music Recognition), prepare image + text pairs:

```python
# Load score image
with open("score_page.png", "rb") as f:
    image_data = f.read()

# Construct multimodal input
model_input = tinker.ModelInput(chunks=[
    types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
    types.ImageChunk(data=image_data, format="png"),
    types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>Convert this music score to Paraff format.<|im_end|>\n")),
    types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>assistant\n")),
])

# Target output (Paraff representation)
target_text = """{"measures": [{"notes": [...]}]}"""
target_tokens = tokenizer.encode(target_text + "<|im_end|>")

# Create training datum
input_tokens = model_input.to_ints()
all_tokens = input_tokens + target_tokens

# Weights: 0 for input (don't compute loss), 1 for target
weights = [0] * len(input_tokens) + [1] * len(target_tokens)

datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),  # input
    loss_fn_inputs={
        "target_tokens": types.TensorData.from_list(all_tokens[1:]),  # shifted targets
        "weights": types.TensorData.from_list(weights[1:]),
    }
)
```

### Step 3: Training Loop

```python
batch_size = 128  # Recommended smaller batch
learning_rate = 5e-5 * 10  # Base LR * LoRA multiplier (adjust per model)

for step in range(num_steps):
    # Get batch of training data
    batch = get_batch(dataset, batch_size)

    # Forward-backward pass
    fwdbwd_result = training_client.forward_backward(
        batch,
        loss_fn="cross_entropy"
    ).result()

    # Optimizer step
    optim_result = training_client.optim_step(
        types.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.98,
            eps=1e-9,
            grad_clip_norm=1.0,
        )
    ).result()

    print(f"Step {step}: loss = {fwdbwd_result.loss}")

    # Periodic checkpointing
    if step % 100 == 0:
        training_client.save_state(name=f"step_{step}")
```

### Step 4: Sampling / Inference

```python
# Save weights for sampling
sampling_client = training_client.save_weights_and_get_sampling_client(
    name="omr_model_v1"
)

# Prepare inference input
test_input = tinker.ModelInput(chunks=[
    types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
    types.ImageChunk(data=test_image_data, format="png"),
    types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>Convert this music score to Paraff format.<|im_end|>\n<|im_start|>assistant\n")),
])

# Generate
result = sampling_client.sample(
    prompt=test_input,
    sampling_params=types.SamplingParams(
        max_tokens=2048,
        temperature=0.0,  # Deterministic for OMR
        stop=["<|im_end|>"],
    ),
    num_samples=1
).result()

output_text = tokenizer.decode(result.sequences[0].tokens)
```

---

## Loss Functions Available

| Loss | Use Case |
|------|----------|
| `cross_entropy` | Supervised learning (OMR primary choice) |
| `importance_sampling` | Policy gradient RL |
| `ppo` | Proximal Policy Optimization |
| `cispo` | Clipped importance sampling |
| `dro` | Direct Reward Optimization |
| Custom | Via `forward_backward_custom()` |

For OMR, use **`cross_entropy`** with appropriate token weights.

---

## Hyperparameter Recommendations

### Learning Rate Formula

```
LR(m) = lr_base × M_LoRA × (2000/H_m)^P_m
```

Where:
- `lr_base = 5e-5`
- `M_LoRA = 10` (for LoRA training)
- `H_m` = model hidden size
- `P_m` = 0.0775 (Qwen) or 0.781 (Llama)

**Utility function:**
```python
from tinker_cookbook.hyperparam_utils import get_lr
lr = get_lr("Qwen/Qwen3-VL-30B-A3B-Instruct")
```

### Batch Size

- Recommended: **128** (smaller = better quality, slower training)
- Minimum steps: **100** (ideally 1000+)

---

## Saving and Loading

```python
# Save for sampling only (faster, smaller)
path = training_client.save_weights_for_sampler(name="v1").result().path

# Save full state (resume training)
path = training_client.save_state(name="checkpoint_100").result().path

# Load and resume
training_client.load_state(path)
```

---

## OMR Task Design Considerations

### Input Format

```
<|im_start|>user
<|vision_start|>{IMAGE}<|vision_end|>
{INSTRUCTION}
<|im_end|>
<|im_start|>assistant
{PARAFF_OUTPUT}
<|im_end|>
```

### Output Format Options

1. **Direct Paraff JSON** - Model outputs structured JSON
2. **Semantic tokens** - Model outputs semantic music representation
3. **MusicXML subset** - Standard music notation format

### Training Data Preparation

For your existing Paraff dataset:

```python
def prepare_omr_datum(score_image_path, paraff_json, tokenizer):
    with open(score_image_path, "rb") as f:
        image_data = f.read()

    # Use your existing Jinja templates
    prompt = render_template("paraffUndPrompt.jinja", ...)
    response = json.dumps(paraff_json)

    # Construct multimodal input
    # ... (as shown above)
```

---

## Integration with Deep Starry

### Comparison: Local vs Tinker Training

| Aspect | Local (current) | Tinker |
|--------|-----------------|--------|
| GPU | Your hardware | Cloud managed |
| Model | Janus-Pro-7B | Qwen3-VL-30B |
| Scaling | Manual | Automatic |
| Cost | Hardware + electricity | API usage |
| Control | Full | High (custom loss supported) |

### Migration Path

1. **Keep local setup** for Janus-Pro-7B experiments
2. **Use Tinker** for larger Qwen3-VL models
3. **Hybrid**: Train on Tinker, download weights for local inference

---

## CLI Commands

```bash
# View available commands
tinker --help

# Training
python -m tinker_cookbook.recipes.sl_basic  # Example SL training

# View logs
cat /tmp/tinker-examples/sl_basic/metrics.jsonl
```

---

## References

- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Tinker Console](https://tinker-console.thinkingmachines.ai/)
