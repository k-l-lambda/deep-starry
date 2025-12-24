# Deep Starry V2L (Vision-to-Language) Task Analysis

Research date: 2025-12-24

## 1. Current Architecture Overview

### Pipeline Flow

```
Score Image (PNG)
      ↓
[SigLIP Vision Encoder] (external, pre-computed)
      ↓
Image Embeddings (576 tokens × 1024 dim) → stored as .pt files
      ↓
[MLP Projector (Aligner)]  1024 → 4096 (2-layer MLP-GELU)
      ↓
[DeepSeek LLaMA 7B]
      ↓
Paraff Language Output
```

### Key Components

| Component | Implementation | Details |
|-----------|----------------|---------|
| Vision Encoder | SigLIP-L (384×384) | Pre-computed, stored in zip |
| Aligner | `MlpProjector` | depth=2, mlp_gelu, frozen |
| LLM | DeepSeek-LLM-7B | Via `LlamaForCausalLM` |
| Trainable | lm_head, model.norm, model.layers.29 | Partial fine-tuning |

---

## 2. Dataset Structure

### Data Files

```
~/data/paraff/
├── ly-vl20250226.csv        # Index file (5,331 samples)
├── ly-vl20250226.zip        # Pre-computed image embeddings
├── ly+cc-vl20250324.csv     # Larger dataset (14,517 samples)
└── ly+cc-vl20250324.zip     # Corresponding embeddings
```

### CSV Format

```csv
index,image,sentence
0,ly20241219.k330_1_114.ly-3c04d39a-1.png,BOM K0 TN2 TD4 S1 Cg G Mu a Osup Osup D32...
```

| Field | Description |
|-------|-------------|
| `index` | Unique ID (matches .pt filename in zip) |
| `image` | Original PNG filename |
| `sentence` | Paraff language representation |

### ZIP Contents

- Contains `{index}.pt` files
- Each .pt: Image embedding tensor (576 × 1024)
- Pre-computed from SigLIP vision encoder
- Total size: ~6GB (ly), ~17GB (ly+cc)

### Dataset Statistics

| Dataset | Samples | Avg Sentence Length | Max Length |
|---------|---------|---------------------|------------|
| ly-vl20250226 | 5,331 | 268 chars | 947 chars |
| ly+cc-vl20250324 | 14,517 | - | - |

---

## 3. Paraff Language Format

Paraff is a domain-specific language for music score representation:

### Example

```
BOM K0 TN2 TD4 S1 Cg G Mu a Osup Osup D32 Bl EslurL G b D32 Br Md c D8 ...EOM
```

### Token Categories

| Token | Meaning | Example |
|-------|---------|---------|
| `BOM/EOM` | Begin/End of Measure | - |
| `K{n}` | Key signature | K0 (C major), K_5 (5 flats) |
| `TN{n}/TD{n}` | Time signature | TN2 TD4 (2/4 time) |
| `S{n}` | Staff number | S1, S2 |
| `C{c}` | Clef | Cg (treble), Cf (bass) |
| `{note}` | Note name | a, b, c, d, e, f, g |
| `D{n}` | Duration | D4 (quarter), D8 (eighth) |
| `Osup/Osub` | Octave up/down | - |
| `Af/As` | Accidentals | Af (flat), As (sharp) |
| `Bl/Br` | Beam left/right | - |
| `EslurL/R` | Slur start/end | - |
| `VB` | Voice/staff break | - |

---

## 4. Model Configuration

### Config: `configs/paraff-visionund-test.yaml`

```yaml
data:
  type: VisionLanguage
  root: ly-vl20250226
  splits: '*1..98/100:99/100'  # 98% train, 2% val
  batch_size: 1

model:
  type: JanusLanguage
  args:
    model_path: ~/data/models/Janus-Pro-7B-language
    dtype: bfloat16
    aligner_cfg:
      depth: 2
      input_dim: 1024
      n_embed: 4096
      projector_type: mlp_gelu
    aligner_weights_path: ~/data/models/Janus-Pro-7B-aligner.pt
    trainable_parameters:
      - lm_head.
      - model.norm.
      - model.layers.29.
    additional_embedding_dims: [100603, 100705]  # Extra tokens

trainer:
  epoch: 20
  epoch_size: 256  # Steps per epoch
  moniter:
    field: acc
    mode: max

optim:
  type: Adam
  scheduler:
    type: InvSqrt
    args:
      n_warmup_steps: 256
      d_model: 4096
```

### Prompt Templates

**User Prompt (`paraffUndPrompt.jinja`):**
```
recognize this image as paraff
```
(Randomized variations: "parse", "understand", "decode", etc.)

**SFT Template (`janusSft.jinja`):**
```
<｜begin▁of▁sentence｜>You are a helpful language and vision assistant...
<|User|>: <image_placeholder>
{question}
<|Assistant|>: Sure.
```

### Special Tokens

| Token | ID Range |
|-------|----------|
| `<image_placeholder>` | Replaced by 576 image tokens |
| `<begin_of_image>` | Before image tokens |
| `<end_of_image>` | After image tokens |
| Additional embeddings | vocab[100603:100705] |

---

## 5. Training Flow

### VisionLanguage Dataset (`visionLanguage.py`)

```python
# For each sample:
1. Load image embedding from zip (576 × 1024 tensor)
2. Render prompt from Jinja template (randomized)
3. Tokenize: prompt + target (Paraff sentence) + EOS + PAD
4. Create input_ids with image placeholder expansion
5. Generate masks: image_seq_mask, target_mask, attention_mask
```

### JanusLanguageLoss (`janus.py`)

```python
# Forward pass:
1. Aligner: img_emb (576×1024) → (576×4096) via MLP
2. Embed input_ids, inject image embeddings at placeholder positions
3. Run LLaMA forward, get logits
4. CrossEntropy loss on target positions only
5. Compute accuracy metric
```

---

## 6. Current Limitations

1. **Pre-computed embeddings**: Vision encoder is frozen, can't end-to-end train
2. **Fixed resolution**: SigLIP uses 384×384, loses fine detail
3. **Partial fine-tuning**: Only last layer + head trainable
4. **Single image**: No multi-page support
5. **Batch size 1**: Memory constraint with 7B model

---

## 7. Comparison: Current vs Tinker Approach

| Aspect | Current (Local) | Tinker (Cloud) |
|--------|-----------------|----------------|
| Vision | Pre-computed SigLIP | End-to-end Qwen3-VL |
| Resolution | 384×384 fixed | Dynamic (native VL) |
| Model | DeepSeek-LLM-7B | Qwen3-VL-30B-A3B |
| Training | Partial fine-tune | LoRA on full model |
| Batch | 1 (memory limited) | Cloud-managed |
| Cost | GPU ownership | API usage |

---

## 8. Files Reference

```
starry/paraff/data/visionLanguage.py   # Dataset class
starry/paraff/models/janus.py          # Model + Loss
starry/janus/projector.py              # MLP Projector
assets/paraffUndPrompt.jinja           # Prompt template
assets/janusSft.jinja                  # SFT template
tools/makeIndexForParaffVL.py          # Data indexing tool
configs/paraff-visionund-test.yaml     # Training config
```
