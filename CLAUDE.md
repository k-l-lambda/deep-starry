# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Starry is a multimodal deep learning framework for music score understanding. It combines vision models (for sheet music images), language models (LLaMA-based), and MIDI sequence processing to understand and manipulate musical scores.

## Common Commands

### Training
```bash
# Main training entry point
python -m train configs/<config>.yaml

# Topology training
python3 ./trainTopology2.py ./configs/test-topology.yaml --tr 2

# Vision-language training (current focus)
python -m train configs/paraff-visionund-test.yaml
```

### Validation
```bash
# Measure validation with checkpoint
CUDA_VISIBLE_DEVICES=0 python -m validateMeasure <training_dir> -d ./configs/<data_config>.yaml -s 0/100 -dv cuda -cp latest.chkpt

# Other validation scripts: validateParaff, validateTopology, validateVision, validateStamp
```

### Data Preprocessing
```bash
python -m preprocessMidiseq configs/midiseq-data.local.yaml
python -m tools.makeIndexForParaffVL <data.yaml> <output_dir>
```

### Model Conversion
```bash
python -m convertToJit <training_dir> -p JitEnc
```

### Utilities
```bash
python -m viewDataset configs/<config>.yaml -p -s 0/100
python -m tools.saveJanusLanguage <training_dir>
```

## Architecture

### Core Pipeline
```
Score Image → Vision Module → Janus Projector → LLM (LLaMA) → Paraff Models → MIDI/Score Output
```

### Main Modules (`starry/`)

- **janus/**: Vision-language bridge. `projector.py` implements MLP/Linear projectors that align image features with LLM embeddings. Current focus uses Deepseek Janus-Pro-7B.

- **paraff/**: Core music sequence processing. Key models:
  - `JanusLanguage` - Vision-language score understanding
  - `SeqShareVAE/SeqShareSE` - Shared encoders for sequences
  - `GraphParaffEncoder` - Score graph encoding
  - `MidiParaffTranslator` - MIDI ↔ score translation

- **topology/**: Score structure understanding with jointer models and BeadPicker.

- **vision/**: CNN-based score image processing (ScoreWidgets, GlyphRecognizer, etc.)

- **melody/**: MIDI encoding and measure-wise processing.

- **transformer/**, **lora/**: Custom transformer implementations and LoRA fine-tuning.

### Infrastructure (`starry/utils/`)

- **config.py**: YAML configuration loading, auto-generates training IDs from `{date}-{filename}` template, persists state to `.state.yaml`
- **trainer.py**: Training loop with checkpointing and monitoring
- **model_factory.py**: Factory pattern for 50+ registered model classes
- **dataset_factory.py**: Dynamic dataset loading

### Configuration System

Configs in `configs/` follow pattern `{module}-{variant}-{date}.yaml`:
```yaml
id: paraff/{date}-{filename}    # Auto-templated

data:
  type: VisionLanguage          # Dataset class from factory
  root: ly-vl20250226
  splits: '*1..98/100:99/100'   # Train/val split syntax

model:
  type: JanusLanguage           # Model class from factory
  args:
    trainable_parameters: [...]  # Selective parameter freezing

trainer:
  moniter:
    field: acc
    mode: max

optim:
  scheduler:
    type: InvSqrt
```

## Environment Variables

Set in `.env` or `.env.local`:
- `TRAINING_DIR` - Checkpoint output (default: `./training`)
- `DATA_DIR` - Dataset root (required)
- `VISION_DATA_DIR` - Vision dataset root
- `CHECK_HOST_CMD` - Health check command (e.g., `pm2 list;nvidia-smi`)

## Testing

Tests are Jupyter notebooks in `tests/`:
- `janus_language_model.ipynb` - Vision-language model tests
- `paraffEncoder.ipynb`, `midiseqEncoder.ipynb` - Encoder tests
- `paraffVisionUnderstand.ipynb` - Vision understanding tests

## Git Workflow

- Main branch: `master`
- Development: `develop`
- Feature branches: `feature/paraff`, `feature/topo-*`, etc.
