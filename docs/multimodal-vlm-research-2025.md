# Open-Source Multimodal Vision-Language Models Research (2025)

Research date: 2025-12-24

## Overview

This document evaluates open-source multimodal VLMs as alternatives to Deepseek Janus-Pro-7B for the music score image → Paraff task. The key requirements are:

1. **Fine-grained OCR capability** - Recognize notes, symbols, barlines
2. **Structure understanding** - Understand score hierarchy and layout
3. **Long context** - Process full score pages

---

## Tier 1: Recommended Models

### Qwen2.5-VL

| Spec | Value |
|------|-------|
| Developer | Alibaba Cloud |
| Sizes | 3B / 7B / 32B / 72B |
| Context | 32K+ tokens (YaRN extension supported) |
| Languages | English, Chinese |
| License | Apache 2.0 |

**Key Strengths:**
- Best-in-class OCR and document understanding
- Dynamic resolution support (advantageous for high-detail score images)
- DocVQA, TextVQA benchmark leader among open models
- 7B version matches Janus-Pro-7B parameter count (easy migration)

**Architecture:**
```
ViT (dynamic resolution) + Qwen2.5-7B + Cross-attention
```

### InternVL3

| Spec | Value |
|------|-------|
| Developer | Shanghai AI Lab (OpenGVLab) |
| Sizes | 1B / 2B / 8B / 26B / 38B / 78B |
| Context | 256K tokens |
| License | Apache 2.0 |

**Key Strengths:**
- MMMU benchmark SOTA: 72.2 (78B version)
- DocVQA: 94.1%, ChartQA: 88.4%, TextVQA: 84.4%
- Best overall open-source VLM performance
- Multiple size options for different deployment scenarios

**Architecture:**
```
InternViT-300M + InternLM2.5 + MLP Projector
```

### GLM-4.1V-9B-Thinking

| Spec | Value |
|------|-------|
| Developer | Zhipu AI |
| Sizes | 9B |
| Context | 128K tokens |

**Key Strengths:**
- Strong document analysis and reasoning
- Chain-of-thought support for complex tasks
- Good balance of size and performance

---

## Tier 2: Strong Alternatives

### Pixtral 12B

| Spec | Value |
|------|-------|
| Developer | Mistral AI |
| Sizes | 12B (400M vision encoder), 124B Large |
| Context | 128K tokens |
| License | Apache 2.0 (12B), Research License (Large) |

**Key Strengths:**
- Excellent instruction following
- Multi-language support (dozens)
- Good performance on complex visual tasks

### Llama 3.2 Vision

| Spec | Value |
|------|-------|
| Developer | Meta |
| Sizes | 11B / 90B |
| Context | 128K tokens |
| License | Llama 3.2 Community License |

**Key Strengths:**
- Strong Meta ecosystem
- Mobile/edge deployment friendly
- Good document OCR capabilities

### Gemma 3

| Spec | Value |
|------|-------|
| Developer | Google DeepMind |
| Sizes | 4B / 12B / 27B |
| Context | 128K tokens |

**Key Strengths:**
- 140+ languages
- Dynamic image segmentation (896×896 base, higher with tiling)
- Efficient training with distillation

### Phi-4 Multimodal

| Spec | Value |
|------|-------|
| Developer | Microsoft |
| Sizes | 5.6B |
| Context | 128K tokens |
| License | MIT |

**Key Strengths:**
- Extremely compact yet powerful
- Unified vision/audio/text processing
- MIT license (most permissive)

---

## Tier 3: Lightweight / Edge Deployment

| Model | Size | Use Case |
|-------|------|----------|
| MiniCPM-V 2.6 | 8B | Mobile/edge |
| Moondream2 | 1.9B | Ultra-lightweight |
| CogVLM2 | 19B | Fine-grained understanding |

---

## Architecture Comparison with Janus-Pro

```
Janus-Pro-7B:    SigLIP-L (384×384) + DeepSeek-LLM-7B + MLP Projector
Qwen2.5-VL-7B:   ViT (dynamic res)  + Qwen2.5-7B     + Cross-attention
InternVL3-8B:    InternViT-300M     + InternLM2.5-7B + MLP Projector
Pixtral-12B:     400M vision enc    + Mistral-12B    + Multimodal decoder
```

**Note:** Qwen2.5-VL's dynamic resolution is particularly beneficial for music scores which require high-resolution detail recognition.

---

## Music Score Understanding Research

Recent academic work on VLMs for music score understanding:

- **"Advancing Visual Music Understanding in Multimodal LLMs"** (2025) - Defines VQA tasks for evaluating MLLMs on sheet music
- **Sheet Music Benchmark (SMB)** - 685 pages for OMR benchmarking (ISMIR 2025)
- **Qwen2-Audio** - Has been explored for music transcription tasks

---

## Recommendation for Deep Starry

**Primary recommendation: Qwen2.5-VL-7B or Qwen2.5-VL-32B**

Rationale:
1. Best OCR/document understanding performance
2. Dynamic resolution handles score detail better
3. 7B matches current Janus-Pro parameter count
4. Apache 2.0 license
5. Active development with Qwen3-VL coming

**Secondary: InternVL3-8B**

If you need stronger overall multimodal reasoning at similar parameter count.

---

## Migration Considerations

### From Janus-Pro to Qwen2.5-VL

1. **Projector architecture change**: Janus uses MLP projector, Qwen uses cross-attention
2. **Vision encoder**: Replace SigLIP-L with Qwen's ViT
3. **Tokenizer**: Switch from DeepSeek tokenizer to Qwen tokenizer
4. **Input format**: Update image placeholder tokens

### Tinker Platform Support

Tinker (thinkingmachines.ai) provides training API with support for:
- Qwen3-VL-235B-A22B-Instruct (MoE)
- Qwen3-VL-30B-A3B-Instruct (MoE)

This enables cloud-based fine-tuning without local GPU infrastructure.

---

## References

- [HuggingFace Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [Qwen2.5-VL GitHub](https://github.com/QwenLM/Qwen2.5-VL)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [Koyeb VLM Comparison](https://www.koyeb.com/blog/best-multimodal-vision-models-in-2025)
