---
name: tune-train-config
description: Measure-first tuning of deep-starry training config trainer/data fields (batch_size, epoch_size, splits, report_step_unit)
metadata:
  tags: deep-starry, training, config, batch-size, epoch-size, gpu-memory, distributed-trainer, tensorboard
---

## When to use

Use this skill when:
- Setting `data.batch_size` / `trainer.epoch_size` / `data.splits` for a deep-starry training run
- A training config errors on a missing trainer key (`epoch` vs `epochs`) or model attribute
- Deciding val-set size, validation cadence, or TensorBoard x-axis (`report_step_unit`)
- Porting a config to the distributed trainer (`trainerQuantitative`)

# Skill: Tuning deep-starry training configs (trainer fields)

Follow this when tuning the `trainer:` / `data:` fields of a deep-starry training config.
The core principle is **measure first, then set**: batch_size is bounded by GPU memory,
epoch_size is set by balancing per-epoch wall time between train and val. Both are derived
by running a few real steps on the actual data, not guessed.

## Two trainers — know which one you're on
- `starry/utils/trainer.py` (single process): uses `self.options['epoch']` (**singular**).
- `starry/utils/trainerQuantitative.py` (**distributed, two processes**): uses
  `self.options['epochs']` (**plural**).
  - rank 0 = TRAINER, rank 1 = VALIDATOR, `PROC_COUNT=2`, so it **occupies at least 2 GPUs**.
  - The two colon-separated segments of `data.splits` are assigned by `splits[rank]`:
    the trainer takes the train segment, the validator the val segment. The two processes
    sync parameters via `broadcastParam` (NCCL).
- **Lesson**: the config's `epoch`/`epochs` key must match the trainer in use, or you get a
  KeyError. The distributed trainer uses `epochs`.

## trainer fields, one by one
| Field | Purpose | Guidance |
|-------|---------|----------|
| `epochs` (distributed) / `epoch` (single) | Upper bound on epoch count | For long distillation runs set it large (e.g. 10000) and rely on `save_mode: best` + manual early-stop judgement |
| `epoch_size` | **Bounds TRAIN only**: each epoch runs `epoch_size // batch_size` steps (train uses `infiniteTraverse`, so it need not cover the full train set). **Unit is samples** | See "Balancing epoch_size" below |
| `report_step_unit` | TensorBoard x-axis unit: `examples` → x = `steps × batch_size` (cumulative samples); otherwise = epoch index | **Prefer `examples`**: after changing epoch_size/batch_size the curves still align on a sample-count axis and stay comparable, instead of breaking when the epoch definition changes |
| `epoch_duration` | **Validator rank only**: on a fresh run (`trainer.latest` falsy) the validator first `sleep(epoch_duration)` seconds so the trainer can finish epoch 0, avoiding broadcasting empty params at startup | resume / want val to start immediately → `0`; fresh run with long epochs → leave default 1800 or set to about one train-epoch in seconds |
| `save_mode` | `best` saves only checkpoints that improve the moniter metric; `all` saves every epoch | use `best` on long runs to save disk |
| `moniter.field` / `.mode` | Metric and direction for picking "best" (`loss`/`min`, or `cos`/`max`, etc.) | for distillation watch val `loss` (=1-cos) → `min` |
| `device` | `cuda` | — |
| `gpus` (distributed, optional) | Physical GPU count; a rank picks its card via `rank % gpus`; None → falls back to `PROC_COUNT` | — |
| `val_batch_size` (distributed, optional) | Validator's batch_size; None → uses `data.batch_size` | val is forward-only and uses less memory, so it can go larger |

## Choosing batch_size: probe the memory ceiling, keep headroom
`data.batch_size`. Steps:
1. On one target GPU, sweep candidate bs from large to small (e.g. 96→64→48→32→16). For each,
   run several fwd+bwd+optimizer steps on **real training batches** and record
   `torch.cuda.max_memory_allocated`.
2. Take the **largest non-OOM value that still leaves comfortable headroom**. Don't sit right
   at the limit — other processes on a shared node, allocator fragmentation, and the
   occasional very long sample can all tip it over.
3. Look at **samples/s** (=`bs / step_time`), not it/s: larger batches often lower it/s
   because each step does more work; once throughput has plateaued, raising bs further only
   buys OOM risk with no gain.

Measured example (LilyletM3 distillation, 143GB card, max_patches=2048 worst case):
bs=48→124GB (against the ceiling, risky), bs=32→83GB (58%, safe), bs=16→39GB. Throughput
plateaus around bs≈32-36 → **chose bs=32** (best samples/s + ample headroom).

> Memory rises steeply with the sequence-length cap. If there's a `max_patches`/`patch_length`
> upper bound, the OOM probe MUST use worst-case batches that hit the cap, otherwise you only
> OOM in production when a long sample shows up.

## Balancing epoch_size: make a train epoch ≈ a val epoch
Goal: one train epoch's wall time ≈ one full val pass's wall time (**val may be slightly
shorter**), so validation runs at a reasonable cadence.
- Mechanism: val always traverses the full val set (fixed time); a train epoch's time =
  `(epoch_size / batch_size) × train_step_time`.
- Measure two numbers: the post-warmup **steady-state train step time** (fwd+bwd) and the
  **full val pass time**. Note a train step is typically ~3× a val step (val is `no_grad`
  forward only — no backward / optimizer).
- Set them equal: `epoch_size ≈ val_epoch_time / train_step_time × batch_size`, rounded to an
  integer multiple of batch_size and slightly above that value (so val is slightly shorter).

> **Degenerate trap**: if the val set is tiny, the balanced epoch_size comes out absurdly
> small (a few dozen steps), so you full-validate every few dozen train steps — validation
> overhead dominates and the train set takes hundreds of epochs to see once. Fix: **enlarge
> the val set first** (also more stable statistics), then balance.

Measured example: with val=1K, bs=32, a val pass was just 8.8s and a train step 822ms →
balancing gave epoch_size≈352 (11 steps/epoch, degenerate). After switching to **val=5K**:
val pass 45.5s, epoch_size=1792 (56×32) → train epoch 46.4s ≈ val epoch 45.5s (ratio 1.02,
val slightly shorter). **Record the post-warmup val rate**: at bs=32, a 5K val pass = 45.5s
≈ 3.45 it/s / 110 samples/s.

## Splits (`data.splits`)
Format `'<train-seg>:<val-seg>'`, each segment a `phases/cycle` filter (see
`starry/utils/parsers.py:parseFilterStr`): a sample falls into phase `index % cycle`, and the
listed phases belong to that segment. A leading `*` shuffles that segment (the train segment
should shuffle).
- e.g. `'*0..18/20:19/20'`: cycle=20, train = phases 0..18 (95%≈95K, shuffled),
  val = phase 19 (5%≈5K).
- To change the val fraction, adjust cycle and the number of val phases. For val≈1% use
  cycle=100 with 1 val phase; for 5% use cycle=20 with 1 val phase.

> Works with `args_variant`: `args_variant: {1: {random_truncate: false}}` makes the **2nd
> segment (val)** use deterministic truncation (head cut, reproducible), while the train
> segment uses random truncation as augmentation. The index matches the segment order in splits.

## End-to-end tuning flow
1. Data in place; confirm `data.root` (mind the `.env.local` `DATA_DIR` override — an absolute
   path is safest).
2. Set `splits`: make val large enough (≥ several K, for stable stats + to avoid the epoch_size
   degeneracy).
3. Measure batch_size: run a memory sweep with worst-case batches, take the value at saturated
   throughput with headroom.
4. Measure epoch_size: after warmup, time a train step and a full val pass, balance to an
   integer multiple of batch_size.
5. Use the right trainer key (`epochs` vs `epoch`); `report_step_unit: examples` makes curves
   comparable across configs.
6. Backfill the config, writing the measured numbers and rationale into the comments (so future
   you / others can review with evidence).
7. Run through warmup, confirm train/val per-epoch times match expectations, then let it run long.
