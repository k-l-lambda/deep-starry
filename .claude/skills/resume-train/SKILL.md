---
name: resume-train
description: Resume (or cleanly stop) a deep-starry training experiment — pass the EXPERIMENT DIRECTORY not the config file, edit the dir's .state.yaml, restore the right checkpoint + LR step count, stop via the __SHUT marker.
metadata:
  tags: deep-starry, training, resume, checkpoint, trainDist, distributed-trainer, state.yaml, __SHUT, learning-rate, gpu
---

## When to use

Use this skill when:
- Resuming a stopped/crashed/diverged deep-starry training run (`trainDist.py` or `train.py`).
- Restarting a run from an earlier (healthy) checkpoint after a divergence.
- Stopping a running train+val experiment cleanly.
- Changing a trainer/optim field (grad_clip, lr, epoch_size…) on an EXISTING run.

If you are creating a BRAND-NEW run or tuning batch_size/epoch_size for the first time, use `tune-train-config` instead — this skill is about continuing an experiment that already has a run directory.

## The one rule that matters: resume by DIRECTORY, not by config file

`trainDist.py` / `train.py` take a positional path that can be EITHER a repo config file OR an existing experiment directory. They behave completely differently:

- **Config file** (`configs/foo.local.yaml`) → `Configuration.createOrLoad` re-resolves the `id:` template (e.g. `id: midi/{date}-{filename}`) to a **NEW dated directory** → a FRESH experiment from scratch (random init, epoch 0, warmup LR). It does NOT continue the old run.
- **Experiment directory** (`/data/training/midi/20260702-…`) → loads that run's frozen `.state.yaml` (with `latest: true`, `steps`, `best`) → `loadCheckpoint` → resumes at the next epoch.

So to RESUME:
```
CUDA_VISIBLE_DEVICES=5,4 DATA_DIR=/data/lilylet/patches PYTHONPATH=. \
  python3 trainDist.py /data/training/midi/20260702-midi-measurewise-trans-selfattn-nota0701
```
Symptom you resumed WRONG (used the config file): log shows `[Epoch 0]`, fresh warmup LR (e.g. 3.16e-5 at step ~1), no `Checkpoint loaded` line, and a NEW `<date>-…` dir appears. A correct resume logs `Checkpoint loaded: …/latest.chkpt` then `[Epoch N+1]`.

## Config edits live in the DIR's .state.yaml, not the repo config

The repo config is read ONLY at first creation. On resume, the trainer reads the experiment directory's `.state.yaml` (a frozen snapshot). So any field you want to change on an existing run — `grad_clip`, `lr_mul`, `epoch_size`, `steps` — must be edited in `<expdir>/.state.yaml`, NOT the repo `.yaml`.

- Back up first: `cp .state.yaml .state.yaml.bak-<why>`.
- 2-space indent under the `trainer:` block; YAML key order doesn't matter for parsing.
- Verify it parses and resolves as expected before launching:
  ```
  python3 -c "from starry.utils.config import Configuration; \
    c=Configuration.createOrLoad('<expdir>'); print(c['trainer'].get('grad_clip'), c['trainer'].get('steps'), c['trainer.latest'])"
  ```

## Which checkpoint gets loaded (and how to resume a HEALTHY earlier state)

`trainer.latest: true` → the trainer loads `latest.chkpt` if it exists, else `best` (`config['best']`). Key facts:
- `loadCheckpoint` sets `start_epoch = checkpoint['epoch'] + 1` and restores model + AdamW optimizer moments.
- A DIVERGED run keeps overwriting `latest.chkpt` with its (bad) latest state, while `best.chkpt` holds the best-monitor state. So after a divergence, resuming from `latest.chkpt` continues the BAD state.
- To resume a healthy earlier state: back up the diverged latest, copy best over it:
  ```
  mv latest.chkpt latest.chkpt.diverged-bak
  cp best.chkpt latest.chkpt          # best.chkpt is a copy of model_<epoch>_loss_<val>.chkpt
  python3 -c "import torch; print(torch.load('latest.chkpt',map_location='cpu',weights_only=False)['epoch'])"  # confirm the epoch
  ```

## The `steps` / learning-rate gotcha (easy to miss)

`loadCheckpoint` restores the model and AdamW moments, but NOT the LR-scheduler step counter. The `ScheduledOptim` gets its `n_steps` from `config['steps']` at CONSTRUCTION (`ScheduledOptim.n_steps = init_step`; each `step()` does `n_steps += 1` then computes LR from `n_steps`). So if you restore an epoch-30 checkpoint but leave a diverged run's `steps: 56320` in `.state.yaml`, training continues at that run's decayed LR — often less than half the LR the checkpoint was actually trained at.

Fix: set `steps` to match the checkpoint's stage. For the InvSqrt schedule (post-warmup):
```
lr = lr_mul * d_model**-0.5 * n_steps**-0.5
```
The right step count for resuming AFTER epoch E (start_epoch = E+1) is:
```
steps = (E + 1) * epoch_size        # bs=1: steps == examples == epoch_size per epoch
```
e.g. resume after epoch 32 with epoch_size 320 → `steps = 33 * 320 = 10560` → LR ~3.04e-5 (vs a wrong 1.31e-5 at steps 56640). Sanity-check by comparing to the LR the original run logged at that epoch. Edit `steps` in `<expdir>/.state.yaml` alongside the checkpoint swap.

## Stopping a run cleanly: the `__SHUT` marker (do NOT pkill)

A concurrent train+val run (distributed `trainDist.py`, 2 ranks) must be stopped by a marker file, not by killing processes:
```
mkdir <expdir>/__SHUT
```
- Both ranks check for `__SHUT` at the TOP of each epoch loop and exit gracefully → it stops at the NEXT epoch/checkpoint boundary (not instantly; a checkpoint is saved first). Waiting one epoch is expected, not a failure.
- On the next resume, startup RENAMES `__SHUT` → `__SHUT-` so it doesn't immediately re-trigger.
- **Do NOT create a new `__SHUT` while a `__SHUT-` already exists** — the rename logic collides.
- **Do NOT `pkill -f <name>`** — on a k8s pod reached via `kubectl exec`, the pattern matches its OWN exec shell's command line → exit 137, kills the wrong thing (or your file-edit shell). SIGTERM is also unreliable: torch multiprocessing spawn workers respawn during teardown. The marker is the only clean way.

## End-to-end resume-from-healthy-checkpoint flow
1. Stop the current run: `mkdir <expdir>/__SHUT`, wait for it to exit at the next epoch boundary; confirm GPUs freed (`nvidia-smi`), only `__SHUT` (no leftover `__SHUT-` you created) present.
2. If resuming an EARLIER healthy state after a divergence: `cp best.chkpt latest.chkpt` (back up the diverged latest first); verify its `epoch`.
3. Edit `<expdir>/.state.yaml`: set `steps = (checkpoint_epoch+1) * epoch_size`, plus any new field (`grad_clip: 1.0`, etc.). Back up first.
4. Verify: `Configuration.createOrLoad('<expdir>')` shows the expected `steps`/`grad_clip`/`latest`.
5. Launch by DIRECTORY: `... python3 trainDist.py <expdir>` (NOT the config file).
6. Confirm the resume took: log shows `Checkpoint loaded`, `[Epoch N+1]`, and the first train-loss line's `lr:` matches your intended value (not fresh warmup, not the diverged decayed value).

## Notes
- `trainDist.py` = distributed (2 ranks, ≥2 GPUs, key `epochs`); `train.py` = single-process (key `epoch`). Both now honor `trainer.grad_clip` + skip non-finite loss.
- On the trainl pod, code updates arrive via a host `git pull` on `/root/work/deep-starry` (hostPath-mounted into the pod); gitignored `.local` configs go via `scp`. But remember: editing the repo config does nothing for an already-running experiment — edit the dir's `.state.yaml`.
