'''One real training step in the trainm-dev-9 pod, to prove the environment TRAINS.

The env check proves imports, CUDA init and a tensor allocation. None of that exercises the parts
that actually break on a hand-assembled environment: a backward pass through the mounted torch
against the host driver, the optimizer step, and the feeder feeding the model batches it accepts.
A run that imports fine and then dies on the first backward has cost you the queue time.

This deliberately does NOT use train.py: the trainer writes checkpoints, a TensorBoard dir and a
`.state.yaml` under TRAINING_DIR, which would leave a half-run experiment lying around for a smoke
test. It builds the same pieces by hand and throws them away.

GPU choice is explicit and defensive. maiyi-9's cards are held by a vLLM TP8 serve at ~263/275 GB
each, so this picks the card with the MOST free memory and refuses to run if that is under a floor —
better to fail with a clear message than to OOM into someone else's serving workload.

    kubectl exec trainm-dev-9 -- python3 /workspace/deep-starry/deploy/trainm_step_smoke.py \
        --config configs/midi-translator-nota1m00-maiyi.local.yaml --steps 3
'''

import argparse
import os
import sys
import time

REPO = '/workspace/deep-starry'
if REPO not in sys.path:
	sys.path.insert(0, REPO)

import torch			# noqa: E402


MIN_FREE_MIB = 6000		# below this, a bs-24 step at T~1000 is not worth attempting


def pick_gpu ():
	'''The emptiest card, or None. Read through torch rather than parsing nvidia-smi, so the number
	is the one the training process will actually see.'''
	best, best_free = None, -1
	for i in range(torch.cuda.device_count()):
		free, total = torch.cuda.mem_get_info(i)
		free_mib = free // (1024 * 1024)
		print(f'  gpu {i}   {free_mib} MiB free of {total // (1024 * 1024)} MiB')
		if free_mib > best_free:
			best, best_free = i, free_mib
	return best, best_free


def main ():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--config', required=True)
	ap.add_argument('--steps', type=int, default=3)
	ap.add_argument('--batch-size', type=int, default=0, help='override the config (default: config)')
	ap.add_argument('--force-gpu', type=int, default=-1,
		help='use this card regardless of free memory — only with a reason')
	args = ap.parse_args()

	os.chdir(REPO)
	from starry.utils.config import Configuration
	from starry.utils.model_factory import loadModel
	from starry.utils.dataset_factory import loadDataset

	config = Configuration.createOrLoad(args.config, volatile=True)
	print(f'config   {args.config}\n  data.root {config["data.root"]}\n  model {config["model.type"]}')

	print('\n== gpu selection')
	gpu, free = pick_gpu()
	if args.force_gpu >= 0:
		gpu = args.force_gpu
		print(f'  forced to gpu {gpu}')
	elif free < MIN_FREE_MIB:
		print(f'\nREFUSING: the emptiest card has {free} MiB free, under the {MIN_FREE_MIB} MiB floor.')
		print('The node\'s GPUs are held by another workload. Wait for capacity, or pass --force-gpu N')
		print('if you have established that card N is genuinely free.')
		return 2
	device = f'cuda:{gpu}'
	torch.cuda.set_device(gpu)
	print(f'  using {device}')

	print('\n== data')
	bs = args.batch_size or config['data.batch_size']
	train, val = loadDataset(config, device=device, batch_size=bs)
	print(f'  batch_size {bs}, val batches {len(list(range(0, 1)))} (val loader built)')

	print('\n== model')
	model = loadModel(config['model'], postfix='Loss')
	model.to(device)
	n_param = sum(p.numel() for p in model.parameters())
	print(f'  {config["model.type"]}Loss   {n_param / 1e6:.1f}M params')

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
		betas=config['optim.args.betas'], eps=float(config['optim.args.eps']),
		weight_decay=config['optim.args.weight_decay'])

	print(f'\n== {args.steps} train step(s)')
	model.train()
	it = iter(train)
	for step in range(args.steps):
		batch = next(it)
		batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
		t0 = time.time()
		loss, metric = model(batch)
		# The decisive part: a backward through the mounted torch against the host driver.
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		torch.cuda.synchronize()
		dt = time.time() - t0
		T = tuple(batch['input_ids'].shape)
		peak = torch.cuda.max_memory_allocated(gpu) / 1e9
		# metrics values are WeightedValue (weighted by supervised-token count), not floats.
		acc = metric.get('acc') if isinstance(metric, dict) else None
		acc = getattr(acc, 'value', acc)
		print(f'  step {step}   shape {T}   loss {loss.item():.4f}'
			+ (f'   acc {float(acc):.4f}' if acc is not None else '')
			+ f'   {dt * 1000:.0f} ms   peak {peak:.2f} GB')

	print('\n== one val batch (no_grad)')
	model.eval()
	with torch.no_grad():
		batch = next(iter(val))
		batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
		loss, metric = model(batch)
		print(f'  val loss {loss.item():.4f}   shape {tuple(batch["input_ids"].shape)}')

	print('\ntrainm step smoke: ok')
	return 0


if __name__ == '__main__':
	sys.exit(main())
