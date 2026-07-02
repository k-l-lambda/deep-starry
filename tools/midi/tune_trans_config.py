'''Tuning probe for midi-measurewise-trans (MidiBgptTrans): measure fwd+bwd+optim peak memory
and step time at bs=1 across representative sequence lengths T (the corpus is uncropped and
heavy-tailed), to pick the safe T cap / filter and the epoch_size balance.

Usage (on the pod, PYTHONPATH=repo root):
  CUDA_VISIBLE_DEVICES=7 python3 tools/midi/tune_trans_config.py \
    --config configs/midi-measurewise-trans-nota20260701.yaml \
    --data-dir /data/lilylet/patches/
'''
import argparse, os, sys, time
import numpy as np
import torch, yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from starry.midi.data.condPatchy import _get_store, CondMidiPatchy
from starry.midi.models.midiBgptTrans import MidiBgptTransLoss


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--data-dir', default=os.environ.get('DATA_DIR', '.'))
	ap.add_argument('--targets', default='6000,10000,15000,20000,28000,40000',
		help='comma T values to probe (nearest real sample chosen for each)')
	ap.add_argument('--warmup', type=int, default=2)
	ap.add_argument('--iters', type=int, default=5)
	args = ap.parse_args()

	cfg = yaml.safe_load(open(args.config))
	margs = cfg['model']['args']
	root = os.path.join(args.data_dir, cfg['data']['root'])
	store = _get_store(root)
	N = len(store)
	Ts = np.array([store.get(i)['patches'].shape[0] for i in range(N)])

	# pick, for each target T, the sample whose length is closest to it.
	targets = [int(x) for x in args.targets.split(',')]
	picks = []
	for t in targets:
		i = int(np.argmin(np.abs(Ts - t)))
		picks.append((i, int(Ts[i])))

	device = 'cuda'
	model = MidiBgptTransLoss(**margs).to(device)
	model.train()
	# optimizer over trainable (frozen encoder excluded), like the real run.
	params = [p for p in model.deducer.parameters() if p.requires_grad]
	opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

	# one dataset instance to reuse its collate on single-sample batches.
	ds = CondMidiPatchy(root, '0/1', device=device, **cfg['data']['args'])

	def make_batch (idx):
		b = ds.collateBatch([ds[idx]])
		return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

	print(f'corpus N={N}  T p50={int(np.percentile(Ts,50))} p90={int(np.percentile(Ts,90))} '
		f'p95={int(np.percentile(Ts,95))} p99={int(np.percentile(Ts,99))} max={int(Ts.max())}')
	print(f'{"T":>8} {"status":>8} {"peak_GB":>9} {"%143":>6} {"step_ms":>9} {"fwd_ms":>8}')
	for idx, T in picks:
		torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
		try:
			batch = make_batch(idx)
			# warmup
			for _ in range(args.warmup):
				opt.zero_grad(set_to_none=True)
				with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
					loss, _ = model(batch)
				loss.backward(); opt.step()
			torch.cuda.synchronize()
			step_t = []; fwd_t = []
			for _ in range(args.iters):
				opt.zero_grad(set_to_none=True)
				t0 = time.time()
				with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
					loss, _ = model(batch)
				torch.cuda.synchronize(); t1 = time.time()
				loss.backward(); opt.step()
				torch.cuda.synchronize(); t2 = time.time()
				fwd_t.append((t1 - t0) * 1e3); step_t.append((t2 - t0) * 1e3)
			peak = torch.cuda.max_memory_allocated() / 1e9
			print(f'{T:>8} {"OK":>8} {peak:>9.1f} {peak/143.77*100:>5.0f}% '
				f'{np.median(step_t):>9.1f} {np.median(fwd_t):>8.1f}')
		except RuntimeError as e:
			msg = 'OOM' if 'out of memory' in str(e).lower() else 'ERR'
			print(f'{T:>8} {msg:>8}  {str(e)[:60]}')
			torch.cuda.empty_cache()


if __name__ == '__main__':
	main()
