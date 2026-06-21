'''Per-prompt ABC (teacher) vs Lilylet (student) embedding comparison.

For the 100K M3 artifact, compute two per-prompt-triple average embeddings and
their similarity:

  ABC side  z_bar^abc_p : average of the stored `m3_embedding` (the frozen CLaMP 3
            / sander-wood/clamp3 ABC M3 teacher output, computed offline by
            preprocessLilyletM3.py) over the items of prompt p.
  Lilylet   z_bar^lyl_p : average of the TRAINED student LilyletM3Encoder's
            embedding of each item's stored Lilylet `patches`, over prompt p.

The prompt triple (period, composer, instrumentation) is the three leading
`%<...>` lines of the source sheet (NotaGen/gradio/inference.py:190-193); it was
dropped from the artifact (drop_style_comments), so it is recovered by reading
each item's source file (resolved from its relative `path`).

Per prompt p we report cos(z_bar^abc_p, z_bar^lyl_p) — how well the student's
group centroid aligns with the teacher's in each prompt's region of the space.

Run on the .51 host (artifact + source trees + GPU local there), repo root:
  python3 -m tools.lilylet.compareAbcLylByPrompt \\
      /data1/datasets/nota/lilylet/m3/20260619/notagen-100k.pt \\
      --student /data1/datasets/nota/lilylet/m3/20260619/student-lr0.06-wu1000-best.chkpt \\
      --abc-dir /data1/datasets/nota/NotaGenX-opus/abc \\
      --lyl-dir /data1/datasets/nota/lilylet/lyl \\
      --device cuda:2 --batch-size 32 \\
      --out /data1/datasets/nota/lilylet/m3/20260619/notagen-100k.abc-vs-lyl.pt \\
      --csv /data1/datasets/nota/lilylet/m3/20260619/notagen-100k.abc-vs-lyl.csv
'''

import argparse
import csv
import logging
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from starry.utils.model_factory import loadModel


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# student LilyletM3Encoder args (lr0.06-wu1000 config; mse_weight harmlessly absent here)
MODEL_ARGS = dict(num_classes=256, hidden_size=768, patch_size=64, patch_length=512,
	patch_num_layers=12, n_head=12, project=True, warm_start=False)


def _read_prompt_triple (src_path):
	'''Three leading `%<style>` lines (period, composer, instrumentation). See
	tools/lilylet/avgM3ByPrompt.py for the format rationale.'''
	styles = []
	with open(src_path, 'r', encoding='utf-8', errors='replace') as f:
		for line in f:
			s = line.strip()
			if not s:
				continue
			if s.startswith('%%'):
				break
			if s.startswith('%'):
				styles.append(s[1:].strip())
				if len(styles) == 3:
					break
				continue
			if s.startswith('X:'):
				continue
			break
	return tuple(styles[:3]) if len(styles) >= 3 else None


def _resolve_source (rel, lyl_dir, abc_dir):
	stem = os.path.splitext(rel)[0]
	if abc_dir:
		abc = os.path.join(abc_dir, stem + '.abc')
		if os.path.exists(abc):
			return abc
	lyl = os.path.join(lyl_dir, rel)
	return lyl if os.path.exists(lyl) else None


@torch.no_grad()
def _encode_student_batch (deducer, patch_list, device):
	'''Encode a list of variable-T patch tensors [T, 64] with the student encoder.
	Pads to the max T in the batch (mask=0 for pad), returns [B, 768] on CPU.'''
	maxT = max(p.shape[0] for p in patch_list)
	B = len(patch_list)
	ps = patch_list[0].shape[1]
	patches = torch.zeros(B, maxT, ps, dtype=torch.long)
	masks = torch.zeros(B, maxT, dtype=torch.long)
	for i, p in enumerate(patch_list):
		T = p.shape[0]
		patches[i, :T] = p.long()
		masks[i, :T] = 1
	emb = deducer(patches.to(device), masks.to(device))   # [B, 768]
	return emb.float().cpu()


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('artifact', type=str, help='merged 100K M3 artifact (.pt)')
	parser.add_argument('--student', type=str, required=True, help='trained LilyletM3Encoder checkpoint (best.chkpt)')
	parser.add_argument('--abc-dir', type=str, default='/data1/datasets/nota/NotaGenX-opus/abc')
	parser.add_argument('--lyl-dir', type=str, default='/data1/datasets/nota/lilylet/lyl')
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--batch-size', type=int, default=32)
	parser.add_argument('--out', type=str, default=None)
	parser.add_argument('--csv', type=str, default=None)
	args = parser.parse_args()

	out_path = args.out or (os.path.splitext(args.artifact)[0] + '.abc-vs-lyl.pt')

	logging.info('Loading artifact %s ...', args.artifact)
	artifact = torch.load(args.artifact, map_location='cpu', weights_only=False)
	items = artifact['items']
	logging.info('%d items', len(items))

	logging.info('Loading student LilyletM3Encoder from %s ...', args.student)
	deducer = loadModel({'type': 'LilyletM3Encoder', 'args': MODEL_ARGS},
		imports=['starry.lilylet.models.m3distill'])
	ck = torch.load(args.student, map_location='cpu', weights_only=False)
	deducer.load_state_dict(ck['model'], strict=True)
	deducer.to(args.device).eval()
	logging.info('Student loaded (epoch %s) on %s', ck.get('epoch'), args.device)

	# accumulate per-prompt running sums for both sides
	abc_sum = {}      # teacher (stored m3_embedding)
	lyl_sum = {}      # student (encoded patches)
	counts = {}
	missing_src = missing_prompt = 0

	# resolve prompt triples first (so we can also skip unresolved items consistently)
	triples = []
	for it in tqdm(items, desc='resolve prompts'):
		src = _resolve_source(it['path'], args.lyl_dir, args.abc_dir)
		if src is None:
			triples.append(None); missing_src += 1; continue
		t = _read_prompt_triple(src)
		if t is None:
			triples.append(None); missing_prompt += 1; continue
		triples.append(t)

	# encode student in batches, accumulate both sides
	buf_idx = []
	def flush (buf):
		if not buf:
			return
		embs = _encode_student_batch(deducer, [items[i]['patches'] for i in buf], args.device)
		for k, i in enumerate(buf):
			t = triples[i]
			abc = items[i]['m3_embedding'].float()
			lyl = embs[k]
			if t in counts:
				abc_sum[t] += abc; lyl_sum[t] += lyl; counts[t] += 1
			else:
				abc_sum[t] = abc.clone(); lyl_sum[t] = lyl.clone(); counts[t] = 1

	for i in tqdm(range(len(items)), desc='encode student'):
		if triples[i] is None:
			continue
		buf_idx.append(i)
		if len(buf_idx) >= args.batch_size:
			flush(buf_idx); buf_idx = []
	flush(buf_idx)

	# per-prompt centroids + similarity
	prompts = []
	for t in sorted(counts, key=lambda k: counts[k], reverse=True):
		n = counts[t]
		abc_avg = abc_sum[t] / n
		lyl_avg = lyl_sum[t] / n
		sim = float(F.cosine_similarity(abc_avg, lyl_avg, dim=0))
		prompts.append(dict(
			period=t[0], composer=t[1], instrumentation=t[2], count=n,
			abc_avg=abc_avg, lyl_avg=lyl_avg, cos=sim,
		))

	# global stats: unweighted (per group) and count-weighted (per piece)
	import statistics
	cos_list = [p['cos'] for p in prompts]
	n_list = [p['count'] for p in prompts]
	mean_unw = sum(cos_list) / len(cos_list)
	mean_w = sum(c * n for c, n in zip(cos_list, n_list)) / sum(n_list)
	median = statistics.median(cos_list)

	out = dict(
		version=1,
		format='notagen-m3-abc-vs-lyl-prompt',
		source_artifact=os.path.abspath(args.artifact),
		student_checkpoint=os.path.abspath(args.student),
		student_epoch=ck.get('epoch'),
		prompts=prompts,
		stats=dict(
			items=len(items), grouped=sum(counts.values()), prompt_groups=len(counts),
			missing_src=missing_src, missing_prompt=missing_prompt,
			cos_mean_unweighted=mean_unw, cos_mean_weighted=mean_w, cos_median=median,
			cos_min=min(cos_list), cos_max=max(cos_list),
		),
	)
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	tmp = out_path + '.tmp'
	torch.save(out, tmp)
	os.replace(tmp, out_path)

	logging.info('Wrote %s', out_path)
	logging.info('Prompt groups: %d  (grouped %d / %d; missing src %d, prompt %d)',
		len(counts), sum(counts.values()), len(items), missing_src, missing_prompt)
	logging.info('Per-group ABC<->lyl centroid cosine:')
	logging.info('  mean (unweighted over %d groups): %.4f', len(prompts), mean_unw)
	logging.info('  mean (weighted by piece count)  : %.4f', mean_w)
	logging.info('  median / min / max              : %.4f / %.4f / %.4f', median, min(cos_list), max(cos_list))
	logging.info('Top 10 groups by count:')
	for p in prompts[:10]:
		logging.info('  n=%5d  cos=%.4f  %s | %s | %s', p['count'], p['cos'], p['period'], p['composer'], p['instrumentation'])
	logging.info('Lowest-cos 5 groups (count>=20):')
	low = sorted([p for p in prompts if p['count'] >= 20], key=lambda p: p['cos'])[:5]
	for p in low:
		logging.info('  n=%5d  cos=%.4f  %s | %s | %s', p['count'], p['cos'], p['period'], p['composer'], p['instrumentation'])

	if args.csv:
		with open(args.csv, 'w', newline='', encoding='utf-8') as f:
			w = csv.writer(f)
			w.writerow(['period', 'composer', 'instrumentation', 'count', 'abc_lyl_cos'])
			for p in prompts:
				w.writerow([p['period'], p['composer'], p['instrumentation'], p['count'], '%.6f' % p['cos']])
		logging.info('Wrote CSV %s', args.csv)


if __name__ == '__main__':
	main()
