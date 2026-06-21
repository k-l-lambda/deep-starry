'''Generate samples from LilyNota for the top prompt groups and score them with the
paper's CLaMP-Score methodology (NotaGen 2502.18008v5, eq. 1; cf. NotaGen/RL/data.py).

CLaMP-Score of a generated piece x for prompt p:
    c_x = cos(z_x, z_bar_p)
where z_x is the piece's embedding and z_bar_p is the average embedding of prompt
p's ground-truth set. Here:
  - z_x        : the trained Lilylet student (LilyletM3Encoder) embedding of the
                 GENERATED Lilylet text (patchified at patch_size=64, style dropped —
                 exactly as preprocessLilyletM3.py prepares the training patches).
  - z_bar_p    : the per-prompt `lyl_avg` from notagen-100k.abc-vs-lyl.pt (the student's
                 average over that prompt's 100K ground-truth pieces) — SAME encoder /
                 same space as z_x (no cross-model bias), the user's chosen reference.

Two phases:
  --mode sweep  : on ONE group, generate a small batch at each of several temperatures
                  (wide range) and report mean CLaMP-Score per temperature → pick the best.
  --mode final  : with the chosen --temperature, generate --samples per top-K group and
                  report each group's CLaMP-Score distribution.

Reproducible: each (group, sample) seeds torch.manual_seed deterministically.

Run locally (RTX 3090) from repo root, in the venv:
  python3 -m tools.lilylet.scoreLilyNotaByPrompt --mode sweep ...
  python3 -m tools.lilylet.scoreLilyNotaByPrompt --mode final --temperature <best> ...
'''

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModel
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator
from starry.lilylet.data.patchifier import LilyletTokenizer, patchify_text


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# student LilyletM3Encoder args (lr0.06-wu1000 config)
STUDENT_ARGS = dict(num_classes=256, hidden_size=768, patch_size=64, patch_length=512,
	patch_num_layers=12, n_head=12, project=True, warm_start=False)
M3_PATCH_SIZE = 64


def _seed_text (period, composer, instrumentation):
	'''The corpus prompt: three leading `%<style>` lines (period/composer/instrumentation),
	matching the .lyl source layout. The generator continues header + body from this.'''
	return '%%%s\n%%%s\n%%%s' % (period, composer, instrumentation)


@torch.no_grad()
def _encode_student (deducer, tokenizer, text, device, min_patches=4, max_unknown_ratio=0.05):
	'''Encode generated Lilylet text into the student's [768] embedding, using the SAME
	patchify path as preprocessLilyletM3.py (patch_size=64, style comments dropped).

	Returns (z, info) where z is a [768] float32 CPU tensor (or None if unusable) and info
	is a dict with validity signals ORTHOGONAL to the CLaMP-Score: `n_patches` (body length)
	and `unknown_ratio` (garbled-token fraction). These gate out *degenerate* generations
	(empty / too-short / high-gibberish) without ever looking at the score itself — the
	paper rejects on syntax/alignment (check_alignment_unrotated), not on CLaMP-Score.'''
	patches, unknowns = patchify_text(text, tokenizer, file='<gen>',
		patch_size=M3_PATCH_SIZE, patch_length=2048, patch_stream=True,
		drop_style_comments=True)
	n_patches = 0 if patches is None else int(patches.shape[0])
	n_unknown = sum(hit['count'] for hit in unknowns) if unknowns else 0
	# total tokens ~= n_patches * patch_size (upper bound; pad inflates it, fine as denom)
	denom = max(1, n_patches * M3_PATCH_SIZE)
	unknown_ratio = n_unknown / denom
	info = dict(n_patches=n_patches, unknown_ratio=unknown_ratio, valid=True)

	if patches is None or n_patches < min_patches or unknown_ratio > max_unknown_ratio:
		info['valid'] = False
		return None, info

	p = patches.long().unsqueeze(0).to(device)              # [1, T, 64]
	m = torch.ones(1, p.shape[1], dtype=torch.long, device=device)
	z = deducer(p, m)[0].float().cpu()                      # [768]
	return z, info


def _gen_and_score (gen, deducer, tokenizer, ref_vec, seed_txt, n, base_seed,
	temperature, top_k, top_p, max_patches, measures, device, label='',
	max_attempts_factor=4, min_patches=4, max_unknown_ratio=0.05):
	'''Generate until n VALID samples are scored, re-rolling the seed on degenerate
	generations (validity gate is orthogonal to the score — see _encode_student). Caps
	total attempts at n*max_attempts_factor to avoid an infinite loop on a hopeless temp.
	Returns (scores, meta) where meta records attempts / rejects for transparency.'''
	scores = []
	attempts = 0
	rejects = 0
	max_attempts = n * max_attempts_factor
	seed = base_seed
	while len(scores) < n and attempts < max_attempts:
		torch.manual_seed(seed)
		seed += 1
		attempts += 1
		text = gen.generate(prompt_text=seed_txt, max_patches=max_patches,
			temperature=temperature, top_k=top_k, top_p=top_p, measures=measures)
		z, info = _encode_student(deducer, tokenizer, text, device,
			min_patches=min_patches, max_unknown_ratio=max_unknown_ratio)
		if z is None:
			rejects += 1
			continue
		scores.append(float(F.cosine_similarity(z, ref_vec, dim=0)))
	if len(scores) < n:
		logging.warning('  [%s] only %d/%d valid after %d attempts (%d rejected)',
			label, len(scores), n, attempts, rejects)
	meta = dict(attempts=attempts, rejects=rejects, reject_rate=(rejects / attempts if attempts else 0.0))
	return scores, meta


def _stats (scores):
	import statistics
	if not scores:
		return dict(n=0, mean=float('nan'), median=float('nan'), min=float('nan'), max=float('nan'), std=float('nan'))
	return dict(n=len(scores), mean=sum(scores) / len(scores), median=statistics.median(scores),
		min=min(scores), max=max(scores),
		std=(statistics.pstdev(scores) if len(scores) > 1 else 0.0))


def _topk_mean (scores, portion=0.1):
	'''Mean of the top-`portion` highest scores — the CLaMP-DPO *selection* score (the paper
	takes the top-10% as the chosen set). BIASED upward vs the raw mean (it reports the
	best-of-N tail), so always label it separately. At least 1 sample is kept.'''
	if not scores:
		return float('nan')
	k = max(1, int(round(len(scores) * portion)))
	return sum(sorted(scores, reverse=True)[:k]) / k


def _load_all (args):
	logging.info('Loading prompt-avg reference %s ...', args.ref)
	ref = torch.load(args.ref, map_location='cpu', weights_only=False)
	groups = {(p['period'], p['composer'], p['instrumentation']): p for p in ref['prompts']}

	logging.info('Loading LilyNota generator from %s ...', args.lilynota)
	config = Configuration.createOrLoad(args.lilynota_config or os.path.dirname(args.lilynota), volatile=True)
	tok_path = config['data.args.tokenizer_path']
	if not os.path.isabs(tok_path):
		repo = next(p for p in [os.getcwd()] if os.path.isdir(os.path.join(p, 'starry')))
		tok_path = os.path.join(repo, tok_path)
	gen = LilyletPatchyGenerator.from_config(config, args.lilynota, tokenizer_path=tok_path, device=args.device)

	logging.info('Loading student LilyletM3Encoder from %s ...', args.student)
	deducer = loadModel({'type': 'LilyletM3Encoder', 'args': STUDENT_ARGS},
		imports=['starry.lilylet.models.m3distill'])
	ck = torch.load(args.student, map_location='cpu', weights_only=False)
	deducer.load_state_dict(ck['model'], strict=True)
	deducer.to(args.device).eval()

	tokenizer = LilyletTokenizer(tok_path)
	logging.info('All loaded (student epoch %s) on %s', ck.get('epoch'), args.device)
	return groups, gen, deducer, tokenizer


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ref', default='/home/camus/data/lilylet/m3/notagen-100k.abc-vs-lyl.pt',
		help='prompt-avg artifact with per-group lyl_avg (and abc_avg) reference vectors')
	parser.add_argument('--lilynota', default='/home/camus/data/models/LilyNota/best.chkpt')
	parser.add_argument('--lilynota-config', default=None, help='config/dir for model.args (default: checkpoint dir .state.yaml)')
	parser.add_argument('--student', default='/home/camus/data/models/deep-starry-logs/lilylet/20260620-lilylet-m3-distill-0619-lr0.06-wu1000/best.chkpt')
	parser.add_argument('--ref-side', choices=['lyl', 'abc'], default='lyl', help='which group-average to score against (lyl_avg = same student space, recommended)')
	parser.add_argument('--mode', choices=['sweep', 'final'], required=True)
	parser.add_argument('--top', type=int, default=5, help='final: number of top groups (by count) to score')
	parser.add_argument('--samples', type=int, default=50, help='final: samples per group')
	parser.add_argument('--sweep-samples', type=int, default=12, help='sweep: samples per temperature')
	parser.add_argument('--temps', default='0.5,0.7,0.9,1.1,1.3,1.5', help='sweep: comma temperatures')
	parser.add_argument('--temperature', type=float, default=0.9, help='final: chosen temperature')
	parser.add_argument('--top-k', type=int, default=20)
	parser.add_argument('--top-p', type=float, default=0.95)
	parser.add_argument('--max-patches', type=int, default=1024)
	parser.add_argument('--measures', type=int, default=None, help='force a measure count (default: model decides)')
	parser.add_argument('--min-patches', type=int, default=4, help='validity gate: reject generations with fewer body patches (degenerate/empty)')
	parser.add_argument('--max-unknown-ratio', type=float, default=0.05, help='validity gate: reject generations whose unknown-token fraction exceeds this (gibberish)')
	parser.add_argument('--seed-base', type=int, default=1000)
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--out', default=None)
	args = parser.parse_args()

	groups, gen, deducer, tokenizer = _load_all(args)

	ref_key = 'lyl_avg' if args.ref_side == 'lyl' else 'abc_avg'
	# order groups by count desc
	ordered = sorted(groups.values(), key=lambda p: p['count'], reverse=True)

	if args.mode == 'sweep':
		g = ordered[0]
		seed_txt = _seed_text(g['period'], g['composer'], g['instrumentation'])
		ref_vec = g[ref_key].float()
		temps = [float(t) for t in args.temps.split(',')]
		logging.info('=== TEMPERATURE SWEEP on group: %s | %s | %s (n_gt=%d) ===',
			g['period'], g['composer'], g['instrumentation'], g['count'])
		logging.info('ref=%s, %d samples/temp, top_k=%d top_p=%.2f', ref_key, args.sweep_samples, args.top_k, args.top_p)
		results = []
		for t in temps:
			scores, gmeta = _gen_and_score(gen, deducer, tokenizer, ref_vec, seed_txt,
				args.sweep_samples, args.seed_base, t, args.top_k, args.top_p,
				args.max_patches, args.measures, args.device, label='T=%.2f' % t,
				min_patches=args.min_patches, max_unknown_ratio=args.max_unknown_ratio)
			st = _stats(scores)
			top10 = _topk_mean(scores, 0.1)
			results.append(dict(temperature=t, **st, top10pct=top10, gen_meta=gmeta, scores=scores))
			logging.info('  T=%.2f  CLaMP mean=%.4f median=%.4f std=%.4f [min=%.4f max=%.4f] top10%%=%.4f  (valid n=%d, reject %.0f%%)',
				t, st['mean'], st['median'], st['std'], st['min'], st['max'], top10, st['n'], 100 * gmeta['reject_rate'])
		best = max((r for r in results if r['n'] > 0), key=lambda r: r['mean'])
		logging.info('BEST temperature by mean CLaMP-Score: %.2f (mean=%.4f)', best['temperature'], best['mean'])
		out = dict(format='lilynota-clamp-score-sweep', group=dict(period=g['period'], composer=g['composer'], instrumentation=g['instrumentation'], count=g['count']),
			ref_side=args.ref_side, top_k=args.top_k, top_p=args.top_p, sweep_samples=args.sweep_samples,
			gating=dict(min_patches=args.min_patches, max_unknown_ratio=args.max_unknown_ratio),
			results=results, best_temperature=best['temperature'])
	else:
		top = ordered[:args.top]
		logging.info('=== FINAL scoring: top %d groups, %d samples each, T=%.2f, ref=%s ===',
			len(top), args.samples, args.temperature, ref_key)
		group_results = []
		for gi, g in enumerate(top):
			seed_txt = _seed_text(g['period'], g['composer'], g['instrumentation'])
			ref_vec = g[ref_key].float()
			scores, gmeta = _gen_and_score(gen, deducer, tokenizer, ref_vec, seed_txt,
				args.samples, args.seed_base + gi * 10000, args.temperature, args.top_k, args.top_p,
				args.max_patches, args.measures, args.device,
				label='%s/%s' % (g['composer'][:12], g['instrumentation'][:8]),
				min_patches=args.min_patches, max_unknown_ratio=args.max_unknown_ratio)
			st = _stats(scores)
			top10 = _topk_mean(scores, 0.1)
			group_results.append(dict(period=g['period'], composer=g['composer'], instrumentation=g['instrumentation'],
				count=g['count'], **st, top10pct=top10, gen_meta=gmeta, scores=scores))
			logging.info('  [%d] %s | %s | %s  CLaMP mean=%.4f median=%.4f std=%.4f [min=%.4f max=%.4f] top10%%=%.4f (valid n=%d/%d, reject %.0f%%)',
				gi, g['period'], g['composer'], g['instrumentation'], st['mean'], st['median'], st['std'], st['min'], st['max'], top10, st['n'], args.samples, 100 * gmeta['reject_rate'])
		alls = [c for gr in group_results for c in gr['scores']]
		overall = _stats(alls)
		logging.info('OVERALL CLaMP-Score (pooled %d valid): mean=%.4f median=%.4f std=%.4f | top10%%=%.4f',
			overall['n'], overall['mean'], overall['median'], overall['std'], _topk_mean(alls, 0.1))
		out = dict(format='lilynota-clamp-score-final', temperature=args.temperature, ref_side=args.ref_side,
			top_k=args.top_k, top_p=args.top_p, samples=args.samples,
			gating=dict(min_patches=args.min_patches, max_unknown_ratio=args.max_unknown_ratio),
			groups=group_results, overall=overall)

	out_path = args.out or ('/home/camus/data/lilylet/m3/lilynota-clamp-score-%s.pt' % args.mode)
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	torch.save(out, out_path)
	logging.info('Wrote %s', out_path)


if __name__ == '__main__':
	main()
