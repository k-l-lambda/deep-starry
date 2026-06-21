'''Group the 100K M3 distillation artifact by NotaGen prompt triple and average
the M3 embeddings per group — the per-prompt reference vector \\bar{z}_p of the
CLaMP-DPO "CLaMP 2 Score" (NotaGen paper, eq. 1; cf. NotaGen/RL/data.py
average_npy over a prompt's ground-truth feature set).

NotaGen's prompt is the three leading `%<...>` comment lines of a sheet:
    %<period>
    %<composer>
    %<instrumentation>
(see NotaGen/gradio/inference.py:190-193). preprocessLilyletM3.py dropped these
(`drop_style_comments=True`), so they are NOT in the artifact — we recover them
by reading each item's source file (resolved from its stored relative `path`).

For each prompt triple p we collect every item whose source carries that triple
and average their stored `m3_embedding` ([768] float16) into \\bar{z}_p (float32):

    z_bar_p = mean_{x in X_p} z_x          (arithmetic mean over real features)

Output (.pt): version 1 artifact with
    prompts: list of dicts {period, composer, instrumentation, count, avg_embedding[768] f32}
plus stats. Optionally also dumps a CSV summary (prompt, count) for inspection.

Run on the .51 host (artifact + source trees are local there), from repo root:
  python3 -m tools.lilylet.avgM3ByPrompt \\
      /data1/datasets/nota/lilylet/m3/20260619/notagen-100k.pt \\
      --lyl-dir /data1/datasets/nota/lilylet/lyl \\
      --abc-dir /data1/datasets/nota/NotaGenX-opus/abc \\
      --out /data1/datasets/nota/lilylet/m3/20260619/notagen-100k.prompt-avg.pt
'''

import argparse
import csv
import logging
import os
import sys

import torch
from tqdm import tqdm


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _read_prompt_triple (src_path):
	'''Read the three leading `%<style>` comment lines (period, composer,
	instrumentation) from a NotaGen source file (.abc or .lyl). The triple sits at
	the very top; an ABC file may have an `X:` line before it (skipped). Returns
	(period, composer, instrumentation) or None if fewer than 3 style lines found.'''
	styles = []
	with open(src_path, 'r', encoding='utf-8', errors='replace') as f:
		for line in f:
			s = line.strip()
			if not s:
				continue
			if s.startswith('%%'):       # `%%score` etc. ends the style block
				break
			if s.startswith('%'):
				styles.append(s[1:].strip())
				if len(styles) == 3:
					break
				continue
			if s.startswith('X:'):       # ABC index line precedes the style block
				continue
			break                        # any other content => style block ended
	if len(styles) < 3:
		return None
	return tuple(styles[:3])


def _resolve_source (rel, lyl_dir, abc_dir):
	'''Resolve an item's stored relative `path` (e.g. `00/00/<hash>.lyl`) to a
	source file from which the prompt triple can be read. Prefer the ABC tree
	(authoritative for NotaGen prompts); fall back to the .lyl itself.'''
	stem = os.path.splitext(rel)[0]
	if abc_dir:
		abc = os.path.join(abc_dir, stem + '.abc')
		if os.path.exists(abc):
			return abc
	lyl = os.path.join(lyl_dir, rel)
	if os.path.exists(lyl):
		return lyl
	return None


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('artifact', type=str, help='the merged 100K M3 artifact (.pt) from mergeLilyletM3.py')
	parser.add_argument('--lyl-dir', type=str, default='/data1/datasets/nota/lilylet/lyl', help='root of the .lyl source tree (for path resolution / fallback prompt read)')
	parser.add_argument('--abc-dir', type=str, default='/data1/datasets/nota/NotaGenX-opus/abc', help='root of the mirrored .abc source tree (authoritative prompt source)')
	parser.add_argument('--out', type=str, default=None, help='output .pt path (default: <artifact>.prompt-avg.pt)')
	parser.add_argument('--csv', type=str, default=None, help='optional CSV summary path (period,composer,instrumentation,count)')
	args = parser.parse_args()

	out_path = args.out or (os.path.splitext(args.artifact)[0] + '.prompt-avg.pt')

	logging.info('Loading artifact %s ...', args.artifact)
	artifact = torch.load(args.artifact, map_location='cpu', weights_only=False)
	items = artifact['items']
	logging.info('%d items', len(items))

	# accumulate sum + count per prompt triple (float32 running sum for accuracy)
	sums = {}
	counts = {}
	missing_src = 0
	missing_prompt = 0
	for it in tqdm(items):
		src = _resolve_source(it['path'], args.lyl_dir, args.abc_dir)
		if src is None:
			missing_src += 1
			continue
		triple = _read_prompt_triple(src)
		if triple is None:
			missing_prompt += 1
			continue
		emb = it['m3_embedding'].to(torch.float32)
		if triple in sums:
			sums[triple] += emb
			counts[triple] += 1
		else:
			sums[triple] = emb.clone()
			counts[triple] = 1

	prompts = []
	for triple in sorted(counts, key=lambda t: counts[t], reverse=True):
		n = counts[triple]
		avg = sums[triple] / n
		prompts.append(dict(
			period=triple[0], composer=triple[1], instrumentation=triple[2],
			count=n, avg_embedding=avg,
		))

	out = dict(
		version=1,
		format='notagen-m3-prompt-avg',
		source_artifact=os.path.abspath(args.artifact),
		prompts=prompts,
		stats=dict(
			items=len(items),
			grouped=sum(counts.values()),
			prompt_groups=len(counts),
			missing_src=missing_src,
			missing_prompt=missing_prompt,
		),
	)
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	tmp = out_path + '.tmp'
	torch.save(out, tmp)
	os.replace(tmp, out_path)

	logging.info('Wrote %s', out_path)
	logging.info('Prompt groups: %d  (grouped %d / %d items; missing src %d, missing prompt %d)',
		len(counts), sum(counts.values()), len(items), missing_src, missing_prompt)
	logging.info('Top 10 groups by count:')
	for p in prompts[:10]:
		logging.info('  %5d  %s | %s | %s', p['count'], p['period'], p['composer'], p['instrumentation'])

	if args.csv:
		with open(args.csv, 'w', newline='', encoding='utf-8') as f:
			w = csv.writer(f)
			w.writerow(['period', 'composer', 'instrumentation', 'count'])
			for p in prompts:
				w.writerow([p['period'], p['composer'], p['instrumentation'], p['count']])
		logging.info('Wrote CSV summary %s', args.csv)


if __name__ == '__main__':
	main()
