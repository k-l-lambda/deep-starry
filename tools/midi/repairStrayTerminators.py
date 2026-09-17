#!/usr/bin/env python3
'''Repair published midiseq2 pairs that carry a non-final `end_of_track`.

`end_of_track` is the corpus's own terminator and every corpus file holds exactly one, as its last
line. A published file with one in the MIDDLE has the corpus asserting the opposite of its own
invariant -- it teaches the next model that the terminator does not terminate.

The cause is fixed in translateMidiseq2.py (the run now stops at the terminator it emits instead of
8 tokens later, and strips a stray one it could not stop on). This repairs what was published
before that, and splits the work by whether truncating changes the bar count -- because the bar
number is the only thing tying the two arms together:

  GROUP A   no `@measure` directive follows the first terminator, so cutting there removes no bar.
            The regular arm is truncated in place and the rubato arm is not touched, because it
            has nothing to correspond to what was removed. MEASURED at 401 of 485 affected files,
            all with the bar count unchanged and still agreeing with rubato.

  GROUP B   at least one `@measure` follows, so truncating would drop bars from one arm only and
            leave the pair misnumbered. These cannot be repaired textually; the ids are written
            out for a re-run through the fixed tool, which will regenerate both arms together.
            MEASURED at 84 of 485.

Reads nothing but the published text, so it is safe to run against a live translation: a file
being written right now is either absent or complete (the driver publishes as its last step), and
a repair that races one is a no-op the next run picks up.

Run:
  python3 tools/midi/repairStrayTerminators.py --root <corpus>                 # report only
  python3 tools/midi/repairStrayTerminators.py --root <corpus> --apply         # repair group A
  python3 tools/midi/repairStrayTerminators.py --root <corpus> --apply --backup-dir DIR
'''

import argparse
import os
import shutil
import sys


def read_body (path):
	'''Non-empty stripped lines, which is what every count here is defined over.'''
	with open(path, encoding='utf-8') as f:
		return [l.strip() for l in f if l.strip()]


def bar_count (body):
	'''Bars the tool would report: `@measure` directives less a bare trailing terminator.

	`close_final_measure` ends a truncated run on an `@measure N` that opens nothing -- it closes
	the last bar -- and excludes it from its own count. Counting it here would make every
	comparison against the other arm off by one on exactly the files that were trimmed.
	'''
	n = sum(1 for l in body if l.startswith('@measure'))
	if body and body[-1].startswith('@measure'):
		n -= 1
	return n


def classify (body):
	'''-> (cut_index, removed_measures) for a body with a stray terminator, else (None, 0).

	`cut_index` is exclusive: body[:cut_index] ends ON the first terminator.
	'''
	idx = [i for i, l in enumerate(body) if l == 'end_of_track']
	if not idx or idx[0] == len(body) - 1:
		return None, 0
	cut = idx[0] + 1
	return cut, sum(1 for l in body[cut:] if l.startswith('@measure'))


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--root', required=True,
		help='corpus root holding regular/ and rubato/')
	ap.add_argument('--apply', action='store_true',
		help='write the group A repairs; without it nothing is modified')
	ap.add_argument('--backup-dir', default=None,
		help='copy each file here before truncating it (default: <root>/.work/eot-backup)')
	ap.add_argument('--redo-list', default=None,
		help='where to write the group B ids (default: <root>/.work/eot-redo.txt)')
	ap.add_argument('--min-measures', type=int, default=4,
		help='gate the repaired result must still pass, matching the driver (default 4)')
	args = ap.parse_args()

	reg = os.path.join(args.root, 'regular')
	rub = os.path.join(args.root, 'rubato')
	if not os.path.isdir(reg):
		print(f'no regular/ under {args.root}', file=sys.stderr)
		return 2
	backup = args.backup_dir or os.path.join(args.root, '.work', 'eot-backup')
	redo_path = args.redo_list or os.path.join(args.root, '.work', 'eot-redo.txt')

	group_a, group_b, refused = [], [], []
	scanned = 0
	for fn in sorted(os.listdir(reg)):
		if not fn.endswith('.midiseq2.txt'):
			continue
		scanned += 1
		path = os.path.join(reg, fn)
		try:
			body = read_body(path)
		except OSError:
			continue			# being written right now; the next run sees it
		cut, removed = classify(body)
		if cut is None:
			continue
		fid = fn.split('.')[0]
		if removed:
			group_b.append((fid, removed, len(body) - cut))
			continue
		kept = body[:cut]
		bars_before, bars_after = bar_count(body), bar_count(kept)
		rub_path = os.path.join(rub, fn)
		rub_bars = bar_count(read_body(rub_path)) if os.path.exists(rub_path) else None
		# Every one of these must hold, or the pair stops being readable as a pair. A file that
		# fails one is REFUSED rather than repaired -- it goes on the re-run list instead.
		if (bars_after != bars_before or bars_after < args.min_measures
				or (rub_bars is not None and rub_bars != bars_after)):
			refused.append((fid, bars_before, bars_after, rub_bars))
			group_b.append((fid, removed, len(body) - cut))
			continue
		group_a.append((fid, path, kept, len(body) - cut, bars_after))

	print(f'scanned {scanned} published regular/ files under {args.root}')
	print(f'  group A (truncate regular arm in place):      {len(group_a)}')
	print(f'  group B (needs a re-run through both arms):   {len(group_b)}')
	if refused:
		print(f'    of which refused by an invariant check:    {len(refused)}')
		for fid, b, a, r in refused[:8]:
			print(f'      {fid}: bars {b} -> {a}, rubato {r}')
	if group_a:
		total = sum(g[3] for g in group_a)
		print(f'  lines to remove from group A: {total} '
			f'(median {sorted(g[3] for g in group_a)[len(group_a) // 2]} per file)')

	if group_b:
		os.makedirs(os.path.dirname(os.path.abspath(redo_path)), exist_ok=True)
		with open(redo_path, 'w', encoding='utf-8') as f:
			f.write('# ids whose stray end_of_track cannot be removed textually: truncating drops\n'
				'# bars from the regular arm only, so both arms must be regenerated together.\n'
				'#   ./translate_piano0909.sh --workers N --gpus ... --redo --ids ' + redo_path + '\n')
			for fid, removed, lines in sorted(group_b):
				f.write(f'{fid}\n')
		print(f'  group B ids written to {redo_path}')
		for fid, removed, lines in sorted(group_b, key=lambda x: -x[1])[:6]:
			print(f'    {fid}: {removed} @measure and {lines} lines after the terminator')

	if not args.apply:
		print('\nreport only; pass --apply to repair group A')
		return 0

	os.makedirs(backup, exist_ok=True)
	repaired = 0
	for fid, path, kept, removed_lines, bars in group_a:
		shutil.copy2(path, os.path.join(backup, os.path.basename(path)))
		tmp = path + '.eotfix'
		with open(tmp, 'w', encoding='utf-8') as f:
			for line in kept:
				f.write(line + '\n')
		# Verified on the text just written, not on what was intended: a truncation that lost a
		# bar or left a terminator behind must not replace the published file.
		check = read_body(tmp)
		if (check[-1] != 'end_of_track' or bar_count(check) != bars
				or sum(1 for l in check if l == 'end_of_track') != 1):
			os.unlink(tmp)
			print(f'  REFUSED {fid}: the rewritten text failed its own check')
			continue
		os.replace(tmp, path)		# atomic, so a reader never sees a half file
		repaired += 1
	print(f'\nrepaired {repaired} of {len(group_a)} group A files, originals in {backup}')
	return 0


if __name__ == '__main__':
	sys.exit(main())
