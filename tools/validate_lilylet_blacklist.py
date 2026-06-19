"""Validate a lilylet 2-gram blacklist produced by lilylet_blacklist_gen.py.

For every recorded discovery it replays the captured base prefix (a real valid
Lilylet prefix that ended in the 2-gram context) through the parse oracle and
checks:
  - base prefix parses ok or eof (a valid, possibly-incomplete prefix), AND
  - base + forbidden-token text parses with a NON-EOF error (a genuine
    "token not allowed here").
Entries that fail either check are reported as suspect (context too short to
reproduce, or a false positive). Pairs without a captured example (e.g. the
seed, or entries loaded from a prior --resume) are reported separately.

  /usr/local/bin/python tools/validate_lilylet_blacklist.py [--in tools/output/lilylet_blacklist.json]
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lilylet_blacklist_gen import ParseOracle, clean_for_parse, DEFAULT_LILYLET_DIR, DEFAULT_ASSET_DIR

LILYSCRIPT_DIR = os.environ.get('LILYSCRIPT_DIR', '/home/camus/work/LilyScript')
if LILYSCRIPT_DIR not in sys.path:
	sys.path.insert(0, LILYSCRIPT_DIR)


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--in', dest='inp', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'lilylet_blacklist.json'))
	ap.add_argument('--asset-dir', default=DEFAULT_ASSET_DIR)
	ap.add_argument('--lilylet-dir', default=DEFAULT_LILYLET_DIR)
	from lilylet_corpus_ngrams import DEFAULT_CORPUS_DIRS, DEFAULT_INDEX_PATH, DEFAULT_N_MAX
	ap.add_argument('--corpus', nargs='+', default=DEFAULT_CORPUS_DIRS)
	ap.add_argument('--corpus-index', default=DEFAULT_INDEX_PATH)
	ap.add_argument('--n-max', type=int, default=DEFAULT_N_MAX)
	ap.add_argument('--verbose', action='store_true')
	args = ap.parse_args()

	from lilyscript.tokenizer import LilyletTokenizer
	tk = LilyletTokenizer(os.path.join(args.asset_dir, 'lilylet-tokenizer.json'))
	tx = lambda i: tk.text_by_id.get(int(i), '?')
	ctx_text = lambda ctx: ' '.join(repr(tx(i)) for i in ctx) or '<start>'

	data = json.load(open(args.inp))
	blacklist = data['blacklist']
	rpath = args.inp.replace('.json', '_readable.json')
	examples = {}
	if os.path.isfile(rpath):
		rdata = json.load(open(rpath))
		examples = rdata.get('examples', {})

	# corpus index for the non-containment invariant
	from lilylet_corpus_ngrams import load_or_build
	corpus = load_or_build(args.corpus, tk, index_path=args.corpus_index, n_max=args.n_max)

	oracle = ParseOracle(lilylet_dir=args.lilylet_dir)
	legit = suspect = no_example = 0
	suspects = []
	for key, ids in blacklist.items():
		ctx = tuple(int(x) for x in key.split(',')) if key else ()
		for tid in ids:
			tid = int(tid)
			ekey = f'{ctx_text(ctx)} -> {tx(tid)!r}'
			# invariant 1: corpus must NOT contain (context + token)
			in_corpus = corpus.contains(list(ctx) + [tid])
			prefix = examples.get(ekey)
			if prefix is None:
				no_example += 1
				# can still check the corpus invariant for seed/resumed entries
				if in_corpus:
					suspect += 1
					suspects.append((ekey, '<no example>', None, {'in_corpus': True}))
				continue
			bres = oracle.check(prefix)
			base_ok = bool(bres) and (bres.get('ok') or bres.get('eof'))
			pres = oracle.check(prefix + tx(tid))
			is_violation = bool(pres) and (not pres.get('ok')) and (not pres.get('eof'))
			if base_ok and is_violation and not in_corpus:
				legit += 1
				if args.verbose:
					print(f'OK   {ekey}   token={pres.get("token")}')
			else:
				suspect += 1
				suspects.append((ekey, prefix, base_ok, dict(pres, in_corpus=in_corpus)))
	oracle.close()

	for ekey, prefix, base_ok, pres in suspects:
		print(f'SUSPECT {ekey}')
		print(f'        prefix={prefix!r}')
		print(f'        base_valid={base_ok}  probe: ok={pres.get("ok")} eof={pres.get("eof")} '
			f'token={pres.get("token")}  in_corpus={pres.get("in_corpus")}')

	total = legit + suspect
	print(f'\n[validate] {legit}/{total} discoveries legit (oracle-illegal AND not in corpus), '
		f'{suspect} suspect; {no_example} entries without a captured example (seed/resumed).')


if __name__ == '__main__':
	main()
