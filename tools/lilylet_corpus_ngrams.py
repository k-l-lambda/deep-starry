"""Corpus content-token n-gram index for the syntax-blacklist generator.

Calibration: a 2-gram (or n-gram) `context + token` sequence is only blacklisted
if it does NOT appear anywhere in a corpus of known-good Lilylet documents. This
module builds and queries that corpus index, in the SAME content-token-id space
the model generates in:
  - corpus text -> normalize_text (strips `%` comments) -> tokenizer.encode
    -> drop pad/bos/eos -> a flat list of content ids per file.
  - index[n] = set of all length-n id tuples occurring across the corpus.
  - contains(seq): tuple(seq) in index[len] for len <= n_max, else a substring
    scan over the per-file id streams (rare — only when a short n-gram already
    failed to match, so deep scans are uncommon).

The model generates with `[r:x/y]` stream markers and no `%` comments; the corpus
has `%` comments and no markers. Both reduce to the same content tokens here, so
sequences are directly comparable.

CLI:
  python tools/lilylet_corpus_ngrams.py --build         # build + persist the index
  python tools/lilylet_corpus_ngrams.py --probe "]\\n["  # test a literal string
"""

import os
import re
import sys
import json
import argparse

LILYSCRIPT_DIR = os.environ.get('LILYSCRIPT_DIR', '/home/camus/work/LilyScript')
if LILYSCRIPT_DIR not in sys.path:
	sys.path.insert(0, LILYSCRIPT_DIR)

DEFAULT_CORPUS_DIRS = [
	'/home/camus/work/lilylet/tests/output/notagenx-from-abc-meta',   # 215 abc2lyl --meta (multi-header)
	'/home/camus/work/lilylet/tests/output/lyl-corpus-jiuzhang',      # 28 from jiuzhang.51 /tmp/lyl_new
]
DEFAULT_ASSET_DIR = os.path.join(LILYSCRIPT_DIR, 'assets')
DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'lilylet_corpus_ngrams.json')

# default max indexed n-gram length; longer queries fall back to a stream scan.
DEFAULT_N_MAX = 8


def _normalize (text):
	'''Strip `%` comments per line (mirror starry...patchifier.normalize_text) so the
	corpus tokenizes into the same content stream the model generates.'''
	text = text.replace('\r\n', '\n').replace('\r', '\n')
	return '\n'.join(line.split('%', 1)[0].rstrip() for line in text.split('\n'))


class CorpusNgrams:
	'''Content-token n-gram index over a corpus of legal .lyl files.'''

	def __init__ (self, streams, n_max=DEFAULT_N_MAX):
		self.streams = streams                 # list[list[int]] content ids per file
		self.n_max = n_max
		self.index = {n: set() for n in range(1, n_max + 1)}
		for ids in streams:
			L = len(ids)
			for n in range(1, n_max + 1):
				idx = self.index[n]
				for i in range(L - n + 1):
					idx.add(tuple(ids[i:i + n]))

	def contains (self, seq):
		'''True if the content-id sequence occurs anywhere in the corpus.'''
		seq = tuple(int(x) for x in seq)
		n = len(seq)
		if n == 0:
			return True
		if n <= self.n_max:
			return seq in self.index[n]
		# longer than the indexed cap: substring scan over the streams (uncommon)
		for ids in self.streams:
			L = len(ids)
			for i in range(L - n + 1):
				if tuple(ids[i:i + n]) == seq:
					return True
		return False

	def stats (self):
		return {n: len(s) for n, s in self.index.items()}

	# ---- persistence ----

	def save (self, path):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		# store streams (compact) + n_max; the per-length sets are rebuilt on load.
		json.dump({'n_max': self.n_max, 'streams': self.streams}, open(path, 'w'))

	@classmethod
	def load (cls, path):
		data = json.load(open(path))
		return cls(data['streams'], n_max=data.get('n_max', DEFAULT_N_MAX))


def _content_ids (tokenizer, text):
	'''normalize -> encode -> drop pad/bos/eos AND whitespace -> list[int].

	Whitespace tokens (space/newline/tab) are dropped because they are NOT
	grammatically load-bearing between Lilylet tokens (`\\major \\minor` and
	`\\major\\minor` yield the same parse), and — critically — the corpus formats
	boundaries with DOUBLE newlines / blank lines while the model emits single
	`\\n`. Dropping whitespace makes corpus and generation n-grams comparable
	regardless of line-formatting. The discovery/runtime sides drop whitespace the
	same way when forming their context, so the spaces are consistently ignored.'''
	ids = tokenizer.encode(_normalize(text))
	drop = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}
	drop |= set(WHITESPACE_IDS(tokenizer))
	return [i for i in ids if i not in drop]


def WHITESPACE_IDS (tokenizer):
	'''Token ids for space / newline / tab / carriage-return (those present in vocab).'''
	out = []
	for ch in (' ', '\n', '\t', '\r'):
		enc = tokenizer.encode(ch)
		if len(enc) == 1:
			out.append(enc[0])
	return out


def build_index (corpus_dirs, tokenizer, n_max=DEFAULT_N_MAX):
	'''Tokenize every *.lyl across one or more corpus dirs into a content-id stream
	and index it. corpus_dirs may be a single path or a list of paths.'''
	import glob
	if isinstance(corpus_dirs, str):
		corpus_dirs = [corpus_dirs]
	streams = []
	files = []
	for d in corpus_dirs:
		files.extend(sorted(glob.glob(os.path.join(d, '*.lyl'))))
	for f in files:
		try:
			streams.append(_content_ids(tokenizer, open(f, encoding='utf-8').read()))
		except Exception as e:
			print(f'[corpus] skip {os.path.basename(f)}: {e}')
	print(f'[corpus] indexed {len(streams)} files from {len(corpus_dirs)} dir(s), '
		f'{sum(len(s) for s in streams)} content tokens, n_max={n_max}')
	return CorpusNgrams(streams, n_max=n_max)


def load_or_build (corpus_dirs, tokenizer, index_path=DEFAULT_INDEX_PATH, n_max=DEFAULT_N_MAX, rebuild=False):
	'''Load a persisted index, or build (and persist) it if missing/stale.'''
	if not rebuild and index_path and os.path.isfile(index_path):
		try:
			idx = CorpusNgrams.load(index_path)
			if idx.n_max >= n_max:
				return idx
		except Exception as e:
			print(f'[corpus] rebuild (load failed: {e})')
	idx = build_index(corpus_dirs, tokenizer, n_max=n_max)
	if index_path:
		idx.save(index_path)
	return idx


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--corpus', nargs='+', default=DEFAULT_CORPUS_DIRS,
		help='one or more corpus dirs of legal .lyl files')
	ap.add_argument('--asset-dir', default=DEFAULT_ASSET_DIR)
	ap.add_argument('--index', default=DEFAULT_INDEX_PATH)
	ap.add_argument('--n-max', type=int, default=DEFAULT_N_MAX)
	ap.add_argument('--build', action='store_true', help='(re)build + persist the index')
	ap.add_argument('--probe', help='literal string to test for corpus membership (supports \\n, \\t)')
	args = ap.parse_args()

	from lilyscript.tokenizer import LilyletTokenizer
	tk = LilyletTokenizer(os.path.join(args.asset_dir, 'lilylet-tokenizer.json'))
	idx = load_or_build(args.corpus, tk, index_path=args.index, n_max=args.n_max, rebuild=args.build)
	print('[corpus] per-length n-gram counts:', idx.stats())

	if args.probe is not None:
		s = args.probe.encode().decode('unicode_escape')
		drop = {tk.pad_id, tk.bos_id, tk.eos_id} | set(WHITESPACE_IDS(tk))
		ids = [i for i in tk.encode(s) if i not in drop]
		print(f'probe {s!r} -> content ids {ids} ({[tk.text_by_id.get(i) for i in ids]})')
		print('contains:', idx.contains(ids))


if __name__ == '__main__':
	main()
