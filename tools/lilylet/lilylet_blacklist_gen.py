"""Lilylet autoregressive 2-gram syntax-blacklist generator.

Continuously samples the LilyletNotaGen model (int8 ONNX + two-level KV cache,
via LilyScript's torch-free StreamingLilyletGenerator) at high temperature, and
uses the real Lilylet grammar parser as an ORACLE to discover forbidden 2-grams:

	(prev2_id, prev1_id) -> { forbidden_next_id, ... }

meaning "after the content tokens prev2 then prev1, sampling any id in the set
makes the document syntactically illegal". Seeded with
	(\\major, ' ') -> {\\major, \\minor}.

Mechanism (continuous stream + live in-place mask):
  - Each draw is masked by the blacklist set for the current 2-gram context.
  - A drawn token is tentatively appended and the marker-stripped text is sent
    to the parse oracle. If the parser rejects it with a NON-EOF error (a real
    "token not allowed here"), the (context -> token) pair is recorded and the
    token is redrawn with the now-larger mask — so sampling is pushed to find
    NEW violations rather than re-hitting known ones. EOF errors mean a valid-
    but-incomplete prefix and are ignored.

The oracle is a persistent Node process: lilylet/tools/parseOracleServer.ts run
via tsx, speaking line-delimited JSON over stdin/stdout.

Run on system python (onnxruntime + numpy; torch-free):
  /usr/local/bin/python tools/lilylet/lilylet_blacklist_gen.py --max-tokens 4000
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess

# import LilyScript's standalone generator (the int8-ONNX-KV backend)
LILYSCRIPT_DIR = os.environ.get('LILYSCRIPT_DIR', '/home/camus/work/LilyScript')
if LILYSCRIPT_DIR not in sys.path:
	sys.path.insert(0, LILYSCRIPT_DIR)

# our own tools dir (for lilylet_corpus_ngrams), so the harness works from any cwd
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
	sys.path.insert(0, _TOOLS_DIR)

DEFAULT_MODEL_DIR = '/home/camus/data/models/LilyNota/onnx'
DEFAULT_ASSET_DIR = os.path.join(LILYSCRIPT_DIR, 'assets')
DEFAULT_LILYLET_DIR = '/home/camus/work/lilylet'


# A complete stream marker `[r:<digits>/<digits>]` anywhere in the text.
_MARKER_COMPLETE = re.compile(r'\[r:\d+/\d*\]')
# A partial stream marker at the very END of the text: `[r`, `[r:`, `[r:0`,
# `[r:0/`, `[r:0/8` ... — stripped so an in-progress marker never reaches the
# parser. NOTE we require at least `[r`: a BARE trailing `[` is real Lilylet
# content (a header `[composer …]` or a beam `c8[ …`), not a marker, and must
# NOT be stripped — doing so corrupts the 2-gram context for the token after it
# (e.g. `[instrument` would lose its `[`). A bare `[` left in place simply opens
# a bracket, so the structural-depth gate suppresses judgment until it closes.
_MARKER_PARTIAL_TAIL = re.compile(r'\[r(:(\d+(/\d*)?)?)?$')


def clean_for_parse (raw):
	'''Strip stream markers (complete + trailing-partial) from raw generated text
	so the remaining content is valid Lilylet for the parse oracle.'''
	s = _MARKER_COMPLETE.sub('', raw)
	s = _MARKER_PARTIAL_TAIL.sub('', s)
	return s


def _whitespace_ids (tokenizer):
	'''Token ids for space/newline/tab/CR present in the vocab — dropped from the
	content n-gram context so it matches the corpus index space (whitespace isn't
	grammatically load-bearing and corpus/generation format boundaries differently).'''
	out = []
	for ch in (' ', '\n', '\t', '\r'):
		enc = tokenizer.encode(ch)
		if len(enc) == 1:
			out.append(enc[0])
	return out


class ParseOracle:
	'''Persistent Node parse-oracle child (lilylet/tools/parseOracleServer.ts via
	tsx), speaking line-delimited JSON over stdin/stdout. check(text) returns a
	dict {ok, eof, token, text, expected, message}. Respawns if the child dies.'''

	def __init__ (self, lilylet_dir=DEFAULT_LILYLET_DIR, node=None):
		self.lilylet_dir = lilylet_dir
		# prefer the project-local tsx, fall back to npx tsx
		local_tsx = os.path.join(lilylet_dir, 'node_modules', '.bin', 'tsx')
		if node:
			self.cmd = node + ['tools/parseOracleServer.ts']
		elif os.path.isfile(local_tsx):
			self.cmd = [local_tsx, 'tools/parseOracleServer.ts']
		else:
			self.cmd = ['npx', 'tsx', 'tools/parseOracleServer.ts']
		self.proc = None
		self.n_calls = 0
		self._spawn()

	def _spawn (self):
		self.proc = subprocess.Popen(
			self.cmd, cwd=self.lilylet_dir,
			stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
			text=True, bufsize=1)
		# warm-up handshake so the first real call isn't charged the tsx startup cost
		assert self.check('c4|') is not None

	def check (self, text):
		'''Parse `text`; return the oracle result dict. Respawns once on a dead child.'''
		for attempt in (0, 1):
			try:
				self.proc.stdin.write(json.dumps({'code': text}) + '\n')
				self.proc.stdin.flush()
				line = self.proc.stdout.readline()
				if not line:
					raise BrokenPipeError('oracle produced no output')
				self.n_calls += 1
				return json.loads(line)
			except (BrokenPipeError, ValueError, OSError):
				if attempt == 0:
					self._spawn()
				else:
					raise

	def close (self):
		if self.proc and self.proc.poll() is None:
			try:
				self.proc.stdin.close()
			except Exception:
				pass
			try:
				self.proc.wait(timeout=3)
			except Exception:
				self.proc.kill()


class BlacklistMonitor:
	'''Hook passed to StreamingLilyletGenerator.generate_stream. Holds the running
	raw text + a running content-token context + the growing blacklist, and uses a
	ParseOracle to decide whether each drawn token is syntactically legal AND a
	corpus n-gram index to calibrate the context length.

	The generator calls:
	  banned()         -> ids to mask for the next draw (variable-length suffix match)
	  accept(id)->bool -> tentatively append id; True to commit, False to redraw
	  commit_forced(id)-> commit a forced/non-sampled token (bos, prefix, eos-pad)

	On a real parse violation, the context is grown 1-gram -> 2-gram -> ... and at
	each length the `(context + offending token)` content-token sequence is looked
	up in the corpus. The violation is recorded at the SHORTEST length whose sequence
	has NO corpus match — so a pattern the corpus proves legal is never blacklisted.

	Context tokens are the marker-stripped, WHITESPACE-DROPPED content tokens (the
	same space the corpus index lives in: space/newline aren't grammatically
	load-bearing and the corpus formats boundaries with different whitespace).
	'''

	def __init__ (self, gen, oracle, blacklist, corpus, on_discover=None,
		parse_every_token=True, max_ctx=16):
		self.gen = gen
		self.tk = gen.tokenizer
		self.oracle = oracle
		self.blacklist = blacklist                  # dict[tuple(int,...) -> set[int]]
		self.corpus = corpus                        # CorpusNgrams
		self.on_discover = on_discover
		self.parse_every_token = parse_every_token
		self.max_ctx = max_ctx
		self.pad_id, self.bos_id, self.eos_id = gen.pad_id, gen.bos_id, gen.eos_id
		self._ws = set(_whitespace_ids(self.tk))    # space/newline/tab ids
		# Construct-OPENER tokens (`[` header/marker/beam, `<` chord, `{` tuplet/grace,
		# `"` string). Appending one can never be a TRUE violation — the parser's
		# non-EOF error on it is always incompleteness (the construct isn't closed yet),
		# e.g. `]\n[` looks illegal but `[genre …]` is a legal next header. The corpus
		# check can't reliably whitelist these (the content inside varies infinitely —
		# every composer name), so they're excluded from blacklisting structurally.
		self._opener_ids = set()
		for ch in ('[', '<', '{', '"'):
			enc = self.tk.encode(ch)
			if len(enc) == 1:
				self._opener_ids.add(enc[0])
		self.raw = ''                               # full decoded stream (with markers)
		self._ctx_ids = []                          # last <=max_ctx content non-ws ids
		self._clean = ''                            # marker-stripped stream
		self._in_str = False                        # inside an open string literal?
		self._depth = 0                             # open [ { < nesting depth
		self._base_valid = False                    # is the committed prefix a valid Lilylet prefix?
		# index blacklist keys by length for fast variable-length suffix matching
		self._key_lengths = sorted({len(k) for k in blacklist}, reverse=True) if blacklist else []
		self.n_tokens = 0                           # committed sampled tokens
		self.n_violations = 0

	# ---- helpers ----

	def _is_content (self, tid):
		return tid not in (self.pad_id, self.bos_id, self.eos_id)

	def _is_ctx (self, tid):
		'''A token that participates in the content n-gram context: content (not
		pad/bos/eos) and not whitespace (matching the corpus index space).'''
		return self._is_content(tid) and int(tid) not in self._ws

	def _text (self, tid):
		'''Decoded text for a token, mirroring StreamingLilyletGenerator.patch_to_text:
		pad/bos/eos contribute no text (their vocab entries are literal "<pad>" etc.,
		which must never enter the parsed stream).'''
		tid = int(tid)
		if not self._is_content(tid):
			return ''
		return self.tk.text_by_id.get(tid, '')

	def _structural_state (self, clean):
		'''Bracket/brace/angle depth + open-string flag for marker-stripped text.

		A "forbidden 2-gram" is only meaningful at a STRUCTURALLY SETTLED position:
		not inside a string literal and not inside an open [...]/{...}/<...>. Inside
		those, the parser lexes freeform/partial content and reports spurious non-EOF
		errors (e.g. a partial header string `[composer "Bee` -> token=PITCH, or a bare
		`[` prefix -> token=[) that are NOT real token-after-2-gram violations. We
		track depth/string from the committed clean text and only judge when settled.

		Returns (in_string, depth). Quotes toggle the string; brackets count only
		outside strings. `\\<` / `\\>` are escaped (hairpin) tokens, not angle
		brackets — but in clean text they appear as the 2-char run "\\<", and a bare
		'<'/'>' chord delimiter is a lone char, so we skip any '<'/'>' immediately
		preceded by a backslash.'''
		in_str = False
		depth = 0
		i = 0
		n = len(clean)
		while i < n:
			c = clean[i]
			if c == '"' and (i == 0 or clean[i - 1] != '\\'):
				in_str = not in_str
			elif not in_str:
				if c in '[{':
					depth += 1
				elif c in ']}':
					depth = max(0, depth - 1)
				elif c == '<' and (i == 0 or clean[i - 1] != '\\'):
					depth += 1
				elif c == '>' and (i == 0 or clean[i - 1] != '\\'):
					depth = max(0, depth - 1)
			i += 1
		return in_str, depth

	def _sync_ctx (self):
		'''Recompute, from the marker-stripped stream: the content context ids
		(_ctx_ids — last <=max_ctx non-whitespace content tokens), the structural
		state, and whether the committed prefix is a valid Lilylet prefix
		(_base_valid). _ctx_ids is re-encoded from a bounded char suffix (cheap; long
		enough to cover max_ctx tokens). _base_valid is only parsed when structurally
		settled (else the next token is never judged, so we skip the oracle call).'''
		clean = clean_for_parse(self.raw)
		self._clean = clean
		# encode a generous suffix and keep the last max_ctx context tokens
		ids = self.tk.encode(clean[-256:])
		ctx = [i for i in ids if self._is_ctx(i)]
		self._ctx_ids = ctx[-self.max_ctx:]
		self._in_str, self._depth = self._structural_state(clean)
		if not self.parse_every_token or self._in_str or self._depth != 0 or not clean:
			self._base_valid = False
		else:
			res = self.oracle.check(clean)
			self._base_valid = bool(res) and (res.get('ok') or res.get('eof'))

	# ---- generator-facing API ----

	def banned (self):
		'''Union the forbidden sets of every stored key that is a SUFFIX of the
		current content context (variable-length match: a key (a,b,c) fires when the
		context ends with a,b,c).'''
		if not self._key_lengths:
			return ()
		ctx = self._ctx_ids
		out = set()
		for n in self._key_lengths:
			if n <= len(ctx):
				hit = self.blacklist.get(tuple(ctx[-n:]))
				if hit:
					out |= hit
		return out

	def _record (self, key, tid, res):
		self.blacklist.setdefault(key, set()).add(tid)
		if len(key) not in self._key_lengths:
			self._key_lengths = sorted(set(self._key_lengths) | {len(key)}, reverse=True)
		self.n_violations += 1
		if self.on_discover:
			self.on_discover(key, tid, res, self._clean)

	def accept (self, tid):
		tid = int(tid)
		if not self._is_content(tid):
			self.commit_forced(tid)
			return True

		text = self._text(tid)
		candidate = self.raw + text

		# Correctness invariant: a token is a genuine violation ONLY when the base
		# prefix (everything before it) is ITSELF a valid Lilylet prefix; otherwise
		# the parse failure can't be blamed on this token. Also require a
		# structurally-settled position (outside strings/brackets) as a pre-filter.
		settled = self.parse_every_token and self._base_valid and not self._in_str and self._depth == 0
		# never blacklist a construct-opener (see _opener_ids): its non-EOF parse error
		# is incompleteness, not a real violation, and masking it breaks legal
		# continuations (e.g. the `[` of the next header line).
		if settled and self._is_ctx(tid) and tid not in self._opener_ids \
				and not _MARKER_PARTIAL_TAIL.search(candidate):
			res = self.oracle.check(clean_for_parse(candidate))
			if res and not res.get('ok') and not res.get('eof'):
				# Real "token not allowed here". Calibrate the context length against
				# the corpus: grow n = 1,2,... and find the SHORTEST context whose
				# (context + token) sequence does NOT appear in the corpus — recording
				# at that length avoids banning a pattern the corpus proves legal
				# (e.g. `] [` header-to-header, which the corpus contains).
				ctx = self._ctx_ids
				key = None
				for n in range(1, len(ctx) + 1):
					seq = ctx[-n:] + [tid]
					if not self.corpus.contains(seq):
						key = tuple(ctx[-n:])
						break
				if key is None:
					# Even the full available context + token occurs in the corpus, OR
					# there is no context. If the corpus contains it at every length, it
					# is a legal pattern the model merely placed in an illegal spot we
					# can't capture with a suffix n-gram -> do NOT blacklist (would cause
					# false-positive masking). Skip recording; just reject this draw so
					# generation continues with a different token.
					if ctx and self.corpus.contains(ctx[-len(ctx):] + [tid]):
						return False
					# no usable context (start of stream) -> record a 1-gram on the token
					key = tuple(ctx[-1:]) if ctx else ()
				self._record(key, tid, res)
				return False    # redraw; nothing committed

		# commit
		self.raw = candidate
		self.n_tokens += 1
		self._sync_ctx()
		return True

	def commit_forced (self, tid):
		tid = int(tid)
		self.raw += self._text(tid)
		if self._is_content(tid):
			self.n_tokens += 1
		self._sync_ctx()


# ---- persistence -----------------------------------------------------------

def load_blacklist (path):
	'''Load {"id1,...,idn": [ids...]} -> dict[tuple(int,...) -> set[int]]. Missing -> {}.'''
	bl = {}
	if path and os.path.isfile(path):
		data = json.load(open(path))
		for key, ids in data.get('blacklist', data).items():
			ctx = tuple(int(x) for x in key.split(',')) if key else ()
			bl[ctx] = set(int(i) for i in ids)
	return bl


def save_blacklist (path, blacklist, tk, meta=None, examples=None):
	'''Persist the blacklist as JSON plus a human-readable companion (ids->text).
	Keys are variable-length context tuples. examples: optional {(ctx..., tid) ->
	sample base prefix} written into the readable file for audit/reproducibility.'''
	def tid_text (tid):
		return tk.text_by_id.get(int(tid), '?')
	def ctx_text (ctx):
		return ' '.join(repr(tid_text(i)) for i in ctx)

	compact = {','.join(str(i) for i in ctx): sorted(ids) for ctx, ids in sorted(blacklist.items())}
	readable = {
		ctx_text(ctx): sorted(tid_text(i) for i in ids)
		for ctx, ids in sorted(blacklist.items())
	}
	os.makedirs(os.path.dirname(path), exist_ok=True)
	json.dump({'meta': meta or {}, 'blacklist': compact}, open(path, 'w'), indent=2, ensure_ascii=False)
	rpath = path.replace('.json', '_readable.json')
	rdata = {'meta': meta or {}, 'blacklist': readable}
	if examples:
		rdata['examples'] = {
			(ctx_text(ctx) + ' -> ' + repr(tid_text(t))): prefix
			for (ctx, t), prefix in sorted(examples.items())
		}
	json.dump(rdata, open(rpath, 'w'), indent=2, ensure_ascii=False)
	return path, rpath


# ---- seed -------------------------------------------------------------------

def seed_blacklist (tk):
	'''The user-provided seed entry: after `\\major` then space, `\\major`/`\\minor`
	are illegal. Whitespace is dropped from the context (matching the corpus space),
	so the 1-gram context is just `(\\major,)`.'''
	def one (s):
		ids = tk.encode(s)
		assert len(ids) == 1, f'{s!r} is not a single token: {ids}'
		return ids[0]
	major, minor = one('\\major'), one('\\minor')
	return {(major,): {major, minor}}


# ---- discovery loop ---------------------------------------------------------

def run (args):
	from lilyscript.generator import StreamingLilyletGenerator
	from lilylet_corpus_ngrams import load_or_build

	print(f'[init] loading generator: model={args.model_dir} assets={args.asset_dir}')
	gen = StreamingLilyletGenerator(args.model_dir, args.asset_dir, threads=args.threads)
	tk = gen.tokenizer

	print(f'[init] loading corpus index from {args.corpus}')
	corpus = load_or_build(args.corpus, tk, index_path=args.corpus_index, n_max=args.n_max,
		rebuild=args.rebuild_corpus)
	print(f'[init] corpus n-gram counts: {corpus.stats()}')

	blacklist = load_blacklist(args.out) if args.resume else {}
	for ctx, ids in seed_blacklist(tk).items():
		blacklist.setdefault(ctx, set()).update(ids)
	print(f'[init] blacklist seeded: {len(blacklist)} contexts, '
		f'{sum(len(v) for v in blacklist.values())} banned tokens')

	oracle = ParseOracle(lilylet_dir=args.lilylet_dir, node=(args.node.split() if args.node else None))

	def tid_text (tid):
		return tk.text_by_id.get(int(tid), '?')

	discoveries = []
	examples = {}    # (ctx_tuple, tid) -> a sample valid base prefix that triggered it
	def on_discover (ctx, tid, res, base_prefix):
		ctxs = ' '.join(repr(tid_text(i)) for i in ctx) or '<start>'
		exp = ','.join(e.strip("'") for e in res.get('expected', [])[:6])
		print(f"[blacklist] ({ctxs}) += {tid_text(tid)!r}  [n={len(ctx)}] "
			f"(token={res.get('token')}; expected: {exp}...)")
		discoveries.append((ctx, tid))
		examples.setdefault((ctx, int(tid)), base_prefix)

	t0 = time.time()
	total_tokens = 0
	try:
		for r in range(args.restarts):
			if total_tokens >= args.max_tokens:
				break
			mon = BlacklistMonitor(gen, oracle, blacklist, corpus, on_discover=on_discover,
				parse_every_token=not args.sparse_parse)
			seed = args.seed + r
			print(f'[stream {r}] seed={seed} temp={args.temperature} top_p={args.top_p} '
				f'(tokens so far {total_tokens}/{args.max_tokens})')
			for raw, pretty, done in gen.generate_stream(
				prompt_text=args.prompt, max_patches=args.max_patches,
				temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
				measures=(None if args.measures < 0 else args.measures),
				seed=seed, monitor=mon):
				if done:
					break
			total_tokens += mon.n_tokens
			print(f'[stream {r}] done: {mon.n_tokens} tokens, {mon.n_violations} violations this stream, '
				f'{oracle.n_calls} oracle calls total')
			# checkpoint after every stream
			save_blacklist(args.out, blacklist, tk, meta=_meta(args, total_tokens, t0, oracle), examples=examples)
	except KeyboardInterrupt:
		print('\n[interrupted] saving progress...')
	finally:
		oracle.close()

	p, rp = save_blacklist(args.out, blacklist, tk, meta=_meta(args, total_tokens, t0, oracle), examples=examples)
	dt = time.time() - t0
	n_pairs = sum(len(v) for v in blacklist.values())
	print(f'\n[done] {len(blacklist)} contexts / {n_pairs} forbidden pairs '
		f'({len(discoveries)} new this run) in {dt:.1f}s, {total_tokens} tokens, {oracle.n_calls} oracle calls')
	print(f'[done] wrote {p}\n[done] wrote {rp}')


def _meta (args, total_tokens, t0, oracle):
	return {
		'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k,
		'seed': args.seed, 'restarts': args.restarts, 'measures': args.measures,
		'tokens': total_tokens, 'oracle_calls': oracle.n_calls,
		'elapsed_s': round(time.time() - t0, 1), 'model_dir': args.model_dir,
	}


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--max-tokens', type=int, default=4000, help='stop after this many committed tokens (across restarts)')
	ap.add_argument('--restarts', type=int, default=8, help='max number of fresh streams to sample')
	ap.add_argument('--max-patches', type=int, default=256, help='patch cap per stream')
	ap.add_argument('--temperature', type=float, default=1.3, help='sampling temperature (high -> more violations)')
	ap.add_argument('--top-p', type=float, default=0.98)
	ap.add_argument('--top-k', type=int, default=0)
	ap.add_argument('--seed', type=int, default=0, help='base RNG seed; stream r uses seed+r')
	ap.add_argument('--measures', type=int, default=-1, help='force measure count (-1 = let the model choose)')
	ap.add_argument('--prompt', type=str, default='', help='optional metadata prompt prefix')
	ap.add_argument('--sparse-parse', action='store_true', help='(reserved) skip parsing tokens unlikely to complete a construct')
	ap.add_argument('--threads', type=int, default=None, help='onnxruntime intra-op threads')
	ap.add_argument('--resume', action='store_true', help='load + extend an existing --out blacklist')
	ap.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'lilylet_blacklist.json'))
	ap.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR)
	ap.add_argument('--asset-dir', type=str, default=DEFAULT_ASSET_DIR)
	ap.add_argument('--lilylet-dir', type=str, default=DEFAULT_LILYLET_DIR)
	ap.add_argument('--node', type=str, default='', help='override oracle command, e.g. "node --loader tsx"')
	# corpus calibration
	from lilylet_corpus_ngrams import DEFAULT_CORPUS_DIRS, DEFAULT_INDEX_PATH, DEFAULT_N_MAX
	ap.add_argument('--corpus', nargs='+', default=DEFAULT_CORPUS_DIRS,
		help='one or more dirs of legal .lyl files used to calibrate context length')
	ap.add_argument('--corpus-index', type=str, default=DEFAULT_INDEX_PATH, help='persisted corpus n-gram index')
	ap.add_argument('--n-max', type=int, default=DEFAULT_N_MAX, help='max indexed n-gram length')
	ap.add_argument('--rebuild-corpus', action='store_true', help='force rebuild of the corpus index')
	run(ap.parse_args())


if __name__ == '__main__':
	main()
