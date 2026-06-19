"""Runtime syntax-blacklist mask for live generation.

A lightweight, parse-free counterpart to tools/lilylet_blacklist_gen.py's
BlacklistMonitor. It does NOT call any parser/oracle and has no Node dependency:
it trusts a pre-discovered variable-length n-gram blacklist and masks the
forbidden next tokens during sampling.

Wire-compatible with the ORT/torch generators' `monitor` hook:
  banned()          -> ids to mask for the next draw (suffix match on the context)
  accept(id)->bool  -> always True (we trust the mask; never re-parse)
  commit_forced(id) -> advance the running context for a forced token
Plus mark()/rollback() so a `[r:0/<measures>]` priming re-sample (a probe-then-
discard draw) can rewind the context before the forced redraw.

The context is the last N CONTENT token ids (whitespace dropped, `[r:x/y]` stream
markers excluded), maintained INCREMENTALLY in id space — no per-token tokenizer
call. Markers are detected by buffering a potential `[r:…]` run and classifying
the tiny buffer text with the same regexes _clean uses, so the resulting context
is identical to clean_for_parse()+tokenize but at near-zero cost.

This is a port of LilyScript/lilyscript/mask_monitor.py, kept in sync so the
deep-starry int8-ORT bench path can verify blacklist-masked generation.
"""

import os
import re
import json

# Anchored marker regexes for the id-space state machine — applied to the tiny
# marker BUFFER text (a candidate `[r:…]` run), never the whole stream:
#   - COMPLETE `\[r:\d+/\d*\]` -> drop the buffer entirely (a finished marker)
#   - viable PARTIAL `\[r(:(\d+(/\d*)?)?)?` -> keep buffering; it is excluded from
#       the context (mirrors discovery's _MARKER_PARTIAL_TAIL at end-of-stream)
#   - the lone `[` is a viable partial too, but stays VISIBLE as content (a bare
#       `[` is a header `[composer …]` / beam `c8[`, which _clean keeps)
#   - anything else -> not a marker -> flush the buffer back into the context
_MARK_COMPLETE_FULL = re.compile(r'\[r:\d+/\d*\]\Z')
_MARK_PARTIAL_FULL = re.compile(r'\[r(:(\d+(/\d*)?)?)?\Z')


def _whitespace_ids (tokenizer):
	'''Token ids for space/newline/tab/CR present in the vocab — dropped from the
	content context so it matches the corpus index space the blacklist was keyed in.'''
	out = []
	for ch in (' ', '\n', '\t', '\r'):
		enc = tokenizer.encode(ch)
		if len(enc) == 1:
			out.append(enc[0])
	return out


def load_blacklist (path):
	'''Load a blacklist JSON ({"id1,...,idn": [ids...]} under a top-level "blacklist"
	key, or bare) into dict[tuple(int,...) -> set[int]]. Keys are variable-length
	context n-grams. Missing file -> {}.'''
	if not path or not os.path.isfile(path):
		return {}
	data = json.load(open(path))
	raw = data.get('blacklist', data) if isinstance(data, dict) else {}
	bl = {}
	for key, ids in raw.items():
		ctx = tuple(int(x) for x in key.split(',')) if key else ()
		bl[ctx] = set(int(i) for i in ids)
	return bl


class MaskMonitor:
	'''Parse-free runtime mask. Construct with a generator (for tokenizer + special
	ids; LilyletPatchyGenerator exposes .tokenizer/.pad_id/.bos_id/.eos_id) and a
	variable-length-keyed blacklist dict; pass as `monitor=` to generate. A key
	(a,b,c) masks its forbidden tokens whenever the running content context ends with
	the sequence a,b,c (longest/any suffix match).'''

	def __init__ (self, gen, blacklist):
		self.tk = gen.tokenizer
		self.blacklist = blacklist or {}
		self.pad_id, self.bos_id, self.eos_id = gen.pad_id, gen.bos_id, gen.eos_id
		self._ws = set(_whitespace_ids(self.tk))
		self._key_lengths = sorted({len(k) for k in self.blacklist}, reverse=True) if self.blacklist else []
		self._max_ctx = max(self._key_lengths) if self._key_lengths else 0
		self._lbrack = self.tk.encode('[')[0]
		# Context is maintained INCREMENTALLY in id space — no per-token tokenizer
		# call. The only thing that needs care is excluding `[r:x/y]` stream markers,
		# whose chars share ids with real content. We buffer a *potential* marker
		# (always starting at `[`) and decide using the SAME regexes as _clean applied
		# to the tiny buffer text (id->char is a cheap dict lookup), so the resulting
		# context is provably identical to _clean()+encode of the full stream.
		self._ctx_ids = []        # confirmed visible content ids (whitespace dropped)
		self._buf = []            # token ids of an in-progress potential marker
		self._buf_text = ''       # their concatenated text (starts with '[')
		self._mark = None

	def _is_content (self, tid):
		return tid not in (self.pad_id, self.bos_id, self.eos_id)

	def _is_ctx (self, tid):
		return self._is_content(tid) and int(tid) not in self._ws

	def _text (self, tid):
		tid = int(tid)
		return '' if not self._is_content(tid) else self.tk.text_by_id.get(tid, '')

	def _push (self, tid):
		self._ctx_ids.append(int(tid))
		if len(self._ctx_ids) > self._max_ctx:
			del self._ctx_ids[:-self._max_ctx]

	def _push_content (self, tid):
		'''Route a non-marker token into the context: drop whitespace, keep content.'''
		if self._is_ctx(tid):
			self._push(tid)

	def _flush_buf (self):
		'''The buffered tokens are NOT a marker after all -> they are real content;
		replay them through the content path, then clear the buffer.'''
		buf = self._buf
		self._buf = []
		self._buf_text = ''
		for tid in buf:
			self._push_content(tid)

	def _feed (self, tid):
		'''Advance the incremental context with one committed token id.'''
		tid = int(tid)
		if not self._is_content(tid):
			# pad/bos/eos: not content, and breaks any pending marker buffer.
			if self._buf:
				self._flush_buf()
			return

		if self._buf:
			# extend the potential marker and re-classify the buffer text.
			text = self._buf_text + self._text(tid)
			if _MARK_COMPLETE_FULL.match(text):
				# a full `[r:x/y]` -> the whole buffer contributes nothing; drop it.
				self._buf = []
				self._buf_text = ''
				return
			if _MARK_PARTIAL_FULL.match(text):
				# still a viable marker prefix (`[r`, `[r:`, `[r:5`, `[r:5/`, `[r:5/3`)
				self._buf.append(tid)
				self._buf_text = text
				return
			# not a marker: flush the buffer as content, then process tid afresh below.
			self._flush_buf()

		if tid == self._lbrack:
			# start a potential marker (a lone `[` is itself real content until/unless
			# an `r` follows; that visibility is handled in banned()).
			self._buf = [tid]
			self._buf_text = '['
			return

		self._push_content(tid)

	# ---- generator-facing API ----

	def banned (self):
		'''Union forbidden sets of every stored key that is a suffix of the context.

		The visible context is _ctx_ids plus a pending lone `[` (which _clean keeps);
		a longer `[r…` partial buffer is stripped (matches _MARKER_PARTIAL_TAIL).'''
		if not self._key_lengths:
			return ()
		ctx = self._ctx_ids
		if self._buf_text == '[':
			ctx = ctx + [self._lbrack]
		out = set()
		for n in self._key_lengths:
			if n <= len(ctx):
				hit = self.blacklist.get(tuple(ctx[-n:]))
				if hit:
					out |= hit
		return out

	def accept (self, tid):
		# trust the mask: a non-banned draw is always committed (no re-parse)
		self.commit_forced(tid)
		return True

	def commit_forced (self, tid):
		if self._max_ctx:
			self._feed(tid)

	# ---- priming support (probe-then-discard rewind) ----

	def mark (self):
		'''Snapshot the context so a discarded priming-probe patch can be undone.'''
		self._mark = (list(self._ctx_ids), list(self._buf), self._buf_text)

	def rollback (self):
		'''Rewind to the last mark() (drops a probe patch's effect on the context).'''
		if self._mark is not None:
			self._ctx_ids = list(self._mark[0])
			self._buf = list(self._mark[1])
			self._buf_text = self._mark[2]
