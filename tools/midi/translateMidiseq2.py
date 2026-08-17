'''Whole-file midiseq2 -> midiseq2 translation with MidiTranslator, by sliding window.

MidiTranslator is trained by starry.midi.data.seq2seq2.Seq2Seq2 on CROPS, never whole pieces: one
flat sequence `source... <sep> target... <eos>`, supervised only on the target half. A real file runs
to ~7.7k lines / ~76k tokens, far past any single crop, so translating one end to end means sliding a
window across the source and stitching the generated target halves into one stream.

The mechanic — two windows advancing in lockstep:

	output[]      the generated target stream (append-only; this IS the output)
	prime_start   left edge of a sliding VIEW of its tail; prime = output[prime_start:]
	src_cursor    line cursor into the source file

Each step assembles `src_window <sep> prime`, generates to <eos>, appends every new token to `output`,
then moves `prime_start` to at least `--advance-tokens` target tokens ahead, rounding the cut upward to
 the next complete `<eom>` measure boundary. The internal safety ceiling can retire more old measures if
the retained primer grows too large.

	step0   src[bars 1-6]  <sep> (empty)      -> generates bars 1-6   (~320 tok)
	step1   src[bars 2-7]  <sep> prime[2-6]   -> generates bar 7      (~64 tok)  [default stride]
	step2   src[bars 3-8]  <sep> prime[3-7]   -> generates bar 8

The default stride is one target token, rounded to the next `<eom>`; larger `--advance-tokens` values
request a farther target cut and reduce the number of decoder windows at the cost of a larger jump between
source/target views.

The view has to stay within the trained target-half distribution. Its movement is controlled explicitly by
--advance-tokens (default 1 target token): each step requests at least that many tokens from the target
view, then rounds the cut upward to the next complete `<eom>` measure. This is deliberately a movement
parameter, not a retention-cap parameter. An internal safety cap trims an overgrown primer only when the
model emits more than the trained context can hold; it is not the user-facing stride control.

The FIRST token of every step has <eos> masked out of its logits, so no step can return empty. An empty
step is the one thing the loop cannot use: no <eom> to advance past, no onsets to move the source cursor
by, so it falls through to the anti-stall fallbacks and advances by something unrelated to the music.

An immediate <eos> at the END of a piece is the correct answer, not a stall: the source window ends in
`end_of_track`, the output already carries one, and there is nothing left to translate. Since the mask
means a finished step no longer shows up as an empty one, `finished` is keyed on the model HAVING WANTED
<eos> (both halves must still agree, so a spurious early end_of_track cannot truncate a file that still
has source) and ends the run cleanly instead of grinding through the remaining lines.

The mask is reported, never silent: `stats['eos_forced']` counts every step it fired on and `stalls`
counts the mid-piece subset. A mid-piece override means the model wanted to stop and was overruled --
usually the primer having grown out of band -- so a high count is a defect to chase, not a run that
went well.

Sizing is PER-RUN, not a property of the format: it follows the training config's `line_range`, so the
right --src-window has to be re-measured for whatever checkpoint you are running. Two runs measured,
each over ~800 crops through that run's own args (nota1m, mark_mode tick, pos_style sep):

	line_range [20,256]                         line_range [64,512]  (20260814 l16d256)
	  source half   median 275  p99  635         source half   median 598  p90  967  p99 1236  max 1495
	  target half   median 130  p99  351         target half   median 287  p90  523  p99  738
	  total T       median 409  p99  916         total T       median 916  p90 1481  p99 1844  max 2022

So `max_tokens: 2048` in the training config is NOT a window size — it is a resample cap on total T
that essentially never fires.

--src-window defaults to 640, which is about the p99 of the FIRST column only. On the l16d256 run it is
p55, covering just 54.5% of trained source halves, and that under-sizing measurably hurts. Swept with
tests/midi/translate_accuracy_check.py (10 files, bars 7..18, greedy, e358), onset F1 falls into three
regimes, with every boundary checked per-file rather than by comparing means:

	480 / 640     0.361 / 0.552   under-sized; tied with each other, both clearly beaten
	960 / 1260    0.714 / 0.729   THE PLATEAU; tied with each other (median delta 0.000)
	1500 / 1660   0.388 / 0.167   collapsed

Use 960: it ties 1260 on accuracy while costing 15% less (108 vs 125 s/file). 640 -> 960 is the real
gain (7/10 files, median +0.157).

The collapse lands exactly at the run's trained MAX source half (1495) — 1500 is the first window
exceeding every source half the model has seen. Past it the model MISCOUNTS BARS: notes and local
rhythm survive (pitch F1 still 0.87, and correcting one constant shift recovers e.g. 0.136 -> 0.836)
but the stream lands whole bars off, with best shifts of +960 (exactly one 2/4 bar), +720, +1680; note
counts climb 179 -> 239 while bars generated FALL. Not a budget artifact: T at 1500 is 1821, well under
max_token 2048.

So size the window against the run's own source-half p90..p99 and stay clear of its max. The old "1024
would exceed anything trained" caution was right in kind but specific to the narrow-crop column — here
the equivalent ceiling is ~1495, not ~1001.

Two asymmetries with training worth knowing, both deliberate:

  - The target half here is PRIMED with the previous window's tail so generation continues across the
    seam. Training always started a target half fresh at a crop boundary, so a primed target is
    out-of-distribution; it is what keeps successive windows one stream instead of overlapping
    alternatives. Pass --no-prime to fall back to the trained form.

    Measured (l16d256 e358, 10 files, bars 7..18): the prime is ESSENTIAL and its size is nearly
    irrelevant. --no-prime scores onset F1 0.147 against 0.560 at prime 320 -- worse than any
    src_window setting, losing 7/10 files, three of them from ~1.0 to ~0.1, with anchors scattering
    both directions and no stalls. Being out-of-distribution costs far less than not knowing where in
    the target stream you are. But prime_window 160/240/320 were mutually indistinguishable (every
    pairwise median delta ~0.00) in that sweep; the current default is 2048 so ordinary generated
    blocks are not aggressively trimmed. Re-measure if the checkpoint or generation ceiling changes;
    total spread across 160..740 was 0.085 against src_window's 0.562.

    Above ~520 it degrades (6/10 files worse) because prime_window is a CEILING that only acts when
    the view exceeds it: primes 520 and 740 produce byte-identical runs on 8/10 files, both meaning
    "never trim". So the real variable is whether trim_prime fires, not how large the cap is -- and
    trained target-half statistics are the WRONG yardstick for sizing it, since they describe complete
    crops rather than mid-stream continuations.
  - A training crop's left edge always landed on an @tick line. Here it lands on an arbitrary line
    boundary, because the production source carries no @measure/@tick at all. That costs nothing in
    the token sequence (with source_eom off the feeder discards both directive kinds on the source
    half, so a marked and an unmarked source encode identically) but it is why the input cursor has to
    be advanced by MATCHING note_on counts rather than by a mark-key lookup.

The note_on match is approximate by nature: across the two arms of this corpus only 3 of 99 files have
equal note_on counts (median relative delta 2.1%, max 27.4%), so the cursor drifts. --verbose reports
the drift per step. Matching on onset pitch values is the better scheme once output quality justifies
it; `advance_source_by_onsets` is isolated so it can be swapped without touching the loop.

No KV cache: each step re-runs the whole prefix per token, O(T^2) per token. Meant for inspection-scale
translation, not bulk decoding.

--inspect answers a different question from the accuracy metrics: not "is the output right" but "what
is the model looking at". For every generated pitch token it records the attention over the SOURCE pitch
tokens, reduced by max over layers and heads, and draws the surviving links as onset correspondences --
ONE FIGURE PER SLIDING STEP, since a step is the unit the model works in and superimposing steps would
draw links across distances no forward pass ever spanned. See AttentionInspector for why one extra
forward per window reproduces the generation-time rows exactly rather than approximately, and why it
forces eager attention.

Every generated note keeps at least its strongest link. When nothing clears --inspect-threshold the
argmax alone is kept, drawn dotted and counted apart from passing links everywhere it is reported: a note
with no line at all now means its window held no source notes, not that the threshold cut it. Without
this, a diffuse row and an absent row looked identical on the figure, and those are different claims.

Primer notes are queried too. Their rows come free from the same forward pass, and they answer a separate
question -- where the model looks while READING its carried-in context rather than while producing a note.
They are drawn in their own colour and dash and are excluded from every statistic: a note appears as primer
in step n only after being scored as generated in step n-1, so counting both would count it twice.

Usage:
  python tools/midi/translateMidiseq2.py --run <run_dir> --input a.midiseq2.txt --output b.midiseq2.txt
  python tools/midi/translateMidiseq2.py --run <run_dir> --input a.txt --temperature 0 --verbose
  python tools/midi/translateMidiseq2.py --run <run_dir> --input a.txt --src-window 960 \
      --inspect --inspect-threshold 0.05 --max-steps 8
'''

import argparse
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModel
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer, Midiseq2Vocab
from starry.midi.data.unifiedSeq2Tokenizer import UnifiedSeq2Tokenizer
from starry.lilylet.patchyGenerator import sample_next


DEFAULT_RUN = '/home/claude/training/midi/20260812-midi-translator-nota1m00-sep-l8d512'


# --- vocab token classes ------------------------------------------------------------------

def keyword_tokens (tokenizer):
	'''The event-keyword tokens, derived from the vocab rather than hardcoded.

	A midiseq2 line starts with one of these (note_on, control_change, set_tempo, ...) or with an
	elapse run. Everything else in the vocab is a field: E... elapse, C... channel, #... arg3,
	$... arg4, a single hex nibble, or `_`/`-`. Deriving the set means a vocab change cannot leave a
	stale list behind — the line splitter would otherwise silently glue two events together.
	'''
	out = set()
	for tok in tokenizer.tokens:
		if tok.startswith('<') or tok[0] in 'EC#$' or len(tok) == 1:
			continue
		out.add(tok)
	return out


def is_elapse (tok):
	'''An E-prefixed hex delta. Mirrors Midiseq2Tokenizer._is_elapse.'''
	if len(tok) < 2 or tok[0] != 'E':
		return False
	try:
		int(tok[1:], 16)
		return True
	except ValueError:
		return False


# --- source encoding (mirrors Seq2Seq2._encode) -------------------------------------------

def encode_lines (lines, tokenizer, eom=False):
	'''Lines -> token ids, by the same rules as Seq2Seq2._encode (seq2seq2.py:399).

	Directives are CONTROL, never content: @tick always vanishes; @measure becomes a single <eom>
	only when `eom` is set, and @measure 1 emits nothing either way (it opens the piece rather than
	closing a bar, and <bos> already carries that). Kept as a standalone function so the parity test
	can call it against the feeder's own output.
	'''
	ids = []
	lookup = tokenizer.id_by_token
	unknown = tokenizer.unknown_id
	for line in lines:
		if line.startswith('@measure'):
			if eom and line.split()[1] != '1':
				ids.append(tokenizer.eom_id)
			continue
		if line.startswith('@tick'):
			continue
		for token in line.split():
			ids.append(lookup.get(token, unknown))
	return ids


# --- rendering (ids -> midiseq2 text) -----------------------------------------------------

def render_lines (ids, tokenizer, keywords, first_measure=2):
	'''Token ids -> midiseq2 text lines.

	Re-grouping is by line START: a keyword opens a line, and an elapse run opens one too (elapse
	tokens group with each other, since a delta of 0x1234 renders as `E1000 E230 E4` on one line).
	<eom> becomes an `@measure N` line off a running counter — the token stream carries no number,
	only the boundary. Measure numbering starts at `first_measure` because <eom> marks the OPENING of
	bar N for N >= 2 (@measure 1 emits no token at all), so the first <eom> we see is bar 2.

	<bos>/<eos>/<sep> are structural and dropped. <pad> should never appear; <unknown> is preserved
	verbatim so it shows up in the output as a visible defect rather than being silently dropped.
	'''
	lines = []
	current = []
	measure = first_measure
	skip = {tokenizer.bos_id, tokenizer.eos_id, tokenizer.sep_id, tokenizer.pad_id}

	def flush ():
		if current:
			lines.append(' '.join(current))
			del current[:]

	for tid in ids:
		if tid in skip:
			continue
		if tid == tokenizer.eom_id:
			flush()
			lines.append(f'@measure {measure}')
			measure += 1
			continue
		tok = tokenizer.tokens[tid] if 0 <= tid < len(tokenizer.tokens) else '<unknown>'
		if tok in keywords:
			flush()
			current.append(tok)
		elif is_elapse(tok):
			# an elapse run continues an elapse-only line, but starts a new one after an event
			if current and all(is_elapse(t) for t in current):
				current.append(tok)
			else:
				flush()
				current.append(tok)
		else:
			# a field token belongs to the open line; with no line open (a fragment starting
			# mid-event) keep it rather than drop it, so the defect stays visible
			current.append(tok)
	flush()
	return lines


def count_note_on (ids, tokenizer):
	'''How many note_on events are in this id run.'''
	note_on = tokenizer.id_by_token.get('note_on')
	return sum(1 for t in ids if t == note_on)


# --- token <-> note_on mapping (for attention inspection) ----------------------------------

def note_on_events (ids, tokenizer, keywords, tick0=0, state=None, index0=0, order0=0, marks=None):
	'''Token ids -> the note_on events inside them, each carrying the index of its PITCH token.

	Each event is dict(pitch_index, pitch, channel, onset, order):

	  pitch_index  position IN `ids` of the `#XX` token, shifted by `index0`. This is what an attention
	               row is keyed on, so it is the bridge between a musical note and a matrix coordinate.
	  onset        absolute tick, accumulated from the leading E... runs exactly as
	               seq2CondPachifier.parse_events does. A stream that starts mid-piece therefore needs
	               its `tick0` from the caller, since the run before it is not in `ids`.

	A bare `#` token is NOT enough to identify a pitch: note_off carries one too (verified on the
	corpus, where note_on and note_off each own exactly half of them), and taking those for note_ons
	would put a phantom note on the plot for every real one. So the walk tracks which keyword opened
	the current event and accepts a `#` only under note_on. Channel is optional in the text (C0 is
	omitted), which is why it defaults rather than being required.

	RESUMABLE. Returns (events, abst, state), and accepts the `state` a previous call returned, so a
	stream can be walked in chunks with the same result as walking it whole. That matters because a chunk
	boundary can fall BETWEEN `note_on` and its `#XX` -- a fresh walk would start with no open keyword
	and silently drop that note. `index0`/`order0` shift the emitted indices and order numbers into the
	whole stream's coordinates. tests/midi/translate_midiseq2_check.py asserts chunked == whole at chunk
	sizes 1/2/3/17/256; the small ones are below event length, so they split every event in the corpus.

	Pass a list as `marks` to also collect `(token_index, tick)` for every `<eom>`. It rides along on
	THIS walk on purpose: a second pass over the same ids would have to re-accumulate the elapse runs,
	and two counters over one stream is one counter too many -- they only have to disagree once for the
	bar lines to drift away from the notes they are supposed to bound.
	'''
	events = []
	abst = tick0
	current, onset, channel = state if state is not None else (None, tick0, 0)
	vocab = tokenizer.tokens
	for i, tid in enumerate(ids):
		tok = vocab[tid] if 0 <= tid < len(vocab) else '<oob>'
		if is_elapse(tok):
			abst += int(tok[1:], 16)
			current = None		# an elapse run stands between events, so it closes the open one
			continue
		if tok.startswith('<'):
			# <eom> is a bar boundary and carries no time of its own; <bos>/<sep>/<eos> are structural
			if marks is not None and tok == '<eom>':
				marks.append((index0 + i, abst))
			current = None
			continue
		if tok in keywords:
			current, onset, channel = tok, abst, 0
			continue
		if current == 'note_on':
			if tok.startswith('C'):
				try:
					channel = int(tok[1:], 16)
				except ValueError:
					pass
			elif tok.startswith('#'):
				try:
					pitch = int(tok[1:], 16)
				except ValueError:
					continue
				events.append(dict(pitch_index=index0 + i, pitch=pitch, channel=channel, onset=onset,
					order=order0 + len(events)))
	return events, abst, (current, onset, channel)


def line_token_offsets (lines, tokenizer, eom=False):
	'''Prefix sums of per-line token counts: offsets[i] = tokens produced by lines[:i].

	`encode_lines` has no cross-line state (it concatenates each line's tokens), so encoding lines
	one at a time and concatenating gives exactly the same stream as encoding them together. That is
	what makes this prefix sum valid, and it is what lets a window's local token index be translated
	into an index in the whole file's stream — the source window is line-aligned, so a window that
	starts at line `c` starts at token offset `offsets[c]`.
	'''
	offsets = [0]
	for line in lines:
		offsets.append(offsets[-1] + len(encode_lines([line], tokenizer, eom)))
	return offsets


class AttentionInspector:
	'''Records, for every GENERATED note_on, how strongly it attended to each SOURCE note_on.

	Method. Attention is read from ONE extra forward pass per window rather than from every decoding
	step, and that is exact rather than an approximation: the stack is causal, so the attention row at
	query q depends only on tokens 0..q. Re-running the finished window [prefix ++ generated] and
	reading row q therefore reproduces bit-for-bit the row that was live when token q+1 was produced
	(verified to 7e-09, float32 epsilon). One pass per window instead of attentions on every token is
	roughly a 200x saving.

	Positions must be reproduced exactly, not merely plausibly. RoPE encodes relative offsets, so
	shifting every position by a constant is invisible (measured: identical to 7e-09) -- but this run
	uses pos_style 'sep', whose target half RESTARTS its numbering, and perturbing that changes the
	rows by 3e-03. So the inspection pass is handed the same `positions` list the generation step used.

	`output_attentions=True` also demands eager attention: under sdpa the flag is silently ignored and
	the model returns attentions=None with only a warning, which would otherwise look like a bug here
	rather than in the config.

	Reduction. For one generated pitch token and one source pitch token there are n_layer x n_head
	scores. The default takes the MAX over both, per the brief: a single head in a single layer
	pointing hard at a source note is the evidence of interest, and averaging would bury it under the
	many heads doing positional or local work.

	Memory. output_attentions materialises n_layer x n_head x T x T floats at once, which for this run
	(16 x 8, float32) is 0.78 GiB at the T=1281 that src_window 960 produces and 2.0 GiB at the 2048
	ceiling. Step 0 is the worst case, since it generates a whole window in one shot. The reduction
	slices out only the pitch-token rows and columns, so what is KEPT is small; the peak is set by the
	forward pass itself and scales as T squared. Large --src-window plus --inspect is the combination to
	watch.

	Query convention (--inspect-query). Causally, the row that CHOSE a pitch token belongs to the
	position BEFORE it -- the model was at `note_on` when it picked `#3c`. That is 'producer', the
	default, and it answers "which source note produced this note". 'self' instead reads the row at
	the pitch token itself, answering "what does this note look at once written". They are different
	questions and the distinction is easy to lose silently, so it is an explicit flag.

	Measured on one file (e366, src_window 960, 4 steps, 258 generated notes), they are the SAME signal
	traded off differently, not two findings:

		                        notes linked   pitch match   |order delta| <= 3
		producer                 188 (73%)         58%              53%
		self                      79 (31%)         81%              80%
		-- on the 73 notes both linked --
		producer                                   81%              75%
		self                                       82%              81%

	On shared notes they are indistinguishable and name the SAME source note 88% of the time. The whole
	difference is recall: producer links 115 further notes at 43% pitch / 39% order, well above the ~10%
	null but far below its own top band. So 'producer' is the default because the correspondence FIGURE
	wants coverage, and 'self' is the one to reach for when a clean, high-precision picture matters more.
	'''

	def __init__ (self, model, tokenizer, keywords, source_lines, source_eom, device,
		reduce='max', query='producer', threshold=0.05, top_k=8, plot_prefix=None):
		self.model = model
		self.tk = tokenizer
		self.keywords = keywords
		self.device = device
		self.reduce = reduce
		self.query = query
		self.threshold = threshold
		self.top_k = top_k
		# When set, each observe() draws its own step's figure before returning, so figures appear as the
		# run proceeds instead of after it. This is sound because `output` is APPEND-ONLY: a note's
		# absolute onset is fixed the moment its tokens are appended, and no later step can move it.
		self.plot_prefix = plot_prefix
		self.plot_paths = []

		# The whole source, once: global token stream, global note_on events, and the per-line offsets
		# that turn a window-local token index into a global one.
		self.src_ids_all = encode_lines(source_lines, tokenizer, source_eom)
		self.src_events, _, _ = note_on_events(self.src_ids_all, tokenizer, keywords)
		self.src_event_by_index = {e['pitch_index']: e for e in self.src_events}
		self.line_offsets = line_token_offsets(source_lines, tokenizer, source_eom)

		# Switch to eager ONCE, before any generation, not lazily at the first observe(): observe runs
		# after its step's generate, so a lazy switch would decode step 0 under sdpa and every later step
		# under eager. The two agree only to float epsilon, which is enough to flip a near-tie argmax, so
		# a mixed run could differ from both a pure sdpa run and a pure eager one. Inspecting therefore
		# costs speed for the whole run, and buys a run that is internally consistent.
		self._enable_eager()

		# (step, output pitch-token index, source pitch-token index, score), filled per step
		self.links = []
		# per-step context, so a step's figure can show the window it actually saw rather than the whole
		# piece: step -> dict(src_orders, out_indices, lines)
		self.windows = {}
		self.steps = 0
		self.attn_calls = 0

		# Incremental walk of the OUTPUT stream. Re-walking the whole stream every step would be
		# O(steps x total) and, worse, would make per-step plotting look like it needs the finished
		# stream. Instead the walk is resumed from where it stopped, carrying the open-keyword state so a
		# step boundary between `note_on` and its `#XX` loses nothing.
		self.out_events = []
		self.out_event_by_index = {}
		# Measure boundaries of the generated stream, (token index, tick). Collected on the SAME walk as the
		# notes, so a bar line can never drift away from the notes it bounds.
		self.out_eoms = []
		self.out_eom_tick = {}
		self._out_walked = 0			# tokens of `output` already consumed
		self._out_tick = 0
		self._out_state = None

	def _enable_eager (self):
		'''Force eager attention, or output_attentions returns None with only a warning.'''
		backbone = getattr(self.model, 'backbone', self.model)
		setter = getattr(backbone, 'set_attn_implementation', None)
		if setter is not None:
			setter('eager')
		else:					# older transformers: the config field is the only lever
			cfg = getattr(backbone, 'config', None)
			if cfg is not None:
				cfg._attn_implementation = 'eager'

	def _extend_output (self, output):
		'''Resume the output walk over whatever `output` has gained since the last call.'''
		if len(output) <= self._out_walked:
			return
		marks = []
		fresh, self._out_tick, self._out_state = note_on_events(
			output[self._out_walked:], self.tk, self.keywords,
			tick0=self._out_tick, state=self._out_state,
			index0=self._out_walked, order0=len(self.out_events), marks=marks)
		self.out_events.extend(fresh)
		for e in fresh:
			self.out_event_by_index[e['pitch_index']] = e
		self.out_eoms.extend(marks)
		for i, t in marks:
			self.out_eom_tick[i] = t
		self._out_walked = len(output)

	def _src_tick_at (self, line):
		'''Onset tick of the first source note at or after `line`, or None past the end.

		Used to place a window cut on the onset axis. A cut is a LINE index, and the axis is in ticks, so
		the cut is shown at the first note the next window will actually see -- which is what the boundary
		means musically.
		'''
		if line is None or line >= len(self.line_offsets):
			return None
		off = self.line_offsets[line]
		for e in self.src_events:
			if e['pitch_index'] >= off:
				return e['onset']
		return None

	def _out_tick_at (self, index):
		'''Onset tick of the first generated note at or after output token `index`, or None.'''
		if index is None:
			return None
		for e in self.out_events:
			if e['pitch_index'] >= index:
				return e['onset']
		return None

	@torch.no_grad()
	def observe (self, step, prefix, positions, new_ids, src_ids, cursor, next_cursor, head, out_base,
		output=None, prime_start=None, next_cursor_real=None, next_prime_start=None):
		'''One window: re-run [prefix ++ new_ids] with attentions, record its links, draw its figure.

		out_base is len(output) BEFORE new_ids were appended, so a generated token's index in the final
		output stream is out_base + (its offset within new_ids).

		`output` is the stream AFTER this step's tokens were appended. Passing it lets the step resolve
		its own notes' absolute onsets and plot immediately -- which is correct rather than merely early,
		because the stream is append-only and those onsets can never change.

		prime_start / next_cursor_real / next_prime_start are the sliding-window bookkeeping, recorded so
		the figure can mark where the neighbouring windows cut. They are optional: without them the figure
		simply carries no cut marks, so a caller that does not track them still gets a valid plot.
		'''
		if output is not None:
			self._extend_output(output)
		if not new_ids:
			return
		ids = list(prefix) + list(new_ids)
		pos = list(positions)
		while len(pos) < len(ids):			# generation continues the run by +1 per token
			pos.append(pos[-1] + 1)

		src_off = 1 if head else 0			# <bos> sits ahead of the source half on the first window
		src_lo, src_hi = src_off, src_off + len(src_ids)
		# Source pitch tokens are the attention KEYS we care about. Map each one to its global event.
		key_cols, key_events = [], []
		win_events, _, _ = note_on_events(src_ids, self.tk, self.keywords)
		for e in win_events:
			g = self.line_offsets[cursor] + e['pitch_index']
			ge = self.src_event_by_index.get(g)
			if ge is None:
				continue					# window boundary landed oddly; skip rather than guess
			key_cols.append(src_lo + e['pitch_index'])
			key_events.append(ge)
		if not key_cols:
			return

		# Generated pitch tokens are the QUERIES. Their indices are relative to new_ids, so shift into
		# both prefix coordinates (to index the attention matrix) and output coordinates (to name them).
		gen_events, _, _ = note_on_events(new_ids, self.tk, self.keywords)
		if not gen_events:
			return
		prefix_len = len(prefix)
		queries = []
		for e in gen_events:
			q = prefix_len + e['pitch_index']
			if self.query == 'producer':
				q -= 1						# the row that CHOSE this token sits one position earlier
			if q < 0 or q >= len(ids):
				continue
			queries.append((q, out_base + e['pitch_index'], 'gen'))
		# PRIMER pitch tokens are queries too. They are the tail of the prefix -- tokens [prime_start,
		# out_base) of `output` re-fed as this step's target context -- so their rows are already in this same
		# forward pass and reading them costs nothing but a few more sliced rows. What they answer is a
		# DIFFERENT question: not "which source note produced this note" (the model did not produce them here,
		# it was handed them) but "which source note is the model looking at while reading this context".
		# Recorded and drawn apart from the generated links for exactly that reason, and kept out of the
		# agreement statistics, which are about production.
		prime_lo = out_base if prime_start is None else prime_start
		prime_len = out_base - prime_lo
		# The primer occupies the prefix's tail by construction. Verified rather than assumed: if the caller's
		# bookkeeping and the prefix disagree, the indices would name the wrong tokens, and a plausible-looking
		# wrong link is worse than no primer links at all.
		prime_base = prefix_len - prime_len
		if (output is not None and prime_len > 0 and prime_base >= 0
				and list(prefix[prime_base:]) == list(output[prime_lo:out_base])):
			for e in self.out_events:
				if not (prime_lo <= e['pitch_index'] < out_base):
					continue
				q = prime_base + (e['pitch_index'] - prime_lo)
				if self.query == 'producer':
					q -= 1
				if q < 0 or q >= len(ids):
					continue
				queries.append((q, e['pitch_index'], 'prime'))
		if not queries:
			return

		window = torch.tensor([ids], dtype=torch.long, device=self.device)
		pos_t = torch.tensor([pos], dtype=torch.long, device=self.device)
		backbone = getattr(self.model, 'backbone', self.model)
		out = backbone(input_ids=window, attention_mask=None, position_ids=pos_t,
			output_attentions=True)
		attns = out.attentions
		if attns is None or attns[0] is None:
			raise RuntimeError('the backbone returned no attentions; eager attention could not be '
				'enabled, so --inspect cannot work on this transformers build')
		self.attn_calls += 1

		q_idx = torch.tensor([q for q, _, _ in queries], device=self.device)
		k_idx = torch.tensor(key_cols, device=self.device)
		# [layer, head, n_query, n_key] -> reduce over layer and head
		block = torch.stack([a[0][:, q_idx][:, :, k_idx] for a in attns])
		scores = (block.amax(dim=(0, 1)) if self.reduce == 'max'
			else block.mean(dim=(0, 1))).float().cpu()
		del attns, out, block

		for row, (q, out_index, kind) in enumerate(queries):
			vals = scores[row]
			keep = (vals >= self.threshold).nonzero(as_tuple=True)[0]
			if self.top_k and len(keep) > self.top_k:
				order = vals[keep].argsort(descending=True)[:self.top_k]
				keep = keep[order]
			if not len(keep):
				# Nothing cleared the threshold, so keep the ARGMAX alone. Every generated note then has
				# somewhere it looked most, and a note with no line means only that its window held no
				# source notes -- not that the threshold happened to cut it. The distinction matters for
				# reading the figures: dropping these notes silently made diffuse attention look like
				# absent attention, and diffuse-but-consistent is a different claim from nothing.
				# Flagged below threshold so it is never counted as evidence alongside a passing link.
				keep = vals.argmax().reshape(1)
			for c in keep.tolist():
				self.links.append((step, out_index, key_events[c]['pitch_index'], float(vals[c]), kind))
		# What this step could possibly have linked, recorded even when nothing passed the threshold: a
		# step whose window held source notes and produced notes but drew NO line is itself the finding,
		# and it would be invisible if only surviving links were kept.
		# The primer is the tail of PREVIOUSLY generated output handed back as this step's target prefix:
		# tokens [prime_start, out_base). Recorded separately from the generated notes because the model
		# did not choose it this step -- it is context, and showing it as generated would overstate what
		# the step produced.
		# Cut marks. On the source lane the PREVIOUS window's own end is used, not this window's start:
		# they are the same boundary only in the sense that one follows the other, and this window's start
		# is by construction the leftmost note drawn, so marking it would put a line at x=0 that says
		# nothing. The previous window's END falls INSIDE this one -- that is the overlap the sliding
		# window depends on, it is worth seeing, and the figure also PINS the output lane to it.
		prev = self.windows.get(step - 1)
		self.windows[step] = dict(
			src_orders=[e['order'] for e in key_events],
			out_indices=[i for _, i, k in queries if k == 'gen'],
			prime_indices=[e['pitch_index'] for e in self.out_events
				if prime_lo <= e['pitch_index'] < out_base],
			# Measure boundaries inside what this step DRAWS -- primer included, since the primer is drawn.
			# Indices, not ticks: resolve_step turns them into ticks, so the streaming and deferred paths go
			# through one conversion instead of two.
			eom_indices=[i for i, _ in self.out_eoms if prime_lo <= i < len(output or ())],
			cuts=dict(src_prev_line=(prev['lines'][1] if prev else None),
				src_next_line=next_cursor_real,
				# only when something actually rolled out: on the first step the view starts at 0 with no
				# primer, and a mark there would invent a slide that never happened
				out_prev=(prime_start if prime_start is not None and prime_start < out_base else None),
				out_next=next_prime_start),
			lines=(cursor, next_cursor))
		self.steps += 1
		if self.plot_prefix:
			self.plot_step(step)

	def _pairs (self, links):
		'''Turn recorded (step, out_index, src_index, score) links into resolved, score-sorted pairs.'''
		pairs = []
		for step, out_index, src_index, score, kind in links:
			oe = self.out_event_by_index.get(out_index)
			se = self.src_event_by_index.get(src_index)
			if oe is None or se is None:
				continue
			pairs.append(dict(step=step, score=score, kind=kind,
				out_onset=oe['onset'], out_pitch=oe['pitch'], out_order=oe['order'],
				src_onset=se['onset'], src_pitch=se['pitch'], src_order=se['order']))
		pairs.sort(key=lambda p: -p['score'])
		return pairs

	def resolve_step (self, step):
		'''One step's resolved links and window. Needs only what that step already appended.'''
		w = self.windows.get(step)
		if w is None:
			return [], None
		c = w.get('cuts') or {}
		window = dict(lines=w['lines'],
			src=[self.src_events[o] for o in w['src_orders'] if o < len(self.src_events)],
			out=[self.out_event_by_index[i] for i in w['out_indices']
				if i in self.out_event_by_index],
			prime=[self.out_event_by_index[i] for i in w.get('prime_indices', ())
				if i in self.out_event_by_index],
			eoms=[self.out_eom_tick[i] for i in w.get('eom_indices', ())
				if i in self.out_eom_tick],
			# resolved to ticks here rather than in the plot, so the figure stays a pure function of the
			# window and the deferred path draws exactly what the streaming path drew
			cuts=dict(
				src_prev=self._src_tick_at(c.get('src_prev_line')),
				src_next=self._src_tick_at(c.get('src_next_line')),
				out_prev=self._out_tick_at(c.get('out_prev')),
				out_next=self._out_tick_at(c.get('out_next'))))
		return self._pairs([l for l in self.links if l[0] == step]), window

	def plot_step (self, step):
		'''Draw and report one step's figure. Called from observe() during the run.'''
		pairs, window = self.resolve_step(step)
		if window is None or (not pairs and not window['out']):
			return None
		path = f'{self.plot_prefix}.step{step:03d}.png'
		os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
		plot_attention_step(pairs, window, step, path, self.threshold)
		self.plot_paths.append(path)
		gen = [p for p in pairs if p.get('kind', 'gen') == 'gen']
		weak = sum(1 for p in gen if p['score'] < self.threshold)
		print(f'[attn] step {step}: {len(gen) - weak} link(s) >= {self.threshold}'
			f'{f" + {weak} fallback" if weak else ""}'
			f'{f" + {len(pairs) - len(gen)} primer" if len(pairs) > len(gen) else ""} over '
			f'{len(window["src"])} src / {len(window["out"])} gen notes  {span_note(window)}  '
			f'-> {os.path.basename(path)}')
		return path

	def resolve_all (self, output=None):
		'''Everything, for the end-of-run report. `output` only tops up the incremental walk.'''
		if output is not None:
			self._extend_output(output)
		windows = {}
		for step in self.windows:
			_, w = self.resolve_step(step)
			if w is not None:
				windows[step] = w
		return self._pairs(self.links), self.out_events, self.src_events, windows


# --- model ---------------------------------------------------------------------------------

def resolve_checkpoint (run, config, explicit=None):
	'''Pick the weights file. config['best'] first, then best.chkpt, then latest.chkpt.

	Order matters: save_mode 'best' records the winning snapshot's NAME in config['best'] (e.g.
	model_167_loss_1.517e-01.chkpt) and prunes the others, and some run layouts have no best.chkpt at
	all — so defaulting straight to best.chkpt would fail against a freshly trained run. latest.chkpt
	is last because it is whatever epoch finished most recently, not the best one.
	'''
	if explicit:
		return explicit
	best = config['best']
	if best:
		path = os.path.join(run, best)
		if os.path.exists(path):
			return path
	for name in ('best.chkpt', 'latest.chkpt'):
		path = os.path.join(run, name)
		if os.path.exists(path):
			return path
	raise FileNotFoundError(f'no checkpoint found in {run}')


def resolve_tokenizer (run, config):
	'''Load the vocabulary the CHECKPOINT was trained against, preferring the run's own copy.

	The vocabulary is positional: every id in the checkpoint's embedding rows means whatever row it
	sat on during training. A run created from a config with `assets: [Midiseq2Vocab]` pins that
	mapping beside its weights, so the run directory — not the current checkout — is the authority.
	Order: the run-local copy, then whatever `data.args.vocab_path` names, then the repository asset
	(with a warning, since nothing then ties the ids to the checkpoint).
	'''
	local = os.path.join(run, Midiseq2Vocab.FILENAME)
	if os.path.isfile(local):
		return Midiseq2Tokenizer(local), local

	configured = (config['data.args'] or {}).get('vocab_path')
	if configured:
		path = configured if os.path.isabs(configured) else os.path.join(run, configured)
		if UnifiedSeq2Tokenizer.matches(path):
			# A mixed Lilylet/midiseq2 run: its content ids are offset and one half is not midiseq2 at
			# all, so this script's renderer would emit nonsense rather than fail.
			raise ValueError(f'{path} is a mixed unified vocabulary; this tool translates '
				f'midiseq2 -> midiseq2 only')
		return Midiseq2Tokenizer(path), path

	print(f'[warn] {run} pins no vocabulary; falling back to the repository asset. If it has been '
		f'edited since training, the rendered tokens will be wrong.')
	return Midiseq2Tokenizer(), None


def load_model (run, checkpoint, device):
	'''Build the bare MidiTranslator from the run's .state.yaml and load weights (eval mode).

	No postfix='Loss' — the training wrapper (MidiTranslatorLoss) adds the loss and the per-type
	metric buffers, none of which inference needs, and the checkpoint holds only `deducer`'s
	state_dict anyway. strict=False because the wrapper's persistent=False buffers are absent from
	the blob by design.
	'''
	config = Configuration.createOrLoad(run, volatile=True)
	model = loadModel(config['model'], imports=config['imports'])
	blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
	state = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
	missing, unexpected = model.load_state_dict(state, strict=False)
	if missing or unexpected:
		print(f'[load] {len(missing)} missing, {len(unexpected)} unexpected keys')
	epoch = blob.get('epoch') if isinstance(blob, dict) else None
	print(f'[load] {os.path.basename(checkpoint)}' + (f' (epoch {epoch})' if epoch is not None else ''))
	return config, model.to(device).eval()


# --- positions (mirrors Seq2Seq2._positions) ----------------------------------------------

def positions_for (pos_style, n_source, n_target):
	'''RoPE positions for a `n_source ++ <sep> ++ n_target` layout. Length n_source + 1 + n_target.

	  'flat'   0, 1, 2, ... T-1
	  'sep'    source ends at -2, <sep> = -1, target starts at 0

	The two are equivalent to the model — both are one arithmetic run, and RoPE reads only relative
	distance, so a uniform shift is invisible. Continuation just keeps counting up from the last value,
	which is why generation can extend either style by +1 per token.

	'absolute' is refused by the caller: it places each half on its own FILE's token axis, which a
	sliding window does not define (the output file's token count is not known until we are done).
	'''
	total = n_source + 1 + n_target
	if pos_style == 'flat':
		return list(range(total))
	if pos_style == 'sep':
		return list(range(-(n_source + 1), n_target))
	raise ValueError(f'unsupported pos_style {pos_style!r} for sliding-window inference')


class SlidingTranslator:
	'''Translates one midiseq2 file by sliding a window across it.

	Holds no per-file state: `translate` owns the cursors, so one instance can process many files.
	'''

	def __init__ (self, model, tokenizer, pos_style='sep', src_window=640, max_token=2048,
		device='cpu', prime=True, temperature=0.0, top_k=0, top_p=1.0, source_eom=False,
		advance_tokens=1, prime_window=2048):
		self.model = model
		self.tk = tokenizer
		self.pos_style = pos_style
		self.src_window = src_window
		self.max_token = max_token
		self.device = torch.device(device)
		self.prime = prime
		self.advance_tokens = max(1, int(advance_tokens))
		# Safety ceiling on the target-half view. This is separate from the user-facing stride: the model can
		# generate several measures in one step, so the retained primer may grow beyond its trained context.
		# The ceiling trims that accidental growth; it does not determine how many measures we intentionally
		# advance per step.
		self.prime_window = prime_window
		# mirrors the feeder's source_eom: whether @measure becomes <eom> on the SOURCE half
		self.source_eom = source_eom
		# temperature 0 = greedy argmax (reproducible); anything else goes through sample_next
		self.temperature = temperature
		self.top_k = top_k
		self.top_p = top_p
		self.keywords = keyword_tokens(tokenizer)

	# --- one step ------------------------------------------------------------------------

	def source_window (self, lines, cursor):
		'''Lines from `cursor` whose ids fit in src_window. Returns (ids, next_line_after_window).

		Stops on a whole line — a half-encoded event would be a token sequence the model never saw.
		Always takes at least one content line, so a single line longer than the budget cannot stall
		the loop.
		'''
		ids = []
		index = cursor
		while index < len(lines):
			chunk = encode_lines([lines[index]], self.tk, self.source_eom)
			if ids and len(ids) + len(chunk) > self.src_window:
				break
			ids.extend(chunk)
			index += 1
		return ids, index

	def build_prefix (self, src_ids, prime_ids, head):
		'''Assemble the prefix and its positions, matching Seq2Seq2._assemble.

			[<bos> if head] src... <sep> [<bos> if head] prime...

		<bos> goes on BOTH halves iff the crop reaches the start of the piece (seq2seq2.py:507) — so
		only on the first window. <eos> is never in a prefix: it is what generation produces to stop.
		'''
		source = ([self.tk.bos_id] if head else []) + src_ids
		target = ([self.tk.bos_id] if head else []) + list(prime_ids)
		ids = source + [self.tk.sep_id] + target
		# positions_for lays out the full n_source+1+n_target run; the prefix is that run truncated to
		# what we actually have, and generation continues it by +1 per token.
		positions = positions_for(self.pos_style, len(source), len(target))
		return ids, positions, len(source)

	@torch.no_grad()
	def generate (self, prefix_ids, prefix_positions, temperature, top_k, top_p):
		'''Autoregressive continuation to <eos> or the max_token ceiling. Returns (new ids, eos_forced).

		The <eos> is NOT included in the return: it terminates this window, but the output stream is
		one continuous piece of music, so an <eos> in the middle of it would be a stray token.

		The FIRST token of a step has <eos> masked out of the logits, so every step emits at least one
		token. Without it a step could return nothing at all, and an empty step is the one thing this
		loop cannot use: it has no <eom> to advance past and no onsets to move the source cursor by, so
		the run leans entirely on the anti-stall fallbacks to make any progress.

		Reported, not hidden. The mask changes what the model was going to do, so `eos_forced` says it
		happened and the caller counts it. A step that wanted <eos> immediately is usually the primer
		having grown outside the trained band (see trim_prime), and masking makes the run continue rather
		than making that condition go away -- so a high forced count is evidence to act on, not a
		success. The mask applies ONLY to the first position; from the second token on, <eos> ends the
		window normally.
		'''
		ids = list(prefix_ids)
		positions = list(prefix_positions)
		out = []
		forced = False
		while len(ids) < self.max_token:
			window = torch.tensor([ids], dtype=torch.long, device=self.device)
			pos = torch.tensor([positions], dtype=torch.long, device=self.device)
			logits = self.model(window, None, pos)[0, -1, :]
			if not out:
				# -inf, not a small penalty: the observed degenerate case had <eos> at logit 13.16
				# against 7.71 for next-best, which any finite margin would have to be tuned against.
				# Cloned so the mask cannot leak into anything holding this tensor.
				logits = logits.clone()
				forced = bool(logits.argmax().item() == self.tk.eos_id)
				logits[self.tk.eos_id] = float('-inf')
			nxt = (int(logits.argmax().item()) if not temperature
				else sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p))
			if nxt == self.tk.eos_id:
				break
			out.append(nxt)
			ids.append(nxt)
			positions.append(positions[-1] + 1)
		return out, forced

	# --- advancing -----------------------------------------------------------------------

	def advance_output (self, output, prime_start):
		'''Move the view at least `advance_tokens` target tokens, then round upward to the next `<eom>`.

		The threshold is measured only in the generated target stream: first choose `prime_start +
		advance_tokens`, then scan forward to the first complete measure boundary and return the token
		immediately after that `<eom>`. Thus actual movement is never less than requested unless output ends.
		If no boundary exists after the threshold, advance to the available threshold; an empty view does not
		move.
		'''
		span = len(output) - prime_start
		if span <= 0:
			return prime_start
		threshold = min(len(output), prime_start + self.advance_tokens)
		for i in range(threshold, len(output)):
			if output[i] == self.tk.eom_id:
				return i + 1
		# No complete boundary after the requested threshold: move by the available token distance.
		return threshold

	def trim_prime (self, output, prime_start):
		'''Apply the internal target-view safety ceiling, independently of the public stride.

		Necessary because one step can generate far more than the requested stride while advance_output rolls
		out exactly one, so the view GROWS. Left alone it walks out of the trained distribution and the
		model stops: observed at prime 432 (trained target halves are median 130, p99 351, max 468),
		where the model emitted <eos> immediately with logit 13.16 against 7.71 for the next-best token
		— from its point of view a target half that long is simply finished. Two consecutive steps then
		generated nothing.

		Trimming moves by whole measures so the view still begins on a bar boundary. A single measure
		longer than the budget cannot be split that way, so it is cut at the token level rather than
		left to stall the run.
		'''
		limit = self.prime_window
		if not limit:
			return prime_start
		while len(output) - prime_start > limit:
			nxt = None
			for i in range(prime_start, len(output)):
				if output[i] == self.tk.eom_id:
					nxt = i + 1
					break
			if nxt is None or nxt == prime_start:
				return len(output) - limit
			prime_start = nxt
		return prime_start

	def finished (self, output, src_ids):
		'''True when the piece is over: the source window has run out of music and the output ends it.

		Both halves have to agree, so an early spurious end_of_track from the model cannot truncate a
		file that still has source left. `end_of_track` is the corpus's own terminator (every file ends
		with one), which is why the model emits <eos> right after producing it.
		'''
		eot = self.tk.id_by_token.get('end_of_track')
		return eot in src_ids and eot in output[-8:]

	def advance_source_by_onsets (self, lines, cursor, onsets):
		'''Advance the source cursor past `onsets` note_on lines. Returns the new cursor.

		This is the fallback matching scheme: the production source carries no @measure, so there is no
		mark key to look up, and note_on count is the one quantity both arms nominally share. They do
		not share it exactly — across this corpus only 3 of 99 files have equal note_on counts (median
		relative delta 2.1%, max 27.4%) — so the cursor drifts, and translate() reports it. Matching on
		onset PITCH values would be tighter; swap this function, the loop needs no change.

		Stops after the last matched note_on rather than consuming the trailing note_off/elapse tail:
		those belong to the next window's context, and dropping them would lose events.
		'''
		if onsets <= 0:
			return cursor
		seen = 0
		index = cursor
		while index < len(lines):
			if lines[index].startswith('note_on'):
				seen += 1
				if seen >= onsets:
					return index + 1
			index += 1
		return len(lines)

	# --- whole file ----------------------------------------------------------------------

	def translate (self, lines, verbose=False, max_steps=0, inspector=None):
		'''Slide across `lines`, returning (output_ids, stats).

		The invariant that makes this terminate: every step either advances src_cursor or, at EOF,
		breaks. prime_start advances by at least one token per step while the view is non-empty.
		'''
		output = []
		prime_start = 0
		cursor = 0
		step = 0
		stalls = 0
		forced = 0			# steps whose first token was <eos> before the mask removed it
		done = False
		start_time = time.time()

		while cursor < len(lines):
			if max_steps and step >= max_steps:
				break
			src_ids, next_cursor = self.source_window(lines, cursor)
			if not src_ids:
				break
			prime_ids = output[prime_start:] if self.prime else []
			prefix, positions, _ = self.build_prefix(src_ids, prime_ids, head=(step == 0))
			if len(prefix) >= self.max_token:
				print(f'[warn] step {step}: prefix {len(prefix)} >= max_token {self.max_token}, '
					f'nothing left to generate; stopping')
				break

			new_ids, eos_forced = self.generate(prefix, positions, self.temperature, self.top_k,
				self.top_p)
			out_base = len(output)		# before the extend: maps a new_ids offset onto the output stream
			output.extend(new_ids)
			step_prime_start = prime_start
			forced += eos_forced
			# An immediate <eos> is the RIGHT answer once the piece is over: the source window ends in
			# end_of_track and the output already carries one, so there is nothing left to translate.
			# Stopping here (rather than grinding through the remaining source lines producing nothing) is
			# what makes the run end cleanly.
			#
			# The first-token mask means new_ids is never empty, so this can no longer be reached by
			# testing for an empty step -- it is keyed on the model HAVING WANTED <eos> instead. Without
			# this the end-of-piece stop would be unreachable and every finished run would grind to EOF
			# emitting whatever the mask forced out of it.
			if eos_forced:
				if self.finished(output, src_ids):
					done = True
					break
				# not finished: the model wanted to stop early and was overridden. Counted as a stall
				# because that is what it is -- a step that produced only because it was made to.
				stalls += 1

			before = prime_start
			prime_start = self.advance_output(output, prime_start)
			# A step can generate several measures while the requested stride rolls out only some of them;
			# the internal safety ceiling may retire additional old measures to keep the primer in-band.
			prime_start = self.trim_prime(output, prime_start)
			rolled = output[before:prime_start]
			onsets = count_note_on(rolled, self.tk)
			src_before = cursor
			cursor = self.advance_source_by_onsets(lines, cursor, onsets)
			# a step that consumed no source line would repeat the same window forever
			if cursor <= src_before:
				cursor = min(next_cursor, len(lines)) if next_cursor > src_before else src_before + 1

			if inspector is not None:
				# Observed after the advance, not before it: the figure marks where the NEXT window cuts,
				# and that is only known once the roll-out has been computed. Still inside the same step, so
				# the figure is still written the moment the step is over.
				inspector.observe(step, prefix, positions, new_ids, src_ids, src_before, next_cursor,
					step == 0, out_base, output=output, prime_start=step_prime_start,
					next_cursor_real=cursor, next_prime_start=prime_start)

			if verbose:
				print(f'  step {step:4d}  src[{src_before}:{cursor}] {len(src_ids):5d} tok  '
					f'prime {len(prime_ids):5d}  gen {len(new_ids):5d}  '
					f'rolled {len(rolled):4d} tok / {onsets:3d} onsets  '
					f'out {len(output):7d}  T {len(prefix) + len(new_ids):5d}'
					f'{"  <eos> forced" if eos_forced else ""}')
			step += 1

		stats = dict(steps=step, output_tokens=len(output), source_lines=len(lines),
			consumed_lines=cursor, stalls=stalls, eos_forced=forced, done=done,
			elapsed=time.time() - start_time)
		return output, stats


# --- output file ---------------------------------------------------------------------------

HEADER_KEYWORDS = ('ticks_per_beat', 'format_type')


def source_header (lines):
	'''The source's leading header lines (ticks_per_beat / format_type).

	A windowed fragment parses without them — the grammar's start rule accepts any statement sequence
	and Builder defaults to 480 TPB — but it would then decode at whatever the default happens to be.
	Carrying the source's own header over keeps the output on the same tick grid as its input.
	'''
	out = []
	for line in lines:
		if line.split()[:1] and line.split()[0] in HEADER_KEYWORDS:
			out.append(line)
		elif out:
			break
	return out


def compose_output (body_lines, fallback_header):
	'''Final file lines: header, `@measure 1`, then the body.

	The first window is a head crop, so the model generates the file's own `ticks_per_beat` /
	`format_type` lines itself — prepending the source's unconditionally duplicated them (observed:
	the header appeared twice, with `@measure 1` stranded between the two copies). So the generated
	header is used when present, and the source's serves only as a fallback for a run that produced
	none (e.g. --no-prime resuming mid-piece, where the model has no reason to emit one).

	`@measure 1` is written explicitly because @measure 1 emits no <eom> by design, so the opening bar
	has no directive of its own in the token stream; it goes AFTER the header so the file reads in the
	same order the corpus does.
	'''
	head = source_header(body_lines)
	rest = body_lines[len(head):]
	if not head:
		head = fallback_header
	return head + ['@measure 1'] + rest


def write_output (path, lines):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		for line in lines:
			f.write(line + '\n')


def report_output (body_lines, stats):
	'''Post-checks that catch the failure modes worth catching: off-vocab tokens and broken numbering.'''
	measures = [int(l.split()[1]) for l in body_lines if l.startswith('@measure')]
	unknown = sum(l.split().count('<unknown>') for l in body_lines)
	print(f'[out] {len(body_lines)} lines, {stats["output_tokens"]} tokens, '
		f'{len(measures)} measures, {stats["steps"]} steps, {stats["elapsed"]:.1f}s')
	if unknown:
		print(f'[warn] {unknown} <unknown> token(s) in output — a rendering or vocab bug')
	# numbering starts at the explicit `@measure 1` and every <eom> adds the next bar
	if measures and measures != list(range(1, 1 + len(measures))):
		print(f'[warn] measure numbers not contiguous from 1: {measures[:12]}...')
	if stats['stalls']:
		print(f'[warn] {stats["stalls"]} step(s) wanted <eos> mid-piece and were overridden by the '
			f'first-token mask')
	if stats.get('eos_forced'):
		# Includes the final end-of-piece step, which is legitimate; stalls counts only the mid-piece
		# ones. The gap between the two numbers is how many overrides were the run ending normally.
		print(f'[info] first-token <eos> mask fired on {stats["eos_forced"]} step(s) '
			f'({stats["stalls"]} mid-piece)')
	consumed, total = stats['consumed_lines'], stats['source_lines']
	# reaching end_of_track before the last source line is normal: the tail of a source file is its own
	# note_off/end_of_track run, which the model translates in one window rather than one per line.
	if consumed < total and not stats['done']:
		print(f'[warn] stopped after {consumed}/{total} source lines')


def attention_null (pairs, best, draws=200, seed=0):
	'''What the agreement numbers would be if attention carried NO correspondence information.

	Without this the observed rates are unreadable. "58% of best links have matching pitch" sounds
	middling until you know that drawing a random source note from the same step's candidate pool gives
	10%. The pool is deliberately restricted to notes that were actually in that step's attention keys,
	so the null is "attention points somewhere in the window it could see", not "anywhere in the piece"
	-- the weaker, and therefore harder to beat, baseline.
	'''
	import random

	pool = {}
	for p in pairs:
		pool.setdefault(p['step'], set()).add((p['src_order'], p['src_pitch']))
	pool = {k: sorted(v) for k, v in pool.items()}
	rng = random.Random(seed)
	pm, oa, n = 0, 0, 0
	for _ in range(draws):
		for p in best:
			cand = pool.get(p['step'])
			if not cand:
				continue
			order, pitch = rng.choice(cand)
			n += 1
			pm += pitch == p['out_pitch']
			oa += abs(p['out_order'] - order) <= 3
	return (100 * pm / n, 100 * oa / n) if n else (0.0, 0.0)


def report_attention (pairs, out_events, src_events, threshold=None, top=20):
	'''The ranking, plus the numbers that say whether it means anything.

	A ranking alone is easy to over-read: the top rows always look like alignment because the top rows
	of any monotone-ish attention will. Three things are reported underneath, in increasing order of how
	much they tell you:

	1. Coverage -- how many generated notes cleared the threshold. Every note also keeps its single best
	   link as a FALLBACK when nothing cleared it, so "linked at all" is ~100% by construction and says
	   nothing; the number that carries information is how many passed, reported separately from the
	   fallbacks. The agreement statistics below are computed on PASSING links only, so a run's headline
	   numbers stay comparable to one taken before fallbacks existed; the fallback subset is scored on its
	   own line, where a rate near the null is the expected and honest result.
	2. Agreement against a null (attention_null), as a multiple. Pitch match and ORDER delta are the
	   usable measures. ONSET delta is reported but is NOT evidence about attention: the irregular
	   source carries rubato while the score output is quantized, so the two streams are already offset
	   by a few hundred ticks before attention enters. Measured on one file, the attention's median
	   onset delta was +377 and a plain gen#i<->src#i identity mapping over the same 117 notes gave
	   +246, so most of that number is the streams, not the model.
	3. Dose-response -- agreement binned by score. This is the strongest of the three: if attention
	   encodes correspondence, a HIGHER score must mean a MORE correct link, and a flat profile falsifies
	   that even when the overall rate looks good. Measured: 75% pitch match in the top bin falling
	   monotonically to 43% in the weakest.
	'''
	if not pairs:
		print('[attn] no links at all — no step had both source and generated notes')
		return
	# Primer links are excluded from every number below. They are rows read while INGESTING carried-in
	# context, not rows that produced a note in this step, and a note that appears as primer in step n was
	# already scored as generated in step n-1 -- folding them in would count the same note twice under two
	# different questions. They are drawn on the figures and counted in its title, and nowhere else.
	n_prime_links = sum(1 for p in pairs if p.get('kind', 'gen') == 'prime')
	pairs = [p for p in pairs if p.get('kind', 'gen') == 'gen']
	if n_prime_links:
		print(f'[attn] {n_prime_links} primer-context link(s) drawn on the figures, excluded from the '
			f'statistics below (they are context reads, not productions)')
	if not pairs:
		print('[attn] no generated-note links at all')
		return
	thr = 0.0 if threshold is None else threshold
	passing = [p for p in pairs if p['score'] >= thr]
	print(f'[attn] {len(passing)} link(s) >= {thr} over {len(out_events)} generated / '
		f'{len(src_events)} source note_on ({len(pairs) - len(passing)} sub-threshold fallback link(s))')

	# best link per note, split by whether it cleared the bar. Fallbacks are kept out of the headline
	# numbers: they are every note's argmax regardless of strength, so folding them in would dilute the
	# measured rates toward the null and make a run look worse for a reason that is not about the model.
	all_best = {}
	for p in pairs:					# pairs are score-sorted, so the first per note is its best
		all_best.setdefault(p['out_order'], p)
	best = [p for p in all_best.values() if p['score'] >= thr]
	fallback = [p for p in all_best.values() if p['score'] < thr]
	print(f'[attn] {len(best)}/{len(out_events)} generated notes have a link >= {thr} '
		f'({100 * len(best) / max(1, len(out_events)):.0f}%); '
		f'{len(fallback)} carry only a sub-threshold best link')
	if not best:
		print(f'[attn] nothing cleared {thr} — lower --inspect-threshold; '
			f'the figures still show each note\'s best link')
		return
	n = max(1, len(best))
	same_pitch = sum(1 for p in best if p['out_pitch'] == p['src_pitch'])
	near_order = sum(1 for p in best if abs(p['out_order'] - p['src_order']) <= 3)
	pm_null, oa_null = attention_null(pairs, best)
	pm, oa = 100 * same_pitch / n, 100 * near_order / n
	print(f'[attn] best link: pitch match {same_pitch}/{n} ({pm:.0f}%, null {pm_null:.0f}% '
		f'-> {pm / max(pm_null, 1e-9):.1f}x), '
		f'|order delta| <= 3 on {near_order}/{n} ({oa:.0f}%, null {oa_null:.0f}% '
		f'-> {oa / max(oa_null, 1e-9):.1f}x)')
	orders = sorted(p['out_order'] - p['src_order'] for p in best)
	print(f'[attn] order delta of best link: median {orders[len(orders) // 2]:+d}, '
		f'p10 {orders[len(orders) // 10]:+d}, p90 {orders[9 * len(orders) // 10]:+d}')
	deltas = sorted(p['out_onset'] - p['src_onset'] for p in best)
	if deltas:
		mid = deltas[len(deltas) // 2]
		# reported for completeness, but the two streams are offset before attention enters -- see the
		# docstring; do not read this as an attention measurement
		print(f'[attn] onset delta of best link: median {mid:+d} ticks, '
			f'range {deltas[0]:+d}..{deltas[-1]:+d} (streams are pre-offset; not attention evidence)')
	if fallback:
		# Scored separately, and expected to sit near the null: these are argmaxes of diffuse rows. If they
		# instead scored well, that would be the finding -- it would mean the threshold is discarding real
		# correspondence, and the bar is set too high.
		fp = 100 * sum(1 for p in fallback if p['out_pitch'] == p['src_pitch']) / len(fallback)
		fo = 100 * sum(1 for p in fallback
			if abs(p['out_order'] - p['src_order']) <= 3) / len(fallback)
		print(f'[attn] sub-threshold fallbacks ({len(fallback)} note(s), not counted above): '
			f'pitch match {fp:.0f}%, |order delta| <= 3 on {fo:.0f}% '
			f'(null {pm_null:.0f}%/{oa_null:.0f}% — near the null is the expected result)')
	print('[attn] dose-response — agreement by score bin (must FALL with score to mean anything):')
	print('       score bin       n   pitch  |order|<=3')
	# spans the whole range, fallbacks included, so the profile can be read continuously through the
	# threshold rather than stopping at it
	bins = ((0.15, 1.01), (0.10, 0.15), (0.08, 0.10), (0.06, 0.08), (0.04, 0.06), (0.02, 0.04),
		(0.0, 0.02))
	for lo, hi in bins:
		g = [p for p in all_best.values() if lo <= p['score'] < hi]
		if not g:
			continue
		gp = 100 * sum(1 for p in g if p['out_pitch'] == p['src_pitch']) / len(g)
		go = 100 * sum(1 for p in g if abs(p['out_order'] - p['src_order']) <= 3) / len(g)
		mark = '' if lo >= thr else '  (fallback)'
		print(f'       [{lo:.2f},{min(hi, 1.0):.2f})  {len(g):5d}   {gp:4.0f}%     {go:4.0f}%{mark}')
	print(f'[attn] top {min(top, len(pairs))} links by score:')
	print('       score  step  gen#  onset  pitch    src#  onset  pitch   dt')
	for p in pairs[:top]:
		print(f'       {p["score"]:.3f}  {p["step"]:4d}  {p["out_order"]:4d} {p["out_onset"]:6d}  '
			f'{p["out_pitch"]:5d}    {p["src_order"]:4d} {p["src_onset"]:6d}  {p["src_pitch"]:5d}  '
			f'{p["out_onset"] - p["src_onset"]:+5d}')


SRC_COLOR = '#2f5f9f'
OUT_COLOR = '#a5432f'
PRIME_LINK_COLOR = '#4a7c59'
# Floor for link opacity. Weight is normalised within the step, so a step with one dominant link would
# otherwise render the rest at an alpha that rounds to invisible -- and a link that is counted in the title
# but cannot be seen is a figure disagreeing with its own caption.
MIN_LINK_ALPHA = 0.1


def plot_attention_step (pairs, window, step, path, threshold, subtitle=''):
	'''One sliding-window step: PITCH against onset, both sequences on one pitch axis.

	One figure PER STEP rather than one for the file, because a step is the unit the model actually
	works in. Superimposing all steps would overlay windows that never saw each other and let a late
	window's links cross a region an early window could not reach, which reads as long-range attention
	that no forward pass ever performed. Only this step's window is drawn -- the source notes that were
	in its attention keys and the notes it generated -- so the plotted extent IS the step's reach.

	Layout. TWO STACKED PANELS, each sequence its own: source above, generated below. Every axis is per
	sequence -- x is onset normalised to that sequence's own tick extent (source ticks labelled along the
	top edge, generated along the bottom), y is pitch with its own scale on each panel. Links are drawn
	across the panel boundary, so a link's geometry is the diagnostic:

		vertical       same pitch AND same relative position -- the corresponding note
		tilted         relative position disagrees; the tilt is how far
		ends at
		different
		heights        pitch disagrees

	The two pitch axes deliberately share one RANGE (the union of both sequences, padded) even though
	they are drawn and ticked separately. Independent ranges would rescale each panel to its own tessitura
	and a same-pitch link would stop being vertical, which is the one thing the geometry is for.

	The output lane also carries the PRIMER -- the tail of earlier output fed back as this step's target
	prefix -- as hollow squares against the filled triangles of what this step generated. Both carry links,
	but they are not the same claim and are not drawn alike: a solid dark line to a triangle says "this is
	what the model looked at while producing this note", while a dashed green line to a square says "this is
	what it looked at while reading this carried-in context". Only the first is production, so only the first
	enters the agreement statistics.

	Thin olive verticals on the output lane are `<eom>`, the model's own bar lines. They make a horizontal
	displacement musically readable: a link landing a bar late and one landing a beat late are different
	errors, and on a bare onset axis both merely look shifted.

	Both lanes carry dashed window cuts, and each names a specific boundary:

		source  prev    where the PREVIOUS window ended -- it falls inside this one, and the material left
		                of it is the overlap the two windows share. Also the ANCHOR: see the x-axis note
		                below, where the output lane is pinned to it
		source  next    where the NEXT window starts; everything right of it is about to be re-read
		output  prev    where this step's primer began, i.e. what the previous slide rolled into view
		output  next    where the next step's view starts; generated notes left of it will have rolled out

	A boundary that does not exist (no previous window on step 0, no next at the last step) draws no line
	rather than a line at the edge, so a mark is always a real slide.

	How the two lanes are put on one x axis. The source lane always spans 0..1 over its own window. The
	output lane is then placed by TWO PINS, which is exactly enough to fix an affine map -- offset and scale
	both, with nothing arbitrary left to choose:

		left pin    the leftmost note the output lane DRAWS -- the primer's first, when there is a primer --
		            lands on x=0, where the source window's first note is
		cut pin     the step's FIRST GENERATED note lands on the source lane's `prev cut`, the position the
		            previous window ended at and where this step's generation picks up

	Both are real correspondences rather than conveniences. The primer IS this step's output for the overlap
	region: the stretch of source left of `prev cut` that this window shares with the previous one. So the
	pins say "the overlap lines up, and so does the handover", and everything right of the cut -- the part
	this step actually produced -- is then read in units of that overlap.

	The RIGHT ends therefore do not line up, and are not meant to: how far past the cut the generated block
	runs is how much this step produced, and a step that ran long draws long. The span ratio in the title is
	the same quantity in numbers.

	Degeneracies fall back instead of dividing by zero or inverting the axis. With no primer, or an anchor at
	or left of x=0, the left pin has nothing to fix or would demand a non-positive scale: the cut pin is kept
	(it is the more specific claim) and the scale is borrowed from the source lane. With no `prev cut` or no
	generated notes at all -- step 0 -- there is no correspondence to assert and the lane normalises over its
	own extent from x=0. The x label names which of the three is in force.

	Why not one absolute tick axis for both. The two streams do not share a tick origin -- the irregular
	source carries rubato while the score output is quantised, so they sit a few hundred ticks apart before
	attention enters (measured: an identity gen#i<->src#i mapping already shows a +246 median offset).
	Plotting both on raw ticks makes every link lean by that offset, which looks like a finding and is not.
	The pins remove the offset by construction.

	What the earlier per-lane normalisation cost, and this no longer does: it stretched each lane to its OWN
	extent, so the output was fitted to the same width as the source no matter how much or little it had
	generated -- a step whose output ran half the source window's duration looked well-placed. Measured on one
	file, steps 2 and 3 ran 0.50x and 0.58x. Under the pins that difference is on the figure.
	'''
	import matplotlib
	matplotlib.use('Agg')					# file output only; no display on a training box
	import matplotlib.pyplot as plt
	from matplotlib.patches import ConnectionPatch
	from matplotlib.ticker import MaxNLocator

	src, out = window['src'], window['out']
	prime = window.get('prime') or []
	cuts = window.get('cuts') or {}

	def extent (events):
		ons = [e['onset'] for e in events]
		return (min(ons), max(ons)) if ons else (0, 0)

	src_lo, src_hi = extent(src)
	out_lo, out_hi = extent(list(prime) + list(out))
	gen_lo = extent(out)[0]					# the cut pin's note; only meaningful when `out` is non-empty

	def norm_s (tick):
		# a degenerate extent (one note, or all notes on one tick) has no meaningful position: centre it
		# rather than dividing by zero
		return 0.5 if src_hi <= src_lo else (tick - src_lo) / (src_hi - src_lo)

	# The output lane is placed by TWO pins, which is exactly enough to fix an affine map -- offset and scale
	# both, with nothing left over to choose:
	#
	#	left    the leftmost note the lane draws (the primer's first, when there is a primer) lands on x=0,
	#	        where the source window's first note is
	#	cut     the first GENERATED note lands on the source's `prev cut`, where this step picks up
	#
	# Both are real correspondences, not conveniences: the primer IS the output for the overlap region, the
	# stretch of source left of `prev cut` that this window shares with the previous one. So the two pins say
	# "the overlap lines up and the handover lines up", and the generated block's width is then read in units
	# of that overlap.
	#
	# Degeneracies fall back rather than divide by zero or invert the axis. No primer (or a primer starting on
	# the first generated tick) leaves nothing to pin the left edge with, and an anchor at or left of x=0
	# would need a zero or negative scale: in both cases the cut pin is kept -- it is the more specific claim
	# -- and the scale is borrowed from the source lane. With no cut or no generated note at all (step 0)
	# there is no correspondence to assert, so the lane normalises over its own extent.
	anchor_tick = cuts.get('src_prev')
	anchor_x = norm_s(anchor_tick) if anchor_tick is not None else None
	can_anchor = anchor_x is not None and bool(out) and src_hi > src_lo
	if can_anchor and gen_lo > out_lo and anchor_x > 1e-9:
		align = 'two-pin'
		o_gain = anchor_x / (gen_lo - out_lo)
		o_base, o_base_x = out_lo, 0.0
	elif can_anchor:
		align = 'cut'
		o_gain = 1.0 / (src_hi - src_lo)		# x per output tick == x per source tick
		o_base, o_base_x = gen_lo, anchor_x
	else:
		align = 'extent'
		o_gain = 0.0 if out_hi <= out_lo else 1.0 / (out_hi - out_lo)
		o_base, o_base_x = out_lo, (0.5 if out_hi <= out_lo else 0.0)

	def norm_o (tick):
		return o_base_x + (tick - o_base) * o_gain

	# hspace is set AFTER tight_layout, not here: tight_layout computes spacing itself and warns that
	# results may be incorrect when hspace has been pinned in gridspec_kw.
	fig, (ax_s, ax_o) = plt.subplots(2, 1,
		figsize=(max(9, min(30, (len(src) + len(out)) / 4.5)), 6.4))

	# One shared pitch RANGE across the two panels, each still drawn and ticked on its own axis: a
	# same-pitch link must come out vertical, and per-panel autoscaling would tilt it by whatever the two
	# tessituras happen to differ by.
	pitches = [e['pitch'] for e in src] + [e['pitch'] for e in out] + [e['pitch'] for e in prime]
	p_lo, p_hi = (min(pitches), max(pitches)) if pitches else (60, 61)
	pad = max(2, (p_hi - p_lo) * 0.08)

	# Limits come from everything actually DRAWN on either lane, not from a fixed 0..1. With the anchor the
	# output lane can start left of 0 (its primer runs back from the anchor) and end right of 1 (a generated
	# block longer in ticks than the source window), and a note outside the limits is a note silently dropped
	# from a figure whose title still counts it. Both panels get the SAME limits: equal x must be equal
	# screen position or a same-position link stops reading as vertical.
	xs = [0.0, 1.0]
	xs += [norm_s(e['onset']) for e in src]
	xs += [norm_o(e['onset']) for e in list(prime) + list(out)]
	for key, fn in (('src_prev', norm_s), ('src_next', norm_s),
			('out_prev', norm_o), ('out_next', norm_o)):
		if cuts.get(key) is not None:
			xs.append(fn(cuts[key]))
	# `<eom>` deliberately does NOT vote. Bar lines are minor gridlines and already skip themselves when out of
	# range, so letting one distant boundary widen the axis would shrink the notes for a line that is not even
	# drawn -- measured: a single out-of-range <eom> stretched the limits from -0.36..1.03 to -3.53..4.53.
	# Clamped: one far-out cut or a very long primer would otherwise squash the step itself to nothing. What
	# falls outside is counted into the title rather than quietly discarded.
	x_lo = max(-3.5, min(xs)) - 0.03
	x_hi = min(4.5, max(xs)) + 0.03
	# Ticks step by a round fraction across whatever range is visible, coarsening as the range widens so the
	# labels stay readable. Fractions outside 0..1 are labelled by the same affine maps, so they read as the
	# real ticks of the material before and after the window.
	span_x = x_hi - x_lo
	tick_step = 0.25 if span_x <= 2.2 else (0.5 if span_x <= 4.5 else 1.0)
	n_lo = int(math.ceil(x_lo / tick_step))
	n_hi = int(math.floor(x_hi / tick_step))
	fracs = [n * tick_step for n in range(n_lo, n_hi + 1)]
	for ax in (ax_s, ax_o):
		ax.set_xlim(x_lo, x_hi)
		ax.set_ylim(p_lo - pad, p_hi + pad)
		ax.set_xticks(fracs)
		ax.set_ylabel('pitch')
		ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
		ax.grid(alpha=0.18, zorder=1)
		# The links live on ax_s with clipping off so they can reach down into ax_o, but ax_o is drawn
		# after ax_s and its opaque background painted over exactly the part that crossed into it. Making
		# both patches transparent lets the segments stay visible over the whole crossing; the figure's own
		# white background still provides the surface.
		ax.patch.set_visible(False)

	# source panel: its ticks read along the TOP edge, so the two x scales sit at the figure's outside
	ax_s.xaxis.set_ticks_position('top')
	ax_s.xaxis.set_label_position('top')
	ax_s.set_xticklabels([f'{src_lo + f * (src_hi - src_lo):.0f}' for f in fracs])
	ax_s.set_xlabel(f'source onset (ticks {src_lo}..{src_hi}, normalised)', color=SRC_COLOR)
	ax_s.tick_params(axis='x', colors=SRC_COLOR)
	for side in ('top', 'left'):
		ax_s.spines[side].set_color(SRC_COLOR)

	# Labels INVERT norm_o rather than re-deriving a map of their own. Two expressions for one placement is one
	# too many: they only have to disagree once for the axis to be labelled with ticks the notes above it do
	# not have.
	def unnorm_o (x):
		return o_base if not o_gain else o_base + (x - o_base_x) / o_gain

	ax_o.set_xticklabels([f'{unnorm_o(f):.0f}' for f in fracs])
	ax_o.set_xlabel(
		f'generated onset (ticks {out_lo}..{out_hi}'
		+ (f', {out_lo} pinned to source start and first generated {gen_lo} to prev cut @{anchor_tick})'
			if align == 'two-pin' else
			f', first generated {gen_lo} pinned to prev cut @{anchor_tick}, source scale)'
			if align == 'cut' else ', normalised)'),
		color=OUT_COLOR)
	ax_o.tick_params(axis='x', colors=OUT_COLOR)
	for side in ('bottom', 'left'):
		ax_o.spines[side].set_color(OUT_COLOR)

	if src:
		ax_s.scatter([norm_s(e['onset']) for e in src], [e['pitch'] for e in src],
			s=26, marker='o', c=SRC_COLOR, edgecolors='white', linewidths=0.4,
			label=f'source note_on ({len(src)})', zorder=4)
		ax_s.legend(loc='upper right', fontsize=8, framealpha=0.9)
	# The output lane holds two different things and they must not read alike: notes THIS STEP generated
	# (filled triangles, the model's work, the only ones links attach to) and primer notes carried in from
	# earlier steps as context (hollow squares). Shape carries the distinction, not just colour, so it
	# survives greyscale and colour-blind viewing.
	if prime:
		ax_o.scatter([norm_o(e['onset']) for e in prime], [e['pitch'] for e in prime],
			s=26, marker='s', facecolors='none', edgecolors=OUT_COLOR, linewidths=0.7, alpha=0.65,
			label=f'primer note_on ({len(prime)})', zorder=3)
	if out:
		ax_o.scatter([norm_o(e['onset']) for e in out], [e['pitch'] for e in out],
			s=30, marker='v', c=OUT_COLOR, edgecolors='white', linewidths=0.4,
			label=f'generated note_on ({len(out)})', zorder=4)
	# ax_o's legend is built AFTER the <eom> lines below, so the bar-line entry can be in it.

	# Window cuts, dashed, on each lane's own axis: where the previous window was cut and where the next
	# one will be. They are what makes the step's reach legible -- material to the right of `next` is about
	# to be re-read by the following window, and the primer to the left of 0 is what that window will
	# carry over. A cut outside the drawn range is skipped rather than clamped, which would put it at a
	# tick it does not belong to.
	for ax, keys, fn in ((ax_s, ('src_prev', 'src_next'), norm_s),
			(ax_o, ('out_prev', 'out_next'), norm_o)):
		for key, style in zip(keys, ('prev', 'next')):
			tick = cuts.get(key)
			if tick is None:
				continue					# no such neighbour (first or last step), so nothing to mark
			x = fn(tick)
			if not (x_lo <= x <= x_hi):
				continue
			ax.axvline(x, color='#6a6a6a', lw=1.0, ls=(0, (4, 3)) if style == 'prev' else (0, (1, 2)),
				alpha=0.75, zorder=5)
			ax.annotate(f'{style} cut @{tick}', xy=(x, 1.0), xycoords=('data', 'axes fraction'),
				xytext=(2, -9), textcoords='offset points', fontsize=7, color='#5a5a5a')

	# Measure boundaries of the generated stream. They are the model's own barring, not the source's, and
	# they are what makes a horizontal position musically readable: a link landing a bar late is a different
	# error from one landing a beat late, and without bar lines both just look "shifted". Drawn thin and
	# behind the notes, and unlabelled -- one label per bar would crowd the lane for no information, since
	# the tick axis already gives the position.
	seen = set()
	for tick in sorted(window.get('eoms') or []):
		x = norm_o(tick)
		if not (x_lo <= x <= x_hi):
			continue
		key = round(x, 4)
		if key in seen:					# two boundaries on one tick would just thicken the same line
			continue
		seen.add(key)
		ax_o.axvline(x, color='#8a7f5a', lw=0.7, alpha=0.55, zorder=1,
			label=(f'<eom> ({len(window.get("eoms") or [])})' if len(seen) == 1 else None))
	if out or prime or seen:
		ax_o.legend(loc='upper right', fontsize=8, framealpha=0.9)

	# Links cross the panel boundary, so they are figure-level artists in DATA coordinates of the two
	# axes rather than lines inside either one.
	peak = max((p['score'] for p in pairs), default=0.0)		# the real peak, for the title
	hi = peak or 1.0
	# Floor the normaliser at the threshold. Weight is normalised WITHIN the step so a modest step does not
	# render blank beside a confident one -- but a step whose every link is a sub-threshold fallback has no
	# strong link to normalise against, and dividing by its own weak peak would draw those faint rows at
	# full strength. Flooring keeps a weak step looking weak.
	hi = max(hi, threshold) if threshold else hi
	n_weak = n_prime = 0
	for p in pairs:
		w = min(1.0, p['score'] / hi)
		# A link that did not clear the threshold is the note's argmax kept so no generated note is left
		# unexplained. It is drawn DOTTED and fainter: still visible as "this is where it looked most",
		# never mistakable for evidence at the same standing as a passing link.
		weak = threshold is not None and p['score'] < threshold
		n_weak += weak and p.get('kind', 'gen') == 'gen'
		# Primer links answer a different question -- what the model reads while ingesting its own carried-in
		# context, not what produced a note here -- so they get their own colour and dash. Same weight scale,
		# because the scores come from the same reduction.
		prime_link = p.get('kind', 'gen') == 'prime'
		n_prime += prime_link
		# Alpha floors at MIN_LINK_ALPHA. A link drawn at 0.02 is a link that was recorded and then hidden,
		# which is the worst of both: the count in the title says it is there and the figure says it is not.
		alpha = (0.10 + 0.28 * w) if weak else (0.14 + 0.66 * w)
		if prime_link:
			alpha *= 0.7				# recessive: context-reading is the secondary claim on this figure
		alpha = max(MIN_LINK_ALPHA, alpha)
		link = ConnectionPatch(
			xyA=(norm_s(p['src_onset']), p['src_pitch']), coordsA=ax_s.transData,
			xyB=(norm_o(p['out_onset']), p['out_pitch']), coordsB=ax_o.transData,
			color=PRIME_LINK_COLOR if prime_link else '#3a3a3a',
			lw=(0.3 + 0.9 * w) if weak else (0.4 + 1.8 * w), alpha=alpha,
			linestyle=((0, (5, 2)) if prime_link else (0, (1, 3)) if weak else '-'), zorder=2)
		# Added to an AXES, not the figure: a figure-level artist makes the figure incompatible with
		# tight_layout ("results might be incorrect"), and a layout warning is not worth shipping. Living
		# on ax_s with clipping off, the segment still reaches down into ax_o.
		link.set_clip_on(False)
		ax_s.add_artist(link)
	lo_line, hi_line = window['lines']
	# The span ratio is the number behind what the pins make visible: how far past the cut the generated block
	# runs is how much this step produced. Kept in the title because on an unpinned step (no prev cut) the lane
	# is still normalised to its own extent and the width says nothing -- and because "roughly this wide" is
	# not the same as 0.50x.
	# From span_ratio, NOT from the drawn extents: the output lane's extent includes the primer, and the ratio
	# is a claim about what this step GENERATED against the source window it read. Computing it from
	# out_hi - out_lo would silently start measuring primer + generated and disagree with the text report.
	ratio = span_note(window)
	# Counted honestly: solid links cleared the threshold, dotted ones are per-note fallbacks. Reporting a
	# single total would let a step of nothing but fallbacks read as a step full of links.
	n_strong = len(pairs) - n_weak - n_prime
	counts = (f'{n_strong} link(s) >= {threshold}'
		+ (f' + {n_weak} fallback' if n_weak else '')
		+ (f' + {n_prime} primer' if n_prime else ''))
	# Notes the clamped limits pushed off the canvas. Named in the title because the legend still counts them:
	# a figure whose caption says 40 notes while 6 are outside the axes is a figure lying about its own
	# contents, and the clamp exists precisely so one outlier cannot squash the rest.
	off = sum(1 for x in ([norm_s(e['onset']) for e in src]
			+ [norm_o(e['onset']) for e in list(prime) + list(out)])
		if not (x_lo <= x <= x_hi))
	fig.suptitle(f'step {step}  source lines [{lo_line}:{hi_line}]  '
		f'{counts}  peak {peak:.3f}  {ratio}'
		f'{f"  {off} note(s) off-range" if off else ""}{subtitle}', fontsize=10)
	# ConnectionPatch resolves its endpoints from the axes' transData at DRAW time, so laying out after
	# adding the links is safe: the segments follow the panels wherever tight_layout puts them.
	fig.tight_layout(rect=(0, 0.01, 1, 0.93))
	# Now widen the gap: the generated panel's x label and the source panel's top ticks both live in it,
	# and the links have to cross it legibly.
	fig.subplots_adjust(hspace=0.34)
	fig.savefig(path, dpi=130)
	plt.close(fig)


def span_ratio (window):
	'''(generated span, source span, ratio or None) in ticks for one step's window.'''
	so = [e['onset'] for e in window['src']]
	oo = [e['onset'] for e in window['out']]
	if not so or not oo:
		return None, None, None
	ss, os_ = max(so) - min(so), max(oo) - min(oo)
	return os_, ss, (os_ / ss if ss else None)


def span_note (window):
	'''One-line span summary for a step's log line.'''
	os_, ss, r = span_ratio(window)
	if os_ is None:
		return 'span n/a'
	return f'span {os_}/{ss} = {r:.2f}x' if r is not None else f'span {os_}/0'


def report_spans (windows, by_step, threshold=None):
	'''Per-step span ratios, as text beside the figures.

	A ratio far from 1.0 means the step's output covered a different duration than the source window it was
	translating, i.e. accumulated bar-length error. The figures place the output lane by two pins rather than
	stretching it to its own extent, so this is visible there as the generated block's width -- but only on a
	pinned step, and only to the eye. The table states it in numbers for every step, pinned or not.
	'''
	print('[attn] per-step span (generated ticks / source ticks — 1.0 = equal duration):')
	for step in sorted(windows):
		os_, ss, r = span_ratio(windows[step])
		if os_ is None:
			continue
		flag = '  <-- output duration disagrees' if r is not None and not 0.8 <= r <= 1.25 else ''
		shown = f'{r:.2f}x' if r is not None else 'n/a'
		# generated links only: this table is about what the step produced, and primer links belong to a
		# neighbouring step's production
		links = [p for p in by_step.get(step, []) if p.get('kind', 'gen') == 'gen']
		strong = links if threshold is None else [p for p in links if p['score'] >= threshold]
		count = (f'{len(strong)} links' if len(strong) == len(links)
			else f'{len(strong)} links + {len(links) - len(strong)} fallback')
		print(f'       step {step}: {os_:6d} / {ss:6d} = {shown}  ({count}){flag}')
	# A step with source notes and notes of its own, where NOTHING cleared the bar. Since every note now
	# keeps a fallback, such a step still draws lines -- all dotted -- so it is no longer visible as an
	# empty figure and has to be named here instead.
	quiet = []
	for s in sorted(windows):
		links = [p for p in by_step.get(s, []) if p.get('kind', 'gen') == 'gen']
		if not (windows[s]['out'] and windows[s]['src']) or not links:
			continue
		if threshold is not None and all(p['score'] < threshold for p in links):
			quiet.append(s)
	if quiet:
		print(f'[attn] {len(quiet)} step(s) generated notes but cleared the threshold nowhere '
			f'(figure shows fallbacks only): {quiet[:12]}{"..." if len(quiet) > 12 else ""}')


def plot_attention (pairs, windows, prefix, threshold, subtitle=''):
	'''Draw every step's figure in one pass, for a deferred run (--inspect-plot-at-end).

	The streaming path does not come through here: AttentionInspector.plot_step draws each step inside
	observe(). This exists for the deferred case and must produce the same files, which
	tests/midi/translate_midiseq2_check.py asserts.
	'''
	by_step = {}
	for p in pairs:
		by_step.setdefault(p['step'], []).append(p)
	os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
	paths = []
	for step in sorted(windows):
		links, window = by_step.get(step, []), windows[step]
		if not links and not window['out']:
			continue			# a step that generated nothing has no correspondence to show
		path = f'{prefix}.step{step:03d}.png'
		plot_attention_step(links, window, step, path, threshold, subtitle)
		paths.append(path)
	print(f'[attn] wrote {len(paths)} step figure(s): {prefix}.stepNNN.png')
	return paths


def main ():
	ap = argparse.ArgumentParser(description='Translate a whole midiseq2 file with MidiTranslator.')
	ap.add_argument('--run', default=DEFAULT_RUN, help='training run dir (.state.yaml + checkpoint)')
	ap.add_argument('--checkpoint', default=None,
		help="checkpoint path (default: config['best'], then best.chkpt, then latest.chkpt)")
	ap.add_argument('--input', required=True, help='source .midiseq2.txt')
	ap.add_argument('--output', default=None, help='destination .midiseq2.txt')
	ap.add_argument('--max-token', type=int, default=2048,
		help='total-T ceiling; pass the training max_tokens (default 2048)')
	ap.add_argument('--src-window', type=int, default=640,
		help='source-half token budget. 640 suits line_range [20,256] runs; RE-MEASURE per run '
			'(see module docstring) — on the [64,512] l16d256 run use 960')
	ap.add_argument('--no-prime', action='store_true',
		help="don't seed the target half with the previous window's tail (matches training exactly)")
	ap.add_argument('--advance-tokens', type=int, default=1,
		help='minimum target tokens retired per step, rounded up to the next <eom> (default 1)')
	ap.add_argument('--prime-window', type=int, default=2048,
		help='internal target-view safety ceiling in tokens; not the step stride (default 2048)')
	ap.add_argument('--max-steps', type=int, default=0, help='stop after N windows (0 = whole file)')
	ap.add_argument('--temperature', type=float, default=0.0, help='0 = greedy argmax')
	ap.add_argument('--top-k', type=int, default=0)
	ap.add_argument('--top-p', type=float, default=1.0)
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--threads', type=int, default=0)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--verbose', action='store_true')
	ap.add_argument('--inspect', action='store_true',
		help='record, for every generated pitch token, its attention over the source pitch tokens '
			'(one extra exact forward per window; forces eager attention)')
	ap.add_argument('--inspect-threshold', type=float, default=0.05,
		help='links at or above this attention score count as evidence (default 0.05). A note whose '
			'scores all fall below it still keeps its single best link, drawn dotted and reported '
			'separately, so no generated note is left unexplained')
	ap.add_argument('--inspect-top-k', type=int, default=8,
		help='per generated note, keep at most this many source links (0 = all above threshold); the '
			'per-note fallback is always one link regardless')
	ap.add_argument('--inspect-reduce', choices=['max', 'mean'], default='max',
		help="reduction over layers and heads; 'max' (default) is what the brief asks for")
	ap.add_argument('--inspect-query', choices=['producer', 'self'], default='producer',
		help="'producer' reads the row that CHOSE the pitch (one position earlier); 'self' reads the "
			'pitch token row itself. Different questions — see AttentionInspector')
	ap.add_argument('--inspect-plot', default=None,
		help='path PREFIX for the per-step onset-correspondence figures, written as '
			'<prefix>.stepNNN.png, one per sliding window (default: alongside --output)')
	ap.add_argument('--inspect-plot-at-end', action='store_true',
		help='draw all figures after the run instead of one per step as it completes. Same output; '
			'streaming is the default so a long run can be watched rather than waited out')
	ap.add_argument('--inspect-json', default=None, help='dump the full ranking as JSON')
	args = ap.parse_args()

	if args.threads:
		torch.set_num_threads(args.threads)
	torch.manual_seed(args.seed)

	config = Configuration.createOrLoad(args.run, volatile=True)
	checkpoint = resolve_checkpoint(args.run, config, args.checkpoint)
	config, model = load_model(args.run, checkpoint, args.device)

	data_args = config['data.args'] or {}
	pos_style = data_args.get('pos_style', 'flat')
	if pos_style == 'absolute':
		print("[error] pos_style 'absolute' places each half on its own file's token axis, which a "
			"sliding window cannot define (the output's total token count is unknown until done). "
			"Retrain with 'sep'/'flat', or extend this script with an explicit axis.")
		return 1
	trained_max = data_args.get('max_tokens')
	if trained_max and args.max_token != trained_max:
		print(f'[note] --max-token {args.max_token} differs from training max_tokens {trained_max}')
	if data_args.get('source_eom'):
		print('[note] config has source_eom on; the source half will carry <eom> tokens')

	tokenizer, vocab_path = resolve_tokenizer(args.run, config)
	if vocab_path:
		print(f'[vocab] {vocab_path} ({tokenizer.vocab_size} tokens)')

	with open(args.input, 'r', encoding='utf-8') as f:
		lines = f.read().splitlines()
	print(f'[in]  {os.path.basename(args.input)}: {len(lines)} lines, '
		f'pos_style {pos_style}, src_window {args.src_window}, max_token {args.max_token}')

	translator = SlidingTranslator(model, tokenizer, pos_style=pos_style,
		src_window=args.src_window, max_token=args.max_token, device=args.device,
		prime=not args.no_prime, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
		source_eom=bool(data_args.get('source_eom')), advance_tokens=args.advance_tokens,
		prime_window=args.prime_window)

	# The output path has to be settled BEFORE translate runs, because the streaming figures are named
	# from it and are written while the loop is still going.
	out_path = args.output or os.path.join(REPO_ROOT, 'tests', 'output', 'translate_midiseq2',
		os.path.basename(args.input))
	plot_prefix = args.inspect_plot or os.path.splitext(out_path)[0] + '.attn'

	inspector = None
	if args.inspect:
		inspector = AttentionInspector(model, tokenizer, translator.keywords, lines,
			bool(data_args.get('source_eom')), args.device, reduce=args.inspect_reduce,
			query=args.inspect_query, threshold=args.inspect_threshold, top_k=args.inspect_top_k,
			plot_prefix=None if args.inspect_plot_at_end else plot_prefix)
		print(f'[attn] inspecting: reduce {args.inspect_reduce}, query {args.inspect_query}, '
			f'threshold {args.inspect_threshold}, top-k {args.inspect_top_k or "all"}, '
			f'figures {"at end" if args.inspect_plot_at_end else "per step, as generated"}')

	output_ids, stats = translator.translate(lines, verbose=args.verbose, max_steps=args.max_steps,
		inspector=inspector)
	body = render_lines(output_ids, tokenizer, translator.keywords)
	final = compose_output(body, source_header(lines))
	report_output(final, stats)

	write_output(out_path, final)
	print(f'[done] {out_path}')

	if inspector is not None:
		pairs, out_events, src_events, windows = inspector.resolve_all(output_ids)
		by_step = {}
		for p in pairs:
			by_step.setdefault(p['step'], []).append(p)
		if args.inspect_plot_at_end:
			plot_attention(pairs, windows, plot_prefix, args.inspect_threshold,
				subtitle=f'  ({args.inspect_reduce} over layers/heads, {args.inspect_query} row)')
		else:
			print(f'[attn] {len(inspector.plot_paths)} step figure(s) already written during the run: '
				f'{plot_prefix}.stepNNN.png')
		report_spans(windows, by_step, threshold=args.inspect_threshold)
		report_attention(pairs, out_events, src_events, threshold=args.inspect_threshold)
		if args.inspect_json:
			with open(args.inspect_json, 'w', encoding='utf-8') as f:
				json.dump(dict(input=args.input, checkpoint=checkpoint,
					reduce=args.inspect_reduce, query=args.inspect_query,
					threshold=args.inspect_threshold, top_k=args.inspect_top_k,
					src_window=args.src_window, advance_tokens=args.advance_tokens, prime_window=args.prime_window,
					generated_notes=len(out_events), source_notes=len(src_events),
					# prime_notes and cuts are recorded so what a figure DREW is checkable from data,
					# rather than only by looking at the image: an empty primer or a missing cut is a
					# figure quietly short of a mark, and that should be visible here.
					steps={str(s): dict(lines=w['lines'],
							src_notes=len(w['src']), out_notes=len(w['out']),
							prime_notes=len(w.get('prime') or []),
							eoms=list(w.get('eoms') or []),
							cuts=w.get('cuts') or {})
						for s, w in sorted(windows.items())},
					links=pairs), f, indent=2)
			print(f'[attn] wrote {args.inspect_json}')
	return 0


if __name__ == '__main__':
	sys.exit(main())
