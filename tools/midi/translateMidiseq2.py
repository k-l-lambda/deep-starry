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
then moves `prime_start` past the first <eom> at or after it — the measure that rolls out of the view
is final. Both windows' left edges therefore advance one measure per step:

	step0   src[bars 1-6]  <sep> (empty)      -> generates bars 1-6   (~320 tok)
	step1   src[bars 2-7]  <sep> prime[2-6]   -> generates bar 7      (~64 tok)
	step2   src[bars 3-8]  <sep> prime[3-7]   -> generates bar 8

In steady state each step finalizes one measure and generates one, so the cost is ~one measure of
decoding per measure of output rather than a full re-generation per window.

The view has to be CAPPED (--prime-window, default 320), because a step can generate several measures
while only one rolls out, so the view grows. Left alone it leaves the trained distribution and the model
simply stops: observed at prime 432 (trained target halves are median 130, p99 351, max 468), where
<eos> came out with logit 13.16 against 7.71 for the next-best token — from the model's point of view a
target half that long is already finished. `trim_prime` rolls whole extra measures out to keep it in band.

An immediate <eos> at the END of a piece is the correct answer, not a stall: the source window ends in
`end_of_track`, the output already carries one, and there is nothing left to translate. `finished`
detects that (both halves must agree, so a spurious early end_of_track cannot truncate a file that still
has source) and ends the run cleanly instead of grinding through the remaining lines producing nothing.

Sizing, measured over 800 crops through the live training config (nota1m-00, mark_mode tick,
line_range [20,256], pos_style sep):

	source half tokens   median 275   p90 490   p99 635   max 1001
	target half tokens   median 130   p90 270   p99 351   max  468
	total T              median 409   p90 740   p99 916   max 1037

So `max_tokens: 2048` in the training config is NOT a window size — it is a resample cap on total T
that essentially never fires. --src-window defaults to 640 (about the trained p99), which puts
steady-state T at ~961, inside the trained p90-p99 band. A 1024-token source window would exceed the
longest source half the model has ever seen (1001) and push T past 1400, which training never saw.

Two asymmetries with training worth knowing, both deliberate:

  - The target half here is PRIMED with the previous window's tail so generation continues across the
    seam. Training always started a target half fresh at a crop boundary, so a primed target is
    out-of-distribution; it is what keeps successive windows one stream instead of overlapping
    alternatives. Pass --no-prime to fall back to the trained form.
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

Usage:
  python tools/midi/translateMidiseq2.py --run <run_dir> --input a.midiseq2.txt --output b.midiseq2.txt
  python tools/midi/translateMidiseq2.py --run <run_dir> --input a.txt --temperature 0 --verbose
'''

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModel
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer
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
		prime_window=320):
		self.model = model
		self.tk = tokenizer
		self.pos_style = pos_style
		self.src_window = src_window
		self.max_token = max_token
		self.device = torch.device(device)
		self.prime = prime
		# Cap on the target-half view. Trained target halves run median 130 / p99 351 / max 468, and a
		# prime past that reads to the model as an already-finished target: at 432 it emitted <eos>
		# immediately and the run stalled. 320 keeps the view inside the trained band with room for the
		# measure being generated.
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
		'''Autoregressive continuation to <eos> or the max_token ceiling. Returns new ids only.

		The <eos> is NOT included in the return: it terminates this window, but the output stream is
		one continuous piece of music, so an <eos> in the middle of it would be a stray token.
		'''
		ids = list(prefix_ids)
		positions = list(prefix_positions)
		out = []
		while len(ids) < self.max_token:
			window = torch.tensor([ids], dtype=torch.long, device=self.device)
			pos = torch.tensor([positions], dtype=torch.long, device=self.device)
			logits = self.model(window, None, pos)[0, -1, :]
			nxt = (int(logits.argmax().item()) if not temperature
				else sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p))
			if nxt == self.tk.eos_id:
				break
			out.append(nxt)
			ids.append(nxt)
			positions.append(positions[-1] + 1)
		return out

	# --- advancing -----------------------------------------------------------------------

	def advance_output (self, output, prime_start):
		'''Move the view's left edge one measure. Returns the new prime_start.

		One measure = up to and including the first <eom> at or after prime_start. With no <eom> in the
		view, fall back to half the CURRENT TARGET WINDOW — len(output[prime_start:]) // 2, i.e. the
		whole target region the model saw this step, not just the newly generated part.

		Guarantees forward motion of at least one token whenever the view is non-empty, so the loop
		cannot stall on a window that generated nothing and holds no <eom>.
		'''
		for i in range(prime_start, len(output)):
			if output[i] == self.tk.eom_id:
				return i + 1
		span = len(output) - prime_start
		return prime_start + max(1, span // 2) if span else prime_start

	def trim_prime (self, output, prime_start):
		'''Roll extra measures out of the view until the prime fits prime_window. Returns prime_start.

		Necessary because one step can generate far more than one measure while advance_output rolls
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

	def translate (self, lines, verbose=False, max_steps=0):
		'''Slide across `lines`, returning (output_ids, stats).

		The invariant that makes this terminate: every step either advances src_cursor or, at EOF,
		breaks. prime_start advances by at least one token per step while the view is non-empty.
		'''
		output = []
		prime_start = 0
		cursor = 0
		step = 0
		stalls = 0
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

			new_ids = self.generate(prefix, positions, self.temperature, self.top_k, self.top_p)
			output.extend(new_ids)
			if not new_ids:
				# An immediate <eos> is the RIGHT answer once the piece is over: the source window ends
				# in end_of_track and the output already carries one, so there is nothing left to
				# translate. Stopping here (rather than grinding through the remaining source lines
				# producing nothing) is what makes the run end cleanly. Only count it as a stall when
				# the music is NOT finished, which is the case worth warning about.
				if self.finished(output, src_ids):
					done = True
					break
				stalls += 1

			before = prime_start
			prime_start = self.advance_output(output, prime_start)
			# a step can generate several measures while only one rolls out, so the view grows; trim it
			# back into the trained band or the model starts answering <eos> immediately
			prime_start = self.trim_prime(output, prime_start)
			rolled = output[before:prime_start]
			onsets = count_note_on(rolled, self.tk)
			src_before = cursor
			cursor = self.advance_source_by_onsets(lines, cursor, onsets)
			# a step that consumed no source line would repeat the same window forever
			if cursor <= src_before:
				cursor = min(next_cursor, len(lines)) if next_cursor > src_before else src_before + 1

			if verbose:
				print(f'  step {step:4d}  src[{src_before}:{cursor}] {len(src_ids):5d} tok  '
					f'prime {len(prime_ids):5d}  gen {len(new_ids):5d}  '
					f'rolled {len(rolled):4d} tok / {onsets:3d} onsets  '
					f'out {len(output):7d}  T {len(prefix) + len(new_ids):5d}')
			step += 1

		stats = dict(steps=step, output_tokens=len(output), source_lines=len(lines),
			consumed_lines=cursor, stalls=stalls, done=done, elapsed=time.time() - start_time)
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
		print(f'[warn] {stats["stalls"]} step(s) generated nothing mid-piece (forced advance)')
	consumed, total = stats['consumed_lines'], stats['source_lines']
	# reaching end_of_track before the last source line is normal: the tail of a source file is its own
	# note_off/end_of_track run, which the model translates in one window rather than one per line.
	if consumed < total and not stats['done']:
		print(f'[warn] stopped after {consumed}/{total} source lines')


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
		help='source-half token budget (default 640, about the trained p99)')
	ap.add_argument('--no-prime', action='store_true',
		help="don't seed the target half with the previous window's tail (matches training exactly)")
	ap.add_argument('--prime-window', type=int, default=320,
		help='cap on the target-half view; past ~430 the model answers <eos> at once (default 320)')
	ap.add_argument('--max-steps', type=int, default=0, help='stop after N windows (0 = whole file)')
	ap.add_argument('--temperature', type=float, default=0.0, help='0 = greedy argmax')
	ap.add_argument('--top-k', type=int, default=0)
	ap.add_argument('--top-p', type=float, default=1.0)
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--threads', type=int, default=0)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--verbose', action='store_true')
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

	tokenizer = Midiseq2Tokenizer(data_args['vocab_path']) if data_args.get('vocab_path') \
		else Midiseq2Tokenizer()

	with open(args.input, 'r', encoding='utf-8') as f:
		lines = f.read().splitlines()
	print(f'[in]  {os.path.basename(args.input)}: {len(lines)} lines, '
		f'pos_style {pos_style}, src_window {args.src_window}, max_token {args.max_token}')

	translator = SlidingTranslator(model, tokenizer, pos_style=pos_style,
		src_window=args.src_window, max_token=args.max_token, device=args.device,
		prime=not args.no_prime, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
		source_eom=bool(data_args.get('source_eom')), prime_window=args.prime_window)

	output_ids, stats = translator.translate(lines, verbose=args.verbose, max_steps=args.max_steps)
	body = render_lines(output_ids, tokenizer, translator.keywords)
	final = compose_output(body, source_header(lines))
	report_output(final, stats)

	out_path = args.output or os.path.join(REPO_ROOT, 'tests', 'output', 'translate_midiseq2',
		os.path.basename(args.input))
	write_output(out_path, final)
	print(f'[done] {out_path}')
	return 0


if __name__ == '__main__':
	sys.exit(main())
