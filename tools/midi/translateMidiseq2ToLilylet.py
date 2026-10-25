'''Whole-file performance midiseq2 -> Lilylet score, one bar at a time, with two models.

    stage 1  bar lines   an irregular -> regular MidiTranslator (e.g. 20260909-...-lines360-l16d256)
                         slides over the source exactly as tools/midi/translateMidiseq2.py does, with
                         --align-advance on, and writes the bar lines it generated (<eom>) back onto
                         the source through the note correspondence (`annotate_source`). Its regular
                         output is kept beside the result for inspection.
    stage 2  notation    an InterleavedSeq2 MidiTranslator (e.g. 20261022-...-interleaved-nota1m-lyl)
                         reads the annotated source bar by bar, laid out the way the feeder lays out a
                         training crop, and each generated bar becomes history for the next:

        <bos>? %style\\n...\\n  src1 <sep> lyl1 <eom>  ...  src_k <sep>   ->  generate lyl_k to <eom>

The two stages run in sequence, not interleaved: stage 1 settles every bar line before stage 2 starts.
Feeding Lilylet back into the bar lines would only matter if the notation were to correct them, and
nothing here asks it to.

Why the bar lines come from the regular model's <eom> and not from a meter grid. A grid every
`numerator * 1920 / denominator` ticks is what a score means, but neither stage can rely on it: on
test202608 only 67/99 scores keep one meter from bar 1 (26 open with a \\partial, 35 change meter),
and the regular model's own output drifts off its declared grid within a few bars and never returns
(9 saved decodes: final drift -4800..+31440 ticks). A <eom> is decided per bar, so one wrong bar does
not move every later one. Measured against ground truth on 6 nota1m-100 files with every directive
stripped: 92% of bar starts land on the true source note, 83% of whole bars match the training unit.

The cut convention. `annotate_source` opens a bar immediately before its first note_on line, while the
feeder's unit opens at the `@measure` line, which the corpus puts before the elapse run leading into
that note. `bar_units` moves that one elapse line across the boundary. Measured on the interleaved
checkpoint's val crops (teacher-forced target CE): exact training cut 0.1188, this rule 0.1189,
annotate_source's own 0.1200. Being a NOTE off costs far more (+47..63% at +-1 on every bar), so the
accuracy of the bar lines, not their convention, is what limits this pipeline.

Stage 2's context follows the trained crop distribution: at most `--max-units` bars (the run's
line_range upper end) and at most `--max-tokens` ids including room for the bar being generated,
dropping the oldest bar first. <bos> leads the context only while bar 1 is still in it, as in training.

`--time N/D` pins the meter in both stages, at decode time rather than by editing the output: stage 1
forces the fields of every `time_signature` the regular model writes (so it draws its bar lines
against that meter), and stage 2 forces the text after every `\\time ` it writes. Only a meter the model
was going to state is pinned; nothing is inserted where it wrote none, and the report says how often
each happened.

Not handled: the Lilylet `[field "..."]` metadata block (staves, instruments), which the model was
never trained to write; decode-time block masking (an off-block id is dropped and counted, not
prevented); repeat structure (a performance is played order, so the score comes out unrolled).

Example:
    python tools/midi/translateMidiseq2ToLilylet.py \\
        --regular-run ~/data/models/deep-starry-logs/midi/20260909-midi-translator-nota1m0909-lines360-l16d256 \\
        --lilylet-run ~/data/models/deep-starry-logs/midi/20261022-midi-translator-interleaved-nota1m-lyl \\
        --input ~/data/midi/test202606/midiseq2/I-YIgmEZ0ss.midiseq2.txt \\
        --output tests/output/translate_lilylet/I-YIgmEZ0ss.lyl --device cuda
'''

import argparse
import collections
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from starry.utils.config import Configuration
from starry.midi.data.seq2seq2 import Seq2Seq2
from starry.midi.data.unifiedSeq2Tokenizer import UnifiedSeq2Tokenizer
from starry.lilylet.data.patchifier import LilyletTokenizer
from starry.midi.models.midiTranslator import KVDecoder
from tools.midi.translateMidiseq2 import (SlidingTranslator, assert_midiseq2, compose_output,
	is_elapse_line, load_model, reconcile_terminators, render_lines, report_output, resolve_checkpoint,
	resolve_tokenizer, source_header, write_output)


LILYLET_ASSET = os.path.join(REPO_ROOT, 'assets', 'lilylet-tokenizer.json')

# Stage 1's alignment guards: the recipe tests/output/yt-annotate-sep1033 measured on real recordings
# (over-generation stop and trim on both the miss and the reuse/density axes).
ANNOTATE_RECIPE = dict(align_advance=True, align_stop_window=16, align_stop_rate=0.6,
	align_trim_rate=0.3, align_stop_reuse=0.34, align_trim_density=2.0)


# --- meter ----------------------------------------------------------------------------------

def parse_meter (text):
	'''`N/D` -> (N, D). D must be a power of two, as a notated meter's is.'''
	try:
		numerator, denominator = (int(x) for x in text.split('/'))
	except ValueError:
		raise argparse.ArgumentTypeError(f'--time wants N/D, e.g. 4/4 or 6/8; got {text!r}')
	if not (1 <= numerator <= 32 and denominator in (1, 2, 4, 8, 16, 32)):
		raise argparse.ArgumentTypeError(f'--time {text}: numerator 1..32, denominator a power of two')
	return numerator, denominator


def bar_ticks (meter, ticks_per_beat=480):
	numerator, denominator = meter
	return numerator * ticks_per_beat * 4 // denominator


class MeterTranslator (SlidingTranslator):
	'''Stage 1 with the meter pinned: every `time_signature` the regular model writes gets `N _ D _ 1 8 _ 8`.

	The fields are midiseq2's minimal hex nibbles, written exactly as the corpus writes them
	(`time_signature 6 _ 8 _ 1 8 _ 8`; the denominator is literal, not MIDI's power-of-two exponent).
	The tail `1 8 _ 8` -- 24 clocks per click, 8 thirty-seconds per quarter -- is the constant every
	corpus file carries. Forced, not post-edited: the model then lays its bar lines down against the
	meter it has itself just "written", which is the only way the meter can reach the bar lines.
	'''

	def __init__ (self, *args, meter, **kwargs):
		super().__init__(*args, **kwargs)
		numerator, denominator = meter
		fields = list(format(numerator, 'x')) + ['_'] + list(format(denominator, 'x')) + ['_', '1', '8', '_', '8']
		self._meter_ids = [self.tk.id_by_token[t] for t in fields]
		self._signature_id = self.tk.id_by_token['time_signature']
		self.meter_forced = 0

	def constrain_logits (self, logits, out):
		# How far into a time_signature line this window's output is: the tokens since its last keyword.
		try:
			start = len(out) - 1 - out[::-1].index(self._signature_id)
		except ValueError:
			return logits
		done = len(out) - start - 1
		if done >= len(self._meter_ids) or out[start + 1:] != self._meter_ids[:done]:
			return logits
		if done == 0:
			self.meter_forced += 1
		pinned = torch.full_like(logits, float('-inf'))
		pinned[self._meter_ids[done]] = 0.0
		return pinned


def meter_conformance (lines, meter):
	'''Share of a midiseq2 file's bars (last excluded) whose tick length is the meter's bar. -> (on, of).'''
	length, abst, starts = bar_ticks(meter), 0, []
	for line in lines:
		if line.startswith('@measure'):
			starts.append(abst)
		elif is_elapse_line(line):
			abst += sum(int(t[1:], 16) for t in line.split())
	lengths = [b - a for a, b in zip(starts, starts[1:])]
	return sum(x == length for x in lengths), len(lengths)


# --- stage 1: bar lines ---------------------------------------------------------------------

def annotate_bars (args, lines):
	'''Run the regular model over the source. -> (annotated source lines, regular output lines, stats).'''
	config = Configuration.createOrLoad(args.regular_run, volatile=True)
	checkpoint = resolve_checkpoint(args.regular_run, config, args.regular_checkpoint)
	config, model = load_model(args.regular_run, checkpoint, args.device)
	if config['model.type'] != 'MidiTranslator':
		raise SystemExit(f'[error] stage 1 needs a decoder-only MidiTranslator, got {config["model.type"]}')
	data_args = config['data.args'] or {}
	tokenizer, _ = resolve_tokenizer(args.regular_run, config)
	assert_midiseq2(lines, args.input, tokenizer)

	common = dict(pos_style=data_args.get('pos_style', 'flat'), src_window=args.src_window,
		max_token=data_args.get('max_tokens') or 2048, device=args.device,
		source_eom=bool(data_args.get('source_eom')), prime_window=args.prime_window, **ANNOTATE_RECIPE)
	translator = (MeterTranslator(model, tokenizer, meter=args.time, **common) if args.time
		else SlidingTranslator(model, tokenizer, **common))
	started = time.time()
	output_ids, stats = translator.translate(lines)
	regular = compose_output(render_lines(output_ids, tokenizer, translator.keywords), source_header(lines))
	report_output(regular, stats)
	if (stats.get('align') or {}).get('failed'):
		raise SystemExit('[fail] stage 1: no measure survived the alignment trim, so there are no bar lines')
	# translate() has already settled `_align_kept_measures`; annotate exactly the bars it kept.
	annotated, ann_stats = translator.annotate_source(lines, translator._align_kept_measures)
	regular, _ = reconcile_terminators(regular, annotated)
	print(f'[stage1] {ann_stats["bars"]} bars in {time.time() - started:.1f}s; '
		f'{ann_stats["covered_notes"]}/{ann_stats["src_notes"]} source note_on inside them, '
		f'{ann_stats["empty_bars"]} with no note of their own, '
		f'{ann_stats["dropped_tail_notes"]} past the last bar line')
	if args.time:
		on, of = meter_conformance(regular, args.time)
		print(f'[stage1] meter {args.time[0]}/{args.time[1]}: time_signature forced {translator.meter_forced}x; '
			f'{on}/{of} regular bars are exactly {bar_ticks(args.time)} ticks')
		if not translator.meter_forced:
			print('[warn] the regular model wrote no time_signature, so the meter never reached stage 1')
	del model, translator
	if args.device.startswith('cuda'):
		torch.cuda.empty_cache()
	return annotated, regular, ann_stats


def bar_units (lines):
	'''Annotated source lines -> [(bar number, lines)], cut the way InterleavedSeq2 cuts a unit.

	Bar 1 owns everything before `@measure 2`, header included, as a head crop's first unit does.
	`@measure 1` and `@tick` are dropped (the feeder emits neither inside a unit). At every later
	`@measure`, an elapse line directly in front of it moves into the new bar -- see the module
	docstring -- unless it is the only line the previous bar holds, which would leave that bar with
	nothing at all.

	`annotate_source` closes the last translated bar with a terminal `@measure` and drops the source past
	it, so a trailing unit with no event line is folded back rather than decoded as a bar of its own.
	'''
	units = [[1, []]]
	for line in lines:
		if line.startswith('@measure'):
			number = int(line.split()[1])
			if number == 1:
				continue
			current = units[-1][1]
			moved = [current.pop()] if len(current) > 1 and is_elapse_line(current[-1]) else []
			units.append([number, moved])
		elif not line.startswith('@tick'):
			units[-1][1].append(line)
	if len(units) > 1 and not any(not is_elapse_line(l) for l in units[-1][1]):
		units[-2][1].extend(units.pop()[1])
	return [(number, body) for number, body in units]


# --- stage 2: notation ------------------------------------------------------------------------

class _Codec:
	'''InterleavedSeq2's own encoders, bound without building a dataset (which would need a corpus).

	Borrowed rather than restated so a unit is encoded byte for byte as the feeder encoded it in training.
	'''

	_mixed_encode_midi = Seq2Seq2._mixed_encode_midi
	_mixed_encode_lilylet = Seq2Seq2._mixed_encode_lilylet

	def __init__ (self, tokenizer, lilylet_tokenizer):
		self.tokenizer = tokenizer
		self.lilylet_tokenizer = lilylet_tokenizer


def load_notation_model (args):
	'''Stage 2's model, codec and trained bounds, refusing a run that is not the interleaved direction.'''
	config = Configuration.createOrLoad(args.lilylet_run, volatile=True)
	data_args = config['data.args'] or {}
	if (config['data.type'] != 'InterleavedSeq2' or data_args.get('source_format') != 'midiseq2'
			or data_args.get('target_format') != 'lilylet'):
		raise SystemExit(f'[error] stage 2 needs an InterleavedSeq2 midiseq2 -> lilylet run, got '
			f'{config["data.type"]} {data_args.get("source_format")} -> {data_args.get("target_format")}')
	if data_args.get('pos_style', 'flat') != 'flat':
		raise SystemExit('[error] InterleavedSeq2 positions are flat; this run says otherwise')
	checkpoint = resolve_checkpoint(args.lilylet_run, config, args.lilylet_checkpoint)
	config, model = load_model(args.lilylet_run, checkpoint, args.device)
	codec = _Codec(UnifiedSeq2Tokenizer(config['data.args.vocab_path']), LilyletTokenizer(LILYLET_ASSET))
	line_range = data_args.get('line_range') or [12, 12]
	upper = line_range[1] if isinstance(line_range, (list, tuple)) else line_range
	return model, codec, int(data_args.get('max_tokens') or 2048), int(upper)


def lilylet_text (ids, tok):
	'''Generated bar ids -> (text, kept ids, dropped count). Only Lilylet content survives.'''
	lo, size = tok.lilylet_offset, tok.blocks['lilylet']['size']
	kept = [i for i in ids if lo <= i < lo + size]
	return ''.join(tok.tokens[i] for i in kept), kept, len(ids) - len(kept)


def generate_bar (model, context, stop_id, max_new, temperature, device, force_after=None, forced=()):
	'''Greedy/sampled continuation of `context` to `stop_id`, on the KV cache. -> (ids, times forced).

	`model.generate` has no hook for forcing tokens, so this is its cached loop with one addition: once
	the output ends with `force_after`, the next tokens are `forced`, one per step, whatever the model
	scores. Used to pin the text after `\\time ` to the requested meter. Positions are flat, as
	InterleavedSeq2 trains them.
	'''
	decoder = KVDecoder(model)
	logits = decoder.prefill(context, list(range(len(context))), device=device)
	position, out, queue, times = len(context), [], [], 0
	for _ in range(max_new):
		if queue:
			nxt = queue.pop(0)
		elif temperature and temperature > 0:
			nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item())
		else:
			nxt = int(logits.argmax().item())
		out.append(nxt)
		if nxt == stop_id:
			break
		if force_after and out[-len(force_after):] == force_after and not queue:
			queue, times = list(forced), times + 1
		logits = decoder.step(nxt, position, device=device)
		position += 1
	return out, times


def decode_bars (model, codec, units, style_ids, args, max_tokens, max_units):
	'''Generate one Lilylet bar per unit. -> (bar texts, stats).'''
	tok = codec.tokenizer
	device = torch.device(args.device)
	sources = [codec._mixed_encode_midi(body, False) for _, body in units]
	# `\time 4/4` tokenizes as `\time`, ` `, `4`, `/`, `4`, and the corpus always follows the meter with a
	# space; forcing that space too keeps the next element from being glued onto the denominator.
	force_after = forced = None
	if args.time:
		force_after = codec._mixed_encode_lilylet('\\time ')
		forced = codec._mixed_encode_lilylet(f'{args.time[0]}/{args.time[1]} ')
	history, texts = [], []
	stats = collections.Counter()
	started = time.time()
	for k in range(len(units)):
		first = max(0, k - (max_units - 1))
		while True:
			context = ([tok.bos_id] if first == 0 else []) + style_ids
			for j in range(first, k):
				context += sources[j] + [tok.sep_id] + history[j] + [tok.eom_id]
			context += sources[k] + [tok.sep_id]
			if first == k or len(context) + args.max_bar_tokens <= max_tokens:
				break
			first += 1
		generated, times = generate_bar(model, context, tok.eom_id, args.max_bar_tokens, args.temperature,
			device, force_after, forced)
		stats['meter_forced'] += times
		stats['no_time'] += bool(args.time) and not times
		closed = bool(generated) and generated[-1] == tok.eom_id
		text, kept, dropped = lilylet_text(generated[:-1] if closed else generated, tok)
		history.append(kept)
		texts.append(text)
		stats['generated'] += len(generated)
		stats['unclosed'] += not closed
		stats['off_block'] += dropped
		stats['empty_source'] += not any(line.startswith('note_on') for line in units[k][1])
		print(f'[bar {k + 1:3d}/{len(units)}] src {len(sources[k]):4d} ids | context {len(context):4d} '
			f'(bars {first + 1}..{k + 1}) -> {len(kept):4d} ids{"" if closed else "  UNCLOSED"}'
			f'{f"  {dropped} off-block" if dropped else ""}', flush=True)
		if args.verbose:
			print('    ' + text.rstrip('\n').replace('\n', '\n    '))
	stats['seconds'] = round(time.time() - started, 1)
	return texts, stats


def compose_lilylet (style_lines, texts):
	'''Style block, then one bar per paragraph closed by `%N`, the layout of the corpus files.'''
	out = [line.rstrip('\n') for line in style_lines]
	if out:
		out.append('')
	for number, text in enumerate(texts, 1):
		out.append(f'{text.rstrip()} %{number}')
		out.append('')
	return out


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--regular-run', required=True, help='stage 1: irregular -> regular MidiTranslator run dir')
	ap.add_argument('--regular-checkpoint', default=None)
	ap.add_argument('--lilylet-run', required=True, help='stage 2: InterleavedSeq2 midiseq2 -> lilylet run dir')
	ap.add_argument('--lilylet-checkpoint', default=None)
	ap.add_argument('--input', required=True, help='performance .midiseq2.txt (no @measure needed)')
	ap.add_argument('--annotated', default=None,
		help='skip stage 1 and read bar lines from this @measure-annotated source instead')
	ap.add_argument('--output', required=True, help='destination .lyl; stage 1 files are written beside it')
	ap.add_argument('--time', type=parse_meter, default=None, metavar='N/D',
		help='pin the meter, e.g. 4/4: forced into the regular model\'s time_signature (so its bar lines\n'
			'follow it) and after every \\time the notation model writes (default: both models choose)')
	ap.add_argument('--style', action='append', default=[],
		help='a Lilylet style comment line, e.g. --style Classical --style Keyboard (repeatable)')
	ap.add_argument('--src-window', type=int, default=960, help='stage 1 source window (default 960)')
	ap.add_argument('--prime-window', type=int, default=320, help='stage 1 primer ceiling (default 320)')
	ap.add_argument('--max-units', type=int, default=0, help="stage 2 bars per context (default: the run's line_range upper)")
	ap.add_argument('--max-bar-tokens', type=int, default=512, help='stage 2 cap per generated bar (default 512)')
	ap.add_argument('--temperature', type=float, default=0.0, help='stage 2 sampling; 0 = greedy')
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--verbose', action='store_true', help='print every generated bar')
	args = ap.parse_args()
	torch.manual_seed(args.seed)

	with open(args.input, 'r', encoding='utf-8') as f:
		lines = f.read().splitlines()
	stem = os.path.splitext(args.output)[0]
	if args.annotated:
		with open(args.annotated, 'r', encoding='utf-8') as f:
			annotated = f.read().splitlines()
		print(f'[stage1] skipped; bar lines from {args.annotated}')
		if args.time:
			print('[warn] --time with --annotated pins only the notation: the bar lines were drawn elsewhere')
	else:
		annotated, regular, _ = annotate_bars(args, lines)
		write_output(stem + '.regular.midiseq2.txt', regular)
		write_output(stem + '.annotated.midiseq2.txt', annotated)
		print(f'[stage1] {stem}.regular.midiseq2.txt\n[stage1] {stem}.annotated.midiseq2.txt')

	units = bar_units(annotated)
	model, codec, max_tokens, max_units = load_notation_model(args)
	style_lines = [s if s.startswith('%') else '%' + s for s in args.style]
	# The feeder's own style encoding: each line with its newline, then the blank line that ends the block.
	style_ids = codec._mixed_encode_lilylet(''.join(s + '\n' for s in style_lines) + '\n') if style_lines else []
	print(f'[stage2] {len(units)} bars, style {len(style_lines)} line(s), context <= {args.max_units or max_units} '
		f'bars / {max_tokens} ids')
	texts, stats = decode_bars(model, codec, units, style_ids, args, max_tokens, args.max_units or max_units)

	result = compose_lilylet(style_lines, texts)
	write_output(args.output, result)
	print(f'[stage2] {len(texts)} bars, {stats["generated"]} ids in {stats["seconds"]}s; '
		f'{stats["unclosed"]} bar(s) hit --max-bar-tokens without <eom>, {stats["off_block"]} off-block id(s) '
		f'dropped, {stats["empty_source"]} bar(s) with no source note_on')
	if args.time:
		print(f'[stage2] meter {args.time[0]}/{args.time[1]}: forced after \\time {stats["meter_forced"]}x; '
			f'{stats["no_time"]} bar(s) wrote no \\time at all')
	print(f'[done] {args.output}')
	return 0


if __name__ == '__main__':
	sys.exit(main())
