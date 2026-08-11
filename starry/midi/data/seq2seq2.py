'''Seq2Seq2 — a paired-midiseq2 feeder: one midiseq2 corpus in, another out.

Given TWO directories of `.midiseq2.txt` files holding the same pieces under the same basenames —
e.g. test202608's `midi-seq2-score/` (from the lilylet AST), `midi-seq2/` (abc -> MusicXML ->
MuseScore) and `midi-seq2-irregular/` (regular + pianistMockerMIDI perturbation) — every pairing is
a supervised translation task. This feeder crops a random window out of the SOURCE file, locates the
window covering the same music in the TARGET file, and emits ONE flat id sequence:

    <bos>? source...  <sep>  <bos>? target... <eos>

with a mask over the target half. The two wrappers do NOT play the same role:

    <bos>  conditional and symmetric — on both halves iff the crop reaches the START of the piece,
           so the model can tell an opening from an interior fragment.
    <eos>  unconditional, target only — it terminates the GENERATED half and nothing else. The source
           is a read-only condition whose extent is plain to see, so an <eos> there marks nothing new;
           and on the target it has to mean "this crop is finished" rather than "the piece ended",
           since most crops are mid-piece and a model whose stop token is rare does not learn to stop.

Alignment is by MARK IDENTITY, never by line number or tick arithmetic. A mark is a midiseq2
directive naming a score position, and `mark_mode` picks which kind counts:

    'measure'  @measure N       key = N                 (default)
    'tick'     @tick T          key = (measure, T)      @measure lines are NOT marks here

Crop boundaries always land on marks, so the target window is found by looking up the boundary keys
rather than by measuring anything. Measured on test202608, `@measure` keys are present in both arms
for every mark (0 missing of 8097), while `@tick` keys go missing 0.12% of the time regular->irregular
and 3.08% regular->score — hence `_align`'s outward walk.

Directives are CONTROL, not content: `@tick` and `@measure` lines never become tokens. The one
exception is the target's `@measure`, which becomes a single `<eom>` so the decoder gets its bar
boundaries (skipped for `@measure 1`, which marks the start of the piece rather than a boundary
within it). `source_eom` mirrors that on the source half if wanted.

Unlike its siblings in this package (seq2CondPatchy, seq2CondSplitPatchy) this feeder reads TEXT at
runtime instead of a packed `.pt`, so a change of crop policy needs no re-pack. Note the consequence
for `splits`: the filter is POSITIONAL (`i % cycle in phases`), so the file list must be sorted
deterministically or val silently leaks into train.

Batch contract:
	input_ids   LongTensor [B, T]   source ++ <sep> ++ target, right-padded with <pad>
	masks       LongTensor [B, T]   1 = real token
	target_mask LongTensor [B, T]   1 = a supervised target position (strictly after <sep>)
	sep_index   LongTensor [B]      position of <sep> in each row

Loss convention: for a next-token model, compare `logits[:, i - 1]` against `input_ids[:, i]` at
every `i` where target_mask is 1. `<sep>` is therefore the last context position before the first
supervised token, and nothing in the source half is ever a target.
'''

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs
from ...utils.registry import register_dataset
from .seq2CondPachifier import Midiseq2Tokenizer


# A mark key is either a measure number ('measure' mode) or a (measure, tick) pair ('tick' mode).
MarkKey = Any
# (line_index, key) — the line the directive sits on, and what identifies it across files.
Mark = Tuple[int, MarkKey]


class _File:
	'''One parsed midiseq2 file: its lines, its marks, and a key -> line-numbers index.

	`marks` is in file order. `lines_of` maps a key to EVERY line carrying it, because the key is not
	unique in general: the irregular arm has 730 duplicate (measure, tick) keys across 15 files (the
	perturbation can move two events onto the same score tick). `_align` relies on that list.
	'''

	def __init__ (self, path: str, mark_mode: str):
		with open(path, 'r', encoding='utf-8') as f:
			self.lines: List[str] = f.read().splitlines()
		self.marks: List[Mark] = []
		self.lines_of: Dict[MarkKey, List[int]] = {}
		# The measure number is tracked as parse state in BOTH modes: in 'tick' mode an @measure line
		# is not itself a mark, but it still tells us which measure the following @tick values are in
		# (a bare tick repeats every bar and would collide across the piece).
		measure: Optional[int] = None
		for index, line in enumerate(self.lines):
			if line.startswith('@measure'):
				measure = int(line.split()[1])
				if mark_mode != 'measure':
					continue
				key: MarkKey = measure
			elif line.startswith('@tick'):
				if mark_mode != 'tick':
					continue
				key = (measure, int(line.split()[1]))
			else:
				continue
			self.marks.append((index, key))
			self.lines_of.setdefault(key, []).append(index)

	def __len__ (self) -> int:
		return len(self.lines)


# Parsed files are shared across dataset instances built from the same paths within a process, so the
# train and val splits of one config do not each hold their own copy. Keyed by (path, mark_mode)
# because the marks depend on the mode.
_FILE_CACHE: Dict[Tuple[str, str], _File] = {}


def _get_file (path: str, mark_mode: str) -> _File:
	key = (path, mark_mode)
	parsed = _FILE_CACHE.get(key)
	if parsed is None:
		parsed = _File(path, mark_mode)
		_FILE_CACHE[key] = parsed
	return parsed


def _is_directive (line: str) -> bool:
	return line.startswith('@measure') or line.startswith('@tick')


def _line_range (line_range: Any) -> Tuple[int, int]:
	'''Normalize the `line_range` option to an inclusive (lo, hi) pair.

	A [lo, hi] sequence draws the cap per crop; a scalar means a fixed cap, i.e. (n, n). Written as one
	place so the config can say either and everything downstream sees a pair.
	'''
	if isinstance(line_range, (list, tuple)):
		if len(line_range) != 2:
			raise ValueError(f'line_range must be [lo, hi], got {line_range!r}')
		lo, hi = int(line_range[0]), int(line_range[1])
	else:
		lo = hi = int(line_range)
	if lo < 1 or hi < lo:
		raise ValueError(f'line_range must satisfy 1 <= lo <= hi, got {line_range!r}')
	return lo, hi


def _measures_in (file: _File, start: int, end: int) -> List[Tuple[int, int]]:
	'''[(line_index, measure_number)] for the @measure lines in [start, end).

	Independent of mark_mode: @measure lines are read straight off the text, so the numbers are the
	score's own bar numbers whether or not @measure is what bounds a crop in this mode. @measure 1 is
	included — it is a real bar number, even though _encode emits no <eom> for it.
	'''
	out: List[Tuple[int, int]] = []
	for index in range(start, min(end, len(file.lines))):
		line = file.lines[index]
		if line.startswith('@measure'):
			out.append((index, int(line.split()[1])))
	return out


@register_dataset
class Seq2Seq2 (Dataset):
	'''Paired-midiseq2 feeder. See the module docstring for the batch contract.'''

	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		splits = splits.split(':')

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return tuple(
			cls(root, split, device=device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)

	def __init__ (self, root, split, device='cpu', shuffle=False,
		source_dir='midi-seq2-score', target_dir='midi-seq2', mark_mode='measure',
		line_range=(20, 256), p_head=0.15, p_tail=0.15, source_eom=False,
		max_tokens=0, resample_tries=8, align_retries=4,
		random_crop=None, seed=0, vocab_path=None, **_):
		super().__init__()
		self.device = device
		self.shuffle = shuffle
		if mark_mode not in ('measure', 'tick'):
			raise ValueError(f'mark_mode must be "measure" or "tick", got {mark_mode!r}')
		self.mark_mode = mark_mode
		self.source_root = os.path.join(root, source_dir)
		self.target_root = os.path.join(root, target_dir)
		# line_range bounds the SOURCE crop only; the target length follows from mark alignment and is
		# direction-dependent (score->regular targets run ~2x the source, regular->irregular ~1.3x).
		# max_tokens, if set, bounds the ASSEMBLED sequence by RESAMPLING — truncating would cut a
		# boundary off its mark and break the very alignment this feeder exists to provide.
		#
		# The cap is drawn UNIFORMLY from [lo, hi] per crop. Growth is greedy up to whatever cap it gets,
		# so a single fixed value would make every crop as long as it can be — measured on this corpus
		# the crop lands at 0.95 of the cap (p10 0.83) in tick mode, meaning the model would only ever
		# see near-max windows and would have to extrapolate to short ones. A scalar is accepted and
		# means a fixed cap, i.e. [n, n].
		self.line_range = _line_range(line_range)
		# The upper bound, which is what a length assertion means by "within the cap".
		self.max_lines = self.line_range[1]
		self.p_head = p_head
		self.p_tail = p_tail
		self.source_eom = source_eom
		self.max_tokens = max_tokens
		self.resample_tries = resample_tries
		self.align_retries = align_retries
		# Deterministic crops for val: default follows the split's shuffle flag (as m3distill does),
		# so train augments and val is reproducible epoch to epoch.
		self.random_crop = shuffle if random_crop is None else random_crop
		self.seed = seed
		self.tokenizer = Midiseq2Tokenizer(vocab_path) if vocab_path else Midiseq2Tokenizer()

		# The split filter is positional, so the file list MUST be deterministically ordered or the
		# train/val partition shifts with directory iteration order.
		names = sorted(
			set(os.listdir(self.source_root)) & set(os.listdir(self.target_root))
		)
		self.names = [name for name in names if name.endswith('.txt')]
		if not self.names:
			raise RuntimeError(
				f'no shared .txt basenames between {self.source_root} and {self.target_root}')

		phases, cycle = parseFilterStr(split)
		self.indices = [i for i in range(len(self.names)) if i % cycle in phases]

	def __len__ (self) -> int:
		return len(self.indices)

	# --- crop selection -------------------------------------------------------------------
	#
	# A crop is a pair of mark indices (a, z) covering source marks a..z-1, converted to lines as:
	#
	#	start_line = 0 if a == 0 else marks[a].line			 (a == 0: nothing precedes the first
	#														  mark but the header, so take it)
	#	end_line   = len(lines) if z == len(marks) else marks[z].line
	#
	# a == 0 therefore MEANS start-of-piece (which is what drives <bos>) and z == len(marks) means
	# end-of-piece (reported as `tail`, but no longer a wrapper condition — see _assemble's <eos> note).
	# In 'measure' mode a == 0 is the `@measure 1` mark, so the "ignore the
	# opening @measure 1 and set the boundary to the beginning" rule falls out of the same arithmetic
	# rather than needing a special case; in 'tick' mode it generalizes to the first @tick.

	def _bounds (self, source: _File, a: int, z: int) -> Tuple[int, int]:
		start = 0 if a == 0 else source.marks[a][0]
		end = len(source.lines) if z >= len(source.marks) else source.marks[z][0]
		return start, end

	def _pick_crop (self, source: _File, rng: random.Random) -> Tuple[int, int]:
		'''Choose (a, z) — the source mark range. Grows by WHOLE marks, so a boundary can never land
		mid-measure; that is what makes the target lookup a key lookup instead of a search.

		The line cap is drawn per crop from `line_range` (a fixed value collapses to itself), off the
		SAME rng as the mode and the start so a deterministic crop stays deterministic.
		'''
		count = len(source.marks)
		if count == 0:
			# a file with no marks at all: the only honest crop is the whole thing.
			return 0, 0
		lo, hi = self.line_range
		# Uniform over lines, not over marks: the cap is what the caller reasons about, and marks vary
		# wildly in span (a measure is ~104 lines, a tick mark ~15), so sampling marks would make the
		# realized length distribution depend on the mark mode.
		limit = lo if lo == hi else rng.randint(lo, hi)
		roll = rng.random()
		if roll < self.p_head:
			mode = 'head'
		elif roll < self.p_head + self.p_tail:
			mode = 'tail'
		else:
			mode = 'middle'

		if mode == 'tail':
			# grow LEFTWARD from EOF: the smallest a whose span to EOF still fits.
			z = count
			a = count - 1
			while a > 0:
				start, end = self._bounds(source, a - 1, z)
				if end - start > limit:
					break
				a -= 1
			return a, z

		a = 0 if mode == 'head' else rng.randrange(0, count)
		z = a + 1
		while z < count:
			start, end = self._bounds(source, a, z + 1)
			if end - start > limit:
				break
			z += 1
		return a, z

	# --- target alignment -----------------------------------------------------------------

	def _walk_out (self, source: _File, target: _File, index: int, direction: int) -> int:
		'''From source mark `index`, step by `direction` until the key also exists in the target.

		Returns the mark index found, or a sentinel outside the range meaning "clamp to the edge":
		-1 for start-of-piece, len(marks) for end-of-piece. Walking outward (never inward) keeps the
		window a SUPERSET of the requested music, so a missing mark costs a little extra context
		rather than a silently truncated target.

		Only 'tick' mode reaches the loop in practice — every @measure key is present in both arms
		across the whole corpus. Measured cost when it does: median 2-5 marks, max 31.
		'''
		i = index
		while 0 <= i < len(source.marks):
			if source.marks[i][1] in target.lines_of:
				return i
			i += direction
		return -1 if direction < 0 else len(source.marks)

	def _align (self, source: _File, target: _File, a: int, z: int) -> Optional[Tuple[int, int]]:
		'''Source mark range (a, z) -> target line range, by mark key.

		Two passes, and the second is not optional. First the boundaries: the left edge takes the FIRST
		target line carrying its key and the right edge the LAST, so a duplicated key (the irregular arm
		has 730 of them) widens the window rather than inverting it.

		Then the interior is swept, because the boundaries alone do not bound it. The perturbation can
		REORDER tick marks, so a key that sits inside the source range can land outside the window its
		two boundary keys define — e.g. source marks 51..67 spanning keys (5,360)..(6,720) gave target
		lines 488..643 while (5,840) sat at target line 479, ahead of the window. Extending to swallow
		every shared interior key is what makes "the target covers the same music" true rather than
		merely usually true. It fires on ~0.5% of crops and does not move the median length.

		Returns None if the range still comes out empty, which lets the caller retry or fall back.
		'''
		# a == 0 already means start-of-piece, so it needs no key lookup; same for z at the end.
		if a <= 0:
			start = 0
		else:
			left = self._walk_out(source, target, a, -1)
			start = 0 if left < 0 else min(target.lines_of[source.marks[left][1]])

		if z >= len(source.marks):
			end = len(target.lines)
		else:
			right = self._walk_out(source, target, z, 1)
			# Exclusive, mirroring the source side: the boundary mark's own line is not included, so
			# the segment holds exactly the music between the two marks.
			end = len(target.lines) if right >= len(source.marks) \
				else max(target.lines_of[source.marks[right][1]])

		# Second pass: pull the window out to cover every interior key the target shares, wherever the
		# target happens to place it.
		for _, key in source.marks[a:z]:
			for line in target.lines_of.get(key, ()):
				start = min(start, line)
				end = max(end, line + 1)
		# ...but never past the edges those boundaries pinned.
		if a <= 0:
			start = 0
		if z >= len(source.marks):
			end = len(target.lines)

		if end <= start:
			return None
		return start, end

	# --- token assembly -------------------------------------------------------------------

	def _encode (self, lines: Sequence[str], eom: bool) -> List[int]:
		'''Lines -> ids. Directives never become content tokens; with `eom` on, an @measure line
		contributes one <eom> instead. @measure 1 is skipped: it opens the piece rather than closing a
		bar, and <bos> already carries that.'''
		ids: List[int] = []
		lookup = self.tokenizer.id_by_token
		unknown = self.tokenizer.unknown_id
		for line in lines:
			if line.startswith('@measure'):
				if eom and line.split()[1] != '1':
					ids.append(self.tokenizer.eom_id)
				continue
			if line.startswith('@tick'):
				continue
			for token in line.split():
				ids.append(lookup.get(token, unknown))
		return ids

	def _assemble (self, source: _File, target: _File, a: int, z: int,
		align: Tuple[int, int]) -> Tuple[List[int], int]:
		'''Build the joined id sequence and the index of its <sep>.'''
		s_start, s_end = self._bounds(source, a, z)
		t_start, t_end = align
		# <bos> reflects the SOURCE crop reaching the START of the piece, and appears on both halves —
		# _align clamps the target range to the same edge, so the two agree.
		head = a <= 0
		# <eos> is NOT conditional and NOT symmetric: the target half always ends with it, the source
		# half never carries it. The source is a read-only condition whose extent the model can simply
		# see, so an <eos> there marks nothing it does not already know. On the target, <eos> is the only
		# way generation can stop — and it has to mean "this crop is finished", not "the piece ended",
		# because a mid-piece crop is the common case (0.7 of them by p_head/p_tail). Making it
		# conditional on reaching the end of the piece would leave most targets unterminated and teach
		# the model that stopping is rare.
		#
		# `tail` therefore no longer affects the wrappers; it still decides <bos> placement upstream and
		# is reported by describe().
		def wrap (ids: List[int]) -> List[int]:
			return ([self.tokenizer.bos_id] if head else []) + ids

		source_ids = wrap(self._encode(source.lines[s_start:s_end], self.source_eom))
		target_ids = wrap(self._encode(target.lines[t_start:t_end], True)) + [self.tokenizer.eos_id]
		return source_ids + [self.tokenizer.sep_id] + target_ids, len(source_ids)

	# --- item -----------------------------------------------------------------------------

	def describe (self, index: int) -> Dict[str, Any]:
		'''The whole crop decision for one sample, ids included.

		`_item` is a thin wrapper over this. Visualization and diagnostics need what the id sequence
		cannot carry — which lines were cropped, which marks bound them, and above all the MEASURE
		NUMBERS, since _encode turns @measure N into a bare <eom>. Returning it from the feeder rather
		than re-deriving it outside keeps the two from drifting apart.

		Keys: name, source, target (_File), a, z (source mark range), source_range, target_range
		(line slices), ids, sep, head, tail, source_measures, target_measures — the latter two being
		[(line_index, measure_number)] for every @measure line inside that half's range, @measure 1
		included (it is a real bar number even though it emits no <eom>).
		'''
		name = self.names[index]
		source = _get_file(os.path.join(self.source_root, name), self.mark_mode)
		target = _get_file(os.path.join(self.target_root, name), self.mark_mode)
		# A deterministic crop still varies BY SAMPLE (so val covers head/tail/middle) but not by
		# epoch; seeding on the index is what gives both.
		rng = random if self.random_crop else random.Random(self.seed ^ (index * 2654435761))

		# The crop that wins is kept WHOLE — ids together with the (a, z, align) that produced them.
		# Keeping only the ids would leave the ranges describing whichever attempt happened to be last.
		best: Optional[Tuple[List[int], int, int, int, Tuple[int, int]]] = None
		for attempt in range(max(1, self.resample_tries)):
			a, z = self._pick_crop(source, rng)
			align = self._align(source, target, a, z)
			if align is None:
				# Widen by a mark on each side and retry; a wider window has more chance of hitting a
				# key both files share.
				for _ in range(self.align_retries):
					a = max(0, a - 1)
					z = min(len(source.marks), z + 1)
					align = self._align(source, target, a, z)
					if align is not None:
						break
			if align is None:
				# Last resort: the whole piece. Both files always have line 0 and EOF.
				align = (0, len(target.lines))
				a, z = 0, len(source.marks)
			ids, sep = self._assemble(source, target, a, z, align)
			if best is None or len(ids) < len(best[0]):
				best = (ids, sep, a, z, align)
			if not self.max_tokens or len(ids) <= self.max_tokens:
				break
		# If every attempt overshot max_tokens we keep the SHORTEST one rather than truncating: a
		# truncated tail would leave the target unterminated and unaligned with its final mark.
		ids, sep, a, z, align = best
		s_start, s_end = self._bounds(source, a, z)
		t_start, t_end = align
		return dict(name=name, source=source, target=target, a=a, z=z,
			source_range=(s_start, s_end), target_range=(t_start, t_end),
			ids=ids, sep=sep, head=a <= 0, tail=z >= len(source.marks),
			source_measures=_measures_in(source, s_start, s_end),
			target_measures=_measures_in(target, t_start, t_end))

	def _item (self, index: int) -> Tuple[torch.Tensor, int]:
		case = self.describe(index)
		return torch.tensor(case['ids'], dtype=torch.long), case['sep']

	def __getitem__ (self, index):
		return self._item(self.indices[index])

	def __iter__ (self):
		indices = self.indices.copy()
		if self.shuffle:
			order = torch.randperm(len(indices)).tolist()
			indices = [indices[i] for i in order]
		for index in indices:
			yield self._item(index)

	def collateBatch (self, batch):
		sequences = [ex[0] for ex in batch]
		input_ids = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_id)
		masks = pad_sequence(
			[torch.ones(len(ids), dtype=torch.long) for ids in sequences], batch_first=True, padding_value=0)
		# The supervised region is everything strictly after <sep>, padding excluded. Built from the
		# recorded sep index rather than by searching for the id, so a <sep> that ever appeared inside
		# a half could not be mistaken for the boundary.
		target_mask = torch.zeros_like(input_ids)
		for row, (ids, sep) in enumerate(batch):
			target_mask[row, sep + 1:len(ids)] = 1
		sep_index = torch.tensor([ex[1] for ex in batch], dtype=torch.long)

		return dict(
			input_ids=input_ids.to(self.device),
			masks=masks.to(self.device),
			target_mask=target_mask.to(self.device),
			sep_index=sep_index.to(self.device),
		)

