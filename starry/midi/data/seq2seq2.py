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

`start_jitter` (default 0 = off) is an augmentation: for crops that do NOT begin at the piece's start, it
offsets the source crop's first line by round(gauss(0, start_jitter)). Every crop otherwise begins exactly
ON a mark line, which inference cannot reproduce — a sliding window over a production file has no
@measure/@tick to land on and starts mid-measure (see tools/midi/translateMidiseq2.py). The target stays
mark-aligned; only the source's leading context moves.

Unlike its siblings in this package (seq2CondPatchy, seq2CondSplitPatchy) this feeder reads TEXT at
runtime instead of a packed `.pt`, so a change of crop policy needs no re-pack. Note the consequence
for `splits`: the filter is POSITIONAL (`i % cycle in phases`), so the file list must be sorted
deterministically or val silently leaks into train.

Two on-disk layouts, chosen by `packed` (None = auto-detect, so existing configs are unaffected):

    loose    <root>/<arm>/<name>          one file per sample per arm — what the corpus is built as
    packed   <root>/<XX>.zip              one archive per SHARD, holding <arm>/<name> for both arms,
                                          where XX is the first two characters of the name

The packed form exists because midiseq2 is repetitive text: measured compression is 0.067 (score arm)
and 0.144 (irregular), taking nota1m's 124 G of loose files to ~15 G, and it replaces ~1.5M inodes with
~256 archives. It costs nothing at training time — parsing is 99% of the per-file cost (5.6 ms) and a
decompressed read is 0.1-0.4 ms of it. `tools/midi/packMidiseq2Shards.py` builds one, including the
`index.json` manifest it enumerates from. Both layouts produce byte-identical batches; that is what
tests/midi/seq2seq2_archive_check.py asserts.

`pos_style` picks the RoPE position convention (see `_positions`):

    'flat'      0, 1, 2, ... T-1                       the plain default
    'sep'       source ends at -2, <sep> = -1, target starts at 0
    'absolute'  <sep> = -1 as above, but each half sits on its OWN FILE's token axis: the target
                file's first token is 0 counting up, the source file's last token is -2 counting
                down, so a crop carries WHERE in the piece it came from.

'flat' and 'sep' emit different numbers but are equivalent to the model — both are one arithmetic run,
and RoPE reads only relative distance, so a uniform shift changes nothing. 'sep' buys readability (a
position's sign says which half it is in). Only 'absolute' changes what the model sees.

Batch contract:
	input_ids    LongTensor [B, T]   source ++ <sep> ++ target, right-padded with <pad>
	masks        LongTensor [B, T]   1 = real token
	target_mask  LongTensor [B, T]   1 = a supervised target position (strictly after <sep>). A crop
	                                 whose sampled start offset was nonzero (start_jitter) also drops
	                                 the target's first bar, up to and including its first <eom>: the
	                                 source is missing the head of that bar, so it is not derivable
	                                 from the context. 0 offset -> the whole target half is supervised
	sep_index    LongTensor [B]      position of <sep> in each row
	position_ids LongTensor [B, T]   RoPE positions per pos_style; the pad tail CONTINUES each row's
	                                 run rather than taking a constant, so a padded row stays
	                                 numerically identical to its unpadded self

Loss convention: for a next-token model, compare `logits[:, i - 1]` against `input_ids[:, i]` at
every `i` where target_mask is 1. `<sep>` is therefore the last context position before the first
supervised token, and nothing in the source half is ever a target.
'''

import json
import os
import random
import zipfile
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs
from ...utils.registry import register_dataset
from .seq2CondPachifier import Midiseq2Tokenizer


_FORMATS = ('midiseq2', 'lilylet')


def _format (value: str) -> str:
	value = value.lower()
	if value == 'midi':
		value = 'midiseq2'
	if value not in _FORMATS:
		raise ValueError(f'format must be one of {_FORMATS}, got {value!r}')
	return value


def _sample_id (name: str) -> str:
	base = os.path.basename(name)
	for suffix in ('.midiseq2.txt', '.lyl', '.txt'):
		if base.endswith(suffix):
			return base[:-len(suffix)]
	return os.path.splitext(base)[0]


def _recursive_files (root: str, suffix: str) -> Dict[str, str]:
	out: Dict[str, str] = {}
	if not os.path.isdir(root):
		return out
	for directory, _, files in os.walk(root):
		for name in files:
			if not name.endswith(suffix):
				continue
			path = os.path.join(directory, name)
			key = _sample_id(name)
			if key in out:
				raise ValueError(f'duplicate normalized sample id {key!r} under {root!r}')
			out[key] = path
	return out


# Position-id conventions, see Seq2Seq2._positions. 'flat' and 'sep' are equivalent to the model
# (RoPE is relative); only 'absolute' carries where the crop was taken from.
_POS_STYLES = ('flat', 'sep', 'absolute')

# A mark key is either a measure number ('measure' mode) or a (measure, tick) pair ('tick' mode).
MarkKey = Any
# (line_index, key) — the line the directive sits on, and what identifies it across files.
Mark = Tuple[int, MarkKey]


class _DirSource:
	'''The loose-file layout: `<root>/<arm>/<name>`, one file per sample per arm.

	`key` identifies the backing store in the parse cache. It is the absolute root, so two datasets
	built from the same directory share parsed files while two different corpora never collide.
	'''

	def __init__ (self, root: str):
		self.root = os.path.abspath(root)
		self.key = ('dir', self.root)

	def names (self, arm: str) -> List[str]:
		return [n for n in os.listdir(os.path.join(self.root, arm)) if n.endswith('.txt')]

	def read (self, arm: str, name: str) -> str:
		with open(os.path.join(self.root, arm, name), 'r', encoding='utf-8') as f:
			return f.read()

	def describe (self) -> str:
		return self.root


class _ZipSource:
	'''The sharded-archive layout: `<root>/<XX>.zip`, each holding `<arm>/<name>` for one shard.

	A shard is the first `SHARD_CHARS` characters of the name, so a sample's archive is computable from
	its name alone — no index is needed to route a read, only to enumerate.

	Both arms live in the SAME shard archive. That keeps a pair in one file (one open handle serves
	both halves of a sample) at the cost of having to rewrite a shard to regenerate one arm.

	Handles are opened lazily and held: measured 0.11 ms/read with the handle held against 5.42 ms
	reopening per read, a 50x difference, because reopening re-reads the central directory every time.
	They are keyed by PID so a forked DataLoader worker opens its own rather than inheriting a shared
	file offset — `dataset_factory` currently passes no `num_workers`, but a silently corrupt read is
	not the failure mode to leave armed.
	'''

	SHARD_CHARS = 2
	MANIFEST = 'index.json'
	# Open archives held per source instance — see _zip. 128 keeps two datasets (train + val) inside a
	# 1024 fd limit with room for torch's own descriptors, while still covering half a 256-shard corpus.
	MAX_HANDLES = 128

	def __init__ (self, root: str):
		self.root = os.path.abspath(root)
		self.key = ('zip', self.root)
		self._handles: 'OrderedDict[Tuple[int, str], zipfile.ZipFile]' = OrderedDict()
		self._manifest = self._load_manifest()

	@classmethod
	def detect (cls, root: str) -> bool:
		'''True when `root` looks like a packed corpus: a manifest, or at least one `<XX>.zip`.'''
		if not os.path.isdir(root):
			return False
		if os.path.isfile(os.path.join(root, cls.MANIFEST)):
			return True
		return any(n.endswith('.zip') and len(n) == cls.SHARD_CHARS + 4 for n in os.listdir(root))

	def shard_of (self, name: str) -> str:
		return name[:self.SHARD_CHARS]

	def _load_manifest (self) -> Optional[Dict[str, Any]]:
		'''The manifest is JSON, not YAML, and deliberately so.

		Every other packed dataset in this repo ships `index.yaml` (events.py, semantics.py,
		scoreFault.py). That does not scale to this corpus: `yaml.safe_load` of a 200k-name index took
		16.4 s against 50 ms for the same data as JSON, a 330x difference. At ~1.5M names the YAML
		convention would cost minutes per process start.

		Absent, enumeration falls back to reading every shard's central directory (measured 10.3 s over
		256 shards) — correct but slower, so a packed corpus should carry one.
		'''
		path = os.path.join(self.root, self.MANIFEST)
		if not os.path.isfile(path):
			return None
		with open(path, 'r', encoding='utf-8') as f:
			return json.load(f)

	def _zip (self, shard: str) -> zipfile.ZipFile:
		'''An open handle for `shard`, LRU-bounded because handles are file descriptors.

		One ZipFile holds one fd. A 256-shard corpus read by both the train and val datasets wants 512,
		against a soft `ulimit -n` of 1024 on the training box — plus torch's and CUDA's own fds. Rather
		than leave that to chance, the least-recently-used archive is closed once the bound is reached.

		The bound is per source instance, so it is a per-dataset budget. It trades a reopen (measured
		5.4 ms, against 0.11 ms for a held handle) for an fd, and only on a shard that has not been
		touched in the last MAX_HANDLES reads — with a shuffled sampler over a large corpus that is rare
		enough not to matter, and the alternative is EMFILE mid-epoch.
		'''
		key = (os.getpid(), shard)
		handle = self._handles.get(key)
		if handle is None:
			handle = zipfile.ZipFile(os.path.join(self.root, f'{shard}.zip'), 'r')
			self._handles[key] = handle
			while len(self._handles) > self.MAX_HANDLES:
				_, stale = self._handles.popitem(last=False)
				stale.close()
		else:
			self._handles.move_to_end(key)
		return handle

	def names (self, arm: str) -> List[str]:
		'''Every name present for `arm`, from the manifest when there is one.

		The manifest's per-shard lists are the pair intersection the packer wrote, so both arms return
		the same set and the caller's intersection is a no-op. Without a manifest each archive's central
		directory is read instead, which is where the two arms can legitimately differ.
		'''
		if self._manifest is not None:
			out: List[str] = []
			for names in self._manifest['shards'].values():
				out.extend(names)
			return out
		prefix = f'{arm}/'
		out = []
		for entry in sorted(os.listdir(self.root)):
			if not entry.endswith('.zip'):
				continue
			shard = entry[:-4]
			out.extend(n[len(prefix):] for n in self._zip(shard).namelist()
				if n.startswith(prefix) and n.endswith('.txt'))
		return out

	def read (self, arm: str, name: str) -> str:
		with self._zip(self.shard_of(name)).open(f'{arm}/{name}', 'r') as f:
			return f.read().decode('utf-8')

	def describe (self) -> str:
		shards = len(self._manifest['shards']) if self._manifest else '?'
		return f'{self.root} ({shards} shards, manifest={self._manifest is not None})'


def _make_source (root: str, packed: Optional[bool] = None):
	'''Pick the layout. `packed=None` auto-detects, which is what keeps existing configs untouched.'''
	if packed is None:
		packed = _ZipSource.detect(root)
	return _ZipSource(root) if packed else _DirSource(root)


class _File:
	'''One parsed midiseq2 file: its lines, its marks, and a key -> line-numbers index.

	`marks` is in file order. `lines_of` maps a key to EVERY line carrying it, because the key is not
	unique in general: the irregular arm has 730 duplicate (measure, tick) keys across 15 files (the
	perturbation can move two events onto the same score tick). `_align` relies on that list.

	`token_before[k]` is how many CONTENT tokens precede line k in the whole file — the prefix sum
	`pos_style='absolute'` needs to place a crop on the file's own token axis. Directive lines
	contribute 0, matching _encode, so this counts exactly the tokens _encode would emit for the whole
	file. It is a prefix sum rather than a per-crop rescan because describe() is called once per sample
	per epoch, and files run to 76k tokens.
	'''

	def __init__ (self, text: str, mark_mode: str):
		self.lines: List[str] = text.splitlines()
		self.marks: List[Mark] = []
		self.lines_of: Dict[MarkKey, List[int]] = {}
		# len(lines) + 1 entries, so token_before[len(lines)] is the file's total token count.
		self.token_before: List[int] = [0] * (len(self.lines) + 1)
		# The measure number is tracked as parse state in BOTH modes: in 'tick' mode an @measure line
		# is not itself a mark, but it still tells us which measure the following @tick values are in
		# (a bare tick repeats every bar and would collide across the piece).
		measure: Optional[int] = None
		for index, line in enumerate(self.lines):
			if line.startswith('@measure'):
				measure = int(line.split()[1])
				if mark_mode != 'measure':
					self.token_before[index + 1] = self.token_before[index]
					continue
				key: MarkKey = measure
			elif line.startswith('@tick'):
				if mark_mode != 'tick':
					self.token_before[index + 1] = self.token_before[index]
					continue
				key = (measure, int(line.split()[1]))
			else:
				# Content: every whitespace-separated token becomes one id in _encode.
				self.token_before[index + 1] = self.token_before[index] + len(line.split())
				continue
			# A mark line is a directive, so it contributes no content token either.
			self.token_before[index + 1] = self.token_before[index]
			self.marks.append((index, key))
			self.lines_of.setdefault(key, []).append(index)

	def __len__ (self) -> int:
		return len(self.lines)


# Parsed files are shared across dataset instances built from the same source within a process, so the
# train and val splits of one config do not each hold their own copy. Keyed by
# (source_key, arm, name, mark_mode) — the marks depend on the mode, and the source key keeps two
# corpora with colliding basenames apart.
#
# BOUNDED, and it has to be. A parsed _File costs ~11.6x its on-disk size (lines, marks, lines_of and
# the token_before prefix sum are all Python objects), so caching a full nota1m arm — 736k files — would
# want ~400 GB of RAM. Unbounded was fine only while every run used one 3.5k-sample shard.
#
# The bound costs little because the cache's value falls away with corpus size: at epoch_size 1200 a
# given file is revisited every ~3 epochs at shard-00 scale but only every ~509 epochs over the full
# corpus, so beyond a few thousand entries an unbounded cache mostly holds files it will not see again.
# Re-reading is cheap next to re-parsing anyway — parsing is 99% of the 5.6 ms per-file cost, and a
# decompressed read is 0.1-0.4 ms of it.
_FILE_CACHE: 'OrderedDict[Tuple[Any, str, str, str], _File]' = OrderedDict()

# Entries, not bytes. Datasets raise it via `max_cached_files`; the default holds a shard-sized working
# set (~2 arms x 3.6k names) without approaching the memory wall.
_CACHE_LIMIT = 8192


def _set_cache_limit (limit: int) -> None:
	'''Raise the shared bound. Never lowers it: two datasets share this cache, and the one asking for
	less would otherwise evict the other's working set.'''
	global _CACHE_LIMIT
	if limit > 0:
		_CACHE_LIMIT = max(_CACHE_LIMIT, limit)


def _get_file (source, arm: str, name: str, mark_mode: str) -> _File:
	key = (source.key, arm, name, mark_mode)
	parsed = _FILE_CACHE.get(key)
	if parsed is None:
		parsed = _File(source.read(arm, name), mark_mode)
		_FILE_CACHE[key] = parsed
		while len(_FILE_CACHE) > _CACHE_LIMIT:
			_FILE_CACHE.popitem(last=False)		# evict least-recently-used
	else:
		_FILE_CACHE.move_to_end(key)
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
		max_tokens=0, resample_tries=8, align_retries=4, start_jitter=0.0,
		random_crop=None, seed=0, vocab_path=None, pos_style='flat',
		packed=None, max_cached_files=0, source_format='midiseq2',
		target_format='midiseq2', measures_path=None, **_):
		super().__init__()
		self.device = device
		self.shuffle = shuffle
		self.source_format = _format(source_format)
		self.target_format = _format(target_format)
		if self.source_format == self.target_format == 'lilylet':
			raise ValueError('Lilylet -> Lilylet is not supported; at least one side must be midiseq2')
		self.mixed = self.source_format != self.target_format
		if mark_mode not in ('measure', 'tick'):
			raise ValueError(f'mark_mode must be "measure" or "tick", got {mark_mode!r}')
		self.mark_mode = mark_mode
		if pos_style not in _POS_STYLES:
			raise ValueError(f'pos_style must be one of {sorted(_POS_STYLES)}, got {pos_style!r}')
		self.pos_style = pos_style
		# The backing store. A directory root and a packed root differ only here: `packed=None`
		# auto-detects, so an existing config keeps reading loose files with no change.
		self.source = _make_source(root, packed)
		self.arm_source = source_dir
		self.arm_target = target_dir
		# Retained because callers reach for them (the check suite and the notebook build a sibling
		# dataset out of them). Under a packed root they name a path that does not exist on disk, so
		# they are for display and for `os.path.basename` only — read through `self.source`.
		self.source_root = os.path.join(root, source_dir)
		self.target_root = os.path.join(root, target_dir)
		_set_cache_limit(max_cached_files)
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
		# Augmentation: std dev (in LINES) of a normal offset applied to the source crop's start, for
		# crops that do NOT begin at the piece's start. 0 = off.
		#
		# Why it exists: every crop this feeder builds starts exactly ON a mark line, because growth
		# steps whole marks. Inference cannot do that — a sliding window over a production file has no
		# @measure/@tick to land on (tools/midi/translateMidiseq2.py advances the source cursor by
		# matching note_on counts instead), so its windows start mid-measure at an arbitrary line. The
		# model therefore meets an input distribution at inference that training never showed it.
		# Jittering the start teaches it that the source may begin anywhere.
		#
		# Head crops are excluded: a == 0 IS the <bos> condition, and moving that start would either
		# claim start-of-piece while skipping the header or contradict the <bos> the assembler emits.
		# The target is NOT jittered — it stays mark-aligned, which is the supervision signal.
		if start_jitter < 0:
			raise ValueError(f'start_jitter must be >= 0, got {start_jitter!r}')
		self.start_jitter = float(start_jitter)
		# Deterministic crops for val: default follows the split's shuffle flag (as m3distill does),
		# so train augments and val is reproducible epoch to epoch.
		self.random_crop = shuffle if random_crop is None else random_crop
		self.seed = seed
		self._split = split
		if self.mixed:
			if packed:
				raise ValueError('packed archives are supported only for midiseq2 -> midiseq2')
			if mark_mode != 'measure':
				raise ValueError('mixed Lilylet alignment requires mark_mode="measure"')
			from .unifiedSeq2Tokenizer import UnifiedSeq2Tokenizer
			from ...lilylet.data.patchifier import LilyletTokenizer
			self.tokenizer = UnifiedSeq2Tokenizer(vocab_path) if vocab_path else UnifiedSeq2Tokenizer()
			lilylet_asset = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
				os.path.dirname(os.path.abspath(__file__))))), 'assets', 'lilylet-tokenizer.json')
			self.lilylet_tokenizer = LilyletTokenizer(lilylet_asset)
		else:
			self.tokenizer = Midiseq2Tokenizer(vocab_path) if vocab_path else Midiseq2Tokenizer()

		if self.mixed:
			self._init_mixed(root, source_dir, target_dir, measures_path)
		else:
			# The split filter is positional, so the file list MUST be deterministically ordered or the
			# train/val partition shifts with directory iteration order.
			names = sorted(
				set(self.source.names(self.arm_source)) & set(self.source.names(self.arm_target))
			)
			self.names = [name for name in names if name.endswith('.txt')]
			if not self.names:
				raise RuntimeError(f'no shared .txt basenames between {self.arm_source!r} and '
					f'{self.arm_target!r} in {self.source.describe()}')

		phases, cycle = parseFilterStr(split)
		self.indices = [i for i in range(len(self.names)) if i % cycle in phases]

	def __len__ (self) -> int:
		return len(self.indices)

	def _init_mixed (self, root: str, source_dir: str, target_dir: str,
		measures_path: Optional[str]) -> None:
		'''Discover cross-format pairs and validate repeat-aware measure maps.'''
		lyl_dir = source_dir if self.source_format == 'lilylet' else target_dir
		midi_dir = source_dir if self.source_format == 'midiseq2' else target_dir
		self._lyl_files = _recursive_files(os.path.join(root, lyl_dir), '.lyl')
		self._midi_files = _recursive_files(os.path.join(root, midi_dir), '.txt')
		path = measures_path or os.path.join(root, 'metadata', 'measures.json')
		with open(path, 'r', encoding='utf-8') as f:
			all_measures = json.load(f)
		if not isinstance(all_measures, dict):
			raise ValueError(f'{path!r} must contain an object keyed by normalized sample id')
		self._measure_maps: Dict[str, List[int]] = {}
		for sample_id in sorted(set(self._lyl_files) & set(self._midi_files)):
			record = all_measures.get(sample_id)
			if not isinstance(record, dict) or record.get('ok') is False:
				continue
			rows = record.get('measures')
			if not isinstance(rows, list) or not rows:
				continue
			mapping: List[int] = []
			for expected, row in enumerate(rows, 1):
				if not isinstance(row, dict) or row.get('index') != expected:
					raise ValueError(f'{path}: {sample_id}: played measure indices must be contiguous from 1')
				source_measure = row.get('source_measure')
				if not isinstance(source_measure, int) or source_measure < 1:
					raise ValueError(f'{path}: {sample_id}: invalid source_measure at played measure {expected}')
				mapping.append(source_measure)
			self._measure_maps[sample_id] = mapping
		self.names = sorted(self._measure_maps)
		if not self.names:
			raise RuntimeError(f'no valid Lilylet/midiseq2 pairs between {lyl_dir!r} and {midi_dir!r}')

	def _mixed_lilylet_measures (self, sample_id: str) -> List[str]:
		from ...lilylet.data.patchifier import split_lilylet_document, split_measures
		with open(self._lyl_files[sample_id], 'r', encoding='utf-8') as f:
			# Header and style metadata are not score measures and must not enter alignment.
			_, body_lines = split_lilylet_document(f.read())
		measures = split_measures(body_lines)
		maximum = max(self._measure_maps[sample_id])
		if len(measures) < maximum:
			raise ValueError(f'{sample_id}: measures.json references source measure {maximum}, '
				f'but the Lilylet body has only {len(measures)} measures')
		return measures

	def _mixed_midi (self, sample_id: str) -> _File:
		with open(self._midi_files[sample_id], 'r', encoding='utf-8') as f:
			parsed = _File(f.read(), 'measure')
		keys = [key for _, key in parsed.marks]
		expected = list(range(1, len(self._measure_maps[sample_id]) + 1))
		if keys != expected:
			raise ValueError(f'{sample_id}: MIDI @measure marks do not match measures.json played indices')
		return parsed

	def _mixed_pick (self, count: int, rng: random.Random) -> Tuple[int, int]:
		maximum = min(count, max(1, self.line_range[1]))
		minimum = min(maximum, self.line_range[0])
		length = minimum if minimum == maximum else rng.randint(minimum, maximum)
		roll = rng.random()
		if roll < self.p_head:
			start = 0
		elif roll < self.p_head + self.p_tail:
			start = count - length
		else:
			start = rng.randrange(0, count - length + 1)
		return start, start + length

	def _mixed_midi_text (self, midi: _File, played: Sequence[int]) -> List[str]:
		lines: List[str] = []
		for number in played:
			start = midi.marks[number - 1][0]
			end = midi.marks[number][0] if number < len(midi.marks) else len(midi.lines)
			lines.extend(midi.lines[start:end])
		return lines

	def _mixed_encode_midi (self, lines: Sequence[str], eom: bool) -> List[int]:
		lookup = {self.tokenizer.tokens[self.tokenizer.midiseq2_offset + i]: self.tokenizer.midi_id(i)
			for i in range(self.tokenizer.blocks['midiseq2']['size'])}
		ids: List[int] = []
		for line in lines:
			if line.startswith('@measure'):
				if eom and line.split()[1] != '1':
					ids.append(self.tokenizer.midiseq2_eom_id)
				continue
			if line.startswith('@tick'):
				continue
			ids.extend(lookup.get(token, self.tokenizer.midiseq2_unknown_id) for token in line.split())
		return ids

	def _mixed_positions (self, n_source: int, n_target: int) -> List[int]:
		if self.pos_style == 'flat':
			return list(range(n_source + 1 + n_target))
		return list(range(-(n_source + 1), n_target))

	def _describe_mixed_once (self, index: int, rng: random.Random) -> Dict[str, Any]:
		sample_id = self.names[index]
		lyl = self._mixed_lilylet_measures(sample_id)
		midi = self._mixed_midi(sample_id)
		mapping = self._measure_maps[sample_id]
		if self.source_format == 'lilylet':
			a, z = self._mixed_pick(len(lyl), rng)
			source_numbers = list(range(a + 1, z + 1))
			wanted = set(source_numbers)
			played = [i for i, source_number in enumerate(mapping, 1) if source_number in wanted]
			if not played:
				raise ValueError(f'{sample_id}: Lilylet crop has no played measures in measures.json')
			source_body = self.lilylet_tokenizer.encode(''.join(lyl[n - 1] for n in source_numbers))
			target_body = self._mixed_encode_midi(self._mixed_midi_text(midi, played), True)
			head = a == 0
			source_range, target_range = (a, z), (min(played) - 1, max(played))
		else:
			a, z = self._mixed_pick(len(mapping), rng)
			played = list(range(a + 1, z + 1))
			source_numbers = [mapping[n - 1] for n in played]
			source_body = self._mixed_encode_midi(self._mixed_midi_text(midi, played), self.source_eom)
			target_body = self.lilylet_tokenizer.encode(''.join(lyl[n - 1] for n in source_numbers))
			head = a == 0
			source_range, target_range = (a, z), (min(source_numbers) - 1, max(source_numbers))
		source_bos = self.tokenizer.bos_id if self.source_format == 'lilylet' else self.tokenizer.midiseq2_bos_id
		target_bos = self.tokenizer.bos_id if self.target_format == 'lilylet' else self.tokenizer.midiseq2_bos_id
		target_eos = self.tokenizer.eos_id if self.target_format == 'lilylet' else self.tokenizer.midiseq2_eos_id
		source_ids = ([source_bos] if head else []) + source_body
		target_ids = ([target_bos] if head else []) + target_body + [target_eos]
		ids = source_ids + [self.tokenizer.sep_id] + target_ids
		sep = len(source_ids)
		return dict(name=sample_id, source=midi if self.source_format == 'midiseq2' else lyl,
			target=midi if self.target_format == 'midiseq2' else lyl, a=a, z=z, jitter=0, skip=0,
			source_range=source_range, target_range=target_range, ids=ids, sep=sep,
			positions=self._mixed_positions(len(source_ids), len(target_ids)), head=head,
			tail=z >= (len(lyl) if self.source_format == 'lilylet' else len(mapping)),
			pos_bos_on_sep=False, source_measures=source_numbers if self.source_format == 'lilylet' else played,
			target_measures=played if self.target_format == 'midiseq2' else source_numbers)

	def _describe_mixed (self, index: int) -> Dict[str, Any]:
		rng = random if self.random_crop else random.Random(self.seed ^ (index * 2654435761))
		best = None
		for _ in range(max(1, self.resample_tries)):
			case = self._describe_mixed_once(index, rng)
			if best is None or len(case['ids']) < len(best['ids']):
				best = case
			if not self.max_tokens or len(case['ids']) <= self.max_tokens:
				return case
		return best

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

	def _bounds (self, source: _File, a: int, z: int, jitter: int = 0) -> Tuple[int, int]:
		'''Mark range -> source line range. `jitter` shifts the START only (see _pick_jitter).

		The default 0 is what _pick_crop's growth loop uses: it measures candidate spans, and a jittered
		measurement would make the realized length depend on an offset drawn for a different purpose.
		'''
		start = 0 if a == 0 else source.marks[a][0]
		end = len(source.lines) if z >= len(source.marks) else source.marks[z][0]
		if jitter and a != 0:
			# Clamp inside the file and keep at least one line: a start at or past `end` would encode an
			# empty source half, and a negative one would index from the tail.
			start = max(0, min(start + jitter, end - 1))
		return start, end

	def _supervise_from (self, target_ids: List[int], jitter: int) -> int:
		'''Index into the TARGET half where supervision starts. 0 = supervise all of it.

		A jittered source loses the head of its first measure, so the target's first bar is no longer
		derivable from what the model can see — supervising it would train the model to invent the part
		that was cropped away. Everything from the first <eom> on is still fully covered, so that is
		where supervision begins.

		Keyed on the SAMPLED offset, not on the `start_jitter` setting: the draw is normal about 0, so a
		crop can come out at exactly 0 even with jitter enabled (~27% of them at std 8, since the offset
		is rounded to a whole line). Those crops are still mark-aligned and lose nothing, so they keep
		full supervision.

		Returns 0 when the offset is 0 (nothing was lost) and when the target holds no <eom> (there is no
		bar boundary to skip to). The latter cannot arise from an offset crop: describe() cancels the
		offset for exactly those crops, because excluding "up to the first <eom>" would otherwise exclude
		the entire target and leave the sample with an empty mask.
		'''
		if not jitter:
			return 0
		try:
			return target_ids.index(self.tokenizer.eom_id) + 1
		except ValueError:
			return 0

	def _pick_jitter (self, a: int, rng: random.Random) -> int:
		'''Normal offset (in lines) for the source crop's start. 0 for head crops and when disabled.

		Symmetric about the mark, so the start can fall either side of it — the point is that the model
		stops being able to assume the first line it sees opens a measure. `_align` deliberately keeps
		using the UNJITTERED mark `a` to place the target, so the target window still covers the same
		music and only the source's leading context is perturbed.
		'''
		if not self.start_jitter or a == 0:
			return 0
		return round(rng.gauss(0, self.start_jitter))

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

	def _encode (self, lines: Sequence[str], eom: bool, base: int = 0) -> Tuple[List[int], List[int]]:
		'''Lines -> (ids, token_index).

		Directives never become content tokens; with `eom` on, an @measure line contributes one <eom>
		instead. @measure 1 is skipped: it opens the piece rather than closing a bar, and <bos> already
		carries that.

		`token_index[i]` is the running token index of ids[i] on the FILE's own axis, starting from
		`base` (pass the crop's `token_before[start]` to get file-absolute indices). Content tokens
		advance it; an <eom> takes the next index too, since it occupies a slot in the sequence and must
		therefore occupy a position. Returning it from here rather than deriving it from `token_before`
		is what keeps positions and ids the same length — `token_before` counts content only, so it
		undercounts a target half by exactly its <eom> count.
		'''
		ids: List[int] = []
		index: List[int] = []
		cursor = base
		lookup = self.tokenizer.id_by_token
		unknown = self.tokenizer.unknown_id
		for line in lines:
			if line.startswith('@measure'):
				if eom and line.split()[1] != '1':
					ids.append(self.tokenizer.eom_id)
					index.append(cursor)
					cursor += 1
				continue
			if line.startswith('@tick'):
				continue
			for token in line.split():
				ids.append(lookup.get(token, unknown))
				index.append(cursor)
				cursor += 1
		return ids, index

	def _positions (self, source: _File, target: _File, src_index: Sequence[int],
		tgt_index: Sequence[int], head: bool, n_source: int, n_target: int) -> List[int]:
		'''Position ids for the assembled sequence, per `pos_style`. Length is n_source + 1 + n_target
		(the +1 being <sep>), so it lines up with the ids one-for-one.

		  'flat'      0, 1, 2, ... T-1                             the plain default
		  'sep'       source ends at -2, <sep> = -1, target starts at 0
		  'absolute'  <sep> = -1 as above, but each half is placed on its OWN FILE's token axis:
		              the target FILE's first token is 0 and counts up, the source FILE's last token
		              is -2 and counts down. A crop therefore carries WHERE it was taken from.

		RoPE reads only relative distance, so 'flat' and 'sep' produce byte-identical model output —
		both are one arithmetic run, and a uniform shift is invisible to RoPE. 'sep' buys readability
		(a position's sign says which half it is in), not behaviour. Only 'absolute' changes what the
		model sees, because the gap between the halves then varies per crop.

		Two properties make 'absolute' safe, and they are the reason it is preferred over adding a fixed
		constant offset to the source: the source is always <= -2 and the target always >= 0, so the
		halves can never collide on a position id, and can never invert their order. A fixed offset
		cannot promise either once a file is longer than the offset (measured on this corpus, a -N/2
		offset put 0.4% of crops in collision and 1.25% inverted).
		'''
		if self.pos_style == 'flat':
			return list(range(n_source + 1 + n_target))

		if self.pos_style == 'sep':
			# One contiguous run through -2, -1, 0: 'flat' shifted by -(n_source + 1), nothing more.
			return list(range(-(n_source + 1), n_target))

		# --- 'absolute' ---------------------------------------------------------------------
		# Each content token keeps its own file's token index; only the structural wrappers are placed
		# relative to them, taking the slot just outside the content they wrap.
		#
		# target: the file's FIRST token is 0, so a crop's index is used as-is and a later crop sits
		#         further right.
		# source: the file's LAST token is -2, so an index p maps to p - L - 1 (L = the file's total
		#         token count) and an earlier crop sits further left.
		total_src = source.token_before[len(source.lines)]
		src_pos = [p - total_src - 1 for p in src_index]
		tgt_pos = list(tgt_index)

		# <bos> precedes its half's content, <eos> follows the target's. Both are derived from the
		# neighbouring content position rather than assumed, so an empty half cannot produce a gap.
		if head:
			src_pos = [(src_pos[0] if src_pos else -2) - 1] + src_pos
			tgt_pos = [(tgt_pos[0] if tgt_pos else 0) - 1] + tgt_pos
		tgt_pos = tgt_pos + [(tgt_pos[-1] if tgt_pos else -1) + 1]

		# A head crop has t_start == 0 (true of every head crop in this corpus), so the target's content
		# starts at 0 and its <bos> takes -1 — the same position as <sep>. Both are structural markers
		# on the same boundary and their embeddings still tell them apart; what matters is that no two
		# CONTENT tokens ever share a position. describe() reports this as `pos_bos_on_sep`.
		assert len(src_pos) == n_source and len(tgt_pos) == n_target, (
			f'position/id length mismatch: {len(src_pos)} vs {n_source}, {len(tgt_pos)} vs {n_target}')
		return src_pos + [-1] + tgt_pos

	def _assemble (self, source: _File, target: _File, a: int, z: int,
		align: Tuple[int, int], jitter: int = 0) -> Tuple[List[int], int, List[int]]:
		'''Build the joined id sequence, the index of its <sep>, and the position ids.'''
		s_start, s_end = self._bounds(source, a, z, jitter)
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

		src_body, src_index = self._encode(
			source.lines[s_start:s_end], self.source_eom, source.token_before[s_start])
		tgt_body, tgt_index = self._encode(
			target.lines[t_start:t_end], True, target.token_before[t_start])
		source_ids = wrap(src_body)
		target_ids = wrap(tgt_body) + [self.tokenizer.eos_id]
		positions = self._positions(source, target, src_index, tgt_index, head,
			len(source_ids), len(target_ids))
		return source_ids + [self.tokenizer.sep_id] + target_ids, len(source_ids), positions

	# --- item -----------------------------------------------------------------------------

	def describe (self, index: int) -> Dict[str, Any]:
		if self.mixed:
			return self._describe_mixed(index)
		'''The whole crop decision for one sample, ids included.

		`_item` is a thin wrapper over this. Visualization and diagnostics need what the id sequence
		cannot carry — which lines were cropped, which marks bound them, and above all the MEASURE
		NUMBERS, since _encode turns @measure N into a bare <eom>. Returning it from the feeder rather
		than re-deriving it outside keeps the two from drifting apart.

		Keys: name, source, target (_File), a, z (source mark range), jitter (the start offset in lines,
		0 unless start_jitter is on), source_range, target_range (line slices), ids, sep, head, tail,
		source_measures, target_measures — the latter two being [(line_index, measure_number)] for every
		@measure line inside that half's range, @measure 1 included (it is a real bar number even though
		it emits no <eom>).

		`source_range` already has the jitter applied, so it is the range the ids were built from.
		'''
		name = self.names[index]
		source = _get_file(self.source, self.arm_source, name, self.mark_mode)
		target = _get_file(self.source, self.arm_target, name, self.mark_mode)
		# A deterministic crop still varies BY SAMPLE (so val covers head/tail/middle) but not by
		# epoch; seeding on the index is what gives both.
		rng = random if self.random_crop else random.Random(self.seed ^ (index * 2654435761))

		# The crop that wins is kept WHOLE — ids together with the (a, z, align) that produced them.
		# Keeping only the ids would leave the ranges describing whichever attempt happened to be last.
		best: Optional[Tuple[List[int], int, List[int], int, int, Tuple[int, int], int]] = None
		for attempt in range(max(1, self.resample_tries)):
			a, z = self._pick_crop(source, rng)
			# Drawn per attempt and carried with the crop, so the reported source_range is the one the
			# ids were actually built from rather than a re-draw.
			jitter = self._pick_jitter(a, rng)
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
			# Both fallbacks above can move `a` to 0, which IS the head condition; re-derive so a jitter
			# drawn for an interior crop cannot survive onto a head one and shift <bos> off line 0.
			if a == 0:
				jitter = 0
			ids, sep, positions = self._assemble(source, target, a, z, align, jitter)
			# A jittered crop drops its first bar from supervision (see _supervise_from), which needs an
			# <eom> to mark where the bar ends. 26% of target halves have none — a crop can sit inside a
			# single bar — and excluding "up to the first <eom>" would then exclude everything and leave
			# the sample with an empty mask (a nan loss). Drop the JITTER instead of the supervision: the
			# crop reverts to mark-aligned, which is always a valid sample.
			if jitter and self.tokenizer.eom_id not in ids[sep + 1:]:
				jitter = 0
				ids, sep, positions = self._assemble(source, target, a, z, align, jitter)
			if best is None or len(ids) < len(best[0]):
				best = (ids, sep, positions, a, z, align, jitter)
			if not self.max_tokens or len(ids) <= self.max_tokens:
				break
		# If every attempt overshot max_tokens we keep the SHORTEST one rather than truncating: a
		# truncated tail would leave the target unterminated and unaligned with its final mark.
		ids, sep, positions, a, z, align, jitter = best
		s_start, s_end = self._bounds(source, a, z, jitter)
		t_start, t_end = align
		# <bos> can share <sep>'s position id under 'absolute' (see _positions); reported rather than
		# hidden, since it is the one place two tokens coincide.
		# Where supervision starts INSIDE the target half (0 = all of it). Nonzero only when this crop's
		# SAMPLED offset is nonzero — a crop that drew 0 is mark-aligned and keeps full supervision.
		skip = self._supervise_from(ids[sep + 1:], jitter)
		return dict(name=name, source=source, target=target, a=a, z=z, jitter=jitter, skip=skip,
			source_range=(s_start, s_end), target_range=(t_start, t_end),
			ids=ids, sep=sep, positions=positions, head=a <= 0, tail=z >= len(source.marks),
			pos_bos_on_sep=self.pos_style == 'absolute' and a <= 0 and positions[sep + 1] == -1,
			source_measures=_measures_in(source, s_start, s_end),
			target_measures=_measures_in(target, t_start, t_end))

	def _item (self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
		'''(ids, sep, positions, skip). `skip` is how many leading TARGET tokens go unsupervised —
		0 for every unjittered crop, so the tuple's first three elements are unchanged.'''
		case = self.describe(index)
		return (torch.tensor(case['ids'], dtype=torch.long), case['sep'],
			torch.tensor(case['positions'], dtype=torch.long), case['skip'])

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
		#
		# `skip` moves that start further right for a crop whose sampled start offset was nonzero: its
		# source is missing the head of its first bar, so the target's first bar is not derivable from
		# what the model sees and supervising it would teach invention. It is 0 for every mark-aligned
		# crop, which is all of them when start_jitter is off.
		target_mask = torch.zeros_like(input_ids)
		for row, (ids, sep, _, skip) in enumerate(batch):
			target_mask[row, sep + 1 + skip:len(ids)] = 1
		sep_index = torch.tensor([ex[1] for ex in batch], dtype=torch.long)
		# Padded slots CONTINUE each row's run rather than taking a constant fill. A constant would put
		# a real position (e.g. 0, which every style uses) on a pad slot and break the arithmetic run,
		# which is measurable: RoPE rotates every position before the attention mask drops it, so a
		# padded 'flat' batch stopped matching its unpadded self by 4e-2 until this continued instead.
		# The mask still excludes these slots from attention; this only keeps the geometry consistent.
		width = input_ids.shape[1]
		position_ids = torch.stack([
			torch.cat([ex[2], torch.arange(1, width - len(ex[2]) + 1) + ex[2][-1]])
			if len(ex[2]) < width else ex[2]
			for ex in batch
		])

		return dict(
			input_ids=input_ids.to(self.device),
			masks=masks.to(self.device),
			target_mask=target_mask.to(self.device),
			sep_index=sep_index.to(self.device),
			position_ids=position_ids.to(self.device),
		)

