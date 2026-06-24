'''
MIDI-text -> NotaGen/bGPT patch artifact packing.

Counterpart of starry/lilylet/data/patchifier.py for the MIDI-text modality. Walks a
directory of MidiText `.txt` files (music-widgets' line-per-event MIDI serialization),
tokenizes each into fixed-size event patches with starry.midi.tokenizer.MidiTokenizer
(ONE EVENT = ONE PATCH), and writes a NotaGen-style `.pt` artifact the MidiPatchy
dataset (starry/midi/data/patchy.py) consumes.

Key difference from the Lilylet packer
---------------------------------------
Lilylet documents are short (a few hundred patches) so the packer bakes ONE fixed
window per file at pack time (`patch_length` cap with a random head/tail/middle crop).
MIDI songs are long and uneven (hundreds to tens of thousands of events). Cropping at
pack time would freeze one window per song for the whole run. Instead we store the
FULL patch sequence per file (uint8, compact) and let MidiPatchy random-crop a
`patch_length` window at load time — every epoch sees a different slice of each song.

Artifact layout matches the Lilylet packer so the dataset's _ItemStore can be reused
verbatim (single-file version 1, or sharded version 2 with a lazy-loaded index).

  item = dict(path=<rel .txt path>, patches=uint8[T, patch_size], dropped=int)

`dropped` is the count of excluded/unrecognized event lines (text/sysex/...), kept for
corpus transparency (the Lilylet packer stores per-file `unknowns`; MIDI's tokenizer
drops whole non-hex events instead of emitting <unknown>, so we record the drop count).
'''

import os
from typing import Any, Dict, List

import torch

from ..tokenizer import MidiTokenizer


def find_midi_text_files (source_dir: str) -> List[str]:
	'''All `.txt` files under source_dir (MidiText documents), sorted for determinism.'''
	results: List[str] = []
	for root, _, files in os.walk(source_dir):
		for name in files:
			if name.endswith('.txt'):
				results.append(os.path.join(root, name))
	return sorted(results)


def patchify_midi_text (
	text: str,
	tokenizer: MidiTokenizer,
) -> (torch.Tensor, int):
	'''Encode one MidiText document into a uint8 patch tensor [T, patch_size].

	Stores the FULL event sequence (framed by <bos>/<eos> patches) — NO patch_length
	cap here; MidiPatchy crops a window at load time. Returns (patches, dropped) where
	dropped is the number of excluded/unrecognized event lines.
	'''
	patches, dropped = tokenizer.encode_patches(text, add_special_patches=True)
	# Token ids fit in 0..vocab_size-1 (37 << 256), so store compactly as uint8, exactly
	# like the Lilylet packer. The per-item mask is always all-ones (real padding only at
	# batch time), so it is not stored; MidiPatchy reconstructs it from the patch count.
	tensor = torch.tensor(patches, dtype=torch.uint8) if patches else torch.zeros((0, tokenizer.patch_size), dtype=torch.uint8)
	return tensor, len(dropped)


def _shard_path (output_path: str, shard_index: int) -> str:
	# foo.midi-notagen.pt -> foo.midi-notagen.shard00000.pt
	base, ext = os.path.splitext(output_path)
	return f'{base}.shard{shard_index:05d}{ext}'


def _make_item (file_path: str, source_dir: str, tokenizer: MidiTokenizer):
	with open(file_path, 'r', encoding='utf-8') as f:
		text = f.read()
	patches, dropped = patchify_midi_text(text, tokenizer)
	rel = os.path.relpath(file_path, source_dir)
	return dict(path=rel, patches=patches, dropped=dropped), dropped


# --- parallel worker plumbing (one tokenizer per process) -----------------------
_WORKER: Dict[str, Any] = {}


def _worker_init (patch_size, source_dir):
	_WORKER['tokenizer'] = MidiTokenizer(patch_size=patch_size)
	_WORKER['source_dir'] = source_dir


def _worker_process_range (task):
	'''Patchify a contiguous slice of files AND write its own shards to disk.
	Returns (worker_index, [(tmp_shard_path, count), ...], dropped_total). No item data
	flows back to the parent, so the parent is never a serialization bottleneck.'''
	wid, file_list, output_path, shard_size = task
	tokenizer = _WORKER['tokenizer']
	source_dir = _WORKER['source_dir']
	base, ext = os.path.splitext(output_path)
	shards: List = []
	dropped_total = 0
	buffer: List = []

	def flush ():
		nonlocal buffer
		if not buffer:
			return
		tmp = f'{base}.w{wid:04d}.s{len(shards):05d}{ext}'
		torch.save(dict(items=buffer), tmp)
		shards.append((tmp, len(buffer)))
		buffer = []

	for fp in file_list:
		item, n_dropped = _make_item(fp, source_dir, tokenizer)
		buffer.append(item)
		dropped_total += n_dropped
		if len(buffer) >= shard_size:
			flush()
	flush()
	return wid, shards, dropped_total


def pack_midi_notagen_parallel (
	source_dir: str,
	output_path: str,
	patch_size: int = 16,
	patch_length: int = 2048,
	shard_size: int = 20000,
	num_workers: int = 0,
	log=None,
) -> Dict[str, Any]:
	'''Parallel sharded packing with WORKER-SIDE sharding (workers write own shards).'''
	import multiprocessing as mp

	tokenizer = MidiTokenizer(patch_size=patch_size)
	files = find_midi_text_files(source_dir)
	config = dict(patch_size=patch_size, patch_length=patch_length)
	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

	if not num_workers or num_workers <= 0:
		num_workers = max(1, (os.cpu_count() or 2) - 1)
	if shard_size <= 0:
		shard_size = 20000
	num_workers = min(num_workers, max(1, len(files)))

	n = len(files)
	per = (n + num_workers - 1) // num_workers
	tasks = []
	for wid in range(num_workers):
		chunk = files[wid * per:(wid + 1) * per]
		if chunk:
			tasks.append((wid, chunk, output_path, shard_size))
	if log:
		log('dispatching %d files to %d workers (~%d files/worker)' % (n, len(tasks), per))

	ctx = mp.get_context('fork')
	with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(patch_size, source_dir)) as pool:
		results = pool.map(_worker_process_range, tasks)

	# rename per-worker shards into global contiguous order, build index
	results.sort(key=lambda r: r[0])
	shards: List[Dict[str, Any]] = []
	dropped_total = 0
	for wid, wshards, w_dropped in results:
		dropped_total += w_dropped
		for tmp_path, count in wshards:
			final_path = _shard_path(output_path, len(shards))
			os.replace(tmp_path, final_path)
			shards.append(dict(file=os.path.basename(final_path), count=count))
	if log:
		log('wrote %d shards, %d files, %d dropped lines' % (len(shards), n, dropped_total))

	index = dict(
		version=2,
		format='midi-notagen-patches-sharded',
		tokenizer=dict(vocab_size=tokenizer.vocab_size, patch_size=patch_size),
		config=config,
		shards=shards,
		stats=dict(files=n, dropped_total=dropped_total, shards=len(shards),
			shard_size=shard_size, num_workers=len(tasks)),
	)
	torch.save(index, output_path)
	return index


def pack_midi_notagen (
	source_dir: str,
	output_path: str,
	patch_size: int = 16,
	patch_length: int = 2048,
	shard_size: int = 0,
) -> Dict[str, Any]:
	'''Pack MidiText `.txt` files under source_dir into a NotaGen-style patch artifact.

	shard_size = 0: write a single artifact at output_path (version 1).
	shard_size > 0: write each `shard_size` items to a separate .pt shard plus an index
	    file at output_path (version 2). Items are flushed shard-by-shard so memory stays
	    bounded for very large corpora.
	'''
	tokenizer = MidiTokenizer(patch_size=patch_size)
	files = find_midi_text_files(source_dir)
	config = dict(patch_size=patch_size, patch_length=patch_length)
	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

	# --- legacy single-file mode ---
	if not shard_size or shard_size <= 0:
		items = []
		dropped_total = 0
		for file_path in files:
			item, n_dropped = _make_item(file_path, source_dir, tokenizer)
			items.append(item)
			dropped_total += n_dropped
		artifact = dict(
			version=1,
			format='midi-notagen-patches',
			tokenizer=dict(vocab_size=tokenizer.vocab_size, patch_size=patch_size),
			config=config,
			items=items,
			stats=dict(files=len(files), dropped_total=dropped_total),
		)
		torch.save(artifact, output_path)
		return artifact

	# --- sharded mode ---
	shards: List[Dict[str, Any]] = []
	dropped_total = 0
	buffer: List[Dict[str, Any]] = []

	def flush ():
		nonlocal buffer
		if not buffer:
			return
		shard_path = _shard_path(output_path, len(shards))
		torch.save(dict(items=buffer), shard_path)
		shards.append(dict(file=os.path.basename(shard_path), count=len(buffer)))
		buffer = []

	for file_path in files:
		item, n_dropped = _make_item(file_path, source_dir, tokenizer)
		buffer.append(item)
		dropped_total += n_dropped
		if len(buffer) >= shard_size:
			flush()
	flush()

	index = dict(
		version=2,
		format='midi-notagen-patches-sharded',
		tokenizer=dict(vocab_size=tokenizer.vocab_size, patch_size=patch_size),
		config=config,
		shards=shards,
		stats=dict(files=len(files), dropped_total=dropped_total, shards=len(shards), shard_size=shard_size),
	)
	torch.save(index, output_path)
	return index
