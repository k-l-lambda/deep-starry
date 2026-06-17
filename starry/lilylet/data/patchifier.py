import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch


# A metadata header line `[field "..."]`. The field name may carry digits/hyphens
# for per-staff instrument keys (`[instrument-1-2 "Piano" "Pno."]`), and the line may
# hold more than one quoted string (full name + abbreviation).
METADATA_RE = re.compile(r'^\[[A-Za-z][A-Za-z0-9-]*\s+".*"\]$')
# A leading style comment line (`--styles-in-comments` output: %<period>, %<composer>,
# %<instrumentation>). These sit at the very top of the document and carry the style
# conditioning, so — unlike inline/measure-end `%` comments — they must be PRESERVED
# as part of the metadata block. A bare `%` directive (`%%...`) is not a style line.
STYLE_COMMENT_RE = re.compile(r'^%(?!%).*$')
MEASURE_END_RE = re.compile(r'\|\s*$')
# Voice separator is `\\` and part separator is `\\\` in serialized Lilylet.
# Split at runs of 2+ backslashes so patches never cross voice/part boundaries,
# analogous to NotaGen splitting tunebody at ABC barlines.
VOICE_SEP_RE = re.compile(r'(\\{2,})')


@dataclass
class UnknownHit:
	file: str
	char: str
	codePoint: str
	utf8Hex: str
	count: int = 0


class LilyletTokenizer:
	def __init__(self, tokenizer_path: str):
		self.path = tokenizer_path
		with open(tokenizer_path, 'r', encoding='utf-8') as f:
			self.artifact = json.load(f)

		self.vocab = self.artifact['vocab']
		self.id_by_token = {entry['token']: entry['id'] for entry in self.vocab}
		self.text_by_id = {entry['id']: entry.get('text', entry['token']) for entry in self.vocab}
		self.unknown_id = self.id_by_token.get('<unknown>', 3)
		self.pad_id = self.id_by_token.get('<pad>', 0)
		self.bos_id = self.id_by_token.get('<bos>', 1)
		self.eos_id = self.id_by_token.get('<eos>', 2)

		fixed = [entry['token'] for entry in self.vocab if entry.get('type') == 'protected']
		self.fixed_tokens = sorted(set(fixed), key=lambda token: (-len(token), token))

	def encode(self, text: str, file: str = '', unknowns: Dict[Tuple[str, str], UnknownHit] | None = None) -> List[int]:
		ids: List[int] = []
		i = 0
		while i < len(text):
			matched = None
			for token in self.fixed_tokens:
				if text.startswith(token, i):
					matched = token
					break
			if matched is not None:
				ids.append(self.id_by_token[matched])
				i += len(matched)
				continue

			char = text[i]
			code_point = ord(char)
			data = char.encode('utf-8')
			if char in self.id_by_token:
				ids.append(self.id_by_token[char])
			else:
				emitted_unknown = False
				for byte in data:
					if 0x08 <= byte <= 0x7f and byte in self.text_by_id:
						ids.append(byte)
					else:
						emitted_unknown = True
				if emitted_unknown:
					ids.append(self.unknown_id)
					if unknowns is not None:
						key = (file, char)
						hit = unknowns.get(key)
						if hit is None:
							hit = UnknownHit(
								file=file,
								char=char,
								codePoint=f'U+{code_point:04X}',
								utf8Hex=' '.join(f'{byte:02x}' for byte in data),
							)
							unknowns[key] = hit
						hit.count += 1
			i += 1
		return ids


def normalize_text(text: str) -> str:
	text = text.replace('\r\n', '\n').replace('\r', '\n')
	return '\n'.join(line.split('%', 1)[0].rstrip() for line in text.split('\n'))


def split_lilylet_document(text: str) -> Tuple[List[str], List[str]]:
	# Scan the RAW (un-normalized) lines for the leading metadata block first: it can
	# hold both `[field "..."]` lines and leading `%<style>` comment lines (the
	# `--styles-in-comments` format). Those style comments must survive — normalize_text
	# strips every `%`, so normalizing before this scan would drop them. The body (after
	# the block) is normalized to remove inline / measure-end `%` comments as before.
	raw_lines = [line for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n') if line.strip()]
	metadata: List[str] = []
	body_start = 0
	for i, line in enumerate(raw_lines):
		stripped = line.strip()
		if METADATA_RE.match(stripped) or STYLE_COMMENT_RE.match(stripped):
			metadata.append(stripped + '\n')
			body_start = i + 1
		else:
			break
	body_text = '\n'.join(raw_lines[body_start:])
	body_lines = [line + '\n' for line in normalize_text(body_text).split('\n') if line.strip()]
	return metadata, body_lines


def split_measures(body_lines: List[str]) -> List[str]:
	measures: List[str] = []
	current: List[str] = []
	for line in body_lines:
		current.append(line)
		if MEASURE_END_RE.search(line.rstrip('\n')):
			measures.append(''.join(current))
			current = []
	if current:
		measures.append(''.join(current))
	return measures


def split_voice_segments(measure: str) -> List[str]:
	"""Split a measure into voice/part segments at `\\\\` and `\\\\\\` separators,
	keeping each separator attached to its preceding segment."""
	parts = VOICE_SEP_RE.split(measure)
	segments: List[str] = []
	for i in range(0, len(parts), 2):
		chunk = parts[i]
		sep = parts[i + 1] if i + 1 < len(parts) else ''
		segment = chunk + sep
		if segment:
			segments.append(segment)
	return segments


def split_patches(ids: List[int], patch_size: int, eos_id: int) -> List[List[int]]:
	if len(ids) % patch_size != 0:
		ids = ids + [eos_id]
	return [ids[i:i + patch_size] for i in range(0, len(ids), patch_size)]


def pad_patch(patch: List[int], patch_size: int, pad_id: int) -> List[int]:
	return patch + [pad_id] * (patch_size - len(patch))


def special_patch(kind: str, patch_size: int, bos_id: int, eos_id: int) -> List[int]:
	if kind == 'bos':
		return [bos_id] * (patch_size - 1) + [eos_id]
	if kind == 'eos':
		return [bos_id] + [eos_id] * (patch_size - 1)
	raise ValueError(f'Unknown special patch kind: {kind}')


def patchify_text(
	text: str,
	tokenizer: LilyletTokenizer,
	file: str = '',
	patch_size: int = 16,
	patch_length: int = 2048,
	patch_stream: bool = True,
	add_special_patches: bool = True,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
	unknowns: Dict[Tuple[str, str], UnknownHit] = {}
	metadata_lines, body_lines = split_lilylet_document(text)
	measures = split_measures(body_lines)

	if patch_stream:
		total = len(measures)
		measures = [f'[r:{i}/{total - i - 1}]' + measure for i, measure in enumerate(measures)]

	metadata_patches: List[List[int]] = []
	for line in metadata_lines:
		metadata_patches.extend(split_patches(tokenizer.encode(line, file, unknowns), patch_size, tokenizer.eos_id))

	def body_to_patches(chunks: List[str]) -> List[List[int]]:
		patches: List[List[int]] = []
		for chunk in chunks:
			for segment in split_voice_segments(chunk):
				patches.extend(split_patches(tokenizer.encode(segment, file, unknowns), patch_size, tokenizer.eos_id))
		return patches

	body_patches = body_to_patches(measures)

	if add_special_patches:
		metadata_patches = [special_patch('bos', patch_size, tokenizer.bos_id, tokenizer.eos_id)] + metadata_patches
		body_patches = body_patches + [special_patch('eos', patch_size, tokenizer.bos_id, tokenizer.eos_id)]

	patches = metadata_patches + body_patches
	if len(patches) > patch_length:
		if patch_stream and measures:
			choices = ['head'] if len(measures) == 1 else ['head', 'tail', 'middle']
			choice = random.choice(choices)
			if choice == 'head':
				body_patches = body_to_patches(measures)
			else:
				start = len(measures) - 1 if choice == 'tail' else random.randrange(1, len(measures))
				body_patches = body_to_patches(measures[start:])
			if add_special_patches:
				body_patches = body_patches + [special_patch('eos', patch_size, tokenizer.bos_id, tokenizer.eos_id)]
			patches = metadata_patches + body_patches
		patches = patches[:patch_length]

	padded = [pad_patch(patch, patch_size, tokenizer.pad_id) for patch in patches]
	unknown_list = [hit.__dict__ for hit in unknowns.values()]
	# Token ids fit in 0..255 (vocab size 256), so store patches compactly as uint8.
	# The per-item mask is always all-ones (real padding only happens at batch time),
	# so it is not stored; the dataset reconstructs it from the patch count.
	return torch.tensor(padded, dtype=torch.uint8), unknown_list


def find_lilylet_files(source_dir: str) -> List[str]:
	results: List[str] = []
	for root, _, files in os.walk(source_dir):
		for name in files:
			if name.endswith('.lyl'):
				results.append(os.path.join(root, name))
	return sorted(results)


def _shard_path(output_path: str, shard_index: int) -> str:
	# foo.lilylet-notagen.pt -> foo.lilylet-notagen.shard00000.pt
	base, ext = os.path.splitext(output_path)
	return f'{base}.shard{shard_index:05d}{ext}'


# --- parallel worker plumbing (one tokenizer per process) ---
_WORKER = {}


def _worker_init(tokenizer_path, source_dir, patch_size, patch_length, patch_stream):
	_WORKER['tokenizer'] = LilyletTokenizer(tokenizer_path)
	_WORKER['source_dir'] = source_dir
	_WORKER['patch_size'] = patch_size
	_WORKER['patch_length'] = patch_length
	_WORKER['patch_stream'] = patch_stream


def _worker_make_item(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		text = f.read()
	rel = os.path.relpath(file_path, _WORKER['source_dir'])
	patches, unknowns = patchify_text(
		text,
		_WORKER['tokenizer'],
		file=rel,
		patch_size=_WORKER['patch_size'],
		patch_length=_WORKER['patch_length'],
		patch_stream=_WORKER['patch_stream'],
	)
	return dict(path=rel, patches=patches, unknowns=unknowns), sum(hit['count'] for hit in unknowns)


def _worker_process_range(task):
	'''Patchify a contiguous slice of files AND write its own shards to disk.
	Returns (worker_index, [(tmp_shard_path, count), ...], unknown_total).
	No item data flows back to the parent, so the parent is never a
	serialization bottleneck and CPU scales near-linearly.'''
	wid, file_list, output_path, shard_size = task
	base, ext = os.path.splitext(output_path)
	shards = []
	unknown_total = 0
	buffer = []

	def flush():
		nonlocal buffer
		if not buffer:
			return
		tmp = f'{base}.w{wid:04d}.s{len(shards):05d}{ext}'
		torch.save(dict(items=buffer), tmp)
		shards.append((tmp, len(buffer)))
		buffer = []

	for fp in file_list:
		item, n_unknown = _worker_make_item(fp)
		buffer.append(item)
		unknown_total += n_unknown
		if len(buffer) >= shard_size:
			flush()
	flush()
	return wid, shards, unknown_total


def pack_lilylet_notagen_parallel(
	source_dir: str,
	output_path: str,
	tokenizer_path: str,
	patch_size: int = 16,
	patch_length: int = 1024,
	patch_stream: bool = True,
	shard_size: int = 20000,
	num_workers: int = 0,
	chunksize: int = 64,
	log=None,
) -> Dict[str, Any]:
	'''Parallel sharded packing with WORKER-SIDE sharding (workers write own shards).'''
	import multiprocessing as mp

	tokenizer = LilyletTokenizer(tokenizer_path)
	vocab_size = max(entry['id'] for entry in tokenizer.vocab) + 1
	files = find_lilylet_files(source_dir)
	config = dict(patch_size=patch_size, patch_length=patch_length, patch_stream=patch_stream)
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
	with ctx.Pool(
		processes=num_workers,
		initializer=_worker_init,
		initargs=(tokenizer_path, source_dir, patch_size, patch_length, patch_stream),
	) as pool:
		results = pool.map(_worker_process_range, tasks)

	# rename per-worker shards into global contiguous order, build index
	results.sort(key=lambda r: r[0])
	shards: List[Dict[str, Any]] = []
	unknown_total = 0
	for wid, wshards, w_unknown in results:
		unknown_total += w_unknown
		for tmp_path, count in wshards:
			final_path = _shard_path(output_path, len(shards))
			os.replace(tmp_path, final_path)
			shards.append(dict(file=os.path.basename(final_path), count=count))
	if log:
		log('wrote %d shards, %d files, %d unknowns' % (len(shards), n, unknown_total))

	index = dict(
		version=2,
		format='lilylet-notagen-patches-sharded',
		tokenizer=dict(path=tokenizer_path, vocab_size=vocab_size),
		config=config,
		shards=shards,
		stats=dict(files=n, unknown_total=unknown_total, shards=len(shards),
			shard_size=shard_size, num_workers=len(tasks)),
	)
	torch.save(index, output_path)
	return index


def pack_lilylet_notagen(
	source_dir: str,
	output_path: str,
	tokenizer_path: str,
	patch_size: int = 16,
	patch_length: int = 2048,
	patch_stream: bool = True,
	shard_size: int = 0,
) -> Dict[str, Any]:
	'''Pack .lyl files under source_dir into a NotaGen-style patch artifact.

	shard_size = 0: write a single artifact at output_path (legacy, version 1).
	shard_size > 0: write each `shard_size` items to a separate .pt shard and write
	    an index file at output_path (version 2) that lists the shards. Items are
	    flushed shard-by-shard so memory stays bounded for very large corpora.
	'''
	tokenizer = LilyletTokenizer(tokenizer_path)
	files = find_lilylet_files(source_dir)
	vocab_size = max(entry['id'] for entry in tokenizer.vocab) + 1
	config = dict(patch_size=patch_size, patch_length=patch_length, patch_stream=patch_stream)

	def make_item(file_path: str):
		with open(file_path, 'r', encoding='utf-8') as f:
			text = f.read()
		patches, unknowns = patchify_text(
			text,
			tokenizer,
			file=os.path.relpath(file_path, source_dir),
			patch_size=patch_size,
			patch_length=patch_length,
			patch_stream=patch_stream,
		)
		return dict(path=os.path.relpath(file_path, source_dir), patches=patches, unknowns=unknowns), \
			sum(hit['count'] for hit in unknowns)

	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

	# --- legacy single-file mode ---
	if not shard_size or shard_size <= 0:
		items = []
		unknown_total = 0
		for file_path in files:
			item, n_unknown = make_item(file_path)
			items.append(item)
			unknown_total += n_unknown
		artifact = dict(
			version=1,
			format='lilylet-notagen-patches',
			tokenizer=dict(path=tokenizer_path, vocab_size=vocab_size),
			config=config,
			items=items,
			stats=dict(files=len(files), unknown_total=unknown_total),
		)
		torch.save(artifact, output_path)
		return artifact

	# --- sharded mode ---
	shards: List[Dict[str, Any]] = []
	unknown_total = 0
	buffer: List[Dict[str, Any]] = []

	def flush():
		nonlocal buffer
		if not buffer:
			return
		shard_index = len(shards)
		shard_path = _shard_path(output_path, shard_index)
		torch.save(dict(items=buffer), shard_path)
		shards.append(dict(file=os.path.basename(shard_path), count=len(buffer)))
		buffer = []

	for file_path in files:
		item, n_unknown = make_item(file_path)
		buffer.append(item)
		unknown_total += n_unknown
		if len(buffer) >= shard_size:
			flush()
	flush()

	index = dict(
		version=2,
		format='lilylet-notagen-patches-sharded',
		tokenizer=dict(path=tokenizer_path, vocab_size=vocab_size),
		config=config,
		shards=shards,
		stats=dict(files=len(files), unknown_total=unknown_total, shards=len(shards), shard_size=shard_size),
	)
	torch.save(index, output_path)
	return index

