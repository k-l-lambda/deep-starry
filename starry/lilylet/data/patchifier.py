import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch


METADATA_RE = re.compile(r'^\[[A-Za-z][A-Za-z-]*\s+".*"\]$')
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
	lines = [line for line in normalize_text(text).split('\n') if line.strip()]
	metadata: List[str] = []
	body_start = 0
	for i, line in enumerate(lines):
		if METADATA_RE.match(line.strip()):
			metadata.append(line + '\n')
			body_start = i + 1
		else:
			break
	body_lines = [line + '\n' for line in lines[body_start:]]
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
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
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
	masks = [1] * len(padded)
	unknown_list = [hit.__dict__ for hit in unknowns.values()]
	# Token ids fit in 0..255 (vocab size 256), so store patches compactly as uint8.
	return torch.tensor(padded, dtype=torch.uint8), torch.tensor(masks, dtype=torch.uint8), unknown_list


def find_lilylet_files(source_dir: str) -> List[str]:
	results: List[str] = []
	for root, _, files in os.walk(source_dir):
		for name in files:
			if name.endswith('.lyl'):
				results.append(os.path.join(root, name))
	return sorted(results)


def pack_lilylet_notagen(
	source_dir: str,
	output_path: str,
	tokenizer_path: str,
	patch_size: int = 16,
	patch_length: int = 2048,
	patch_stream: bool = True,
) -> Dict[str, Any]:
	tokenizer = LilyletTokenizer(tokenizer_path)
	files = find_lilylet_files(source_dir)
	items = []
	unknown_total = 0
	for file_path in files:
		with open(file_path, 'r', encoding='utf-8') as f:
			text = f.read()
		patches, mask, unknowns = patchify_text(
			text,
			tokenizer,
			file=os.path.relpath(file_path, source_dir),
			patch_size=patch_size,
			patch_length=patch_length,
			patch_stream=patch_stream,
		)
		unknown_total += sum(hit['count'] for hit in unknowns)
		items.append(dict(
			path=os.path.relpath(file_path, source_dir),
			patches=patches,
			mask=mask,
			unknowns=unknowns,
		))

	artifact = dict(
		version=1,
		format='lilylet-notagen-patches',
		tokenizer=dict(path=tokenizer_path, vocab_size=max(entry['id'] for entry in tokenizer.vocab) + 1),
		config=dict(patch_size=patch_size, patch_length=patch_length, patch_stream=patch_stream),
		items=items,
		stats=dict(files=len(files), unknown_total=unknown_total),
	)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	torch.save(artifact, output_path)
	return artifact
