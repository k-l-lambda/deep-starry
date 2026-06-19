'''CLaMP 3 M3 (symbolic) patchilizer + encoding helpers.

The M3 *model* (the BERT-style `M3PatchEncoder`, its config, and the weight
loader) lives in `starry/lilylet/models/m3.py` and is re-exported here for
backward compatibility. This module keeps the input-side pieces:

  - `M3Patchilizer`: segments music text into fixed-size character patches. It
    adapts to two grammars via `syntax=`: 'abc' (the original CLaMP bar/line
    rules, also covering MTF) and 'lilylet' (deep-starry's Lilylet document
    structure, reusing the splitters in `starry/lilylet/data/patchifier.py`).
    The character encoding itself (bos + ord(c)... + eos, one-hot 128) is
    identical for both — only the bar/segment splitting differs.
  - `encode_abc`: run an `M3PatchEncoder` over a file/string and return the
    UN-pooled per-patch hidden states `[num_patches, M3_HIDDEN_SIZE]` (no
    avg-pool, no projection), following extract_clamp2's get_normalized=False flow.

Reference: NotaGen/clamp2/utils.py (M3Patchilizer) and
NotaGen/clamp2/extract_clamp2.py (the extraction flow).
'''

import os
import re
import random

import torch
from unidecode import unidecode

from .data.patchifier import split_lilylet_document, split_measures, split_voice_segments
from .models.m3 import (
	M3PatchEncoder, build_m3_config, load_m3_encoder,
	PATCH_SIZE, PATCH_LENGTH, M3_HIDDEN_SIZE, PATCH_NUM_LAYERS, NUM_CLASSES,
	PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, MASK_TOKEN_ID,
)


SYNTAXES = ('abc', 'lilylet')


class M3Patchilizer:
	'''Segments music text into fixed-size character patches for the M3 encoder.

	Each patch is a length-`patch_size` list of ids: bos + ord(c)... + eos,
	right-padded with pad_token_id. The `syntax` option selects how text is split
	into patches before that char encoding:
	  - 'abc'     : CLaMP's original rules — header lines (`X:`/`K:`/… or `%%`)
	                become whole patches, body lines are split at ABC barlines.
	                Also handles MTF (`ticks_per_beat …`). Ported from NotaGen.
	  - 'lilylet' : deep-starry Lilylet — leading metadata/style block lines become
	                patches, the body is split into measures then voice/part
	                segments (reusing starry.lilylet.data.patchifier splitters).
	'''

	def __init__ (self, syntax='abc'):
		if syntax not in SYNTAXES:
			raise ValueError('M3Patchilizer syntax must be one of %s, got %r' % (SYNTAXES, syntax))
		self.syntax = syntax
		self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
		self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
		self.pad_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.eos_token_id = EOS_TOKEN_ID
		self.mask_token_id = MASK_TOKEN_ID

	def split_bars (self, body):
		bars = re.split(self.regexPattern, ''.join(body))
		bars = list(filter(None, bars))  # remove empty strings
		if bars[0] in self.delimiters:
			bars[1] = bars[0] + bars[1]
			bars = bars[1:]
		bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
		return bars

	def bar2patch (self, bar, patch_size=PATCH_SIZE):
		patch = [self.bos_token_id] + [ord(c) for c in bar] + [self.eos_token_id]
		patch = patch[:patch_size]
		patch += [self.pad_token_id] * (patch_size - len(patch))
		return patch

	def patch2bar (self, patch):
		return ''.join(chr(idx) if idx > self.mask_token_id else '' for idx in patch)

	def _segment_abc (self, item):
		'''ABC / MTF segmentation (original CLaMP rules). Returns string patches.'''
		lines = re.findall(r'.*?\n|.*$', item)
		lines = list(filter(None, lines))  # remove empty lines

		patches = []
		if lines[0].split(" ")[0] == "ticks_per_beat":
			patch = ""
			for line in lines:
				if patch.startswith(line.split(" ")[0]) and (len(patch) + len(" ".join(line.split(" ")[1:])) <= PATCH_SIZE - 2):
					patch = patch[:-1] + "\t" + " ".join(line.split(" ")[1:])
				else:
					if patch:
						patches.append(patch)
					patch = line
			if patch != "":
				patches.append(patch)
		else:
			for line in lines:
				if len(line) > 1 and ((line[0].isalpha() and line[1] == ':') or line.startswith('%%')):
					patches.append(line)
				else:
					bars = self.split_bars(line)
					if bars:
						bars[-1] += '\n'
						patches.extend(bars)
		return patches

	def _segment_lilylet (self, item):
		'''Lilylet segmentation. Leading metadata/style lines become patches; the
		body is split into measures, then each measure into voice/part segments
		(reusing the deep-starry Lilylet splitters). Returns string patches.'''
		metadata_lines, body_lines = split_lilylet_document(item)
		patches = list(metadata_lines)  # each already carries a trailing '\n'
		for measure in split_measures(body_lines):
			segments = split_voice_segments(measure)
			if segments:
				segments[-1] = segments[-1].rstrip('\n') + '\n'  # delimit the measure, mirror ABC
				patches.extend(segments)
		return patches

	def encode (self, item, patch_size=PATCH_SIZE, add_special_patches=False, truncate=False, random_truncate=False):
		item = unidecode(item)

		if self.syntax == 'lilylet':
			patches = self._segment_lilylet(item)
		else:
			patches = self._segment_abc(item)

		if add_special_patches:
			bos_patch = chr(self.bos_token_id) * patch_size
			eos_patch = chr(self.eos_token_id) * patch_size
			patches = [bos_patch] + patches + [eos_patch]

		if len(patches) > PATCH_LENGTH and truncate:
			choices = ["head", "tail", "middle"]
			choice = random.choice(choices)
			if choice == "head" or random_truncate == False:
				patches = patches[:PATCH_LENGTH]
			elif choice == "tail":
				patches = patches[-PATCH_LENGTH:]
			else:
				start = random.randint(1, len(patches) - PATCH_LENGTH)
				patches = patches[start:start + PATCH_LENGTH]

		patches = [self.bar2patch(patch) for patch in patches]

		return patches

	def decode (self, patches):
		return ''.join(self.patch2bar(patch) for patch in patches)


def _read_music (text_or_path, filter_comments=True):
	'''Accept a path or a raw string; return the text. When `filter_comments` is
	set (ABC), drop `%` comment lines but keep `%%` directives (extract_clamp2.py:74-78).
	Lilylet leading `%` lines are style metadata, not comments, so they are kept.'''
	if '\n' not in text_or_path and os.path.exists(text_or_path):
		with open(text_or_path, 'r', encoding='utf-8') as f:
			lines = f.readlines()
	else:
		lines = text_or_path.splitlines(keepends=True)

	if filter_comments:
		lines = [line for line in lines if not (line.startswith('%') and not line.startswith('%%'))]
	return ''.join(lines)


def encode_abc (abc, encoder, patchilizer, device=None):
	'''Encode one music file/string into UN-pooled M3 hidden states.

	Returns a tensor [num_patches, M3_HIDDEN_SIZE] (no avg-pool, no projection).
	The `patchilizer` (its `syntax`) decides whether the input is read as ABC or
	Lilylet. Mirrors extract_clamp2.extract_feature with get_normalized=False:
	segment the patch sequence into <=PATCH_LENGTH chunks, encode each, strip
	padded rows, and concatenate (de-overlapping the trailing window).
	'''
	if device is None:
		device = next(encoder.parameters()).device

	item = _read_music(abc, filter_comments=(patchilizer.syntax == 'abc'))
	input_data = patchilizer.encode(item, add_special_patches=True)
	input_data = torch.tensor(input_data)  # [N, PATCH_SIZE]
	max_len = PATCH_LENGTH

	segments = [input_data[i:i + max_len] for i in range(0, len(input_data), max_len)]
	segments[-1] = input_data[-max_len:]  # last chunk = trailing window (may overlap)

	hidden_list = []
	with torch.no_grad():
		for segment in segments:
			masks = torch.cat((torch.ones(segment.size(0)), torch.zeros(max_len - segment.size(0))), 0)
			pad = torch.ones((max_len - segment.size(0), PATCH_SIZE), dtype=segment.dtype) * patchilizer.pad_token_id
			padded = torch.cat((segment, pad), 0)
			hidden = encoder(padded.unsqueeze(0).to(device), masks.unsqueeze(0).to(device))['last_hidden_state']
			hidden = hidden[:, :int(masks.sum().item()), :]
			hidden_list.append(hidden[0])

	# de-overlap the trailing window (extract_clamp2.py:124-127)
	remainder = len(input_data) % max_len
	if remainder and len(hidden_list) > 1:
		hidden_list[-1] = hidden_list[-1][-remainder:]

	return torch.concat(hidden_list, 0)
