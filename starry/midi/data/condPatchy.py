'''CondMidiPatchy — feeder for the conditioned-MIDI measurewise joint-sequence dataset.

Consumes the artifact written by starry.midi.data.condPatchifier.pack: each item is ONE
joint sequence [lilylet score patches] ++ [midi event patches], with per-patch modality and
measure indices. This feeder builds, at batch time, the custom windowed/block attention mask
that couples midi measures to lilylet measures, and the supervision mask that trains ONLY the
midi segment (lilylet is a read-only condition).

Batch contract (consumed by starry.midi.models.condBgpt.CondMidiBGPT):
	input_patches   LongTensor [B, T, patch_size]   token ids (lilylet ids then midi ids)
	input_masks     LongTensor [B, T]               1 = real patch, 0 = padding
	input_targets   LongTensor [B, T]               1 = supervised next-patch target (midi only)
	input_positions LongTensor [B, T]               0..T-1 per sample (padded 0)
	attn_mask       BoolTensor [B, 1, T, T]          True = query may attend key (custom mask)
	modality        LongTensor [B, T]               0 = lilylet, 1 = midi
	lyl_counts      LongTensor [B]                   lilylet-segment length per sample

The 4D bool attn_mask is passed straight through to the HF Llama backbone (transformers
>=4.53 returns a 4D attention_mask as-is; SDPA reads bool True=attend).
'''

import bisect
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs
from ...utils.registry import register_dataset
from .condPatchifier import SHARDED_FORMAT


class _ItemStore:
	'''Lazy sharded store (mirrors starry.midi.data.patchy._ItemStore, new format string).'''

	def __init__ (self, root):
		self.root = root
		self.artifact = torch.load(root, map_location='cpu', weights_only=False)
		assert self.artifact.get('format') == SHARDED_FORMAT, \
			f'unexpected format {self.artifact.get("format")!r} (want {SHARDED_FORMAT!r})'
		self._dir = os.path.dirname(root)
		self._shards = self.artifact['shards']
		self._offsets = []
		total = 0
		for shard in self._shards:
			self._offsets.append(total)
			total += shard['count']
		self._total = total
		self._cache = {}

	def __len__ (self):
		return self._total

	def _load_shard (self, shard_index):
		cached = self._cache.get(shard_index)
		if cached is None:
			path = os.path.join(self._dir, self._shards[shard_index]['file'])
			cached = torch.load(path, map_location='cpu', weights_only=False)['items']
			self._cache[shard_index] = cached
		return cached

	def get (self, index):
		shard_index = bisect.bisect_right(self._offsets, index) - 1
		local = index - self._offsets[shard_index]
		return self._load_shard(shard_index)[local]


_STORE_CACHE = {}


def _get_store (root):
	store = _STORE_CACHE.get(root)
	if store is None:
		store = _ItemStore(root)
		_STORE_CACHE[root] = store
	return store


def build_vis (modality, own_meas, src_meas, w_midi, w_cross):
	'''Build the [T, T] boolean visibility mask (True = query q may attend key k).

	modality  [T] long  0 = lilylet, 1 = midi
	own_meas  [T] long  patch's own measure (lyl j / midi i; 0 = prefix/header)
	src_meas  [T] long  lilylet-aligned measure (lyl j / midi source_measure(i); 0 = prefix/header)
	w_midi    int       midi->midi window in MEASURES
	w_cross   int       midi->lilylet window in MEASURES

	Rules (all under the global causal floor k <= q):
	  lyl -> lyl   : full causal
	  lyl -> midi  : never (also forbidden by causal since midi is later)
	  midi -> lyl  : j == 0 (lyl prompt/header) -> always; i == 0 (midi header) -> never;
	                 else (i >= j) and (i < j + w_cross), with i = src[q], j = src[k]
	  midi -> midi : header (own 0) -> full causal within header; else c in [a-w_midi+1, a],
	                 with a = own[q], c = own[k]
	  eom patches  : carry a normal own/src measure, so they fall through the midi rules with
	                 NO special-casing.
	'''
	T = modality.shape[0]
	dev = modality.device
	idx = torch.arange(T, device=dev)
	causal = idx[:, None] >= idx[None, :]				# [T,T] k<=q

	q_lyl = (modality == 0)[:, None].expand(T, T)
	k_lyl = (modality == 0)[None, :].expand(T, T)
	q_midi = ~q_lyl
	k_midi = ~k_lyl

	own_q = own_meas[:, None].expand(T, T)
	own_k = own_meas[None, :].expand(T, T)
	src_q = src_meas[:, None].expand(T, T)
	src_k = src_meas[None, :].expand(T, T)

	vis = torch.zeros(T, T, dtype=torch.bool, device=dev)

	# lyl -> lyl : full causal
	vis |= q_lyl & k_lyl & causal

	# midi -> lyl : cross block by lilylet-aligned measure
	cross = q_midi & k_lyl & causal
	j_zero = src_k == 0								# lyl prompt/header: global condition
	i_zero = src_q == 0								# midi header: sees nothing on the lyl side
	in_window = (src_q >= src_k) & (src_q < src_k + w_cross)
	cross_vis = torch.where(j_zero, torch.ones_like(vis),
		torch.where(i_zero, torch.zeros_like(vis), in_window))
	vis |= cross & cross_vis

	# midi -> midi : windowed causal by midi measure
	mm = q_midi & k_midi & causal
	header = (own_q == 0) | (own_k == 0)			# midi header region: full causal
	win = (own_k > own_q - w_midi) & (own_k <= own_q)
	vis |= mm & (header | win)

	# lyl -> midi stays False (causal already forbids it).
	return vis


@register_dataset
class CondMidiPatchy (Dataset):
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

	def __init__ (self, root, split, device='cpu', shuffle=False, pad_id=0,
		w_midi=2, w_cross=2, patch_length=0, max_patches=0, random_crop=True, **_):
		super().__init__()
		self.device = device
		self.shuffle = shuffle
		self.pad_id = pad_id
		self.w_midi = w_midi
		self.w_cross = w_cross
		# patch_length is UNSUPPORTED here: per-song measure coupling forbids cropping the JOINT
		# sequence, and silently ignoring a nonzero value is a footgun (other feeders use
		# patch_length as the length bound, so a config that sets it but forgets max_patches would
		# build a full O(T^2) attention mask and OOM). Fail loud; use max_patches to bound T.
		if patch_length:
			raise ValueError(
				'CondMidiPatchy does not support patch_length (per-song measure coupling forbids '
				'cropping the joint sequence); set patch_length=0 and use max_patches to bound T '
				'by cropping only the midi body to whole measures.')
		self.patch_length = patch_length
		# max_patches: cap the joint length T by cropping ONLY the midi body to whole measures
		# (the lilylet condition + midi header are always kept in full). 0 = no cap. The O(T^2)
		# decoder self-attention memory is driven by the midi segment (L is ~9% of T), so this
		# keeps the full score condition while bounding the memory-dominant midi window. On the
		# train split a random measure window is taken (augmentation); otherwise the head window.
		self.max_patches = max_patches
		self.random_crop = random_crop
		self.store = _get_store(root)

		phases, cycle = parseFilterStr(split)
		self.indices = [i for i in range(len(self.store)) if i % cycle in phases]

	def __len__ (self):
		return len(self.indices)

	def _item (self, index):
		item = self.store.get(index)
		patches = item['patches'].long()				# [T, patch_size]
		modality = item['modality'].long()				# [T]
		own_meas = item['measures'].long()				# [T]
		src_meas = item['src_measures'].long()			# [T]
		lyl_count = int(item['lyl_count'])
		drop_targets = []		# cropped-coord positions whose supervision target must be forced off
		if self.max_patches and patches.shape[0] > self.max_patches:
			patches, modality, own_meas, src_meas, lyl_count, drop_targets = self._crop_midi(
				patches, modality, own_meas, src_meas, lyl_count)
		return patches, modality, own_meas, src_meas, lyl_count, drop_targets

	def _crop_midi (self, patches, modality, own_meas, src_meas, lyl_count):
		'''Crop the MIDI body to a contiguous run of whole measures so the joint length fits
		max_patches, keeping the lilylet prefix + the measures the kept midi window needs, and the
		midi header (own_meas==0) intact.

		The midi self-attention (window w_midi measures) and the midi->lyl cross window only need
		a local measure neighbourhood, so a contiguous measure window is a valid training example;
		per-patch own/src measure ids are preserved, so build_vis couples it to the lilylet score
		exactly as in the uncropped song. A random window start (train) augments; head otherwise.

		The lilylet TAIL is trimmed too: a midi query at source measure src_q cross-attends only
		lyl keys with src_meas in (src_q - w_cross, src_q] plus the global prefix (src_meas == 0).
		Since lyl->lyl is causal and lyl is never supervised, any lyl patch whose measure exceeds
		the highest kept-midi source measure is attended by nobody -> drop it. lyl src_meas is
		monotonic non-decreasing, so the survivors are the contiguous prefix and their RoPE
		positions 0..new_lyl_count-1 are unchanged (only a suffix is removed).

		Returns the kept tensors plus `drop_targets`: cropped-coord positions whose next-patch
		supervision must be dropped because the crop broke their causal context:
		  - a mid-body window start (start_i > 0): the first kept BODY patch would be predicted from
		    the midi header as if it were measure 1, but its true preceding body measures were cut.
		  - a hard-clamp tail cut (a single measure exceeding budget): the last kept patch is a
		    partial measure with no natural <eom>, so we don't teach that artificial boundary.
		'''
		T = patches.shape[0]
		is_midi = modality == 1
		midi_pos = torch.nonzero(is_midi, as_tuple=False).flatten()
		if midi_pos.numel() == 0:
			return patches, modality, own_meas, src_meas, lyl_count, []
		m0 = int(midi_pos[0])								# first midi position (== lyl_count)
		# midi header patches (own_meas == 0) sit CONTIGUOUSLY at the midi segment head; always
		# keep them. Count the contiguous run (not a global ==0 sum) so a malformed later measure-0
		# patch can't push body_start into the body; assert none appear after the header.
		midi_own = own_meas[m0:]
		header_len = 0
		while header_len < midi_own.numel() and int(midi_own[header_len]) == 0:
			header_len += 1
		assert not bool((midi_own[header_len:] == 0).any()), \
			'non-contiguous midi header patches (own_meas==0 after the header run)'
		body_start = m0 + header_len						# first real (measured) midi patch
		# budget for the midi BODY after keeping lyl + midi header.
		fixed = body_start									# lyl_count + header_len
		budget = self.max_patches - fixed
		if budget <= 0:
			# pathological (huge lyl+header); fall back to a hard head cut at max_patches.
			keep = slice(0, self.max_patches)
			return (patches[keep], modality[keep], own_meas[keep], src_meas[keep],
				min(lyl_count, self.max_patches), [])
		body_own = own_meas[body_start:]					# measure id per body patch
		# distinct played measures in first-occurrence order; assert monotonic non-decreasing so we
		# don't silently rely on torch.unique's sort and so upstream corruption surfaces.
		measures, last = [], None
		for mm in body_own.tolist():
			assert last is None or mm >= last, 'non-monotonic midi body measure ids'
			if mm != last:
				measures.append(mm); last = mm
		# greedily assemble the largest contiguous measure window (from a chosen start) <= budget.
		# precompute per-measure patch counts and their start offsets within the body.
		counts = {int(mm): int((body_own == mm).sum()) for mm in measures}
		# choose a start measure: random (train) or first (val/head).
		if self.random_crop and len(measures) > 1:
			start_i = int(torch.randint(0, len(measures), (1,)))
		else:
			start_i = 0
		# extend the window forward from start_i while it fits; if the very first measure alone
		# exceeds budget, take just that measure (a single measure is the atomic unit).
		chosen, acc = [], 0
		for mm in measures[start_i:]:
			c = counts[mm]
			if chosen and acc + c > budget:
				break
			chosen.append(mm); acc += c
			if acc >= budget:
				break
		chosen_set = set(chosen)
		body_keep = torch.tensor([mm in chosen_set for mm in body_own.tolist()], dtype=torch.bool)
		# Trim the lilylet TAIL: the kept midi patches (header + chosen body) reach up to source
		# measure `max_kept_src`; any lyl patch with a higher measure is cross-attended by nobody
		# (lyl->lyl is causal, lyl is unsupervised). lyl src_meas is monotonic non-decreasing, so
		# the survivors are the prefix [0, new_lyl_count) and their RoPE positions are unchanged.
		midi_keep_rel = torch.cat((torch.ones(header_len, dtype=torch.bool), body_keep))	# over [m0:]
		max_kept_src = int(src_meas[m0:][midi_keep_rel].max())
		lyl_src = src_meas[:m0]
		assert lyl_src.numel() == 0 or bool((lyl_src[1:] >= lyl_src[:-1]).all()), \
			'non-monotonic lilylet src_meas (tail trim assumes a non-decreasing prefix)'
		new_lyl_count = int((lyl_src <= max_kept_src).sum())	# contiguous prefix (monotonic)
		# assemble the kept index mask: lyl prefix + midi header + chosen body measures. The lyl
		# tail (new_lyl_count..m0) is dropped; the midi header at [m0, body_start) is always kept.
		keep_mask = torch.zeros(T, dtype=torch.bool)
		keep_mask[:new_lyl_count] = True					# kept lilylet prefix
		keep_mask[m0:body_start] = True						# midi header
		keep_mask[body_start:] = body_keep
		keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
		# cropped-coord position of the first kept BODY patch: kept lyl prefix + midi header sit at
		# the front (budget>0 guarantees they fit), so it lands at new_lyl_count + header_len.
		cropped_body_start = new_lyl_count + header_len
		# Hard safety clamp: a single measure larger than the budget (dense/sustained bars do
		# occur) would still overflow, so cap the total length at max_patches by dropping the
		# TAIL midi patches. lyl + header sit at the front and are preserved; the truncated tail
		# just ends the midi window early (a causal-valid PARTIAL measure, no natural <eom>).
		clamped = keep_idx.numel() > self.max_patches
		if clamped:
			keep_idx = keep_idx[:self.max_patches]
		# drop targets broken by the crop (see docstring), in cropped coordinates.
		drop_targets = []
		if start_i > 0:
			drop_targets.append(cropped_body_start)			# first kept body patch: cross-boundary predict
		if clamped:
			drop_targets.append(int(keep_idx.numel()) - 1)	# partial-measure tail: no natural <eom>
		return (patches[keep_idx], modality[keep_idx], own_meas[keep_idx],
			src_meas[keep_idx], new_lyl_count, drop_targets)

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
		patches_list = [ex[0] for ex in batch]
		B = len(batch)
		Tmax = max(p.shape[0] for p in patches_list)
		ps = patches_list[0].shape[1]

		input_patches = pad_sequence(patches_list, batch_first=True, padding_value=self.pad_id)	# [B,Tmax,ps]
		input_masks = torch.zeros(B, Tmax, dtype=torch.long)
		modality = torch.zeros(B, Tmax, dtype=torch.long)
		input_targets = torch.zeros(B, Tmax, dtype=torch.long)
		input_positions = torch.zeros(B, Tmax, dtype=torch.long)
		lyl_counts = torch.zeros(B, dtype=torch.long)
		attn_mask = torch.zeros(B, 1, Tmax, Tmax, dtype=torch.bool)

		for b, (patches, mod, own_meas, src_meas, lyl_count, drop_targets) in enumerate(batch):
			T = patches.shape[0]
			input_masks[b, :T] = 1
			modality[b, :T] = mod
			# Positions restart per modality: lilylet patches 0..L-1, midi patches 0..Mp-1.
			# Each segment is its own RoPE coordinate frame (the midi sequence does not inherit
			# the lilylet length as an offset), matching the dual-embedding split.
			input_positions[b, :lyl_count] = torch.arange(lyl_count, dtype=torch.long)
			input_positions[b, lyl_count:T] = torch.arange(T - lyl_count, dtype=torch.long)
			lyl_counts[b] = lyl_count

			# Supervision: ONLY the midi segment is a prediction target. The very first midi
			# patch (index lyl_count) is NOT a target — we never force the lilylet segment's
			# last hidden state to emit the first midi patch across the modality boundary
			# (at generation the deterministic midi header is fed as a prompt instead).
			tgt = (mod == 1).long()
			tgt[0] = 0					# defensive (lyl_count >= 1 always, so index 0 is lyl)
			if lyl_count < T:
				tgt[lyl_count] = 0		# drop the cross-boundary prediction
			# Crop-broken targets: a mid-body window start has no preceding body context, and a
			# hard-clamped partial tail has no natural <eom>; _crop_midi flags those positions.
			for pos in drop_targets:
				if 0 <= pos < T:
					tgt[pos] = 0
			input_targets[b, :T] = tgt

			vis = build_vis(mod, own_meas, src_meas, self.w_midi, self.w_cross)
			attn_mask[b, 0, :T, :T] = vis
			# pad query rows attend self so SDPA softmax stays finite (outputs discarded).
			if T < Tmax:
				diag = torch.arange(T, Tmax)
				attn_mask[b, 0, diag, diag] = True

		return dict(
			input_patches=input_patches.to(self.device),
			input_masks=input_masks.to(self.device),
			input_targets=input_targets.to(self.device),
			input_positions=input_positions.to(self.device),
			attn_mask=attn_mask.to(self.device),
			modality=modality.to(self.device),
			lyl_counts=lyl_counts.to(self.device),
		)


