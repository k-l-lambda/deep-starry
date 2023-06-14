
import numpy as np
import torch
from torch.utils.data import IterableDataset

from ...utils.parsers import parseFilterStr, mergeArgs
from .paraffFile import ParaffFile



class SentenceShift (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		splits = splits.split(':')

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return (
			cls(root, split, device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)


	def __init__ (self, root, split, device, shuffle, n_seq, descriptor_drop=0.1, descriptor_drop_sigma=0., BOM='BOM', EOM='EOM', **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.descriptor_drop = descriptor_drop
		self.descriptor_drop_sigma = descriptor_drop_sigma

		phases, cycle = parseFilterStr(split)

		file = ParaffFile(open(root, 'rb'))

		padding_zeros = [0] * (n_seq + 1 - file.sentence_align_size)
		sentences = [s + padding_zeros for i, s in enumerate(file.sentences) if i % cycle in phases]
		self.entries = torch.tensor(sentences, dtype=torch.uint8)

		#self.tokens = file.tokens
		self.id_BOM = file.tokens.index(BOM)
		self.id_EOM = file.tokens.index(EOM)


	def __iter__ (self):
		entries = self.entries.clone()

		if self.shuffle:
			entries = entries[torch.randperm(self.entries.shape[0])]

		mtx_sum = torch.triu(torch.ones(entries.shape[-1], entries.shape[-1]), diagonal=0)

		body_mask = (entries == self.id_BOM).float()
		body_mask = body_mask.matmul(mtx_sum)

		drop_p_pow = torch.randn(body_mask.shape[0], dtype=torch.float) * self.descriptor_drop_sigma
		drop_p = torch.pow(self.descriptor_drop, torch.exp(drop_p_pow))[:, None]

		drops = (1 - body_mask) * (torch.rand_like(body_mask) < drop_p)
		indices = torch.arange(body_mask.shape[-1])[None, :].repeat(drops.shape[0], 1)
		for idx, drop in zip(indices, drops):
			idx_mask = idx[drop == 0]
			idx[:idx_mask.shape[0]] = idx_mask
		indices = indices.long().clip(max=entries.shape[-1] - 1)

		n_descs = (1 - body_mask - drops).int().sum(dim=1).tolist()

		# drop descriptors
		for i, idx in enumerate(indices):
			# shuffle descriptors
			if self.shuffle:
				n_desc = n_descs[i]
				idx[:n_desc] = idx[:n_desc][torch.randperm(n_desc)]

			entries[i] = entries[i].index_select(0, idx)
			body_mask[i] = body_mask[i].index_select(0, idx)

		body_mask = body_mask.bool() & (entries != 0)
		body_mask[entries == self.id_EOM] = False

		for entry, mask in zip(entries, body_mask):
			yield entry[:-1], entry[1:], mask[:-1]


	def __len__ (self):
		return len(self.entries)


	def collateBatch (self, batch):
		input_ids = [ex[0] for ex in batch]
		output_ids = [ex[1] for ex in batch]
		body_mask = [ex[2] for ex in batch]

		input_ids = torch.stack(input_ids, axis=0).to(self.device)
		output_ids = torch.stack(output_ids, axis=0).to(self.device)
		body_mask = torch.stack(body_mask, axis=0).to(self.device)

		return dict(input_ids=input_ids, output_ids=output_ids, body_mask=body_mask)
