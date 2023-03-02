
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


	def __init__ (self, root, split, device, shuffle, n_seq, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle

		phases, cycle = parseFilterStr(split)

		file = ParaffFile(open(root, 'rb'))

		padding_zeros = [0] * (n_seq + 1 - file.sentence_align_size)
		sentences = [s + padding_zeros for i, s in enumerate(file.sentences) if i % cycle in phases]
		self.entries = torch.tensor(sentences, dtype=torch.int8)


	def __iter__ (self):
		if self.shuffle:
			self.entries = self.entries[torch.randperm(self.entries.shape[0])]

		for entry in self.entries:
			yield entry[:-1], entry[1:]


	def __len__ (self):
		return len(self.entries)


	def collateBatch (self, batch):
		input_ids = [ex[0] for ex in batch]
		output_ids = [ex[1] for ex in batch]

		input_ids = torch.stack(input_ids, axis=0).to(self.device)
		output_ids = torch.stack(output_ids, axis=0).to(self.device)

		return dict(input_ids=input_ids, output_ids=output_ids)
