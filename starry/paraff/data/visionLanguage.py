
import torch
from torch.utils.data import IterableDataset

from ...utils.parsers import parseFilterStr, mergeArgs
import pandas as pd



class VisionLanguage (IterableDataset):
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


	def __init__ (self, root, split, device, shuffle, prompt_template, **_):
		super().__init__()

		total_table = pd.read_csv(root + '.csv')

		phases, cycle = parseFilterStr(split)
		self.table = total_table.iloc[[i for i in range(len(total_table)) if i % cycle in phases]]

		# TODO:


	def __len__ (self):
		return len(self.table)


	def __iter__ (self):
		# TODO:
		for _, row in self.table.iterrows():
			yield row['image'], row['sentence']


	def collateBatch (self, batch):
		sentence = [ex[1] for ex in batch]

		return dict(sentence=sentence)
