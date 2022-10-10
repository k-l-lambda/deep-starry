
import os
import yaml
from torch.utils.data import IterableDataset

from ...utils.parsers import parseFilterStr, mergeArgs



class VocalPitch (IterableDataset):
	@classmethod
	def loadPackage (cls, root, args, splits='*0/1', device='cpu', args_variant=None):
		splits = splits.split(':')

		utterances_file = open(os.path.join(root, 'utterances.yaml'), 'r', encoding='utf-8')
		utterances = yaml.safe_load(utterances_file)

		bias_limit = args.get('bias_limit', 0.3)
		ids = [ut for ut in utterances['utterances'] if utterances['utterances'][ut].get('bias', 1) < bias_limit]
		print('ids:', ids)

		def loadEntries (split):
			phases, cycle = parseFilterStr(split)

			return [id for i, id in enumerate(ids) if i % cycle in phases]

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return (
			cls(root, loadEntries(split), device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)


	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None):
		return cls.loadPackage(root, args, splits, device, args_variant=args_variant)


	def __init__ (self, root, ids, device, shuffle, **_):
		self.ids = ids


	def __len__(self):
		return len(self.ids)


	def __iter__ (self):
		pass


	def collateBatch (self, batch):
		pass
