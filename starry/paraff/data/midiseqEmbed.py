
import os
import dill as pickle
import torch
from torch.utils.data import IterableDataset

from ...utils.parsers import parseFilterStr, mergeArgs
from .paragraph import MeasureLibrary



class MidiseqEmbed (IterableDataset):
	measure_lib = {}


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


	@classmethod
	def loadMeasures (cls, paraff_path, n_seq, encoder_config=None):
		if paraff_path in cls.measure_lib:
			return cls.measure_lib[paraff_path]

		cls.measure_lib[paraff_path] = MeasureLibrary(open(paraff_path, 'rb'), n_seq, encoder_config)

		return cls.measure_lib[paraff_path]


	def __init__ (self, root, split, device, shuffle, n_seq_paraff=256, paraff_encoder=None, **_):
		super().__init__()

		paraff_path = root + '.paraff'
		midiseq_path = root + '.midiseq.pkl'

		self.measure = self.loadMeasures(paraff_path, n_seq_paraff, paraff_encoder)

		self.midiseq = pickle.load(open(midiseq_path, 'rb'))


	def __len__ (self):
		# TODO:
		return self.measure.entries.shape[0]
	

	def __iter__ (self):
		pass


	def collateBatch (self, batch):
		pass
