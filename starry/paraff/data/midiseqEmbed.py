
import os
import dill as pickle
import numpy as np
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
	def loadMeasures (cls, paraff_path, n_seq, root, device):
		if paraff_path in cls.measure_lib:
			return cls.measure_lib[paraff_path]

		summaries_path = root + '-midiseq-measures.pt'
		summaries = torch.load(summaries_path, map_location=device)

		with open(paraff_path, 'rb') as paraff_file:
			cls.measure_lib[paraff_path] = MeasureLibrary(paraff_file, n_seq, summaries)

		return cls.measure_lib[paraff_path]


	def __init__ (self, root, split, device, shuffle, n_seq_paraff=256, blend_p=0, blend_length_sigma=0.2, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle

		paraff_path = root + '-midiseq.paraff'
		midiseq_path = root + '.midiseq.pkl'

		self.midiseq = pickle.load(open(midiseq_path, 'rb'))

		phases, cycle = parseFilterStr(split)
		scoreIndices = list(map(int, self.midiseq['scoreIndices']))
		startidx, endidx = scoreIndices[:-1], scoreIndices[1:]
		self.spans = [span for i, span in enumerate(zip(startidx, endidx)) if i % cycle in phases]

		self.measure = self.loadMeasures(paraff_path, n_seq_paraff, root, self.device)

		self.blend_p = blend_p
		self.blend_length_sigma = blend_length_sigma


	def __len__ (self):
		return sum([span[1] - span[0] for span in self.spans])


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.spans)
		else:
			torch.manual_seed(0)
			np.random.seed(1)

		for span in self.spans:
			sidx, eidx = span
			for idx in range(sidx, eidx):
				summary = self.measure.summaries[idx]
				seq = self.midiseq['seqs'][idx]

				if idx < eidx - 1 and self.blend_p > 0 and np.random.rand() < self.blend_p:
					next_summary = self.measure.summaries[idx + 1]
					next_seq = self.midiseq['seqs'][idx + 1]

					k = np.random.rand()
					k1 = min(1, k * np.exp(np.random.randn() * self.blend_length_sigma))
					k2 = min(1, (1 - k) * np.exp(np.random.randn() * self.blend_length_sigma))
					print(f'{k1=}, {k2=}')

					seq1, seq2 = np.array(seq), np.array(next_seq)
					seq1, seq2 = seq1[seq1 != 0], seq2[seq2 != 0]

					n_seq1 = max(1, int(len(seq1) * k1))
					n_seq2 = max(1, int(len(seq2) * k2))
					print(f'{n_seq1=}, {n_seq2=}')

					seq_cat = np.concatenate([seq1[-n_seq1:], seq2[:n_seq2]])
					blend_seq = np.zeros_like(seq)
					blend_seq[:len(seq_cat)] = seq_cat[:len(blend_seq)]

					blend_summary = summary * k1 + next_summary * k2

					yield blend_summary, blend_seq.tolist()
				else:
					yield summary, seq


	def collateBatch (self, batch):
		def extract (i, padding=False, dtype=None):
			tensors = [ex[i] for ex in batch]
			if padding:
				n_seq = max([len(t) for t in tensors])
				tensor = torch.zeros(len(batch), n_seq, dtype=dtype)
				for i, t in enumerate(tensors):
					tensor[i, :len(t)] = torch.tensor(t, dtype=dtype)

				return tensor.to(self.device)

			return torch.stack(tensors, axis=0).to(self.device)

		summary, seq = extract(0), extract(1, padding=True, dtype=torch.long)

		return dict(
			summary=summary,
			seq=seq,
		)
