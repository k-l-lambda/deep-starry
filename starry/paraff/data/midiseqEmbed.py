
#import os
import dill as pickle
import numpy as np
import torch
from torch.utils.data import IterableDataset

from ...utils.parsers import parseFilterStr, mergeArgs
from .paragraph import MeasureLibrary
from ..midiseq import T2I, ID_PEDAL0
from ...utils.registry import register_dataset



MSUM = T2I['MSUM']
BOS = T2I['BOS']
EOS = T2I['EOS']


def wrapSentence (seq):
	# in causal mask, EOS shouldn't see MSUM
	wseq = [MSUM, BOS] + seq + [EOS]

	decoding_mask = [1] * (len(wseq) - 1) + [0]

	return wseq, decoding_mask


@register_dataset
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
	def loadMeasures (cls, paraff_path, root, device):
		if paraff_path in cls.measure_lib:
			return cls.measure_lib[paraff_path]

		summaries_path = root + '-midiseq-measures.pt'
		summaries = torch.load(summaries_path, map_location=device, weights_only=True)

		with open(paraff_path, 'rb') as paraff_file:
			cls.measure_lib[paraff_path] = MeasureLibrary(paraff_file, summaries=summaries)

		return cls.measure_lib[paraff_path]


	def __init__ (self, root, split, device, shuffle, blend_p=0, blend_length_sigma=0.2, n_seq_max=512, drop_pedal_p=0, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.n_seq_max = n_seq_max

		paraff_path = root + '-midiseq.paraff'
		midiseq_path = root + '.midiseq.pkl'

		self.midiseq = pickle.load(open(midiseq_path, 'rb'))

		phases, cycle = parseFilterStr(split)
		scoreIndices = list(map(int, self.midiseq['scoreIndices']))
		startidx, endidx = scoreIndices[:-1], scoreIndices[1:]
		self.spans = [span for i, span in enumerate(zip(startidx, endidx)) if i % cycle in phases]

		self.measure = self.loadMeasures(paraff_path, root, self.device)

		self.blend_p = blend_p
		self.blend_length_sigma = blend_length_sigma

		self.drop_pedal_p = drop_pedal_p


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
				drop_pedal = np.random.rand() < self.drop_pedal_p

				summary = self.measure.summaries[idx]
				seq = self.midiseq['seqs'][idx][:self.n_seq_max - 3]

				if drop_pedal:
					seq = [id for id in seq if id < ID_PEDAL0]

				if idx < eidx - 1 and self.blend_p > 0 and np.random.rand() < self.blend_p:
					next_summary = self.measure.summaries[idx + 1]
					next_seq = self.midiseq['seqs'][idx + 1]

					if drop_pedal:
						next_seq = [id for id in next_seq if id < ID_PEDAL0]

					k = np.random.rand()
					k1 = min(1, k * np.exp(np.random.randn() * self.blend_length_sigma))
					k2 = min(1, (1 - k) * np.exp(np.random.randn() * self.blend_length_sigma))
					#print(f'{k1=}, {k2=}')

					seq1, seq2 = seq, next_seq
					n_seq1 = min(max(1, int(len(seq1) * k1)), self.n_seq_max - 4)
					n_seq2 = min(max(1, int(len(seq2) * k2)), self.n_seq_max - 3 - n_seq1)
					#print(f'{n_seq1=}, {n_seq2=}')

					blend_seq = seq1[-n_seq1:] + seq2[:n_seq2]
					assert len(blend_seq) <= self.n_seq_max, f'blend_seq out of n_seq_max: {len(blend_seq)}'

					blend_summary = summary * k1 + next_summary * k2

					yield blend_summary, *wrapSentence(blend_seq)
				else:
					yield summary, *wrapSentence(seq)


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

		summary, seq, decoding_mask = extract(0), extract(1, padding=True, dtype=torch.long), extract(2, padding=True, dtype=torch.bool)

		return dict(
			summary=summary,
			seq=seq,
			decoding_mask=decoding_mask,
		)
