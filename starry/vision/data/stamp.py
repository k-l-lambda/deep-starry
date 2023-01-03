
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .utils import collateBatch, loadSplittedDatasets
from .score import makeReader, listAllScoreNames



class Stamp (IterableDataset):
	@staticmethod
	def load (root, args, splits, device='cpu', args_variant=None):
		return loadSplittedDatasets(Stamp, root=root, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__(self, root, split='0/1', device='cpu', labels=None, epoch_size=100, shuffle=False):
		super().__init__()

		self.reader, self.root = makeReader(root)
		self.device = device
		self.labels = labels
		self.shuffle = shuffle
		self.epoch_size = epoch_size

		self.names = [listAllScoreNames(self.reader, split, label) for label in labels]
		#print('names:', self.names)


	def __len__ (self):
		return self.epoch_size


	def __iter__ (self):
		for i in range(self.epoch_size):
			li = np.random.randint(0, len(self.labels)) if self.shuffle else i % len(self.labels)
			label = self.labels[li]
			names = self.names[li]

			if len(names) == 0:
				continue

			name = names[np.random.randint(0, len(names))]
			source = self.reader.readImage(os.path.join(label, name + ".png"))
			source = torch.from_numpy(source)

			# TODO: center crop with flicker

			yield source, li


	def collateBatch (self, batch):
		feature = torch.stack([ex[0] for ex in batch], dim=0).to(self.device)
		label = torch.tensor([ex[1] for ex in batch], dtype=torch.long).to(self.device)

		return feature, label
