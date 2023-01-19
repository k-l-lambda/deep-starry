
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


	def __init__(self, root, split='0/1', device='cpu', labels=None, epoch_size=100, shuffle=False, crop_size=32, bias_sigma=0.2):
		super().__init__()

		self.reader, self.root = makeReader(root)
		self.device = device
		self.labels = labels
		self.shuffle = shuffle
		self.epoch_size = epoch_size
		self.crop_size = crop_size
		self.bias_sigma = bias_sigma

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

			bias_max = (source.shape[-1] - self.crop_size) // 2
			bias_x, bias_y = np.random.randn() * self.bias_sigma, np.random.randn() * self.bias_sigma
			bias_x = int(max(-bias_max, min(bias_max, bias_x)))
			bias_y = int(max(-bias_max, min(bias_max, bias_y)))
			#print('bias:', bias_x, bias_y)

			# center crop
			source = source[bias_max + bias_x:bias_max + bias_x + self.crop_size, bias_max + bias_y:bias_max + bias_y + self.crop_size]

			source = source.float() / 255.

			yield source, li


	def collateBatch (self, batch):
		feature = torch.stack([ex[0][None, :] for ex in batch], dim=0).to(self.device)
		label = torch.tensor([ex[1] for ex in batch], dtype=torch.long).to(self.device)

		return feature, label
