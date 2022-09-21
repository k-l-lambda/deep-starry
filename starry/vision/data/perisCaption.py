
import os
import time
import numpy as np
import random
import logging
import pandas as pd
from torch.utils.data import IterableDataset

from .utils import loadSplittedDatasets, listAllImageNames
from .score import makeReader



FIGURE_WORD = os.getenv('FIGURE_WORD')


def perisCaption (record):
	#print('record:', record)
	style = 'painting' if record['PA'] else ('doll' if record['DOLL'] else 'photo')

	modifiers = []
	if record['score'] >= 6:
		modifiers.append('<p6+>')
	if record['score'] >= 7:
		modifiers.append('<p7+>')
	if record['score'] >= 8:
		modifiers.append('<p8+>')
	if record['score'] >= 9:
		modifiers.append('<p9+>')
	if record['LOLI']:
		modifiers.append('young')

	descriptions = []
	if record['SE']:
		descriptions.append('SE')
	if record['SM']:
		descriptions.append('SM')
	if record['NF']:
		descriptions.append('no face')
	if record['identity'] and type(record['identity']) is str:
		descriptions.append(f'name "{record["identity"]}"')

	return ', '.join([f'a {style} of a {" ".join(modifiers)} {FIGURE_WORD}'] + descriptions)


class PerisCaption (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, labels, device='cpu', args_variant=None):
		return loadSplittedDatasets(cls, root=root, labels=labels, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, labels, split='0/1', device='cpu', shuffle=False, **_):
		self.reader, self.root = makeReader(root)
		self.shuffle = shuffle
		self.device = device

		dataframes = pd.read_csv(labels)
		self.labels = dict(zip(dataframes['hash'], dataframes.to_dict('records')))

		self.names = listAllImageNames(self.reader, split)
		self.names = [name for name in self.names if self.labels.get(name)]


	def __iter__ (self):
		if self.shuffle:
			random.shuffle(self.names)
			np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))

		for i, name in enumerate(self.names):
			filename = f'{name}.jpg'
			if not self.reader.exists(filename):
				self.names.remove(name)
				logging.warn('image file missing, removed: %s', name)
				continue
			source = self.reader.readImage(filename)
			if source is None:
				self.names.remove(name)
				logging.warn('image reading failed, removed: %s', name)
				continue

			if len(source.shape) < 3:
				source = source.reshape(source.shape + (1,))
			if source.shape[2] == 1:
				source = np.concatenate((source, source, source), axis=2)
			elif source.shape[2] > 3:
				source = source[:, :, :3]

			source = (source / 255.0).astype(np.float32)

			caption = perisCaption(self.labels[name])

			yield source, caption


	def __len__ (self):
		return len(self.names)
