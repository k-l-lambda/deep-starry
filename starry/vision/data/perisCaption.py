
import os
import time
import numpy as np
import torch
import random
import logging
import pandas as pd
import json
from torch.utils.data import IterableDataset

from .utils import loadSplittedDatasets, listAllImageNames
from .score import makeReader



FIGURE_WORD = os.getenv('FIGURE_WORD')


class AliasWord:
	def __init__(self, config):
		self.alias = {}
		if config:
			with open(config) as file:
				self.alias = json.load(file)


	def __call__ (self, word):
		if word in self.alias:
			return self.alias[word]

		return word


class PerisCaption (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, labels, device='cpu', args_variant=None):
		return loadSplittedDatasets(cls, root=root, labels=labels, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, labels, tokenizer, split='0/1', shuffle=False, alias=None, **_):
		self.reader, self.root = makeReader(root)
		self.shuffle = shuffle
		#self.device = device
		self.tokenizer = tokenizer

		dataframes = pd.read_csv(labels)
		self.labels = dict(zip(dataframes['hash'], dataframes.to_dict('records')))

		self.names = listAllImageNames(self.reader, split)
		self.names = [name for name in self.names if self.labels.get(name)]

		self.alias = AliasWord(alias)


	def perisCaption (self, record):
		#print('record:', record)
		style = 'painting' if record.get('PA') else ('doll' if record['DOLL'] else 'photo')

		modifiers = []
		if record['score'] >= 6:
			modifiers.append(self.alias('<p6+>'))
		if record['score'] >= 7:
			modifiers.append(self.alias('<p7+>'))
		if record['score'] >= 8:
			modifiers.append(self.alias('<p8+>'))
		if record['score'] >= 9:
			modifiers.append(self.alias('<p9+>'))
		if record['LOLI']:
			modifiers.append(self.alias('LOLI'))

		descriptions = []
		if record['SE']:
			descriptions.append(self.alias('SE'))
		if record['SM']:
			descriptions.append(self.alias('SM'))
		if record['NF']:
			descriptions.append(self.alias('NF'))
		if record['identity'] and type(record['identity']) is str:
			descriptions.append(f'name "{record["identity"]}"')

		return ', '.join([f'a {style} of a {" ".join(modifiers)} {FIGURE_WORD}'] + descriptions)


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

			source = (source / (255.0 / 2.) - 1).astype(np.float32)

			caption = self.perisCaption(self.labels[name])

			token_dict = self.tokenizer(caption,
				padding='max_length',
				truncation=True,
				max_length=self.tokenizer.model_max_length,
				return_tensors='pt')

			example = {
				'text': caption,
				'input_ids': token_dict.input_ids[0],
				'attention_mask': token_dict.attention_mask[0],
				'pixel_values': torch.from_numpy(source).permute(2, 0, 1)
			}

			yield example


	def __len__ (self):
		return len(self.names)
