
import os
import time
import numpy as np
import torch
import random
import logging
import pandas as pd
import json
from torch.utils.data import IterableDataset
from torchvision import transforms
import torchvision.transforms.functional as F

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


class SquarePad:
	def __init__ (self, padding_mode):
		self.padding_mode = padding_mode


	def __call__(self, image):
		_, h, w = image.shape
		max_wh = np.max([w, h])
		lp = (max_wh - w) // 2
		rp = max_wh - w - lp
		tp = (max_wh - h) // 2
		bp = max_wh - h - tp
		padding = (lp, tp, rp, bp)

		return F.pad(image, padding, 0, self.padding_mode)


class PerisCaption (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, labels, device='cpu', args_variant=None):
		return loadSplittedDatasets(cls, root=root, labels=labels, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, labels, tokenizer, split='0/1', shuffle=False, filter=None, resolution=512, alias=None, **_):
		self.reader, self.root = makeReader(root)
		self.shuffle = shuffle
		self.tokenizer = tokenizer

		dataframes = pd.read_csv(labels)
		if filter is not None:
			fn = eval(f'lambda _1: {filter}')
			dataframes = dataframes[fn(dataframes)]
		self.labels = dict(zip(dataframes['hash'], dataframes.to_dict('records')))

		self.names = listAllImageNames(self.reader, split)
		self.names = [name for name in self.names if self.labels.get(name)]

		self.alias = AliasWord(alias)

		self.transform = transforms.Compose([
			SquarePad(padding_mode='reflect'),
			transforms.Resize(resolution),
			transforms.RandomHorizontalFlip(p=0.5),
		])


	def perisCaption (self, record):
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

		return ', '.join([f'a peris {style} of a {" ".join(modifiers)} {FIGURE_WORD}'] + descriptions)


	def __iter__ (self):
		if self.shuffle:
			random.shuffle(self.names)
			#np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))

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

			# skip too oblong image to avoid reflection padding error
			h, w = source.shape[:2]
			long_edge, short_edge = max(h, w), min(h, w)
			if long_edge / short_edge > 2.6:
				continue

			source = (source / (255.0 / 2.) - 1).astype(np.float32)
			source = torch.from_numpy(source).permute(2, 0, 1)
			source = self.transform(source)

			caption = self.perisCaption(self.labels[name])

			token_dict = self.tokenizer(caption,
				padding='max_length',
				truncation=True,
				max_length=self.tokenizer.model_max_length,
				return_tensors='pt')

			example = {
				#'text': caption,
				'input_ids': token_dict.input_ids[0],
				#'attention_mask': token_dict.attention_mask[0],
				'pixel_values': source,
			}

			yield example


	def __len__ (self):
		return len(self.names)
