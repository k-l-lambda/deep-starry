
import os
import numpy as np
import random
import time
import logging
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from .utils import loadSplittedDatasets
from .score import makeReader, parseFilterStr
#from .augmentor2 import Augmentor2



def listAllImageNames (reader, filterStr, dir='/'):
	# split file name & ext name
	all_names = [os.path.splitext(name)[0] for name in reader.listFiles(dir)]

	if filterStr is None:
		return all_names

	phases, cycle = parseFilterStr(filterStr)

	return [name for i, name in enumerate(all_names) if (i % cycle) in phases]


def collateBatchSingle (batch, device):
	assert len(batch) == 1

	source, labels = batch[0]
	source = source.reshape((1,) + source.shape)
	source = torch.from_numpy(source).permute(0, 3, 1, 2).to(device)

	#target = torch.tensor([[labels[field] for field in fields]], dtype=torch.float32).to(device)

	return source, labels


class BalanceLabeledPeris (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, labels, device='cpu', args_variant=None):
		return loadSplittedDatasets(cls, root=root, labels=labels, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, labels, label_fields, split='0/1', device='cpu', epoch_n=None, groups=[], augmentor={}, shuffle=False, score_binary=None, **_):
		self.reader, self.root = makeReader(root)
		self.shuffle = shuffle
		self.label_fields = label_fields
		self.device = device
		self.epoch_n = epoch_n
		self.score_binary = score_binary

		names = listAllImageNames(self.reader, split)

		dataframes = pd.read_csv(labels)
		dataframes = dataframes[dataframes.hash.isin(names)]

		group_fns = [eval(f'lambda _1: {code}') for code in groups]
		df_groups = [dataframes[fn(dataframes)] for fn in group_fns]

		self.labels = [group.to_dict('records') for group in df_groups]

		self.augmentor = Augmentor2(augmentor, device=device)

		def iter_records (records):
			while True:
				if self.shuffle:
					#np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))
					random.shuffle(records)

				for record in records:
					yield record

		self.record_its = [iter(iter_records(records)) for records in self.labels if len(records) > 0]

		def iter_groups ():
			while True:
				for g, it in enumerate(self.record_its):
					yield next(it)
		self.groups_it = iter(iter_groups())


	def getImage (self, name):
		filename = f'{name}.jpg'
		if not self.reader.exists(filename):
			logging.warn('image file missing, removed: %s', name)
			return
		source = self.reader.readImage(filename)
		if source is None:
			logging.warn('image reading failed, removed: %s', name)
			return

		if len(source.shape) < 3:
			source = source.reshape(source.shape + (1,))
		if source.shape[2] == 1:
			source = np.concatenate((source, source, source), axis=2)
		elif source.shape[2] > 3:
			source = source[:, :, :3]

		source = (source / 255.0).astype(np.float32)

		return source


	def __iter__ (self):
		for i in range(self.epoch_n):
			record = next(self.groups_it)
			source = self.getImage(record['hash'])
			if source is None:
				continue

			if self.score_binary:
				score = record['score']
				record['score_binary'] = [1 if score >= threshold else 0 for threshold in self.score_binary['thresholds']]

			yield source, record


	def __len__ (self):
		return self.epoch_n


	def collateBatch (self, batch):
		source, labels = collateBatchSingle(batch, self.device)
		source, labels = self.augmentor.augment(source, labels)

		target = torch.tensor([[labels[field] for field in self.label_fields]], dtype=torch.float32).to(self.device)

		if len(self.label_fields) == 1 and len(target.shape) > 2:
			target = target.squeeze(1)

		return source, target
