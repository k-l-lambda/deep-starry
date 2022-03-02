
import yaml
from fs import open_fs
import numpy as np
import dill as pickle
import torch
from torch.utils.data import IterableDataset

from ...utils.parsers import parseFilterStr
from ..event_element import TARGET_FIELDS



class EventCluster (IterableDataset):
	@classmethod
	def loadPackage (cls, url, splits='*0/1', device='cpu', **kwargs):
		splits = splits.split(':')
		package = open_fs(url)
		index_file = package.open('index.yaml', 'r')
		index = yaml.safe_load(index_file)

		def loadEntries (split):
			phases, cycle = parseFilterStr(split)

			ids = [id for i, id in enumerate(index['groups']) if i % cycle in phases]

			return [entry for entry in index['examples'] if entry['group'] in ids]

		return tuple(map(lambda split: cls(
			package, loadEntries(split), device, shuffle='*' in split, **kwargs,
		), splits))


	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None):
		url = f'zip://{root}' if root.endswith('.zip') else root
		return cls.loadPackage(url, splits, device, **args)


	def __init__ (self, package, entries, device, shuffle=False, stability_base=10, position_drift=0):
		self.package = package
		self.entries = entries
		self.shuffle = shuffle
		self.device = device

		self.stability_base = stability_base
		self.position_drift = position_drift


	def __len__(self):
		return len(self.entries)


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.entries)
		else:
			torch.manual_seed(0)
			np.random.seed(len(self.entries))

		for entry in self.entries:
			with self.package.open(entry['filename'], 'rb') as file:
				tensors = pickle.load(file)
				yield tensors


	def collateBatch (self, batch):
		assert len(batch) == 1

		tensors = batch[0]
		feature_shape = tensors['feature'].shape
		batch_size = feature_shape[0]
		n_seq = feature_shape[1]

		# extend batch dim for single tensors
		elem_type = tensors['type'].repeat(batch_size, 1).long()
		staff = tensors['staff'].repeat(batch_size, 1).long()

		# noise augment for feature
		stability = np.random.power(self.stability_base)
		error = torch.rand(*feature_shape) > stability
		chaos = torch.exp(torch.randn(*feature_shape))
		feature = tensors['feature']
		feature[error] = chaos[error]

		# augment for position
		x = tensors['x']
		y1 = tensors['y1']
		y2 = tensors['y2']
		ox, oy = (torch.rand(batch_size, 1) - 0.2) * 24, (torch.rand(batch_size, 1) - 0.2) * 12
		if self.position_drift > 0:
			x += torch.randn(batch_size, n_seq) * self.position_drift + ox

			# exclude BOS, EOS from global Y offset
			y1[:, 1:-1] += torch.randn(batch_size, n_seq - 2) * self.position_drift + oy
			y2[:, 1:-1] += torch.randn(batch_size, n_seq - 2) * self.position_drift + oy

		result = {
			'type': elem_type,
			'staff': staff,
			'feature': feature,
			'x': x,
			'y1': y1,
			'y2': y2,
			'matrixH': tensors['matrixH'].repeat(batch_size, 1),
		}

		for field in TARGET_FIELDS:
			result[field] = tensors[field].repeat(batch_size, 1)

		# to device
		for key in result:
			result[key] = result[key].to(self.device)

		return result