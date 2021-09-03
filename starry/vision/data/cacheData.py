
import numpy as np
from fs import open_fs
import secrets
import torch
from torch.utils.data import IterableDataset



def npSave (file, a):
	np.save(file, a.astype(np.float16))

def npLoad (file):
	return np.load(file).astype(np.float32)


class CachedIterableDataset (IterableDataset):
	def __init__ (self, enable_cache, device='cpu'):
		self.enable_cache = enable_cache
		self.cached = False
		self.sample_count = 0
		self.sample_queue = []
		self.device = device

		if self.enable_cache:
			self.id = secrets.token_hex(8)
			self.fs = open_fs(f'mem://{id}')


	def __iter__ (self):
		self.sample_index = 0

		if self.cached:
			for self.sample_index in range(self.sample_count):
				self.sample_queue.append((True, self.sample_index))
				yield self.sample_index
		else:
			for data in self.iterImpl():
				self.sample_queue.append((False, self.sample_index))
				yield data
				self.sample_index += 1

			self.sample_count = self.sample_index
			if self.enable_cache:
				self.cached = True


	def collateBatch (self, batch):
		feature, label = None, None

		batch_size = len(batch)
		cached, index = self.sample_queue[0]
		batch_id = f'{index}:{index + batch_size}'
		#print('collateBatch:', cached, batch_id)

		self.sample_queue = self.sample_queue[batch_size:]

		if cached:
			feature = npLoad(self.fs.open(f'feature-{batch_id}.npy', 'rb'))
			label = npLoad(self.fs.open(f'label-{batch_id}.npy', 'rb'))
		else:
			#print('sample_index:', self.sample_index)
			feature, label = self.collateBatchImpl(batch)

			if self.enable_cache:
				#print('save:', feature.size, label.size)
				npSave(self.fs.open(f'feature-{batch_id}.npy', 'wb'), feature)
				npSave(self.fs.open(f'label-{batch_id}.npy', 'wb'), label)

		feature = torch.from_numpy(feature).to(self.device)
		label = torch.from_numpy(label).to(self.device)

		return feature, label


	def iterImpl (self):
		pass


	def collateBatchImpl (self, batch):
		pass
