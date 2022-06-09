
from random import sample
import yaml
from fs import open_fs
import numpy as np
import dill as pickle
import torch
from torch.utils.data import IterableDataset

from ...utils.parsers import parseFilterStr, mergeArgs



class NotationPair (IterableDataset):
	@classmethod
	def loadPackage (cls, url, args, splits='*0/1', device='cpu', args_variant=None):
		splits = splits.split(':')
		package = open_fs(url)
		files = [file for step in package.walk(filter=['*.pkl']) for file in step.files]

		def loadEntries (split):
			phases, cycle = parseFilterStr(split)

			return [file for i, file in enumerate(files) if i % cycle in phases]

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return (
			cls(package, loadEntries(split), device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)


	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None):
		url = f'zip://{root}' if root.endswith('.zip') else root
		return cls.loadPackage(url, args, splits, device, args_variant=args_variant)


	def __init__ (self, package, entries, device, shuffle=False, seq_len=0x100, ci_bias_sigma=5, use_cache=False):
		self.package = package
		self.entries = entries
		self.shuffle = shuffle
		self.device = device

		self.seq_len = seq_len
		self.ci_bias_sigma = ci_bias_sigma

		self.entry_cache = {} if use_cache else None

		with self.package.open('index.yaml', 'r') as file:
			index_info = yaml.safe_load(file)
			ex = [info for info in index_info['examples'] if any(entry for entry in entries if entry.name.startswith(info['name']))]

			sample_ns = [[max(sample['length'] - seq_len, 0) + 1 for sample in info['samples']] for info in ex]
			self.n_examples = sum(n for nn in sample_ns for n in nn)


	def readEntryFromPackage (self, filename):
		with self.package.open(filename, 'rb') as file:
			return pickle.load(file)


	def readEntry (self, filename):
		if self.entry_cache is None:
			return self.readEntryFromPackage(filename)

		data = self.entry_cache.get(filename)
		if data is None:
			data = self.readEntryFromPackage(filename)
			self.entry_cache[filename] = data

		return {**data}


	def __len__(self):
		return self.n_examples


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.entries)
		else:
			torch.manual_seed(0)
			np.random.seed(len(self.entries))

		for entry in self.entries:
			pair = self.readEntry(entry.name)
			criterion = pair['criterion']
			criterion_len = len(criterion['time'])
			criterion_indices = [*range(criterion_len)]

			# TODO: augment pitch and velocity

			for sample in pair['samples']:
				sample_len = len(sample['time'])

				for si in range(1, sample_len):
					ci0 = sample['ci'][si - 1].item()
					s_time0 = sample['time'][si - 1]
					c_time0 = criterion['time'][ci0]
					s0i = max(si - self.seq_len, 0)

					center_ci = round(ci0 - 1 + np.random.randn() * self.ci_bias_sigma)
					ci_range = (max(0, center_ci - self.seq_len // 2), min(criterion_len, center_ci + self.seq_len // 2))
					ci_range_len = ci_range[1] - ci_range[0]

					c_ci = criterion_indices[ci_range[0]:ci_range[1]]
					s_ci = sample['ci'][s0i:si]
					cis = torch.tensor([(c_ci.index(ci) + 1 if ci in c_ci else 0) for ci in s_ci], dtype=torch.long)

					s_time, s_pitch, s_velocity, ci = torch.zeros(self.seq_len, dtype=torch.float32), torch.zeros(self.seq_len, dtype=torch.long), torch.zeros(self.seq_len, dtype=torch.float32), torch.zeros(self.seq_len, dtype=torch.long)
					s_time[-si:], s_pitch[-si:], s_velocity[-si:] = sample['time'][s0i:si] - s_time0, sample['pitch'][s0i:si], sample['velocity'][s0i:si]
					ci[-si:] = cis

					c_time, c_pitch, c_velocity = torch.zeros(self.seq_len, dtype=torch.float32), torch.zeros(self.seq_len, dtype=torch.long), torch.zeros(self.seq_len, dtype=torch.float32)
					c_time[:ci_range_len], c_pitch[:ci_range_len], c_velocity[:ci_range_len] = criterion['time'][ci_range[0]:ci_range[1]] - c_time0, criterion['pitch'][ci_range[0]:ci_range[1]] - c_pitch[0], criterion['velocity'][ci_range[0]:ci_range[1]] - c_velocity[0]

					yield c_time, c_pitch, c_velocity, s_time, s_pitch, s_velocity, ci


	def collateBatch (self, batch):
		def extract (i):
			stack = [tensors[i].unsqueeze(0) for tensors in batch]
			return torch.cat(stack, dim=0).to(self.device)

		return {
			'criterion': (extract(0), extract(1), extract(2)),
			'sample': (extract(3), extract(4), extract(5)),
			'ci': extract(6),
		}