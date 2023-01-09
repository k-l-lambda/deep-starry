
import random
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


	def __init__ (self, package, entries, device, shuffle=False, seq_len=0x100, ci_bias_sigma=5, use_cache=False,
		ci_center_position=0.5, st_scale_sigma=0, ci_bias_constant=-1, random_time0=0., guid_rate=0, guid_rate_sigma=0.4):
		self.package = package
		self.entries = entries
		self.shuffle = shuffle
		self.device = device

		self.seq_len = seq_len
		self.ci_bias_sigma = ci_bias_sigma
		self.ci_bias_constant = ci_bias_constant
		self.ci_center_position = ci_center_position
		self.st_scale_sigma = st_scale_sigma
		self.random_time0 = random_time0
		self.guid_rate = guid_rate
		self.guid_rate_sigma = guid_rate_sigma

		self.entry_cache = {} if use_cache else None

		# dummy profile_check
		self.profile_check = lambda x: x

		with self.package.open('index.yaml', 'r') as file:
			index_info = yaml.safe_load(file)
			ex = [info for info in index_info['examples'] if any(entry for entry in entries if entry.name.startswith(info['name']))]

			sample_ns = [[sample['length'] + 1 for sample in info['samples']] for info in ex]
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

		n_post_center = int(self.seq_len * self.ci_center_position)

		for entry in self.entries:
			pair = self.readEntry(entry.name)
			criterion = pair['criterion']
			criterion_len = len(criterion['time'])
			#criterion_indices = [*range(criterion_len)]

			if self.shuffle:
				np.random.shuffle(pair['samples'])
			for sample in pair['samples']:
				sample_len = len(sample['time'])

				sample_ctime = torch.index_select(criterion['time'], 0, sample['ci'])

				for si in range(1, sample_len + 1):
					self.profile_check('iter.0')

					ci0 = sample['ci'][si - 1].item()
					s_time0 = sample['time'][si - 1]
					s0i = max(si - self.seq_len, 0)

					pitch_bias = random.randint(-12, 12) if self.shuffle else 0

					#self.profile_check('iter.1')

					center_ci = max(0, min(criterion_len - 1, round(ci0 + self.ci_bias_constant + np.random.randn() * self.ci_bias_sigma)))
					ci_range = (max(0, center_ci - (self.seq_len - n_post_center) + 1), min(criterion_len, center_ci + n_post_center + 1))
					ci_range_len = ci_range[1] - ci_range[0]
					c_time0 = criterion['time'][center_ci]
					#self.profile_check('iter.1.1')

					#c_ci = criterion_indices[ci_range[0]:ci_range[1]]
					s_ci = sample['ci'][s0i:si]
					#self.profile_check('iter.1.2')
					cis = torch.tensor([(ci - ci_range[0] + 1 if (ci >= ci_range[0] and ci < ci_range[1]) else 0) for ci in s_ci], dtype=torch.long)

					#self.profile_check('iter.2')

					# velocity blur
					velocity_bias = torch.randn(si - s0i).int() if self.shuffle else 0

					st_scale = 1 if self.st_scale_sigma == 0 else np.exp(np.random.randn() * self.st_scale_sigma)

					#self.profile_check('iter.3')

					s_time, s_pitch, s_velocity, ci = torch.zeros(self.seq_len, dtype=torch.float32), torch.zeros(self.seq_len, dtype=torch.long), torch.zeros(self.seq_len, dtype=torch.float32), torch.zeros(self.seq_len, dtype=torch.long)
					s_time[-si:] = (sample['time'][s0i:si] - s_time0) * st_scale
					s_pitch[-si:] = sample['pitch'][s0i:si] + pitch_bias
					s_velocity[-si:] = sample['velocity'][s0i:si] + velocity_bias
					ci[-si:] = cis

					s_guid, s_guid_mask = torch.zeros(self.seq_len, dtype=torch.float32), torch.zeros(self.seq_len, dtype=torch.bool)
					s_guid[-si:] = sample_ctime[s0i:si] - c_time0
					n_guid = int(self.seq_len * (1 - (1 - self.guid_rate) * np.exp(np.random.randn() * self.guid_rate_sigma)))
					n_guid = min(n_guid, self.seq_len - 1)
					if si > self.seq_len - n_guid:
						s_guid_mask[-si:n_guid] = True
					s_ng_mask = torch.logical_not(s_guid_mask)
					s_guid[s_ng_mask] = 0

					#self.profile_check('iter.4')

					c_time, c_pitch, c_velocity = torch.zeros(self.seq_len, dtype=torch.float32), torch.zeros(self.seq_len, dtype=torch.long), torch.zeros(self.seq_len, dtype=torch.float32)
					c_time[:ci_range_len] = criterion['time'][ci_range[0]:ci_range[1]] - c_time0
					c_pitch[:ci_range_len] = criterion['pitch'][ci_range[0]:ci_range[1]] + pitch_bias
					c_velocity[:ci_range_len] = criterion['velocity'][ci_range[0]:ci_range[1]]

					if self.random_time0 > 0:
						bias_s, bias_c = (np.random.rand() - 0.5) * 2 * self.random_time0, (np.random.rand() - 0.5) * 2 * self.random_time0
						s_time += bias_s
						c_time += bias_c
						s_guid += bias_c

					self.profile_check('iter.-1')

					yield c_time, c_pitch, c_velocity, s_time, s_pitch, s_velocity, ci, s_guid, s_ng_mask


	def collateBatch (self, batch):
		#self.profile_check('collateBatch.0')

		def extract (i):
			stack = [tensors[i].unsqueeze(0) for tensors in batch]
			return torch.cat(stack, dim=0).to(self.device)

		result = {
			'criterion': (extract(0), extract(1), extract(2)),
			'sample': (extract(3), extract(4), extract(5), extract(7), extract(8)) if self.guid_rate > 0 else (extract(3), extract(4), extract(5)),
			'ci': extract(6),
		}

		#self.profile_check('collateBatch.-1')

		return result
