
import random
import yaml
from fs import open_fs
import numpy as np
import dill as pickle
import torch
from torch.utils.data import IterableDataset

from starry.melody.notation import KEYBOARD_SIZE

from ...utils.parsers import parseFilterStr, mergeArgs



class FramePair (IterableDataset):
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


	def __init__ (self, package, entries, device, shuffle=False, c_seq_len=0x20, s_seq_len=0x10, ci_bias_sigma=5, use_cache=False,
		ci_center_position=0.5, st_scale_sigma=0, ci_bias_constant=-1, random_time0=0):
		self.package = package
		self.entries = entries
		self.shuffle = shuffle
		self.device = device

		self.c_seq_len = c_seq_len
		self.s_seq_len = s_seq_len
		self.ci_bias_sigma = ci_bias_sigma
		self.ci_bias_constant = ci_bias_constant
		self.ci_center_position = ci_center_position
		self.st_scale_sigma = st_scale_sigma
		self.random_time0 = random_time0

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

		n_post_center = int(self.c_seq_len * self.ci_center_position)

		for entry in self.entries:
			pair = self.readEntry(entry.name)
			criterion = pair['criterion']
			criterion_len = len(criterion['time'])

			if self.shuffle:
				np.random.shuffle(pair['samples'])
			for sample in pair['samples']:
				sample_len = len(sample['time'])

				for si in range(1, sample_len + 1):
					self.profile_check('iter.0')

					seen_chis = sample['chi'][:si]
					ci0 = seen_chis[seen_chis >= 0][-1].item()
					s_time0 = sample['time'][si - 1]
					s0i = max(si - self.s_seq_len, 0)

					pitch_bias = random.randint(-12, 12) if self.shuffle else 0

					#self.profile_check('iter.1')

					center_ci = max(0, min(criterion_len - 1, round(ci0 + self.ci_bias_constant + np.random.randn() * self.ci_bias_sigma)))
					ci_range = (max(0, center_ci - (self.c_seq_len - n_post_center) + 1), min(criterion_len, center_ci + n_post_center + 1))
					ci_range_len = ci_range[1] - ci_range[0]
					c_time0 = criterion['time'][center_ci]
					#self.profile_check('iter.1.1')

					s_ci = sample['chi'][s0i:si]
					#self.profile_check('iter.1.2')
					cis = torch.tensor([(ci - ci_range[0] + 1 if (ci >= ci_range[0] and ci < ci_range[1]) else 0) for ci in s_ci], dtype=torch.long)

					#self.profile_check('iter.2')

					st_scale = 1 if self.st_scale_sigma == 0 else np.exp(np.random.randn() * self.st_scale_sigma)

					#self.profile_check('iter.3')

					s_time, s_frame, ci = torch.zeros(self.s_seq_len, dtype=torch.float32), torch.zeros((self.s_seq_len, KEYBOARD_SIZE), dtype=torch.float32), torch.zeros(self.s_seq_len, dtype=torch.long)
					s_time[-si:] = (sample['time'][s0i:si] - s_time0) * st_scale
					frame_filling = sample['frame'][s0i:si]
					if pitch_bias != 0:
						frame_filling = torch.cat((frame_filling[:, -pitch_bias:], frame_filling[:, :-pitch_bias]), dim=1)

					frame_filling = -torch.log((1 - frame_filling).clip(min=1e-12)) * torch.exp(torch.randn(*frame_filling.shape) + 0.4)

					# noise in frame
					gain = torch.exp(torch.randn(frame_filling.shape[0], 1) * 1.2 - 4.8)
					#gain = torch.exp(torch.ones(frame_filling.shape[0], 1) * -3.6)
					frame_filling += torch.exp(torch.randn(*frame_filling.shape)) * gain
					frame_filling = torch.tanh(frame_filling)

					s_frame[-frame_filling.shape[0]:] = frame_filling

					ci[-si:] = cis

					#self.profile_check('iter.4')

					c_time, c_frame = torch.zeros(self.c_seq_len, dtype=torch.float32), torch.zeros((self.c_seq_len, KEYBOARD_SIZE), dtype=torch.float32)
					c_time[:ci_range_len] = criterion['time'][ci_range[0]:ci_range[1]] - c_time0
					c_frame[:ci_range_len] = criterion['frame'][ci_range[0]:ci_range[1]]
					if pitch_bias != 0:
						c_frame = torch.cat((c_frame[:, -pitch_bias:], c_frame[:, :-pitch_bias]), dim=1)

					if self.random_time0 > 0:
						s_time += (np.random.rand() - 0.5) * 2 * self.random_time0
						c_time += (np.random.rand() - 0.5) * 2 * self.random_time0

					self.profile_check('iter.-1')

					#if ci.max() <= 0:
					#	print('empty example:', ci_range, n_post_center, center_ci, ci0 + self.ci_bias_constant)

					yield c_time, c_frame, s_time, s_frame, ci


	def collateBatch (self, batch):
		#self.profile_check('collateBatch.0')

		def extract (i):
			stack = [tensors[i].unsqueeze(0) for tensors in batch]
			return torch.cat(stack, dim=0).to(self.device)

		result = {
			'criterion': (extract(0), extract(1)),
			'sample': (extract(2), extract(3)),
			'ci': extract(4),
		}

		#self.profile_check('collateBatch.-1')

		return result
