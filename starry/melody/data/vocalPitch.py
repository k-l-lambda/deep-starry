
import os
import yaml
import struct
import numpy as np
import torch
from torch.utils.data import IterableDataset

from ...utils.perlin1d import Perlin1d
from ...utils.parsers import parseFilterStr, mergeArgs
from ..vocal import PITCH_RANGE, PITCH_SUBDIV, GAIN_RANGE



def distort (y, noise, xp):
	return torch.from_numpy(np.interp(noise, xp, y))


def sampleXp (y, size, x_max):
	xp = [i * x_max / (size - 1) for i in range(size)]
	xp_sharp = [*xp]

	# sharp zero cliff
	for i in range(len(xp) - 1):
		if y[i] and y[i + 1] == 0:
			xp_sharp[i + 1] = xp_sharp[i] + 1e-8
		elif y[i + 1] and y[i] == 0:
			xp_sharp[i] = xp_sharp[i + 1] - 1e-8

	return xp, xp_sharp


def peakFilter (y):
	size = len(y)
	y0, y1 = torch.zeros(size + 1), torch.zeros(size + 1)
	y0[:size] = y
	y1[1:] = y
	d0 = y0 - y1
	d1 = -d0[1:]

	non_peak = torch.logical_or(d0[:size] < 0, d1 <= 0)
	y[non_peak] = 0


class VocalPitch (IterableDataset):
	@classmethod
	def loadPackage (cls, root, args, splits='*0/1', device='cpu', args_variant=None):
		splits = splits.split(':')

		utterances_file = open(os.path.join(root, 'utterances.yaml'), 'r', encoding='utf-8')
		utterances = yaml.safe_load(utterances_file)

		bias_limit = args.get('bias_limit', 0.3)
		ids = [ut for ut in utterances['utterances'] if utterances['utterances'][ut].get('bias', 1) < bias_limit]

		def loadEntries (split):
			phases, cycle = parseFilterStr(split)

			return [id for i, id in enumerate(ids) if i % cycle in phases]

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return (
			cls(root, loadEntries(split), device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)


	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None):
		return cls.loadPackage(root, args, splits, device, args_variant=args_variant)


	def __init__ (self, root, ids, device, shuffle, seq_align=4, augmentor={}, **_):
		self.root = root
		self.ids = ids
		self.shuffle = shuffle
		self.device = device
		self.seq_align = seq_align
		self.augmentor = augmentor
		self.perlin = Perlin1d()


	def __len__(self):
		return len(self.ids)


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.ids)
		else:
			torch.manual_seed(0)

		for id in self.ids:
			pitchBuffer = open(os.path.join(self.root, 'pitches', f'{id}.pitch'), 'rb').read()
			pitch7Buffer = open(os.path.join(self.root, 'pitch7', f'{id}.pitch7'), 'rb').read()

			n_frame = len(pitch7Buffer)
			assert len(pitchBuffer) == n_frame * 4

			pitchArr = torch.tensor(struct.unpack('H' * (n_frame * 2), pitchBuffer), dtype=torch.long).reshape((-1, 2))
			pitch7Arr = struct.unpack('B' * n_frame, pitch7Buffer)

			#pitch = pitchArr[:, 0] // 64
			pitch = torch.div(pitchArr[:, 0], 64, rounding_mode='floor')
			gain = pitchArr[:, 1].float()

			pitch[pitch > 0] -= PITCH_RANGE[0] * PITCH_SUBDIV
			pitch[pitch < 0] = 0
			pitch[pitch >= (PITCH_RANGE[1] - PITCH_RANGE[0]) * PITCH_SUBDIV] = 0

			gain[gain > 0] -= GAIN_RANGE[0]
			gain /= GAIN_RANGE[1] - GAIN_RANGE[0]

			head = torch.tensor([f >> 7 for f in pitch7Arr], dtype=torch.float32)

			pitch, gain, head = self.augment(pitch, gain, head)

			yield n_frame, pitch, gain, head


	def augment (self, pitch, gain, head):
		if self.augmentor.get('time_distortion'):
			cy_miu, cy_sigma = self.augmentor['time_distortion']['cycle']['miu'], self.augmentor['time_distortion']['cycle']['sigma']
			am_miu, am_sigma = self.augmentor['time_distortion']['amplitude']['miu'], self.augmentor['time_distortion']['amplitude']['sigma']

			cycle = cy_miu * np.exp(np.random.randn() * cy_sigma)
			amplitude = am_miu * np.exp(np.random.randn() * am_sigma)
			#print('time_distortion:', cycle, amplitude)

			distortion = self.perlin.integral(len(pitch), cycle, amplitude=amplitude)
			xp, xp_sharp = sampleXp(pitch, len(distortion), distortion[-1])

			pitch = distort(pitch, distortion, xp_sharp)
			gain = distort(gain, distortion, xp_sharp)
			head = distort(head, distortion, xp)
			peakFilter(head)

		return pitch, gain, head



	def collateBatch (self, batch):
		n_frame_max = max(fields[0] for fields in batch)
		n_frame_max = int(np.ceil(n_frame_max / self.seq_align)) * self.seq_align

		def extract (i):
			tensor = torch.zeros((len(batch), n_frame_max), dtype=batch[0][i].dtype)
			for n, tensors in enumerate(batch):
				tensor[n, :len(tensors[i])] = tensors[i]

			return tensor.to(self.device)

		result = {
			'pitch': extract(1),
			'gain': extract(2),
			'head': extract(3),
		}

		return result
