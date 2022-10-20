
import os
import yaml
import dill as pickle
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


def backEdgeFilter (y):
	y1 = torch.zeros_like(y)
	y1[:-1] = y[1:]
	edge = torch.logical_and(y > 0, y1 == 0)
	y[torch.logical_not(edge)] = 0


def frontEdgeFilter (y):
	y1 = torch.zeros_like(y)
	y1[1:] = y[:-1]
	edge = torch.logical_and(y > 0, y1 == 0)
	y[torch.logical_not(edge)] = 0


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


	def __init__ (self, root, ids, device, shuffle, seq_align=4, with_tonf=False, with_nonf=False, augmentor={}, **_):
		self.root = root
		self.ids = ids
		self.shuffle = shuffle
		self.device = device
		self.seq_align = seq_align
		self.augmentor = augmentor
		self.perlin = Perlin1d()
		self.with_tonf = with_tonf
		self.with_nonf = with_nonf

		# load midi compilation
		with open(os.path.join(self.root, 'midi-compilation.pickle'), 'rb') as file:
			self.midi = pickle.load(file)


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

			midi = self.midi[id]
			midi_pitch, midi_tick, midi_rtick = midi['pitches'], midi['ticks'], midi['rticks']
			tonf = midi.get('tonf') if self.with_tonf else None
			nonf = midi.get('nonf') if self.with_nonf else None

			midi_pitch = torch.clip(midi_pitch, min=PITCH_RANGE[0], max=PITCH_RANGE[1])
			midi_pitch -= PITCH_RANGE[0]

			pitch, gain, head, tonf, nonf, midi_tick, midi_rtick = self.augment(pitch, gain, head, tonf, nonf, midi_tick, midi_rtick)

			yield n_frame, pitch, gain, head, tonf, nonf, midi_pitch, midi_tick, midi_rtick


	def augment (self, pitch, gain, head, tonf, nonf, midi_tick, midi_rtick):
		silence = pitch == 0

		if self.augmentor.get('time_distortion'):
			cy_miu, cy_sigma = self.augmentor['time_distortion']['cycle']['miu'], self.augmentor['time_distortion']['cycle']['sigma']
			am_miu, am_sigma = self.augmentor['time_distortion']['amplitude']['miu'], self.augmentor['time_distortion']['amplitude']['sigma']

			cycle = cy_miu * np.exp(np.random.randn() * cy_sigma)
			amplitude = am_miu * np.exp(np.random.randn() * am_sigma)
			#print('time_distortion:', cycle, amplitude)

			distortion = self.perlin.integralExp(len(pitch), cycle, amplitude=amplitude)
			xp, xp_sharp = sampleXp(pitch, len(distortion), distortion[-1])

			pitch = distort(pitch, distortion, xp_sharp).long()
			gain = distort(gain, distortion, xp_sharp).float()

			silence = pitch == 0

			head = distort(head, distortion, xp).float()
			head[silence] = 0
			#peakFilter(head)
			frontEdgeFilter(head)
			head[head > 0] = 1

			if self.with_tonf:
				tonf = distort(tonf, distortion, xp).float()

		if self.augmentor.get('offkey'):
			cy_miu, cy_sigma = self.augmentor['offkey']['cycle']['miu'], self.augmentor['offkey']['cycle']['sigma']
			am_miu, am_sigma = self.augmentor['offkey']['amplitude']['miu'], self.augmentor['offkey']['amplitude']['sigma']
			cycle = cy_miu * np.exp(np.random.randn() * cy_sigma)
			cycle = max(len(pitch) / 32 + 1, cycle)
			amplitude = am_miu * np.exp(np.random.randn() * am_sigma)

			bias = amplitude * np.random.randn()

			offkey = bias + torch.from_numpy(self.perlin.integralLinear(len(pitch), cycle)) * amplitude * 4 / cycle
			offkey[silence] = 0
			offkey = torch.round(offkey).long()

			#print('offkey:', cycle, amplitude, bias, (offkey.max().item(), offkey.min().item()))

			pitch += offkey
			pitch[pitch < 0] = 0
			pitch[pitch >= (PITCH_RANGE[1] - PITCH_RANGE[0]) * PITCH_SUBDIV] = 0

		if self.augmentor.get('gain'):
			miu, sigma = self.augmentor['gain']['scaling']['miu'], self.augmentor['gain']['scaling']['sigma']
			scaling = miu * np.exp(np.random.randn() * sigma)
			#print('gain scaling:', scaling)

			gain *= scaling

		if self.augmentor.get('tonf'):
			scale = np.random.choice(self.augmentor['tonf']['scales'])
			tonf *= scale
			midi_tick *= scale
			midi_rtick *= scale

		if self.augmentor.get('nonf'):
			if self.augmentor['nonf'].get('max') is not None:
				nonf = nonf.clip(max=self.augmentor['nonf']['max'])
				midi_rtick = midi_rtick.clip(max=self.augmentor['nonf']['max'])

		return pitch, gain, head, tonf, nonf, midi_tick, midi_rtick


	def collateBatch (self, batch):
		n_frame_max = max(fields[0] for fields in batch)
		n_frame_max = int(np.ceil(n_frame_max / self.seq_align)) * self.seq_align

		n_note_max = max(len(fields[6]) for fields in batch)

		def extract (i, n_seq=n_frame_max):
			tensor = torch.zeros((len(batch), n_seq), dtype=batch[0][i].dtype)
			for n, tensors in enumerate(batch):
				tensor[n, :len(tensors[i])] = tensors[i]

			return tensor.to(self.device)

		mask = torch.zeros((len(batch), n_frame_max), dtype=torch.float)
		for n, tensors in enumerate(batch):
			mask[n, :tensors[0]] = 1
		mask = mask.bool().to(self.device)

		result = {
			'pitch': extract(1),
			'gain': extract(2),
			'head': extract(3),
			'tonf': extract(4) if self.with_tonf else None,
			'nonf': extract(5) if self.with_nonf else None,
			'midi_pitch': extract(6, n_seq=n_note_max),
			'midi_tick': extract(7, n_seq=n_note_max),
			'midi_rtick': extract(8, n_seq=n_note_max),
			'mask': mask,
		}

		return result
