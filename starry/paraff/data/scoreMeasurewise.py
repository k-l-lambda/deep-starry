
import os
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import json
import yaml
from tqdm import tqdm
import dill as pickle
import numpy as np
from fs import open_fs

from ...utils.parsers import parseFilterStr, mergeArgs
from .paraffFile import ParaffFile
from ...melody.measurewiseMIDI import NOTE_MIN, NOTE_MAX
from ...melody.data.measurewise import normalFactor



class ScoreMeasurewise (IterableDataset):
	measure_lib = {}


	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		splits = splits.split(':')

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return (
			cls(root, split, device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)


	def __init__ (self, root, split, device, shuffle, n_seq_word, n_seq_midi,
		descriptor_drop=0.1, descriptor_drop_sigma=0., seq_tail_padding=0,
		strength_sigma=0.1, strength_pow_sigma=0.6, key_shift_sigma=5, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.n_seq_word = n_seq_word
		self.n_seq_midi = n_seq_midi
		self.descriptor_drop = descriptor_drop
		self.descriptor_drop_sigma = descriptor_drop_sigma
		self.seq_tail_padding = seq_tail_padding
		self.strength_sigma = strength_sigma
		self.strength_pow_sigma = strength_pow_sigma
		self.key_shift_sigma = key_shift_sigma

		phases, cycle = parseFilterStr(split)

		#print('reading index')
		index = yaml.safe_load(open(root, 'r'))
		paraff_path = os.path.join(os.path.dirname(root), index['paraff'])
		measurewise_path = os.path.join(os.path.dirname(root), index['paraff'].replace('.paraff', '-measurewise.zip'))
		#print('opening measurewise')
		measurewise = open_fs(f'zip://{measurewise_path}')

		groups = [group for i, group in enumerate(index['groups']) if i % cycle in phases]
		#print('groups:', len(groups))
		self.paragraphs = [paragraph for paragraph in index['paragraphs'] if paragraph['group'] in groups and measurewise.isfile(f'{paragraph["name"]}.measurewise.json.pkl')]
		self.n_measure = sum(paragraph['sentenceRange'][1] - paragraph['sentenceRange'][0] for paragraph in self.paragraphs)

		# load measurewise
		for paragraph in tqdm(self.paragraphs, desc="Loading measurewise"):
			paragraph['midi'] = pickle.load(measurewise.openbin(f'{paragraph["name"]}.measurewise.json.pkl'))
			assert paragraph['sentenceRange'][1] - paragraph['sentenceRange'][0] == len(paragraph['midi']), f'{paragraph["name"]} measure number mismatched: {paragraph["sentenceRange"][1] - paragraph["sentenceRange"][0]} vs {len(paragraph["midi"])}'


	def __len__ (self):
		return self.n_measure


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.paragraphs)
		else:
			torch.manual_seed(0)
			np.random.seed(1)

		for paragraph in self.paragraphs:
			midi = paragraph['midi']
			for mi in range(len(midi)):
				events = midi.slice(mi, mi + 8, pre=1, n_seq=self.n_seq_midi, aug_time_index=np.random.randint(0x1000000))[:self.n_seq_midi]
				n_event = len(events)
				if n_event == 0:
					continue

				padding = [0] * (self.n_seq_midi - len(events))

				type_ = torch.tensor([e['type'] for e in events] + padding, dtype=torch.uint8)
				pitch = torch.tensor([e['pitch'] for e in events] + padding, dtype=torch.uint8)
				strength = torch.tensor([e['strength'] for e in events] + padding, dtype=torch.float32)
				time = torch.tensor([e['time'] for e in events] + padding, dtype=torch.float32)
				measure = torch.tensor([e['measure'] for e in events] + padding, dtype=torch.int8)

				order = time[:n_event].argsort()
				type_[:n_event] = type_[:n_event][order]
				pitch[:n_event] = pitch[:n_event][order]
				strength[:n_event] = strength[:n_event][order]
				time[:n_event] = time[:n_event][order]

				is_note = (pitch >= NOTE_MIN) & (pitch <= NOTE_MAX)
				is_positive = measure >= 0

				if self.key_shift_sigma > 0:
					key_shift = int(np.random.randn() * self.key_shift_sigma)
					pitch[is_note] += key_shift
					pitch[is_note] = pitch[is_note].clip(min=NOTE_MIN, max=NOTE_MAX)

				t0 = time[is_positive][0] - (normalFactor() * 0.4e+3 if self.shuffle else 0)
				time[:n_event] -= t0

				if self.strength_sigma > 0 or self.strength_pow_sigma > 0:
					strength *= (torch.randn_like(strength) * self.strength_sigma).exp()
					strength = strength.pow(normalFactor(self.strength_pow_sigma))

				yield type_, pitch, strength, time, measure


	def collateBatch (self, batch):
		def extract (i):
			tensors = [ex[i] for ex in batch]

			return torch.stack(tensors, axis=0).to(self.device)

		type_, pitch, strength, time, measure = extract(0), extract(1), extract(2), extract(3), extract(4)

		return dict(
			type=type_,
			pitch=pitch,
			strength=strength,
			time=time,
			measure=measure,
		)
