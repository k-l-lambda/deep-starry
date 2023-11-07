
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
from ...melody.measurewiseMIDI import NOTE_MIN, NOTE_MAX
from ...melody.data.measurewise import normalFactor
from .paragraph import MeasureLibrary



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


	@classmethod
	def loadMeasures (cls, paraff_path, n_seq, encoder_config=None):
		if paraff_path in cls.measure_lib:
			return cls.measure_lib[paraff_path]

		cls.measure_lib[paraff_path] = MeasureLibrary(open(paraff_path, 'rb'), n_seq, encoder_config)

		return cls.measure_lib[paraff_path]


	def __init__ (self, root, split, device, shuffle, n_seq_word, n_seq_midi,
		seq_tail_padding=0, strength_sigma=0.1, strength_pow_sigma=0.6, key_shift_sigma=5,
		premier_drop=0, premier_head_drop=0, premier_drop_sigma=0, consumption_augment=None,
		head_measure_only=False, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.n_seq_word = n_seq_word
		self.n_seq_midi = n_seq_midi
		self.seq_tail_padding = seq_tail_padding
		self.strength_sigma = strength_sigma
		self.strength_pow_sigma = strength_pow_sigma
		self.key_shift_sigma = key_shift_sigma
		self.premier_drop = premier_drop
		self.premier_head_drop = premier_head_drop
		self.premier_drop_sigma = premier_drop_sigma
		self.consumption_augment = consumption_augment
		self.head_measure_only = head_measure_only

		phases, cycle = parseFilterStr(split)

		#print('reading index')
		index = yaml.safe_load(open(root, 'r'))
		measurewise_path = os.path.join(os.path.dirname(root), index['paraff'].replace('.paraff', '-measurewise.zip'))
		#print('opening measurewise')
		measurewise = open_fs(f'zip://{measurewise_path}')

		groups = [group for i, group in enumerate(index['groups']) if i % cycle in phases]
		#print('groups:', len(groups))
		self.paragraphs = [paragraph for paragraph in index['paragraphs'] if paragraph['group'] in groups and measurewise.isfile(f'{paragraph["name"]}.measurewise.json.pkl')]
		self.n_measure = len(self.paragraphs) if self.head_measure_only else sum(paragraph['sentenceRange'][1] - paragraph['sentenceRange'][0] for paragraph in self.paragraphs)

		paraff_path = os.path.join(os.path.dirname(root), index['paraff'])
		self.measure = self.loadMeasures(paraff_path, n_seq_word, None)

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
			mi_begin, mi_end = paragraph['sentenceRange']
			entries = self.measure.entries[mi_begin:mi_end]

			measure_size = (entries != 0).int().sum(dim=-1)

			midi = paragraph['midi']
			for mi in range(len(midi)):
				pid = entries[mi] if self.head_measure_only else entries[:mi + 1].flatten()
				pid = pid[pid != 0]

				body_mask = torch.zeros_like(pid).bool()
				body_mask[-measure_size[mi] + 1:] = True

				tail_padding = 0
				if self.seq_tail_padding > 0:
					tail_padding = np.random.randint(self.seq_tail_padding) + 1
					pid = F.pad(pid, (self.n_seq_word, tail_padding), value=0)
					body_mask = F.pad(body_mask, (self.n_seq_word, tail_padding), value=False)
				else:
					pid = F.pad(pid, (0, self.n_seq_word), value=0)
					body_mask = F.pad(body_mask, (0, self.n_seq_word), value=False)

				if self.head_measure_only:
					pid = pid[:self.n_seq_word + 1]
					body_mask = body_mask[:self.n_seq_word + 1]

				input_id = pid[-self.n_seq_word - 1:-1]
				output_id = pid[-self.n_seq_word:]
				body_mask = body_mask[-self.n_seq_word:]

				if self.premier_drop > 0 or self.premier_head_drop > 0:
					premier_drop_p = self.premier_drop * np.exp(np.random.randn() * self.premier_drop_sigma)
					premier_head_drop_p = self.premier_head_drop * np.exp(np.random.randn())
					premier_mask = torch.logical_not(body_mask)
					premier_head_mask = premier_mask & (input_id < 42)

					input_id = input_id.clone()
					input_id[premier_mask & (torch.rand_like(input_id, dtype=torch.float32) < premier_drop_p)] = 0
					input_id[premier_head_mask & (torch.rand_like(input_id, dtype=torch.float32) < premier_head_drop_p)] = 0

				position = torch.arange(self.n_seq_word, dtype=torch.int16)
				if not self.head_measure_only:
					position += - self.n_seq_word + measure_size[mi] + tail_padding - 1

				n_pre = 0 if self.head_measure_only else 1
				events = midi.slice(mi, mi + 8, pre=n_pre, n_seq=self.n_seq_midi, aug_time_index=np.random.randint(0x1000000))[:self.n_seq_midi]
				n_event = len(events)
				if n_event == 0:
					continue

				if self.head_measure_only and mi > 0 and n_event < self.n_seq_midi // 2:
					break

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

				consumption = torch.ones_like(time)
				consumption[is_positive] *= -1

				t1 = time[is_positive][0]
				if self.consumption_augment:
					consumption += -(time - t1) * self.consumption_augment.get('slope', 0.04e-3)
					consumption += torch.randn_like(consumption) * (((time - t1) / self.consumption_augment.get('sigma', 0.6e-3)).square() * -1).exp()

					consumption = torch.sigmoid(consumption * self.consumption_augment.get('frozen', 4.))

				if self.key_shift_sigma > 0:
					key_shift = int(np.random.randn() * self.key_shift_sigma)
					pitch[is_note] += key_shift
					pitch[is_note] = pitch[is_note]
				pitch[is_note] = pitch[is_note].clip(min=NOTE_MIN, max=NOTE_MAX)

				t0 = time[0].item() + (np.random.randn() * 0.2e+3 if self.shuffle else 0)
				time[:n_event] -= t0

				if self.strength_sigma > 0 or self.strength_pow_sigma > 0:
					strength *= (torch.randn_like(strength) * self.strength_sigma).exp()
					strength = strength.pow(normalFactor(self.strength_pow_sigma))

				yield input_id, output_id, body_mask, position, type_, pitch, strength, time, measure, consumption


	def collateBatch (self, batch):
		def extract (i):
			tensors = [ex[i] for ex in batch]

			return torch.stack(tensors, axis=0).to(self.device)

		input_id, output_id, body_mask, position, type_, pitch, strength, time, measure, consumption = (extract(i) for i in range(10))

		return dict(
			input_id=input_id,
			output_id=output_id,
			body_mask=body_mask,
			position=position,
			type=type_,
			pitch=pitch,
			strength=strength,
			time=time,
			measure=measure,
			consumption=consumption,
		)
