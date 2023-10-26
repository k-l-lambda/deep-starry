
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


	def __init__ (self, root, split, device, shuffle, n_seq_word,
		descriptor_drop=0.1, descriptor_drop_sigma=0., seq_tail_padding=0, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.n_seq_word = n_seq_word
		self.descriptor_drop = descriptor_drop
		self.descriptor_drop_sigma = descriptor_drop_sigma
		self.seq_tail_padding = seq_tail_padding

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
			pass


	def collateBatch (self, batch):
		pass
