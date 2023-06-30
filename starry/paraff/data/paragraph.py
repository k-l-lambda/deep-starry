
import os
import torch
from torch.utils.data import IterableDataset
import yaml
from tqdm import tqdm

from ...utils.parsers import parseFilterStr, mergeArgs
from .paraffFile import ParaffFile



PHID_BOS = 3


class PhasedParagraph (IterableDataset):
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


	def __init__ (self, root, split, device, shuffle, n_seq_word, n_seq_phase,
		descriptor_drop=0.1, descriptor_drop_sigma=0., **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.n_seq_word = n_seq_word
		self.n_seq_phase = n_seq_phase
		self.descriptor_drop = descriptor_drop
		self.descriptor_drop_sigma = descriptor_drop_sigma

		phases, cycle = parseFilterStr(split)

		index = yaml.safe_load(open(root, 'rb'))
		paraff_path = os.path.join(os.path.dirname(root), index['paraff'])
		groups = [group for i, group in enumerate(index['groups']) if i % cycle in phases]
		paragraphs = [paragraph for paragraph in index['paragraphs'] if paragraph['group'] in groups]

		ph_ids = torch.zeros(len(paragraphs), n_seq_phase, dtype=torch.uint8)
		#ph_body_mask = torch.zeros(len(paragraphs), n_seq_phase, dtype=torch.bool)
		ph_f_num = torch.zeros(len(paragraphs), n_seq_phase, dtype=torch.int16)
		ph_b_num = torch.zeros(len(paragraphs), n_seq_phase, dtype=torch.int16)
		# TODO: ph_summary

		for i, paragraph in enumerate(tqdm(paragraphs, 'load paragraphs')):
			descriptors, types, numbers = paragraph['descriptors'], paragraph['phaseTypes'], paragraph['phaseNumbers']
			n_desc = len(descriptors)
			types = types[:n_seq_phase - n_desc]
			numbers = numbers[:n_seq_phase - n_desc]

			assert not (None in descriptors), 'invalid descriptors: %s, %s' % (paragraph['name'], descriptors)

			ph_ids[i, :n_desc] = torch.tensor(descriptors, dtype=ph_ids.dtype)
			ph_ids[i, n_desc:n_desc + len(types)] = torch.tensor(types, dtype=ph_ids.dtype)
			#ph_body_mask[i] = ph_ids[i] == PHID_BOS
			ph_f_num[i, n_desc:n_desc + len(types)] = torch.tensor([fb[0] for fb in numbers], dtype=ph_f_num.dtype)
			ph_b_num[i, n_desc:n_desc + len(types)] = torch.tensor([fb[1] for fb in numbers], dtype=ph_b_num.dtype)

		self.paragraphs = dict(ids=ph_ids, f_num=ph_f_num, b_num=ph_b_num)
		self.n_measure = sum(paragraph['sentenceRange'][1] - paragraph['sentenceRange'][0] for paragraph in paragraphs)

		'''paraff = ParaffFile(paraff_path)

		padding_zeros = [0] * (n_seq_word + 1 - paraff.sentence_align_size)
		sentences = [s + padding_zeros for i, s in paraff.sentences]
		self.entries = torch.tensor(sentences, dtype=torch.uint8)

		self.id_BOM = paraff.tokens.index('BOM')
		self.id_EOM = paraff.tokens.index('EOM')'''


	def __len__ (self):
		return self.n_measure


	def collateBatch (self, batch):
		pass
