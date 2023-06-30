
import os
import torch
from torch.utils.data import IterableDataset
import yaml
from tqdm import tqdm

from ...utils.parsers import parseFilterStr, mergeArgs
from .paraffFile import ParaffFile



PHID_MEASURE = 3


class MeasureLibrary:
	def __init__(self, file, n_seq, encoder_config):
		paraff = ParaffFile(file)
		self.tokens = paraff.tokens

		padding_zeros = [0] * (n_seq + 1 - paraff.sentence_align_size)
		sentences = [s + padding_zeros for s in paraff.sentences]
		self.entries = torch.tensor(sentences, dtype=torch.uint8)

		encoder = torch.jit.load(encoder_config['weight']).to(encoder_config['device'])
		encoder.eval()
		batch_size = encoder_config.get('batch_size', 1)

		codes = []
		sigma = torch.zeros(1).to(encoder_config['device'])
		with torch.no_grad():
			for ei in tqdm(range(0, self.entries.shape[0], batch_size), 'Encoding measures'):
				es = self.entries[ei:ei + batch_size].to(encoder_config['device'])
				z = encoder(es, sigma)
				codes.append(z)

		self.summaries = torch.concatenate(codes, dim=0)


class PhasedParagraph (IterableDataset):
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
	def loadMeasures (cls, paraff_path, n_seq, encoder_config):
		if paraff_path in cls.measure_lib:
			return cls.measure_lib[paraff_path]

		cls.measure_lib[paraff_path] = MeasureLibrary(open(paraff_path, 'rb'), n_seq, encoder_config)

		return cls.measure_lib[paraff_path]


	def __init__ (self, root, split, device, shuffle, n_seq_word, n_seq_phase, encoder,
		descriptor_drop=0.1, descriptor_drop_sigma=0., **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.n_seq_word = n_seq_word
		self.n_seq_phase = n_seq_phase
		self.descriptor_drop = descriptor_drop
		self.descriptor_drop_sigma = descriptor_drop_sigma

		phases, cycle = parseFilterStr(split)

		index = yaml.safe_load(open(root, 'r'))
		paraff_path = os.path.join(os.path.dirname(root), index['paraff'])
		groups = [group for i, group in enumerate(index['groups']) if i % cycle in phases]
		paragraphs = [paragraph for paragraph in index['paragraphs'] if paragraph['group'] in groups]

		self.measure = self.loadMeasures(paraff_path, n_seq_word, encoder)

		self.d_summary = self.measure.summaries.shape[-1]

		ph_id = torch.zeros(len(paragraphs), n_seq_phase, dtype=torch.uint8)
		#ph_body_mask = torch.zeros(len(paragraphs), n_seq_phase, dtype=torch.bool)
		ph_f_num = torch.zeros(len(paragraphs), n_seq_phase, dtype=torch.int16)
		ph_b_num = torch.zeros(len(paragraphs), n_seq_phase, dtype=torch.int16)
		ph_ranges = []

		for i, paragraph in enumerate(tqdm(paragraphs, 'Load paragraphs')):
			descriptors, types, numbers, range = paragraph['descriptors'], paragraph['phaseTypes'], paragraph['phaseNumbers'], paragraph['sentenceRange']
			n_desc = len(descriptors)
			types = types[:n_seq_phase - n_desc]
			numbers = numbers[:n_seq_phase - n_desc]

			assert not (None in descriptors), 'invalid descriptors: %s, %s' % (paragraph['name'], descriptors)

			ph_id[i, :n_desc] = torch.tensor(descriptors, dtype=ph_id.dtype)
			ph_id[i, n_desc:n_desc + len(types)] = torch.tensor(types, dtype=ph_id.dtype)
			#ph_body_mask[i] = ph_id[i] == PHID_BOS
			ph_f_num[i, n_desc:n_desc + len(types)] = torch.tensor([fb[0] for fb in numbers], dtype=ph_f_num.dtype)
			ph_b_num[i, n_desc:n_desc + len(types)] = torch.tensor([fb[1] for fb in numbers], dtype=ph_b_num.dtype)

			#ph_summary[i, ph_id[i] == PHID_BOS] = self.measure.summaries[range[0]:range[1]]
			ph_ranges.append(range)

		self.paragraphs = dict(id=ph_id, f_num=ph_f_num, b_num=ph_b_num, range=torch.tensor(ph_ranges, dtype=torch.int32))
		self.n_measure = sum(paragraph['sentenceRange'][1] - paragraph['sentenceRange'][0] for paragraph in paragraphs)


	def __len__ (self):
		return self.n_measure


	def __iter__ (self):
		if self.shuffle:
			disorder = torch.randperm(self.paragraphs['id'].shape[0])

			for key in self.paragraphs:
				self.paragraphs[key] = self.paragraphs[key][disorder]

			for i in range(self.paragraphs['id'].shape[0]):
				measure_begin, measure_end = self.paragraphs['range'][i].tolist()

				ph_id = self.paragraphs['id'][i]
				ph_f_num = self.paragraphs['f_num'][i]
				ph_b_num = self.paragraphs['b_num'][i]
				ph_summary = torch.zeros(self.n_seq_phase, self.d_summary)
				ph_body_mask = torch.zeros_like(ph_id).bool()
				ph_summary[ph_body_mask] = self.measure.summaries[measure_begin:measure_end]

				ph_mask_idx = torch.arange(ph_id.shape[0])[ph_id == PHID_MEASURE]

				entris = self.measure.entries[measure_begin:measure_end]
				pids = entris.flatten()
				pids = pids[pids != 0]
				pids_arange = torch.arange(pids.shape[0], dtype=torch.int16)
				measure_size = (entris != 0).int().sum(dim=-1)

				for mi in range(measure_begin, measure_end):
					ids_begin = measure_size[:mi - measure_begin].sum()
					ids_end = measure_size[:mi - measure_begin + 1].sum()
					ids = pids[max(0, ids_end - self.n_seq_word):ids_end]

					input_ids = torch.zeros(self.n_seq_word, dtype=torch.uint8)
					output_ids = torch.zeros(self.n_seq_word, dtype=torch.uint8)
					body_mask = torch.zeros(self.n_seq_word, dtype=torch.bool)
					position = torch.zeros(self.n_seq_word, dtype=torch.int16)

					input_ids[:ids.shape[0] - 1] = ids[:-1]
					output_ids[:ids.shape[0] - 1] = ids[1:]

					body_mask[:ids.shape[0] - 1] = pids_arange[max(0, ids_end - self.n_seq_word):ids_end - 1] >= ids_begin

					position[:ids.shape[0] - 1] = pids_arange[max(0, ids_end - self.n_seq_word):ids_end - 1] - ids_begin

					yield ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask, input_ids, output_ids, body_mask, position

					ph_body_mask[ph_mask_idx[mi - measure_begin].item()] = True


	def collateBatch (self, batch):
		def extract (i):
			tensors = [ex[i] for ex in batch]
			return torch.stack(tensors, axis=0).to(self.device)

		ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask, input_ids, output_ids, body_mask, position = [extract(i) for i in range(9)]

		return dict(
			ph_id=ph_id,
			ph_f_num=ph_f_num,
			ph_b_num=ph_b_num,
			ph_summary=ph_summary,
			ph_body_mask=ph_body_mask,
			input_ids=input_ids,
			output_ids=output_ids,
			body_mask=body_mask,
			position=position,
		)
