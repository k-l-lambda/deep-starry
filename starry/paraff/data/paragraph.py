
import os
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import yaml
from tqdm import tqdm
import dill as pickle
import numpy as np

from ...utils.parsers import parseFilterStr, mergeArgs
from .paraffFile import ParaffFile
from .timewiseGraph import TG_EOS



PHID_MEASURE = 3


class MeasureLibrary:
	def __init__(self, file, n_seq, encoder_config, semantic_file=None):
		paraff = ParaffFile(file)
		self.tokens = paraff.tokens

		padding_zeros = [0] * (n_seq + 1 - paraff.sentence_align_size)
		sentences = [s + padding_zeros for s in paraff.sentences]
		self.entries = torch.tensor(sentences, dtype=torch.uint8)

		file.close()

		if encoder_config is not None:
			encoder = torch.jit.load(encoder_config['weight']).to(encoder_config['device'])
			encoder.eval()
			batch_size = encoder_config.get('batch_size', 1)

			codes = []
			sigma = torch.zeros(1).to(encoder_config['device'])
			with torch.no_grad():
				for ei in tqdm(range(0, self.entries.shape[0], batch_size), 'Encoding measures'):
					es = self.entries[ei:ei + batch_size].to(encoder_config['device'])
					if encoder_config.get('test'):
						z = torch.randn(batch_size, 256)
					else:
						z = encoder(es, sigma).cpu()
					codes.append(z)

			self.summaries = torch.cat(codes, dim=0)
		else:
			self.summaries = torch.zeros(self.entries.shape[0], 256)

		if semantic_file:
			self.semantic_tensors = pickle.load(semantic_file)
			semantic_file.close()


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
	def loadMeasures (cls, paraff_path, n_seq, encoder_config, with_graph=False):
		if paraff_path in cls.measure_lib:
			return cls.measure_lib[paraff_path]

		semantic_file = None
		if with_graph:
			semantic_path = paraff_path.replace('.paraff', '-semantic.pkl')
			semantic_file = open(semantic_path, 'rb')

		cls.measure_lib[paraff_path] = MeasureLibrary(open(paraff_path, 'rb'), n_seq, encoder_config, semantic_file=semantic_file)

		return cls.measure_lib[paraff_path]


	@staticmethod
	def traverseParagraph (paragraphs, max_len):
		segments = []

		step_size = max_len - max_len // 4

		for pg in paragraphs:
			pg_size = len(pg['phaseTypes'])
			if pg_size <= max_len:
				segments.append(pg)
			else:
				n_desc = len(pg['descriptors'])
				sl, sr = pg['sentenceRange']

				for i in range(0, pg_size, step_size - n_desc):
					si = min(i, pg_size - max_len)
					n_sentence = pg['phaseTypes'][si:si + max_len - n_desc].count(PHID_MEASURE)
					segment = dict(
						name=f'{pg["name"]}_{si}',
						group=pg['group'],
						descriptors=pg['descriptors'],
						phaseTypes=pg['phaseTypes'][si:si + max_len - n_desc],
						phaseNumbers=pg['phaseNumbers'][si:si + max_len - n_desc],
						sentenceRange=[sl, sl + n_sentence],
					)

					segments.append(segment)
					sl += n_sentence
		#print('segments:', '\n'.join([p['name'] for p in segments]))

		return segments


	def __init__ (self, root, split, device, shuffle, n_seq_word, n_seq_phase, encoder=None,
		descriptor_drop=0.1, descriptor_drop_sigma=0., with_summary=False, summary_id=1, with_graph=False,
		seq_tail_padding=0, graph_augmentor=None, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.n_seq_word = n_seq_word
		self.n_seq_phase = n_seq_phase
		self.descriptor_drop = descriptor_drop
		self.descriptor_drop_sigma = descriptor_drop_sigma
		self.with_summary = with_summary
		self.summary_id = summary_id
		self.with_graph = with_graph
		self.seq_tail_padding = seq_tail_padding
		self.graph_augmentor = graph_augmentor
		self.with_summary_encoder = encoder is not None

		phases, cycle = parseFilterStr(split)

		index = yaml.safe_load(open(root, 'r'))
		paraff_path = os.path.join(os.path.dirname(root), index['paraff'])
		groups = [group for i, group in enumerate(index['groups']) if i % cycle in phases]
		paragraphs = [paragraph for paragraph in index['paragraphs'] if paragraph['group'] in groups]
		paragraphs = PhasedParagraph.traverseParagraph(paragraphs, n_seq_phase)	# limit paragraph length less than n_seq_phase

		self.vocab = index['vocab']

		self.measure = self.loadMeasures(paraff_path, n_seq_word, encoder, with_graph=with_graph)

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
		else:
			torch.manual_seed(0)
			np.random.seed(1)

		for i in range(self.paragraphs['id'].shape[0]):
			measure_begin, measure_end = self.paragraphs['range'][i].tolist()

			ph_id = self.paragraphs['id'][i]
			ph_indecies = torch.arange(ph_id.shape[0])
			ph_descriptors = ph_id > PHID_MEASURE
			ph_f_num = self.paragraphs['f_num'][i]
			ph_b_num = self.paragraphs['b_num'][i]
			ph_summary = torch.zeros(self.n_seq_phase, self.d_summary)
			ph_body_mask = torch.zeros_like(ph_id).bool()
			ph_is_measure = ph_id == PHID_MEASURE
			if self.with_summary_encoder:
				ph_summary[ph_is_measure] = self.measure.summaries[measure_begin:measure_end]
			ph_body_idx = ph_indecies[ph_is_measure]
			ph_body_mask[ph_indecies < ph_body_idx[0].item()] = True

			# random drop decriptors
			drop_p_pow = torch.randn(1) * self.descriptor_drop_sigma
			drop_p = torch.pow(self.descriptor_drop, torch.exp(drop_p_pow))
			#descriptors_mask = torch.rand_like(ph_id, dtype=torch.float32) < drop_p
			#ph_body_mask[ph_descriptors & descriptors_mask] = False
			n_descriptors = ph_descriptors.int().sum().item()

			entries = self.measure.entries[measure_begin:measure_end]
			pids = entries.flatten()
			pids = pids[pids != 0]
			pids = F.pad(pids, (1, 0), 'constant', 0)
			pids_arange = torch.arange(pids.shape[0], dtype=torch.int32)
			measure_size = (entries != 0).int().sum(dim=-1)
			measure_size[0] += 1

			for mi in range(measure_begin, measure_end):
				ids_begin = measure_size[:mi - measure_begin].sum().item()
				ids_end = measure_size[:mi - measure_begin + 1].sum().item()

				tail_padding = np.random.randint(self.seq_tail_padding) + 1 if self.seq_tail_padding > 0 else 0
				ids_front = max(0, ids_end - (self.n_seq_word - tail_padding))
				ids = pids[ids_front:ids_end]

				if self.with_summary:
					ids = ids.clone()
					ids[0] = self.summary_id

				input_ids = torch.zeros(self.n_seq_word, dtype=torch.uint8)
				output_ids = torch.zeros(self.n_seq_word, dtype=torch.uint8)
				body_mask = torch.zeros(self.n_seq_word, dtype=torch.bool)
				position = torch.zeros(self.n_seq_word, dtype=torch.int16)

				input_ids[:ids.shape[0] - 1] = ids[:-1]
				output_ids[:ids.shape[0] - 1] = ids[1:]

				body_mask[:ids.shape[0] - 1] = pids_arange[ids_front:ids_end - 1] >= ids_begin

				position[:ids.shape[0] - 1] = pids_arange[ids_front:ids_end - 1] - ids_begin

				if self.with_summary:
					position[0] = 0
					#body_mask[0] = True

				ph_next_mask = torch.zeros_like(ph_id).bool()
				ph_next_mask[ph_body_idx[mi - measure_begin].item()] = True

				# random drop decriptors
				ph_body_mask_m = ph_body_mask.clone()
				ph_body_mask_m[ph_descriptors] = torch.rand(n_descriptors, dtype=torch.float32) >= drop_p

				basic_fields = (
					ph_id, ph_f_num, ph_b_num, ph_summary.clone(), ph_body_mask_m, ph_next_mask,
					input_ids, output_ids, body_mask, position,
				)

				if not self.with_graph:
					yield basic_fields
				else:
					tg_fields = [self.measure.semantic_tensors[k][mi].clone() for k in ['semantic', 'staff', 'x', 'y', 'sy1', 'sy2', 'confidence']]
					tg_fields = self.augmentGraphFields(*tg_fields)
					yield (*basic_fields, *tg_fields)

				ph_body_mask[ph_body_idx[mi - measure_begin].item()] = True


	def collateBatch (self, batch):
		def extract (i):
			tensors = [ex[i] for ex in batch]
			return torch.stack(tensors, axis=0).to(self.device)

		ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask, ph_next_mask, input_ids, output_ids, body_mask, position = [extract(i) for i in range(10)]

		basic_dict = dict(
			ph_id=ph_id,
			ph_f_num=ph_f_num,
			ph_b_num=ph_b_num,
			ph_summary=ph_summary,
			ph_body_mask=ph_body_mask,
			ph_next_mask=ph_next_mask,
			input_ids=input_ids,
			output_ids=output_ids,
			body_mask=body_mask,
			position=position,
		)

		if not self.with_graph:
			return basic_dict
		else:
			tg_id, tg_staff, tg_x, tg_y, tg_sy1, tg_sy2, tg_confidence = [extract(i) for i in range(10, 17)]

			return {
				**basic_dict,
				**dict(
					tg_id=tg_id,
					tg_staff=tg_staff,
					tg_x=tg_x,
					tg_y=tg_y,
					tg_sy1=tg_sy1,
					tg_sy2=tg_sy2,
					tg_confidence=tg_confidence,
				),
			}


	def augmentGraphFields (self, *fields):
		if self.graph_augmentor is None:
			return fields

		id, staff, x, y, sy1, sy2, confidence = fields

		drop_base = self.graph_augmentor.get('drop_p', 0)
		drop_sigma = self.graph_augmentor.get('drop_sigma', 0)
		drop_p = drop_base ** np.exp(np.random.randn() * drop_sigma)

		drop_mask = (torch.rand_like(id, dtype=torch.float) < drop_p) & (id > TG_EOS)
		id[drop_mask] = 0

		x_factor_sigma = self.graph_augmentor.get('x_factor_sigma', 0)
		x_factor = np.exp(np.random.randn() * x_factor_sigma)
		x *= x_factor

		drift_sigma = self.graph_augmentor.get('drift_sigma', 0)
		x += torch.randn_like(x) * drift_sigma

		y_drift = torch.randn_like(y) * drift_sigma
		y += y_drift
		sy1 += y_drift
		sy2 += y_drift

		# initialize confidence for null value
		confidence = confidence.float()
		null_confidence = confidence < 0
		confidence[null_confidence] = torch.randn_like(confidence[null_confidence]) * 5.

		confidence_sigma = self.graph_augmentor.get('confidence_sigma', 0)
		confidence *= (torch.randn_like(confidence) * confidence_sigma).exp()

		return id, staff, x, y, sy1, sy2, confidence
