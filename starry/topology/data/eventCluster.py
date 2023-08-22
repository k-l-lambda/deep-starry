
import yaml
from fs import open_fs
import numpy as np
import dill as pickle
import torch
from torch.utils.data import IterableDataset
import torch.nn.functional as F

from ...utils.parsers import parseFilterStr, mergeArgs
from ..event_element import TARGET_FIELDS, EventElementType



class EventCluster (IterableDataset):
	@classmethod
	def loadPackage (cls, url, args, splits='*0/1', device='cpu', args_variant=None):
		splits = splits.split(':')
		package = open_fs(url)
		index_file = package.open('index.yaml', 'r')
		index = yaml.safe_load(index_file)

		def loadEntries (split):
			phases, cycle = parseFilterStr(split)

			ids = [id for i, id in enumerate(index['groups']) if i % cycle in phases]

			return [entry for entry in index['examples'] if entry['group'] in ids]

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


	def __init__ (self, package, entries, device, shuffle=False, stability_base=10, position_drift=0, stem_amplitude=None,
		chaos_exp=-1, chaos_flip=False, batch_slice=None, use_cache=True, with_beading=False, time8th_drop=0, event_drop=0):
		self.package = package
		self.entries = entries
		self.shuffle = shuffle
		self.device = device

		self.stability_base = stability_base
		self.chaos_exp = chaos_exp
		self.chaos_flip = chaos_flip
		self.position_drift = position_drift
		self.stem_amplitude = stem_amplitude
		self.batch_slice = batch_slice
		self.with_beading = with_beading
		self.time8th_drop = time8th_drop
		self.event_drop = event_drop

		self.entry_cache = {} if use_cache else None


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
		return len(self.entries)


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.entries)
		else:
			torch.manual_seed(0)
			np.random.seed(len(self.entries))

		for entry in self.entries:
			yield self.readEntry(entry['filename'])


	def collateBatch (self, batch):
		assert len(batch) == 1

		tensors = batch[0]
		if tensors.get('fake') is None:
			tensors['fake'] = 1 - tensors['confidence']

		feature_shape = tensors['feature'].shape
		batch_size = feature_shape[0]
		if self.batch_slice is not None:
			batch_size = min(batch_size, self.batch_slice)
		n_seq = feature_shape[1]

		# to device
		for key in tensors:
			tensors[key] = tensors[key].to(self.device)

		# extend batch dim for single tensors
		elem_type = tensors['type'].repeat(batch_size, 1).long()
		staff = tensors['staff'].repeat(batch_size, 1).long()

		# noise augment for feature
		feature = tensors['feature'][:batch_size]
		stability = np.random.power(self.stability_base)
		error = torch.rand(*feature.shape, device=self.device) > stability
		if self.chaos_flip:
			feature_err_tanh = torch.tanh(feature[error] / 0.4).clip(min=0.0000001, max=0.9999999)
			feature[error] = torch.atanh(1 - feature_err_tanh)
		else:
			chaos = torch.exp(torch.randn(*feature.shape, device=self.device) + self.chaos_exp)
			feature[error] *= chaos[error]

		# sort division[3:] & dots
		feature[:, :, 3:7], _ = feature[:, :, 3:7].sort(descending=True)
		feature[:, :, 7:9], _ = feature[:, :, 7:9].sort(descending=True)

		# stemDirection amplitude
		if self.stem_amplitude:
			power = torch.randn((batch_size, 1, 1), device=self.device) * self.stem_amplitude['sigma'] + self.stem_amplitude['mu']
			feature[:, :, 12:14] *= torch.exp(power)

		# augment for position
		x = tensors['x'][:batch_size]
		pivotX = tensors['pivotX'][:batch_size]
		y1 = tensors['y1'][:batch_size]
		y2 = tensors['y2'][:batch_size]
		ox, oy = (torch.rand(batch_size, 1, device=self.device) - 0.2) * 24, (torch.rand(batch_size, 1, device=self.device) - 0.2) * 12
		if self.position_drift > 0:
			dx = torch.randn(batch_size, n_seq, device=self.device) * self.position_drift + ox
			x += dx
			pivotX += dx

			# exclude BOS, EOS from global Y offset
			y1[:, 1:-1] += torch.randn(batch_size, n_seq - 2, device=self.device) * self.position_drift + oy
			y2[:, 1:-1] += torch.randn(batch_size, n_seq - 2, device=self.device) * self.position_drift + oy

		if self.with_beading:
			order_max = tensors['order'].max().item()
			beading_tip = torch.multinomial(torch.ones(order_max), batch_size, replacement=batch_size > order_max).to(self.device)	# (batch_size)
			beading_pos = tensors['order'][None, :].repeat(batch_size, 1) - (beading_tip[:, None] + 1)								# (batch_size, n_seq)
			beading_pos[beading_pos > 0] = 0

			# move BOS pos to ahead of last voice
			voice_head_pos = torch.tensor([i for i in range(order_max) if i == 0 or (not i in tensors['order'])], dtype=beading_pos.dtype, device=self.device)
			vhs = voice_head_pos[None, :].repeat(batch_size, 1) - (beading_tip[:, None] + 1)
			#print('vhs:', vhs)
			for i, vh in enumerate(vhs):
				vh[vh >= 0] = beading_pos[i, 0]
			beading_pos[:, 0] = vhs.max(dim=-1).values
			beading_pos = beading_pos.to(self.device)

			matrixH = tensors['matrixH'].reshape(n_seq - 1, n_seq - 1)

			successor = torch.zeros(batch_size, n_seq).bool()
			order_list = tensors['order'].tolist()
			for i, tip in enumerate(beading_tip.tolist()):
				ti = order_list.index(tip) if tip in order_list else 0
				successor[i, 1:] = (matrixH[:, ti] > 0) & (beading_pos[i, 1:] == 0)

			# regularize duration feature fields
			fixed_indices = (beading_pos < 0) & (elem_type > 2)
			fixed_feature = feature[fixed_indices]
			fixed_division = tensors['division'].repeat(batch_size, 1)[fixed_indices].long()
			fixed_dots = tensors['dots'].repeat(batch_size, 1)[fixed_indices]

			fixed_feature[:, :7] = F.one_hot(fixed_division, num_classes=9)[:, :7]
			fixed_feature[:, 7] = (fixed_dots > 0).float()
			fixed_feature[:, 8] = (fixed_dots > 1).float()

			feature[fixed_indices] = fixed_feature

			# dropout events
			if self.event_drop > 0:
				is_event = (elem_type == EventElementType.CHORD) | (elem_type == EventElementType.REST)
				event_rollout = torch.rand_like(elem_type, dtype=torch.float32, device=elem_type.device) < self.event_drop
				event_dropout = is_event & event_rollout & torch.logical_not(successor) #& torch.logical_not(beading_pos < 0)
				elem_type[event_dropout] = EventElementType.PAD

			result = {
				'type': elem_type,
				'staff': staff,
				'feature': feature,
				'x': x,
				'pivotX': pivotX,
				'y1': y1,
				'y2': y2,
				'tickDiff': tensors['tickDiff'].unsqueeze(0).repeat(batch_size, 1, 1),
				'maskT': tensors['maskT'].unsqueeze(0).repeat(batch_size, 1, 1),
				'beading_pos': beading_pos,
				'successor': successor.float().to(self.device),
			}
		else:
			result = {
				'type': elem_type,
				'staff': staff,
				'feature': feature,
				'x': x,
				'pivotX': pivotX,
				'y1': y1,
				'y2': y2,
				'matrixH': tensors['matrixH'].repeat(batch_size, 1),
				'tickDiff': tensors['tickDiff'].unsqueeze(0).repeat(batch_size, 1, 1),
				'maskT': tensors['maskT'].unsqueeze(0).repeat(batch_size, 1, 1),
			}

		result['time8th'] = tensors['time8th'].repeat(batch_size)
		result['time8th'][torch.rand(result['time8th'].shape) < self.time8th_drop] = 0
		result['time8th'] = result['time8th'].to(self.device)

		for field in TARGET_FIELDS:
			result[field] = tensors[field].repeat(batch_size, 1)

		for field in ['division', 'dots', 'beam', 'stemDirection']:
			result[field] = result[field].long()

		return result
