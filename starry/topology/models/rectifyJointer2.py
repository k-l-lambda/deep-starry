
import torch
import torch.nn as nn

from ..event_element import TARGET_DIM, EventElementType
from ...utils.weightedValue import WeightedValue
from .modules import EventArgsEncoder, SieveJointer2, RectifierParser2, JaggedLoss, CrossEntropy
from .rectifyJointer import EncoderLayerStack, Encoder, Decoder



class RectifySieveJointer2 (nn.Module):
	def __init__ (self, n_trunk_layers=1, n_rectifier_layers=1, n_source_layers=2, n_target_layers=1, n_sieve_layers=1,
			d_model=512, d_inner=2048, angle_cycle=1000, feature_activation=None, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		self.event_encoder = EventArgsEncoder(d_model, angle_cycle=angle_cycle, feature_activation=feature_activation)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.trunk_encoder = EncoderLayerStack(n_trunk_layers, **encoder_args)
		self.rectifier_encoder = Encoder(n_rectifier_layers, **encoder_args, scale_emb=scale_emb)
		self.target_encoder = Encoder(n_target_layers, **encoder_args, scale_emb=scale_emb)
		self.sieve_encoder = Encoder(n_sieve_layers, **encoder_args, scale_emb=scale_emb)
		self.source_encoder = Decoder(n_source_layers, **encoder_args, scale_emb=scale_emb)

		self.rec_out = nn.Linear(d_model, TARGET_DIM)
		self.rec_parser = RectifierParser2()

		self.jointer = SieveJointer2(d_model)


	def forward (self, stype, staff, feature, x, y1, y2):	# dict(name -> T(n, seq, xtar)), list(n, T((n - 1) * (n - 1)))
		x = self.event_encoder(stype, staff, feature, x, y1, y2)	# (n, seq, d_model)

		mask_pad = stype != 5	# EventElementType.PAD
		mask = mask_pad.unsqueeze(-2)

		x = self.trunk_encoder(x, mask)

		rec = self.rectifier_encoder(x, mask)
		rec = self.rec_out(rec)
		rec = self.rec_parser(rec)

		target = self.target_encoder(x, mask)
		sieve = self.sieve_encoder(x, mask)
		source = self.source_encoder(x, target, mask)

		mask_src = mask_pad & (stype != 1)	# EventElementType.BOS
		mask_tar = mask_pad & (stype != 2)	# EventElementType.EOS

		j = self.jointer(source, target, sieve, mask_src, mask_tar)

		return rec, j


DEFAULT_ERROR_WEIGHTS = [
	5,		# topo
	1e-3,	# tick
	3,		# division
	3,		# dots
	1,		# beam
	0.1,	# stemDirection
	3,		# grace
	3,		# warped
	3,		# full measure
	3,		# fake
]


class RectifySieveJointer2Loss (nn.Module):
	def __init__ (self, decisive_confidence=0.5, error_weights=DEFAULT_ERROR_WEIGHTS, loss_weights=[10, 1e-6],
		init_gain_n=6, **kw_args):
		super().__init__()

		self.error_weights = error_weights
		self.loss_weights = [*loss_weights] + [1] * 20

		self.j_metric = JaggedLoss(decisive_confidence)
		self.deducer = RectifySieveJointer2(**kw_args)

		self.mse = nn.MSELoss()
		self.ce = CrossEntropy()
		self.bce = nn.BCELoss()

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=init_gain_n ** -0.5)


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())


	def forward (self, batch):
		inputs = (batch['type'], batch['staff'], batch['feature'], batch['x'], batch['y1'], batch['y2'])
		rec, matrixH = self.deducer(*inputs)

		#is_entity = batch['type'] != EventElementType.PAD
		is_rest = batch['type'] == EventElementType.REST
		is_chord = batch['type'] == EventElementType.CHORD
		is_event = is_rest | is_chord

		#n_elements = is_entity.sum().item()
		n_events = is_event.sum().item()
		n_chords = is_chord.sum().item()
		n_rests = is_rest.sum().item()
		n_rel_tick = batch['maskT'].sum().item()

		loss_topo, acc_topo = self.j_metric(matrixH, batch['matrixH'])

		loss_tick = self.mse(rec['tick'], batch['tick'])
		error_tick = torch.sqrt(loss_tick)

		n_seq = batch['tick'].shape[-1]
		tick_src = rec['tick'].unsqueeze(-1).repeat(1, 1, n_seq)
		tick_tar = rec['tick'].unsqueeze(-2).repeat(1, n_seq, 1)
		pred_tick_diff = (tick_src - tick_tar)
		loss_rel_tick = self.mse(pred_tick_diff.masked_select(batch['maskT']), batch['tickDiff'].masked_select(batch['maskT']))
		error_rel_tick = torch.sqrt(loss_rel_tick)

		loss_tick += loss_rel_tick

		eos = batch['type'] == EventElementType.EOS
		error_duration = torch.sqrt(self.mse(rec['tick'][eos], batch['tick'][eos]))

		loss_division = self.ce(rec['division'], batch['division'], mask=is_event)
		val_division = torch.argmax(rec['division'], dim=-1) == batch['division']
		acc_division = val_division.float().mean()

		loss_dots = self.ce(rec['dots'], batch['dots'], mask=is_event)
		val_dots = torch.argmax(rec['dots'], dim=-1) == batch['dots']
		acc_dots = val_dots.float().mean()

		loss_beam = self.ce(rec['beam'], batch['beam'], mask=is_event)
		val_beam = torch.argmax(rec['beam'], dim=-1)[is_chord] == batch['beam'][is_chord]
		acc_beam = val_beam.float().mean()

		loss_direction = self.ce(rec['stemDirection'], batch['stemDirection'], mask=is_event)
		val_direction = torch.argmax(rec['stemDirection'], dim=-1)[is_chord] == batch['stemDirection'][is_chord]
		acc_direction = val_direction.float().mean()

		loss_grace = self.bce(rec['grace'][is_event], batch['grace'][is_event])
		val_grace = rec['grace'][is_chord] > 0.5
		acc_grace = (val_grace == batch['grace'][is_chord]).float().mean()

		loss_warped = self.bce(rec['timeWarped'][is_event], batch['timeWarped'][is_event])
		val_warped = rec['timeWarped'] > 0.5
		acc_warped = (val_warped == batch['timeWarped']).float().mean()

		loss_full = self.bce(rec['fullMeasure'][is_event], batch['fullMeasure'][is_event])
		val_full = rec['fullMeasure'][is_rest] > 0.5
		acc_full = (val_full == batch['fullMeasure'][is_rest]).float().mean()

		loss_fake = self.bce(rec['fake'][is_event], batch['fake'][is_event])
		val_fake = rec['fake'] > 0.5
		acc_fake = (val_fake == batch['fake']).float().mean()

		#loss = loss_topo * 10 + loss_tick * 1e-6 + loss_division + loss_dots + loss_beam + loss_direction + loss_grace + loss_warped + loss_full + loss_fake
		loss = sum([w * l for w, l in zip(self.loss_weights, [
			loss_topo, loss_tick, loss_division, loss_dots, loss_beam, loss_direction, loss_grace, loss_warped, loss_full, loss_fake,
		])])

		wv = WeightedValue.from_value

		metrics = dict(
			acc_topo			=wv(acc_topo.item()),
			loss_topo			=wv(loss_topo.item()),
			error_tick			=wv(error_tick.item(), n_events),
			error_rel_tick		=wv(error_rel_tick.item(), n_rel_tick),
			error_duration		=wv(error_duration.item()),
			acc_division		=wv(acc_division.item(), n_events),
			acc_dots			=wv(acc_dots.item(), n_events),
			acc_beam			=wv(acc_beam.item(), n_chords),
			acc_stemDirection	=wv(acc_direction.item(), n_chords),
			acc_grace			=wv(acc_grace.item(), n_chords),
			acc_timeWarped		=wv(acc_warped.item(), n_events),
			acc_fullMeasure		=wv(acc_full.item(), n_rests),
			acc_fake			=wv(acc_fake.item(), n_events),
		)

		return loss, metrics


	def stat (self, metrics, n_batch):
		accuracy=dict(
			topo			=metrics['acc_topo'].value,
			tick			=metrics['error_tick'].value,
			rel_tick		=metrics['error_rel_tick'].value,
			duration		=metrics['error_duration'].value,
			division		=metrics['acc_division'].value,
			dots			=metrics['acc_dots'].value,
			beam			=metrics['acc_beam'].value,
			stemDirection	=metrics['acc_stemDirection'].value,
			grace			=metrics['acc_grace'].value,
			timeWarped		=metrics['acc_timeWarped'].value,
			fullMeasure		=metrics['acc_fullMeasure'].value,
			fake			=metrics['acc_fake'].value,
		)

		errors = [
			1 - accuracy['topo'],
			accuracy['rel_tick'] + accuracy['duration'],
			1 - accuracy['division'],
			1 - accuracy['dots'],
			1 - accuracy['beam'],
			1 - accuracy['stemDirection'],
			1 - accuracy['grace'],
			1 - accuracy['timeWarped'],
			1 - accuracy['fullMeasure'],
			1 - accuracy['fake'],
		]
		general_error = sum(
			err * w for err, w in zip(errors, self.error_weights)
		) / sum(self.error_weights)

		return dict(
			loss_topo=metrics['loss_topo'].value,
			accuracy=accuracy,
			general_error=general_error,
		)
