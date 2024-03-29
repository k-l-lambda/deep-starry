
from typing import Optional
import torch
import torch.nn as nn

from ..event_element import EventElementType, VTICK_BASE
from ...utils.weightedValue import WeightedValue
from .modules import EventOrderedEncoder, EventEncoderV4, CrossEntropy, RectifierParser2, RectifierParser3
from .rectifyJointer import EncoderLayerStack, DEFAULT_ERROR_WEIGHTS
from ...modules.classInt import Int2PClass



RectifierParsers = {
	'v2': RectifierParser2,
	'v3': RectifierParser3,
}


class BeadPicker (nn.Module):
	def __init__ (self, n_layers=1, angle_cycle=1000, d_position=512, feature_activation=None, zero_candidates=False,
			with_time8th=False,
			d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64, dropout=0.1, rectifier_version='v2', **_):
		super().__init__()

		self.with_time8th = with_time8th

		event_encoder_class = EventEncoderV4 if with_time8th else EventOrderedEncoder
		self.event_encoder = event_encoder_class(d_model, angle_cycle=angle_cycle, d_position=d_position,
			feature_activation=feature_activation, zero_candidates=zero_candidates)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.attention = EncoderLayerStack(n_layers, **encoder_args)

		rectifier_class = RectifierParsers[rectifier_version]

		self.out = nn.Linear(d_model, 1 + rectifier_class.VEC_CHANNELS)
		self.rec_parser = rectifier_class()
		self.sigmoid = nn.Sigmoid()

		self.PAD = EventElementType.PAD


	def forward (self, stype, staff, feature, x, y1, y2, beading_pos, time8th: Optional[torch.Tensor] =None):
		if time8th is not None:
			x = self.event_encoder(stype, staff, feature, x, y1, y2, beading_pos, time8th)	# (n, seq, d_model)
		else:
			x = self.event_encoder(stype, staff, feature, x, y1, y2, beading_pos)	# (n, seq, d_model)

		mask_pad = stype != self.PAD
		mask = mask_pad.unsqueeze(-2)

		x = self.attention(x, mask)
		x = self.out(x)

		successor = self.sigmoid(x[:, :, 0])
		rec = self.rec_parser(x[:, :, 1:])

		return successor, rec


class BeadPickerOnnx (BeadPicker):
	def __init__ (self, out_temperature=1, **kw_args):
		super().__init__(**kw_args)

		self.out_temperature = out_temperature


	def forward (self, *args):
		successor, rec = super().forward(*args)

		rec['division'] = torch.softmax(rec['division'] / self.out_temperature, dim=-1)
		rec['dots'] = torch.softmax(rec['dots'] / self.out_temperature, dim=-1)
		rec['beam'] = torch.softmax(rec['beam'] / self.out_temperature, dim=-1)
		rec['stemDirection'] = torch.softmax(rec['stemDirection'] / self.out_temperature, dim=-1)

		return (successor, *[rec[key] for key in
			['tick', 'division', 'dots', 'beam', 'stemDirection', 'grace', 'timeWarped', 'fullMeasure', 'fake']])


class BeadPickerLoss (nn.Module):
	def __init__ (self, decisive_confidence=0.5, error_weights=DEFAULT_ERROR_WEIGHTS, loss_weights=[10, 1e-6],
		usePivotX=False, init_gain_n=2, freeze=None, **kw_args):
		super().__init__()

		self.error_weights = error_weights
		self.loss_weights = [*loss_weights] + [1] * 20
		self.decisive_confidence = decisive_confidence
		self.usePivotX = usePivotX
		self.use_vtick = kw_args['rectifier_version'] == 'v3'

		self.deducer = BeadPicker(**kw_args)

		self.mse = nn.MSELoss()
		self.ce = CrossEntropy()
		self.bce = nn.BCELoss()
		self.bce_logits = nn.BCEWithLogitsLoss()

		self.tick2vec = Int2PClass(VTICK_BASE)

		if freeze is not None:
			for param in self.deducer.parameters():
				param.requires_grad = False

			layer_n = freeze.get('train_layer_n', 0)
			train_out = freeze.get('train_out', True)

			if layer_n > 0:
				layers = self.deducer.attention.layer_stack[-layer_n:]
				for layer in layers:
					for param in layer.parameters():
						param.requires_grad = True

			if train_out:
				for param in self.deducer.out.parameters():
					param.requires_grad = True
		else:
			# initialize parameters
			for p in self.parameters():
				if p.dim() > 1:
					nn.init.xavier_uniform_(p, gain=init_gain_n ** -0.5)


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())


	def forward (self, batch):
		inputs = (
			batch['type'],
			batch['staff'],
			batch['feature'],
			batch['pivotX'] if self.usePivotX else batch['x'],
			batch['y1'],
			batch['y2'],
			batch['beading_pos'],
			batch['time8th'],
		)
		pred_suc, rec = self.deducer(*inputs)

		is_entity = batch['type'] != EventElementType.PAD
		is_rest = batch['type'] == EventElementType.REST
		is_chord = batch['type'] == EventElementType.CHORD
		is_event = is_rest | is_chord
		is_candidate = is_entity & (batch['beading_pos'] == 0)
		is_ce = is_candidate & is_event
		is_fixed = batch['beading_pos'] < 0

		n_candidates = is_candidate.sum().item()
		n_events = is_event.sum().item()
		n_ce = is_ce.sum().item()
		n_chords = is_chord.sum().item()
		n_rests = is_rest.sum().item()
		n_rel_tick = batch['maskT'].sum().item()
		n_fixed = is_fixed.sum().item()

		loss_suc = self.bce(pred_suc[is_entity], batch['successor'][is_entity])
		err_suc = 1 - ((pred_suc[is_candidate] > self.decisive_confidence).float() == batch['successor'][is_candidate]).float().mean()

		if not self.use_vtick:
			loss_tick = self.mse(rec['tick'], batch['tick'])
			err_tick = torch.sqrt(loss_tick.detach())

			loss_tick_fixed = self.mse(rec['tick'][is_fixed], batch['tick'][is_fixed])
			err_tick_fixed = torch.sqrt(loss_tick_fixed.detach())
		else:
			target_vtick = self.tick2vec(batch['tick'])
			loss_tick = self.bce_logits(rec['vtick'], target_vtick)
			err_tick = torch.sqrt(self.mse(rec['tick'], batch['tick']).detach())

			loss_tick_fixed = self.bce_logits(rec['vtick'][is_fixed], target_vtick[is_fixed])
			err_tick_fixed = torch.sqrt(self.mse(rec['tick'][is_fixed], batch['tick'][is_fixed]).detach())

		n_seq = batch['tick'].shape[-1]
		tick_src = rec['tick'].unsqueeze(-1).repeat(1, 1, n_seq)
		tick_tar = rec['tick'].unsqueeze(-2).repeat(1, n_seq, 1)
		pred_tick_diff = (tick_src - tick_tar)
		loss_rel_tick = self.mse(pred_tick_diff.masked_select(batch['maskT']), batch['tickDiff'].masked_select(batch['maskT']))
		err_rel_tick = torch.sqrt(loss_rel_tick.detach())

		if not self.use_vtick:
			loss_tick += loss_rel_tick

		eos = batch['type'] == EventElementType.EOS
		err_duration = torch.sqrt(self.mse(rec['tick'][eos], batch['tick'][eos]).detach())

		loss_division = self.ce(rec['division'], batch['division'], mask=is_event)
		val_division = torch.argmax(rec['division'][is_ce], dim=-1) == batch['division'][is_ce]
		err_division = 1 - val_division.float().mean()

		loss_dots = self.ce(rec['dots'], batch['dots'], mask=is_event)
		val_dots = torch.argmax(rec['dots'][is_ce], dim=-1) == batch['dots'][is_ce]
		err_dots = 1 - val_dots.float().mean()

		loss_beam = self.ce(rec['beam'], batch['beam'], mask=is_event)
		val_beam = torch.argmax(rec['beam'], dim=-1)[is_chord] == batch['beam'][is_chord]
		err_beam = 1 - val_beam.float().mean()

		loss_direction = self.ce(rec['stemDirection'], batch['stemDirection'], mask=is_event)
		val_direction = torch.argmax(rec['stemDirection'], dim=-1)[is_chord] == batch['stemDirection'][is_chord]
		err_direction = 1 - val_direction.float().mean()

		loss_grace = self.bce(rec['grace'][is_event], batch['grace'][is_event])
		val_grace = rec['grace'][is_chord] > 0.5
		err_grace = 1 - (val_grace == batch['grace'][is_chord]).float().mean()

		loss_warped = self.bce(rec['timeWarped'][is_event], batch['timeWarped'][is_event])
		val_warped = rec['timeWarped'][is_event] > 0.5
		err_warped = 1 - (val_warped == batch['timeWarped'][is_event]).float().mean()

		loss_full = self.bce(rec['fullMeasure'][is_event], batch['fullMeasure'][is_event])
		val_full = rec['fullMeasure'][is_rest] > 0.5
		err_full = 1 - (val_full == batch['fullMeasure'][is_rest]).float().mean()

		loss_fake = self.bce(rec['fake'][is_event], batch['fake'][is_event])
		val_fake = rec['fake'][is_event] > 0.5
		err_fake = 1 - (val_fake == batch['fake'][is_event]).float().mean()

		loss = sum([w * l for w, l in zip(self.loss_weights, [
			loss_suc, loss_tick_fixed, loss_tick, loss_division, loss_dots, loss_beam, loss_direction, loss_grace, loss_warped, loss_full, loss_fake,
		])])

		wv = WeightedValue.from_value

		metrics = dict(
			err_suc				=wv(err_suc.item(), n_candidates),
			loss_suc			=wv(loss_suc.item()),
			err_tick			=wv(err_tick.item(), n_events),
			err_tick_fixed		=wv(err_tick_fixed.item(), n_fixed),
			err_rel_tick		=wv(err_rel_tick.item(), n_rel_tick),
			err_duration		=wv(err_duration.item()),
			err_division		=wv(err_division.item(), n_ce),
			err_dots			=wv(err_dots.item(), n_ce),
			err_beam			=wv(err_beam.item(), n_chords),
			err_stemDirection	=wv(err_direction.item(), n_chords),
			err_grace			=wv(err_grace.item(), n_chords),
			err_timeWarped		=wv(err_warped.item(), n_events),
			err_fullMeasure		=wv(err_full.item(), n_rests),
			err_fake			=wv(err_fake.item(), n_events),
		)

		return loss, metrics


	def stat (self, metrics, n_batch):
		error = dict(
			suc				=metrics['err_suc'].value,
			tick			=metrics['err_tick'].value,
			tick_fixed		=metrics['err_tick_fixed'].value,
			rel_tick		=metrics['err_rel_tick'].value,
			duration		=metrics['err_duration'].value,
			division		=metrics['err_division'].value,
			dots			=metrics['err_dots'].value,
			beam			=metrics['err_beam'].value,
			stemDirection	=metrics['err_stemDirection'].value,
			grace			=metrics['err_grace'].value,
			timeWarped		=metrics['err_timeWarped'].value,
			fullMeasure		=metrics['err_fullMeasure'].value,
			fake			=metrics['err_fake'].value,
		)

		errors = [
			error['suc'],
			error['tick_fixed'] + error['duration'],
			error['division'],
			error['dots'],
			error['beam'],
			error['stemDirection'],
			error['grace'],
			error['timeWarped'],
			error['fullMeasure'],
			error['fake'],
		]
		general_error = sum(
			err * w for err, w in zip(errors, self.error_weights)
		) / sum(self.error_weights)

		return dict(
			loss_suc=metrics['loss_suc'].value,
			error=error,
			general_error=general_error,
		)
