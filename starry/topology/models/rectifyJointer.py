
import torch
import torch.nn as nn

from ...transformer.layers import EncoderLayer, DecoderLayer
from ..event_element import TARGET_DIM, EventElementType
from .modules import EventEncoder, SieveJointer, RectifierParser, JaggedLoss, CrossEntropy



class EncoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, x, mask=None):	# (n, seq, d_word)
		enc_output = x
		for enc_layer in self.layer_stack:
			enc_output, _ = enc_layer(enc_output, mask)

		return enc_output


class DecoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, dec_input, enc_output, mask=None):	# (n, seq, d_word)
		dec_output = dec_input
		for layer in self.layer_stack:
			dec_output, _1, _2 = layer(dec_output, enc_output, mask, mask)

		return dec_output


class Encoder (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.stack = EncoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model


	def forward (self, x, mask=None):
		if self.scale_emb:
			x *= self.d_model ** 0.5

		x = self.dropout(x)
		x = self.layer_norm(x)

		x = self.stack(x, mask)

		return x


class Decoder (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.stack = DecoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model


	def forward (self, x, enc_output, mask=None):
		if self.scale_emb:
			x *= self.d_model ** 0.5

		x = self.dropout(x)
		x = self.layer_norm(x)

		x = self.stack(x, enc_output, mask)

		return x


class RectifySieveJointer (nn.Module):
	def __init__ (self, n_trunk_layers, n_rectifier_layers, n_source_layers=2, n_target_layers=1, n_sieve_layers=1,
			d_model=512, d_inner=2048, angle_cycle=1000, n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False):
		super().__init__()

		self.event_encoder = EventEncoder(d_model, angle_cycle=angle_cycle)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.trunk_encoder = EncoderLayerStack(n_trunk_layers, **encoder_args)
		self.rectifier_encoder = Encoder(n_rectifier_layers, **encoder_args, scale_emb=scale_emb)
		self.target_encoder = Encoder(n_target_layers, **encoder_args, scale_emb=scale_emb)
		self.sieve_encoder = Encoder(n_sieve_layers, **encoder_args, scale_emb=scale_emb)
		self.source_encoder = Decoder(n_source_layers, **encoder_args, scale_emb=scale_emb)

		self.rec_out = nn.Linear(d_model, TARGET_DIM)
		self.rec_parser = RectifierParser()

		self.jointer = SieveJointer(d_model)


	def forward (self, inputs):	# dict(name -> T(n, seq, xtar)), list(n, T((n - 1) * (n - 1)))
		x = self.event_encoder(inputs)	# (n, seq, d_model)

		mask_pad = inputs['type'] != EventElementType.PAD
		mask = mask_pad.unsqueeze(-2)

		x = self.trunk_encoder(x, mask)

		rec = self.rectifier_encoder(x, mask)
		rec = self.rec_out(rec)
		rec = self.rec_parser(rec)

		target = self.target_encoder(x, mask)
		sieve = self.sieve_encoder(x, mask)
		source = self.source_encoder(x, target, mask)

		mask_src = mask_pad & (inputs['type'] != EventElementType.BOS)
		mask_tar = mask_pad & (inputs['type'] != EventElementType.EOS)

		j = self.jointer(source, target, sieve, mask_src, mask_tar)

		return rec, j


class RectifySieveJointerLoss (nn.Module):
	def __init__ (self, decisive_confidence=0.5, **kw_args):
		super().__init__()

		self.j_metric = JaggedLoss(decisive_confidence)
		self.deducer = RectifySieveJointer(**kw_args)

		self.mse = nn.MSELoss()
		self.ce = CrossEntropy()
		self.bce = nn.BCELoss()

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)


	def forward (self, batch):
		rec, matrixH = self.deducer(batch)

		loss_topo, acc_topo = self.j_metric(matrixH, batch['matrixH'])

		loss_tick = self.mse(rec['tick'], batch['tick'])
		error_tick = torch.sqrt(loss_tick)

		loss_division = self.ce(rec['division'], batch['division'])
		val_division = torch.argmax(rec['division'], dim=-1) == batch['division']
		acc_division = val_division.float().mean()

		loss_dots = self.ce(rec['dots'], batch['dots'])
		val_dots = torch.argmax(rec['dots'], dim=-1) == batch['dots']
		acc_dots = val_dots.float().mean()

		loss_beam = self.ce(rec['beam'], batch['beam'])
		val_beam = torch.argmax(rec['beam'], dim=-1) == batch['beam']
		acc_beam = val_beam.float().mean()

		loss_direction = self.ce(rec['stemDirection'], batch['stemDirection'])
		val_direction = torch.argmax(rec['stemDirection'], dim=-1) == batch['stemDirection']
		acc_direction = val_direction.float().mean()

		loss_grace = self.bce(rec['grace'], batch['grace'])
		val_grace = rec['grace'] > 0.5
		acc_grace = (val_grace == batch['grace']).float().mean()

		loss_warped = self.bce(rec['timeWarped'], batch['timeWarped'])
		val_warped = rec['timeWarped'] > 0.5
		acc_warped = (val_warped == batch['timeWarped']).float().mean()

		loss_full = self.bce(rec['fullMeasure'], batch['fullMeasure'])
		val_full = rec['fullMeasure'] > 0.5
		acc_full = (val_full == batch['fullMeasure']).float().mean()

		loss_confidence = self.bce(rec['confidence'], batch['confidence'])
		val_confidence = rec['confidence'] > 0.5
		acc_confidence = (val_confidence == batch['confidence']).float().mean()

		loss = loss_topo * 10 + loss_tick * 1e-6 + loss_division + loss_dots + loss_beam + loss_direction + loss_grace + loss_warped + loss_full + loss_confidence

		metrics = dict(
			acc_topo			=acc_topo,
			loss_topo			=loss_topo,
			error_tick			=error_tick,
			acc_division		=acc_division,
			acc_dots			=acc_dots,
			acc_beam			=acc_beam,
			acc_stemDirection	=acc_direction,
			acc_grace			=acc_grace,
			acc_timeWarped		=acc_warped,
			acc_fullMeasure		=acc_full,
			acc_confidence		=acc_confidence,
		)

		return loss, metrics
