
import torch.nn as nn

from ...transformer.layers import EncoderLayer, DecoderLayer
from ..event_element import TARGET_DIM, EventElementType
from .modules import EventEncoder, SieveJointer, RectifierParser



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
