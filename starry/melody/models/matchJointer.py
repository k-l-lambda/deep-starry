
import torch
import torch.nn as nn

from .modules import Encoder, Decoder, Jointer



class MatchJointerRaw (nn.Module):
	def __init__ (self, n_layers_ce=1, n_layers_se=1, n_layers_sd=1,
			d_model=1, d_inner=2048, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.c_encoder = Encoder(n_layers_ce, **encoder_args, scale_emb=scale_emb)
		self.s_encoder = Encoder(n_layers_se, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = Decoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_input, s_input):
		c_output = self.c_encoder(c_input)
		s_output = self.s_encoder(s_input)

		s_output = self.s_decoder(s_output, c_output)

		return self.jointer(s_output, c_output)
