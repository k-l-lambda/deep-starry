
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...transformer.layers import EncoderLayer
from ...transformer.models import PositionalEncoding



class HeadSummaryEncoder (nn.Module):
	def __init__ (self, n_src_vocab, n_layers,
			d_model, pad_idx, dropout=0.1, n_position=200, scale_emb=False, **kw_args):
		super().__init__()

		self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=pad_idx)
		self.position_enc = PositionalEncoding(d_model, n_position=n_position)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model=d_model, dropout=dropout, **kw_args)
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.scale_emb = scale_emb
		self.d_model = d_model


	# src_seq:		(n, seq)
	# src_mask:		(n, seq)
	# summary_emb:	(n, d_model)
	def forward (self, src_seq, src_mask, summary_emb, return_attns=False):
		enc_slf_attn_list = []

		enc_output = self.src_word_emb(src_seq)	# (n, seq, d_model)
		enc_output[:, 0] += summary_emb	# add summary embedding on the first element

		if self.scale_emb:
			enc_output *= self.d_model ** 0.5
		enc_output = self.dropout(self.position_enc(enc_output))
		enc_output = self.layer_norm(enc_output)

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
			enc_slf_attn_list += [enc_slf_attn] if return_attns else []

		if return_attns:
			return enc_output, enc_slf_attn_list

		return enc_output
