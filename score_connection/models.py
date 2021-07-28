
import torch.nn as nn
import numpy as np

from ..transformer.Layers import EncoderLayer
from .semantic_element import SemanticElementType



STAFF_MAX = 8


class Embedder(nn.Module):
	def __init__ (self, d_word_vec):
		super().__init__()

		self.type_emb = nn.Embedding(SemanticElementType.MAX, d_word_vec, padding_idx = SemanticElementType.PAD)
		self.staff_emb = nn.Embedding(STAFF_MAX, d_word_vec)

	def forward (self, src_seq):
		types = src_seq[:, 0]
		staves = src_seq[:, 1]
		positions = src_seq[:, 2:]

		return self.type_emb(types) + self.staff_emb(staves) + positions


class Encoder(nn.Module):
	def __init__ (self, d_word_vec, n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.src_word_emb = Embedder(d_word_vec)

		self.dropout = nn.Dropout(p=dropout)

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model

	def forward (self, src_seq, src_mask, return_attns=False):
		enc_slf_attn_list = []

		# -- Forward
		enc_output = self.src_word_emb(src_seq)
		if self.scale_emb:
			enc_output *= self.d_model ** 0.5
		enc_output = self.dropout(enc_output)
		enc_output = self.layer_norm(enc_output)

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
			enc_slf_attn_list += [enc_slf_attn] if return_attns else []

		if return_attns:
			return enc_output, enc_slf_attn_list

		return enc_output,
