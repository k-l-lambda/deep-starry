
import torch.nn as nn
import numpy as np

from transformer.layers import EncoderLayer
from transformer.models import get_pad_mask, get_subsequent_mask
from .semantic_element import SemanticElementType, STAFF_MAX



class Embedder (nn.Module):
	def __init__ (self, d_word_vec):
		super().__init__()

		self.type_emb = nn.Embedding(SemanticElementType.MAX, d_word_vec, padding_idx = SemanticElementType.PAD)
		self.staff_emb = nn.Embedding(STAFF_MAX, d_word_vec)

	# seq: (n, seq, 2)
	def forward (self, seq):
		types = seq[:, :, 0]
		staves = seq[:, :, 1]

		return self.type_emb(types) + self.staff_emb(staves)


class Encoder (nn.Module):
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

	# seq_id:		(n, seq, 2)
	# seq_position:	(n, seq, d_word)
	# mask:			(n, seq, seq)
	def forward (self, seq_id, seq_position, mask, return_attns=False):	# (n, seq, d_word)
		enc_slf_attn_list = []

		# -- Forward
		enc_output = self.src_word_emb(seq_id) + seq_position

		if self.scale_emb:
			enc_output *= self.d_model ** 0.5

		enc_output = self.dropout(enc_output)
		enc_output = self.layer_norm(enc_output)

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
			enc_slf_attn_list += [enc_slf_attn] if return_attns else []

		if return_attns:
			return enc_output, enc_slf_attn_list

		return enc_output,


class TransformJointer (nn.Module):
	def __init__ (self, d_word_vec=512, d_model=512, d_inner=2048,
			n_source_layers=6, n_target_layers=1,
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False):
		super().__init__()

		self.source_encoder = Encoder(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.target_encoder = Encoder(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p) 

		assert d_model == d_word_vec, \
		'To facilitate the residual connections, \
		 the dimensions of all module outputs shall be the same.'


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD) #& get_subsequent_mask(seq_type)

		source_code, = self.source_encoder(seq_id, seq_position, seq_mask)
		target_code, = self.target_encoder(seq_id, seq_position, seq_mask)

		print('source_code:', source_code.shape)
		print('target_code:', target_code.shape)
		# TODO:


	def train (self, batch):
		pred = self.forward(batch['seq_id'], batch['seq_position'], batch['mask'])
		matrixH = batch['matrixH']

		# TODO: compute loss
