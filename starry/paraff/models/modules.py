
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...transformer.layers import EncoderLayer, DecoderLayer
from ...transformer.models import PositionalEncoding
from ...modules.positionEncoder import SinusoidEncoder
from ..data.timewiseGraph import SEMANTIC_MAX, STAFF_MAX



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


class AttentionStack (nn.Module):
	def __init__ (self, n_layers=6, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model=d_model, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)
			for _ in range(n_layers)])


	def forward (self, x, mask):
		for enc_layer in self.layer_stack:
			x, enc_slf_attn = enc_layer(x, slf_attn_mask=mask)

		return x


class InteractiveAttentionStack (nn.Module):
	def __init__ (self, n_layers=6, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model=d_model, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)
			for _ in range(n_layers)])


	def forward (self, x, mask, other, other_mask):
		for layer in self.layer_stack:
			x, dec_slf_attn, dec_enc_attn = layer(x, other, slf_attn_mask=mask, dec_enc_attn_mask=other_mask)

		return x


class TimewiseGraphEncoder (nn.Module):
	def __init__ (self, n_semantic=SEMANTIC_MAX, n_staff=STAFF_MAX, d_hid=128, angle_cycle=1000):
		super().__init__()

		self.n_semantic = n_semantic
		self.n_staff = n_staff

		self.position_enc = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_hid)

		self.output_dim = self.n_semantic + self.n_staff + 1 + d_hid * 3


	def forward (self, id, staff, x, y, sy1, sy2, confidence):
		vec_id = F.one_hot(id.long(), num_classes=self.n_semantic)
		vec_staff = F.one_hot(staff.long(), num_classes=self.n_staff)
		vec_x = self.position_enc(x.float())
		vec_y = self.position_enc(y.float())
		vec_sy1 = self.position_enc(sy1.float())
		vec_sy2 = self.position_enc(sy2.float())

		return torch.cat([vec_id.float(), vec_staff.float(), confidence.unsqueeze(-1), vec_x, vec_y, vec_sy1 + vec_sy2], dim=-1)


class DecoderWithPosition (nn.Module):
	def __init__ (self, n_trg_vocab, d_model, d_word_vec, n_layers, n_head, d_k, d_v, d_inner,
		pad_idx=0, dropout=0.1, scale_emb=True, angle_cycle=10000):
		super().__init__()

		self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
		self.position_enc = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_model)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.attention = InteractiveAttentionStack(n_layers, d_model, d_inner, n_head, d_k, d_v, dropout)
		self.d_model = d_model
		self.scale_emb = scale_emb


	def forward(self, trg_seq, position, trg_mask, source, src_mask):
		x = self.trg_word_emb(trg_seq)

		if self.scale_emb:
			x *= self.d_model ** 0.5

		x += self.position_enc(position)
		x = self.dropout(x)
		x = self.layer_norm(x)

		return self.attention(x, trg_mask, source, src_mask)
