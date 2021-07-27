
from enum import Enum
import torch.nn as nn

from transformer.Layers import EncoderLayer



class SemanticElementType(Enum):
	BOS					= 0,
	PAD					= 1,

	NoteheadS0			= 2,
	NoteheadS1			= 3,
	NoteheadS2			= 4,
	GraceNoteheadS0		= 5,
	vline_Stem			= 6,
	Flag3				= 7,
	BeamLeft			= 8,
	BeamContinue		= 9,
	BeamRight			= 10,
	Dot					= 11,
	Rest0				= 12,
	Rest1				= 13,
	Rest2				= 14,
	Rest3				= 15,
	Rest4				= 16,
	Rest5				= 17,
	Rest6				= 18,


class Embedder(nn.Module):
	def __init__ (self, n_vocab, d_word_vec, padding_idx = SemanticElementType.PAD):
		super().__init__()

	def forward (self):
		pass


class Encoder(nn.Module):
	def __init__ (
			self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1, n_position=200, scale_emb=False):

		super().__init__()

		self.src_word_emb = Embedder(n_src_vocab, d_word_vec)

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
