
import torch.nn as nn
import numpy as np
import logging

from ..transformer.layers import EncoderLayer
from ..transformer.models import get_pad_mask, get_subsequent_mask
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

		# normalize for inner product
		enc_output = nn.functional.normalize(enc_output, dim=-1)

		if return_attns:
			return enc_output, enc_slf_attn_list

		return enc_output,


class TransformJointer (nn.Module):
	def __init__ (self, d_model=512, d_inner=2048,
			n_source_layers=6, n_target_layers=1,
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.source_encoder = Encoder(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.target_encoder = Encoder(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)

		assert d_model == d_word_vec, \
		'To facilitate the residual connections, \
		 the dimensions of all module outputs shall be the same.'


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD) #& get_subsequent_mask(seq_type)

		source_code, = self.source_encoder(seq_id, seq_position, seq_mask)
		target_code, = self.target_encoder(seq_id, seq_position, seq_mask)

		results = []
		for i, (code_src, code_tar) in enumerate(zip(source_code, target_code)):
			msk_src, msk_tar = mask[i, 0].unsqueeze(-1), mask[i, 1].unsqueeze(-1)
			code_src = code_src.masked_select(msk_src).reshape(-1, self.d_model)	# (src_joint, d_model)
			code_tar = code_tar.masked_select(msk_tar).reshape(-1, self.d_model)	# (tar_joint, d_model)

			code_src = code_src.unsqueeze(-2)			# (src_joint, 1, d_model)
			code_tar = code_tar.unsqueeze(-1)			# (tar_joint, d_model, 1)

			src_joints = code_src.shape[0]
			tar_joints = code_tar.shape[0]

			code_src = code_src.unsqueeze(1).repeat(1, tar_joints, 1, 1)	# (src_joint, tar_joints, 1, d_model)
			code_tar = code_tar.repeat(src_joints, 1, 1, 1)					# (src_joint, tar_joint, d_model, 1)

			result = code_src.matmul(code_tar).clamp(min=0).flatten()					# (src_joint * tar_joint)
			results.append(result)

		return results


class TransformJointerLoss (nn.Module):
	def __init__ (self, decisive_confidence=0.5, **kw_args):
		super().__init__()

		self.decisive_confidence = decisive_confidence
		self.deducer = TransformJointer(**kw_args)

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p) 

	def forward (self, batch):
		pred = self.deducer(batch['seq_id'], batch['seq_position'], batch['mask'])
		matrixH = batch['matrixH']

		loss = 0
		#samples = 0
		ground_positive, ground_negative, true_positive, true_negative = 0, 0, 0, 0
		for pred_i, truth in zip(pred, matrixH):
			# skip empty prediction
			if len(pred_i) == 0:
				#logging.warn('empty mask.')
				continue

			truth = truth[:len(pred_i)]
			loss += nn.functional.binary_cross_entropy(pred_i, truth)

			pred_binary = pred_i > self.decisive_confidence
			target_binary = truth > 0
			#errors += sum(pred_binary ^ target_binary)
			#samples += len(pred_i)
			ground_positive += sum(target_binary)
			ground_negative += sum(~target_binary)
			true_positive += sum(pred_binary & target_binary)
			true_negative += sum(~pred_binary & ~target_binary)

		loss /= len(pred)

		ground_positive = max(ground_positive, 1)
		ground_negative = max(ground_negative, 1)
		accuracy = (true_positive / ground_positive) * (true_negative / ground_negative)

		return loss, accuracy
