
import torch.nn as nn
import logging

from ..transformer.models import get_pad_mask, get_subsequent_mask
from .semantic_element import SemanticElementType, STAFF_MAX
from .modules import Encoder, Encoder1, EncoderBranch2, Jointer, JaggedLoss



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



class TransformJointerH (nn.Module):
	def __init__ (self, d_model=512, d_inner=2048,
			n_source_layers=6, n_target_layers=1,
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.source_encoder = Encoder1(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.target_encoder = Encoder1(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)

		assert d_model == d_word_vec, \
		'To facilitate the residual connections, \
		 the dimensions of all module outputs shall be the same.'


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD) #& get_subsequent_mask(seq_type)

		source_code = self.source_encoder(seq_id, seq_position, seq_mask)
		target_code = self.target_encoder(seq_id, seq_position, seq_mask)

		results = self.jointer(source_code, target_code, mask[:, 0], mask[:, 1])

		return results


class TransformJointerHLoss (nn.Module):
	def __init__ (self, decisive_confidence=0.5, **kw_args):
		super().__init__()

		self.metric = JaggedLoss(decisive_confidence)
		self.deducer = TransformJointerH(**kw_args)

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p) 

	def forward (self, batch):
		pred = self.deducer(batch['seq_id'], batch['seq_position'], batch['mask'])
		matrixH = batch['matrixH']

		loss, accuracy = self.metric(pred, matrixH)

		return loss, accuracy
