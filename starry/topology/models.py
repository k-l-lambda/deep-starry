
import torch.nn as nn
import logging

from ..transformer.models import get_pad_mask, get_subsequent_mask
from .semantic_element import SemanticElementType, STAFF_MAX
from .modules import Encoder, Encoder1, Decoder1, EncoderBranch2, Jointer, SieveJointer, JaggedLoss



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
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False, with_temperature=False):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.source_encoder = Encoder1(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.target_encoder = Encoder1(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)

		self.jointer = Jointer(d_model, with_temperature=with_temperature)

		assert d_model == d_word_vec, \
		'To facilitate the residual connections, \
		 the dimensions of all module outputs shall be the same.'


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD)

		source_code = self.source_encoder(seq_id, seq_position, seq_mask)
		target_code = self.target_encoder(seq_id, seq_position, seq_mask)

		results = self.jointer(source_code, target_code, mask[:, 0], mask[:, 1])

		return results


class TransformJointerHLoss (nn.Module):
	def __init__ (self, model_class=TransformJointerH, decisive_confidence=0.5, **kw_args):
		super().__init__()

		self.metric = JaggedLoss(decisive_confidence)
		self.deducer = model_class(**kw_args)

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward (self, batch):
		pred = self.deducer(batch['seq_id'], batch['seq_position'], batch['mask'])
		matrixH = batch['matrixH']

		loss, accuracy = self.metric(pred, matrixH)

		return loss, {'acc_h': accuracy}


class TransformJointerHV (nn.Module):
	def __init__ (self, d_model=512, d_inner=2048,
			n_source_layers=6, n_target_layers=(0, 1, 1),
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False, with_temperature=False):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.source_encoder = Encoder1(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.target_encoder = EncoderBranch2(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)

		self.jointerH = Jointer(d_model, with_temperature=with_temperature)
		self.jointerV = Jointer(d_model, triu_mask=True, with_temperature=with_temperature)


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD)

		source_code = self.source_encoder(seq_id, seq_position, seq_mask)
		target_code, v_code = self.target_encoder(seq_id, seq_position, seq_mask)

		h_results = self.jointerH(source_code, target_code, mask[:, 0], mask[:, 1])
		v_results = self.jointerV(v_code, v_code, mask[:, 2], mask[:, 2])

		return h_results, v_results


class TransformJointerHVLoss (nn.Module):
	def __init__ (self, model_class=TransformJointerHV, decisive_confidence=0.5, **kw_args):
		super().__init__()

		self.metric = JaggedLoss(decisive_confidence)
		self.deducer = model_class(**kw_args)

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward (self, batch):
		predH, predV = self.deducer(batch['seq_id'], batch['seq_position'], batch['mask'])
		matrixH = batch['matrixH']
		matrixV = batch['matrixV']

		loss_h, acc_h = self.metric(predH, matrixH)
		loss_v, acc_v = self.metric(predV, matrixV)

		loss = loss_h + loss_v

		accuracy = {
			'acc_h': acc_h,
			'acc_v': acc_v,
		}

		return loss, accuracy


class TransformJointerH_ED (nn.Module):
	def __init__ (self, d_model=512, d_inner=2048,
			n_source_layers=6, n_target_layers=1,
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False, with_temperature=False):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.source_encoder = Decoder1(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.target_encoder = Encoder1(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)

		self.jointer = Jointer(d_model, with_temperature=with_temperature)


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD)

		target_code = self.target_encoder(seq_id, seq_position, seq_mask)
		source_code = self.source_encoder(seq_id, seq_position, target_code, seq_mask)

		results = self.jointer(source_code, target_code, mask[:, 0], mask[:, 1])

		return results


class TransformJointerH_EDLoss (TransformJointerHLoss):
	def __init__ (self, **kw_args):
		super().__init__(TransformJointerH_ED, **kw_args)


class TransformJointerHV_EDD (nn.Module):
	def __init__ (self, d_model=512, d_inner=2048,
			n_source_layers=6, n_target_layers=1, n_v_layers=1,
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False, with_temperature=False):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.source_encoder = Decoder1(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.target_encoder = Encoder1(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.v_encoder = Decoder1(d_word_vec, n_v_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)

		self.jointer_h = Jointer(d_model, with_temperature=with_temperature)
		self.jointer_v = Jointer(d_model, triu_mask=True, with_temperature=with_temperature)


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD)

		target_code = self.target_encoder(seq_id, seq_position, seq_mask)
		source_code = self.source_encoder(seq_id, seq_position, target_code, seq_mask)
		v_code = self.v_encoder(seq_id, seq_position, source_code, seq_mask)

		h_results = self.jointer_h(source_code, target_code, mask[:, 0], mask[:, 1])
		v_results = self.jointer_v(v_code, v_code, mask[:, 2], mask[:, 2])

		return h_results, v_results


class TransformJointerHV_EDDLoss (TransformJointerHVLoss):
	def __init__ (self, **kw_args):
		super().__init__(TransformJointerHV_EDD, **kw_args)


class TransformSieveJointerH (nn.Module):
	def __init__ (self, d_model=512, d_inner=2048,
			n_source_layers=6, n_target_layers=1, n_sieve_layers=1,
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.source_encoder = Decoder1(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.target_encoder = Encoder1(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)
		self.sieve_encoder = Encoder1(d_word_vec, n_sieve_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb)

		self.jointer = SieveJointer(d_model)


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD)

		target = self.target_encoder(seq_id, seq_position, seq_mask)
		sieve = self.sieve_encoder(seq_id, seq_position, seq_mask)
		source = self.source_encoder(seq_id, seq_position, target, seq_mask)

		results = self.jointer(source, target, sieve, mask[:, 0], mask[:, 1])

		return results


class TransformSieveJointerHLoss (TransformJointerHLoss):
	def __init__ (self, **kw_args):
		super().__init__(TransformSieveJointerH, **kw_args)


class TransformSieveJointerHV (nn.Module):
	def __init__ (self, d_model=512, d_inner=2048,
			n_source_layers=6, n_target_layers=1, n_sieve_layers=1, n_v_layers=1,
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False, n_type=SemanticElementType.MAX, n_staff=STAFF_MAX):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.source_encoder = Decoder1(d_word_vec, n_source_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb, n_type=n_type, n_staff=n_staff)
		self.target_encoder = Encoder1(d_word_vec, n_target_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb, n_type=n_type, n_staff=n_staff)
		self.sieve_encoder = Encoder1(d_word_vec, n_sieve_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb, n_type=n_type, n_staff=n_staff)
		self.v_encoder = Decoder1(d_word_vec, n_v_layers, n_head, d_k, d_v, d_model, d_inner, dropout=dropout, scale_emb=scale_emb, n_type=n_type, n_staff=n_staff)

		self.jointer_h = SieveJointer(d_model)
		self.jointer_v = Jointer(d_model, triu_mask=True)


	def forward (self, seq_id, seq_position, mask):
		seq_type = seq_id[:, :, 0]
		seq_mask = get_pad_mask(seq_type, SemanticElementType.PAD)

		target = self.target_encoder(seq_id, seq_position, seq_mask)
		sieve = self.sieve_encoder(seq_id, seq_position, seq_mask)
		source = self.source_encoder(seq_id, seq_position, target, seq_mask)
		vcode = self.v_encoder(seq_id, seq_position, source, seq_mask)

		h_results = self.jointer_h(source, target, sieve, mask[:, 0], mask[:, 1])
		v_results = self.jointer_v(vcode, vcode, mask[:, 2], mask[:, 2])

		return h_results, v_results


class TransformSieveJointerHVLoss (TransformJointerHVLoss):
	def __init__ (self, **kw_args):
		super().__init__(TransformSieveJointerHV, **kw_args)
