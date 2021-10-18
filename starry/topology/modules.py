
import torch
import torch.nn as nn

from ..transformer.layers import EncoderLayer, DecoderLayer
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


class EncoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, x, mask):	# (n, seq, d_word)
		enc_output = x
		for enc_layer in self.layer_stack:
			enc_output, _ = enc_layer(enc_output, slf_attn_mask=mask)

		return enc_output


class DecoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, dec_input, enc_output, mask):	# (n, seq, d_word)
		dec_output = dec_input
		for layer in self.layer_stack:
			dec_output, _1, _2 = layer(dec_output, enc_output, slf_attn_mask=mask, dec_enc_attn_mask=mask)

		return dec_output


class Encoder1 (nn.Module):
	def __init__ (self, d_word_vec, n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.src_word_emb = Embedder(d_word_vec)

		self.dropout = nn.Dropout(p=dropout)

		self.stack = EncoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model

	# seq_id:		(n, seq, 2)
	# seq_position:	(n, seq, d_word)
	# mask:			(n, seq, seq)
	def forward (self, seq_id, seq_position, mask):	# (n, seq, d_word), (n, seq, d_word)
		# -- Forward
		enc_output = self.src_word_emb(seq_id) + seq_position

		if self.scale_emb:
			enc_output *= self.d_model ** 0.5

		enc_output = self.dropout(enc_output)
		enc_output = self.layer_norm(enc_output)

		enc_output = self.stack(enc_output, mask)

		return enc_output


class EncoderBranch2 (nn.Module):
	def __init__ (self, d_word_vec, n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.src_word_emb = Embedder(d_word_vec)

		self.dropout = nn.Dropout(p=dropout)

		n_layers_trunk, n_layers_left, n_layers_right = n_layers

		self.stack_trunk = EncoderLayerStack(n_layers_trunk, n_head, d_k, d_v, d_model, d_inner, dropout)
		self.stack_left = EncoderLayerStack(n_layers_left, n_head, d_k, d_v, d_model, d_inner, dropout)
		self.stack_right = EncoderLayerStack(n_layers_right, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model

	# seq_id:		(n, seq, 2)
	# seq_position:	(n, seq, d_word)
	# mask:			(n, seq, seq)
	def forward (self, seq_id, seq_position, mask):	# (n, seq, d_word), (n, seq, d_word)
		enc_output = self.src_word_emb(seq_id) + seq_position

		if self.scale_emb:
			enc_output *= self.d_model ** 0.5

		enc_output = self.dropout(enc_output)
		enc_output = self.layer_norm(enc_output)

		enc_output = self.stack_trunk(enc_output, mask)
		enc_out_left = self.stack_left(enc_output, mask)
		enc_out_right = self.stack_right(enc_output, mask)

		return enc_out_left, enc_out_right


class Decoder1 (nn.Module):
	def __init__ (self, d_word_vec, n_layers, n_head, d_k, d_v,
			d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.src_word_emb = Embedder(d_word_vec)

		self.dropout = nn.Dropout(p=dropout)

		self.stack = DecoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model

	# seq_id:		(n, seq, 2)
	# seq_position:	(n, seq, d_word)
	# mask:			(n, seq, seq)
	# enc_output:	(n, seq, d_word)
	def forward (self, seq_id, seq_position, enc_output, mask):	# (n, seq, d_word)
		dec_output = self.src_word_emb(seq_id) + seq_position

		if self.scale_emb:
			dec_output *= self.d_model ** 0.5

		dec_output = self.dropout(dec_output)
		dec_output = self.layer_norm(dec_output)

		dec_output = self.stack(dec_output, enc_output, mask)

		return dec_output


class Jointer (nn.Module):
	def __init__ (self, d_model, triu_mask=False, with_temperature=False):
		super().__init__()

		self.d_model = d_model

		if triu_mask:
			mask = torch.triu(torch.ones((0x200, 0x200))) == 0
			self.register_buffer('triu_mask', mask, persistent=False)
		else:
			self.triu_mask = None

		#self.temperature = torch.zeros((1,))
		#if with_temperature:
		#	self.temperature = nn.Parameter(self.temperature)

	def forward (self, source, target, mask_src, mask_tar):
		# normalize for inner product
		source = nn.functional.normalize(source, dim=-1)
		target = nn.functional.normalize(target, dim=-1)

		results = []
		for i, (code_src, code_tar) in enumerate(zip(source, target)):
			msk_src, msk_tar = mask_src[i].unsqueeze(-1), mask_tar[i].unsqueeze(-1)
			code_src = code_src.masked_select(msk_src).reshape(-1, self.d_model)	# (src_joint, d_model)
			code_tar = code_tar.masked_select(msk_tar).reshape(-1, self.d_model)	# (tar_joint, d_model)

			code_src = code_src.unsqueeze(-2)										# (src_joint, 1, d_model)
			code_tar = code_tar.unsqueeze(-1)										# (tar_joint, d_model, 1)

			src_joints = code_src.shape[0]
			tar_joints = code_tar.shape[0]

			code_src = code_src.unsqueeze(1).repeat(1, tar_joints, 1, 1)			# (src_joint, tar_joints, 1, d_model)
			code_tar = code_tar.repeat(src_joints, 1, 1, 1)							# (src_joint, tar_joint, d_model, 1)

			result = code_src.matmul(code_tar).clamp(min=0)							# (src_joint, tar_joint)
			if self.triu_mask is not None:
				result = result.squeeze(-1).squeeze(-1).masked_select(self.triu_mask[:result.shape[0], :result.shape[1]])
			else:
				result = result.flatten()
			#result *= torch.exp(self.temperature)

			results.append(result)

		return results


class SieveJointer (nn.Module):
	def __init__ (self, d_model, triu_mask=False):
		super().__init__()

		self.d_model = d_model

		if triu_mask:
			mask = torch.triu(torch.ones((0x200, 0x200))) == 0
			self.register_buffer('triu_mask', mask, persistent=False)
		else:
			self.triu_mask = None

	def forward (self, source, target, sieve, mask_src, mask_tar):
		# normalize for inner product
		sieve = nn.functional.normalize(sieve, dim=-1)
		source = nn.functional.normalize(source * sieve, dim=-1)
		target = nn.functional.normalize(target, dim=-1)

		results = []
		for i, (code_src, code_tar) in enumerate(zip(source, target)):
			msk_src, msk_tar = mask_src[i].unsqueeze(-1), mask_tar[i].unsqueeze(-1)
			code_src = code_src.masked_select(msk_src).reshape(-1, self.d_model)	# (src_joint, d_model)
			code_tar = code_tar.masked_select(msk_tar).reshape(-1, self.d_model)	# (tar_joint, d_model)

			code_src = code_src.unsqueeze(-2)										# (src_joint, 1, d_model)
			code_tar = code_tar.unsqueeze(-1)										# (tar_joint, d_model, 1)

			src_joints = code_src.shape[0]
			tar_joints = code_tar.shape[0]

			code_src = code_src.unsqueeze(1).repeat(1, tar_joints, 1, 1)			# (src_joint, tar_joints, 1, d_model)
			code_tar = code_tar.repeat(src_joints, 1, 1, 1)							# (src_joint, tar_joint, d_model, 1)

			result = code_src.matmul(code_tar).clamp(min=0)							# (src_joint, tar_joint)
			if self.triu_mask is not None:
				result = result.squeeze(-1).squeeze(-1).masked_select(self.triu_mask[:result.shape[0], :result.shape[1]])
			else:
				result = result.flatten()
			results.append(result)

		return results


class JaggedLoss (nn.Module):
	def __init__ (self, decisive_confidence=0.5, **kw_args):
		super().__init__()

		self.decisive_confidence = decisive_confidence

	def forward (self, pred, truth):
		loss = 0
		ground_positive, ground_negative, true_positive, true_negative = 0, 0, 0, 0
		for pred_i, truth_i in zip(pred, truth):
			# skip empty prediction
			if len(pred_i) == 0:
				continue

			# protect NAN in pred
			pred_i = torch.where(torch.isnan(pred_i), torch.zeros_like(pred_i), pred_i)
			pred_i = torch.where(torch.isinf(pred_i), torch.zeros_like(pred_i), pred_i)

			truth_i = truth_i[:len(pred_i)]
			loss += nn.functional.binary_cross_entropy(pred_i, truth_i)

			pred_binary = pred_i > self.decisive_confidence
			target_binary = truth_i > 0

			ground_positive += sum(target_binary)
			ground_negative += sum(~target_binary)
			true_positive += sum(pred_binary & target_binary)
			true_negative += sum(~pred_binary & ~target_binary)

		loss /= len(pred)

		ground_positive = max(ground_positive, 1)
		ground_negative = max(ground_negative, 1)
		accuracy = (true_positive / ground_positive) * (true_negative / ground_negative)

		return loss, accuracy
