
#import torch
import torch.nn as nn
import torch.nn.functional as F

from ...transformer.layers import EncoderLayer, DecoderLayer



class EncoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, x, mask=None):	# (n, seq, d_word)
		enc_output = x
		for enc_layer in self.layer_stack:
			enc_output, _ = enc_layer(enc_output, mask)

		return enc_output


class DecoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, dec_input, enc_output, mask=None):	# (n, seq, d_word)
		dec_output = dec_input
		for layer in self.layer_stack:
			dec_output, _1, _2 = layer(dec_output, enc_output, mask, mask)

		return dec_output


class Encoder (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.stack = EncoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model


	def forward (self, x, mask=None):
		if self.scale_emb:
			x *= self.d_model ** 0.5

		x = self.dropout(x)
		x = self.layer_norm(x)

		x = self.stack(x, mask)

		return x


class Decoder (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.stack = DecoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model


	def forward (self, x, enc_output, mask=None):
		if self.scale_emb:
			x *= self.d_model ** 0.5

		x = self.dropout(x)
		x = self.layer_norm(x)

		x = self.stack(x, enc_output, mask)

		return x


class Jointer (nn.Module):
	def __init__ (self, d_model):
		super().__init__()

		self.d_model = d_model


	def forward (self, source, target, mask_src, mask_tar):
		# normalize for inner product
		source = F.normalize(source, dim=-1)
		target = F.normalize(target, dim=-1)

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
			result = result.flatten()

			results.append(result)

		return results
