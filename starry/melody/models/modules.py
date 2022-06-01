
import torch
import torch.nn as nn
#import torch.nn.functional as F

from ...transformer.layers import EncoderLayer, DecoderLayer



FLOAT32_EPS = 1e-12 #torch.finfo(torch.float32).eps
FLOAT32_MAX = torch.finfo(torch.float32).max


def normalizeL2 (v, dim=-1):
	#return F.normalize(v, dim=dim)
	#return v / torch.linalg.vector_norm(v, dim=dim).clamp(min=FLOAT32_EPS, max=FLOAT32_MAX)
	#return v / torch.norm(v, p=2, dim=dim).clamp(min=FLOAT32_EPS, max=FLOAT32_MAX)
	sq_sum = torch.sum(v * v, dim=dim, keepdim=True)
	norm = torch.sqrt(sq_sum).clamp(min=FLOAT32_EPS, max=FLOAT32_MAX)
	return v / norm


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


	def forward (self, source, target):
		# normalize for inner product
		code_src = normalizeL2(source)	# (n, src_joint, d_model)
		code_tar = normalizeL2(target)	# (n, tar_joint, d_model)

		code_tar = code_tar.transpose(1, 2)
		result = code_src.matmul(code_tar).clamp(min=0, max=FLOAT32_MAX)		# (n, src_joint, tar_joint)
		#result = result.flatten()

		return result, code_src, code_tar
