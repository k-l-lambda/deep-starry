
from typing import Optional
import torch

from .injection import LoraInjectedLinear
from ..transformer.sub_layers import MultiHeadAttention
from ..transformer.layers import EncoderLayer



class LoraMultiHeadAttention (MultiHeadAttention):
	def __init__(self, r=4, alpha=1., bias=False, n_head=8, d_model=256, d_k=32, d_v=32, dropout=0.1):
		super().__init__(n_head, d_model, d_k, d_v, dropout)

		self.q_lora = LoraInjectedLinear(d_model, n_head * d_k, bias=bias, r=r, dropout_p=dropout, alpha=alpha)
		self.v_lora = LoraInjectedLinear(d_model, n_head * d_k, bias=bias, r=r, dropout_p=dropout, alpha=alpha)


	def initialize (self):
		self.q_lora.initialize()
		self.v_lora.initialize()


	def freezeTrunk (self):
		for module in [self.w_qs, self.w_ks, self.w_vs, self.fc]:
			for p in module.parameters():
				p.requires_grad = False


	# overload
	def train (self, mode=True):
		super().train(mode=False)

		self.training = mode

		self.q_lora.train(mode=mode)
		self.v_lora.train(mode=mode)

		return self


	def forward(self, q, k, v, mask: Optional[torch.Tensor] =None):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

		residual = q

		# Pass through the pre-attention projection: b x lq x (n*dv)
		# Separate different heads: b x lq x n x dv
		q = self.w_qs(q) + self.q_lora(q)
		q = q.view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v) + self.v_lora(v)
		v = v.view(sz_b, len_v, n_head, d_v)

		# Transpose for attention dot product: b x n x lq x dv
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		if mask is not None:
			mask = mask.unsqueeze(1)   # For head axis broadcasting.

		q, attn = self.attention(q, k, v, mask=mask)

		# Transpose to move the head dimension back: b x lq x n x dv
		# Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
		q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
		q = self.dropout(self.fc(q))
		q += residual

		q = self.layer_norm(q)

		return q, attn


class LoraEncoderLayer (EncoderLayer):
	def __init__ (self, r=4, alpha=1., bias=False, d_inner=1024, **kw_args):
		super().__init__(d_inner=d_inner, **kw_args)

		self.slf_attn = LoraMultiHeadAttention(r=r, alpha=alpha, bias=bias, **kw_args)


	#def wrap (self, plain_layer):
	#	self.load_state_dict(plain_layer.state_dict(), strict=False)

	#	return self


	def initialize (self):
		self.slf_attn.initialize()


	def freezeTrunk (self):
		self.slf_attn.freezeTrunk()

		for p in self.pos_ffn.parameters():
			p.requires_grad = False
