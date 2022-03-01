
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.positionEncoder import SinusoidEncoderXYY
from ...transformer.layers import EncoderLayer
from ..event_element import FEATURE_DIM, TARGET_DIM, STAFF_MAX, EventElementType



class EncoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, x):	# (n, seq, d_word)
		enc_output = x
		for enc_layer in self.layer_stack:
			enc_output, _ = enc_layer(enc_output)

		return enc_output


class Encoder (nn.Module):
	def __init__ (self, n_trunk_layers, n_jointer_layers, n_rectifier_layers, n_head, d_k, d_v, d_model, d_inner, angle_cycle=1000,
			dropout=0.1, scale_emb=False):
		super().__init__()

		d_position = d_model - EventElementType.MAX - STAFF_MAX - FEATURE_DIM

		self.position_encoder = SinusoidEncoderXYY(angle_cycle=angle_cycle, d_hid=d_position)

		#self.dropout = nn.Dropout(p=dropout)
		#self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.trunk_stack = EncoderLayerStack(n_trunk_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
		self.jointer_stack = EncoderLayerStack(n_jointer_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
		self.rectifier_stack = EncoderLayerStack(n_rectifier_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.rec_out = nn.Linear(d_model, TARGET_DIM)

		self.scale_emb = scale_emb
		self.d_model = d_model


	# etype:		(n, seq)
	# staff:		(n, seq)
	# feature:		(n, seq, FEATURE_DIM)
	# x:			(n, seq)
	# y1:			(n, seq)
	# y2:			(n, seq)
	def forward (self, etype, staff, feature, x, y1, y2):	# (n, seq, d_word), (n, seq, d_word)
		vec_type = F.one_hot(etype, num_classes=EventElementType.MAX)
		vec_staff = F.one_hot(staff, num_classes=STAFF_MAX)
		pos = self.position_encoder(x, y1, y2)

		x = torch.cat([vec_type, vec_staff, feature, pos], dim=-1)

		if self.scale_emb:
			x *= self.d_model ** 0.5

		#x = self.dropout(x)
		#x = self.layer_norm(x)

		x = self.trunk_stack(x)

		rec = self.rectifier_stack(x)
		rec = self.rec_out(x)

		j = self.jointer_stack(x)

		return rec, j


class RectifyJointer (nn.Module):
	pass
