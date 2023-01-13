
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...transformer.layers import EncoderLayer, DecoderLayer
from ...transformer.sub_layers import MultiHeadAttention, PositionwiseFeedForward
from ...modules.positionEncoder import SinusoidEncoder
from ..notation import PITCH_MAX, PITCH_OCTAVE_MAX, PITCH_OCTAVE_SIZE, VELOCITY_MAX, KEYBOARD_SIZE



FLOAT32_EPS = 1e-12 #torch.finfo(torch.float32).eps
FLOAT32_MAX = torch.finfo(torch.float32).max


def normalizeL2 (v, dim=-1):
	#return F.normalize(v, dim=dim)
	#return v / torch.linalg.vector_norm(v, dim=dim).clamp(min=FLOAT32_EPS, max=FLOAT32_MAX)
	#return v / torch.norm(v, p=2, dim=dim).clamp(min=FLOAT32_EPS, max=FLOAT32_MAX)
	sq_sum = torch.sum(v * v, dim=dim, keepdim=True)
	norm = torch.sqrt(sq_sum).clamp(min=FLOAT32_EPS, max=FLOAT32_MAX)
	return v / norm


class ThinDecoderLayer(nn.Module):
	def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
		super().__init__()

		self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

	def forward(self, dec_input, enc_output, dec_enc_attn_mask=None):
		dec_output, dec_enc_attn = self.enc_attn(dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
		dec_output = self.pos_ffn(dec_output)

		return dec_output, dec_enc_attn


class EncoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, x, mask: Optional[torch.Tensor]=None):	# (n, seq, d_word)
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


	def forward (self, dec_input, enc_output, mask: Optional[torch.Tensor]=None):	# (n, seq, d_word)
		dec_output = dec_input
		for layer in self.layer_stack:
			dec_output, _1, _2 = layer(dec_output, enc_output, mask, mask)

		return dec_output


class ThinDecoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			ThinDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, dec_input, enc_output, mask=None):	# (n, seq, d_word)
		dec_output = dec_input
		for layer in self.layer_stack:
			dec_output, _ = layer(dec_output, enc_output, mask)

		return dec_output


class Encoder (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.stack = EncoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model


	def forward (self, x, mask: Optional[torch.Tensor]=None):
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


	def forward (self, x, enc_output, mask: Optional[torch.Tensor]=None):
		if self.scale_emb:
			x *= self.d_model ** 0.5

		x = self.dropout(x)
		x = self.layer_norm(x)

		x = self.stack(x, enc_output, mask)

		return x


class ThinDecoder (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, scale_emb=False):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.stack = ThinDecoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model


	def forward (self, x, enc_output, mask: Optional[torch.Tensor]=None):
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

		code_tar_trans = code_tar.transpose(-2, -1)
		result = code_src.matmul(code_tar_trans).clamp(min=0, max=1)		# (n, src_joint, tar_joint)
		#result = result.flatten()

		return result, code_src, code_tar


class NoteEncoder (nn.Module):
	def __init__ (self, d_model=128, d_time=64, angle_cycle=100000):
		super().__init__()

		self.time_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_time)
		self.embed = nn.Linear(d_time + PITCH_MAX + 1, d_model)


	# x: (time, pitch, velocity)
	def forward (self, x):
		time, pitch, velocity = x

		vec_time = self.time_encoder(time)	# (n, seq, d_time)
		vec_pitch = F.one_hot(pitch.long(), num_classes=PITCH_MAX).float()	# (n, seq, PITCH_MAX)
		scaler_velocity = (velocity.float() / VELOCITY_MAX).unsqueeze(-1)	# (n, seq, 1)

		x = torch.cat([vec_time, vec_pitch, scaler_velocity], dim=-1)	# (n, seq, d_time + PITCH_MAX + 1)

		return self.embed(x)	# (n, seq, d_model)


def encodePitchByOctave (pitch):
	octave = torch.div(pitch, PITCH_OCTAVE_SIZE).long()
	step = torch.remainder(pitch, PITCH_OCTAVE_SIZE)

	vec_octave = F.one_hot(octave, num_classes=PITCH_OCTAVE_MAX).float()	# (..., PITCH_OCTAVE_MAX)
	vec_step = F.one_hot(step, num_classes=PITCH_OCTAVE_SIZE).float()		# (..., PITCH_OCTAVE_MAX)

	return torch.cat([vec_octave, vec_step], dim=-1)		# (..., PITCH_OCTAVE_MAX + PITCH_OCTAVE_SIZE)


class SoftIndex (nn.Module):
	def __init__ (self, seq_len=64, scale=1):
		super().__init__()

		self.scale = scale

		mtx_diff = torch.diag(torch.ones(seq_len), 0) - torch.diag(torch.ones(seq_len - 1), -1)
		mtx_diff[0, 0] = 0
		mtx_diff = self.mtx_diff[None, :, :]

		mtx_sum = 1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
		mtx_sum = self.mtx_sum[None, :, :]

		self.register_buffer('mtx_diff', mtx_diff, persistent=False)
		self.register_buffer('mtx_sum', mtx_sum, persistent=False)


	# x shape: (n, seq)
	def forward (self, x):
		x = self.mtx_diff.matmul(x[:, :, None])
		x = torch.tanh(x / self.scale)
		x = self.mtx_sum.matmul(x)

		return x[:, :, 0]


class NoteEncoder2 (nn.Module):
	def __init__ (self, d_model=128, d_time=64, angle_cycle=100000, softindex_scale=0):
		super().__init__()

		self.softindex = SoftIndex(scale=softindex_scale) if softindex_scale > 0 else None

		self.time_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_time)
		self.embed = nn.Linear(d_time + PITCH_OCTAVE_MAX + PITCH_OCTAVE_SIZE + 1, d_model)


	# x: (time, pitch, velocity)
	def forward (self, x):
		time, pitch, velocity = x

		if self.softindex is not None:
			time = self.softindex(time)

		vec_time = self.time_encoder(time)	# (n, seq, d_time)
		vec_pitch = encodePitchByOctave(pitch.long())	# (n, seq, PITCH_OCTAVE_MAX + PITCH_OCTAVE_SIZE)
		scaler_velocity = (velocity.float() / VELOCITY_MAX).unsqueeze(-1)	# (n, seq, 1)

		x = torch.cat([vec_time, vec_pitch, scaler_velocity], dim=-1)	# (n, seq, d_time + PITCH_OCTAVE_MAX + PITCH_OCTAVE_SIZE + 1)

		return self.embed(x)	# (n, seq, d_model)

# TODO: relative time encoder?


class TimeGuidEncoder (nn.Module):
	def __init__ (self, d_model=128, d_time=64, angle_cycle=100000):
		super().__init__()

		self.time_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_time)
		self.embed = nn.Linear(d_time, d_model)

		self.d_time = d_time


	# x: (time, mask)
	def forward (self, x):
		time, mask = x

		vec_time = self.time_encoder(time.float())	# (n, seq, d_time)

		if mask is not None:
			vec_time[mask.unsqueeze(-1).repeat(1, 1, self.d_time)] = 0

		return self.embed(vec_time)	# (n, seq, d_model)


class FrameEncoder (nn.Module):
	def __init__ (self, d_model=128, d_time=64, angle_cycle=100000):
		super().__init__()

		self.time_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_time)
		self.embed = nn.Linear(d_time + KEYBOARD_SIZE, d_model)


	# x: (time, frame)
	def forward (self, x):
		time, frame = x

		vec_time = self.time_encoder(time)	# (n, seq, d_time)

		x = torch.cat([vec_time, frame], dim=-1)	# (n, seq, d_time + KEYBOARD_SIZE)

		return self.embed(x)	# (n, seq, d_model)
