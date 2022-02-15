
import torch
import torch.nn as nn
import numpy as np



class SinusoidEncoder (nn.Module):
	def __init__(self, angle_cycle=1000, d_hid=0x200):
		super().__init__()

		band = torch.arange(0, d_hid // 2) * (-2. / d_hid)
		band = torch.pow(angle_cycle * 2 * np.pi, band).unsqueeze(0).unsqueeze(0) # (1, 1, d_hid / 2)
		self.register_buffer('band', band, persistent=False)


	# x: (batch, seq)
	def forward (self, x): # -> (batch, seq, d_hid)
		x = x.unsqueeze(-1).repeat(1, 1, self.band.shape[-1])	# (batch, seq, d_hid / 2)
		x *= self.band

		vec = torch.zeros((x.shape[0], x.shape[1], x.shape[-1] * 2), device=x.device) # (batch, seq, d_hid)
		vec[:, :, 0::2] = torch.sin(x)
		vec[:, :, 1::2] = torch.cos(x)

		return vec



class SinusoidEncoderXYY (nn.Module):
	def __init__(self, angle_cycle=1000, d_hid=0x200):
		super().__init__()

		self.encoder = SinusoidEncoder(angle_cycle, d_hid // 2)


	# pos: (batch, seq, 3)
	def forward (self, x, y1, y2):	# -> (batch, seq, d_hid)
		x = self.encoder(x)
		y1 = self.encoder(y1)
		y2 = self.encoder(y2)

		return torch.cat((x, y1 + y2), dim=2)
