
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

		vec = torch.zeros((x.shape[0], x.shape[1], x.shape[-1] * 2)) # (batch, seq, d_hid)
		vec[:, :, 0::2] = torch.sin(x)
		vec[:, :, 1::2] = torch.cos(x)

		return vec



class SinusoidEncoderXYY (nn.Module):
	def __init__(self, angle_cycle=1000, d_hid=0x200):
		super().__init__()

		self.encoder = SinusoidEncoder(angle_cycle, d_hid)


	# pos: (batch, seq, 3)
	def forward (self, pos):	# -> (batch, seq, 2, d_hid)
		x = self.encoder(pos[:, :, 0])
		y1 = self.encoder(pos[:, :, 1])
		y2 = self.encoder(pos[:, :, 2])

		return torch.stack((x, y1 + y2), dim=2)
