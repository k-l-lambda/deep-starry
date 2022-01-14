
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
	def forward (self, x):
		x = x.unsqueeze(-1).repeat(1, 1, self.band.shape[-1])	# (batch, seq, d_hid / 2)
		x *= self.band

		vec = torch.zeros((x.shape[0], x.shape[1], x.shape[-1] * 2)) # (batch, seq, d_hid)
		vec[:, :, 0::2] = torch.sin(x)
		vec[:, :, 1::2] = torch.cos(x)

		return vec
