
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vocal import PITCH_RANGE, PITCH_SUBDIV



N_PITCH_CLASS = (PITCH_RANGE[1] - PITCH_RANGE[0]) * PITCH_SUBDIV


class VocalEncoder (nn.Module):
	def __init__ (self, d_model=128):
		super().__init__()

		self.embed = nn.Linear(N_PITCH_CLASS + 1, d_model)


	def forward (self, pitch, gain):
		vec_pitch = F.one_hot(pitch.long(), num_classes=N_PITCH_CLASS).float()
		gain = gain.unsqueeze(-1)

		x = torch.cat([vec_pitch, gain], dim=-1)
		x = self.embed(x)

		return x
