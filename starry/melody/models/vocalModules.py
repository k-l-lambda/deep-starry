
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vocal import PITCH_RANGE, PITCH_SUBDIV
from ...modules.positionEncoder import SinusoidEncoder



N_VOCAL_PITCH_CLASS = (PITCH_RANGE[1] - PITCH_RANGE[0]) * PITCH_SUBDIV
N_MIDI_PITCH_CLASS = PITCH_RANGE[1] - PITCH_RANGE[0]


class VocalEncoder (nn.Module):
	def __init__ (self, d_model=128):
		super().__init__()

		self.embed = nn.Linear(N_VOCAL_PITCH_CLASS + 1, d_model)


	def forward (self, pitch, gain):
		vec_pitch = F.one_hot(pitch.long(), num_classes=N_VOCAL_PITCH_CLASS).float()
		gain = gain.unsqueeze(-1)

		x = torch.cat([vec_pitch, gain], dim=-1)
		x = self.embed(x)

		return x


class MidiEncoder (nn.Module):
	def __init__ (self, d_model=128, d_time=256, angle_cycle=1e+5):
		super().__init__()

		self.embed = nn.Linear(d_time + N_MIDI_PITCH_CLASS + 1, d_model)
		self.time_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_time)


	def forward (self, pitch, tick):
		vec_time = self.time_encoder(tick)	# (n, seq, d_time)
		vec_pitch = F.one_hot(pitch.long(), num_classes=N_MIDI_PITCH_CLASS).float()
		tick = tick.unsqueeze(-1)

		x = torch.cat([vec_time, vec_pitch, tick], dim=-1)
		x = self.embed(x)

		return x
