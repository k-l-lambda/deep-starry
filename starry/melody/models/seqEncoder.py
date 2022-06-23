
import torch.nn as nn

from .modules import NoteEncoder2



class SequenceEncoder (nn.Module):
	def __init__ (self, ranges=[4, 8, 16, 32, 64], d_model=128, d_time=64, angle_cycle=100000):
		super().__init__()

		self.ranges = ranges
		self.n_seq = ranges[-1]

		self.note_encoder = NoteEncoder2(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)


	def loadFromMatchJointer (self, mj):
		self.encoder = mj.encoder


	# seq_t: 		(n, seq)	float32
	# seq_p: 		(n, seq)	int8
	# seq_v: 		(n, seq)	int8
	def forward (self, seq_t, seq_p, seq_v):	# (n, seq, n_ranges, d_model)
		##endpoints = endpoints[endpoints > 0].long()
		vec = self.note_encoder((seq_t, seq_p, seq_v))
		vec = self.encoder(vec)

		return vec
