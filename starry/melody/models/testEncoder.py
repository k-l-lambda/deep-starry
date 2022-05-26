
import torch.nn as nn

from ...transformer.layers import EncoderLayer



class TestEncoder (nn.Module):
	def __init__(self, d_model):
		super().__init__()

		self.layer = EncoderLayer(d_model, 4, 4, 4, 4)


	def forward(self, enc_input):
		out, _ = self.layer(enc_input)

		return out
