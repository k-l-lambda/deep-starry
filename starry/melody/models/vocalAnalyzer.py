
import torch.nn as nn
import torch.nn.functional as F

from ...conformer.modules import Linear
from .conformerU import ConformerEncoderU
from .vocalModules import VocalEncoder



class VocalAnalyzer (nn.Module):
	def __init__ (self, n_class=1, encoder_dim=128, **args):
		super().__init__()

		self.vocalEncoder = VocalEncoder(encoder_dim)
		self.encoder = ConformerEncoderU(encoder_dim=encoder_dim, **args)
		self.fc = Linear(encoder_dim, n_class, bias=False)


	def forward (self, pitch, gain):
		x = self.vocalEncoder(pitch, gain)
		x = self.encoder(x)
		x = self.fc(x)
		x = F.sigmoid(x)

		return x


class VocalAnalyzerLoss (nn.Module):
	def __init__ (self, output_field='head', **args):
		super().__init__()

		self.deducer = VocalAnalyzer(**args)
		self.output_field = output_field


	def forward (self, batch):
		target = batch[self.output_field].unsqueeze(-1)
		pred = self.deducer(batch['pitch'], batch['gain'])

		loss = F.binary_cross_entropy(pred, target)

		acc = ((pred > 0.5).float() == target).float().mean()

		return loss, {'acc': acc}
