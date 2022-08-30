
import torch.nn as nn
import torch.nn.functional as F

from ...unet_ddpm import UNetModel



class Reconstructor (nn.Module):
	def __init__(self, channels=3, resolution_factor=4, layers=4, num_res_blocks=2, dropout=0.1):
		super().__init__()

		self.resolution_factor = resolution_factor

		channel_mult = tuple(2 ** i for i in range(layers))

		self.backbone = UNetModel(
			in_channels=channels,
			out_channels=channels,
			model_channels=32,
			num_res_blocks=num_res_blocks,
			attention_resolutions=[channel_mult[-1]],
			channel_mult=channel_mult,
			dropout=dropout,
		)


	def forward (self, x):
		_, _, h, w = x.shape
		f = self.resolution_factor

		x = F.interpolate(x, (h * f, w * f), mode='bilinear')

		return self.backbone(x)


class ReconstructorLoss (nn.Module):
	def __init__ (self, **kwargs):
		super().__init__()

		self.deducer = Reconstructor(**kwargs)


	def forward (self, batch):
		x, y = batch
		y_ = self.deducer(x)

		loss = F.mse_loss(y_, y)

		return loss, {}
