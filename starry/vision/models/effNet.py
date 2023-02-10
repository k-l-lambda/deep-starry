
import torch
import torch.nn as nn
import torchvision.models as models



LAST_CHANNELS = {
	'efficientnet_b0': 1280,
	'efficientnet_b1': 1280,
	'efficientnet_b2': 1408,
	'efficientnet_b3': 1536,
	'efficientnet_b4': 1792,
	'efficientnet_b5': 2048,
	'efficientnet_b6': 2304,
	'efficientnet_b7': 2560,
}


class HeadlessEffNet (nn.Module):
	def __init__ (self, backbone='efficientnet_b0', mono_channel=False, **kw_args):
		super().__init__()

		self.mono_channel = mono_channel

		self.last_channel = LAST_CHANNELS[backbone]

		backbone = getattr(models, backbone)(num_classes=1, input_channels=1, **kw_args)
		self.features = backbone.features
		self.avgpool = backbone.avgpool


	def forward (self, x):
		if self.mono_channel:
			x = x.repeat(1, 3, 1, 1)

		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		return x
