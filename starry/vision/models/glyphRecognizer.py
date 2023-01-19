
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .effNet import HeadlessEffNet



class GlyphRecognizer (nn.Module):
	def __init__ (self, n_classes=2, size=(32, 32),
		backbones=['efficientnet_b0', 'efficientnet_b0', 'efficientnet_b0'],
		dropout=0.2):
		super().__init__()

		self.crops = [nn.Identity(), *[transforms.CenterCrop((size[0] // (2 ** i), size[1] // (2 ** i))) for i in range(1, len(backbones))]]

		self.backbones = [HeadlessEffNet(backbone=backbone, mono_channel=True) for backbone in backbones]

		lastconv_output_channels = sum(backbone.last_channel for backbone in self.backbones)

		self.classifier = nn.Sequential(
			nn.Dropout(p=dropout, inplace=True),
			nn.Linear(lastconv_output_channels, n_classes),
		)


	# feature: (n, h, w)
	def forward (self, feature):
		xs = []
		for crop, backbone in zip(self.crops, self.backbones):
			x = crop(feature)
			x = backbone(x)
			xs.append(x)

		x = torch.concat(xs, dim=-1)

		x = self.classifier(x)

		return x
