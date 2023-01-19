
import torch
import torch.nn as nn
import torchvision.models as models



class HeadlessEffNet (nn.Module):
	def __init__ (self, backbone='efficientnet_b0'):
		super().__init__()

		backbone = getattr(models, backbone)(num_classes=1, input_channels=1)

		self.features = backbone.features
		self.avgpool = backbone.avgpool


	def forward (self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		return x
