
import torch
import torch.nn as nn
import torchvision.models as models



class PerisSimple (nn.Module):
	def __init__ (self, backbone='efficientnet_b7', channels=1):
		super().__init__()

		self.backbone = getattr(models, backbone)(num_classes=channels)


	def forward (self, feature):
		y = self.backbone(feature)
		return torch.log(torch.relu(y) + 1e-9)


class PerisSimpleLoss (nn.Module):
	def __init__ (self, loss='mse_loss', **kwargs):
		super().__init__()

		self.deducer = PerisSimple(**kwargs)
		self.loss = getattr(nn.functional, loss)


	def forward (self, batch):
		feature, target = batch
		pred = self.deducer(feature)
		loss = self.loss(pred, target)
		error = torch.sqrt(loss)

		return loss, {'error': error}
