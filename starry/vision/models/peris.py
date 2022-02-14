
import torch
import torch.nn as nn
import torchvision.models as models



class PerisSimple (nn.Module):
	def __init__ (self, backbone='efficientnet_b7', channels=1):
		super().__init__()

		self.backbone = getattr(models, backbone)(num_classes=channels)


	def forward (self, feature):
		return self.backbone(feature)


class PerisSimpleLoss (nn.Module):
	def __init__ (self, labels=['score'], loss='mse_loss', **kwargs):
		super().__init__()

		self.labels = labels
		self.deducer = PerisSimple(channels=len(labels), **kwargs)
		self.loss = getattr(nn.functional, loss)


	def forward (self, feature, target):
		target_tensor = torch.tensor([[ex[label] for label in self.labels] for ex in target], dtype=torch.float32).to(feature.device)
		pred = self.deducer(feature)
		loss = self.loss(pred, target_tensor)
		error = torch.sqrt(loss)

		return loss, {'error': error}
