
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
		error = torch.sqrt(loss).item()

		return loss, {'error': error}


class PerisBinary (nn.Module):
	def __init__ (self, backbone='efficientnet_b7', channels=1):
		super().__init__()

		self.backbone = getattr(models, backbone)(num_classes=channels)


	def forward (self, feature):
		y = self.backbone(feature)
		return torch.sigmoid(y)


class PerisBinaryLoss (nn.Module):
	def __init__ (self, loss='binary_cross_entropy', **kwargs):
		super().__init__()

		self.deducer = PerisBinary(**kwargs)
		self.loss = getattr(nn.functional, loss)


	def forward (self, batch):
		feature, target = batch
		pred = self.deducer(feature)
		loss = self.loss(pred, target)

		target_binary = target > 0
		pred_binary = pred > 0.5
		acc = (pred_binary == target_binary).float().mean()

		return loss, {'acc': acc.item()}


class PerisClass (nn.Module):
	def __init__ (self, backbone='efficientnet_b7', channels=1):
		super().__init__()

		self.backbone = getattr(models, backbone)(num_classes=channels)


	def forward (self, feature):
		y = self.backbone(feature)
		return torch.nn.functional.softmax(y, dim=-1)


class PerisClassLoss (nn.Module):
	def __init__ (self, loss='cross_entropy', **kwargs):
		super().__init__()

		self.deducer = PerisClass(**kwargs)
		self.loss = getattr(nn.functional, loss)


	def forward (self, batch):
		feature, target = batch
		pred = self.deducer(feature)
		loss = self.loss(pred, target)

		target_cls = torch.argmax(target, dim=-1)
		pred_cls = torch.argmax(pred, dim=-1)
		acc = (target_cls == pred_cls).float().mean()

		return loss, {'acc': acc.item()}
