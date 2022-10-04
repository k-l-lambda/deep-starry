
import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from transformers.models.clip.configuration_clip import CLIPVisionConfig



class ClipVisionBinary (nn.Module):
	def __init__ (self, channels, clip_config):
		super().__init__()

		self.backbone = CLIPVisionTransformer(CLIPVisionConfig(**clip_config))
		self.projection = nn.Linear(clip_config['hidden_size'], channels, bias=False)


	def forward (self, feature):
		y = self.backbone(feature).pooler_output
		y = self.projection(y)

		return torch.sigmoid(y)


class ClipVisionBinaryLoss (nn.Module):
	def __init__ (self, loss='binary_cross_entropy', **kwargs):
		super().__init__()

		self.deducer = ClipVisionBinary(**kwargs)
		self.loss = getattr(nn.functional, loss)


	def forward (self, batch):
		feature, target = batch
		pred = self.deducer(feature)
		loss = self.loss(pred, target)

		target_binary = target > 0
		pred_binary = pred > 0.5
		acc = (pred_binary == target_binary).float().mean()

		return loss, {'acc': acc.item()}



class ClipVisionClass (nn.Module):
	def __init__ (self, channels, clip_config):
		super().__init__()

		self.backbone = CLIPVisionTransformer(CLIPVisionConfig(**clip_config))
		self.projection = nn.Linear(clip_config['hidden_size'], channels, bias=False)


	def forward (self, feature):
		y = self.backbone(feature).pooler_output
		y = self.projection(y)

		return torch.nn.functional.softmax(y, dim=-1)


class ClipVisionClassLoss (nn.Module):
	def __init__ (self, loss='cross_entropy', **kwargs):
		super().__init__()

		self.deducer = ClipVisionClass(**kwargs)
		self.loss = getattr(nn.functional, loss)


	def forward (self, batch):
		feature, target = batch
		pred = self.deducer(feature)
		loss = self.loss(pred, target)

		target_cls = torch.argmax(target, dim=-1)
		pred_cls = torch.argmax(pred, dim=-1)
		acc = (target_cls == pred_cls).float().mean()

		return loss, {'acc': acc.item()}
