
import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionTransformer



class ClipVisionBinary (nn.Module):
	def __init__ (self, channels, clip_config):
		super().__init__()

		self.backbone = CLIPVisionTransformer(clip_config)
		self.projection = nn.Linear(clip_config['hidden_size'], channels, bias=False)


	def forward (self, feature):
		y = self.backbone(feature)
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
