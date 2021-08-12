
import torch
import torch.nn as nn

from ..unet import UNet



class ScoreWidgets (nn.Module):
	def __init__ (self, in_channels, out_channels, mask, backbone, freeze_mask, mask_channels = 2):
		super().__init__()

		if mask['type'] == 'unet':
			depth = mask['unet_depth']
			init_width = mask['unet_init_width']
			self.mask = UNet(in_channels, mask_channels, depth = depth, init_width = init_width)

		trunk_channels = in_channels + mask_channels

		if backbone['type'] == 'unet':
			depth = backbone['unet_depth']
			init_width = backbone['unet_init_width']
			self.backbone = UNet(trunk_channels, out_channels, depth = depth, init_width = init_width)

		self.freeze_mask = freeze_mask
		if self.freeze_mask:
			for param in self.mask.parameters():
				param.requires_grad = False


	def forward (self, x):
		mask = self.mask(x)
		x = torch.cat([x, mask], dim = 1)
		x = self.backbone(x)

		return torch.sigmoid(x)


	# overload
	def state_dict (self, destination=None, prefix='', keep_vars=False):
		return {
			'mask': self.mask.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
			'backbone': self.backbone.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
		}


	# overload
	def load_state_dict (self, state_dict):
		if state_dict.get('mask'):
			self.mask.load_state_dict(state_dict['mask'])

		if state_dict.get('backbone'):
			self.backbone.load_state_dict(state_dict['backbone'])


	# overload
	def train (self, mode=True):
		self.mask.train(mode and not self.freeze_mask)
		self.backbone.train(mode)


class ScoreWidgetsMask (ScoreWidgets):
	def __init__ (self, **kw_args):
		super().__init__(**kw_args)


	# overload
	def state_dict (self, destination=None, prefix='', keep_vars=False):
		return {
			'mask': self.mask.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
		}


	def forward (self, x):
		x = self.mask(x)
		x = torch.sigmoid(x)
		return x


class ScoreWidgetsInspection (ScoreWidgets):
	def __init__ (self, **kw_args):
		super().__init__(**kw_args)


	def forward (self, x):
		mask = self.mask(x)
		x = torch.cat([x, mask], dim = 1)
		x = self.backbone(x)

		mask = torch.sigmoid(mask)
		result = torch.sigmoid(x)

		return (mask, result)
