
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...unet import UNet
from ...unet_ddpm import UNetModel



class ScoreRegression (nn.Module):
	def __init__ (self, out_channels, backbone, in_channels=1, **_):
		super().__init__()

		if backbone['type'] == 'unet':
			depth = backbone['unet_depth']
			init_width = backbone['unet_init_width']
			self.backbone = UNet(in_channels, out_channels, depth=depth, init_width=init_width)
		elif backbone['type'] == 'unet_ddpm':
			self.backbone = UNetModel(in_channels=in_channels, out_channels=out_channels, **backbone['args'])


	def forward (self, input):
		x = self.backbone(input)

		return x


class ScoreRegressionLoss (nn.Module):
	def __init__ (self, with_mask=False, loss_gradient0=0, channel_weights=[1, 1], loss_func='mse_loss', init_param=True, **kw_args):
		super().__init__()

		self.with_mask = with_mask
		self.loss_gradient0 = loss_gradient0
		self.loss_func = getattr(F, loss_func)

		self.register_buffer('channel_weights', torch.Tensor(channel_weights).view((1, 2, 1, 1)), persistent=False)

		self.deducer = ScoreRegression(**kw_args)

		if init_param:
			# initial parameters
			for param in self.deducer.parameters():
				if param.dim() > 1:
					nn.init.xavier_uniform_(param)


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())


	def loss_mask (self, pred, target, mask=None):
		if mask is None:
			return self.loss_func(pred, target)

		loss_map = self.loss_func(pred, target, reduction='none')

		return (loss_map * mask).sum() / mask.sum()


	def gradientLoss (self, pred, target, mask=None):
		gradient_px = pred[:, :, :, 1:] - pred[:, :, :, :-1]
		gradient_tx = target[:, :, :, 1:] - target[:, :, :, :-1]
		loss = self.loss_mask(gradient_px, gradient_tx, mask[:, :, :, :-1] if mask is not None else None)

		gradient_py = pred[:, :, 1:, :] - pred[:, :, :-1, :]
		gradient_ty = target[:, :, 1:, :] - target[:, :, :-1, :]
		loss += self.loss_mask(gradient_py, gradient_ty, mask[:, :, :-1, :] if mask is not None else None)

		return loss


	def forward (self, batch):
		input, target = batch

		pred = self.deducer(input) * self.channel_weights
		tar = target[:, :2, :, :] * self.channel_weights

		mask = None
		loss = 0
		if self.with_mask:
			mask = target[:, 2:, :, :]
			loss = self.loss_mask(pred, tar, mask)
		else:
			loss = self.loss_mask(pred, tar)

		if self.loss_gradient0 > 0:
			pred0 = pred[:, 0:1, :, :]
			targ0 = target[:, 0:1, :, :]
			loss += self.gradientLoss(pred0, targ0, mask=mask) * self.loss_gradient0 * self.loss_gradient0

		metric = {'loss': loss.item()}

		if not self.training:
			if mask is None:
				mask = torch.ones_like(pred).to(pred.device)

			mask_sum = mask.sum()

			metric['y_error'] = (((pred[:, 0] - tar[:, 0]).abs() * mask / self.channel_weights[:, 0]).sum() / mask_sum).item()
			metric['k_error'] = (((pred[:, 1] - tar[:, 1]).abs() * mask / self.channel_weights[:, 1]).sum() / mask_sum).item()

		return loss, metric
