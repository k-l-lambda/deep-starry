
import torch
import torch.nn as nn

from ...unet import UNet



class ScoreRegression (nn.Module):
	def __init__ (self, out_channels, backbone, in_channels=1, width_mask=False, loss_gradient0=0, channel_weights=[1, 1], use_sigmoid=False):
		super().__init__()

		self.with_mask = width_mask
		self.loss_gradient0 = loss_gradient0

		self.channel_weights = torch.Tensor(channel_weights).reshape((1, 2, 1, 1))

		self.use_sigmoid = use_sigmoid

		if backbone['type'] == 'unet':
			depth = backbone['unet_depth']
			init_width = backbone['unet_init_width']
			self.backbone = UNet(in_channels, out_channels, depth=depth, init_width=init_width)


	def forward (self, input):
		x = self.backbone(input)

		if self.use_sigmoid:
			x = torch.sigmoid(x)

		return x


	# overload
	def to (self, device):
		self.channel_weights.to(device)

		return super().to(device)


	'''@classmethod
	def gradientLoss (cls, pred, target, loss_func, mask = None):
		gradient_px = pred[:, :, :, 1:] - pred[:, :, :, :-1]
		gradient_tx = target[:, :, :, 1:] - target[:, :, :, :-1]
		loss = loss_func(gradient_px, gradient_tx, mask[:, :, :, :-1]) if mask is not None else loss_func(gradient_px, gradient_tx)

		gradient_py = pred[:, :, 1:, :] - pred[:, :, :-1, :]
		gradient_ty = target[:, :, 1:, :] - target[:, :, :-1, :]
		loss += loss_func(gradient_py, gradient_ty, mask[:, :, :-1, :]) if mask is not None else loss_func(gradient_py, gradient_ty)

		return loss


	def loss (self, input, target, loss_func):
		pred = self.forward(input)
		tar = target
		if not self.use_sigmoid:
			pred *= self.channel_weights
			tar = target[:, :2, :, :] * self.channel_weights

		mask = None
		loss = 0
		if self.with_mask:
			mask = target[:, 2:, :, :]
			loss = loss_func(pred, tar, mask)
		else:
			loss = loss_func(pred, tar)

		if self.loss_gradient0 > 0:
			pred0 = pred[:, 0:1, :, :]
			targ0 = target[:, 0:1, :, :]
			loss += self.gradientLoss(pred0, targ0, loss_func, mask = mask) * self.loss_gradient0 * self.loss_gradient0

		return loss


	def validation_loss (self, input, target, loss_func):
		return self.loss(input, target, loss_func)'''
