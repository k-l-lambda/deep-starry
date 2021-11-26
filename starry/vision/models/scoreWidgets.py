
import math
import torch
import torch.nn as nn

from ...unet import UNet
from ..score_semantic import ScoreSemanticDual



class ScoreWidgets (nn.Module):
	def __init__ (self, in_channels, out_channels, mask, backbone, freeze_mask, mask_channels=2, wide_mask=False, **kw_args):
		super().__init__()

		mask_out_channels = mask_channels
		if mask['type'] == 'unet':
			depth = mask['unet_depth']
			init_width = mask['unet_init_width']
			self.mask = UNet(in_channels, mask_channels, depth=depth, init_width=init_width, classify_out=not wide_mask)

			if wide_mask:
				mask_out_channels = init_width

		trunk_channels = in_channels + mask_out_channels

		if backbone['type'] == 'unet':
			depth = backbone['unet_depth']
			init_width = backbone['unet_init_width']
			self.backbone = UNet(trunk_channels, out_channels, depth=depth, init_width=init_width)
		else:
			self.backbone = None

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

		if self.backbone:
			self.backbone.train(mode)


class ScoreWidgetsLoss (nn.Module):
	need_states = True


	def __init__(self, labels, unit_size, out_channels, metric_quota=float('inf'), channel_weights_rate=1e-4, clip_margin=12, **kw_args):
		super().__init__()

		self.labels = labels
		self.unit_size = unit_size
		self.channel_weights_rate = channel_weights_rate
		self.clip_margin = clip_margin
		self.metric_quota = metric_quota

		self.deducer = ScoreWidgets(out_channels=out_channels, **kw_args)

		self.channel_weights = torch.ones(out_channels)
		self.channel_weights_target = torch.ones(out_channels)
		self.metric_cost = 0


	def forward (self, batch):
		feature, target = batch
		pred = self.deducer(feature)

		weights = self.channel_weights.reshape((1, -1, 1, 1)).to(feature.device)

		target = target[:, :, :, self.clip_margin:-self.clip_margin]
		pred = pred[:, :, :, self.clip_margin:-self.clip_margin]

		loss = nn.functional.binary_cross_entropy(pred, target, weight=weights)

		metric = {'bce': loss}

		# update channel weights
		if self.training:
			self.channel_weights = self.channel_weights * (1 - self.channel_weights_rate) + self.channel_weights_target * self.channel_weights_rate
		else:
			if self.metric_cost < self.metric_quota:
				metric['semantic'] = ScoreSemanticDual.create(self.labels, self.unit_size, pred, target)

				cost = (metric['semantic'].points_count - metric['semantic'].true_count) / max(metric['semantic'].true_count, 1)
				cost = (cost ** 1.2) * feature.shape[0]
				self.metric_cost += cost
				#print('metric_cost:', cost, self.metric_cost)

		return loss, metric


	def stat (self, metrics, n_batch):
		result = {
			'bce': metrics['bce'] / n_batch,
		}

		semantic = metrics.get('semantic')
		if semantic is not None:
			stats = metrics['semantic'].stat()
			self.stats = stats

			result['contour'] = stats['accuracy']
			result['channel_weights'] = dict([(label, self.channel_weights[i].item()) for i, label in enumerate(self.labels)])
			result['channel_weights_target'] = dict([(label, self.channel_weights_target[i].item()) for i, label in enumerate(self.labels)])

			details = stats['details']
			result['channel_error'] = dict([(label, 1 - details[label]['feasibility']) for label in self.labels])

			#print('result:', result['channel_error'])

		return result


	def updateStates (self):
		# update channel weights target
		wws = self.stats['loss_weights'] ** 2
		ww_sum = max(wws.sum(), 1e-9)
		self.channel_weights_target = torch.tensor(wws * len(wws) / ww_sum)
		self.metric_cost = 0


	def state_dict (self, destination=None, prefix='', keep_vars=False):
		return {
			'channel_weights': self.channel_weights,
			'channel_weights_target': self.channel_weights_target,
		}


	def load_state_dict (self, state_dict):
		if state_dict.get('channel_weights') is not None:
			self.channel_weights = state_dict['channel_weights'].cpu()
		if state_dict.get('channel_weights_target') is not None:
			self.channel_weights_target = state_dict['channel_weights_target'].cpu()


class ScoreWidgetsMask (ScoreWidgets):
	def __init__ (self, **kw_args):
		super().__init__(out_channels=1, freeze_mask=False, **kw_args)


	# overload
	def state_dict (self, destination=None, prefix='', keep_vars=False):
		return {
			'mask': self.mask.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
		}


	def forward (self, x):
		x = self.mask(x)
		x = torch.sigmoid(x)
		return x


class ScoreWidgetsMaskLoss (nn.Module):
	def __init__(self, **kw_args):
		super().__init__()

		self.deducer = ScoreWidgetsMask(**kw_args)

	def forward (self, batch):
		feature, target = batch
		pred = self.deducer(feature)
		loss = nn.functional.binary_cross_entropy(pred, target)

		return loss, {
			'acc': -math.log(loss),
		}


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
