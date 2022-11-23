
import math
import torch
import torch.nn as nn

from ...unet import UNet
from ..score_semantic import ScoreSemanticDual
from ..contours import Compounder



class ScoreResidueBlockUnet (nn.Module):
	def __init__ (self, in_channels=1, out_channels=3, depth=4, init_width=64):
		super().__init__()

		self.backbone = UNet(in_channels, out_channels, depth=depth, init_width=init_width)

	def forwardBase (self, input):
		x = self.backbone(input)

		return torch.sigmoid(x)

	def forwardRes (self, input, prev):
		x = torch.cat([input, prev], dim = 1)
		x = self.backbone(x)

		return torch.sigmoid(prev + x)


class ScoreResidueBlockUnetBase (ScoreResidueBlockUnet):
	def __init__ (self, **args):
		super().__init__(**args)

	def forward (self, input):
		return self.forwardBase(input)


class ScoreResidueBlockUnetRes (ScoreResidueBlockUnet):
	def __init__ (self, **args):
		super().__init__(**args)

	def forward (self, input, prev):
		return self.forwardRes(input, prev)


class ScoreResidueU (nn.Module):
	def __init__ (self, in_channels, out_channels, residue_blocks,
		base_depth, base_init_width, residue_depth=4, residue_init_width=64,
		freeze_base=False, frozen_res=0, **args):
		super().__init__()

		self.freeze_base = freeze_base
		self.frozen_res = frozen_res

		self.base_block = ScoreResidueBlockUnetBase(in_channels=in_channels, out_channels=out_channels, depth=base_depth, init_width=base_init_width)
		self.res_blocks = nn.ModuleList(modules=[
			ScoreResidueBlockUnetRes(in_channels=in_channels + out_channels, out_channels=out_channels, depth=residue_depth, init_width=residue_init_width)
				for i in range(residue_blocks)
		])

		if self.freeze_base:
			for param in self.base_block.parameters():
				param.requires_grad = False

		for l in range(self.frozen_res):
			for param in self.res_blocks[l].parameters():
				param.requires_grad = False

		self.no_overwrite = False

	def forward (self, input):
		x = self.base_block(input)

		for block in self.res_blocks:
			x = block(input, x)

		return x

	# overload
	def state_dict (self, destination=None, prefix='', keep_vars=False):
		if self.no_overwrite:
			return super().state_dict(destination, prefix, keep_vars)

		return {
			'base': self.base_block.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
			'res': [block.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars) for block in self.res_blocks],
		}

	# overload
	def load_state_dict (self, state_dict):
		if state_dict.get('base'):
			self.base_block.load_state_dict(state_dict['base'])

		if state_dict.get('res'):
			res = state_dict['res']
			for block, state in zip(self.res_blocks, res):
				block.load_state_dict(state)

			# load rest blocks from last res weights
			if len(res) > 0 and len(res) < len(self.res_blocks):
				for i in range(len(res), len(self.res_blocks)):
					self.res_blocks[i].load_state_dict(res[-1])

	# overload
	def train (self, mode=True):
		self.base_block.train(mode and not self.freeze_base)

		for i, block in enumerate(self.res_blocks):
			frozen = i < self.frozen_res
			block.train(mode and not frozen)


class ScoreResidueULoss (nn.Module):
	def __init__(self, compounder, out_channels=3, channel_weights=None, **kw_args):
		super().__init__()

		channel_weights = torch.Tensor(channel_weights) if channel_weights else torch.ones((out_channels))
		self.register_buffer('channel_weights', channel_weights.view((1, out_channels, 1, 1)), persistent=False)

		self.deducer = ScoreResidueU(out_channels=out_channels, **kw_args)
		self.compounder = Compounder(compounder)

	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())

	def forward (self, batch):
		feature, target = batch
		pred = self.deducer(feature)
		loss = nn.functional.binary_cross_entropy(pred, target, weight=self.channel_weights)

		metric = {'acc': -math.log(loss.item())}

		if not self.training:
			compound_pred = self.compounder.compound(pred)
			compound_target = self.compounder.compound(target)
			metric['semantic'] = ScoreSemanticDual.create(self.compounder.labels, 1, compound_pred, compound_target)

		return loss, metric


	def stat (self, metrics, n_batch):
		result = {
			'acc': metrics['acc'] / n_batch,
		}

		semantic = metrics.get('semantic')
		if semantic is not None:
			stats = metrics['semantic'].stat()
			#self.stats = stats

			result['contour'] = stats['accuracy']

		return result


class ScoreResidueUInspection (ScoreResidueU):
	def __init__ (self, **kw_args):
		super().__init__(**kw_args)

	def forward (self, input):
		x = self.base_block(input)
		outputs = [x]

		for block in self.res_blocks:
			x = block((input, x))
			outputs.append(x)

		return outputs
