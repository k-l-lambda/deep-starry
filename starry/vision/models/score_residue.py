
import torch
import torch.nn as nn

from ...flu_net import HeatMapOut, ConIn, ZoomInOutSeq



class ScoreResidueBlock (nn.Module):
	def __init__ (self, in_channels=1, out_channels=3, depth=2, stack_filters=[8, 16, 24]):
		super().__init__()

		self.conIn = ConIn(stack_filters, channels=in_channels)
		self.zoom = ZoomInOutSeq(depth, stack_filters)
		self.heatOut = HeatMapOut(stack_filters, out_channels)

	def forwardBase (self, input):
		x = self.conIn(input)
		x = self.zoom(x)
		x = self.heatOut(x)

		return torch.sigmoid(x)

	def forwardRes (self, xs):
		input, prev = xs

		x = torch.cat([input, prev], dim = 1)
		x = self.conIn(x)
		x = self.zoom(x)
		x = self.heatOut(x)

		return torch.sigmoid(input + x)


class ScoreResidueBlockBase (ScoreResidueBlock):
	def __init__ (self, **args):
		super().__init__(**args)

	def forward (self, input):
		return self.forwardBase(input)


class ScoreResidueBlockRes (ScoreResidueBlock):
	def __init__ (self, **args):
		super().__init__(**args)

	def forward (self, input):
		return self.forwardRes(input)


class ScoreResidue (nn.Module):
	def __init__ (self, in_channels, out_channels, residue_blocks,
		base_depth, base_stack_filters, residue_depth=1, residue_stack_filters=[8, 16, 24],
		freeze_base=False, frozen_res=0, **args):
		super().__init__()

		self.freeze_base = freeze_base
		self.frozen_res = frozen_res

		self.base_block = ScoreResidueBlockBase(in_channels=in_channels, out_channels=out_channels, depth=base_depth, stack_filters=base_stack_filters)
		self.res_blocks = nn.ModuleList(modules=[
			ScoreResidueBlockRes(in_channels=in_channels + out_channels, out_channels=out_channels, depth=residue_depth, stack_filters=residue_stack_filters)
				for i in range(residue_blocks)
		])

		if self.freeze_base:
			for param in self.base_block.parameters():
				param.requires_grad = False

		for l in range(self.frozen_res):
			for param in self.res_blocks[l].parameters():
				param.requires_grad = False

	def forward (self, input):
		x = self.base_block(input)

		for block in self.res_blocks:
			x = block((input, x))

		return x

	'''def loss (self, input, target, loss_func):
		x = self.base_block(input)
		loss = loss_func(x, target)

		for block in self.res_blocks:
			x = block((input, x))
			loss += loss_func(x, target)

		return loss'''

	# overload
	def state_dict (self, destination=None, prefix='', keep_vars=False):
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


class ScoreResidueInspection (ScoreResidue):
	def __init__ (self, **kw_args):
		super().__init__(**kw_args)

	def forward (self, input):
		x = self.base_block(input)
		outputs = [x]

		for block in self.res_blocks:
			x = block((input, x))
			outputs.append(x)

		return outputs
