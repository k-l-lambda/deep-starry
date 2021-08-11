""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .parts import *



class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=True, depth = 4, init_width = 64):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.depth = depth
		factor = 2 if bilinear else 1

		self.inc = DoubleConv(n_channels, init_width)
		self.outc = OutConv(init_width, n_classes)

		downs = []
		ups = []

		for d in range(depth):
			ic = init_width * (2 ** d)
			oc = ic * 2
			if d == depth - 1:
				oc //= factor
			downs.append(Down(ic, oc))
			#print('down c:', ic, oc)

		for d in range(depth):
			ic = init_width * (2 ** (depth - d))
			oc = ic // 2
			if d < depth - 1:
				oc //= factor
			ups.append(Up(ic, oc, bilinear))
			#print('up c:', ic, oc)

		self.downs = nn.ModuleList(modules = downs)
		self.ups = nn.ModuleList(modules = ups)


	def forward(self, input):
		xs = []
		x = self.inc(input)

		for down in self.downs:
			xs.append(x)
			x = down(x)

		xs.reverse()

		for xi, up in zip(xs, self.ups):
			x = up(x, xi)

		logits = self.outc(x)

		return logits
