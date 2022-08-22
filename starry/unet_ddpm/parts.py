""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, gn_channels):
		super().__init__()

		mid_channels = out_channels
		if out_channels > in_channels:
			mid_channels = in_channels * 2
		elif out_channels < in_channels:
			mid_channels = in_channels // 2

		self.side = nn.Identity if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
		self.double_conv = nn.Sequential(
			nn.GroupNorm(in_channels // gn_channels, in_channels),
			nn.SiLU(inplace=True),
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.GroupNorm(mid_channels // gn_channels, mid_channels),
			nn.SiLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
		)

	def forward(self, x):
		xi = self.side(x)
		return xi + self.double_conv(x)


class Down(nn.Module):
	def __init__(self, in_channels, out_channels, gn_channels, res_blocks=1):
		super().__init__()

		self.pool_conv = nn.Sequential(
			ResBlock(in_channels, out_channels, gn_channels),
			*(ResBlock(out_channels, out_channels, gn_channels) for _ in range(res_blocks - 1)),
			nn.AvgPool2d(2),
		)

	def forward(self, x):
		return self.pool_conv(x)


class Up(nn.Module):
	def __init__(self, in_channels, out_channels, gn_channels, res_blocks=1, bilinear=True):
		super().__init__()

		self.conv = nn.Sequential(
			ResBlock(in_channels, out_channels, gn_channels),
			*(ResBlock(out_channels, out_channels, gn_channels) for _ in range(res_blocks - 1)),
		)

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)


	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)
