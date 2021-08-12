
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List



class HardSwish(nn.Module):
	def __init__(self, inplace=True):
		super(HardSwish, self).__init__()
		self.inplace = inplace
	def forward(self, x):
		return x * (F.relu6(x+3, inplace=self.inplace)) / 6

class HardSigmoid(nn.Module):
	def __init__(self, inplace=True):
		super(HardSigmoid, self).__init__()
		self.inplace = inplace
	def forward(self, x):
		return F.relu6(x+3) / 6


ACT_FNS = {
	'RE':nn.ReLU,
	'LRE':nn.LeakyReLU,
	'RE6': nn.ReLU6,
	'HS': HardSwish,
	'HG': HardSigmoid
}


class ConvBN(nn.Sequential):
	r"""
	Args:
		in_planes input channels
		out_planes output channels
		kernel_size (int or tuple): Size of the convolving kernel
		stride (int or tuple, optional): Stride of the convolution. Default: 1
		groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
		nl
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, nl='RE6'):
		if(isinstance(kernel_size, tuple)):
			padding = ((kernel_size[0] - 1) // 2),((kernel_size[1] - 1) // 2)
		else:
			padding = ((kernel_size - 1) // 2)

		a = ACT_FNS[nl]
		super(ConvBN, self).__init__(
			nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
			nn.BatchNorm2d(out_channels),
			a())


class Squeeze(nn.Module):
	def __init__(self, n_features, reduction=4):
		super(Squeeze, self).__init__()
		if n_features % reduction != 0:
			raise ValueError('n_features must be divisible by reduction (default = 4)')
		self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
		self.nonlin1 = ACT_FNS['RE6']()
		self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
		self.nonlin2 = ACT_FNS['HG']()
		self.gap = nn.AdaptiveAvgPool2d(1)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.gap(x)
		y = y.view(b,c)
		y = self.nonlin1(self.linear1(y))
		y = self.nonlin2(self.linear2(y))
		y = x * y.view(b, c, 1, 1)
		return y


class InvertedResidual(nn.Module):

	def __init__(self, inp, oup, stride, expand_ratio, nl, with_se=False, kernel_size=3):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert (stride in [1, 2])
		hidden_dim = int(round((inp * expand_ratio)))
		self.use_res_connect = ((self.stride == 1) and (inp == oup))
		layers = []
		if (expand_ratio != 1):
				layers.append(ConvBN(inp, hidden_dim, kernel_size=1, nl=nl))
		if with_se:
			layers.extend([ConvBN(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, nl=nl), Squeeze(hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		else:
			layers.extend([ConvBN(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, nl=nl), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return (x + self.conv(x))
		else:
			return self.conv(x)


class FusedIBN(nn.Module):

	def __init__(self, inp, oup, stride, expand_ratio, nl):
		super(FusedIBN, self).__init__()
		self.stride = stride
		assert (stride in [1, 2])
		hidden_dim = int(round((inp * expand_ratio)))
		self.use_res_connect = ((self.stride == 1) and (inp == oup))
		layers = []
		layers.extend([ConvBN(inp, hidden_dim, stride=stride, nl=nl), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return (x + self.conv(x))
		else:
			return self.conv(x)


class TuckIBN(nn.Module):

	def __init__(self, inp, oup, stride, compress_ratio, expand_ratio, nl):
		super(TuckIBN, self).__init__()
		self.stride = stride
		assert (stride in [1, 2])
		hidden_dim = int(round((inp * expand_ratio)))
		self.use_res_connect = ((self.stride == 1) and (inp == oup))
		layers = []
		if (expand_ratio != 1):
			layers.append(ConvBN(inp, hidden_dim, kernel_size=1, nl=nl))
		layers.extend([ConvBN(inp, hidden_dim, stride=stride, nl=nl), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return (x + self.conv(x))
		else:
			return self.conv(x)


class RInvertedResidual(nn.Module):

	def __init__(self, inp, oup, stride, expand_ratio, nl, with_se=False):
		super(RInvertedResidual, self).__init__()
		self.stride = stride
		assert (stride in [1, 2])
		hidden_dim = int(round((inp * expand_ratio)))
		self.use_res_connect = ((self.stride == 1) and (inp == oup))
		layers = []
		if (expand_ratio != 1):
			layers.append(ConvBN(inp, hidden_dim, kernel_size=1, nl=nl))
		if with_se:
			layers.extend([ConvBN(hidden_dim, hidden_dim, stride=stride, nl=nl), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		else:
			layers.extend([ConvBN(hidden_dim, hidden_dim, stride=stride, nl=nl), Squeeze(hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return (x + self.conv(x))
		else:
			return self.conv(x)


class WInvertedResidual(nn.Module):

	def __init__(self, inp, oup, kernel_size, stride, expand_ratio, nl, with_se=False):
		super(WInvertedResidual, self).__init__()
		self.stride = stride
		assert (stride in [1, 2])
		hidden_dim = int(round((inp * expand_ratio)))
		self.use_res_connect = ((self.stride == 1) and (inp == oup))
		layers = []
		if (expand_ratio != 1):
				layers.append(ConvBN(inp, hidden_dim, kernel_size=1, nl=nl))
		if with_se:
			layers.extend([ConvBN(hidden_dim, hidden_dim, kernel_size=kernel_size,stride=stride, groups=hidden_dim, nl=nl), Squeeze(hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		else:
			layers.extend([ConvBN(hidden_dim, hidden_dim, kernel_size=kernel_size,stride=stride, groups=hidden_dim, nl=nl), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return (x + self.conv(x))
		else:
			return self.conv(x)


class WFusedIBN(nn.Module):

	def __init__(self, inp, oup, kernel_size, stride, expand_ratio, nl):
		super(WFusedIBN, self).__init__()
		self.stride = stride
		assert (stride in [1, 2])
		hidden_dim = int(round((inp * expand_ratio)))
		self.use_res_connect = ((self.stride == 1) and (inp == oup))
		layers = []
		layers.extend([ConvBN(inp, hidden_dim, kernel_size=kernel_size, stride=stride, nl=nl), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return (x + self.conv(x))
		else:
			return self.conv(x)


class Residual(nn.Module):
	def __init__(self, numIn, numOut):
		super(Residual, self).__init__()
		self.numIn = numIn
		self.numOut = numOut
		self.bn = nn.BatchNorm2d(self.numIn)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
		self.bn1 = nn.BatchNorm2d(self.numOut // 2)
		self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(self.numOut // 2)
		self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

		if self.numIn != self.numOut:
			self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

	def forward(self, x):
		residual = x
		out = self.bn(x)
		out = self.relu(out)
		out = self.conv1(out)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)

		if self.numIn != self.numOut:
			residual = self.conv4(x)

		return out + residual


class SematicEmbbedBlock(nn.Module):
	def __init__(self, high_in_plane, low_in_plane, out_plane):
		super(SematicEmbbedBlock, self).__init__()
		self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)

	def forward(self, high_x, low_x):
		high_x = self.upsample(self.conv3x3(high_x))
		low_x = self.conv1x1(low_x)
		return high_x * low_x


class HeatMapOut(nn.Module):
	
	def __init__(self,SF, nclass):
		super(HeatMapOut, self).__init__()
		self.seb0 = SematicEmbbedBlock(SF[2], SF[1], SF[1])
		self.seb1 = SematicEmbbedBlock(SF[1], SF[0], SF[0])
		self.heatmap = nn.Conv2d(SF[0],nclass,1)

	def forward(self, x:List[torch.Tensor]):
		x0,x1,x2 = x
		up0 = self.seb0(x2,x1)
		up1 = self.seb1(up0,x0)
		out = self.heatmap(up1)
		return out


# shape,shape/2=>shape/2
class SematicEmbbedBlockScore(nn.Module):
	def __init__(self, high_in_plane, low_in_plane, out_plane):
		super(SematicEmbbedBlockScore, self).__init__()
		self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
		self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)

	def forward(self, high_x, low_x):
		high_x = F.interpolate(self.conv3x3(high_x), scale_factor=0.5, mode='bilinear')
		low_x = self.conv1x1(low_x)
		return high_x * low_x


class HeatMapScore(nn.Module):
	
	def __init__(self,SF, nclass):
		super(HeatMapScore, self).__init__()
		self.seb0 = SematicEmbbedBlockScore(SF[0], SF[1], SF[1])
		self.seb1 = SematicEmbbedBlockScore(SF[1], SF[2], SF[2])
		self.heatmap = nn.Conv2d(SF[2],nclass,1)

	def forward(self, x:List[torch.Tensor]):
		x0,x1,x2 = x
		up0 = self.seb0(x0,x1)
		up1 = self.seb1(up0,x2)
		out = self.heatmap(up1)
		return out


class HeatMapOutOpt(nn.Module):
	
	def __init__(self,SF, nclass):
		super(HeatMapOutOpt, self).__init__()
		self.seb0 = SematicEmbbedBlock(SF[2], SF[1], SF[1])
		self.seb1 = SematicEmbbedBlock(SF[1], SF[0], SF[0])
		self.heatmap = nn.Conv2d(SF[0],nclass,1,bias=False)

	def forward(self, x:List[torch.Tensor]):
		x0,x1,x2 = x
		up0 = self.seb0(x2,x1)
		up1 = self.seb1(up0,x0)
		out = self.heatmap(up1)
		return out


class HeatMapOutMult(nn.Module):
	
	def __init__(self,SF, nclass):
		super(HeatMapOutMult, self).__init__()
		self.up = nn.ModuleList()
		for i in reversed(range(len(SF)-1)):
			self.up.append(SematicEmbbedBlock(SF[i+1], SF[i], SF[i]))
		self.heatmap = nn.Conv2d(SF[0],nclass,1)

	def forward(self, x):
		counter=0
		for i in reversed(range(len(x)-1)):
			x[i] = self.up[counter](x[i+1], x[i])
			counter = counter+1
		out = self.heatmap(x[0])
		return out


class ConIn(nn.Module):

	def __init__(self, SF, channels = 1):
		super(ConIn, self).__init__()
		self.conv = ConvBN(channels, SF[0], nl='RE')
		self.stack_conv_down0 = InvertedResidual(inp=SF[0], oup=SF[0], stride=1, expand_ratio=2, nl='RE6')
		self.stack_conv_down_1 = InvertedResidual(inp=SF[0], oup=SF[1], stride=2, expand_ratio=2, nl='RE6')
		self.stack_conv_down_2 = InvertedResidual(inp=SF[1], oup=SF[2], stride=2, expand_ratio=2, nl='RE6')

	def forward(self, x):
		# type: (Tensor) -> List[Tensor]
		t0 = self.conv(x)
		t0 = self.stack_conv_down0(t0)
		t1 = self.stack_conv_down_1(t0)
		t2 = self.stack_conv_down_2(t1)
		return [t0, t1, t2]


class ZoomInOut(nn.Module):

	def __init__(self, SF, with_se=False):
		super(ZoomInOut, self).__init__()
		self.Pool = nn.MaxPool2d(kernel_size=2)
		self.stack_up_sample = nn.Upsample(scale_factor=(2,2), mode='bilinear')
		i = 0
		self.stack_conv0_e0 = InvertedResidual(inp=SF[i], oup=SF[i], stride=1, expand_ratio=2, nl='RE6', with_se=with_se)
		self.stack_conv0_e1= FusedIBN(inp=SF[i], oup=SF[i], stride=1, expand_ratio=2, nl='RE')
		self.stack_conv0_c = ConvBN(SF[i] + SF[i+1], SF[i], kernel_size=1, nl='LRE')
		i = 1
		self.stack_conv1_e0 = InvertedResidual(inp=SF[i], oup=SF[i], stride=1, expand_ratio=2, nl='RE6', with_se=with_se)
		self.stack_conv1_e1= FusedIBN(inp=SF[i], oup=SF[i], stride=1, expand_ratio=2, nl='RE')
		self.stack_conv1_c = ConvBN(SF[i] + SF[i+1] + SF[i-1], SF[i], kernel_size=1, nl='LRE')
		i = 2
		self.stack_conv2_e0 = InvertedResidual(inp=SF[i], oup=SF[i], stride=1, expand_ratio=2, nl='RE6', with_se=with_se)
		self.stack_conv2_e1= FusedIBN(inp=SF[i], oup=SF[i], stride=1, expand_ratio=2, nl='RE')
		self.stack_conv2_c = ConvBN(SF[i] + SF[i-1], SF[i], kernel_size=1, nl='LRE')

	def forward(self, x:List[torch.Tensor]):
		# type: (List[Tensor]) -> List[Tensor]
		#Forward
		x[0] = self.stack_conv0_e0(x[0])
		x[0] = self.stack_conv0_e1(x[0])
		x[1] = self.stack_conv1_e0(x[1])
		x[1] = self.stack_conv1_e1(x[1])
		x[2] = self.stack_conv2_e0(x[2])
		x[2] = self.stack_conv2_e1(x[2])
		#Merge
		i = 0
		up = self.stack_up_sample(x[i+1])
		x0 = torch.cat([x[i], up], dim=1)
		x0 = self.stack_conv0_c(x0)

		i = 1
		up = self.stack_up_sample(x[i+1])
		down = self.Pool(x[i-1])
		x1 = torch.cat([x[i], up, down], dim=1)
		x1 = self.stack_conv1_c(x1)

		i = 2
		down = self.Pool(x[i-1])
		x2 = torch.cat([x[i], down], dim=1)
		x2 = self.stack_conv2_c(x2)

		x[0] = x[0] + x0
		x[1] = x[1] + x1
		x[2] = x[2] + x2

		return x


class ZoomInOutSeq(nn.Module):
	def __init__(self, depth, stack_filters):
		super().__init__()

		self.depth = depth

		inout = [ZoomInOut(stack_filters, False) for i in range(self.depth)]
		self.zoomInOut = nn.ModuleList(modules = inout)

	def forward(self, x):
		for inout in self.zoomInOut:
			x = inout(x)

		return x
