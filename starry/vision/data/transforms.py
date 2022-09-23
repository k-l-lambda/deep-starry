
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F


class SquarePad (torch.nn.Module):
	def __init__ (self, padding_mode):
		super().__init__()

		self.padding_mode = padding_mode


	def forward (self, image):
		h, w = image.shape[-2:]
		max_wh = np.max([w, h])
		lp = (max_wh - w) // 2
		rp = max_wh - w - lp
		tp = (max_wh - h) // 2
		bp = max_wh - h - tp
		padding = (lp, tp, rp, bp)

		return F.pad(image, padding, 0, self.padding_mode)


class SizeLimit (torch.nn.Module):
	def __init__ (self, size):
		super().__init__()

		self.limit = size
		self.resizer = transforms.Resize(size - 1, max_size=size, antialias=True)


	def forward (self, image):
		if max(image.shape[2], image.shape[3]) > self.limit:
			return self.resizer(image)

		return image
