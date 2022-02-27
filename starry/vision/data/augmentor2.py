
import torch
from torchvision import transforms



class SizeLimit (torch.nn.Module):
	def __init__(self, size):
		super().__init__()

		self.limit = size
		self.resizer = transforms.Resize(size - 1, max_size=size, antialias=True)


	def forward(self, image):
		print('image:', image.shape, self.limit)
		if max(image.shape[2], image.shape[3]) > self.limit:
			return self.resizer(image)

		return image


class Augmentor2:
	def __init__(self, options):
		trans = []

		if options.get('size_limit'):
			trans.append(SizeLimit(**options['size_limit']))
		if options.get('affine'):
			trans.append(transforms.RandomAffine(**options['affine']))

		self.composer = transforms.Compose(trans)


	def augment (self, source):
		return self.composer(source)
