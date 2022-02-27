
import torch
from torchvision import transforms



class Masker:
	def __init__(self, model, resize, mask_semantic, reverse=False, flip_RGB=True):
		self.model = model
		self.trans = transforms.Resize(resize)
		#self.blur = transforms.Compose([transforms.GaussianBlur(blur_kernel, sigma=blur_sigma)] * blur_iterations)
		self.softmax = torch.nn.Softmax(dim=0)
		self.mask_semantic = mask_semantic
		self.reverse = reverse
		self.flip_RGB = flip_RGB


	def mask (self, image):
		image_f = torch.flip(image, (0,)) if self.flip_RGB else image
		batched_image = image_f.reshape((1,) + image_f.shape)
		semantics = self.model(self.trans(batched_image))[0][-1][0]
		semantics = self.softmax(semantics * 0.8)

		#blur_image = self.blur(image)
		blur_image = torch.mean(image, dim=(1, 2), keepdim=True)

		bg = semantics[self.mask_semantic]
		bg = bg.reshape((1,) + bg.shape)
		resize = transforms.Resize(image.shape[1:])
		bg = resize(bg)
		fg = 1 - bg
		if self.reverse:
			bg, fg = fg, bg

		inter = blur_image * bg + image * fg

		return inter
