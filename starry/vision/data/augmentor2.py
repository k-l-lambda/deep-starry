
import os
from collections import OrderedDict
import torch
from torchvision import transforms
import numpy as np

#from ...schp import networks as schp_networks
from .masker import Masker
from .transforms import SizeLimit, SquarePad



SCHP_PRETRAINED = os.getenv('SCHP_PRETRAINED')


class SCHPMasker (torch.nn.Module):
	def __init__(self, num_classes, resize, pretrained=SCHP_PRETRAINED, bg_semantic=0, hardness=1, reverse_p=0.5, skip_p=0):
		super().__init__()

		self.reverse_p = reverse_p
		self.skip_p = skip_p
		self.model = schp_networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

		state_dict = torch.load(pretrained)['state_dict']
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:]  # remove `module.`
			new_state_dict[name] = v
		self.model.load_state_dict(new_state_dict)

		self.model.eval()
		for param in self.model.parameters():
			param.requires_grad = False

		self.masker = Masker(self.model, resize=resize, mask_semantic=bg_semantic, hardness=hardness)


	def forward(self, image, labels):
		if np.random.random() < self.skip_p:
			return image, labels

		reversed = np.random.random() < self.reverse_p

		if reversed:
			labels['score'] = 0

		masked = self.masker.mask(image.squeeze(0), reverse=reversed)
		masked = masked.unsqueeze(0)

		return masked, labels


class Augmentor2:
	def __init__(self, options, device='cpu'):
		trans = []
		self.masker = None

		if options.get('size_limit'):
			trans.append(SizeLimit(**options['size_limit']))
		if options.get('size_fixed'):
			trans.append(SquarePad(padding_mode=options['size_fixed']['padding_mode']))
			trans.append(transforms.Resize(options['size_fixed']['size']))
		if options.get('affine'):
			if options['affine'].get('interpolation'):
				if type(options['affine']['interpolation']) == str:
					options['affine']['interpolation'] = transforms.InterpolationMode[options['affine']['interpolation']]
			trans.append(transforms.RandomAffine(**options['affine']))
		#if options.get('masker'):
		#	self.masker = SCHPMasker(**options['masker'])
		#	self.masker.to(device)

		self.composer = transforms.Compose(trans)


	def augment (self, source, labels):
		source = self.composer(source)

		if self.masker:
			source, labels = self.masker(source, labels)

		return source, labels