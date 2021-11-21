
import os
import torch
from collections import OrderedDict

from .networks_stylegan2 import Discriminator as SG2Discriminator
from ...schp import networks as schp_networks



SCHP_CLASSES = int(os.environ.get('SCHP_CLASSES', 18))
SCHP_PRETRAINED = os.environ['SCHP_PRETRAINED']


class HumanDiscriminator(torch.nn.Module):
	def __init__(self, img_resolution, img_channels, **args):
		super().__init__()

		self.human_parser = schp_networks.init_model('resnet101', num_classes=SCHP_CLASSES, pretrained=None)
		HumanDiscriminator.loadHumanParser(self.human_parser, SCHP_PRETRAINED)

		self.normal_d = SG2Discriminator(img_resolution=img_resolution, img_channels=img_channels, **args)
		self.human_d = SG2Discriminator(img_resolution=img_resolution // 4, img_channels=SCHP_CLASSES, **args)


	@staticmethod
	def loadHumanParser (module, weights_path):
		# load schp weights
		state_dict = torch.load(weights_path)['state_dict']
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:]  # remove `module.`
			new_state_dict[name] = v
		module.load_state_dict(new_state_dict)

		module.eval()
		for param in module.parameters():
			param.requires_grad = False


	def forward(self, img, c, update_emas=False, **block_kwargs):
		with torch.no_grad():
			schp_out = self.human_parser(img)
		schp_semantic = schp_out[0][-1]

		x1 = self.normal_d(img, c, update_emas=update_emas, **block_kwargs)
		x2 = self.human_d(schp_semantic, c, update_emas=update_emas, **block_kwargs)

		return torch.cat([x1, x2], dim=1)	# [batch, 2]
