
import torch
import torch.nn.functional as F

from ..utils.predictor import Predictor



class Superredictor (Predictor):
	def __init__(self, config, device='cpu'):
		super().__init__(device=device)

		self.loadModel(config)


	def super (self, image, iterations=1, down=0):
		x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 255.
		x = x.to(self.device)

		with torch.no_grad():
			for i in range(iterations):
				x = self.model(x)

				if down > 0:
					f = 2. ** -down
					x = F.interpolate(x, scale_factor=f, mode='bilinear')

			result = (x * 255).clip(min=0, max=255).to(dtype=torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()

		return result
