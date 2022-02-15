
import os

from .slicedScore import SlicedScore
from .score import MASK
from .utils import loadSplittedDatasets



class ScoreMask (SlicedScore):
	@staticmethod
	def load (root, args, splits, device='cpu', args_variant=None):
		return loadSplittedDatasets(ScoreMask, root=root, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, **kwargs):
		super().__init__(root, **kwargs)


	# override
	def _loadTarget (self, name, *_):
		mask = self.reader.readImage(os.path.join(MASK, name + ".png"))
		#print('mask:', mask.shape)

		if mask is None:
			return None

		return mask[:, :, :2][:, :, ::-1]
