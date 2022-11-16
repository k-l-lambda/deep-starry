
import numpy as np
import cv2

from .slicedScore import SlicedScore
from .utils import loadSplittedDatasets



def makeIndicesArray (shape):
	indices_x = np.tile(np.arange(shape[1], dtype=np.float32)[None, :], (shape[0], 1))
	indices_y = np.tile(np.arange(shape[0], dtype=np.float32)[:, None], (1, shape[1]))

	return indices_y, indices_x


class ScoreGauge (SlicedScore):
	indices = makeIndicesArray((256, 4096))
	dilate_kernel = np.ones((9, 9), np.uint8)


	@staticmethod
	def load (root, args, splits, device='cpu', args_variant=None):
		return loadSplittedDatasets(ScoreGauge, root=root, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, y_unit, with_mask=False, mask_bg_value=0, **kwargs):
		super().__init__(root, **kwargs)

		self.y_unit = y_unit
		self.with_mask = with_mask
		self.mask_bg_value = mask_bg_value


	# override
	def _loadTarget (self, name, source):
		height, width, _ = source.shape
		target = np.zeros([height, width, 3 if self.with_mask else 2], np.float32)

		target[:, :, 0] = (ScoreGauge.indices[0][:height, :width] + 0.5 - height // 2) / self.y_unit
		target[:, :, 1] = ScoreGauge.indices[1][:height, :width]

		if self.with_mask:
			mask = 1 - source
			ret, mask = cv2.threshold(mask, 1/255., 1., cv2.THRESH_BINARY)
			mask = cv2.dilate(mask, ScoreGauge.dilate_kernel, iterations=1)
			mask *= np.exp(-0.5 * (target[:, :, 0] / 6) ** 2)
			mask = np.maximum(mask, self.mask_bg_value)

			target[:, :, 2] = mask

		return target
