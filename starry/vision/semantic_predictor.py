
import numpy as np
import torch

from ..utils.predictor import Predictor
from .images import arrayFromImageStream, writeImageFileFormat, sliceFeature, spliceOutputTensor, MARGIN_DIVIDER
from . import transform
from .chromaticChannels import composeChromaticMap



class SemanticPredictor(Predictor):
	def __init__(self, config, device='cpu', inspect=False):
		super().__init__(device=device)

		self.inspect = inspect
		if inspect:
			config['model.type'] = config['model.type'] + 'Inspection'

		self.loadModel(config)

		self.slicing_width = config['data.slicing_width']
		self.labels = config['data.labels']

		trans = [t for t in config['data.trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)


	def predict (self, streams, output_path=None):
		images = map(lambda stream: arrayFromImageStream(stream), streams)

		graphs = []
		for i, image in enumerate(images):
			if output_path:
				writeImageFileFormat(image, output_path, i, 'feature')

			pieces = sliceFeature(image, width = self.slicing_width, overlapping = 2 / MARGIN_DIVIDER, padding = True)
			pieces = np.array(list(pieces), dtype = np.uint8)
			staves, _ = self.composer(pieces, np.ones((1, 4, 4, 2)))

			with torch.no_grad():
				batch = torch.from_numpy(staves)
				batch = batch.to(self.device)

				output = self.model(batch)

				semantic, mask, tm0, tm1, tm2 = None, None, None, None, None
				if self.inspect:
					mask, trunk_map, semantic = output
					trunk_map = trunk_map if trunk_map is not None else (None, None, None)
					tm0, tm1, tm2 = trunk_map
				else:
					semantic = output	# (batch, channel, height, width)
				semantic, mask, tm0, tm1, tm2 = map(spliceOutputTensor, (semantic, mask, tm0, tm1, tm2))

				if output_path:
					if mask is not None:
						mask = np.concatenate([np.zeros((1, mask.shape[1], mask.shape[2])), mask], axis = 0)
						mask = np.moveaxis(mask, 0, -1)
						mask = np.clip(np.uint8(mask * 255), 0, 255)

						writeImageFileFormat(mask, output_path, i, 'mask')

					if tm0 is not None:
						for (ii, tm) in enumerate([tm0, tm1, tm2]):
							chromatic = composeChromaticMap(tm)
							writeImageFileFormat(chromatic, output_path, i, f'trunk_map{ii}')

					chromatic = composeChromaticMap(semantic)
					writeImageFileFormat(chromatic, output_path, i, 'semantics')

				#graphs.append(score_rep(np.uint8(semantic * 255), self.labels, confidence_table=self.confidence_table))

		return graphs
