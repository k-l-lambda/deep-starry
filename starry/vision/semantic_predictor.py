
from ..utils.predictor import Predictor
from .images import arrayFromImageStream, writeImageFileFormat



class SemanticPredictor(Predictor):
	def __init__(self, config, device='cpu'):
		super().__init__(device=device)

		self.loadModel(config)


	def predict (self, streams, output_path=None):
		images = map(lambda stream: arrayFromImageStream(stream), streams)

		graphs = []
		for i, image in enumerate(images):
			if output_path:
				writeImageFileFormat(image, output_path, i, 'feature')

			'''pieces = slice_feature(image, width = self.image_width, overlapping = 2 / MARGIN_DIVIDER, padding = True)
			pieces = np.array(list(pieces), dtype = np.uint8)
			staves, _ = self.composer(pieces, np.ones((1, 4, 4, 2)))

			with torch.no_grad():
				batch = torch.from_numpy(staves)
				if torch.cuda.is_available():
					batch = batch.cuda()

				output = self.model(batch)

				semantic, mask, tm0, tm1, tm2 = None, None, None, None, None
				if self.inspect:
					mask, trunk_map, semantic = output
					trunk_map = trunk_map if trunk_map is not None else (None, None, None)
					tm0, tm1, tm2 = trunk_map
				else:
					semantic = output	# (batch, channel, height, width)
				semantic, mask, tm0, tm1, tm2 = map(spliceOutputTensor, (semantic, mask, tm0, tm1, tm2))

				#print('output:', mask.shape, semantic.shape)
				#print('tms:', tm0.shape, tm1.shape, tm2.shape)
				if output_path:
					if mask is not None:
						mask = np.concatenate([np.zeros((1, mask.shape[1], mask.shape[2])), mask], axis = 0)
						mask = np.moveaxis(mask, 0, -1)
						mask = np.clip(np.uint8(mask * 255), 0, 255)

						writeImageFile(mask, output_path, i, 'mask')

					if tm0 is not None:
						for (ii, tm) in enumerate([tm0, tm1, tm2]):
							chromatic = composeChromaticMap(tm)
							writeImageFile(chromatic, output_path, i, f'trunk_map{ii}')

					chromatic = composeChromaticMap(semantic)
					writeImageFile(chromatic, output_path, i, 'semantics')

				graphs.append(score_rep(np.uint8(semantic * 255), self.CFG.DATASET_PROTOTYPE.LABELS, confidence_table = self.confidence_table))'''

		return graphs
