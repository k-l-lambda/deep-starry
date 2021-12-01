
import os
import numpy as np
import torch
import logging
import PIL.Image
import cv2

from ..utils.predictor import Predictor
from .layout_predictor import PageLayout
from . import transform



BATCH_SIZE = int(os.environ.get('SCORE_PAGE_PROCESSOR_BATCH_SIZE', '2'))

RESIZE_WIDTH = 600


SYSTEM_HEIGHT_ENLARGE = 0.02
SYSTEM_LEFT_ENLARGE = 0.03
SYSTEM_RIGHT_ENLARGE = 0.01


def detectSystems (image):
	height, width = image.shape

	blur = cv2.GaussianBlur(image, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, -40)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	marginLeft = SYSTEM_LEFT_ENLARGE * width
	marginRight = SYSTEM_RIGHT_ENLARGE * width

	areas = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)

		rw = w / width
		rh = h / width
		if (rw > 0.6 and rh > 0.02) or (rw > 0.12 and rh > 0.2):
			left = max(x - marginLeft, 0)
			right = min(x + w + marginRight, width)

			areas.append({
				'x': left,
				'y': y,
				'width': right - left,
				'height': h,
			})
	areas.sort(key=lambda a: a['y'])

	'''// enlarge heights
	const marginY = SYSTEM_HEIGHT_ENLARGE * matSize.width;
	const maginYMax = marginY * 4;
	const enlarges = areas.map((area, i) => {
		let top = area.y;
		let bottom = area.y + area.height;

		if (i > 0) {
			const lastArea = areas[i - 1];
			top = Math.max(0, Math.min(top - marginY, Math.max(lastArea.y + lastArea.height, top - maginYMax)));
		}
		else
			top = Math.min(top, Math.max(marginY, top - maginYMax));

		if (i < areas.length - 1) {
			const nextArea = areas[i + 1];
			bottom = Math.min(matSize.height, Math.max(bottom + marginY, nextArea.y), bottom + maginYMax);
		}
		else
			bottom = Math.min(matSize.height, bottom + maginYMax);

		return { top, bottom };
	});

	areas.forEach((area, i) => {
		area.y = enlarges[i].top;
		area.height = enlarges[i].bottom - enlarges[i].top;
	});

	if (options.log) {
		const MARK_COLOR1 = new cv2.Scalar(0, 255, 0);
		const MARK_COLOR2 = new cv2.Scalar(255, 0, 0);

		const graph = new cv2.Mat();
		cv2.cvtColor(mat, graph, cv2.COLOR_BGR2RGB);

		for (let i = 0; i < (contours.size() as any); ++i) {
			const contour = contours.get(i);
			const rect = cv2.boundingRect(contour);

			cv2.rectangle(graph, new cv2.Point(rect.x, rect.y), new cv2.Point(rect.x + rect.width, rect.y + rect.height),
				MARK_COLOR1, 1);
		}
		areas.forEach(rect => cv2.rectangle(graph, new cv2.Point(rect.x, rect.y), new cv2.Point(rect.x + rect.width, rect.y + rect.height),
			MARK_COLOR2, 2));
		options.reportMat("contours", graph);
	}'''

	return {
		'areas': areas,
	}


class ScorePageProcessor (Predictor):
	def __init__(self, config, device='cpu', inspect=False):
		super().__init__(device=device)

		self.inspect = inspect
		if inspect:
			config['model.type'] = config['model.type'] + 'Inspection'

		self.loadModel(config)

		data_args = config['data.args'] or config['data']

		trans = [t for t in data_args['trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)


	def predict (self, input_paths, output_folder=None):
		for i in range(0, len(input_paths), BATCH_SIZE):
			images = list(map(lambda path: cv2.imread(path), input_paths[i:i + BATCH_SIZE]))

			# unify images' dimensions
			ratio = min(map(lambda img: img.shape[0] / img.shape[1], images))
			height = int(RESIZE_WIDTH * ratio)
			height -= height % 4
			unified_images = list(map(lambda img: cv2.resize(img, (RESIZE_WIDTH, img.shape[0] * RESIZE_WIDTH // img.shape[1]))[:height], images))
			image_array = np.stack(unified_images, axis=0)

			batch, _ = self.composer(image_array, np.ones((1, 4, 4, 2)))
			batch = torch.from_numpy(batch)
			batch = batch.to(self.device)

			with torch.no_grad():
				output = self.model(batch)
				output = output.cpu().numpy()

				for j, heatmap in enumerate(output):
					layout = PageLayout(heatmap)

					VL, HB, HL = heatmap
					block = (heatmap.max(axis = 0) * 255).astype(np.uint8)
					detection = detectSystems(block)

					yield {
						'theta': layout.theta, 'interval': layout.interval,
						'detection': detection,
					}
