
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from argparse import ArgumentParser
import torch

from starry.utils.config import Configuration
from starry.utils.predictor import Predictor
from starry.vision.chromaticChannels import composeChromaticMap
from starry.vision.images import sliceFeature, splicePieces, softSplicePieces
from starry.vision.datasetViewer import gaugeToRGB, gaugeToFrameFeature



class CameraPredictor (Predictor):
	def __init__ (self, args, request_resolution=(512, 192)):
		super().__init__(device=args.device)

		self.resolution = [*request_resolution]
		self.show_mask = args.mask
		self.show_page = args.page
		self.show_semantic = args.semantic
		self.show_gauge = args.gauge
		self.image_width = request_resolution[0]

		config = Configuration.createdOrLoad(args.config)
		self.loadModel(config)

		PLOTS = 3

		self.figure, axes = plt.subplots(1, PLOTS) if request_resolution[1] > request_resolution[0] else plt.subplots(PLOTS)

		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
			print('VideoCapture open failed')
			return

		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

		ret, frame = self.cap.read()
		print('frame.shape:', frame.shape)
		self.resolution[0] = min(self.resolution[0], frame.shape[1])

		self.images = [ax.imshow(np.zeros((self.resolution[1], self.resolution[0], 3))) for ax in axes]


	def run (self):
		self.animation = FuncAnimation(self.figure, self.updateFrame, interval = 1)
		plt.show()

		self.cap.release()


	def updateFrame (self, _):
		ret, frame = self.cap.read()
		if not ret:
			raise Exception('Can\'t receive frame (stream end?).')

		margin_y = (frame.shape[0] - self.resolution[1]) // 2
		margin_x = (frame.shape[1] - self.resolution[0]) // 2
		if margin_y > 0 and margin_x > 0:
			frame = frame[margin_y:-margin_y, margin_x:-margin_x, :]
		elif margin_y > 0:
			frame = frame[margin_y:-margin_y, :, :]
		elif margin_x > 0:
			frame = frame[:, margin_x:-margin_x, :]
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		pieces = sliceFeature(np.expand_dims(frame, axis = 2), width = self.image_width)
		pieces = np.array(list(pieces), dtype = np.float32) / 255.
		pieces = np.moveaxis(pieces, 3, 1)
		feature = torch.from_numpy(pieces).to(self.device)

		origin = np.stack([frame, frame, frame], axis = 2)
		self.images[0].set_data(origin)

		with torch.no_grad():
			if self.show_mask or self.show_page:
				mask = self.model(feature).cpu().numpy()
				mask = splicePieces(mask, 8, keep_margin = True)[:, :, :frame.shape[1]]
				float_frame = frame / 255.
				if mask.shape[0] == 2:
					mask_plot = np.stack([mask[1, : , :] + float_frame, mask[0, : , :] + float_frame, float_frame], axis = 2)
				else:
					mask_plot = np.stack([mask[0, : , :] + float_frame, mask[1, : , :] + float_frame, mask[2, : , :] + float_frame], axis = 2)
				mask_plot = (np.clip(mask_plot, 0., 1.) * 255).astype(np.uint8)
				self.images[1].set_data(mask_plot)

			if self.show_semantic:
				pred = self.model(feature).cpu().numpy()
				pred = splicePieces(pred, 8, keep_margin = True)
				pred = pred[:, :, :frame.shape[1]]
				semantic_plot = composeChromaticMap(pred)

				self.images[2].set_data(semantic_plot)

			if self.show_gauge:
				pred = self.model(feature).cpu().numpy()
				pred = softSplicePieces(pred, 8)
				pred = pred[:, :, :frame.shape[1]]

				plot = gaugeToRGB(pred)[:, :, ::-1]
				self.images[1].set_data(plot)

				gauge_frame = gaugeToFrameFeature(pred, frame)
				self.images[2].set_data(gauge_frame)

		return self.images[0], self.images[1], self.images[2]


def main ():
	parser = ArgumentParser()
	parser.add_argument('config', type=str, help='config folder')
	parser.add_argument('-m', '--mask', action='store_true', help='score mask mode')
	parser.add_argument('-s', '--semantic', action='store_true', help='score semantic mode')
	parser.add_argument('-g', '--gauge', action='store_true', help='gauge mode')
	parser.add_argument('-p', '--page', action='store_true', help='page layout mode')
	parser.add_argument('-dv', '--device', type=str, default='cuda')
	args = parser.parse_args()

	predictor = CameraPredictor(args, request_resolution=(512, 724) if args.page else ((512, 256) if args.gauge else (512, 192)))
	predictor.run()


if __name__ == '__main__':
	main()
