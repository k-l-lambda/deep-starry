
import os
import sys
import io
import numpy as np
import torch
import logging
import PIL.Image
import cv2
import hashlib
import shutil

from ..utils.predictor import Predictor
from .layout_predictor import PageLayout
from . import transform



BATCH_SIZE = int(os.environ.get('SCORE_PAGE_PROCESSOR_BATCH_SIZE', '1'))

RESIZE_WIDTH = 600


SYSTEM_HEIGHT_ENLARGE = 0.02
SYSTEM_LEFT_ENLARGE = 0.03
SYSTEM_RIGHT_ENLARGE = 0.01

PADDING_LEFT_UNITS = 32
STAFF_HEIGHT_UNITS = 24
UNIT_SIZE = 8


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

	# enlarge heights
	marginY = SYSTEM_HEIGHT_ENLARGE * width
	maginYMax = marginY * 8

	def enlarge(args):
		i, area = args
		top = area['y']
		bottom = top + area['height']

		if i > 0:
			lastArea = areas[i - 1]
			top = max(0, min(top - marginY, max(lastArea['y'] + lastArea['height'], top - maginYMax)))
		else:
			top = min(top, max(marginY, top - maginYMax))

		if i < len(areas) - 1:
			nextArea = areas[i + 1]
			bottom = min(height, max(bottom + marginY, nextArea['y']), bottom + maginYMax)
		else:
			bottom = min(height, bottom + maginYMax)

		return { 'top': top, 'bottom': bottom }
	enlarges = list(map(enlarge, enumerate(areas)))

	for i, area in enumerate(areas):
		area['y'] = enlarges[i]['top']
		area['height'] = enlarges[i]['bottom'] - enlarges[i]['top']

	return {
		'areas': areas,
	}


def panProductLine (line, pan):
	assert len(line) > pan, f'invalid line length: {pan}, {line}'

	line1 = line[:-pan]
	line2 = line[pan:]
	products = line1 * line2

	return products


def detectStavesFromHBL (HB, HL, interval):
	_, HB = cv2.threshold(HB, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	contours, _ = cv2.findContours(HB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	height, width = HB.shape

	STAFF_SIZE_MIN = width * 0.02
	UPSCALE = 4

	upInterval = interval * UPSCALE

	rects = map(cv2.boundingRect, contours)
	rects = filter(lambda rect: rect[2] > STAFF_SIZE_MIN and rect[3] > STAFF_SIZE_MIN, rects)
	rects = sorted(rects, key=lambda rect: rect[1])

	preRects = []
	for rect in rects:
		x, y, w, h = rect

		ri = next((i for i, rc in enumerate(preRects)
			if (y + h / 2) - (rc[1] + rc[3] / 2) < (h + rc[3]) / 2), -1)
		if ri < 0:
			preRects.append(rect)
		else:
			rc = list(preRects[ri])
			if w > rc[2]:
				preRects[ri] = (
					min(x, rc[0]),
					rect[1],
					max(x + w, rc[0] + rc[2]) - x,
					rect[3],
				)
			else:
				rc[0] = min(x, rc[0])
				rc[2] = max(x + w, rc[0] + rc[2]) - rc[0]

	if len(preRects) == 0:
		return { 'reason': '1. no block rectanges detected' }

	phi1 = min(map(lambda rc: rc[0], preRects))
	phi2 = min(map(lambda rc: rc[0] + rc[2], preRects))

	middleRhos = []
	for rect in preRects:
		#logging.info('rect: %s', rect)
		rx, ry, rw, rh = rect
		x = max(rx, 0)
		y = max(round(ry - interval), 0)
		roi = (
			x,
			y,
			min(rw, width - x),
			min(round(rh + interval + ry - y), height - y),
		)
		staffLines = HL[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

		hlineColumn = cv2.resize(staffLines, (1, staffLines.shape[0] * UPSCALE), 0, 0, cv2.INTER_LINEAR).flatten()

		i2 = round(upInterval * 2)
		productionLine2 = panProductLine(hlineColumn, i2)
		productionLine2Max = np.max(productionLine2)
		#logging.info('productionLine2Max: %s', productionLine2Max)
		productionLine2Normalized = np.vectorize(lambda x: np.tanh(x * 4 / productionLine2Max))(productionLine2)
		#logging.info('productionLine2Normalized: %s', productionLine2Normalized)

		convolutionLine = np.zeros(productionLine2.shape)
		convolutionLine += productionLine2Normalized
		convolutionLine[i2:] += productionLine2Normalized[:-i2]
		#logging.info('convolutionLine: %s', convolutionLine)

		convolutionLineMax = np.max(convolutionLine)
		middleY = np.where(convolutionLine == convolutionLineMax)[0][0]

		middleRhos.append(y + middleY / UPSCALE)

	return {
		#'theta': 0,
		'interval': interval,
		'phi1': phi1,
		'phi2': phi2,
		'middleRhos': middleRhos,
	}


def loadImageWithHash (path):
	bytes = open(path, 'rb').read()
	hash = hashlib.md5(bytes).hexdigest()
	image = PIL.Image.open(io.BytesIO(bytes))
	arr = np.array(image)
	if len(arr.shape) > 2 and arr.shape[2] == 4:
		alpha = arr[:, :, 3]
		rgb = arr[:, :, :3]
		rgb[alpha == 0] = 255

		arr = rgb
	elif len(arr.shape) == 2:
		max_brightness = np.max(arr)
		if max_brightness < 255:
			arr *= 255 // max_brightness
		arr = np.stack([arr] * 3, axis=2)

	return arr, hash


def arrayToImageFile (arr, ext='.png'):
	image = PIL.Image.fromarray(arr, 'RGB' if len(arr.shape) == 3 and arr.shape[2] == 3 else 'L')
	fp = io.BytesIO()
	image.save(fp, PIL.Image.registered_extensions()[ext])

	return fp


def resizePageImage (img, size):
	w, h = size
	filled_height = img.shape[0] * w // img.shape[1]
	img = cv2.resize(img, (w, filled_height), interpolation=cv2.INTER_AREA)

	if filled_height < h:
		result = np.ones((h, w, img.shape[2]), dtype=np.uint8) * 255
		result[:filled_height] = img

		return result

	return img[:h]


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
		if output_folder is not None:
			os.makedirs(output_folder, exist_ok=True)

		try:
			for i in range(0, len(input_paths), BATCH_SIZE):
				image_hashes = list(map(loadImageWithHash, input_paths[i:i + BATCH_SIZE]))
				page_filenames = None
				if output_folder:
					page_filenames = []
					for ii, ih in enumerate(image_hashes):
						input_path = input_paths[i + ii]
						_, hash = ih
						# get extension name from input path
						ext = os.path.splitext(input_path)[1]
						filename = hash + ext
						page_filenames.append(filename)
						shutil.copyfile(input_path, os.path.join(output_folder, filename))

						logging.debug('Page image copied: %s', filename)

				images = list(map(lambda ih: ih[0], image_hashes))
				#cv2.imwrite('./output/image0.png', images[0])

				# unify images' dimensions
				ratio = max(map(lambda img: img.shape[0] / img.shape[1], images))
				height = int(RESIZE_WIDTH * ratio)
				height += -height % 4
				unified_images = list(map(lambda img: resizePageImage(img, (RESIZE_WIDTH, height)), images))
				image_array = np.stack(unified_images, axis=0)

				batch, _ = self.composer(image_array, np.ones((1, 4, 4, 2)))
				batch = torch.from_numpy(batch)
				batch = batch.to(self.device)

				with torch.no_grad():
					output = self.model(batch)
					output = output.cpu().numpy()

					for j, heatmap in enumerate(output):
						try:
							layout = PageLayout(heatmap)

							if layout.theta is None:
								yield {
									'theta': None,
									'interval': None,
									'detection': None,
									'page_info': None,
								}
								continue

							image = images[j]
							original_size = (image.shape[1], int(image.shape[1] * ratio))

							# rotation correction
							rot_mat = cv2.getRotationMatrix2D((original_size[0] / 2, original_size[1] / 2), layout.theta * 180 / np.pi, 1)
							image = cv2.warpAffine(image, rot_mat, original_size, flags=cv2.INTER_CUBIC)
							#cv2.imwrite('./output/image.png', image)

							heatmap = np.moveaxis(np.uint8(heatmap * 255), 0, -1)
							heatmap = cv2.resize(heatmap, original_size)
							heatmap = cv2.warpAffine(heatmap, rot_mat, original_size, flags=cv2.INTER_LINEAR)
							#cv2.imwrite(f'./output/heatmap-{i+j}.png', heatmap)

							HB = heatmap[:, :, 1]
							HL = heatmap[:, :, 2]
							block = heatmap.max(axis=2)
							detection = detectSystems(block)

							page_interval = layout.interval * original_size[0] / RESIZE_WIDTH

							for si, area in enumerate(detection['areas']):
								l, r, t, b = map(round, (area['x'], area['x'] + area['width'], area['y'], area['y'] + area['height']))
								hb = HB[t:b, l:r]
								hl = HL[t:b, l:r]
								area['staves'] = detectStavesFromHBL(hb, hl, page_interval)
								#cv2.imwrite(f'./output/hl-{si}.png', hl)

								if area['staves'].get('middleRhos') is None:
									continue

								system_image = image[t:b, l:r, :]
								#cv2.imwrite(f'./output/system-{si}.png', system_image)

								area['staff_images'] = []

								interval = area['staves']['interval']
								staff_size = (round(system_image.shape[1] * UNIT_SIZE / interval), STAFF_HEIGHT_UNITS * UNIT_SIZE)
								for ssi, rho in enumerate(area['staves']['middleRhos']):
									top = round(rho - STAFF_HEIGHT_UNITS * interval / 2)
									bottom = round(rho + STAFF_HEIGHT_UNITS * interval / 2)
									#logging.info('staff: %s, %s', top, bottom)

									if top >= 0 and bottom < system_image.shape[0]:
										staff_image = system_image[top:bottom, :, :]
									else:
										staff_image = np.ones((bottom - top, system_image.shape[1], system_image.shape[2]), dtype=np.uint8) * 255
										bi = system_image.shape[0] - bottom if system_image.shape[0] - bottom < 0 else staff_image.shape[0]
										staff_image[max(-top, 0):bi, :, :] = system_image[max(top, 0):min(bottom, system_image.shape[0]), :, :]
									staff_image = cv2.resize(staff_image, staff_size, interpolation=cv2.INTER_CUBIC)

									#cv2.imwrite(f'./output/staff-{si}-{ssi}.png', staff_image)
									bytes = arrayToImageFile(staff_image).getvalue()
									hash = hashlib.md5(bytes).hexdigest()

									area['staff_images'].append({
										'hash': f'md5:{hash}',
										'position': {
											'x': -area['staves']['phi1'] * UNIT_SIZE / interval,
											'y': -STAFF_HEIGHT_UNITS * UNIT_SIZE / 2,
											'width': staff_size[0],
											'height': staff_size[1],
										},
									})

									if output_folder is not None:
										with open(os.path.join(output_folder, hash + '.png'), 'wb') as f:
											f.write(bytes)
										#logging.info('Staff image wrote: %s.png', hash)

							page_info = {
								'url': 'md5:' + page_filenames[j] if page_filenames is not None else None,
								'size': original_size,
							}

							yield {
								'theta': layout.theta,
								'interval': page_interval,
								'detection': detection,
								'page_info': page_info,
							}
						except:
							yield {
								'theta': None,
								'interval': None,
								'detection': None,
								'page_info': None,
								'reason': str(sys.exc_info()[1]),
							}
		except:
			logging.warn(sys.exc_info()[1])
			yield None
