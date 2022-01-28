
import os
import io
import numpy as np
import PIL.Image
import cv2
import hashlib

from .images import encodeImageBase64



RESIZE_WIDTH = 600
CANVAS_WIDTH_MIN = 1024


SYSTEM_HEIGHT_ENLARGE = 0.02
SYSTEM_LEFT_ENLARGE = 0.03
SYSTEM_RIGHT_ENLARGE = 0.01

#PADDING_LEFT_UNITS = 32
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

		i1 = round(upInterval)
		kernel = np.zeros(i1 * 4 + 1)	# the comb 0f 5
		kernel[::i1] = 1
		convolutionLine = np.convolve(hlineColumn, kernel)
		#logging.info("hlineColumn: %s", hlineColumn)
		#logging.info("convolutionLine: %s", convolutionLine)

		convolutionLineMax = np.max(convolutionLine)
		middleY = np.where(convolutionLine == convolutionLineMax)[0][0] - round(upInterval * 2)

		middleRhos.append(y + middleY / UPSCALE)

	return {
		#'theta': 0,
		'interval': interval,
		'phi1': phi1,
		'phi2': phi2,
		'middleRhos': middleRhos,
	}


def arrayToImageFile (arr, ext='.png'):
	image = PIL.Image.fromarray(arr, 'RGB' if len(arr.shape) == 3 and arr.shape[2] == 3 else 'L')
	fp = io.BytesIO()
	image.save(fp, PIL.Image.registered_extensions()[ext])

	return fp


def scaleDetection (detection, scale):
	areas = list(map(lambda a: {
		'x': a['x'] * scale,
		'y': a['y'] * scale,
		'width': a['width'] * scale,
		'height': a['height'] * scale,
		'staff_images': a['staff_images'],
		'staves': {
			'interval': a['staves']['interval'] * scale,
			'phi1': a['staves']['phi1'] * scale,
			'phi2': a['staves']['phi2'] * scale,
			'middleRhos': list(map(lambda x: x * scale, a['staves']['middleRhos'])),
		},
	}, detection['areas']))

	return {'areas': areas}


class PageLayout:
	def __init__ (self, heatmap):	# channel semantic: [VL, StaffBox, HL]
		lines_map = heatmap[2]
		self.interval = PageLayout.measureInterval(lines_map)

		heatmap = np.uint8(heatmap * 255)
		heatmap = np.moveaxis(heatmap, 0, -1)
		self.image = PIL.Image.fromarray(heatmap, 'RGB')

		staves_map = heatmap[:, :, 1]
		self.theta = PageLayout.measureTheta(staves_map)
		self.heatmap = heatmap


	def json (self):
		image_code = encodeImageBase64(self.image)

		return {
			'image': image_code,
			'theta': self.theta,
			'interval': self.interval,
		}


	def detect (self, image, ratio, output_folder=None):
		original_size = (image.shape[1], image.shape[0])
		aligned_height = int(image.shape[1] * ratio)
		if image.shape[0] < aligned_height:
			image = np.pad(image, ((0, aligned_height - image.shape[0]), (0, 0), (0,0)), mode='constant')
		elif image.shape[0] > aligned_height:
			image = image[:aligned_height]

		# determine canvas size
		canvas_size = (original_size[0], aligned_height)
		while canvas_size[0] < CANVAS_WIDTH_MIN:
			canvas_size = (canvas_size[0] * 2, canvas_size[1] * 2)
		if canvas_size[0] > original_size[0]:
			image = cv2.resize(image, canvas_size)

		# rotation correction
		rot_mat = cv2.getRotationMatrix2D((canvas_size[0] / 2, canvas_size[1] / 2), self.theta * 180 / np.pi, 1)
		image = cv2.warpAffine(image, rot_mat, canvas_size, flags=cv2.INTER_CUBIC)
		#cv2.imwrite(f'./output/image-{i+j}.png', image)

		heatmap = cv2.resize(self.heatmap, (canvas_size[0], round(canvas_size[0] * self.heatmap.shape[0] / self.heatmap.shape[1])), interpolation=cv2.INTER_CUBIC)
		if heatmap.shape[0] > canvas_size[1]:
			heatmap = heatmap[:canvas_size[1]]
		elif heatmap.shape[0] < canvas_size[1]:
			heatmap = np.pad(heatmap, ((0, canvas_size[1] - heatmap.shape[0]), (0, 0), (0,0)), mode='constant')
		heatmap = cv2.warpAffine(heatmap, rot_mat, canvas_size, flags=cv2.INTER_LINEAR)
		#cv2.imwrite(f'./output/heatmap-{i+j}.png', heatmap)

		HB = heatmap[:, :, 1]
		HL = heatmap[:, :, 2]
		block = heatmap.max(axis=2)
		detection = detectSystems(block)

		page_interval = self.interval * original_size[0] / RESIZE_WIDTH
		canvas_interval = self.interval * canvas_size[0] / RESIZE_WIDTH

		for si, area in enumerate(detection['areas']):
			l, r, t, b = map(round, (area['x'], area['x'] + area['width'], area['y'], area['y'] + area['height']))

			system_image = image[t:b, l:r, :]
			#cv2.imwrite(f'./output/system-{si}.png', system_image)

			hb = HB[t:b, l:r]
			hl = HL[t:b, l:r]

			area['staves'] = detectStavesFromHBL(hb, hl, canvas_interval)
			#cv2.imwrite(f'./output/hl-{si}.png', hl)

			if area['staves'].get('middleRhos') is None:
				continue

			area['staff_images'] = []

			interval = area['staves']['interval']
			unit_scaling = UNIT_SIZE / interval
			staff_size = (round(system_image.shape[1] * unit_scaling), STAFF_HEIGHT_UNITS * UNIT_SIZE)
			for ssi, rho in enumerate(area['staves']['middleRhos']):
				map_x = np.tile(np.arange(staff_size[0], dtype=np.float32), (staff_size[1], 1)) / unit_scaling
				map_y = (np.tile(np.arange(staff_size[1], dtype=np.float32), (staff_size[0], 1)).T - staff_size[1] / 2) / unit_scaling + rho
				staff_image = cv2.remap(system_image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

				hash = None
				if output_folder is not None:
					#cv2.imwrite(f'./output/staff-{si}-{ssi}.png', staff_image)
					bytes = arrayToImageFile(staff_image).getvalue()
					hash = hashlib.md5(bytes).hexdigest()

					with open(os.path.join(output_folder, hash + '.png'), 'wb') as f:
						f.write(bytes)
					#logging.info('Staff image wrote: %s.png', hash)

				area['staff_images'].append({
					'hash': f'md5:{hash}' if hash else None,
					'position': {
						'x': -area['staves']['phi1'] * UNIT_SIZE / interval,
						'y': -STAFF_HEIGHT_UNITS * UNIT_SIZE / 2,
						'width': staff_size[0],
						'height': staff_size[1],
					},
				})

		return {
			'theta': self.theta,
			'interval': page_interval,
			'detection': scaleDetection(detection, original_size[0] / canvas_size[0]),
		}


	@staticmethod
	def measureTheta (heatmap):
		edges = cv2.Canny(heatmap, 50, 150, apertureSize = 3)
		lines = cv2.HoughLines(edges, 1, np.pi/18000, round(heatmap.shape[1] * 0.4), min_theta = np.pi * 0.48, max_theta = np.pi * 0.52)
		if lines is None:
			return None

		avg_theta = sum(map(lambda line: line[0][1], lines)) / len(lines)

		return avg_theta - np.pi / 2


	@staticmethod
	def measureInterval (heatmap):
		UPSCALE = 4

		width = heatmap.shape[1]
		heatmap = cv2.resize(heatmap, (heatmap.shape[1] // UPSCALE, heatmap.shape[0] * UPSCALE), interpolation=cv2.INTER_LINEAR)
		#print('upscale heatmap:', heatmap.shape)

		interval_min, interval_max = round(width * 0.002 * UPSCALE), round(width * 0.025 * UPSCALE)

		brights = []
		for y in range(interval_min, interval_max):
			m1, m2 = heatmap[y:], heatmap[:-y]
			p = np.multiply(m1, m2)
			brights.append(np.mean(p))

		# minus scale x2, to weaken 2 intervals activation
		brights = np.array([brights])
		brights2 = cv2.resize(brights, (brights.shape[1] * 2, 1))

		brights = brights.flatten()[interval_min:]
		brights2 = brights2.flatten()[:len(brights)]

		brights -= brights2 * 0.5
		#print('interval_min:', interval_min * 2)
		#print('brights:', brights)

		return (interval_min * 2 + int(np.argmax(brights))) / UPSCALE
