
import re
import math
import numpy as np
import cv2
import yaml



POINT_RADIUS_MAX = 8


def detectPoints(heatmap, vertical_units = 24, otsu = False):
	unit = heatmap.shape[0] / vertical_units
	y0 = heatmap.shape[0] / 2.0

	blur_kernel = (heatmap.shape[0] // 128) * 2 + 1
	heatmap_blur = cv2.GaussianBlur(heatmap, (blur_kernel, blur_kernel), 0) if blur_kernel > 1 else heatmap
	if otsu:
		_, thresh = cv2.threshold(heatmap_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	else:
		thresh = cv2.adaptiveThreshold(heatmap_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	points = []
	for c in contours:
		(x,y),radius = cv2.minEnclosingCircle(c)	# 'minEnclosingCircle' is very slow!

		confidence = 0
		for px in range(max(math.floor(x - radius), 0), min(math.ceil(x + radius), heatmap.shape[1])):
			for py in range(max(math.floor(y - radius), 0), min(math.ceil(y + radius), heatmap.shape[0])):
				confidence += heatmap[py, px] / 255.

		if radius < POINT_RADIUS_MAX:
			points.append({
				'mark': (x, y, radius),
				'x': x / unit,
				'y': (y - y0) / unit,
				'confidence': float(confidence),
			})

	return points


def detectVLines (heatmap, vertical_units = 24, otsu = False):
	unit = heatmap.shape[0] / vertical_units
	y0 = heatmap.shape[0] / 2.0

	blur_kernel = (heatmap.shape[0] // 128) * 2 + 1
	heatmap_blur = cv2.GaussianBlur(heatmap, (blur_kernel, blur_kernel), 0) if blur_kernel > 1 else heatmap
	if otsu:
		_, thresh = cv2.threshold(heatmap_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	else:
		thresh = cv2.adaptiveThreshold(heatmap_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	lines = []
	for contour in contours:
		left, top, width, height = cv2.boundingRect(contour)
		x = (left + width / 2) / unit
		y1 = (top - y0) / unit
		y2 = (top + height - y0) / unit

		confidence = 0
		for px in range(left, left + width):
			for py in range(top, top + height):
				confidence += heatmap[py, px] / 255.

		length = max(height, 2.5)
		confidence /= length
		#print('confidence:', confidence)

		lines.append({
			'x': x,
			'y': y1,
			'extension': {
				'y1': y1,
				'y2': y2,
			},
			'confidence': float(confidence),
			'mark': (left + width / 2, top, top + height),
		})

	return lines


def detectRectangles (heatmap, vertical_units = 24, otsu = False):
	unit = heatmap.shape[0] / vertical_units
	y0 = heatmap.shape[0] / 2.0

	if otsu:
		_, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	else:
		_, thresh = cv2.threshold(heatmap, 92, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	rects = []
	for contour in contours:
		left, top, width, height = cv2.boundingRect(contour)
		if width * height / unit / unit < 2:	# clip too small rectangle
			continue

		x = (left + width / 2) / unit
		y = (top + height / 2) / unit

		confidence = 0
		for px in range(left, left + width):
			for py in range(top, top + height):
				confidence += heatmap[py, px] / 255.

		confidence /= width * height

		rects.append({
			'x': x,
			'y': y - y0 / unit,
			'extension': {
				'width': width / unit,
				'height': height / unit,
			},
			'confidence': float(confidence),
			'mark': (left, top, width, height),
		})

	return rects


def detectBoxes (heatmap, vertical_units = 24, otsu = False):
	if otsu:
		_, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	else:
		_, thresh = cv2.threshold(heatmap, 92, 255, cv2.THRESH_BINARY)
		#thresh = cv2.adaptiveThreshold(heatmap, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -1)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	rects = []
	for contour in contours:
		rect = cv2.minAreaRect(contour)

		pos, size, theta = rect
		confidence = math.sqrt(size[0] * size[1])

		area = size[0] * size[1]
		if area > 100:
			rects.append({
				'x': pos[0],
				'y': pos[1],
				'extension': {
					'width': size[0],
					'height': size[1],
					'theta': theta,
				},
				'confidence': confidence,
				'mark': rect,
			})

	return rects


POINT_NEAR_DISTANCE = 0.2

LINE_TOLERANCE_X = 0.5
LINE_TOLERANCE_Y = 0.8

RECT_TOLERANCE = 0.8


def findNearPoint (point, points):
	for p in points:
		if (p['x'] - point['x']) ** 2 + (p['y'] - point['y']) ** 2 < POINT_NEAR_DISTANCE ** 2:
			return p

	return None


def findNearVLine (line, lines):
	aimExt = line['extension']
	for l in lines:
		ext = l['extension']
		if abs(l['x'] - line['x']) < LINE_TOLERANCE_X and abs(ext['y1'] - aimExt['y1']) < LINE_TOLERANCE_Y and abs(ext['y2'] - aimExt['y2']) < LINE_TOLERANCE_Y:
			return l

	return None


def rectToBox (data):
	x, y = data['x'], data['y']
	w, h = data['extension']['width'], data['extension']['height']
	l, r, t, b = x - w / 2, x + w / 2, y - h / 2, y + h / 2

	return l, r, t, b

def findRect (rect, rects):
	al, ar, at, ab = rectToBox(rect)
	for rc in rects:
		l, r, t, b = rectToBox(rc)
		if l > al - RECT_TOLERANCE and r < ar + RECT_TOLERANCE and t > at - RECT_TOLERANCE and b < ab + RECT_TOLERANCE:
			return rc

	return None


def rectPoints (rect):
	rc = ((rect['x'], rect['y']), (rect['extension']['width'], rect['extension']['height']), rect['extension']['theta'])
	points = np.int0(cv2.boxPoints(rc))
	#points.sort(key = lambda p: p[0] + p[1])

	points = sorted(points, key = lambda p: p[0] + p[1])
	return np.array(points).flatten()

def pointsDistance (p1, p2):
	return np.sum(np.square(p1 - p2)) / len(p1)

def findBox (rect, rects):
	box = rectPoints(rect)

	for rc in rects:
		b = rectPoints(rc)
		distance = pointsDistance(b, box)
		if distance < 6 * 6:
			return rc

	return None


def labelToDetector (label):
	if re.match(r'^vline_', label):
		return (detectVLines, findNearVLine)
	elif re.match(r'^rect_', label):
		return (detectRectangles, findRect)
	elif re.match(r'^box_', label):
		return (detectBoxes, findBox)
	else:
		return (detectPoints, findNearPoint)


def pointBrief (point):
	info = {'x': point['x'], 'y': point['y']}
	if point.get('extension') is not None:
		info['y1'] = point['extension'].get('y1')
		info['y2'] = point['extension'].get('y2')
		info['width'] = point['extension'].get('width')
		info['height'] = point['extension'].get('height')

	return info


def countPoints (points_true, points_pred, finder):
	results = []

	for pt in points_true:
		p = finder(pt, points_pred)
		if p is not None:
			pt['positive'] = True
			p['positive'] = True

	for p in points_true:
		if not p.get('positive'):
			results.append(dict({'value': 1, 'confidence': 0, **pointBrief(p)}))

	for p in points_pred:
		results.append(dict({'value': 1 if p.get('positive') else -1, 'confidence': p['confidence'], **pointBrief(p)}))

	return results


def countHeatmaps (heatmap_true, heatmap_pred, label, unit_size):
	detect, finder = labelToDetector(label)

	vertical_units = heatmap_true.shape[0] // unit_size

	points_true = detect(heatmap_true, vertical_units = vertical_units, otsu = True)
	points_pred = detect(heatmap_pred, vertical_units = vertical_units)

	return countPoints(points_true, points_pred, finder)


def statPoints (points, true_count, negative_weight = 1, positive_weight = 1):
	points.sort(key = lambda p: p['confidence'])

	fake_positive_count = len([p for p in points if p['value'] < 0])
	fake_negative_count = 0
	confidence = 0

	for p in points:
		confidence = p['confidence']
		if confidence > 0 and fake_negative_count * negative_weight >= fake_positive_count * positive_weight:
			break

		if p['value'] > 0:
			fake_negative_count += 1
		else:
			fake_positive_count -= 1

	true_count = max(true_count, 1)

	true_positive_count = len(points) - fake_positive_count
	true_negative_count = len([p for p in points if p['confidence'] < confidence and p['value'] < 0])

	error = fake_negative_count * negative_weight + fake_positive_count * positive_weight
	feasibility = 1 - error / true_count

	pe = max((fake_negative_count + fake_positive_count) / true_count, 1e-100)
	precision = -math.log(pe)

	return confidence, error, precision, feasibility, fake_negative_count, fake_positive_count, true_negative_count, true_positive_count


class Compounder:
	def __init__ (self, config):
		self.list = config['data.args.compound_labels']
		self.labels = list(map(lambda item: item['label'], self.list)) if self.list else config['data.args.labels']

	def compound (self, image):	# (channel, h, w)
		if self.list is not None:
			shape = image.shape
			channels = len(self.list)
			result = np.zeros((channels, shape[1], shape[2]), dtype=np.uint8)
			for l in range(channels):
				for c in self.list[l]['channels']:
					result[l, :, :] = np.maximum(result[l, :, :], image[c, :, :])

			return result

		return image


class Contour:
	def __init__(self, config):
		self.name = 'contour'

		with open('./semantics.yaml', 'r') as stream:
			self.semantics = yaml.safe_load(stream)

		self.compounder = Compounder(config)

		self.unit_size = config.DATASET_PROTOTYPE.UNIT_SIZE

		self.layers = [[] for label in self.compounder.labels]
		self.true_count = 0

	def batch (self, pred, target):
		int_pred = np.uint8(pred * 255)
		int_target = np.uint8(target * 255)
		for n, (pred_map, target_map), in enumerate(zip(int_pred, int_target)): # pred_map: (channel, height, width)
			pred_map_compound = self.compounder.compound(pred_map)
			target_map_compound = self.compounder.compound(target_map)

			for i, label, in enumerate(self.compounder.labels):
				pred_layer = pred_map_compound[i]
				target_layer = target_map_compound[i]

				points = countHeatmaps(target_layer, pred_layer, label, unit_size = self.unit_size)
				self.layers[i] += points
				self.true_count += len([p for p in points if p['value'] > 0])

	def stat (self):
		total_error = 0
		total_true_count = 0

		self.details = {}
		self.loss_weights = []
		for i, layer in enumerate(self.layers):
			label = self.compounder.labels[i]
			neg_weight, pos_weight = self.semantics['loss_weights'].get(label, (1, 1))

			true_count = len([p for p in layer if p['value'] > 0])

			confidence, error, precision, feasibility, fake_neg, fake_pos = statPoints(layer, true_count, neg_weight, pos_weight)
			total_error += error
			total_true_count += true_count

			self.details[label] = {
				'confidence': confidence,
				'precision': precision,
				'feasibility': feasibility,
				'true_count': true_count,
				'errors': f'{fake_neg}-|+{fake_pos}'
			}

			self.loss_weights.append(1 - feasibility)

		self.layers = [[] for label in self.compounder.labels]
		self.true_count = 0

		return -math.log(max(total_error / max(total_true_count, 1), 1e-100))
