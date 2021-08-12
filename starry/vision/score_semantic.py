
import json
import cv2
import numpy as np
import math
import re

from .contours import detectPoints, detectVLines, detectRectangles, detectBoxes



VERTICAL_UNITS = 24.


class ScoreSemantic:
	def __init__(self, heatmaps, labels, confidence_table=None):
		self.data = dict({
			'__prototype': 'SemanticGraph',
			'points': [],
		})

		num_classes = len(labels)
		assert num_classes == len(heatmaps), f'classes - heat maps count mismatch, {num_classes} - {len(heatmaps)}'

		self.marks = []
		for i in range(num_classes):
			semantic = labels[i]

			mean_confidence = 1
			if confidence_table is not None:
				item = confidence_table[i]
				assert item['semantic'] == semantic, f'confidence table semantic mismatch:{item["semantic"]}, {semantic}'

				mean_confidence = max(item['mean_confidence'], 1e-4)

			if re.match(r'^vline_', semantic):
				lines = detectVLines(heatmaps[i], vertical_units=VERTICAL_UNITS)
				for line in lines:
					self.data['points'].append({
						'semantic': semantic,
						'x': line['x'],
						'y': line['y'],
						'extension': line['extension'],
						'confidence': line['confidence'] / mean_confidence,
					})

				self.marks.append({'vlines': np.array([line['mark'] for line in lines], dtype = np.float32)})
			elif re.match(r'^rect_', semantic):
				rectangles = detectRectangles(heatmaps[i], vertical_units=VERTICAL_UNITS)
				for rect in rectangles:
					self.data['points'].append({
						'semantic': semantic,
						'x': rect['x'],
						'y': rect['y'],
						'extension': rect['extension'],
						'confidence': rect['confidence'] / mean_confidence,
					})

				self.marks.append({'rectangles': np.array([rect['mark'] for rect in rectangles], dtype = np.float32)})
			elif re.match(r'^box_', semantic):
				boxes = detectBoxes(heatmaps[i], vertical_units=VERTICAL_UNITS)
				for rect in boxes:
					self.data['points'].append({
						'semantic': semantic,
						'x': rect['x'],
						'y': rect['y'],
						'extension': rect['extension'],
						'confidence': rect['confidence'] / mean_confidence,
					})

				self.marks.append({'boxes': [rect['mark'] for rect in boxes]})
			else:
				points = detectPoints(heatmaps[i], vertical_units=VERTICAL_UNITS)
				for point in points:
					self.data['points'].append({
						'semantic': semantic,
						'x': point['x'],
						'y': point['y'],
						'confidence': point['confidence'] / mean_confidence,
					})

				self.marks.append({'points': np.array([point['mark'] for point in points], dtype = np.float32)})


	def json(self):
		return self.data
