
import math
import numpy as np
import re

from .contours import detectPoints, detectVLines, detectRectangles, detectBoxes, labelToDetector, countHeatmaps, statPoints, logAccuracy
from .scoreSemanticsWeights import LOSS_WEIGHTS



VERTICAL_UNITS = 24.


class ScoreSemantic:
	def __init__ (self, heatmaps, labels, confidence_table=None):
		self.data = dict({
			'__prototype': 'SemanticGraph',
			'points': [],
		})

		assert len(labels) == len(heatmaps), f'classes - heat maps count mismatch, {len(labels)} - {len(heatmaps)}'

		self.marks = []
		for i, semantic in enumerate(labels):
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


	def discern (self, truth_graph):
		points = self.data['points']
		for p in points:
			p['value'] = 0

		labels = set(map(lambda p: p['semantic'], truth_graph['points']))
		for label in labels:
			points_true = [p for p in truth_graph['points'] if p['semantic'] == label]
			points_pred = [p for p in points if p['semantic'] == label]

			for pt in points_true:
				_, finder = labelToDetector(pt['semantic'])
				pp = finder(pt, points_pred)
				if pp is not None:
					pp['value'] = 1


	def json (self):
		return self.data


class ScoreSemanticDual:
	@staticmethod
	def create (labels, unit_size, pred, target):	# pred: (batch, channel, height, width)
		layers = [[] for label in labels]
		true_count = 0

		int_pred = np.uint8(pred.cpu() * 255)
		int_target = np.uint8(target.cpu() * 255)
		for pred_map, target_map in zip(int_pred, int_target): # pred_map: (channel, height, width)
			for i, label, in enumerate(labels):
				pred_layer = pred_map[i]
				target_layer = target_map[i]

				points = countHeatmaps(target_layer, pred_layer, label, unit_size=unit_size)
				layers[i] += points
				true_count += len([p for p in points if p['value'] > 0])

		return ScoreSemanticDual(labels, layers, true_count)


	def __init__ (self, labels, layers, true_count):
		self.labels = labels
		self.layers = layers
		self.true_count = true_count


	def __add__(self, other):
		layers = [l1 + l2 for l1, l2 in zip(self.layers, other.layers)]

		return ScoreSemanticDual(self.labels, layers, self.true_count + other.true_count)


	@property
	def points_count (self):
		return sum(map(lambda points: len(points), self.layers))


	def stat (self):
		total_error = 0
		total_true_count = 0

		details = {}
		loss_weights = []
		for i, layer in enumerate(self.layers):
			label = self.labels[i]
			neg_weight, pos_weight = LOSS_WEIGHTS.get(label, (1, 1))

			true_count = len([p for p in layer if p['value'] > 0])

			confidence, error, acc, feasibility, fake_neg, fake_pos, true_neg, true_pos = statPoints(layer, true_count, neg_weight, pos_weight)
			total_error += error
			total_true_count += true_count

			details[label] = {
				'confidence': confidence,
				'accuracy': acc,
				'feasibility': feasibility,
				'true_count': true_count,
				'errors': f'{fake_neg}-|+{fake_pos}'
			}

			loss_weights.append(1 - feasibility)

		accuracy = logAccuracy(total_error, total_true_count)

		return {
			'total_error': total_error,
			'total_true_count': total_true_count,
			'total_error_rate': total_error / total_true_count,
			'accuracy': accuracy,
			'details': details,
			'loss_weights': np.array(loss_weights),
		}
