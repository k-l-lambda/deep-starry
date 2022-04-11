
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from .event_element import EventElementType, StemDirection, BeamType



class DatasetViewer:
	def __init__(self, config, n_axes=4, show_matrix=False):
		self.n_axes = n_axes
		self.show_matrix = show_matrix


	def show(self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			self.showEventTopology(tensors)


	def showEventCluster (self, ax, inputs, pred_rec=None):
		n_seq = inputs['feature'].shape[0]

		xs = inputs['x']
		y1 = inputs['y1']
		y2 = inputs['y2']
		elem_type = inputs['type']

		for ei in range(n_seq):
			direction = inputs['stemDirection'][ei]
			division = inputs['division'][ei]
			dots = inputs['dots'][ei]
			beam = inputs['beam'][ei]
			warped = inputs['timeWarped'][ei]
			grace = inputs['grace'][ei]
			fullMeasure = inputs['fullMeasure'][ei]
			tick = inputs['tick'][ei]
			features = inputs['feature'][ei]

			x = xs[ei]
			y = y2[ei] if direction == StemDirection.u else y1[ei]
			ty = y2[ei] if direction == StemDirection.d else y1[ei]

			if elem_type[ei] == EventElementType.REST:
				color = 'gray' if fullMeasure else 'g'
				ax.add_patch(patches.Rectangle((x - 0.6, y - 0.6), 1.2, 1.2, fill=division >= 2, facecolor=color, edgecolor=color))

				if division == 0:
					ax.add_patch(patches.Rectangle((x - 0.6, y - 0.6), 1.2, 0.6, fill=True, facecolor=color))
				elif division == 1:
					ax.add_patch(patches.Rectangle((x - 0.6, y), 1.2, 0.6, fill=True, facecolor=color))

				# flags
				if division > 2:
					for fi in range(division - 2):
						fy = y + fi * 0.4
						ax.hlines(fy, x - 1.2, x - 0.6, color='g')

				# dots
				for di in range(dots):
					dx = x + 1.2 + di * 0.4
					ax.add_patch(patches.Circle((dx, y), 0.16, fill=True, facecolor='g'))
			elif elem_type[ei] == EventElementType.CHORD:
				color = 'c' if warped else 'b'
				scale = 0.6 if grace else 1

				# stem
				ax.vlines(x, y1[ei], y2[ei], color=color)

				# head
				head_x = x
				head_x = head_x - 0.7 * scale if direction == StemDirection.u else head_x
				head_x = head_x + 0.7 * scale if direction == StemDirection.d else head_x
				ax.add_patch(patches.Ellipse((head_x, y), 1.4 * scale, 0.8 * scale, fill=division >= 2, facecolor=color, edgecolor=color))

				# flags
				if division > 2:
					left, right = x, x + 0.7
					oy = 0
					if beam == BeamType.Open:
						right = x + 1.2
					elif beam == BeamType.Continue:
						left, right = x - 0.6, x + 0.6
					elif beam == BeamType.Close:
						left, right = x - 1.2, x
					else:
						oy = (0.5 if direction == StemDirection.u else -0.5)
					for fi in range(division - 2):
						fy = ty + oy + fi * (0.9 if direction == StemDirection.u else -0.9)
						ax.hlines(fy, left, right, color=color)

				# dots
				for di in range(dots):
					dx = x + (0.4 if direction == StemDirection.u else 1.8) + di * 0.4
					ax.add_patch(patches.Circle((dx, y), 0.16, fill=True, facecolor=color))
			elif elem_type[ei] == EventElementType.BOS:
				ax.add_patch(patches.Polygon([(x - 0.3, y - 1), (x + 0.3, y - 1), (x, y),], fill=True, facecolor='r'))
			elif elem_type[ei] == EventElementType.EOS:
				ax.add_patch(patches.Polygon([(x - 0.3, y + 1), (x + 0.3, y + 1), (x, y),], fill=True, facecolor='r'))

			# features
			ax.text(x - 0.2, y2[ei] + 0.9, '%d' % tick, color='k', fontsize='small', ha='right')

			def drawDot (i, li, y, color):
				alpha = math.tanh(features[i].item() * 0.4)
				ax.add_patch(patches.Circle((x - 0.4 - li * 0.32, y), 0.14, fill=True, facecolor=color, alpha=alpha))
				ax.add_patch(patches.Circle((x - 0.4 - li * 0.32, y + 0.16), 0.04, fill=True, facecolor='dimgray'))

			if elem_type[ei] in [EventElementType.CHORD, EventElementType.REST]:
				for i in range(0, 3):	# division 0,1,2
					drawDot(i, i, y2[ei], 'orange')
				for i in range(3, 7):	# division 3-
					drawDot(i, i - 3, y2[ei] - 0.4, 'orange')
				for i in range(7, 9):	# dots
					drawDot(i, i - 7, y2[ei] - 1.1, 'lawngreen')
				for i in range(9, 12):	# beam
					drawDot(i, 11 - i, y2[ei] - 1.7, 'navy')
				for i in range(12, 14):	# stemDirection
					drawDot(i, i - 12, y2[ei] - 2.3, 'violet')
				drawDot(14, 0, y2[ei] - 2.9, 'cyan')

			# predicted features
			if pred_rec is not None:
				pred_event = {k: seq[ei] for k, seq in pred_rec.items()}
				pred_event = {k: v.item() if v.numel() == 1 else v.tolist() for k, v in pred_event.items()}
				#print('pred_event:', pred_event)

				ax.text(x + 0.2, y2[ei] + 0.9, '%d' % round(pred_event['tick']), color='maroon', fontsize='small', ha='left')

				def drawDot (value, truth, li, y, color):
					alpha = value
					ax.add_patch(patches.Circle((x + 0.5 + li * 0.32, y), 0.14, fill=True, facecolor=color, alpha=alpha))
					ax.add_patch(patches.Circle((x + 0.5 + li * 0.32, y + 0.16), 0.06 if truth else 0.04, fill=True, facecolor='g' if truth else 'brown'))

				if elem_type[ei] in [EventElementType.CHORD, EventElementType.REST]:
					for i in range(0, 3):	# division 0,1,2
						drawDot(pred_event['division'][i], i == division, i, y2[ei], 'orange')
					for i in range(3, 7):	# division 3-
						drawDot(pred_event['division'][i], i == division, i - 3, y2[ei] - 0.4, 'orange')
					for i, v in enumerate(pred_event['dots']):
						drawDot(v, i == dots, i, y2[ei] - 1.1, 'lawngreen')
					for i, v in enumerate(pred_event['beam']):
						drawDot(v, i == beam, i, y2[ei] - 1.7, 'navy')
					for i, v in enumerate(pred_event['stemDirection']):
						drawDot(v, i == direction, i, y2[ei] - 2.3, 'violet')

					drawDot(pred_event['grace'], grace, 0, y2[ei] - 2.9, 'cyan')
					drawDot(pred_event['timeWarped'], warped, 1, y2[ei] - 2.9, 'darkcyan')
					drawDot(pred_event['fullMeasure'], fullMeasure, 2, y2[ei] - 2.9, 'yellow')
					drawDot(pred_event['fake'], False, 3, y2[ei] - 2.9, 'black')


	def showMatrix (self, ax, truth_matrix, pred_matrix=None):
		#print('showMatrix:', truth_matrix, pred_matrix)
		n_seq = truth_matrix.shape[0]
		assert truth_matrix.shape[1] == n_seq and (pred_matrix is None or (pred_matrix.shape[0] == n_seq and pred_matrix.shape[1] == n_seq))

		ax.set_xlim(-0.2, n_seq + 0.1)
		ax.set_ylim(-0.2, n_seq + 0.1)
		for i in range(n_seq):
			for j in range(n_seq):
				truth = truth_matrix[i, j] > 0
				#positive = pred_matrix[i, j] > 0.5 if pred_matrix is not None else None
				color = 'g' if truth else 'r'

				if truth:
					ax.add_patch(patches.Rectangle((j, i), 0.9, 0.9, fill=False, edgecolor='k'))

				if pred_matrix is not None:
					ax.add_patch(patches.Rectangle((j + 0.1, i + 0.1), 0.7, 0.7, fill=True, facecolor=color, alpha=pred_matrix[i, j].item()))


	def showEventTopology (self, inputs, pred=(None, None)):
		batch_size = min(self.n_axes ** 2, inputs['feature'].shape[0])

		_, axes = plt.subplots(batch_size // self.n_axes, self.n_axes)

		plt.get_current_fig_manager().full_screen_toggle()

		if self.show_matrix:
			plt.figure(1)
			_, axesM = plt.subplots(batch_size // self.n_axes, self.n_axes)

		pred_rec, pred_matrixH = pred

		for i in range(batch_size):
			ax = axes if self.n_axes == 1 else axes[i // self.n_axes, i % self.n_axes]
			ax.invert_yaxis()
			ax.set_aspect(1)

			cluster_inputs = {k: tensor[i] for k, tensor in inputs.items()}
			cluster_rec = pred_rec and {k: tensor[i] for k, tensor in pred_rec.items()}

			self.showEventCluster(ax, cluster_inputs, cluster_rec)

			if self.show_matrix:
				ax = axesM if self.n_axes == 1 else axesM[i // self.n_axes, i % self.n_axes]
				ax.set_aspect(1)

				n_seq = len(cluster_inputs['type'])
				self.showMatrix(ax, cluster_inputs['matrixH'].reshape((n_seq - 1, -1)),
					pred_matrixH and pred_matrixH[i].reshape((n_seq - 1, -1)))

		plt.show()
