
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from .event_element import EventElementType, StemDirection, BeamType



class DatasetViewer:
	def __init__(self, config):
		pass


	def show(self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			self.showEventTopology(tensors)


	def showEventCluster (self, ax, tensors):
		n_seq = tensors['feature'].shape[0]

		xs = tensors['x']
		y1 = tensors['y1']
		y2 = tensors['y2']
		elem_type = tensors['type']

		for ei in range(n_seq):
			direction = tensors['stemDirection'][ei]
			division = tensors['division'][ei]
			dots = tensors['dots'][ei]
			beam = tensors['beam'][ei]
			warped = tensors['timeWarped'][ei]
			grace = tensors['grace'][ei]
			fullMeasure = tensors['fullMeasure'][ei]
			tick = tensors['tick'][ei]
			features = tensors['feature'][ei]

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
			ax.text(x - 0.4, y2[ei] + 0.9, '%d' % tick, color='k', fontsize='x-small', ha='right')
			if elem_type[ei] in [EventElementType.CHORD, EventElementType.REST]:
				for i in range(0, 7):
					alpha = math.tanh(features[i].item())
					ax.add_patch(patches.Circle((x - 0.4 - i * 0.4, y2[ei] - 0.4), 0.16, fill=True, facecolor='orange', alpha=alpha))
					ax.add_patch(patches.Circle((x - 0.4 - i * 0.4, y2[ei] - 0.2), 0.04, fill=True, facecolor='k'))


	def showEventTopology (self, tensors):
		batch_size = min(16, tensors['feature'].shape[0])

		_, axes = plt.subplots(batch_size // 4, 4)

		for i in range(batch_size):
			ax = axes[i // 4, i % 4]
			ax.invert_yaxis()
			ax.set_aspect(1)

			cluster_tensors = {k: tensor[i] for k, tensor in tensors.items()}
			self.showEventCluster(ax, cluster_tensors)

		plt.get_current_fig_manager().full_screen_toggle()
		plt.show()
