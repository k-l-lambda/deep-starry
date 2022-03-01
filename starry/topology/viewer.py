
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .event_element import EventElementType, StemDirection, BeamType



class DatasetViewer:
	def __init__(self, config):
		pass


	def show(self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			self.showEventTopology(tensors)


	def showEventTopology (self, tensors):
		batch_size = min(16, tensors['feature'].shape[0])
		n_seq = tensors['feature'].shape[1]

		_, axes = plt.subplots(batch_size // 4, 4)

		xs = tensors['x']
		y1 = tensors['y1']
		y2 = tensors['y2']
		elem_type = tensors['type'][0]
		stemDirection = tensors['stemDirection'][0]

		for i in range(batch_size):
			ax = axes[i // 4, i % 4]
			ax.invert_yaxis()
			ax.set_aspect(1)

			for ei in range(n_seq):
				direction = stemDirection[ei]
				division = tensors['division'][0, ei]
				dots = tensors['dots'][0, ei]
				beam = tensors['beam'][0, ei]
				warped = tensors['timeWarped'][0, ei]
				grace = tensors['grace'][0, ei]
				x = xs[i, ei]
				y = y2[i, ei] if direction == StemDirection.u else y1[i, ei]
				ty = y2[i, ei] if direction == StemDirection.d else y1[i, ei]

				if elem_type[ei] == EventElementType.REST:
					ax.add_patch(patches.Rectangle((x - 0.6, y - 0.6), 1.2, 1.2, fill=division >= 2, facecolor='g', edgecolor='g'))

					# flags
					if division > 2:
						for fi in range(division - 2):
							fy = y + fi * 0.4
							ax.hlines(fy, x + 0.6, x + 1.2, color='g')

					# dots
					for di in range(dots):
						dx = x + 0.8 + di * 0.4
						ax.add_patch(patches.Circle((dx, y), 0.16, fill=True, facecolor='g'))
				elif elem_type[ei] == EventElementType.CHORD:
					color = 'c' if warped else 'b'
					scale = 0.6 if grace else 1

					# stem
					ax.vlines(x, y1[i, ei], y2[i, ei], color=color)

					# head
					head_x = x
					head_x = head_x - 0.7 * scale if direction == StemDirection.u else head_x
					head_x = head_x + 0.7 * scale if direction == StemDirection.d else head_x
					ax.add_patch(patches.Ellipse((head_x, y), 1.4 * scale, 0.8 * scale, fill=division >= 2, facecolor=color, edgecolor=color))

					# flags
					if division > 2:
						left, right = x, x + 0.7
						if beam == BeamType.Open:
							right = x + 1.2
						elif beam == BeamType.Continue:
							left, right = x - 0.6, x + 0.6
						elif beam == BeamType.Close:
							left, right = x - 1.2, x
						for fi in range(division - 2):
							fy = ty + fi * (0.9 if direction == StemDirection.u else -0.9)
							ax.hlines(fy, left, right, color=color)

					# dots
					for di in range(dots):
						dx = x + (0.4 if direction == StemDirection.u else 1.8) + di * 0.4
						ax.add_patch(patches.Circle((dx, y), 0.16, fill=True, facecolor=color))
				elif elem_type[ei] == EventElementType.BOS:
					ax.add_patch(patches.Polygon([(x - 0.3, y - 1), (x + 0.3, y - 1), (x, y),], fill=True, facecolor='r'))
				elif elem_type[ei] == EventElementType.EOS:
					ax.add_patch(patches.Polygon([(x - 0.3, y + 1), (x + 0.3, y + 1), (x, y),], fill=True, facecolor='r'))

		plt.get_current_fig_manager().full_screen_toggle()
		plt.show()
