
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .event_element import EventElementType, StemDirection



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
				dots = tensors['division'][0, ei]
				beam = tensors['division'][0, ei]
				x = xs[i, ei]
				y = y2[i, ei] if direction == StemDirection.u else y1[i, ei]

				if elem_type[ei] == EventElementType.REST:
					ax.add_patch(patches.Rectangle((x - 0.6, y - 0.6), 1.2, 1.2, fill=division >= 2, facecolor='g', edgecolor='g'))
				elif elem_type[ei] == EventElementType.CHORD:
					# stem
					ax.vlines(x, y1[i, ei], y2[i, ei], color='b')

					# head
					head_x = x
					head_x = head_x - 0.7 if direction == StemDirection.u else head_x
					head_x = head_x + 0.7 if direction == StemDirection.d else head_x
					ax.add_patch(patches.Ellipse((head_x, y), 1.4, 0.8, fill=division >= 2, facecolor='b', edgecolor='b'))
				elif elem_type[ei] == EventElementType.BOS:
					ax.add_patch(patches.Polygon([(x - 0.3, y - 1), (x + 0.3, y - 1), (x, y),], fill=True, facecolor='r'))
				elif elem_type[ei] == EventElementType.EOS:
					ax.add_patch(patches.Polygon([(x - 0.3, y + 1), (x + 0.3, y + 1), (x, y),], fill=True, facecolor='r'))

		plt.get_current_fig_manager().full_screen_toggle()
		plt.show()
