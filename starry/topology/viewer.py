
import logging
import matplotlib.pyplot as plt



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

		x = tensors['x']
		y1 = tensors['y1']
		y2 = tensors['y2']

		for i in range(batch_size):
			ax = axes[i // 4, i % 4]
			ax.invert_yaxis()
			ax.set_aspect(1)

			for ei in range(n_seq):
				ax.vlines(x[i, ei], y1[i, ei], y2[i, ei], color='b')

		plt.get_current_fig_manager().full_screen_toggle()
		plt.show()
