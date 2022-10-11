
import logging
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class VocalViewer:
	def __init__(self, config, n_axes=4):
		self.n_axes = n_axes


	def show(self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			self.showVocal(tensors)


	def showVocal (self, inputs, pred=None):
		batch_size = min(self.n_axes ** 2, inputs['pitch'].shape[0])

		_, axes = plt.subplots(batch_size // self.n_axes, self.n_axes)

		print('inputs:', inputs)

		'''plt.get_current_fig_manager().full_screen_toggle()

		ci = inputs['ci']
		ct, cp, cv = inputs['criterion']
		st, sp, sv = inputs['sample']

		matching, src, tar = pred

		for i in range(batch_size):
			ax = axes if self.n_axes == 1 else axes[i // self.n_axes, i % self.n_axes]
			ax.set_aspect(1)

			self.showMatching(ax, (ct[i], cp[i], cv[i]), (st[i], sp[i], sv[i]), ci[i], matching[i] if matching is not None else None)

		plt.show()'''
