
import logging
#import torch
import matplotlib.pyplot as plt
#import matplotlib.patches as patches

from .vocal import PITCH_RANGE, PITCH_SUBDIV



class VocalViewer:
	def __init__(self, config, n_axes=4):
		self.n_axes = n_axes


	def show(self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			self.showVocalBatch(tensors)


	def showVocalBatch (self, inputs, pred=None):
		batch_size = min(self.n_axes ** 2, inputs['pitch'].shape[0])

		_, axes = plt.subplots(batch_size // self.n_axes, self.n_axes)

		#print('pred:', pred)

		plt.get_current_fig_manager().full_screen_toggle()

		pitch = inputs['pitch']
		gain = inputs['gain']
		head = inputs['head']

		for i in range(batch_size):
			ax = axes if self.n_axes == 1 else axes[i // self.n_axes, i % self.n_axes]
			ax.set_aspect(1)

			self.showVocal(ax, pitch[i], gain[i], head[i], pred[i] if pred is not None else None)

		plt.show()


	def showVocal (self, ax, pitch, gain, head, pred=None):
		#print('pred:', pred.shape)
		positive = pitch > 0
		positive_xs = positive.nonzero()[:, 0]

		width = positive_xs[-1].item()
		pred = pred[:width, 0]

		heads = (head > 0).nonzero()[:, 0]

		ax.set_ylim(PITCH_RANGE[0], PITCH_RANGE[1])
		ax.plot(positive_xs, pitch[positive] / PITCH_SUBDIV + PITCH_RANGE[0], ',')
		ax.vlines(heads, PITCH_RANGE[0], PITCH_RANGE[1], linestyles='dashed')

		if pred is not None:
			values = pred * (PITCH_RANGE[1] - PITCH_RANGE[0]) + PITCH_RANGE[0]
			#ax.step(range(width), values)
			ax.plot(values, color='y')
