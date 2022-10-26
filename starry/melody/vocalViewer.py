
import logging
import torch
import matplotlib.pyplot as plt
#import matplotlib.patches as patches

from .vocal import PITCH_RANGE, PITCH_SUBDIV, TICK_ROUND_UNIT



TONF_SCALING = 0.02


class VocalViewer:
	def __init__(self, config, n_axes=4, detail_mode=False):
		self.n_axes = n_axes
		self.detail_mode = detail_mode

		self.by_index = config['model.type'] == 'VocalAnalyzerNotationJointer'
		self.prepend_midi_zeros = config['data.args.prepend_midi_zeros']


	def show(self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			self.showVocalBatch(tensors)


	def showVocalBatch (self, inputs, pred=None):
		if self.detail_mode:
			batch_size = len(inputs['pitch'])
			for i in range(batch_size):
				pitch, gain, head, nonf, nionf, midi_pitch, midi_rtick = inputs['pitch'][i], inputs['gain'][i], inputs['head'][i], inputs['nonf'][i], inputs['nionf'][i], inputs['midi_pitch'][i], inputs['midi_rtick'][i]
				pred_i = pred[i] if pred is not None else None

				self.showVocalDetails(pitch, gain, head, nonf, nionf, midi_pitch, midi_rtick, pred_i)
		else:
			batch_size = min(self.n_axes ** 2, inputs['pitch'].shape[0])

			_, axes = plt.subplots(batch_size // self.n_axes, self.n_axes)

			#print('pred:', pred)

			plt.get_current_fig_manager().full_screen_toggle()

			pitch = inputs['pitch']
			gain = inputs['gain']
			head = inputs['head']
			tonf = inputs['tonf']

			for i in range(batch_size):
				ax = axes if self.n_axes == 1 else axes[i // self.n_axes, i % self.n_axes]
				ax.set_aspect(1)

				self.showVocal(ax, pitch[i], gain[i], head[i], tonf[i], pred[i] if pred is not None else None)

			plt.show()


	def showVocal (self, ax, pitch, gain, head, tonf, pred=None):
		#print('pred:', pred.shape)
		positive = pitch > 0
		positive_xs = positive.nonzero()[:, 0]

		width = positive_xs[-1].item()

		heads = (head > 0).nonzero()[:, 0]

		ax.set_ylim(PITCH_RANGE[0], PITCH_RANGE[1])
		ax.plot(positive_xs, pitch[positive] / PITCH_SUBDIV + PITCH_RANGE[0], ',')
		ax.vlines(heads, PITCH_RANGE[0], PITCH_RANGE[1], linestyles='dashed')

		ax.plot(gain[:width] * (PITCH_RANGE[1] - PITCH_RANGE[0]) + PITCH_RANGE[0], color='g')

		ax.plot(tonf[:width] * TONF_SCALING + PITCH_RANGE[0], color='m')

		if pred is not None:
			pred = pred[:width, 0]
			#values = pred * (PITCH_RANGE[1] - PITCH_RANGE[0]) + PITCH_RANGE[0]
			values = pred * TONF_SCALING + PITCH_RANGE[0]
			#ax.step(range(width), values)
			ax.plot(values, color='y')


	def showVocalDetails (self, pitch, gain, head, nonf, nionf, midi_pitch, midi_rtick, pred=None):
		positive = pitch > 0
		positive_xs = positive.nonzero()[:, 0]

		width = positive_xs[-1].item()

		heads = (head > 0).nonzero()[:, 0]

		axMIDI = plt.subplot2grid((5, 5), (0, 0), rowspan=4)
		axMatch = plt.subplot2grid((5, 5), (0, 1), colspan=4, rowspan=4)
		axVocal = plt.subplot2grid((5, 5), (4, 1), colspan=4)

		n_notes = int((midi_pitch > 0).sum().item())
		if self.prepend_midi_zeros:
			n_notes += 1

		#tick_range = (0, nonf[width - 1].item() // TICK_ROUND_UNIT + 4)
		tick_range = (0, 100)

		#notes = list(zip(midi_pitch.tolist(), midi_rtick.tolist()))
		#print('notes:', notes)
		axMIDI.set_xlim(PITCH_RANGE[0] + 12, PITCH_RANGE[1] - 12)
		axMIDI.set_ylim(*tick_range)
		axMIDI.plot(midi_pitch[:n_notes] + PITCH_RANGE[0], midi_rtick[:n_notes] / TICK_ROUND_UNIT, marker='o', linestyle='None')

		#axVocal.set_aspect(0.5)
		axVocal.set_ylim(PITCH_RANGE[0], PITCH_RANGE[1])
		axVocal.set_xlim(0, width)
		axVocal.plot(positive_xs, pitch[positive] / PITCH_SUBDIV + PITCH_RANGE[0], ',')
		axVocal.vlines(heads, PITCH_RANGE[0], PITCH_RANGE[1], linestyles='dashed')

		axVocal.plot(gain[:width] * (PITCH_RANGE[1] - PITCH_RANGE[0]) + PITCH_RANGE[0], color='g')

		axMatch.set_ylim(*tick_range)
		axMatch.set_xlim(0, width)
		if self.by_index:
			nonf[:width] = midi_rtick.index_select(0, nionf[:width])
		axMatch.plot(nonf[:width] / TICK_ROUND_UNIT, color='g')

		if pred is not None:
			pred = torch.nn.functional.softmax(pred[:, :min(width, pred.shape[1])], dim=0)
			if self.by_index:
				pred_tick = torch.zeros(100, pred.shape[1], dtype=torch.float)
				for i, tick in enumerate(midi_rtick):
					ti = min(99, int(tick.item()) // TICK_ROUND_UNIT)
					pred_tick[ti, :] = pred[i, :]
				pred = pred_tick

			axMatch.pcolormesh(pred, cmap='Purples')

		plt.get_current_fig_manager().full_screen_toggle()
		plt.show()
