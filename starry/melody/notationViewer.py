
import logging
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .notation import PITCH_BOS



TIME_SCALE = 1e-3


class NotationViewer:
	def __init__(self, config, n_axes=4):
		self.n_axes = n_axes


	def show(self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			self.showMelody(tensors)


	def showMelody (self, inputs, pred=(None, None, None)):
		batch_size = min(self.n_axes ** 2, inputs['ci'].shape[0])

		_, axes = plt.subplots(batch_size // self.n_axes, self.n_axes)

		plt.get_current_fig_manager().full_screen_toggle()

		ci = inputs['ci']
		ct, cp, cv = inputs['criterion']
		st, sp, sv, sg, sgm = inputs['sample']

		matching, src, tar = pred

		for i in range(batch_size):
			ax = axes if self.n_axes == 1 else axes[i // self.n_axes, i % self.n_axes]
			ax.set_aspect(1)

			#print('sg:', sg[i])
			self.showMatching(ax, (ct[i], cp[i], cv[i]), (st[i], sp[i], sv[i], sgm[i]), ci[i], matching[i] if matching is not None else None)

		plt.show()


	def showMatching (self, ax, criterion, sample, ci, matching=None):
		ct, cp, cv = criterion
		st, sp, sv, sgm = sample

		st_n0 = st[sp != 0]
		ct_n0 = ct[cp != 0]
		left, right = torch.min(st_n0).item(), torch.max(st_n0).item()
		bottom, top = torch.min(ct_n0).item(), torch.max(ct_n0).item()
		ax.set_xlim(left * TIME_SCALE - 1, right * TIME_SCALE + 1)
		ax.set_ylim(bottom * TIME_SCALE - 1, top * TIME_SCALE + 1)

		# origin mark
		ax.vlines(0, -0.16, 0.16, color='k')
		ax.hlines(0, -0.16, 0.16, color='k')

		for si, snote in enumerate(zip(st, sp, sv, sgm)):
			cii = ci[si].item() - 1
			sti, spi, svi, sgmi = snote
			sti, spi, svi, sgmi = sti.item(), spi.item(), svi.item(), sgmi.item()
			if spi > 0 and cii >= 0:
				cti, cpi, cvi = ct[cii].item(), cp[cii].item(), cv[cii].item()
				#print('si:', si, cii, sti, cti, sgmi)

				x, y = sti * TIME_SCALE, cti * TIME_SCALE
				if cpi == PITCH_BOS:
					ax.add_patch(patches.Polygon([(x, y), (x - 0.1, y - 0.45), (x + 0.1, y - 0.45)], fill=True, facecolor='g'))
				elif not sgmi:
					ax.add_patch(patches.Rectangle((x - 0.15, y - 0.15), 0.3, 0.3, fill=True, facecolor='g'))
				else:
					ax.add_patch(patches.Circle((x, y), 0.1, fill=True, facecolor='g'))

		if matching is not None:
			#print('matching:', matching.shape, sp.shape)
			for si, ps in enumerate(matching):
				sti, spi = st[si].item(), sp[si].item()
				if spi > 0:
					cii_truth = ci[si].item() - 1
					cii_pred = torch.argmax(ps).item()
					for cii, p in enumerate(ps):
						if p > 0:
							cti, cpi, cvi = ct[cii].item(), cp[cii].item(), cv[cii].item()
							is_truth = cii == cii_truth
							is_pred = cii == cii_pred

							ax.add_patch(patches.Circle((sti * TIME_SCALE, cti * TIME_SCALE), p * 0.4,
								fill=True, alpha=0.7 if is_pred else 0.3,
								facecolor=('c' if is_truth else 'r') if is_pred else 'm'))
