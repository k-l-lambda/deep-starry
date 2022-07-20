
import logging
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches



TIME_SCALE = 1e-3


class FrameViewer:
	def __init__(self, config, n_axes=4):
		self.n_axes = n_axes


	def show(self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			self.showMelody(tensors)


	def showMelody (self, inputs, pred=(None, None, None)):
		batch_size = min(self.n_axes ** 2, inputs['ci'].shape[0])

		_, axes = plt.subplots(batch_size // self.n_axes, self.n_axes * 2)

		plt.get_current_fig_manager().full_screen_toggle()

		ci = inputs['ci']
		ct, cf = inputs['criterion']
		st, sf = inputs['sample']

		matching, src, tar = pred

		for i in range(batch_size):
			ax_row = axes[i // self.n_axes] if self.n_axes > 1 else axes
			ax1, ax2 = ax_row[(i % self.n_axes) * 2], ax_row[(i % self.n_axes) * 2 + 1]
			ax1.set_aspect(1)

			criterion, sample, cii = (ct[i], cf[i]), (st[i], sf[i]), ci[i]
			mi = matching[i] if matching is not None else None

			self.showMap(ax1, criterion, sample, cii, mi)
			self.showSequences(ax2, criterion, sample, cii, mi)

		plt.show()


	def showMap (self, ax, criterion, sample, ci, matching=None):
		ct, cf = criterion
		st, sf = sample

		st_n0 = st[sf.sum(dim=-1) != 0]
		ct_n0 = ct[cf.sum(dim=-1) != 0]
		left, right = torch.min(st_n0).item(), torch.max(st_n0).item()
		bottom, top = torch.min(ct_n0).item(), torch.max(ct_n0).item()
		ax.set_xlim(left * TIME_SCALE - 1, right * TIME_SCALE + 1)
		ax.set_ylim(bottom * TIME_SCALE - 1, top * TIME_SCALE + 1)

		# origin mark
		ax.vlines(0, -0.16, 0.16, color='k')
		ax.hlines(0, -0.16, 0.16, color='k')

		for si, snote in enumerate(zip(st, sf)):
			cii = ci[si].item() - 1
			sti, sfi = snote
			sti, sfi = sti.item(), sfi
			if sfi.sum().item() > 0 and cii >= 0:
				cti, cfi = ct[cii].item(), cf[cii].tolist()
				#print('si:', si, cii, sti, cti)

				ax.add_patch(patches.Circle((sti * TIME_SCALE, cti * TIME_SCALE), 0.1, fill=True, facecolor='g'))

		if matching is not None:
			#print('matching:', matching.shape, sp.shape)
			for si, ps in enumerate(matching):
				sti, sfi = st[si].item(), sf[si]
				if sfi.sum().item() > 0:
					cii_truth = ci[si].item() - 1
					cii_pred = torch.argmax(ps).item()
					for cii, p in enumerate(ps):
						if p > 0:
							cti, cfi = ct[cii].item(), cf[cii].tolist()
							is_truth = cii == cii_truth
							is_pred = cii == cii_pred

							ax.add_patch(patches.Circle((sti * TIME_SCALE, cti * TIME_SCALE), p * 0.4,
								fill=True, alpha=0.8 if is_pred else 0.3,
								facecolor=('c' if is_truth else 'r') if is_pred else 'm'))


	def showSequences (self, ax, criterion, sample, ci, matching=None):
		ct, cf = criterion
		st, sf = sample

		left, right = min(st.min().item(), ct.min().item()), max(st.max().item(), ct.max().item())

		ax.set_xlim(left - 1e+3, right + 1e+3)
		ax.set_ylim(-112, 112)

		#print('ct:', ct.tolist())
		for ci, (cti, cfi) in enumerate(zip(ct, cf)):
			cti, cfi = cti.item(), cfi.tolist()
			w = (ct[ci + 1].item() - cti) if ci < len(ct) - 1 else 0.5e+3
			for p, v in enumerate(cfi):
				ax.add_patch(patches.Rectangle((cti, p + 21), w, 1, fill=True, facecolor='g', alpha=v))


		for si, snote in enumerate(zip(st, sf)):
			sti, sfi = snote
			sti, sfi = sti.item(), sfi.tolist()
			w = (st[si + 1].item() - sti) if si < len(st) - 1 else 0.5e+3
			for p, v in enumerate(sfi):
				ax.add_patch(patches.Rectangle((sti, -(p + 21)), w, 1, fill=True, facecolor='b', alpha=v))
