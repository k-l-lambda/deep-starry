
import logging
import matplotlib.pyplot as plt
import math



class ParaffViewer:
	def __init__ (self, config, show_latent=False, show_graph=False):
		if config['_vocab'] is not None:
			self.vocab = config['_vocab'].split(',')
		self.show_latent = show_latent
		self.show_graph = show_graph


	def show(self, data):
		for i, batch in enumerate(data):
			logging.info('batch: %d', i)

			if self.show_graph:
				self.showGraph(batch)

			plt.show()


	def showBatch (self, batch, inspection):
		#logging.info('mask:', batch['body_mask'])
		target_ids = inspection['target_flat']
		truth = inspection['truth']

		def format_coord (x, y):
			x, y = math.floor(x), math.floor(y)
			token = self.vocab[y] if y < len(self.vocab) else f'[{y}]'
			pred = inspection['pred_flat'][x][y]
			return f'(seq_i, token, pred) = {x},\t{token:>8},\t{pred:8.4f}'
		plt.gca().format_coord = format_coord

		plt.get_current_fig_manager().full_screen_toggle()
		plt.pcolormesh(inspection['pred_flat'].transpose(0, 1).numpy(), cmap='RdBu', vmin=-25, vmax=30)
		plt.xlabel('seq')
		plt.ylabel('vocab id')
		plt.xticks([i for i, _ in enumerate(target_ids)], [self.vocab[id] + ('' if truth[i] else ' *') for i, id in enumerate(target_ids)], rotation=-60)
		plt.yticks([i for i, _ in enumerate(self.vocab)], [token for token in self.vocab])
		plt.colorbar()

		if self.show_latent:
			self.showLatent(inspection['mu'], inspection['logvar'])

		if self.show_graph:
			self.showGraph(batch)

		plt.show()


	def showLatent (self, mu, logvar):
		plt.figure(1)
		fig, axes = plt.subplots(1, 2)

		mu = mu[0].reshape(-1, 32).numpy()
		var = logvar[0].reshape(-1, 32).numpy()
		m0 = axes[0].pcolormesh(mu, cmap='RdBu', vmin=-1, vmax=1)
		axes[1].pcolormesh(var, cmap='RdBu', vmin=-1.5, vmax=1.)
		fig.colorbar(m0)


	def showGraph (self, batch):
		#plt.figure(2)

		plt.gca().invert_yaxis()

		id = batch['tg_id'][0]
		mask = id != 0

		id = id[mask]
		x = batch['tg_x'][0][mask]
		sy1 = batch['tg_sy1'][0][mask]
		sy2 = batch['tg_sy2'][0][mask]

		plt.plot(x, sy1, '.')
