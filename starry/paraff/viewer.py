
import logging
import matplotlib.pyplot as plt
import math



class ParaffViewer:
	def __init__ (self, config):
		self.vocab = config['_vocab'].split(',')


	def show (self, batch, inspection):
		#logging.info('mask:', batch['body_mask'])
		target_ids = inspection['target_flat']
		truth = inspection['truth']

		def format_coord (x, y):
			x, y = math.floor(x), math.floor(y)
			token = self.vocab[y] if y < len(self.vocab) else f'[{y}]'
			pred = inspection['pred_flat'][x][y]
			return f'seq_i={x}, token={token}, pred={pred:.4f}'
		plt.gca().format_coord = format_coord

		plt.get_current_fig_manager().full_screen_toggle()
		plt.pcolormesh(inspection['pred_flat'].transpose(0, 1).numpy(), cmap='RdBu', vmin=-25, vmax=30)
		plt.xlabel('seq')
		plt.ylabel('vocab id')
		plt.xticks([i for i, _ in enumerate(target_ids)], [self.vocab[id] + ('' if truth[i] else ' *') for i, id in enumerate(target_ids)], rotation=-60)
		plt.yticks([i for i, _ in enumerate(self.vocab)], [token for token in self.vocab])
		plt.colorbar()
		plt.show()
