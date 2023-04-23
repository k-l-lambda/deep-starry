
import logging
import matplotlib.pyplot as plt



class ParaffViewer:
	def __init__ (self, config):
		pass


	def show (self, batch, inspection):
		#logging.info('mask:', batch['body_mask'])
		plt.get_current_fig_manager().full_screen_toggle()
		plt.pcolormesh(inspection['pred_flat'].transpose(0, 1).numpy(), cmap='RdBu')
		plt.xlabel('seq')
		plt.ylabel('vocab id')
		plt.colorbar()
		plt.show()
