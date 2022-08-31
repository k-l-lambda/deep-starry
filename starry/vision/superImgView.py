
import logging
import matplotlib.pyplot as plt



class SuperImgView:
	def __init__(self, config):
		self.config = config


	def show (self, data_set):
		for i, batch in enumerate(data_set):
			logging.info(f'batch: {i}')

			self.showBatch(batch)


	def showBatch (self, batch, pred=None):
		x, y = batch
		#print('x:', x.shape)
		#print('y:', y.shape)
		batch_size = len(x)

		_, axes = plt.subplots(2 if pred is None else 3, batch_size)

		def ax (j, i):
			return axes[j] if batch_size == 1 else axes[j][i]

		for i in range(batch_size):
			feature = x[i].permute(1, 2, 0)
			target = y[i].permute(1, 2, 0)

			ax(0, i).imshow(feature)
			ax(1, i).imshow(target)

			if pred is not None:
				p = pred[i].permute(1, 2, 0)
				ax(2, i).imshow(p)

		plt.get_current_fig_manager().full_screen_toggle()
		plt.show()
