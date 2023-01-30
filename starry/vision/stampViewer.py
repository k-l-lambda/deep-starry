
import matplotlib.pyplot as plt



class StampViewer:
	def __init__(self, config):
		self.n_class = config['model.args.n_classes']
		self.labels = config['data.args.labels']

		_, self.axes = plt.subplots(self.n_class, self.n_class)

		for ir, ax_row in enumerate(self.axes):
			for ic, ax in enumerate(ax_row):
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_ticks([])

				if ic == 0:
					ax.set_ylabel(self.labels[ir])

				if ir == 0:
					ax.set_title(self.labels[ic])

		plt.axis('off')
		plt.get_current_fig_manager().full_screen_toggle()


	def appendExample (self, image, label, pred):
		#print('ex:', image.shape, label, pred)
		x = label.item()
		y = pred.argmax().item()

		self.axes[x][y].imshow(image[0])

		plt.draw()
		plt.pause(0.5)
