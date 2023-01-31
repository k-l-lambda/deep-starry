
import math
import numpy as np
import matplotlib.pyplot as plt



class ExpandableCell:
	def __init__ (self, unit_size, capacity=16):
		self.unit_size = unit_size
		self.capacity = capacity
		self.size = 0
		self.screen = None

		levels = int(math.ceil(math.sqrt(capacity)))

		coords = []
		for l in range(levels):
			coords += [(l, i) for i in range(l)]
			coords += [(i, l) for i in range(l + 1)]

		self.coords = coords


	def append (self, image):
		x, y = self.coords[self.size % self.capacity]
		if self.size < self.capacity:
			if x == 0:
				new_screen = np.zeros(((y + 1) * self.unit_size, (y + 1) * self.unit_size))
			elif y == 0:
				new_screen = np.zeros((x * self.unit_size, (x + 1) * self.unit_size))

			if x * y == 0:
				if self.screen is not None:
					h, w = self.screen.shape
					new_screen[:h, :w] = self.screen
				self.screen = new_screen

		self.screen[y * self.unit_size:(y + 1) * self.unit_size, x * self.unit_size:(x + 1) * self.unit_size] = image

		self.size += 1


class StampViewer:
	def __init__ (self, config, cell_capacity=16):
		self.n_class = config['model.args.n_classes']
		self.labels = config['data.args.labels']

		unit_size = config['data.args.crop_size']

		self.cells = [[ExpandableCell(unit_size, cell_capacity) for x in range(self.n_class)] for y in range(self.n_class)]

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

		self.cells[y][x].append(image[0].numpy())
		self.axes[y][x].imshow(self.cells[y][x].screen)

		plt.draw()
		plt.pause(0.01)
