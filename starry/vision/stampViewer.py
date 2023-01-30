
class StampViewer:
	def __init__(self, config):
		self.n_class = config['model.args.n_classes']


	def appendExample (self, image, label, pred):
		print('ex:', image.shape, label, pred)
