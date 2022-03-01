
import logging



class DatasetViewer:
	def __init__(self, config):
		pass


	def show(self, data_set):
		for batch, tensors in enumerate(data_set):
			logging.info('batch: %d', batch)
