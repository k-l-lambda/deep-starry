
import sys
import os
import argparse
import logging
from tqdm import tqdm
import time

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


class Profiler:
	def __init__ (self):
		self.t0 = 0
		self.costs = {}


	def check (self, name):
		now = time.time()
		cost = now - self.t0
		self.t0 = now

		#logging.info('check: [%f]	%s', time.time(), name)

		self.costs[name] = self.costs.get(name, 0)
		self.costs[name] += cost


	def run (self, data):
		data.dataset.profile_check = self.check

		t0 = time.time()
		self.t0 = t0
		n_batch = 0

		for batch in tqdm(data):
			n_batch += 1

		total_cost = time.time() - t0

		# dump results
		logging.info('n_batch: %d', n_batch)
		for k, v in self.costs.items():
			logging.info('	%-16s:	%f,	%.2f%%', k, v / n_batch, v / total_cost * 100)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-s', '--splits', type=str, default='1/10')
	parser.add_argument('-dv', '--device', type=str, default='cuda')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)
	if args.splits is not None:
		config['data.splits'] = args.splits

	data, = loadDataset(config, data_dir=DATA_DIR, device=args.device)

	profiler = Profiler()
	profiler.run(data)


if __name__ == '__main__':
	main()
