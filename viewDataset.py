
import sys
import os
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.melody.notationViewer import NotationViewer
from starry.melody.frameViewer import FrameViewer
from starry.melody.vocalViewer import VocalViewer
from starry.topology.viewer import DatasetViewer as TopoViewer



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-s', '--splits', type=str, default='0/10')
	parser.add_argument('-e', '--eventTopo', action='store_true', help='show event topology data')
	parser.add_argument('-mx', '--show_matrix', action='store_true', help='show matrix view')
	parser.add_argument('-f', '--frame', action='store_true', help='show melody frame data')
	parser.add_argument('-v', '--vocal', action='store_true', help='show vocal data')
	parser.add_argument('-ax', '--n_axes', type=int, default=4)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)
	if args.splits is not None:
		config['data.splits'] = args.splits

	data, = loadDataset(config, data_dir=DATA_DIR)
	if args.eventTopo:
		viewer = TopoViewer(config, n_axes=args.n_axes, show_matrix=args.show_matrix)
	elif args.frame:
		viewer = FrameViewer(config, n_axes=args.n_axes)
	elif args.vocal:
		viewer = VocalViewer(config, n_axes=args.n_axes)
	else:
		viewer = NotationViewer(config, n_axes=args.n_axes)

	viewer.show(data)


if __name__ == '__main__':
	main()
