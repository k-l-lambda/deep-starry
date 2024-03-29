
import sys
import os
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.vision.datasetViewer import DatasetViewer
from starry.topology.viewer import DatasetViewer as TopoViewer
from starry.paraff.viewer import ParaffViewer



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')
VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-s', '--splits', type=str, default='0/10')
	parser.add_argument('-g', '--gauge', action='store_true', help='gauge mode')
	parser.add_argument('-e', '--eventTopo', action='store_true', help='show event topology data')
	parser.add_argument('-p', '--paraff', action='store_true', help='show paraff data')
	parser.add_argument('-mx', '--show_matrix', action='store_true', help='show matrix view')
	parser.add_argument('-tg', '--show_graph', action='store_true', help='show timewise graph')
	parser.add_argument('-ax', '--n_axes', type=int, default=4)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)
	if args.splits is not None:
		config['data.splits'] = args.splits

	topo = args.eventTopo
	data_dir = DATA_DIR if (topo or args.paraff) else VISION_DATA_DIR

	data, = loadDataset(config, data_dir=data_dir)
	if args.paraff:
		viewer = ParaffViewer(config, show_graph=args.show_graph)
	elif topo:
		viewer = TopoViewer(config, n_axes=args.n_axes, show_matrix=args.show_matrix)
	else:
		viewer = DatasetViewer(config, gauge_mode=args.gauge)

	viewer.show(data)


if __name__ == '__main__':
	main()
