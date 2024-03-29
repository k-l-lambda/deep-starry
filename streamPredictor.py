
import sys
import time
import argparse
import logging
import json
import base64
import os
import PIL.Image

from starry.utils.config import Configuration
from starry.topology.predictor import TopologyPredictorH, TopologyPredictorHV
from starry.topology.predictorEvent import EvtopoPredictor
from starry.vision.semantic_predictor import SemanticPredictor
from starry.vision.mask_predictor import MaskPredictor
from starry.vision.gauge_predictor import GaugePredictor
from starry.vision.layout_predictor import LayoutPredictor
from starry.vision.scorePageProcessor import ScorePageProcessor
from starry.vision.scoreSemanticProcessor import ScoreSemanticProcessor
from starry.vision.semanticClusterPredictor import SemanticClusterPredictor
from starry.utils.zero_server import ZeroServer



logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# workaround incomplete registered_extensions list problem in PIL
PIL.Image.registered_extensions()
#logging.info('extensions: %s', PIL.Image.registered_extensions().keys())



LOGFILE = os.environ.get('PREDICTOR_LOG')
DIST_DATE = os.environ.get('DIST_DATE')


PREDICTOR_FACTORY = {
	'topology':			TopologyPredictorH,
	'topologyHV':		TopologyPredictorHV,
	'evtopo':			EvtopoPredictor,
	'semantic':			SemanticPredictor,
	'mask':				MaskPredictor,
	'gauge':			GaugePredictor,
	'layout':			LayoutPredictor,
	'scorePage':		ScorePageProcessor,
	'semanticBatch':	ScoreSemanticProcessor,
	'semanticCluster':	SemanticClusterPredictor,
}


def parseInputLine (line):
	if not ':' in line:
		buffer = base64.b64decode(line)
		return {'buffer': buffer}

	colon = line.index(':')
	protocol = line[:colon]
	body = line[colon + 1:]

	if protocol == 'base64':
		buffer = base64.b64decode(body)
		return {'buffer': buffer}
	elif protocol == 'json':
		return json.loads(body)
	elif protocol == 'echo':
		return {'_echo': body[:-1]}
	else:
		raise ValueError(f'unexpected input protocol: {protocol}')


def outputData (data):
	assert not ('\n' in data), 'stream output data contains "\\n"'
	logging.debug('output json: %d', len(data))
	print(data, end='\n')


def session (predictor):
	logging.info('Waiting for request...')

	args = []
	kwarg = {}

	while True:
		data = sys.stdin.readline()
		logging.info('data read: %d', len(data))
		if not data or len(data) <= 1:
			break

		data = parseInputLine(data)
		if data.get('buffer'):
			if not len(args):
				args.append([])
			args[0].append(data['buffer'])
		else:
			kwarg = {**kwarg, **data}

	if kwarg.get('_echo'):
		print(kwarg['_echo'], end='\n\n', flush = True)
		logging.info(f'Echoed: "{kwarg["_echo"]}"')

		return True

	logging.info('Predicting...')
	t0 = time.time()
	count = 0
	for result in predictor.predict(*args, **kwarg):
		outputData(json.dumps(result))
		count += 1
	t1 = time.time()

	print('', end='\n', flush = True)
	logging.info(f'Prediction results ({count}) sent. ({t1 - t0:.3f}s)')

	return True


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-m', '--mode', type=str, default='topology', help='predictor mode')
	parser.add_argument('-dv', '--device', type=str, default='cpu', help='cpu or cuda')
	parser.add_argument('-i', '--inspect', action='store_true', help='inspect mode')
	parser.add_argument('-p', '--port', type=int, help='zmq server port')

	args = parser.parse_args()

	if LOGFILE is not None:
		log_path = LOGFILE % {'mode': args.mode, 'port': args.port}
		logging.basicConfig(format='%(asctime)s.%(msecs)03d	%(levelname)s	%(message)s', datefmt='%Y%m%d %H:%M:%S', level=logging.INFO,
			force=True,
			handlers=[
				logging.StreamHandler(sys.stderr),
				logging.FileHandler(log_path),
			])

	if DIST_DATE is not None:
		logging.info('DIST: %s', DIST_DATE)

	config = Configuration.createOrLoad(args.config, volatile=True)

	predictorClass = PREDICTOR_FACTORY.get(args.mode)
	if predictorClass is None:
		logging.error(f'Mode "{args.mode}" is not supported.')
		return -1

	predictor = predictorClass(config, device=args.device, inspect=args.inspect)

	if args.port:
		app = ZeroServer(predictor)
		app.bind(args.port)
	else:
		# the initialized signal
		print('', end='\r', flush = True)

		while session(predictor):
			pass


if __name__ == '__main__':
	main()
