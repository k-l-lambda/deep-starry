
import sys
import time
import io
import argparse
import logging
import json

from starry.utils.config import Configuration
import starry.topology as topology



logging.basicConfig(stream=sys.stderr, level=logging.INFO)


PREDICTOR_FACTORY = {
	'topology': topology.predictor.Predictor,
}


def parseInputLine (line):
	if not ':' in line:
		buffer = base64.b64decode(line)
		return {'buffer': io.BytesIO(buffer)}

	colon = line.index(':')
	protocol = line[:colon]
	body = line[colon + 1:]

	if protocol == 'base64':
		buffer = base64.b64decode(body)
		return {'buffer': io.BytesIO(buffer)}
	elif protocol == 'json':
		return json.loads(body)
	else:
		raise ValueError(f'unexpected input protocol: {protocol}')


def outputData (data):
	logging.info('output json: %d', len(data))
	print(data + '\n')


def session (predictor):
	logging.info('Waiting for request...')

	args = []
	kwarg = {}

	while True:
		data = sys.stdin.readline()
		#print('read:', len(data), data[0], flush = True)
		logging.info('data read: %d', len(data))
		if not data or len(data) <= 1:
			break

		data = parseInputLine(data)
		if data.get('buffer'):
			args.append(data['buffer'])
		else:
			kwarg = {**kwarg, **data}

	logging.info('Predicting...')
	t0 = time.time()
	result = predictor.predict(*args, **kwarg)
	t1 = time.time()

	outputData(json.dumps(result))

	print('', flush = True)
	logging.info(f'Prediction result sent. ({t1 - t0:.3f}s)')

	return True


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-m', '--mode', type=str, default='topology', help='predictor mode')

	args = parser.parse_args()

	config = Configuration.create(args.config) if args.config.endswith('.yaml') else Configuration(args.config)

	predictorClass = PREDICTOR_FACTORY.get(args.mode)
	if predictorClass is None:
		logging.error(f'Mode "{args.mode}" is not supported.')
		return -1

	predictor = predictorClass(config)

	while session(predictor):
		pass


if __name__ == '__main__':
	main()
