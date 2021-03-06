import sys
import logging
import zmq
import types
from msgpack import unpackb, packb
from operator import methodcaller
import traceback


class ZeroServer:
	def __init__(self, obj):
		self.obj = obj
		context = zmq.Context()
		self.socket = context.socket(zmq.REP)

	def __del__(self):
		self.socket.close()

	def res_success(self, ret):
		self.socket.send(packb({
			'code': 0,
			'msg': 'success',
			'data': ret
		}, use_bin_type=True))

	def res_error(self, err_type=None):
		err_info = sys.exc_info()
		err_msg = f"{str(err_info[0])} {str(err_info[1])} \n {''.join(traceback.format_tb(err_info[2]))}"
		self.socket.send(packb({
			'code': -1,
			'msg': (err_type and (err_type + '\n\n')) + err_msg
		}, use_bin_type=True))

		logging.error('[ZeroServer.res_error] %s\n%s', err_type, err_msg)

	def bind(self, port):
		address = f'tcp://*:{port}'
		self.socket.bind(address)
		logging.info('ZeroServer is online: %s', address)

		while True:
			logging.info('Waiting for request...')
			buf = self.socket.recv()
			try:
				logging.info('Got request.')
				msg = unpackb(buf, raw=False, use_list=False)

				method = msg.get('method')
				args = msg.get('args', [])
				kwargs = msg.get('kwargs', {})

				try:
					ret = methodcaller(method, *args, **kwargs)(self.obj)
					ret = [*ret] if isinstance(ret, types.GeneratorType) else ret
					self.res_success(ret)
					logging.info('Response sent.')
				except:
					self.res_error('Server handler error:')
			except:
				self.res_error('Parse request params error:')
