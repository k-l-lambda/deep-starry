import sys
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
		self.socket.send(packb({
			'code': -1,
			'msg': (err_type and (err_type + '\n\n')) + f"{str(err_info[0])} {str(err_info[1])} \n {''.join(traceback.format_tb(err_info[2]))} "
		}, use_bin_type=True))

	def bind(self, port):
		self.socket.bind(f"tcp://*:{port}")

		while True:
			buf = self.socket.recv()
			try:
				msg = unpackb(buf, raw=False, use_list=False)

				method = msg.get('method')
				args = msg.get('args', [])
				kwargs = msg.get('kwargs', {})

				try:
					ret = methodcaller(method, *args, **kwargs)(self.obj)
					ret = [*ret] if isinstance(ret, types.GeneratorType) else ret
					self.res_success(ret)
				except:
					self.res_error('Server handler error:')
			except:
				self.res_error('Parse request params error:')
