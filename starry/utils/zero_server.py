
import zmq
from msgpack import unpackb, packb
from operator import methodcaller



class ZeroServer:
	def __init__(self, obj):
		self.obj = obj
		context = zmq.Context()
		self.socket = context.socket(zmq.REP)


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
					self.socket.send(packb({
						'code': 0,
						'msg': 'success',
						'data': ret
					}, use_bin_type=True))
				except Exception as e:
					self.socket.send(packb({
						'code': -1,
						'msg': str(e)
					}, use_bin_type=True))
			except Exception as err:
				self.socket.send(packb({
					'code': -1,
					'msg': str(err),
					'data': None
				}, use_bin_type=True))
