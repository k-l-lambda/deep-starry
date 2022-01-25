
#import sys
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
				name = msg.get('name')
				params = msg.get('params')

				try:
					#sys.stdout.write(f'calling "{name}"')
					#sys.stdout.flush()
					ret = methodcaller(name, params)(self.obj)
					#sys.stdout.write(f'ret: "{ret}"')
					#sys.stdout.flush()
					self.socket.send(packb({
						'code': 0,
						'msg': 'success',
						'data': ret
					}, use_bin_type=True))
					#sys.stdout.write('sent.')
					#sys.stdout.flush()
				except Exception as e:
					self.socket.send(packb({
						'code': -1,
						'msg': str(e)
					}, use_bin_type=True))
					#sys.stdout.write('error.1')
					#sys.stdout.flush()
			except Exception as err:
				self.socket.send(packb({
					'code': -1,
					'msg': str(err),
					'data': None
				}, use_bin_type=True))
				#sys.stdout.write('error.2')
				#sys.stdout.flush()
