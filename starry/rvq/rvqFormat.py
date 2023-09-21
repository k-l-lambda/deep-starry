
# Residual Vector Quantizer (for audio) file format

import struct



'''
	RVQ file specification

	header block
		0-3 bytes		RVQ
		3-8 bytes		\0
		8-12 bytes		sample rate
		12th byte		number of codebook
		13-24 bytes		reserved

	body block
'''


class RVQFile:
	def __init__ (self, sampling_rate, n_codebook, buffer):
		self.sampling_rate = sampling_rate
		self.n_codebook = n_codebook
		self.buffer = buffer


	@staticmethod
	def fromFile (file):
		file.seek(8)
		sampling_rate = struct.unpack('I', file.read(4))[0]

		file.seek(12)
		n_codebook = struct.unpack('B', file.read(1))[0]

		file.seek(24)
		buffer = file.read()

		return RVQFile(sampling_rate, n_codebook, buffer)


	@staticmethod
	def fromAudio (audio, sampling_rate):
		from .encodec import encode

		buf, n_codebook = encode(audio, sr=sampling_rate)

		return RVQFile(sampling_rate, n_codebook, buf)


	def save (self, file):
		file.write(b'RVQ')

		file.seek(8)
		file.write(struct.pack('I', self.sampling_rate))

		file.seek(12)
		file.write(struct.pack('B', self.n_codebook))

		file.seek(24)
		file.write(self.buffer)


	def decodeToAudio (self):
		from .encodec import decode

		return decode(self.buffer, self.n_codebook)
