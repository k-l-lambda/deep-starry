
import struct



'''
	PARAFF file specification

	header block
		0-6 bytes		PARAFF
		6-8 bytes		\0\0
		8-12 bytes		vocab start position
		12-16 bytes		sentences start position
		16-20 bytes		sentence number
		20-24 bytes		sentence align size

	vocab block

	sentence block
'''


class ParaffFile:
	def __init__ (self, file):
		file.seek(8)
		vocab_pos, data_pos, self.n_sentence, self.sentence_align_size = struct.unpack('iiii', file.read(16))

		file.seek(vocab_pos)
		self.tokens = file.read(data_pos - vocab_pos).decode().split(',')

		data = file.read(self.n_sentence * self.sentence_align_size)
		self.sentences = [list(data[i * self.sentence_align_size:(i + 1) * self.sentence_align_size]) for i in range(self.n_sentence)]
