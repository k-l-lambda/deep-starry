
class InfiniteIterator:
	def __init__(self, it):
		self.it = it

	def __iter__(self):
		while True:
			yield from self.it


class FixedLengthIterator:
	def __init__(self, it, length):
		self.length = length
		self.it = InfiniteIterator(it)

	def __len__(self):
		return self.length

	def __iter__(self):
		for i in range(self.length):
			yield next(self.it)
