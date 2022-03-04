
class WeightedValue:
	@staticmethod
	def from_value (value, weight=1):
		return WeightedValue(value * weight, weight)


	def __init__ (self, sum=None, weight=1):
		self.sum = sum
		self.weight = weight


	def __add__ (self, other: 'WeightedValue') -> 'WeightedValue':
		if other.weight != 0:
			if self.weight == 0:
				return other

			return WeightedValue(self.sum + other.sum, self.weight + other.weight)

		return self


	def __str__(self) -> str:
		return f'{self.value} (*{self.weight})'


	@property
	def value (self):
		return self.sum / self.weight
