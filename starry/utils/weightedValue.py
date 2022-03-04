
class WeightedValue:
	def __init__ (self, value, weight=1):
		self.value = value
		self.weight = weight


	def __add__ (self, other: 'WeightedValue') -> 'WeightedValue':
		if other.weight != 0:
			if self.weight == 0:
				return other

			return WeightedValue(self.value + other.value, self.weight + other.weight)

		return self


	def __str__(self) -> str:
		return f'{self.value} ({self.weight})'


	@property
	def mean (self):
		return self.value / self.weight
