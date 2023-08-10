
# map int to prime factorization classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import primesieve
import numpy as np



def factorify (n):
	n = n // 1
	sqrt_n = np.sqrt(n) // 1
	factors = []

	residue = n
	while residue > 1:
		for p in primesieve.primes(sqrt_n):
			if residue % p == 0:
				factors.append(p)
				residue //= p
				break

	return factors


def facprod (factors):
	pfactors = []
	p = 1
	for f in factors:
		pfactors.append(p)
		p *= f

	return pfactors


class Int2PClass (nn.Module):
	def __init__ (self, base):
		super().__init__()

		factors = factorify(base)
		self.factors = factors[::-1]
		self.pfactors = facprod(self.factors)


	def forward (self, x):
		digits = [x.floor_divide(p).remainder(f) for f, p in zip(self.factors, self.pfactors)]
		one_hot = [F.one_hot(d, num_classes=f)[..., 1:] for d, f in zip(digits, self.factors)]

		return torch.cat(one_hot, dim=-1)


class PClass2Int (nn.Module):
	def __init__ (self, base):
		super().__init__()

		factors = factorify(base)
		self.factors = factors[::-1]
		pfactors = facprod(self.factors)

		self.register_buffer('pfactors', torch.tensor(pfactors, dtype=torch.long), persistent=False)

		pos = 0
		self.digit_pos = []
		for f in self.factors:
			self.digit_pos.append(pos)
			pos += f - 1


	def forward (self, logits):
		digits = [F.pad(logits[..., pos:pos + f - 1], (1, 0), value=0).argmax(dim=-1) for f, pos in zip(self.factors, self.digit_pos)]
		digits = torch.stack(digits, dim=-1)

		return torch.inner(digits, self.pfactors)
