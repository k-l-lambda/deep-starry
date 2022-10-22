
import torch.nn as nn



class InvWordEmbed (nn.Module):
	def __init__ (self, hidden_size, vocab_size):
		super().__init__()

		self.fc = nn.Linear(hidden_size, vocab_size)

	def forward (self, x):
		return self.fc(x)
