
import torch
import torch.nn as nn



class LoraInjectedLinear(nn.Module):
	def __init__ (self, in_features, out_features, bias=False, r=4, dropout_p=0.1, alpha=1.0):
		super().__init__()

		assert r <= min(in_features, out_features), f'LoRA rank {r} must be less or equal than {min(in_features, out_features)}'

		self.r = r
		self.linear = nn.Linear(in_features, out_features, bias)
		self.lora_down = nn.Linear(in_features, r, bias=False)
		self.dropout = nn.Dropout(dropout_p)
		self.lora_up = nn.Linear(r, out_features, bias=False)
		self.alpha = alpha
		self.selector = nn.Identity()


	def initialize (self):
		nn.init.normal_(self.lora_down.weight, std=1 / self.r)
		nn.init.zeros_(self.lora_up.weight)


	def forward (self, input):
		return self.linear(input) + self.dropout(self.lora_up(self.selector(self.lora_down(input)))) * self.alpha


	def realize_as_lora (self):
		return self.lora_up.weight.data * self.alpha, self.lora_down.weight.data


	def set_selector_from_diag (self, diag: torch.Tensor):
		# diag is a 1D tensor of size (r,)
		assert diag.shape == (self.r,)
		self.selector = nn.Linear(self.r, self.r, bias=False)
		self.selector.weight.data = torch.diag(diag, device=self.lora_up.weight.device, dtype=self.lora_up.weight.dtype)
