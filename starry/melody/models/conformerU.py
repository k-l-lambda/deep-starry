
import torch.nn as nn

from ...conformer.modules import Linear
from ...conformer.encoder import ConformerBlock



class Downsample1D (nn.Module):
	def __init__ (self, in_channels, out_channels, n_layer):
		super().__init__()

		self.sequential = nn.Sequential(
			*sum([
				[
					nn.Conv1d(in_channels if l == 0 else out_channels, out_channels, kernel_size=3),
					nn.ReLU(),
				] for l in range(n_layer)
			], [])
		)


	def forward (self, inputs):
		x = inputs.permute(0, 2, 1)
		x = self.sequential(x)
		x = x.permute(0, 2, 1)

		return x


class Upsample1D (nn.Module):
	def __init__ (self, in_channels, out_channels, n_layer):
		super().__init__()

		self.sequential = nn.Sequential(
			*sum([
				[
					nn.ConvTranspose1d(in_channels, out_channels if l == n_layer - 1 else in_channels, kernel_size=3),
					nn.ReLU(),
				] for l in range(n_layer)
			], [])
		)


	def forward (self, inputs):
		x = inputs.permute(0, 2, 1)
		x = self.sequential(x)
		x = x.permute(0, 2, 1)

		return x


class ConformerEncoderU (nn.Module):
	def __init__ (
		self,
		input_dim=80,
		encoder_dim=512,
		num_down_layers=2,
		num_layers=17,
		num_attention_heads=8,
		feed_forward_expansion_factor=4,
		conv_expansion_factor=2,
		input_dropout_p=0.1,
		feed_forward_dropout_p=0.1,
		attention_dropout_p=0.1,
		conv_dropout_p=0.1,
		conv_kernel_size=31,
		half_step_residual=True,
	):
		super().__init__()

		self.conv_down = Downsample1D(in_channels=input_dim, out_channels=encoder_dim, n_layer=num_down_layers)
		self.conv_up = Upsample1D(in_channels=encoder_dim, out_channels=encoder_dim, n_layer=num_down_layers)

		self.input_projection = nn.Sequential(
			nn.Dropout(p=input_dropout_p),
		)
		self.layers = nn.ModuleList([ConformerBlock(
			encoder_dim=encoder_dim,
			num_attention_heads=num_attention_heads,
			feed_forward_expansion_factor=feed_forward_expansion_factor,
			conv_expansion_factor=conv_expansion_factor,
			feed_forward_dropout_p=feed_forward_dropout_p,
			attention_dropout_p=attention_dropout_p,
			conv_dropout_p=conv_dropout_p,
			conv_kernel_size=conv_kernel_size,
			half_step_residual=half_step_residual,
		) for _ in range(num_layers)])


	def count_parameters (self):
		return sum([p.numel() for p in self.parameters()])


	def update_dropout (self, dropout_p):
		for name, child in self.named_children():
			if isinstance(child, nn.Dropout):
				child.p = dropout_p


	def forward (self, inputs):
		outputs = self.conv_down(inputs)
		outputs = self.input_projection(outputs)

		for layer in self.layers:
			outputs = layer(outputs)

		outputs = self.conv_up(outputs)

		return outputs
