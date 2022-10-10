
import torch
import torch.nn as nn

#from ...conformer.modules import Linear
from ...modules.positionEncoder import SinusoidEncoder
from ...conformer.encoder import ConformerBlock



class Downsample1D (nn.Module):
	def __init__ (self, in_channels, out_channels):
		super().__init__()

		self.sequential = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=2),
			nn.ReLU(),
		)


	def forward (self, inputs):
		return self.sequential(inputs)


class Upsample1D (nn.Module):
	def __init__ (self, in_channels, out_channels, short_channels):
		super().__init__()

		self.up = nn.Sequential(
			nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
		)
		self.conv = nn.Sequential(
			nn.ConvTranspose1d(in_channels + short_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
		)


	def forward (self, x, xi):
		x = self.up(x)
		x = torch.cat([x, xi], dim=1)

		x = self.conv(x)

		return x


class ConformerEncoderU (nn.Module):
	def __init__ (
		self,
		input_dim=80,
		encoder_dim=128,
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
		angle_cycle=1e+5,
	):
		super().__init__()

		inner_dim = encoder_dim << num_down_layers
		down_channels = [encoder_dim << i for i in range(num_down_layers + 1)]
		#print('down_channels:', down_channels)

		self.downs = nn.ModuleList(modules=[
			Downsample1D(
				in_channels=input_dim if l == 0 else down_channels[l],
				out_channels=down_channels[l + 1],
			) for l in range(num_down_layers)
		])
		self.ups = nn.ModuleList(modules=[
			Upsample1D(
				in_channels=down_channels[-l - 1],
				out_channels=down_channels[-l - 2],
				short_channels=input_dim if l + 1 == num_down_layers else down_channels[-l - 2],
			) for l in range(num_down_layers)
		])

		self.input_projection = nn.Sequential(
			nn.Dropout(p=input_dropout_p),
		)
		self.layers = nn.ModuleList([ConformerBlock(
			encoder_dim=inner_dim,
			num_attention_heads=num_attention_heads,
			feed_forward_expansion_factor=feed_forward_expansion_factor,
			conv_expansion_factor=conv_expansion_factor,
			feed_forward_dropout_p=feed_forward_dropout_p,
			attention_dropout_p=attention_dropout_p,
			conv_dropout_p=conv_dropout_p,
			conv_kernel_size=conv_kernel_size,
			half_step_residual=half_step_residual,
		) for _ in range(num_layers)])

		self.pos_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=inner_dim)


	def count_parameters (self):
		return sum([p.numel() for p in self.parameters()])


	def update_dropout (self, dropout_p):
		for name, child in self.named_children():
			if isinstance(child, nn.Dropout):
				child.p = dropout_p


	def forward (self, inputs):
		x = inputs.permute(0, 2, 1)	# (n, inner, seq)

		xs = []
		for layer in self.downs:
			xs.append(x)
			x = layer(x)

		x = x.permute(0, 2, 1)	# (n, seq, inner)
		x = self.input_projection(x)

		pos = torch.arange(0, x.shape[1], device=inputs.device, dtype=torch.float32).unsqueeze(0)
		x += self.pos_encoder(pos)

		for layer in self.layers:
			x = layer(x)

		#print('x3:', x.shape)

		x = x.permute(0, 2, 1)	# (n, inner, seq)
		xs.reverse()
		for i, layer in enumerate(self.ups):
			xi = xs[i]
			print('x4:', x.shape, xi.shape)
			x = layer(x, xi)

		#print('x5:', x.shape)

		x = x.permute(0, 2, 1)	# (n, seq, inner)

		return x
