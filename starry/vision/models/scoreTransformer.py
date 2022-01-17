
import torch
import torch.nn as nn

from ...modules.positionEncoder import SinusoidEncoderXYY
from ...transformer.layers import EncoderLayer
from ...transformer.models import get_pad_mask
from ..data.scoreFault import SEMANTIC_MAX, STAFF_MAX



class EncoderLayerStack (nn.Module):
	def __init__ (self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])


	def forward (self, x, mask):	# (n, seq, d_word)
		enc_output = x
		for enc_layer in self.layer_stack:
			enc_output, _ = enc_layer(enc_output, slf_attn_mask=mask)

		return enc_output


class Encoder (nn.Module):
	def __init__ (self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, angle_cycle=1000,
			dropout=0.1, scale_emb=False, n_semantic=SEMANTIC_MAX, n_staff=STAFF_MAX):
		super().__init__()

		self.semantic_emb = nn.Embedding(n_semantic, d_word_vec, padding_idx=0)
		self.staff_emb = nn.Embedding(n_staff, d_word_vec, padding_idx=0)

		self.position_encoder = SinusoidEncoderXYY(angle_cycle=angle_cycle, d_hid=d_word_vec)

		self.dropout = nn.Dropout(p=dropout)

		self.stack = EncoderLayerStack(n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.scale_emb = scale_emb
		self.d_model = d_model


	# semantic:		(n, seq)
	# staff:		(n, seq)
	# x:			(n, seq)
	# y1:			(n, seq)
	# y2:			(n, seq)
	# confidence:	(n, seq)
	# mask:			(n, seq, seq)
	def forward (self, semantic, staff, x, y1, y2, confidence, mask):	# (n, seq, d_word), (n, seq, d_word)
		emb = self.semantic_emb(semantic) + self.staff_emb(staff)
		pos = self.position_encoder(x, y1, y2)
		enc_output = emb + pos
		enc_output[:, :, -1] = confidence

		if self.scale_emb:
			enc_output *= self.d_model ** 0.5

		enc_output = self.dropout(enc_output)
		enc_output = self.layer_norm(enc_output)

		enc_output = self.stack(enc_output, mask)

		return enc_output


class ScoreTransformer (nn.Module):
	def __init__ (self, out_channels, d_model=512, d_inner=2048,
			angle_cycle=1000, n_layers=6,
			n_head=8, d_k=64, d_v=64, dropout=0.1, scale_emb=False,
			n_semantic=SEMANTIC_MAX, n_staff=STAFF_MAX):
		super().__init__()

		self.d_model = d_model
		d_word_vec = d_model

		self.encoder = Encoder(d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner,
			angle_cycle=angle_cycle, dropout=dropout, scale_emb=scale_emb, n_semantic=n_semantic, n_staff=n_staff)

		self.output = nn.Linear(d_model, out_channels)


	def forward (self, semantic, staff, x, y1, y2, confidence):
		mask = get_pad_mask(semantic, 0)
		code = self.encoder(semantic, staff, x, y1, y2, confidence, mask)	# (n, seq, d_model)

		return self.output(code)	# (n, seq, out_channels)


class ScoreSemanticValue (ScoreTransformer):
	def __init__(self, **kwargs):
		super().__init__(out_channels = 1, **kwargs)


class ScoreSemanticValueLoss (nn.Module):
	def __init__ (self, **kw_args):
		super().__init__()

		self.deducer = ScoreSemanticValue(**kw_args)

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)


	def forward (self, batch):
		pred = self.deducer(batch['semantic'], batch['staff'], batch['x'], batch['y1'], batch['y2'], batch['confidence'])
		pred = torch.sigmoid(pred[:, :, 0])
		truth = batch['value']

		loss = nn.functional.binary_cross_entropy(pred, truth, weight=batch['mask'])

		fake = torch.logical_and(batch['mask'], torch.logical_xor(batch['value'] > 0, batch['confidence'] >= 1))
		rectified = torch.logical_and(fake, torch.logical_xor(batch['value'] > 0, pred < 0.5))

		real = torch.logical_and(batch['mask'], torch.logical_xor(batch['value'] > 0, batch['confidence'] < 1))
		degenerated = torch.logical_and(real, torch.logical_xor(batch['value'] > 0, pred >= 0.5))

		accurate = torch.logical_and(batch['mask'], torch.logical_xor(batch['value'] > 0, pred < 0.5))

		metric = {
			'bce': loss.item(),
			'total': batch['mask'].sum().item(),
			'accurate': accurate.sum().item(),
			'fake': fake.sum().item(),
			'real': real.sum().item(),
			'rectified': rectified.sum().item(),
			'degenerated': degenerated.sum().item(),
		}

		return loss, metric


	def stat (self, metrics, n_batch):
		total = metrics.get('total')
		accurate = metrics.get('accurate')
		fake = metrics.get('accurate')
		real = metrics.get('real')
		rectified = metrics.get('rectified')
		degenerated = metrics.get('degenerated')

		return {
			'bce': metrics['bce'] / max(n_batch, 1),
			'accuracy': accurate / max(total, 1),
			'base_acc': real / max(total, 1),
			'rectification': rectified / max(fake, 1),
			'degeneration': degenerated / max(real, 1),
		}
