
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...transformer.models import get_pad_mask, get_subsequent_mask
from ...modules.positionEncoder import SinusoidEncoder
from ...lora.transformer import LoraEncoderLayer
from .modules import AttentionStack



class SeqDecoderBase (nn.Module):
	def __init__(self, n_vocab, pad_id=0,
		n_layers=6, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64,
		dropout=0.1, angle_cycle=10000 / (2 * np.pi)):
		super().__init__()

		self.d_model = d_model
		self.pad_id = pad_id

		self.word_emb = nn.Embedding(n_vocab, d_model, padding_idx=pad_id)
		self.word_prj = nn.Linear(d_model, n_vocab, bias=False)
		self.word_prj.weight = self.word_emb.weight

		self.position_enc = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_model)

		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)


	def forward (self, seq: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor] =None):
		mask = get_pad_mask(seq, self.pad_id) if mask is None else mask.unsqueeze(-2)
		mask = mask & get_subsequent_mask(seq)

		#summary = self.latent_emb(latent)

		x = seq.long()
		x = self.word_emb(x)
		#x[:, 0] += summary	# add summary embedding on the first element
		x *= self.d_model ** 0.5	# scale embedding

		x += self.dropout(self.position_enc(pos))
		x = self.layer_norm(x)
		x = self.attention(x, mask)

		x = self.word_prj(x)

		return x


class SeqDecoderBaseLoss (nn.Module):
	def __init__ (self, **kw_args):
		super().__init__()

		self.deducer = SeqDecoderBase(**kw_args)

		for p in self.deducer.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['n_layers'] * 2) ** -0.5)


	def forward (self, batch):
		body_mask = batch['body_mask']
		target = batch['output_ids'].long()
		target_body = target[body_mask]

		pred = self.deducer(batch['input_ids'], batch['position'].float())
		pred_body = pred[body_mask]

		loss = F.cross_entropy(pred_body, target_body)

		pred_ids = pred_body.argmax(dim=-1)
		acc = (pred_ids == target_body).float().mean()

		metric = {
			'acc': acc.item(),
		}

		return loss, metric


class SeqDecoderLora (SeqDecoderBase):
	def __init__(self, lora_config, n_lora_layers=None, d_latent=256, d_model=256, **kw_args):
		super().__init__(d_model=d_model, **kw_args)

		n_lora_layers = n_lora_layers or kw_args['n_layers']

		lora_layers = [LoraEncoderLayer(
			r=lora_config['r'],
			alpha=lora_config['alpha'],
			bias=lora_config['bias'],
			d_model=d_model,
			d_inner=kw_args['d_inner'],
			n_head=kw_args['n_head'],
			d_k=kw_args['d_k'],
			d_v=kw_args['d_v'],
			dropout=kw_args['dropout'],
		) for i in range(n_lora_layers)]
		layers = [lora_layers[i] if i < n_lora_layers else layer for i, layer in enumerate(self.attention.layer_stack)]
		self.attention.layer_stack = nn.ModuleList(layers)

		self.latent_emb = nn.Linear(d_latent, d_model)


	def forward (self, seq: torch.Tensor, pos: torch.Tensor, latent: torch.Tensor, mask: Optional[torch.Tensor] =None):
		mask = get_pad_mask(seq, self.pad_id) if mask is None else mask.unsqueeze(-2)
		mask = mask & get_subsequent_mask(seq)

		summary = self.latent_emb(latent)

		x = seq.long()
		x = self.word_emb(x)
		x[:, 0] += summary	# add summary embedding on the first element
		x *= self.d_model ** 0.5	# scale embedding

		x += self.dropout(self.position_enc(pos))
		x = self.layer_norm(x)
		x = self.attention(x, mask)

		x = self.word_prj(x)

		return x


	def initialize (self):
		for layer in self.attention.layer_stack:
			if hasattr(layer, 'initialize'):
				layer.initialize()

		for p in self.latent_emb.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)


	def freezeTrunk (self):
		for p in self.parameters():
			p.requires_grad = False

		for layer in self.attention.layer_stack:
			if hasattr(layer, 'freezeTrunk'):
				layer.freezeTrunk()
