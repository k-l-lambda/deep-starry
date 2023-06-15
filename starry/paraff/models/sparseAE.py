
# sparse auto encoder

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
#import logging

from ...transformer.models import PositionalEncoding, get_pad_mask, get_subsequent_mask
from ...transformer.layers import EncoderLayer



class AttentionStack (nn.Module):
	def __init__ (self, n_layers=6, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64, dropout=0.1):
		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model=d_model, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)
			for _ in range(n_layers)])


	def forward (self, x, mask):
		for enc_layer in self.layer_stack:
			x, enc_slf_attn = enc_layer(x, slf_attn_mask=mask)

		return x


class SaeEncoder (nn.Module):
	def __init__(self, d_model, word_emb, latent_prj, position_enc, dropout, attention, pad_id, finale_id):
		super().__init__()

		self.d_model = d_model

		self.word_emb = word_emb
		self.latent_prj = latent_prj
		self.position_enc = position_enc
		self.attention = attention
		self.pad_id = pad_id
		self.finale_id = finale_id

		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


	def forward (self, seq: torch.Tensor, mask: Optional[torch.Tensor] =None):
		mask = get_pad_mask(seq, self.pad_id) if mask is None else mask.unsqueeze(-2)
		mask = mask & get_subsequent_mask(seq)

		x = seq.long()
		x = self.word_emb(x)
		x *= self.d_model ** 0.5	# scale embedding
		x = self.dropout(self.position_enc(x))
		x = self.layer_norm(x)
		x = self.attention(x, mask)

		finale = x[seq == self.finale_id]	# (n, d_model)
		latent = self.latent_prj(finale)

		return latent


class SaeDecoder (nn.Module):
	def __init__(self, d_model, word_emb, word_prj, latent_emb, position_enc, dropout, mask_dropout, attention, pad_id):
		super().__init__()

		self.d_model = d_model

		self.word_emb = word_emb
		self.word_prj = word_prj
		self.latent_emb = latent_emb
		self.position_enc = position_enc
		self.attention = attention
		self.pad_id = pad_id
		#self.summary_id = summary_id

		self.dropout = nn.Dropout(p=dropout)
		self.mask_dropout = nn.Dropout(p=mask_dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


	def forward (self, seq: torch.Tensor, latent: torch.Tensor, mask: Optional[torch.Tensor] =None):
		mask = get_pad_mask(seq, self.pad_id) if mask is None else mask.unsqueeze(-2)
		mask[:, 1:] = self.mask_dropout(mask[:, 1:].float()).bool()
		mask = mask & get_subsequent_mask(seq)

		summary = self.latent_emb(latent)

		x = seq.long()
		x = self.word_emb(x)
		x[:, 0] += summary	# add summary embedding on the first element
		x *= self.d_model ** 0.5	# scale embedding

		x = self.dropout(self.position_enc(x))
		x = self.layer_norm(x)
		x = self.attention(x, mask)

		x = self.word_prj(x)

		return x


class SparseAE (nn.Module):
	def __init__ (self, n_vocab, d_latent=0x10000, pad_id=0, summary_id=1, finale_id=5,
		n_layers=6, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64,
		dropout=0.1, mask_dropout=0.2, n_seq_max=512):
		super().__init__()

		self.d_model = d_model
		self.dropout = dropout
		self.mask_dropout = mask_dropout
		self.pad_id = pad_id
		self.summary_id = summary_id
		self.finale_id = finale_id

		self.word_emb = nn.Embedding(n_vocab, d_model, padding_idx=pad_id)
		self.word_prj = nn.Linear(d_model, n_vocab, bias=False)
		self.word_prj.weight = self.word_emb.weight

		self.latent_prj = nn.Linear(d_model, d_latent, bias=False)
		self.latent_emb = nn.Linear(d_latent, d_model)
		self.latent_emb.weight.data = self.latent_prj.weight.data.transpose(0, 1)

		self.position_enc = PositionalEncoding(d_model, n_position=n_seq_max)

		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)

		self.register_buffer('steps', torch.zeros([1], dtype=torch.long, requires_grad=False), persistent=True)


	def getEncoder (self):
		return SaeEncoder(d_model=self.d_model, word_emb=self.word_emb, latent_prj=self.latent_prj, position_enc=self.position_enc,
			dropout=self.dropout, attention=self.attention, pad_id=self.pad_id, finale_id=self.finale_id)


	def getDecoder (self):
		return SaeDecoder(d_model=self.d_model, word_emb=self.word_emb, word_prj=self.word_prj, latent_emb=self.latent_emb,
			position_enc=self.position_enc, dropout=self.dropout, mask_dropout=self.mask_dropout, attention=self.attention, pad_id=self.pad_id)


class SparseAELoss (nn.Module):
	def __init__ (self, n_layers, summary_id, sparse_loss_weight=1, sparse_slope_unit=1e+7, **kw_args):
		super().__init__()

		self.n_layers = n_layers
		self.summary_id = summary_id
		self.sparse_loss_weight = sparse_loss_weight
		self.sparse_slope_unit = sparse_slope_unit

		self.deducer = SparseAE(n_layers=n_layers, summary_id=summary_id, **kw_args)

		self.encoder = self.deducer.getEncoder()
		self.decoder = self.deducer.getDecoder()

		for p in self.deducer.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(self.n_layers * 2) ** -0.5)

		self.freeze_target = 1


	def forward (self, batch):
		x = batch['input_ids']
		mask = batch['body_mask']
		target = batch['output_ids'].long()

		# prepend summary_id on head
		head = torch.tensor([[self.summary_id]], device=x.device).repeat(x.shape[0], 1)
		x = torch.cat([head, x], dim=1)

		head_true = torch.ones_like(head)
		mask1 = torch.cat([head_true, mask], dim=1)

		# update steps
		if self.training:
			self.deducer.steps += 1

		z = self.encoder(x)
		#z = torch.softmax(z * self.deducer.latent_freeze, dim=-1)
		z = torch.softmax(z, dim=-1)
		pred = self.decoder(x, z, mask=mask1)

		pred_flat = pred[:, 1:][mask]
		target_flat = target[mask]

		recons_loss = F.cross_entropy(pred_flat, target_flat)
		sparse_loss = (-z.square().sum(dim=-1).log()).mean()

		sparse_slope = torch.tanh(self.deducer.steps.float() / self.sparse_slope_unit)
		loss = recons_loss + sparse_loss * self.sparse_loss_weight * sparse_slope

		pred_ids = torch.argmax(pred_flat, dim=-1)
		acc = (pred_ids == target_flat).float().mean()

		z_density = 1 - z.max(dim=-1).values.mean()

		metric = {
			'recons_loss': recons_loss.item(),
			'sparse_loss': sparse_loss.item(),
			'acc': acc.item(),
			'z_density': z_density.item(),
			'sparse_slope': sparse_slope.item(),
		}

		if not self.training:
			pred_unconditional = self.decoder(x, torch.zeros_like(z), mask=mask1)
			pred_u_flat = pred_unconditional[:, 1:][mask]
			pred_u_ids = torch.argmax(pred_u_flat, dim=-1)
			acc_uncond = (pred_u_ids == target_flat).float().mean()
			metric['acc_uncond'] = acc_uncond.item()

			zh = z.argmax(dim=-1)
			zh = F.one_hot(zh, z.shape[-1]).float()
			pred_hard = self.decoder(x, zh, mask=mask1)
			pred_hard_flat = pred_hard[:, 1:][mask]
			pred_hard_ids = torch.argmax(pred_hard_flat, dim=-1)
			acc_hard = (pred_hard_ids == target_flat).float().mean()
			metric['acc_hard'] = acc_hard.item()

		return loss, metric
