
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
#import logging

from ...transformer.models import PositionalEncoding, get_pad_mask, get_subsequent_mask
from .sparseAE import AttentionStack



class SeqShareEncoder (nn.Module):
	def __init__(self, d_model, word_emb, latent_prj_mu, latent_prj_var, position_enc, dropout, layer_norm, attention, pad_id, finale_id):
		super().__init__()

		self.d_model = d_model

		self.word_emb = word_emb
		self.latent_prj_mu = latent_prj_mu
		self.latent_prj_var = latent_prj_var
		self.position_enc = position_enc
		self.attention = attention
		self.pad_id = pad_id
		self.finale_id = finale_id

		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = layer_norm


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
		mu = self.latent_prj_mu(finale)
		logvar = self.latent_prj_var(finale)

		return mu, logvar


class SeqShareDecoder (nn.Module):
	def __init__(self, d_model, word_emb, word_prj, latent_emb, position_enc, dropout, layer_norm, mask_dropout, attention, pad_id):
		super().__init__()

		self.d_model = d_model

		self.word_emb = word_emb
		self.word_prj = word_prj
		self.latent_emb = latent_emb
		self.position_enc = position_enc
		self.attention = attention
		self.pad_id = pad_id

		self.dropout = nn.Dropout(p=dropout)
		self.mask_dropout = nn.Dropout(p=mask_dropout)
		self.layer_norm = layer_norm


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


class SeqShareVAE (nn.Module):
	def __init__ (self, n_vocab, d_latent=256, pad_id=0, finale_id=5,
		n_layers=6, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64,
		dropout=0.1, mask_dropout=0.2, n_seq_max=512, **_):
		super().__init__()

		self.d_model = d_model
		self.dropout = dropout
		self.mask_dropout = mask_dropout
		self.pad_id = pad_id
		self.finale_id = finale_id

		self.word_emb = nn.Embedding(n_vocab, d_model, padding_idx=pad_id)
		self.word_prj = nn.Linear(d_model, n_vocab, bias=False)
		self.word_prj.weight = self.word_emb.weight

		self.latent_prj_mu = nn.Linear(d_model, d_latent, bias=False)
		self.latent_prj_var = nn.Linear(d_model, d_latent, bias=False)
		self.latent_emb = nn.Linear(d_latent, d_model)

		self.position_enc = PositionalEncoding(d_model, n_position=n_seq_max)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)


	def getEncoder (self):
		return SeqShareEncoder(d_model=self.d_model, word_emb=self.word_emb,
			latent_prj_mu=self.latent_prj_mu, latent_prj_var=self.latent_prj_var,
			position_enc=self.position_enc, layer_norm=self.layer_norm,
			dropout=self.dropout, attention=self.attention, pad_id=self.pad_id, finale_id=self.finale_id)


	def getDecoder (self):
		return SeqShareDecoder(d_model=self.d_model, word_emb=self.word_emb, word_prj=self.word_prj, latent_emb=self.latent_emb,
			position_enc=self.position_enc, layer_norm=self.layer_norm, dropout=self.dropout, mask_dropout=self.mask_dropout,
			attention=self.attention, pad_id=self.pad_id)


class SeqShareVAEJitEnc (SeqShareVAE):
	def __init__ (self, **kw_args):
		super().__init__(**kw_args)


	# sigma: scalar
	def forward (self, seq: torch.Tensor, sigma: torch.Tensor):
		mask = get_pad_mask(seq, self.pad_id)
		mask = mask & get_subsequent_mask(seq)

		x = seq.long()
		x = self.word_emb(x)
		x *= self.d_model ** 0.5	# scale embedding
		x = self.position_enc(x)
		x = self.layer_norm(x)
		x = self.attention(x, mask)

		finale = x[seq == self.finale_id]	# (n, d_model)
		mu = self.latent_prj_mu(finale)
		logvar = self.latent_prj_var(finale)

		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)

		return mu + eps * std * sigma


class SeqShareVAEJitDec (SeqShareVAE):
	def __init__ (self, **kw_args):
		super().__init__(**kw_args)


	def forward (self, seq: torch.Tensor, latent: torch.Tensor, mask: Optional[torch.Tensor] =None):
		mask = get_pad_mask(seq, self.pad_id) if mask is None else mask.unsqueeze(-2)
		mask = mask & get_subsequent_mask(seq)

		summary = self.latent_emb(latent)

		x = seq.long()
		x = self.word_emb(x)
		x[:, 0] += summary	# add summary embedding on the first element
		x *= self.d_model ** 0.5	# scale embedding

		x = self.position_enc(x)
		x = self.layer_norm(x)
		x = self.attention(x, mask)

		x = self.word_prj(x)

		return x[:, -1]


class SeqShareVAELoss (nn.Module):
	def __init__ (self, n_layers, summary_id, kld_weight=0.001, **kw_args):
		super().__init__()

		self.n_layers = n_layers
		self.summary_id = summary_id
		self.kld_weight = kld_weight

		self.deducer = SeqShareVAE(n_layers=n_layers, **kw_args)

		self.encoder = self.deducer.getEncoder()
		self.decoder = self.deducer.getDecoder()

		for p in self.deducer.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(self.n_layers * 2) ** -0.5)

		self.freeze_target = 1


	def reparameterize (self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)

		return eps * std + mu


	def forward (self, batch):
		x = batch['input_ids']
		mask = batch['body_mask']
		target = batch['output_ids'].long()

		# prepend summary_id on head
		head = torch.tensor([[self.summary_id]], device=x.device).repeat(x.shape[0], 1)
		x = torch.cat([head, x], dim=1)

		head_true = torch.ones_like(head)
		mask1 = torch.cat([head_true, mask], dim=1)

		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)

		pred = self.decoder(x, z, mask=mask1)

		pred_flat = pred[:, 1:][mask]
		target_flat = target[mask]

		recons_loss = F.cross_entropy(pred_flat, target_flat)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

		loss = recons_loss + kld_loss * self.kld_weight

		pred_ids = torch.argmax(pred_flat, dim=-1)
		acc = (pred_ids == target_flat).float().mean()

		metric = {
			'recons_loss': recons_loss.item(),
			'kld_loss': kld_loss.item(),
			'acc': acc.item(),
		}

		return loss, metric


	def inspectRun (self, batch):
		x = batch['input_ids']
		mask = batch['body_mask']
		target = batch['output_ids'].long()

		# prepend summary_id on head
		head = torch.tensor([[self.summary_id]], device=x.device).repeat(x.shape[0], 1)
		x = torch.cat([head, x], dim=1)

		head_true = torch.ones_like(head)
		mask1 = torch.cat([head_true, mask], dim=1)

		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)

		pred = self.decoder(x, z, mask=mask1)

		pred_flat = pred[:, 1:][mask]
		target_flat = target[mask]

		#recons_loss = F.cross_entropy(pred_flat, target_flat)
		#kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

		#loss = recons_loss + kld_loss * self.kld_weight

		pred_ids = torch.argmax(pred_flat, dim=-1)
		acc = (pred_ids == target_flat).float().mean()

		return {
			'pred': pred,
			'pred_flat': pred_flat,
			'target_flat': target[mask],
			'truth': pred_ids == target[mask],
			'acc': acc.item(),
			'mu': mu,
			'logvar': logvar,
			'z': z,
		}
