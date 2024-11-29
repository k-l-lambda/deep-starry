
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...transformer.models import PositionalEncoding, get_pad_mask, get_subsequent_mask
from ..midiseq import T2I, N_VOCAB
from .seqShareVAE import SeqShareDecoderWithPosition as SeqShareSDecoder
from .modules import AttentionStack



# `Share` means encoder & decoder share the common backbone
# SE = Supervised Encoder


PAD = T2I['PAD']
MSUM = T2I['MSUM']
EOS = T2I['EOS']


class SeqShareSEncoder (nn.Module):
	def __init__(self, d_model, word_emb, latent_prj, position_enc, dropout, layer_norm, attention, pad_id=PAD, finale_id=EOS):
		super().__init__()

		self.d_model = d_model

		self.word_emb = word_emb
		self.latent_prj = latent_prj
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
		latent = self.latent_prj(finale)

		return latent


class SeqShareSE (nn.Module):
	def __init__ (self, n_vocab=N_VOCAB, d_latent=256,
		n_layers=6, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64,
		dropout=0.1, mask_dropout=0.2, n_seq_max=512, **_):
		super().__init__()

		self.d_model = d_model
		self.dropout = dropout
		self.mask_dropout = mask_dropout

		self.word_emb = nn.Embedding(n_vocab, d_model, padding_idx=PAD)
		self.word_prj = nn.Linear(d_model, n_vocab, bias=False)
		self.word_prj.weight = self.word_emb.weight

		self.latent_prj = nn.Linear(d_model, d_latent, bias=False)
		self.latent_emb = nn.Linear(d_latent, d_model)

		self.position_enc = PositionalEncoding(d_model, n_position=n_seq_max)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)


	def getEncoder (self) -> SeqShareSEncoder:
		return SeqShareSEncoder(d_model=self.d_model, word_emb=self.word_emb,
			latent_prj=self.latent_prj, position_enc=self.position_enc, layer_norm=self.layer_norm,
			dropout=self.dropout, attention=self.attention, pad_id=PAD, finale_id=EOS)


	def getDecoder (self) -> SeqShareSDecoder:
		return SeqShareSDecoder(d_model=self.d_model, word_emb=self.word_emb, word_prj=self.word_prj, latent_emb=self.latent_emb,
			layer_norm=self.layer_norm, dropout=self.dropout, mask_dropout=self.mask_dropout,
			attention=self.attention, pad_id=PAD)


class SeqShareSELoss (nn.Module):
	def __init__ (self, n_layers, decode_weight=1., **kw_args):
		super().__init__()

		self.n_layers = n_layers
		self.decode_weight = decode_weight
		self.summary_id = MSUM

		self.deducer = SeqShareSE(n_layers=n_layers, **kw_args)

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
		x = batch['seq']
		decoding_mask = batch['decoding_mask']
		encoding_mask = torch.roll(batch['decoding_mask'], shifts=1)
		summary = batch['summary']

		# prepend summary_id on head
		head = torch.tensor([[self.summary_id]], device=x.device).repeat(x.shape[0], 1)
		x1 = torch.cat([head, x], dim=1)

		head_true = torch.ones_like(head)
		decoding_mask1 = torch.cat([head_true, decoding_mask], dim=1)

		z = self.encoder(x, mask=encoding_mask)

		pos = torch.arange(x1.shape[1], device=x1.device).float()[None]
		pred = self.decoder(x1, pos, summary, mask=decoding_mask1)

		pred_flat = pred[:, 1:][decoding_mask]
		target_flat = x[encoding_mask]

		cos_angle = (z * summary).sum(dim=-1) / (z.norm(dim=-1) * summary.norm(dim=-1))
		encode_loss = -((cos_angle + 1.) / 2.).log().mean()

		decode_loss = F.cross_entropy(pred_flat, target_flat)

		loss = encode_loss + decode_loss * self.decode_weight

		pred_ids = torch.argmax(pred_flat, dim=-1)
		acc = (pred_ids == target_flat).float().mean()
		angle = cos_angle.acos().mean()

		metric = {
			'cos_angle': cos_angle.mean().item(),
			'angle': angle.item(),
			'encode_loss': encode_loss.item(),
			'decode_loss': decode_loss.item(),
			'acc': acc.item(),
		}

		if not self.training:
			pred2 = self.decoder(x1, pos, z, mask=decoding_mask1)
			pred2_flat = pred2[:, 1:][decoding_mask]
			pred2_ids = torch.argmax(pred2_flat, dim=-1)
			acc2 = (pred2_ids == target_flat).float().mean()

			metric['acc2'] = acc2.item()

		return loss, metric
