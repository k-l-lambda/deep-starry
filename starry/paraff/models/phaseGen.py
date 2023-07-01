
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...transformer.models import get_subsequent_mask
from ...modules.positionEncoder import SinusoidEncoder
from .sparseAE import AttentionStack
from .seqShareVAE import SeqShareVAE



class PhaseGen (nn.Module):
	def __init__ (self,
			n_vocab, pad_id=0, d_model=256, d_inner=1024,
			n_layers=6, n_head=8, d_k=32, d_v=32,
			dropout=0.1, angle_cycle=10000):
		super().__init__()

		self.d_model = d_model

		self.embedding = nn.Embedding(n_vocab, d_model, padding_idx=pad_id)
		self.pos_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_model // 2)
		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)


	def forward(self, ids, f_pos, b_pos, summary, mask, next):	# -> (n, d_model * n_next)
		trg_mask = mask.unsqueeze(-2) & get_subsequent_mask(ids)

		ids = ids.long()
		x = self.embedding(ids)

		f_pos_code = self.pos_encoder(f_pos.float())
		b_pos_code = self.pos_encoder(b_pos.float())
		x += torch.cat([f_pos_code, b_pos_code], dim=-1)

		x += summary

		x = self.attention(x, mask=trg_mask)

		return x[next].squeeze(1)


class PhaseGenLoss (nn.Module):
	def __init__ (self,
		d_model=256, word_decoder_config={}, random_base=False, latent_l2_reg=0., **kw_args):
		super().__init__()

		self.summary_id = word_decoder_config['summary_id']
		self.random_base = random_base
		self.latent_l2_reg = latent_l2_reg

		self.deducer = PhaseGen(d_model=d_model, **kw_args)

		self.word_decoder = SeqShareVAE(d_latent=d_model, **word_decoder_config).getDecoder()


	def forward(self, batch):
		body_mask = batch['body_mask']
		target = batch['output_ids'].long()
		target_body = target[body_mask]

		ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask, ph_next_mask = batch['ph_id'], batch['ph_f_num'], batch['ph_b_num'], batch['ph_summary'], batch['ph_body_mask'], batch['ph_next_mask']
		ph_summary[ph_next_mask] = torch.randn_like(ph_summary[ph_next_mask]) if self.random_base else torch.zeros_like(ph_summary[ph_next_mask])

		latent = self.deducer(ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask | ph_next_mask, ph_next_mask)
		latent_delta = latent - ph_summary[ph_next_mask]
		pred = self.word_decoder(batch['input_ids'], latent)
		pred_body = pred[body_mask]

		latent_l2 = latent_delta.square().mean()
		loss = F.cross_entropy(pred_body, target_body) + latent_l2 * self.latent_l2_reg

		pred_ids = pred_body.argmax(dim=-1)
		acc = (pred_ids == target_body).float().mean()

		return loss, {
			'acc': acc.item(),
			'latent_l2': latent_l2,
		}
