
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ...transformer.models import get_subsequent_mask
from ...modules.positionEncoder import SinusoidEncoder
from .sparseAE import AttentionStack
from .seqShareVAE import SeqShareVAE
from .seqDecoder import SeqDecoderLora



class PhaseGen (nn.Module):
	def __init__ (self,
			n_vocab, pad_id=0, d_model=256, d_inner=1024,
			n_layers=6, n_head=8, d_k=32, d_v=32,
			dropout=0.1, angle_cycle=10000, **_):
		super().__init__()

		self.d_model = d_model

		self.embedding = nn.Embedding(n_vocab, d_model, padding_idx=pad_id)
		self.pos_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_model // 2)
		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)


	def forward (self, ids, f_pos, b_pos, summary, mask, next):	# -> (n, d_model * n_next)
		trg_mask = mask.unsqueeze(-2) & get_subsequent_mask(ids)

		ids = ids.long()
		x = self.embedding(ids)

		f_pos_code = self.pos_encoder(f_pos.float())
		b_pos_code = self.pos_encoder(b_pos.float())
		x += torch.cat([f_pos_code, b_pos_code], dim=-1)

		x += summary

		x = self.attention(x, mask=trg_mask)

		return x[next]


class PhaseGenDecoder (nn.Module):
	def __init__ (self, d_model=256, word_decoder_config={}, **_):
		super().__init__()

		vae = SeqShareVAE(d_latent=d_model, **word_decoder_config)
		self.word_decoder = vae.getDecoderWithPos()


	def load_state_dict (self, state_dict, strict=True):
		return self.word_decoder.load_state_dict(state_dict['word_decoder'], strict)


	def forward (self, input_ids, position, latent):
		return self.word_decoder(input_ids, position.float(), latent)


class PhaseGenLoss (nn.Module):
	need_states = True


	def __init__ (self,
		d_model=256, summary_id=1, word_decoder_config=None, word_decoder_pretrain=None,
		lora_decoder_config=None, lora_decoder_pretrain=None,
		random_base=False, latent_l2_reg=0., mask_score_primer=False, **kw_args):
		super().__init__()

		self.summary_id = summary_id
		self.random_base = random_base
		self.latent_l2_reg = latent_l2_reg
		self.mask_score_primer = mask_score_primer

		self.deducer = PhaseGen(d_model=d_model, **kw_args)

		for p in self.deducer.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['n_layers'] * 2) ** -0.5)

		if word_decoder_config is not None:
			if 'summary_id' in word_decoder_config:
				self.summary_id = word_decoder_config['summary_id']

			vae = SeqShareVAE(d_latent=d_model, **word_decoder_config)

			if word_decoder_pretrain is not None:
				checkpoint = torch.load(word_decoder_pretrain['weight'], map_location='cpu')
				vae.load_state_dict(checkpoint['model'], strict=False)

				defreeze_layers = word_decoder_pretrain.get('defreeze_layers', -1)
				init_layers = word_decoder_pretrain.get('init_layers', 0)
				if defreeze_layers >= 0:
					for param in vae.parameters():
						param.requires_grad = False
					for l in range(defreeze_layers):
						for param in vae.attention.layer_stack[-l - 1].parameters():
							param.requires_grad = True
							if l < init_layers and p.dim() > 1:
								nn.init.xavier_uniform_(param, gain=(word_decoder_config['n_layers'] * 2) ** -0.5)
				else:
					freeze_layers = word_decoder_pretrain.get('freeze_layers', 0)
					for l in range(freeze_layers):
						for param in vae.attention.layer_stack[l].parameters():
							param.requires_grad = False
			else:
				for p in vae.parameters():
					if p.dim() > 1:
						nn.init.xavier_uniform_(p, gain=(word_decoder_config['n_layers'] * 2) ** -0.5)

			self.word_decoder = vae.getDecoderWithPos()
		elif lora_decoder_config is not None:
			self.word_decoder = SeqDecoderLora(**lora_decoder_config)

			if lora_decoder_pretrain is not None:
				checkpoint = torch.load(lora_decoder_pretrain['weight'], map_location='cpu')
				self.word_decoder.load_state_dict(checkpoint['model'], strict=False)
				logging.info('lora decoder weight loaded: %s', lora_decoder_pretrain['weight'])

			self.word_decoder.initialize()
			self.word_decoder.freezeTrunk()


	def forward (self, batch):
		body_mask = batch['body_mask']
		body_summary_mask = body_mask | (batch['input_ids'] == self.summary_id)
		target = batch['output_ids'].long()
		target_body = target[body_mask]

		ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask, ph_next_mask = batch['ph_id'], batch['ph_f_num'], batch['ph_b_num'], batch['ph_summary'], batch['ph_body_mask'], batch['ph_next_mask']
		ph_summary[ph_next_mask] = torch.randn_like(ph_summary[ph_next_mask]) if self.random_base else torch.zeros_like(ph_summary[ph_next_mask])

		latent = self.deducer(ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask | ph_next_mask, ph_next_mask)
		latent_delta = latent - ph_summary[ph_next_mask]

		word_mask = body_summary_mask if self.mask_score_primer else None
		pred = self.word_decoder(batch['input_ids'], batch['position'].float(), latent, mask=word_mask)
		pred_body = pred[body_mask]

		latent_l2 = latent_delta.square().mean()
		loss = F.cross_entropy(pred_body, target_body) + latent_l2 * self.latent_l2_reg

		pred_ids = pred_body.argmax(dim=-1)
		acc = (pred_ids == target_body).float().mean()

		metric = {
			'acc': acc.item(),
			'latent_l2': latent_l2.item(),
		}

		if not self.training:
			zero_latent = torch.zeros_like(latent)

			pred_zl = self.word_decoder(batch['input_ids'], batch['position'].float(), zero_latent)
			pred_np = self.word_decoder(batch['input_ids'], batch['position'].float(), latent, mask=body_summary_mask)
			pred_zlnp = self.word_decoder(batch['input_ids'], batch['position'].float(), zero_latent, mask=body_summary_mask)

			error = 1 - acc
			error_zero_latent = 1 - (pred_zl[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_no_primer = 1 - (pred_np[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_zero_latent_no_primer = 1 - (pred_zlnp[body_mask].argmax(dim=-1) == target_body).float().mean()

			metric['error'] = error.item()
			metric['error_zero_latent'] = error_zero_latent.item()
			metric['error_no_primer'] = error_no_primer.item()
			metric['error_zero_latent_no_primer'] = error_zero_latent_no_primer.item()

		return loss, metric


	def state_dict (self, destination=None, prefix='', keep_vars=False):
		return {
			'word_decoder': self.word_decoder.state_dict(),
		}


	def load_state_dict (self, state_dict, **_):
		if state_dict.get('word_decoder') is not None:
			self.word_decoder.load_state_dict(state_dict['word_decoder'])


	def updateStates (self):
		pass
