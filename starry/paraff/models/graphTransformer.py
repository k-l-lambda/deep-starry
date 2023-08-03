
#from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
#import logging

from ...transformer.models import get_subsequent_mask
from ..data.timewiseGraph import SEMANTIC_MAX, STAFF_MAX, TG_EOS
from .modules import AttentionStack, TimewiseGraphEncoder
from .seqShareVAE import SeqShareVAE



class GraphParaffEncoder (nn.Module):
	def __init__(self, d_model=256, n_semantic=SEMANTIC_MAX, n_staff=STAFF_MAX, d_position=128, angle_cycle=1000,
		d_inner=1024, n_layers=6, n_head=8, d_k=32, d_v=32, dropout=0.1):
		super().__init__()

		self.d_model = d_model

		self.cat = TimewiseGraphEncoder(n_semantic=n_semantic, n_staff=n_staff, d_hid=d_position, angle_cycle=angle_cycle)
		self.embedding = nn.Linear(self.cat.output_dim, d_model, bias=False)
		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)


	def forward (self, ids, staff, confidence, x, y, sy1, y2, mask):	# -> (n, d_model * n_next)
		trg_mask = mask.unsqueeze(-2) & get_subsequent_mask(ids)

		h = self.cat(ids, staff, confidence, x, y, sy1, y2)
		h = self.embedding(h)
		h = self.attention(h, mask=trg_mask)

		return h


class GraphParaffEncoderLoss (nn.Module):
	need_states = True


	def __init__(self, d_model=256, word_decoder_config=None, word_decoder_pretrain=None, **kw_args):
		super().__init__()

		assert d_model == word_decoder_config['d_model'], 'd_model mismatch: %d, %d' % (d_model, word_decoder_config['d_model'])

		self.deducer = GraphParaffEncoder(d_model=d_model, **kw_args)

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

				for param in vae.parameters():
					param.requires_grad = False

				defreeze_layers = word_decoder_pretrain.get('defreeze_layers', -1)
				init_layers = word_decoder_pretrain.get('init_layers', 0)
				if defreeze_layers >= 0:
					for l in range(defreeze_layers):
						for param in vae.attention.layer_stack[-l - 1].parameters():
							param.requires_grad = True
							if l < init_layers and p.dim() > 1:
								nn.init.xavier_uniform_(param, gain=(word_decoder_config['n_layers'] * 2) ** -0.5)

			self.word_decoder = vae.getDecoderWithPos()


	def forward (self, batch):
		body_mask = batch['body_mask']
		word_mask = body_mask | (batch['input_ids'] == self.summary_id)
		target = batch['output_ids'].long()
		target_body = target[body_mask]

		tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2 = batch['tg_id'], batch['tg_staff'], batch['tg_confidence'], batch['tg_x'], batch['tg_y'], batch['tg_sy1'], batch['tg_sy2']
		tg_mask = tg_id != 0
		tg_eos = tg_id == TG_EOS

		graph_code = self.deducer(tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2, mask=tg_mask)
		latent = graph_code[tg_eos]

		pred = self.word_decoder(batch['input_ids'], batch['position'].float(), latent)
		pred_body = pred[body_mask]

		loss = F.cross_entropy(pred_body, target_body)

		pred_ids = pred_body.argmax(dim=-1)
		acc = (pred_ids == target_body).float().mean()

		metric = {
			'acc': acc.item(),
		}

		if not self.training:
			zero_latent = torch.zeros_like(latent)

			pred_zl = self.word_decoder(batch['input_ids'], batch['position'].float(), zero_latent)
			pred_np = self.word_decoder(batch['input_ids'], batch['position'].float(), latent, mask=word_mask)
			pred_zlnp = self.word_decoder(batch['input_ids'], batch['position'].float(), zero_latent, mask=word_mask)

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
