
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
#import logging

from ...transformer.models import get_subsequent_mask, get_pad_mask, Decoder
from ..graphSemantics import SEMANTIC_MAX, STAFF_MAX, TG_EOS, TG_PAD
from ..vocab import ID_PAD, ID_VB, ID_EOM
from .modules import AttentionStack, TimewiseGraphEncoder, DecoderWithPosition
from .seqShareVAE import SeqShareVAE



# GraphParaffEncoder -------------------------------------------------------------------------------------------------------
class GraphParaffEncoder (nn.Module):
	def __init__(self, d_model=256, n_semantic=SEMANTIC_MAX, n_staff=STAFF_MAX, d_position=128, angle_cycle=1000,
		d_inner=1024, n_layers=6, n_head=8, d_k=32, d_v=32, dropout=0.1, unidirectional=True):
		super().__init__()

		self.d_model = d_model
		self.unidirectional = unidirectional

		self.cat = TimewiseGraphEncoder(n_semantic=n_semantic, n_staff=n_staff, d_hid=d_position, angle_cycle=angle_cycle)
		self.embedding = nn.Linear(self.cat.output_dim, d_model, bias=False)
		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)


	def forward (self, ids, staff, confidence, x, y, sy1, sy2, mask):	# -> (n, n_seq, d_model)
		trg_mask = mask.unsqueeze(-2)
		if self.unidirectional:
			trg_mask = trg_mask & get_subsequent_mask(ids)

		h = self.cat(ids, staff, confidence, x, y, sy1, sy2)
		h = self.embedding(h)
		h = self.attention(h, mask=trg_mask)

		return h


class GraphParaffEncoderTail (nn.Module):
	def __init__(self, word_decoder_config, word_decoder_pretrain, **kw_args):
		super().__init__()

		self.encoder = GraphParaffEncoder(**kw_args)

		self.id_pad = TG_PAD
		self.id_eos = TG_EOS


	def load_state_dict (self, state_dict, **_):
		return self.encoder.load_state_dict(state_dict, **_)


	def forward (self, ids, staff, confidence, x, y, sy1, sy2):	# -> (n, d_model)
		mask = ids == self.id_pad
		h = self.encoder(ids, staff, confidence, x, y, sy1, sy2, mask)

		return h[ids == self.id_eos]


class GraphParaffEncoderDecoder (nn.Module):
	def __init__(self, d_model=256, word_decoder_config=None, **_):
		super().__init__()

		vae = SeqShareVAE(d_latent=d_model, **word_decoder_config)
		self.word_decoder = vae.getDecoderWithPos()


	def load_state_dict (self, state_dict, strict=True):
		return self.word_decoder.load_state_dict(state_dict['word_decoder'], strict)


	def forward (self, input_ids, position, latent):
		return self.word_decoder(input_ids, position.float(), latent)


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
		tg_mask = tg_id != TG_PAD
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


# GraphParaffSummaryEncoder -------------------------------------------------------------------------------------------------------
class GraphParaffSummaryEncoder (nn.Module):
	def __init__(self, d_model=256, n_semantic=SEMANTIC_MAX, n_staff=STAFF_MAX, d_position=128, angle_cycle=1000,
		d_inner=1024, n_layers=6, n_head=8, d_k=32, d_v=32, dropout=0.1):
		super().__init__()

		self.d_model = d_model

		self.cat = TimewiseGraphEncoder(n_semantic=n_semantic, n_staff=n_staff, d_hid=d_position, angle_cycle=angle_cycle)
		self.embedding = nn.Linear(self.cat.output_dim, d_model, bias=False)
		self.attention = AttentionStack(d_model=d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)
		self.summary_emb = nn.Linear(d_model, d_model, bias=False)


	# prev_summary: (n, d_model)
	# ids, staff, confidence, x, y, sy1, sy2, mask: (n, n_seq, d_model)
	def forward (self, prev_summary, ids, staff, confidence, x, y, sy1, sy2, mask):	# -> (n, n_seq, d_model)
		trg_mask = mask.unsqueeze(-2) & get_subsequent_mask(ids)

		prev_emb = self.summary_emb(prev_summary).unsqueeze(-2)

		h = self.cat(ids, staff, confidence, x, y, sy1, sy2)
		h = self.embedding(h)
		h += prev_emb
		h = self.attention(h, mask=trg_mask)

		return h


class GraphParaffSummaryEncoderLoss (nn.Module):
	def __init__(self, d_model=256, word_decoder_config=None, word_decoder_pretrain=None, **kw_args):
		super().__init__()

		assert d_model == word_decoder_config['d_model'], 'd_model mismatch: %d, %d' % (d_model, word_decoder_config['d_model'])

		self.deducer = GraphParaffSummaryEncoder(d_model=d_model, **kw_args)

		self.mse = nn.MSELoss()

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

			self.word_decoder = vae.getDecoderWithPos()


	def forward (self, batch):
		body_mask = batch['body_mask']
		word_mask = body_mask | (batch['input_ids'] == self.summary_id)
		target = batch['output_ids'].long()
		target_body = target[body_mask]

		ph_summary, ph_next_mask = batch['ph_summary'], batch['ph_next_mask']
		current_summary = ph_summary[ph_next_mask]

		ph_prev_mask = F.pad(ph_next_mask.unsqueeze(0), (0, 1), 'circular').squeeze(0)[..., 1:]
		ph_summary[..., :-1, :][ph_prev_mask[..., :-1]] = 0 # zero out prev summary at head (put to tail by circular padding)
		prev_summary = ph_summary[ph_prev_mask]

		tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2 = batch['tg_id'], batch['tg_staff'], batch['tg_confidence'], batch['tg_x'], batch['tg_y'], batch['tg_sy1'], batch['tg_sy2']
		tg_mask = tg_id != TG_PAD
		tg_eos = tg_id == TG_EOS

		graph_code = self.deducer(prev_summary, tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2, mask=tg_mask)
		latent = graph_code[tg_eos]

		pred = self.word_decoder(batch['input_ids'], batch['position'].float(), latent, mask=word_mask)
		pred_body = pred[body_mask]

		#loss = F.cross_entropy(pred_body, target_body)
		loss = self.mse(latent, current_summary)

		pred_ids = pred_body.argmax(dim=-1)
		acc = (pred_ids == target_body).float().mean()

		metric = {
			'acc': acc.item(),
		}

		if not self.training:
			zero_latent = torch.zeros_like(latent)

			pred_zl = self.word_decoder(batch['input_ids'], batch['position'].float(), zero_latent)

			error = 1 - acc
			error_zero_latent = 1 - (pred_zl[body_mask].argmax(dim=-1) == target_body).float().mean()

			metric['error'] = error.item()
			metric['error_zero_latent'] = error_zero_latent.item()

		return loss, metric


# GraphParaffTranslator ----------------------------------------------------------------------------------------------------
class GraphParaffTranslator (nn.Module):
	def __init__(self, d_model, encoder_config, decoder_config, with_pos=False, **_):
		super().__init__()

		self.with_pos = with_pos

		self.encoder = GraphParaffEncoder(d_model=d_model, **encoder_config)

		decoder_class = DecoderWithPosition if with_pos else Decoder
		self.decoder = decoder_class(d_model=d_model, d_word_vec=d_model, pad_idx=ID_PAD, **decoder_config)

		self.word_prj = nn.Linear(d_model, decoder_config['n_trg_vocab'], bias=False)
		self.word_prj.weight = self.decoder.trg_word_emb.weight

		self.TG_PAD = TG_PAD
		self.ID_PAD = ID_PAD


	def forward (self, ids, staff, confidence, x, y, sy1, sy2, premier, position: Optional[torch.Tensor]=None):	# -> (n, n_seq, n_vocab)
		source_mask = ids != self.TG_PAD
		target_mask = get_pad_mask(premier, self.ID_PAD) & get_subsequent_mask(premier)

		graph_code = self.encoder(ids, staff, confidence, x, y, sy1, sy2, source_mask)
		if self.with_pos:
			decoder_out = self.decoder(premier.long(), position, target_mask, graph_code, source_mask.unsqueeze(-2))
		else:
			decoder_out, = self.decoder(premier.long(), target_mask, graph_code, source_mask.unsqueeze(-2))
		result = self.word_prj(decoder_out)

		return result


class GraphParaffTranslatorOnnx (GraphParaffTranslator):
	pass
	'''def forward(self, ids, staff, confidence, x, y, sy1, sy2, premier, position):
		n_point = (ids != self.TG_PAD).sum().item()
		ids = ids[:, :n_point]
		staff = staff[:, :n_point]
		confidence = confidence[:, :n_point]
		x = x[:, :n_point]
		y = y[:, :n_point]
		sy1 = sy1[:, :n_point]
		sy2 = sy2[:, :n_point]

		return super().forward(ids, staff, confidence, x, y, sy1, sy2, premier, position)'''


class GraphParaffTranslatorLoss (nn.Module):
	def __init__(self, word_weights=None, vocab=[], **kw_args):
		super().__init__()

		self.deducer = GraphParaffTranslator(**kw_args)

		if word_weights is not None:
			n_vocab = kw_args['decoder_config']['n_trg_vocab']
			ww = torch.tensor([word_weights[word] if word in word_weights else 1 for word in vocab[:n_vocab]], dtype=torch.float)
			assert len(ww) == kw_args['decoder_config']['n_trg_vocab'], f'invalid word weights shape {len(ww)} vs decoder_config.n_trg_vocab({n_vocab})'
			self.register_buffer('word_weights', ww, persistent=False)

		for p in self.deducer.encoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['encoder_config']['n_layers'] * 2) ** -0.5)

		for p in self.deducer.decoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['decoder_config']['n_layers'] * 2) ** -0.5)


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())


	def forward (self, batch):
		body_mask = batch['body_mask']
		target = batch['output_ids'].long()
		target_body = target[body_mask]

		input_ids = batch['input_ids']
		tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2 = batch['tg_id'], batch['tg_staff'], batch['tg_confidence'], batch['tg_x'], batch['tg_y'], batch['tg_sy1'], batch['tg_sy2']
		position = batch['position'].float()

		pred = self.deducer(tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2, input_ids, position=position)
		pred_body = pred[body_mask]

		ce_weight = self.word_weights if hasattr(self, 'word_weights') else None
		loss = F.cross_entropy(pred_body, target_body, weight=ce_weight)

		pred_ids = pred_body.argmax(dim=-1)
		acc = (pred_ids == target_body).float().mean()

		metric = {
			'acc': acc.item(),
		}

		if not self.training:
			boundary_mask = (target_body == ID_VB) | (target_body == ID_EOM)
			acc_boundary = (pred_ids[boundary_mask] == target_body[boundary_mask]).float().mean()
			metric['acc_boundary'] = acc_boundary.item()

			zero_tg_id = torch.zeros_like(tg_id)
			np_input_ids = torch.zeros_like(input_ids)
			np_input_ids[body_mask] = input_ids[body_mask]

			pred_zl = self.deducer(zero_tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2, input_ids, position=position)
			pred_np = self.deducer(tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2, np_input_ids, position=position)
			pred_zlnp = self.deducer(zero_tg_id, tg_staff, tg_confidence, tg_x, tg_y, tg_sy1, tg_sy2, np_input_ids, position=position)

			error = 1 - acc
			error_zero_latent = 1 - (pred_zl[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_no_primer = 1 - (pred_np[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_zero_latent_no_primer = 1 - (pred_zlnp[body_mask].argmax(dim=-1) == target_body).float().mean()

			metric['error'] = error.item()
			metric['error_zero_latent'] = error_zero_latent.item()
			metric['error_no_primer'] = error_no_primer.item()
			metric['error_zero_latent_no_primer'] = error_zero_latent_no_primer.item()

		return loss, metric
