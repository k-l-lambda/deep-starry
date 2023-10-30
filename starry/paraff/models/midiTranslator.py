
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
#import logging

from ...transformer.models import get_subsequent_mask, get_pad_mask, Decoder
from ..vocab import ID_PAD, ID_VB, ID_EOM
from .modules import AttentionStack, DecoderWithPosition, MidiEventEncoder



class MEBiEncoder (nn.Module):
	def __init__(self, n_layer, n_type=4, n_pitch=91, pos_encoder='sinusoid', angle_cycle=100e+3, d_model=128, d_inner=512, n_head=8, d_k=16, d_v=16, dropout=0.1) -> None:
		super().__init__()

		self.event_enc = MidiEventEncoder(d_model, n_type, n_pitch, pos_encoder=pos_encoder, angle_cycle=angle_cycle)
		self.attention = AttentionStack(n_layer, d_model=d_model, n_head=n_head, d_k=d_k, d_v=d_v, d_inner=d_inner, dropout=dropout)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


	def forward (self, t, p, s, time, mask):
		x = self.event_enc(t, p, s, time)
		x = self.dropout(x)
		x = self.layer_norm(x)
		x = self.attention(x, mask)

		return x


class MidiParaffTranslator (nn.Module):
	def __init__(self, d_model, encoder_config, decoder_config, with_pos=False, **_):
		super().__init__()

		self.with_pos = with_pos

		self.encoder = MEBiEncoder(d_model=d_model, **encoder_config)

		self.decoder = DecoderWithPosition(d_model=d_model, d_word_vec=d_model, pad_idx=ID_PAD, **decoder_config)

		self.word_prj = nn.Linear(d_model, decoder_config['n_trg_vocab'], bias=False)
		self.word_prj.weight = self.decoder.trg_word_emb.weight

		self.ID_PAD = ID_PAD


	def forward (self, t, p, s, time, premier, position):	# -> (n, n_seq, n_vocab)
		# bidirectional mask
		source_mask = (t != 0).unsqueeze(-2)
		target_mask = get_pad_mask(premier, self.ID_PAD) & get_subsequent_mask(premier)

		# TODO: output midi event mask

		midi_emb = self.encoder(t, p, s, time, source_mask)
		decoder_out = self.decoder(premier.long(), position, target_mask, midi_emb, source_mask)
		result = self.word_prj(decoder_out)

		return result


class MidiParaffTranslatorLoss (nn.Module):
	def __init__(self, word_weights=None, vocab=[], **kw_args):
		super().__init__()

		self.deducer = MidiParaffTranslator(**kw_args)

		if word_weights is not None:
			n_vocab = kw_args['decoder_config']['n_trg_vocab']
			ww = torch.tensor([word_weights[word] if word in word_weights else 1 for word in vocab[:n_vocab]], dtype=torch.float)
			assert len(ww) == kw_args['decoder_config']['n_trg_vocab'], f'invalid word weights shape {len(ww)} vs decoder_config.n_trg_vocab({n_vocab})'
			self.register_buffer('word_weights', ww, persistent=False)

		for p in self.deducer.encoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['encoder_config']['n_layer'] * 2) ** -0.5)

		for p in self.deducer.decoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['decoder_config']['n_layers'] * 2) ** -0.5)


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())


	def forward (self, batch):
		body_mask = batch['body_mask']
		target = batch['output_id'].long()
		target_body = target[body_mask]

		input_id = batch['input_id']
		t, p, s, time = batch['type'], batch['pitch'], batch['strength'], batch['time']
		position = batch['position'].float()

		pred = self.deducer(t, p, s, time, input_id, position=position)
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

			zero_t = torch.zeros_like(t)
			np_input_id = torch.zeros_like(input_id)
			np_input_id[body_mask] = input_id[body_mask]

			pred_zl = self.deducer(zero_t, p, s, time, input_id, position=position)
			pred_np = self.deducer(t, p, s, time, np_input_id, position=position)
			pred_zlnp = self.deducer(zero_t, p, s, time, np_input_id, position=position)

			error = 1 - acc
			error_zero_latent = 1 - (pred_zl[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_no_primer = 1 - (pred_np[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_zero_latent_no_primer = 1 - (pred_zlnp[body_mask].argmax(dim=-1) == target_body).float().mean()

			metric['error'] = error.item()
			metric['error_zero_latent'] = error_zero_latent.item()
			metric['error_no_primer'] = error_no_primer.item()
			metric['error_zero_latent_no_primer'] = error_zero_latent_no_primer.item()

		return loss, metric
