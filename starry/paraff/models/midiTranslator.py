
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
#import logging

from ...transformer.models import get_subsequent_mask, get_pad_mask
from ..vocab import ID_PAD, ID_VB, ID_EOM
from .modules import AttentionStack, DecoderWithPosition, MidiEventEncoder, InteractiveAttentionStack



'''class MEBiEncoder (nn.Module):
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

		return x'''


class MidiParaffTranslator (nn.Module):
	def __init__(self, d_model, encoder_config, decoder_config, n_midi_dec_layer=2, d_midi_dec=1, n_head=8, d_k=32, d_v=32, d_inner=1024, dropout=0.1, with_pos=False, **_):
		super().__init__()

		self.with_pos = with_pos

		#self.encoder = MEBiEncoder(d_model=d_model, **encoder_config)
		self.midi_enc = MidiEventEncoder(d_model, encoder_config['n_type'], encoder_config['n_pitch'], pos_encoder=encoder_config['pos_encoder'], angle_cycle=encoder_config['angle_cycle'])
		self.att_midi_enc = AttentionStack(encoder_config['n_layer'], d_model=d_model, n_head=n_head, d_k=d_k, d_v=d_v, d_inner=d_inner, dropout=dropout)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

		self.paraff_decoder = DecoderWithPosition(d_model=d_model, d_word_vec=d_model, pad_idx=ID_PAD, n_head=n_head, d_k=d_k, d_v=d_v, d_inner=d_inner, dropout=dropout, **decoder_config)

		self.word_prj = nn.Linear(d_model, decoder_config['n_trg_vocab'], bias=False)
		self.word_prj.weight = self.paraff_decoder.trg_word_emb.weight

		self.att_midi_dec = InteractiveAttentionStack(n_midi_dec_layer, d_model, d_inner, n_head, d_k, d_v, dropout)
		self.midi_prj = nn.Linear(d_model, d_midi_dec, bias=False)

		self.ID_PAD = ID_PAD


	def forward (self, t, p, s, time, premier, position, midi_mask=None):	# -> (n, n_seq, n_vocab)
		source_mask = (t != 0).unsqueeze(-2)	# bidirectional mask
		paraff_mask = get_pad_mask(premier, self.ID_PAD)
		target_mask = paraff_mask & get_subsequent_mask(premier)

		midi_emb = self.midi_enc(t, p, s, time)
		midi_emb = self.dropout(midi_emb)
		midi_emb = self.layer_norm(midi_emb)

		x = self.att_midi_enc(midi_emb, source_mask)

		x = self.paraff_decoder(premier.long(), position, target_mask, x, source_mask if midi_mask is None else midi_mask.unsqueeze(-2))
		paraff_out = self.word_prj(x)

		x = self.att_midi_dec(midi_emb, source_mask, x, paraff_mask)
		midi_out = self.midi_prj(x)

		return paraff_out, midi_out


class MidiParaffTranslatorLoss (nn.Module):
	def __init__(self, word_weights=None, midi_weight=1, mask_measure=None, vocab=[], **kw_args):
		super().__init__()

		self.deducer = MidiParaffTranslator(**kw_args)

		if word_weights is not None:
			n_vocab = kw_args['decoder_config']['n_trg_vocab']
			ww = torch.tensor([word_weights[word] if word in word_weights else 1 for word in vocab[:n_vocab]], dtype=torch.float)
			assert len(ww) == kw_args['decoder_config']['n_trg_vocab'], f'invalid word weights shape {len(ww)} vs decoder_config.n_trg_vocab({n_vocab})'
			self.register_buffer('word_weights', ww, persistent=False)

		self.midi_weight = midi_weight
		self.mask_measure = mask_measure

		for p in self.deducer.att_midi_enc.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['encoder_config']['n_layer'] * 2) ** -0.5)

		for p in self.deducer.paraff_decoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['decoder_config']['n_layers'] * 2) ** -0.5)

		for p in self.deducer.att_midi_dec.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(kw_args['n_midi_dec_layer'] * 2) ** -0.5)


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())


	def forward (self, batch):
		body_mask = batch['body_mask']
		target = batch['output_id'].long()
		target_body = target[body_mask]

		input_id = batch['input_id']
		t, p, s, time = batch['type'], batch['pitch'], batch['strength'], batch['time']
		position = batch['position'].float()

		midi_mask = None
		if self.mask_measure is not None:
			midi_mask = batch['measure'] == self.mask_measure

		pred_id, pred_midi = self.deducer(t, p, s, time, input_id, position=position, midi_mask=midi_mask)
		pred_body = pred_id[body_mask]

		midi_body = t != 0
		target_midi = batch['measure'] == 0
		target_midi = target_midi[midi_body]
		pred_midi = pred_midi.squeeze(-1)[midi_body]

		midi_loss = F.binary_cross_entropy_with_logits(pred_midi, target_midi.float())
		midi_error = 1 - ((pred_midi >= 0) == target_midi).float().mean()

		ce_weight = self.word_weights if hasattr(self, 'word_weights') else None
		paraff_loss = F.cross_entropy(pred_body, target_body, weight=ce_weight)

		loss = paraff_loss + self.midi_weight * midi_loss

		pred_ids = pred_body.argmax(dim=-1)
		acc = (pred_ids == target_body).float().mean()

		metric = {
			'acc': acc.item(),
			'paraff_loss': paraff_loss.item(),
			'midi_loss': midi_loss.item(),
			'midi_error': midi_error.item(),
		}

		if not self.training:
			boundary_mask = (target_body == ID_VB) | (target_body == ID_EOM)
			acc_boundary = (pred_ids[boundary_mask] == target_body[boundary_mask]).float().mean()
			metric['acc_boundary'] = acc_boundary.item()

			zero_t = torch.zeros_like(t)
			np_input_id = torch.zeros_like(input_id)
			np_input_id[body_mask] = input_id[body_mask]

			pred_zl, _ = self.deducer(zero_t, p, s, time, input_id, position=position, midi_mask=midi_mask)
			pred_np, _ = self.deducer(t, p, s, time, np_input_id, position=position, midi_mask=midi_mask)
			pred_zlnp, _ = self.deducer(zero_t, p, s, time, np_input_id, position=position, midi_mask=midi_mask)

			error = 1 - acc
			error_zero_latent = 1 - (pred_zl[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_no_primer = 1 - (pred_np[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_zero_latent_no_primer = 1 - (pred_zlnp[body_mask].argmax(dim=-1) == target_body).float().mean()

			metric['error'] = error.item()
			metric['error_zero_latent'] = error_zero_latent.item()
			metric['error_no_primer'] = error_no_primer.item()
			metric['error_zero_latent_no_primer'] = error_zero_latent_no_primer.item()

		return loss, metric
