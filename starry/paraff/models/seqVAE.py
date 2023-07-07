
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ...transformer.models import Encoder, get_pad_mask, get_subsequent_mask
from ...transformer.layers import EncoderLayer
from ...lora.transformer import LoraEncoderLayer
from .modules import HeadSummaryEncoder



class SeqvaeEncoderMean (nn.Module):
	def __init__ (self,
			n_vocab, n_layers=6, pad_id=0, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64,
			dropout=0.1, n_seq_max=512, scale_emb=True):
		super().__init__()

		self.pad_id = pad_id
		self.d_model = d_model

		self.encoder = Encoder(
			n_src_vocab=n_vocab, n_position=n_seq_max,
			d_word_vec=d_model, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers-1, n_head=n_head, d_k=d_k, d_v=d_v,
			pad_idx=pad_id, dropout=dropout, scale_emb=scale_emb)

		self.out_mu = EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
		self.out_var = EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)


	# seq:	(n, seq)
	# mask:	(n, seq)
	def forward(self, seq, mask: Optional[torch.Tensor] =None):
		if mask is None:
			mask = torch.ones_like(seq)	# double directional mask

		seq = seq.long()
		enc_output, *_ = self.encoder(seq, mask.unsqueeze(-2))

		mu, *_ = self.out_mu(enc_output)		# (n, seq, d_model)
		logvar, *_ = self.out_var(enc_output)

		# reduce along sequence dim
		mask = mask.unsqueeze(-1)
		mask_sum = mask.sum(dim=1)
		mu = mu.masked_fill(mask == 0, 0).sum(dim=1) / mask_sum
		logvar = (logvar.masked_fill(mask == 0, 0).sum(dim=1) / mask_sum).clip(max=80)	# avoid inf after exp

		return mu, logvar


class SeqvaeEncoderFinale (nn.Module):
	def __init__ (self,
			n_vocab, n_layers=6, pad_id=0, finale_id=2, d_model=512, d_emb=512, d_inner=2048, n_head=8, d_k=64, d_v=64,
			dropout=0.1, n_seq_max=512, scale_emb=True):
		super().__init__()

		self.pad_id = pad_id
		self.finale_id = finale_id
		self.d_model = d_model

		self.encoder = Encoder(
			n_src_vocab=n_vocab, n_position=n_seq_max,
			d_word_vec=d_model, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers-1, n_head=n_head, d_k=d_k, d_v=d_v,
			pad_idx=pad_id, dropout=dropout, scale_emb=scale_emb)

		self.out_mu = nn.Linear(d_model, d_emb, bias=False)
		self.out_var = nn.Linear(d_model, d_emb, bias=False)


	# seq:	(n, seq)
	# mask:	(n, seq)
	def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] =None):
		mask = get_pad_mask(seq, self.pad_id) if mask is None else mask.unsqueeze(-2)
		mask = mask & get_subsequent_mask(seq)

		seq = seq.long()
		enc_output, *_ = self.encoder(seq, mask)

		finale = enc_output[seq == self.finale_id]	# (n, d_model)

		mu = self.out_mu(finale)		# (n, d_emb)
		logvar = self.out_var(finale)	# (n, d_emb)

		return mu, logvar


class SeqvaeEncoderJit (nn.Module):
	def __init__(self, n_vocab, encoder_type, d_enc_model, d_model, n_encoder_layer, encoder_scale_emb,
			pad_id=0, finale_id=2, d_inner=2048, n_head=8, d_k=64, d_v=64,
			n_seq_max=512, **kw_args):
		super().__init__()

		assert encoder_type == 'finale', 'encoder type not supported: %s' % encoder_type

		self.encoder = SeqvaeEncoderFinale(n_vocab, d_model=d_enc_model, d_emb=d_model, n_layers=n_encoder_layer, scale_emb=encoder_scale_emb,
			pad_id=pad_id, finale_id=finale_id, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, n_seq_max=n_seq_max)

		self.deducer = nn.ModuleList([self.encoder])	# placeholder to load/save checkpoint


	def forward (self, input_ids, var_factor: float=1):
		mu, logvar = self.encoder(input_ids)
		z = mu

		if var_factor != 0:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			z += eps * std * var_factor

		return z


class SeqvaeDecoderHead (nn.Module):
	def __init__ (self,
			n_vocab, n_layers=6, pad_id=0, d_model=512, dropout=0.1, n_seq_max=512,
			emb_prj_weight_sharing=True, scale_emb=True, **kw_args):
		super().__init__()

		self.pad_id = pad_id
		self.summary_id = n_vocab - 1

		self.scale_prj = scale_emb and emb_prj_weight_sharing
		self.d_model = d_model

		self.decoder = HeadSummaryEncoder(
				n_src_vocab=n_vocab, n_position=n_seq_max,
				d_model=d_model, n_layers=n_layers,
				pad_idx=pad_id, dropout=dropout, scale_emb=not emb_prj_weight_sharing, **kw_args)

		self.word_prj = nn.Linear(d_model, n_vocab, bias=False)

		if emb_prj_weight_sharing:
			# Share the weight between target word embedding & last dense layer
			self.word_prj.weight = self.decoder.src_word_emb.weight


	# seq: (n, seq)
	# summary: (n, d_model)
	def forward(self, seq, summary):
		head = torch.tensor([[self.summary_id]], device=seq.device).repeat(seq.shape[0], 1)
		seq = torch.cat([head, seq], dim=1)	# prepend head element
		trg_mask = get_pad_mask(seq, self.pad_id) & get_subsequent_mask(seq)

		seq = seq.long()
		dec_output = self.decoder(seq, trg_mask, summary)
		seq_logit = self.word_prj(dec_output)
		if self.scale_prj:
			seq_logit *= self.d_model ** -0.5

		return seq_logit[:, 1:]	# clip head element


class SeqvaeDecoderHeadLora (SeqvaeDecoderHead):
	def __init__(self, n_vocab, r=4, alpha=1., bias=False, n_lora_layers=6, n_seq_max=512,
		n_layers=6, emb_prj_weight_sharing=True, scale_emb=True, **kw_args):
		super().__init__(n_vocab, n_layers=n_layers, n_seq_max=n_seq_max, emb_prj_weight_sharing=emb_prj_weight_sharing, scale_emb=scale_emb, **kw_args)

		lora_layers = [LoraEncoderLayer(r=r, alpha=alpha, bias=bias, **kw_args) for i in range(n_lora_layers)]
		layers = [lora_layers[i] if i < n_lora_layers else layer for i, layer in enumerate(self.decoder.layer_stack)]
		self.decoder.layer_stack = nn.ModuleList(layers)


	def initialize (self):
		for layer in self.decoder.layer_stack:
			if hasattr(layer, 'initialize'):
				layer.initialize()


	def freezeTrunk (self):
		for layer in self.decoder.layer_stack:
			if hasattr(layer, 'freezeTrunk'):
				layer.freezeTrunk()
			else:
				for p in layer.parameters():
					p.requires_grad = False


class SeqvaeLoss (nn.Module):
	# encoder_type: 'mean'|'finale'
	def __init__ (self, n_vocab, encoder_type='mean',
		d_model=512, n_encoder_layer=6, n_decoder_layer=6,
		d_enc_model=512, finale_id=2,
		encoder_scale_emb=True, decoder_scale_emb=True, emb_prj_weight_sharing=True,
		kld_weight=0.001, encoder_init_gain=0, lora_config=None, **kw_args):
		super().__init__()

		self.kld_weight = kld_weight
		#self.encoder_type = encoder_type

		if encoder_type == 'finale':
			self.encoder = SeqvaeEncoderFinale(n_vocab, d_model=d_enc_model, d_emb=d_model, finale_id=finale_id, n_layers=n_encoder_layer, scale_emb=encoder_scale_emb, **kw_args)
		else:
			self.encoder = SeqvaeEncoderMean(n_vocab, d_model=d_model, n_layers=n_encoder_layer, scale_emb=encoder_scale_emb, **kw_args)

		if lora_config is not None:
			self.decoder = SeqvaeDecoderHeadLora(n_vocab, d_model=d_model, n_layers=n_decoder_layer, scale_emb=decoder_scale_emb,
				emb_prj_weight_sharing=emb_prj_weight_sharing, **lora_config, **kw_args)
		else:
			self.decoder = SeqvaeDecoderHead(n_vocab + 1, d_model=d_model, n_layers=n_decoder_layer, scale_emb=decoder_scale_emb,
				emb_prj_weight_sharing=emb_prj_weight_sharing, **kw_args)

		self.deducer = nn.ModuleList([self.encoder, self.decoder])	# placeholder to load/save checkpoint

		if encoder_type == 'finale':
			for p in self.encoder.parameters():
				if p.dim() > 1:
					nn.init.xavier_uniform_(p, gain=(n_encoder_layer * 2) ** -0.5)
		else:
			for p in self.encoder.parameters():
				nn.init.normal_(p, std=encoder_init_gain)

		for p in self.decoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(n_decoder_layer * 2) ** -0.5)

		if lora_config is not None:
			self.decoder.initialize()
			self.decoder.freezeTrunk()


	def preloadDecoder (self, other):
		m, u = self.decoder.decoder.load_state_dict(other.decoder.state_dict(), strict=False)
		if len(u) > 0:
			logging.warn('[SeqvaeLoss.preloadDecoder] unexpected states: %s', u)


	def reparameterize (self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)

		return eps * std + mu


	def forward (self, batch):
		mask = batch['body_mask']

		mu, logvar = self.encoder(batch['input_ids'], mask=mask)
		z = self.reparameterize(mu, logvar)	# (n, d_model)
		pred = self.decoder(batch['input_ids'], z)
		target = batch['output_ids'].long()

		pred_flat = pred[mask]
		target_flat = target[mask]

		recons_loss = F.cross_entropy(pred_flat, target_flat)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

		loss = recons_loss + kld_loss * self.kld_weight

		pred_ids = torch.argmax(pred_flat, dim=-1)
		acc = (pred_ids == target_flat).float().mean()

		return loss, {
			'acc': acc.item(),
			'recons_loss': recons_loss.item(),
			'kld_loss': kld_loss.item(),
		}


	def inspectRun (self, batch):
		mask = batch['body_mask']

		mu, logvar = self.encoder(batch['input_ids'])
		z = self.reparameterize(mu, logvar)	# (n, d_model)
		pred = self.decoder(batch['input_ids'], z)
		target = batch['output_ids'].long()

		pred_flat = pred[mask]
		target_flat = target[mask]

		recons_loss = F.cross_entropy(pred_flat, target_flat)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

		loss = recons_loss + kld_loss * self.kld_weight

		pred_ids = torch.argmax(pred_flat, dim=-1)
		acc = (pred_ids == target_flat).float().mean()

		return {
			'mu': mu,
			'logvar': logvar,
			'z': z,
			'pred': pred,
			'pred_flat': pred_flat,
			'target_flat': target_flat,
			'truth': pred_ids == target_flat,
			'acc': acc.item(),
			'recons_loss': recons_loss.item(),
			'kld_loss': kld_loss.item(),
		}
