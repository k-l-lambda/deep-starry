
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...transformer.models import Encoder, get_pad_mask, get_subsequent_mask
from ...transformer.layers import EncoderLayer
from .modules import HeadSummaryEncoder



class SeqvaeEncoder (nn.Module):
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


	def forward(self, seq):
		trg_mask = get_pad_mask(seq, self.pad_id)

		seq = seq.long()
		enc_output, *_ = self.encoder(seq, trg_mask)

		mu, *_ = self.out_mu(enc_output)
		logvar, *_ = self.out_var(enc_output)

		# sum along sequence dim
		mu = mu.sum(dim=1)
		logvar = logvar.sum(dim=1).clip(max=80)	# avoid inf after exp

		return mu, logvar


class SeqvaeDecoderHead (nn.Module):
	def __init__ (self,
			n_vocab, n_layers=6, pad_id=0, d_model=512, d_inner=2048,
			n_head=8, d_k=64, d_v=64,
			dropout=0.1, n_seq_max=512,
			emb_prj_weight_sharing=True, scale_emb=True):
		super().__init__()

		self.pad_id = pad_id
		self.summary_id = n_vocab - 1

		self.scale_prj = scale_emb and emb_prj_weight_sharing
		self.d_model = d_model

		self.decoder = HeadSummaryEncoder(
			n_src_vocab=n_vocab, n_position=n_seq_max,
			d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			pad_idx=pad_id, dropout=dropout, scale_emb=not emb_prj_weight_sharing)

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


class SeqvaeLoss (nn.Module):
	def __init__ (self, n_vocab, n_encoder_layer=6, n_decoder_layer=6,
		encoder_scale_emb=True, decoder_scale_emb=True, emb_prj_weight_sharing=True,
		kld_weight=0.001, **kw_args):
		super().__init__()

		self.kld_weight = kld_weight

		self.encoder = SeqvaeEncoder(n_vocab, n_layers=n_encoder_layer, scale_emb=encoder_scale_emb, **kw_args)
		self.decoder = SeqvaeDecoderHead(n_vocab + 1, n_layers=n_decoder_layer, scale_emb=decoder_scale_emb,
			emb_prj_weight_sharing=emb_prj_weight_sharing, **kw_args)

		self.deducer = nn.ModuleList([self.encoder, self.decoder])	# placeholder to load/save checkpoint

		for p in self.encoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(n_encoder_layer * 2) ** -0.5) 

		for p in self.decoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=(n_decoder_layer * 2) ** -0.5)


	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)

		return eps * std + mu


	def forward(self, batch):
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

		return loss, {
			'acc': acc.item(),
			'recons_loss': recons_loss.item(),
			'kld_loss': kld_loss.item(),
		}
