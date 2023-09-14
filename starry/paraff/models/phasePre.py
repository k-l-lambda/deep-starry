
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ...transformer.models import get_subsequent_mask
from ...modules.positionEncoder import SinusoidEncoder
from .sparseAE import AttentionStack
from ..vocab import ID_MSUM



class PhasePre (nn.Module):
	def __init__ (self,
			n_type, n_vocab, pad_id=0, sum_id=ID_MSUM, d_phase=128, d_token=128, d_summary=256,
			n_layers=6, d_inner=1024, n_head=8, d_k=32, d_v=32,
			dropout=0.1, angle_cycle=10000, **_):
		super().__init__()

		self.d_phase = d_phase
		self.d_token = d_token
		self.d_model = d_phase + d_token

		self.pad_id = pad_id
		self.sum_id = sum_id

		self.ph_emb = nn.Embedding(n_type, d_phase, padding_idx=pad_id)
		self.summary_emb = nn.Linear(d_summary, d_phase, bias=False)
		self.pos_encoder_half = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_phase // 2)
		self.pos_encoder = SinusoidEncoder(angle_cycle=angle_cycle, d_hid=d_phase)

		self.word_emb = nn.Embedding(n_vocab, d_token, padding_idx=pad_id)
		self.word_prj = nn.Linear(self.d_model, n_vocab, bias=False)

		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

		self.attention = AttentionStack(d_model=self.d_model, n_layers=n_layers, dropout=dropout, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v)


	# phid:		(n, n_phase)
	# f_pos:	(n, n_phase)
	# b_pos:	(n, n_phase)
	# summary:	(n, n_phase, d_summary)
	# ph_mask:	(n, n_phase)
	# next:		(n, n_phase)
	# id:		(n, n_word)
	# id_pos:	(n, n_word)
	# id_mask:	(n, n_word)
	def forward (self, phid, f_pos, b_pos, summary, ph_mask, next, id, id_pos, id_mask: Optional[torch.Tensor]=None):	# -> (n, n_word, n_vocab)
		n_batch = id.shape[0]
		n_word = id.shape[1]
		n_phase = phid.shape[1]
		n_seq = n_phase + n_word

		id_mask = (id != self.pad_id) if id_mask is None else id_mask
		mask = torch.cat([ph_mask, id_mask], dim=-1)
		mask = mask.unsqueeze(-2) & (1 - torch.triu(torch.ones((1, n_seq, n_seq), device=mask.device), diagonal=1)).bool()

		phid = F.pad(phid, (0, n_word), value=0)

		next_f, next_b = f_pos[next], b_pos[next]
		next_f, next_b = next_f.view(n_batch, -1).repeat(1, n_word), next_b.view(n_batch, -1).repeat(1, n_word)

		f_pos = torch.cat([f_pos, next_f], dim=-1)
		b_pos = torch.cat([b_pos, next_b], dim=-1)
		f_pos_code = self.pos_encoder_half(f_pos.float())
		b_pos_code = self.pos_encoder_half(b_pos.float())
		phase_position = torch.cat([f_pos_code, b_pos_code], dim=-1)
		phase_position[:, :n_phase][torch.logical_not(ph_mask | next)] = 0

		phases = self.ph_emb(phid.long())
		phases += phase_position
		phases[:, :summary.shape[1]] += self.summary_emb(summary)

		id = F.pad(id, (n_phase, 0), value=self.sum_id)
		words = self.word_emb(id.long())
		words[:, -n_word:] += self.pos_encoder(id_pos.float())

		x = torch.cat([phases, words], dim=-1)
		x *= self.d_model ** 0.5	# scale embedding
		x = self.dropout(x)
		x = self.layer_norm(x)
		x = self.attention(x, mask=mask)
		x = self.word_prj(x)

		return x[:, -n_word:]


class PhasePreLoss (nn.Module):
	def __init__ (self, **kw_args):
		super().__init__()

		self.deducer = PhasePre(**kw_args)


	def forward (self, batch):
		body_mask = batch['body_mask']
		target = batch['output_ids'].long()
		target_body = target[body_mask]

		ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask, ph_next_mask = batch['ph_id'], batch['ph_f_num'], batch['ph_b_num'], batch['ph_summary'], batch['ph_body_mask'], batch['ph_next_mask']
		ph_summary[ph_next_mask] = torch.zeros_like(ph_summary[ph_next_mask])

		id = batch['input_ids']
		id_pos = batch['position']

		pred = self.deducer(ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask, ph_next_mask,
			id, id_pos)
		pred_body = pred[body_mask]

		loss = F.cross_entropy(pred_body, target_body)

		pred_ids = pred_body.argmax(dim=-1)
		acc = (pred_ids == target_body).float().mean()

		metric = {
			'acc': acc.item(),
		}

		if not self.training:
			zero_mask = torch.zeros_like(ph_body_mask)

			pred_zl = self.deducer(ph_id, ph_f_num, ph_b_num, ph_summary, zero_mask, ph_next_mask, id, id_pos)
			pred_np = self.deducer(ph_id, ph_f_num, ph_b_num, ph_summary, ph_body_mask, ph_next_mask, id, id_pos, body_mask)
			pred_zlnp = self.deducer(ph_id, ph_f_num, ph_b_num, ph_summary, zero_mask, ph_next_mask, id, id_pos, body_mask)

			error = 1 - acc
			error_zero_latent = 1 - (pred_zl[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_no_primer = 1 - (pred_np[body_mask].argmax(dim=-1) == target_body).float().mean()
			error_zero_latent_no_primer = 1 - (pred_zlnp[body_mask].argmax(dim=-1) == target_body).float().mean()

			metric['error'] = error.item()
			metric['error_zero_latent'] = error_zero_latent.item()
			metric['error_no_primer'] = error_no_primer.item()
			metric['error_zero_latent_no_primer'] = error_zero_latent_no_primer.item()

		return loss, metric
