
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...transformer.models import Encoder, get_pad_mask, get_subsequent_mask



class TokenGen (nn.Module):
	def __init__ (self,
			n_vocab, pad_id=0, d_model=512, d_inner=2048,
			n_layers=6, n_head=8, d_k=64, d_v=64,
			dropout=0.1, n_seq_max=512,
			emb_prj_weight_sharing=False, scale_emb_or_prj='prj'):
		super().__init__()

		self.pad_id = pad_id

		# scale_emb_or_prj:
		#	'emb': multiply \sqrt{d_model} to embedding output
		#	'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
		#	None: no multiplication

		assert scale_emb_or_prj in ['emb', 'prj', None]
		scale_emb = (scale_emb_or_prj == 'emb') if emb_prj_weight_sharing else False
		self.scale_prj = (scale_emb_or_prj == 'prj') if emb_prj_weight_sharing else False
		self.d_model = d_model

		self.decoder = Encoder(
			n_src_vocab=n_vocab, n_position=n_seq_max,
			d_word_vec=d_model, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			pad_idx=pad_id, dropout=dropout, scale_emb=scale_emb)

		self.word_prj = nn.Linear(d_model, n_vocab, bias=False)

		if emb_prj_weight_sharing:
			# Share the weight between target word embedding & last dense layer
			self.word_prj.weight = self.decoder.src_word_emb.weight


	def forward(self, seq):
		trg_mask = get_pad_mask(seq, self.pad_id) & get_subsequent_mask(seq)

		seq = seq.long()
		dec_output, *_ = self.decoder(seq, trg_mask)
		seq_logit = self.word_prj(dec_output)
		if self.scale_prj:
			seq_logit *= self.d_model ** -0.5

		return seq_logit


class TokenGenLoss (nn.Module):
	def __init__ (self, **kw_args):
		super().__init__()

		self.deducer = TokenGen(**kw_args)

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p) 


	def forward (self, batch):
		pred = self.deducer(batch['input_ids'])
		target = batch['output_ids'].long()
		pred_ncs = pred.permute(0, 2, 1)

		loss = F.cross_entropy(pred_ncs, target)

		non_zero = target != 0
		pred_ids = torch.argmax(pred, dim=-1)
		acc = (pred_ids[non_zero] == target[non_zero]).float().mean()

		return loss, {'acc': acc.item()}
