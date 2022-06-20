
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Encoder, Decoder, Jointer, NoteEncoder, NoteEncoder2



class MatchJointerRaw (nn.Module):
	def __init__ (self, n_layers_ce=1, n_layers_se=1, n_layers_sd=1,
			d_model=1, d_inner=4, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.c_encoder = Encoder(n_layers_ce, **encoder_args, scale_emb=scale_emb)
		self.s_encoder = Encoder(n_layers_se, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = Decoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_input, s_input):
		c_output = self.c_encoder(c_input)
		s_output = self.s_encoder(s_input)

		s_output = self.s_decoder(s_output, c_output)

		return self.jointer(s_output, c_output)


class MatchJointer1 (nn.Module):
	def __init__ (self, n_layers_ce=1, n_layers_se=1, n_layers_sd=1,
			d_model=128, d_time=64, angle_cycle=100000, d_inner=512, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		self.note_encoder = NoteEncoder(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.c_encoder = Encoder(n_layers_ce, **encoder_args, scale_emb=scale_emb)
		self.s_encoder = Encoder(n_layers_se, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = Decoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_time, c_pitch, c_velocity, s_time, s_pitch, s_velocity):
		vec_c = self.note_encoder((c_time, c_pitch, c_velocity))
		vec_s = self.note_encoder((s_time, s_pitch, s_velocity))

		vec_c = self.c_encoder(vec_c)
		vec_s = self.s_encoder(vec_s)

		vec_s = self.s_decoder(vec_s, vec_c)

		return self.jointer(vec_s, vec_c)


class MatchJointerLossGeneric (nn.Module):
	def __init__ (self, deducer_class, **kw_args):
		super().__init__()

		self.deducer = deducer_class(**kw_args)

		self.bce = nn.BCELoss()

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)


	def forward (self, batch):
		criterion, sample, ci = batch['criterion'], batch['sample'], batch['ci']
		c_len = criterion[0].shape[1]

		matching_truth = F.one_hot(ci, num_classes=c_len + 1).float()[:, :, 1:]
		matching_pred, code_src, code_tar = self.deducer(*criterion, *sample)

		sample_mask = sample[1] > 0
		sample_mask_c1 = torch.logical_and(sample_mask, ci > 0)
		sample_mask8 = sample_mask.clone()
		sample_mask8[:, :-8] = False

		loss = self.bce(matching_pred[sample_mask], matching_truth[sample_mask])

		# pad 1 element at beginning of each sequence as none-matching
		matching_pred_1 = torch.cat([torch.ones((*matching_pred.shape[:-1], 1), device=matching_pred.device) * 1e-3, matching_pred], dim=-1)
		ci_pred = torch.argmax(matching_pred_1, dim=-1)

		corrects = (ci_pred[sample_mask] == ci[sample_mask]).sum().item()
		accuracy = corrects / torch.numel(ci[sample_mask])

		corrects_c1 = (ci_pred[sample_mask_c1] == ci[sample_mask_c1]).sum().item()
		acc_c1 = corrects_c1 / torch.numel(ci[sample_mask_c1])

		corrects8 = (ci_pred[sample_mask8] == ci[sample_mask8]).sum().item()
		acc_tail8 = corrects8 / torch.numel(ci[sample_mask8])

		corrects_tip = (ci_pred[:, -1] == ci[:, -1]).sum().item()
		acc_tip = corrects_tip / torch.numel(ci[:, -1])

		return loss, {
			'acc_full': accuracy,
			'acc_c1': acc_c1,
			'acc_tail8': acc_tail8,
			'acc_tip': acc_tip,
		}


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())


	def stat (self, metrics, n_batch):
		return dict(
			accuracy={k: v / n_batch for k, v in metrics.items()},
			acc=metrics['acc_tail8'] / n_batch,
		)


class MatchJointer1Loss (MatchJointerLossGeneric):
	def __init__(self, **kw_args):
		super().__init__(deducer_class=MatchJointer1, **kw_args)


class MatchJointer2 (nn.Module):
	def __init__ (self, n_layers_ce=1, n_layers_se=1, n_layers_sd=1,
			d_model=128, d_time=16, angle_cycle=10e+3, d_inner=512, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		self.note_encoder = NoteEncoder2(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.c_encoder = Encoder(n_layers_ce, **encoder_args, scale_emb=scale_emb)
		self.s_encoder = Encoder(n_layers_se, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = Decoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_time, c_pitch, c_velocity, s_time, s_pitch, s_velocity):
		vec_c = self.note_encoder((c_time, c_pitch, c_velocity))
		vec_s = self.note_encoder((s_time, s_pitch, s_velocity))

		vec_c = self.c_encoder(vec_c)
		vec_s = self.s_encoder(vec_s)

		vec_s = self.s_decoder(vec_s, vec_c)

		return self.jointer(vec_s, vec_c)


class MatchJointer2Loss (MatchJointerLossGeneric):
	def __init__(self, **kw_args):
		super().__init__(deducer_class=MatchJointer2, **kw_args)


class MatchJointer3 (nn.Module):
	def __init__ (self, n_layers_enc=1, n_layers_sd=1,
			d_model=128, d_time=16, angle_cycle=10e+3, d_inner=512, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		self.note_encoder = NoteEncoder2(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.encoder = Encoder(n_layers_enc, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = Decoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_time, c_pitch, c_velocity, s_time, s_pitch, s_velocity):
		vec_c = self.note_encoder((c_time, c_pitch, c_velocity))
		vec_s = self.note_encoder((s_time, s_pitch, s_velocity))

		vec_c = self.encoder(vec_c)
		vec_s = self.encoder(vec_s)

		vec_s = self.s_decoder(vec_s, vec_c)

		return self.jointer(vec_s, vec_c)


class MatchJointer3Loss (MatchJointerLossGeneric):
	def __init__(self, **kw_args):
		super().__init__(deducer_class=MatchJointer3, **kw_args)
