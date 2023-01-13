
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Encoder, Decoder, ThinDecoder, Jointer, NoteEncoder, NoteEncoder2, TimeGuidEncoder
from ...utils.weightedValue import WeightedValue



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
	def __init__ (self, deducer_class, init_gain_n=1, reg_orthogonality=0, reg_orth_exclude_pitch=False,
		main_acc='acc_tail8', exp_focal=0, **kw_args):
		super().__init__()

		self.deducer = deducer_class(**kw_args)

		if exp_focal != 0:
			SEQ_LEN_MAX = 0x100
			focal_weight = torch.exp(torch.arange(SEQ_LEN_MAX).float() / exp_focal).flip(0)
			self.register_buffer('focal_weight', focal_weight, persistent=False)

		self.reg_orthogonality = reg_orthogonality
		self.reg_orth_exclude_pitch = reg_orth_exclude_pitch
		self.main_acc = main_acc

		# initialize parameters
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p, gain=init_gain_n ** -0.5)


	def forward (self, batch):
		criterion, sample, ci = batch['criterion'], batch['sample'], batch['ci']
		c_len = criterion[0].shape[1]

		matching_truth = F.one_hot(ci, num_classes=c_len + 1).float()[:, :, 1:]
		matching_pred, code_src, code_tar = self.deducer(*criterion, *sample)

		sample_mask = sample[1] > 0
		sample_mask_c1 = torch.logical_and(sample_mask, ci > 0)
		sample_mask8 = sample_mask.clone()
		sample_mask8[:, :-8] = False
		if len(sample) > 3:
			sample_mask8 = torch.logical_and(sample_mask8, sample[4])

		#loss = self.bce(matching_pred[sample_mask], matching_truth[sample_mask])
		weight = None
		if hasattr(self, 'focal_weight'):
			#print('focal_weight:', self.focal_weight)
			weight = self.focal_weight[-matching_truth.shape[1]:]
			mean = weight.mean()
			weight = (weight / mean).view(1, -1, 1).repeat(matching_truth.shape[0], 1, matching_truth.shape[2])
			weight = weight[sample_mask]
		loss = F.binary_cross_entropy(matching_pred[sample_mask], matching_truth[sample_mask], weight=weight)

		loss_orth = 0
		if self.reg_orthogonality > 0:
			code_tar_transposed = code_tar.transpose(-2, -1)
			tartar = code_tar.matmul(code_tar_transposed)
			mask = (torch.triu(torch.ones(tartar.shape[0], c_len, c_len), diagonal=1) == 0).to(tartar.device)

			if self.reg_orth_exclude_pitch:
				cp = criterion[1]
				cp_v = cp.view(cp.shape[0], cp.shape[1], 1)
				cp_h = cp.view(cp.shape[0], 1, cp.shape[1])
				cp_v, cp_h = torch.broadcast_tensors(cp_v, cp_h)
				mask = torch.logical_and(mask, cp_v != cp_h)

			loss_orth = torch.abs(tartar[mask]).mean()

			loss += loss_orth * self.reg_orthogonality

		# pad 1 element at beginning of each sequence as none-matching
		matching_pred_1 = torch.cat([torch.ones((*matching_pred.shape[:-1], 1), device=matching_pred.device) * 1e-3, matching_pred], dim=-1)
		ci_pred = torch.argmax(matching_pred_1, dim=-1)

		corrects = (ci_pred[sample_mask] == ci[sample_mask]).sum().item()
		accuracy = WeightedValue(corrects, torch.numel(ci[sample_mask]))

		corrects_c1 = (ci_pred[sample_mask_c1] == ci[sample_mask_c1]).sum().item()
		acc_c1 = WeightedValue(corrects_c1, torch.numel(ci[sample_mask_c1]))

		corrects8 = (ci_pred[sample_mask8] == ci[sample_mask8]).sum().item()
		acc_tail8 = WeightedValue(corrects8, torch.numel(ci[sample_mask8]))

		corrects_tip = (ci_pred[:, -1] == ci[:, -1]).sum().item()
		acc_tip = WeightedValue(corrects_tip, torch.numel(ci[:, -1]))

		metric = dict(loss_orth=WeightedValue(loss_orth.item()), acc_full=accuracy, acc_c1=acc_c1, acc_tail8=acc_tail8, acc_tip=acc_tip)

		if len(sample) > 3:
			ng_mask = sample[4]
			guid_mask = torch.logical_not(ng_mask)
			ng_ci_mask = torch.logical_and(ng_mask, sample_mask_c1)
			corrects_guid = (ci_pred[guid_mask] == ci[guid_mask]).sum().item()
			corrects_ng = (ci_pred[ng_ci_mask] == ci[ng_ci_mask]).sum().item()

			metric['acc_guid'] = WeightedValue(corrects_guid, torch.numel(ci[guid_mask]))
			metric['acc_ng'] = WeightedValue(corrects_ng, torch.numel(ci[ng_ci_mask]))

		return loss, metric


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())


	def stat (self, metrics, n_batch):
		return dict(
			accuracy={k: v.value for k, v in metrics.items()},
			acc=metrics[self.main_acc].value,
		)


class MatchJointer1Loss (MatchJointerLossGeneric):
	def __init__(self, **kw_args):
		super().__init__(deducer_class=MatchJointer1, **kw_args)


# use NoteEncoder2
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


# share encoder between sample & criterion
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


# use ThinDecoder
class MatchJointer4 (nn.Module):
	def __init__ (self, n_layers_enc=1, n_layers_sd=1,
			d_model=128, d_time=16, angle_cycle=10e+3, d_inner=512, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		self.note_encoder = NoteEncoder2(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.encoder = Encoder(n_layers_enc, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = ThinDecoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_time, c_pitch, c_velocity, s_time, s_pitch, s_velocity):
		vec_c = self.note_encoder((c_time, c_pitch, c_velocity))
		vec_s = self.note_encoder((s_time, s_pitch, s_velocity))

		vec_c = self.encoder(vec_c)
		vec_s = self.encoder(vec_s)

		vec_s = self.s_decoder(vec_s, vec_c)

		return self.jointer(vec_s, vec_c)


class MatchJointer4Loss (MatchJointerLossGeneric):
	def __init__(self, **kw_args):
		super().__init__(deducer_class=MatchJointer4, **kw_args)


# add s_guid input on MatchJointer2
class MatchJointer2Plus (nn.Module):
	def __init__ (self, n_layers_ce=1, n_layers_se=1, n_layers_sd=1,
			d_model=128, d_time=16, angle_cycle=10e+3, d_inner=512, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		self.note_encoder = NoteEncoder2(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)

		self.guid_encoder = TimeGuidEncoder(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.c_encoder = Encoder(n_layers_ce, **encoder_args, scale_emb=scale_emb)
		self.s_encoder = Encoder(n_layers_se, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = Decoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_time, c_pitch, c_velocity, s_time, s_pitch, s_velocity, s_guid, s_guid_mask):
		vec_c = self.note_encoder((c_time, c_pitch, c_velocity)) + self.guid_encoder((c_time, None))
		vec_s = self.note_encoder((s_time, s_pitch, s_velocity)) + self.guid_encoder((s_guid, s_guid_mask))

		vec_c = self.c_encoder(vec_c)
		vec_s = self.s_encoder(vec_s)

		vec_s = self.s_decoder(vec_s, vec_c)

		return self.jointer(vec_s, vec_c)


class MatchJointer2PlusLoss (MatchJointerLossGeneric):
	def __init__(self, **kw_args):
		super().__init__(deducer_class=MatchJointer2Plus, **kw_args)


# index (instead of time) s_guid
class MatchJointer2PlusI (nn.Module):
	def __init__ (self, n_layers_ce=1, n_layers_se=1, n_layers_sd=1,
			d_model=128, d_time=16, angle_cycle=10e+3, d_inner=512, n_head=8, d_k=64, d_v=64,
			softindex_scale=0, dropout=0.1, scale_emb=False, **_):
		super().__init__()

		self.note_encoder = NoteEncoder2(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle, softindex_scale=softindex_scale)

		self.guid_encoder = TimeGuidEncoder(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.c_encoder = Encoder(n_layers_ce, **encoder_args, scale_emb=scale_emb)
		self.s_encoder = Encoder(n_layers_se, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = Decoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_time, c_pitch, c_velocity, s_time, s_pitch, s_velocity, s_ci, s_ng_mask):
		cci = (torch.arange(0, c_time.shape[1], device=s_ci.device) + 1).repeat(c_time.shape[0], 1)

		vec_c = self.note_encoder((c_time, c_pitch, c_velocity)) + self.guid_encoder((cci, None))
		vec_s = self.note_encoder((s_time, s_pitch, s_velocity)) + self.guid_encoder((s_ci, s_ng_mask))

		vec_c = self.c_encoder(vec_c)
		vec_s = self.s_encoder(vec_s)

		vec_s = self.s_decoder(vec_s, vec_c)

		return self.jointer(vec_s, vec_c)


class MatchJointer2PlusILoss (MatchJointerLossGeneric):
	def __init__(self, **kw_args):
		super().__init__(deducer_class=MatchJointer2PlusI, **kw_args)
