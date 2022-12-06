
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Encoder, Decoder, Jointer, FrameEncoder



class FrameMatchJointer (nn.Module):
	def __init__ (self, n_layers_ce=1, n_layers_sd=1,
			d_model=128, d_time=16, angle_cycle=10e+3, d_inner=512, n_head=8, d_k=64, d_v=64,
			dropout=0.1, scale_emb=False, **_):
		super().__init__()

		self.frame_encoder = FrameEncoder(d_model=d_model, d_time=d_time, angle_cycle=angle_cycle)

		encoder_args = dict(n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)

		self.c_encoder = Encoder(n_layers_ce, **encoder_args, scale_emb=scale_emb)
		self.s_decoder = Decoder(n_layers_sd, **encoder_args, scale_emb=scale_emb)

		self.jointer = Jointer(d_model)


	def forward (self, c_time, c_frame, s_time, s_frame):
		vec_c = self.frame_encoder((c_time, c_frame))
		vec_s = self.frame_encoder((s_time, s_frame))

		vec_c = self.c_encoder(vec_c)

		vec_s = self.s_decoder(vec_s, vec_c)

		return self.jointer(vec_s, vec_c)


class FrameMatchJointerLoss (nn.Module):
	def __init__ (self, init_gain_n=1, reg_orthogonality=0, reg_orth_exclude_pitch=False,
		main_acc='acc_tail8', exp_focal=0, **kw_args):
		super().__init__()

		self.deducer = FrameMatchJointer(**kw_args)

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

		sample_mask = sample[1].sum(dim=-1) > 0
		sample_mask_c1 = torch.logical_and(sample_mask, ci > 0)
		sample_mask8 = sample_mask.clone()
		sample_mask8[:, :-8] = False

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
		accuracy = corrects / max(1, torch.numel(ci[sample_mask]))

		corrects_c1 = (ci_pred[sample_mask_c1] == ci[sample_mask_c1]).sum().item()
		acc_c1 = corrects_c1 / max(1, torch.numel(ci[sample_mask_c1]))

		corrects8 = (ci_pred[sample_mask8] == ci[sample_mask8]).sum().item()
		acc_tail8 = corrects8 / max(1, torch.numel(ci[sample_mask8]))

		corrects_tip = (ci_pred[:, -1] == ci[:, -1]).sum().item()
		acc_tip = corrects_tip / max(1, torch.numel(ci[:, -1]))

		return loss, {
			'loss_orth': loss_orth,
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
			acc=metrics[self.main_acc] / n_batch,
		)
