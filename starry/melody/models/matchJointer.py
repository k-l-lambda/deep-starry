
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Encoder, Decoder, Jointer, NoteEncoder



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


class MatchJointer1Loss (nn.Module):
	def __init__ (self, **kw_args):
		super().__init__()

		self.deducer = MatchJointer1(**kw_args)

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

		loss = self.bce(matching_pred, matching_truth)

		matching_pred_1 = torch.cat([torch.ones((*matching_pred.shape[:-1], 1), device=matching_pred.device) * 1e-3, matching_pred], dim=-1)
		ci_pred = torch.argmax(matching_pred_1, dim=-1)
		corrects = (ci_pred == ci).sum().item()
		accuracy = corrects / torch.numel(ci)

		return loss, {
			'accuracy': accuracy,
		}


	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())
