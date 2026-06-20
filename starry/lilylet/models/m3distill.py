'''ABC→Lilylet M3 distillation: a Lilylet symbolic encoder trained to match the
frozen CLaMP 3 ABC M3 encoder's pooled embedding.

The teacher (ABC M3 encoder, starry/lilylet/models/m3.py) is run offline by
tools/lilylet/preprocessLilyletM3.py, which stores the mean-pooled [768] target per
piece. This student reads the paired Lilylet patches, encodes them with the SAME
M3 architecture (one-hot char patches → linear patch-embedding → BERT over the patch
sequence), masked-mean-pools to a single [768] vector, projects, and is trained by
cosine regression to the teacher target. Distillation (not contrastive): the pairing
is deterministic and the teacher is frozen, so we want absolute alignment in the
teacher's space.

Inference model (deducer): LilyletM3Encoder
Loss model (training wrapper): LilyletM3EncoderLoss

Batch contract (from LilyletM3Distill.collateBatch):
	input_patches:    LongTensor  [B, T, patch_size]  Lilylet token ids in [0, num_classes)
	input_masks:      LongTensor  [B, T]              1 for real patch, 0 for padding
	target_embedding: FloatTensor [B, hidden]         frozen ABC M3 pooled target
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertModel, PreTrainedModel

from .m3 import (
	M3PatchEncoder, build_m3_config, load_m3_encoder,
	PATCH_SIZE, PATCH_LENGTH, M3_HIDDEN_SIZE, PATCH_NUM_LAYERS, NUM_CLASSES,
)
from ...utils.registry import register_model


def _masked_mean (features, masks):
	'''Masked average over the patch (time) dim. features [B, T, H], masks [B, T].
	Matches CLaMP's avg_pooling: sum of real-patch features / real-patch count.'''
	masks = masks.unsqueeze(-1).to(features.dtype)            # [B, T, 1]
	summed = (features * masks).sum(dim=1)                    # [B, H]
	count = masks.sum(dim=1).clamp(min=1.0)                   # [B, 1]
	return summed / count


@register_model
class LilyletM3Encoder (nn.Module):
	'''Student Lilylet M3 encoder.

	Same architecture as the ABC M3 encoder (M3PatchEncoder: one-hot patches →
	linear patch-embedding → BERT), plus a masked-mean pool and an optional
	projection so the pooled output lives in the teacher's [hidden] space.

	Args (from config['model.args']):
		num_classes:      character vocabulary / one-hot size (Lilylet manual tokenizer = 256)
		hidden_size:      encoder + output hidden size (must match teacher; 768)
		patch_size:       tokens per patch (M3 = 64)
		patch_length:     max patches (position-embedding cap; 512)
		patch_num_layers: BERT encoder layers (12)
		n_head:           attention heads (default hidden_size // 64)
		project:          add a final Linear(hidden, hidden) head (default True)
		warm_start:       if True, initialize the encoder from the frozen ABC M3 teacher
		                  weights (same arch); requires num_classes == teacher's (128)
		teacher_weights:  optional explicit teacher weight path for warm_start
	'''

	def __init__ (self, num_classes=NUM_CLASSES, hidden_size=M3_HIDDEN_SIZE,
		patch_size=PATCH_SIZE, patch_length=PATCH_LENGTH, patch_num_layers=PATCH_NUM_LAYERS,
		n_head=None, project=True, warm_start=False, teacher_weights=None, **_):
		super().__init__()

		# NOTE: PATCH_SIZE / hidden are module-level in m3.py; M3PatchEncoder reads them
		# globally, so we keep them at the defaults (64 / 768) and only vary num_classes.
		self.num_classes = num_classes
		self.hidden_size = hidden_size

		config = BertConfig(
			vocab_size=1,
			hidden_size=hidden_size,
			num_hidden_layers=patch_num_layers,
			num_attention_heads=n_head or (hidden_size // 64),
			intermediate_size=hidden_size * 4,
			max_position_embeddings=patch_length,
		)
		self.encoder = M3PatchEncoder(config, num_classes=num_classes)

		if warm_start:
			# copy the frozen ABC M3 tower as initialization (same arch; needs matching num_classes)
			teacher = load_m3_encoder(weights_path=teacher_weights, device='cpu')
			self.encoder.load_state_dict(teacher.state_dict(), strict=True)

		self.proj = nn.Linear(hidden_size, hidden_size) if project else None
		if self.proj is not None:
			nn.init.normal_(self.proj.weight, std=0.02)

	def forward (self, patches: torch.Tensor, masks: torch.Tensor):
		'''
		patches: LongTensor [B, T, patch_size]  Lilylet token ids
		masks:   LongTensor [B, T]              1 for real patch, 0 for padding
		Returns: FloatTensor [B, hidden]        pooled (+projected) Lilylet embedding

		Sequences longer than the position-embedding cap (patch_length) are split into
		consecutive ≤cap chunks; each chunk is encoded and masked-mean-pooled, then the
		chunk vectors are length-weighted-averaged. This mirrors how the ABC teacher
		(extract_clamp2, get_normalized path) aggregates over-long pieces, so the student
		target stays consistent. Within the cap it is a single encoder pass.
		'''
		cap = self.encoder.config.max_position_embeddings
		T = patches.shape[1]

		if T <= cap:
			hidden = self.encoder(patches, masks)['last_hidden_state']   # [B, T, hidden]
			pooled = _masked_mean(hidden, masks)                         # [B, hidden]
		else:
			seg_sum = None
			seg_weight = None
			for start in range(0, T, cap):
				p = patches[:, start:start + cap]
				m = masks[:, start:start + cap]
				h = self.encoder(p, m)['last_hidden_state']
				w = m.sum(dim=1, keepdim=True).to(h.dtype)               # [B, 1] real patches in chunk
				chunk_pooled = _masked_mean(h, m) * w                    # un-normalize: sum of features
				seg_sum = chunk_pooled if seg_sum is None else seg_sum + chunk_pooled
				seg_weight = w if seg_weight is None else seg_weight + w
			pooled = seg_sum / seg_weight.clamp(min=1.0)                 # [B, hidden]

		if self.proj is not None:
			pooled = self.proj(pooled)
		return pooled


@register_model
class LilyletM3EncoderLoss (nn.Module):
	'''Training wrapper: cosine-regression distillation against the frozen teacher target.

	loss = (1 - cos(student, target)) + mse_weight * MSE(student, target)
	The cosine term is the primary objective (downstream uses cosine similarity); a
	small optional MSE term anchors the magnitude / discourages collapse.
	'''

	def __init__ (self, mse_weight=0.0, **kw_args):
		super().__init__()
		self.deducer = LilyletM3Encoder(**kw_args)
		self.mse_weight = mse_weight

	def training_parameters (self):
		# all student params are trained; distributed trainer broadcasts these from the
		# trainer rank. (cf. LilyletNotaGenLoss / ScoreRegressionLoss)
		return list(self.deducer.parameters()) + list(self.deducer.buffers())

	def validation_parameters (self):
		# no validation-only parameters to broadcast back from the validator rank
		return []

	def forward (self, batch):
		pred = self.deducer(batch['input_patches'], batch['input_masks'])   # [B, hidden]
		target = batch['target_embedding']                                 # [B, hidden]

		cos = F.cosine_similarity(pred, target, dim=-1)                    # [B]
		cos_loss = (1.0 - cos).mean()
		loss = cos_loss
		mse = F.mse_loss(pred, target)
		if self.mse_weight > 0:
			loss = loss + self.mse_weight * mse

		with torch.no_grad():
			# expose all loss components so the trainer logs / TB-plots them: 'cos' is the
			# primary alignment metric; 'cos_loss' (=1-cos) lets a +MSE run be compared on
			# the same scale as a pure-cosine run; 'mse' shows the raw magnitude term so the
			# mse_weight can be judged (not drowned out / not dominating).
			metric = {'cos': cos.mean().item(), 'cos_loss': cos_loss.item(), 'mse': mse.item()}
		return loss, metric

