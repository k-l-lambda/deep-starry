
'''
NotaGen-style hierarchical patch/char model for Lilylet, adapted to deep-starry conventions.

Reference: /home/camus/work/NotaGen/pretrain/utils.py

The model has two levels:
	- PatchLevelDecoder: encodes a sequence of patches (each patch is a fixed-length
	  sequence of token ids) into per-patch hidden states with a GPT2 backbone.
	- CharLevelDecoder: an auto-regressive GPT2 LM that, conditioned on a patch's encoded
	  hidden state, generates the token ids inside that patch.

Inference model (deducer): LilyletNotaGen
Loss model (training wrapper): LilyletNotaGenLoss

Batch contract (from LilyletPatchy.collateBatch):
	input_patches: LongTensor [B, T, patch_size]   token ids in [0, char_vocab_size)
	input_masks:   LongTensor [B, T]               1 for real patch, 0 for padding
'''

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, PreTrainedModel


PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


class PatchLevelDecoder (PreTrainedModel):
	'''Encodes patches into per-patch hidden states (auto-regressive over patches).'''

	config_class = GPT2Config

	def __init__ (self, config, patch_size, char_vocab_size):
		super().__init__(config)

		self.patch_size = patch_size
		self.char_vocab_size = char_vocab_size

		self.patch_embedding = nn.Linear(patch_size * char_vocab_size, config.n_embd)
		nn.init.normal_(self.patch_embedding.weight, std=0.02)

		self.base = GPT2Model(config)

	def forward (self, patches: torch.Tensor, masks: Optional[torch.Tensor] = None):
		# patches: [B, T, patch_size] -> one-hot [B, T, patch_size, char_vocab_size]
		patches = F.one_hot(patches.long(), num_classes=self.char_vocab_size).to(self.patch_embedding.weight.dtype)
		patches = patches.reshape(len(patches), -1, self.patch_size * self.char_vocab_size)
		patches = self.patch_embedding(patches)

		if masks is None:
			return self.base(inputs_embeds=patches)
		return self.base(inputs_embeds=patches, attention_mask=masks)


class CharLevelDecoder (PreTrainedModel):
	'''Generates the tokens within a patch, conditioned on the patch hidden state.'''

	config_class = GPT2Config

	def __init__ (self, config):
		super().__init__(config)

		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.base = GPT2LMHeadModel(config)

	def forward (self, encoded_patches: torch.Tensor, target_patches: torch.Tensor):
		# target_patches: [N, patch_size]; prepend BOS -> [N, patch_size + 1]
		target_patches = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id, target_patches), dim=1)

		# labels: mask padding positions with -100
		labels = target_patches.clone()
		labels = labels.masked_fill(labels == self.special_token_id, -100)

		# attention mask over chars: 1 where label is valid, 0 where -100
		target_masks = torch.ones_like(labels)
		target_masks = target_masks.masked_fill(labels == -100, 0)

		# char embeddings, replace first position with the encoded patch state
		inputs_embeds = F.embedding(target_patches, self.base.transformer.wte.weight)
		inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1)

		return self.base(inputs_embeds=inputs_embeds, attention_mask=target_masks, labels=labels)


class LilyletNotaGen (nn.Module):
	'''Inference model: hierarchical patch-level + char-level decoders.

	Args (from config['model.args']):
		char_vocab_size: token vocabulary size (Lilylet manual tokenizer = 256)
		patch_size: tokens per patch (16)
		patch_length: max patches per document (used for position embeddings)
		hidden_size: GPT2 embedding dim
		patch_num_layers: layers of the patch-level GPT2
		char_num_layers: layers of the char-level GPT2
		n_head: attention heads (default hidden_size // 64)
	'''

	def __init__ (self, char_vocab_size=256, patch_size=16, patch_length=2048,
		hidden_size=768, patch_num_layers=12, char_num_layers=3, n_head=None, **_):
		super().__init__()

		self.char_vocab_size = char_vocab_size
		self.patch_size = patch_size
		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.eos_token_id = EOS_TOKEN_ID

		n_head = n_head or max(1, hidden_size // 64)

		patch_config = GPT2Config(
			num_hidden_layers=patch_num_layers,
			max_length=patch_length,
			max_position_embeddings=patch_length,
			n_embd=hidden_size,
			num_attention_heads=n_head,
			vocab_size=1,
		)
		char_config = GPT2Config(
			num_hidden_layers=char_num_layers,
			max_length=patch_size + 1,
			max_position_embeddings=patch_size + 1,
			hidden_size=hidden_size,
			num_attention_heads=n_head,
			vocab_size=char_vocab_size,
		)

		self.patch_level_decoder = PatchLevelDecoder(patch_config, patch_size, char_vocab_size)
		self.char_level_decoder = CharLevelDecoder(char_config)

	def forward (self, patches: torch.Tensor, masks: torch.Tensor):
		'''
		patches: [B, T, patch_size] token ids
		masks:   [B, T] 1 for real patch, 0 for padding
		Returns (output, target_patches):
			output: char-level GPT2 output (has .loss and .logits) for next-patch prediction
			target_patches: [N, patch_size] the target tokens aligned to each prediction
		'''
		patches = patches.reshape(len(patches), -1, self.patch_size)
		encoded_patches = self.patch_level_decoder(patches, masks)['last_hidden_state']

		# Pair each encoded patch with the *next* patch's tokens (next-patch prediction):
		# - left_shift_masks selects encoded positions that have a valid following patch
		#   (i.e. drop the last real patch of each sequence).
		# - masks[:, 0] = 0 drops the first patch from the targets, so target i aligns to
		#   encoded position i-1.
		masks = masks.clone()
		left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
		masks[:, 0] = 0

		encoded_patches = encoded_patches[left_shift_masks == 1]
		target_patches = patches[masks == 1]

		return self.char_level_decoder(encoded_patches, target_patches), target_patches


class LilyletNotaGenLoss (nn.Module):
	'''Training wrapper: computes loss + metrics from a LilyletPatchy batch.'''

	def __init__ (self, **kw_args):
		super().__init__()

		self.deducer = LilyletNotaGen(**kw_args)

	def _char_accuracy (self, output, target_patches):
		# next-token accuracy over valid (non-pad) char positions, matching the
		# label shift the char-level GPT2 applies internally.
		char_targets = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.deducer.bos_token_id, target_patches), dim=1)
		labels = char_targets.masked_fill(char_targets == self.deducer.special_token_id, -100)
		shift_logits = output.logits[:, :-1, :]
		shift_labels = labels[:, 1:]
		valid = shift_labels != -100
		if not valid.any():
			return 0.0
		return (shift_logits.argmax(dim=-1)[valid] == shift_labels[valid]).float().mean().item()

	def forward (self, batch):
		output, target = self.deducer(batch['input_patches'], batch['input_masks'])

		with torch.no_grad():
			acc = self._char_accuracy(output, target)

		return output.loss, {'acc': acc}

	def inspectRun (self, batch):
		output, target = self.deducer(batch['input_patches'], batch['input_masks'])

		return {
			'loss': output.loss.item(),
			'acc': self._char_accuracy(output, target),
			'logits': output.logits,
			'target_patches': target,
			'n_patches': int(target.shape[0]),
		}
