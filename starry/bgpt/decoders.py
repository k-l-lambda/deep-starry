'''
Generic bGPT-style hierarchical patch/token decoders.

This is the reusable two-level transformer architecture shared by NotaGen/bGPT-style
models (cf. NotaGen pretrain/utils.py: PatchLevelDecoder + CharLevelDecoder, and the
bGPT paper "Byte models are digital world simulators"). It is modality-agnostic — any
patchified sequence (Lilylet text, MIDI events, raw bytes) can use it.

Two levels:
	- PatchLevelDecoder: encodes a sequence of patches (each patch is a fixed-length
	  sequence of token ids) into per-patch hidden states with a GPT2/Llama backbone.
	- TokenLevelDecoder: an auto-regressive LM that, conditioned on a patch's encoded
	  hidden state, generates the token ids inside that patch.

Modality-specific deducers (e.g. starry.lilylet.models.notagen.LilyletNotaGen) compose
these two and define the config/batch contract.
'''

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Model, GPT2LMHeadModel, PreTrainedModel
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers import PretrainedConfig


PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


def token_embedding_weight (base):
	'''Return the input token-embedding weight of a HF base model, regardless of
	architecture: GPT2 stores it at `transformer.wte`, Llama at `model.embed_tokens`.'''
	if hasattr(base, 'transformer'):			# GPT2Model / GPT2LMHeadModel
		return base.transformer.wte.weight
	if hasattr(base, 'model'):				# LlamaForCausalLM
		return base.model.embed_tokens.weight
	if hasattr(base, 'embed_tokens'):			# LlamaModel
		return base.embed_tokens.weight
	raise AttributeError(f'cannot locate token embedding on {type(base).__name__}')


class PatchLevelDecoder (PreTrainedModel):
	'''Encodes patches into per-patch hidden states (auto-regressive over patches).'''

	config_class = PretrainedConfig

	def __init__ (self, config, patch_size, token_vocab_size):
		super().__init__(config)

		self.patch_size = patch_size
		self.token_vocab_size = token_vocab_size

		hidden = getattr(config, 'n_embd', None) or config.hidden_size
		self.patch_embedding = nn.Linear(patch_size * token_vocab_size, hidden)
		nn.init.normal_(self.patch_embedding.weight, std=0.02)

		self.base = LlamaModel(config) if isinstance(config, LlamaConfig) else GPT2Model(config)

	def forward (self, patches: torch.Tensor, masks: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.Tensor] = None):
		'''
		patches: LongTensor [B, T, patch_size]   token ids in [0, token_vocab_size)
		masks:   LongTensor [B, T] or None       1 for real patch, 0 for padding
		                                         (None = attend all positions)
		position_ids: LongTensor [B, T] or None  optional explicit patch positions. When
		                                         None, the HF base model uses its default
		                                         local 0..T-1 positions.
		Returns: the HF base model output; `.last_hidden_state` is [B, T, hidden].
		'''
		# patches: [B, T, patch_size] -> one-hot [B, T, patch_size, token_vocab_size]
		patches = F.one_hot(patches.long(), num_classes=self.token_vocab_size).to(self.patch_embedding.weight.dtype)
		# flatten each patch's one-hot into a single vector -> [B, T, patch_size * token_vocab_size]
		patches = patches.reshape(len(patches), -1, self.patch_size * self.token_vocab_size)
		# project to per-patch embeddings -> [B, T, hidden]
		patches = self.patch_embedding(patches)

		if masks is None:
			return self.base(inputs_embeds=patches, position_ids=position_ids)
		return self.base(inputs_embeds=patches, attention_mask=masks, position_ids=position_ids)


class TokenLevelDecoder (PreTrainedModel):
	'''Generates the tokens within a patch, conditioned on the patch hidden state.'''

	config_class = PretrainedConfig

	def __init__ (self, config):
		super().__init__(config)

		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.base = LlamaForCausalLM(config) if isinstance(config, LlamaConfig) else GPT2LMHeadModel(config)

	def forward (self, encoded_patches: torch.Tensor, target_patches: torch.Tensor):
		'''
		encoded_patches: FloatTensor [N, hidden]        per-patch hidden state (1 per target patch)
		target_patches:  LongTensor  [N, patch_size]    the token ids to teacher-force
		Returns: HF CausalLM output over [N, patch_size + 1] positions, with
			.loss   scalar cross-entropy (padding positions masked via -100 labels)
			.logits FloatTensor [N, patch_size + 1, token_vocab_size]
		'''
		# target_patches: [N, patch_size]; prepend BOS -> [N, patch_size + 1]
		target_patches = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id, target_patches), dim=1)

		# labels: mask padding positions with -100
		labels = target_patches.clone()
		labels = labels.masked_fill(labels == self.special_token_id, -100)

		# attention mask over tokens: 1 where label is valid, 0 where -100
		target_masks = torch.ones_like(labels)
		target_masks = target_masks.masked_fill(labels == -100, 0)

		# token embeddings, replace first position with the encoded patch state
		inputs_embeds = F.embedding(target_patches, token_embedding_weight(self.base))
		inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1)

		return self.base(inputs_embeds=inputs_embeds, attention_mask=target_masks, labels=labels)
