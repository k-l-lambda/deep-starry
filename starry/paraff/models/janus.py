
import os
from typing import List
from transformers import LlamaForCausalLM
import torch
import torch.nn as nn

# workaround error: cannot import name 'Mapping' from 'collections'
import collections
import collections.abc
for type_name in collections.abc.__all__:
	setattr(collections, type_name, getattr(collections.abc, type_name))

from attrdict import AttrDict

from ...topology.models.modules import CrossEntropy
from ...janus import MlpProjector



class JanusLanguage (LlamaForCausalLM):
	pass
	#def __init__ (self):
	#	super().__init__()


class JanusLanguageLoss (nn.Module):
	def __init__ (self, model_path: str, aligner_cfg: dict, aligner_weights_path: str, trainable_parameters: List[str], dtype='float32'):
		super().__init__()

		self.dtype = getattr(torch, dtype)

		self.deducer = JanusLanguage.from_pretrained(os.path.expanduser(model_path))
		self.deducer.to(self.dtype)

		self.aligner = MlpProjector(AttrDict(aligner_cfg))

		aligner_weights = torch.load(os.path.expanduser(aligner_weights_path), weights_only=True)
		self.aligner.load_state_dict(aligner_weights)
		self.aligner.to(self.dtype)

		for name, param in self.deducer.named_parameters():
			if not any(name.startswith(p) for p in trainable_parameters):
				param.requires_grad = False

		self.ce = CrossEntropy()


	def forward (self, batch):
		input_ids = batch['input_ids']
		target_ids = torch.roll(input_ids, shifts=-1, dims=1)

		inputs_embeds = self.deducer.get_input_embeddings()(input_ids)
		image_seq_mask = batch['image_seq_mask']

		image_embedding = self.aligner(batch['img_emb'].to(self.dtype))
		image_embedding = image_embedding.reshape((-1, image_embedding.shape[-1]))
		inputs_embeds[image_seq_mask] = image_embedding

		attention_mask = batch['attention_mask']
		target_mask = batch['target_mask']

		logits = self.deducer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

		loss = self.ce(logits, target_ids, mask=target_mask)

		pred_ids = torch.argmax(logits, dim=-1)
		acc = (pred_ids == target_ids).float().mean()

		return loss, dict(loss=loss, acc=acc)


	@torch.inference_mode()
	def inspectRun (self, batch):
		input_ids = batch['input_ids']
		target_ids = torch.roll(input_ids, shifts=-1, dims=1)

		inputs_embeds = self.deducer.get_input_embeddings()(input_ids)
		image_seq_mask = batch['image_seq_mask']

		image_embedding = self.aligner(batch['img_emb'].to(self.dtype))
		image_embedding = image_embedding.reshape((-1, image_embedding.shape[-1]))
		inputs_embeds[image_seq_mask] = image_embedding

		attention_mask = batch['attention_mask']
		target_mask = batch['target_mask']

		logits = self.deducer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

		pred_ids = torch.argmax(logits, dim=-1)

		target_flat = target_ids[target_mask]
		pred_flat = logits[target_mask]

		truth = (pred_ids == target_ids)[target_mask]

		return dict(
			target_flat=target_flat,
			pred_flat=pred_flat,
			truth=truth,
		)
