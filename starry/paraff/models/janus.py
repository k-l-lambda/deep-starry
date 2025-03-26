
import os
from typing import List
from transformers import LlamaForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

# workaround error: cannot import name 'Mapping' from 'collections'
import collections
import collections.abc
for type_name in collections.abc.__all__:
	setattr(collections, type_name, getattr(collections.abc, type_name))

from attrdict import AttrDict

from ...topology.models.modules import CrossEntropy
from ...janus import MlpProjector



class JanusLanguage (nn.Module):
	def __init__ (self, trainable_parameters, model_path, dtype='float32', additional_embedding_dims=None, **_):
		super().__init__()

		dtype = getattr(torch, dtype)

		self.janus = LlamaForCausalLM.from_pretrained(os.path.expanduser(model_path))
		self.janus.to(dtype)

		self.trainable_parameters = [
			key for key in self.janus.state_dict().keys()
			if any(key.startswith(p) for p in trainable_parameters)
		]

		self.additional_embedding_dims = additional_embedding_dims

		if self.additional_embedding_dims is not None:
			hidden_size = self.janus.config.hidden_size
			n_vocab = additional_embedding_dims[1] - additional_embedding_dims[0]
			self.add_embedding = nn.Parameter(torch.zeros(n_vocab, hidden_size, dtype=dtype))


	def embed_input (self, input_ids):
		emb = self.janus.get_input_embeddings()(input_ids)

		if self.additional_embedding_dims is not None:
			weights = torch.zeros((self.janus.config.vocab_size, emb.shape[-1]), dtype=self.add_embedding.dtype, device=emb.device)
			weights[self.additional_embedding_dims[0]:self.additional_embedding_dims[1]] = self.add_embedding

			emb += F.embedding(input_ids, weights)

		return emb


	def forward (self, input_ids, image_masks, image_embeddings, attention_mask):
		inputs_embeds = self.embed_input(input_ids)
		for mask, emb in zip(image_masks, image_embeddings):
			inputs_embeds[mask] = emb

		logits = self.janus(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

		return logits


	def state_dict (self):
		super_dict = self.janus.state_dict()

		if self.additional_embedding_dims is not None:
			states = dict(add_embedding=self.add_embedding)
		for key in self.trainable_parameters:
			states[key] = super_dict[key]

		return states


	def load_state_dict (self, state_dict, strict=True, assign: bool = False):
		if self.additional_embedding_dims is not None:
			self.add_embedding.data.copy_(state_dict.pop('add_embedding'))

		return self.janus.load_state_dict(state_dict, strict=False, assign=assign)


	def save_pretrained (self, path):
		if self.additional_embedding_dims is not None:
			weights = torch.zeros((self.janus.config.vocab_size, self.janus.config.hidden_size), dtype=self.add_embedding.dtype)
			weights[self.additional_embedding_dims[0]:self.additional_embedding_dims[1]] = self.add_embedding

			with torch.no_grad():
				self.janus.get_input_embeddings().weight += weights

		self.janus.save_pretrained(path)


class JanusLanguageLoss (nn.Module):
	def __init__ (self, aligner_cfg: dict, aligner_weights_path: str, trainable_parameters: List[str], dtype='float32', **kwargs):
		super().__init__()

		self.dtype = getattr(torch, dtype)

		self.deducer = JanusLanguage(**kwargs, trainable_parameters=trainable_parameters, dtype=dtype)

		self.aligner = MlpProjector(AttrDict(aligner_cfg))

		aligner_weights = torch.load(os.path.expanduser(aligner_weights_path), weights_only=True)
		self.aligner.load_state_dict(aligner_weights)
		self.aligner.to(self.dtype)

		for name, param in self.deducer.janus.named_parameters():
			if not any(name.startswith(p) for p in trainable_parameters):
				param.requires_grad = False

		self.ce = CrossEntropy()


	def forward (self, batch):
		input_ids = batch['input_ids']
		target_ids = torch.roll(input_ids, shifts=-1, dims=1)

		image_seq_mask = batch['image_seq_mask']

		image_embedding = self.aligner(batch['img_emb'].to(self.dtype))
		image_embedding = image_embedding.reshape((-1, image_embedding.shape[-1]))

		attention_mask = batch['attention_mask']
		target_mask = batch['target_mask']

		logits = self.deducer(input_ids=input_ids, image_masks=[image_seq_mask], image_embeddings=[image_embedding], attention_mask=attention_mask)

		loss = self.ce(logits, target_ids, mask=target_mask)

		pred_ids = torch.argmax(logits, dim=-1)
		acc = (pred_ids == target_ids).float().mean()

		return loss, dict(loss=loss.item(), acc=acc.item())


	@torch.inference_mode()
	def inspectRun (self, batch):
		input_ids = batch['input_ids']
		target_ids = torch.roll(input_ids, shifts=-1, dims=1)

		image_seq_mask = batch['image_seq_mask']

		image_embedding = self.aligner(batch['img_emb'].to(self.dtype))
		image_embedding = image_embedding.reshape((-1, image_embedding.shape[-1]))

		attention_mask = batch['attention_mask']
		target_mask = batch['target_mask']

		logits = self.deducer(input_ids=input_ids, image_masks=[image_seq_mask], image_embeddings=[image_embedding], attention_mask=attention_mask)

		pred_ids = torch.argmax(logits, dim=-1)

		target_flat = target_ids[target_mask]
		pred_flat = logits[target_mask]

		truth = (pred_ids == target_ids)[target_mask]

		return dict(
			target_flat=target_flat,
			pred_flat=pred_flat,
			truth=truth,
		)
