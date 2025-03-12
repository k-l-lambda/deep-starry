
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

from ...janus import MlpProjector



class JanusLanguage (LlamaForCausalLM):
	pass
	#def __init__ (self):
	#	super().__init__()


class JanusLanguageLoss (nn.Module):
	def __init__ (self, model_path: str, aligner_cfg: dict, aligner_weights_path: str, trainable_parameters: List[str]):
		super().__init__()

		self.deducer = JanusLanguage.from_pretrained(os.path.expanduser(model_path))
		self.dtype = self.deducer.dtype

		self.aligner = MlpProjector(AttrDict(aligner_cfg))

		aligner_weights = torch.load(os.path.expanduser(aligner_weights_path))
		self.aligner.load_state_dict(aligner_weights)

		for name, param in self.deducer.named_parameters():
			if not any(name.startswith(p) for p in trainable_parameters):
				param.requires_grad = False


	def forward (self, batch):
		inputs_embeds = self.deducer.get_input_embeddings()(batch['input_ids'])
		image_seq_mask = batch['image_seq_mask']

		image_embedding = self.aligner(batch['img_emb'].to(self.dtype))
		image_embedding = image_embedding.reshape((-1, image_embedding.shape[-1]))
		inputs_embeds[image_seq_mask] = image_embedding

		attention_mask = batch['attention_mask']
		logits = self.deducer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

		# TODO:
		return logits, {}
