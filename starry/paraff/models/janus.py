
from typing import List
from transformers import LlamaForCausalLM
import torch
import torch.nn as nn

from ...janus import MlpProjector



class JanusLanguage (LlamaForCausalLM):
	pass
	#def __init__ (self):
	#	super().__init__()


class JanusLanguageLoss (nn.Module):
	def __init__ (self, model_path: str, aligner_cfg, trainable_parameters: List[str]):
		super().__init__()

		self.deducer = JanusLanguage.from_pretrained(model_path)

		self.aligner = MlpProjector(aligner_cfg)

		for name, param in self.deducer.named_parameters():
			if name not in trainable_parameters:
				param.requires_grad = False


	def forward (self, batch):
		inputs_embeds = self.deducer.get_input_embeddings()(batch['input_ids'])
		image_seq_mask = batch['image_seq_mask']

		image_embedding = self.aligner(batch['img_emb'])
		inputs_embeds[image_seq_mask] = image_embedding

		# TODO: call deducer
