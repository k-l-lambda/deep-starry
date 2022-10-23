
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel
import itertools
import types

from ...utils.config import Configuration
from .modules import InvWordEmbed
from .overwriteCLIPTextTransformer import CLIPTextTransformer_forward



class ClipTextGenerator (nn.Module):
	def __init__(self, text_encoder_path, unembed_checkpoint_path=None, **_):
		super().__init__()

		self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, subfolder='text_encoder')
		self.text_encoder.text_model.forward = types.MethodType(CLIPTextTransformer_forward, self.text_encoder.text_model)

		self.unembed = InvWordEmbed(self.text_encoder.config.hidden_size, self.text_encoder.config.vocab_size)

		if unembed_checkpoint_path is not None:
			checkpoint = torch.load(unembed_checkpoint_path, map_location='cpu')
			self.unembed.load_state_dict(checkpoint['model'])


	def forward (self, ids, mask=None):
		x = self.text_encoder(ids, attention_mask=mask).last_hidden_state
		x = self.unembed(x)

		return x


class ClipTextGeneratorLoss (nn.Module):
	def __init__(self, unembed_path, **args):
		super().__init__()

		config_unembed = Configuration.createOrLoad(unembed_path)
		unembed_checkpoint_path = config_unembed.localPath(config_unembed['best']) if config_unembed['best'] else None

		self.deducer = ClipTextGenerator(unembed_checkpoint_path=unembed_checkpoint_path, **args)

		# freeze part modules
		for param in self.deducer.text_encoder.text_model.embeddings.token_embedding.parameters():
			param.requires_grad = False
		if unembed_checkpoint_path is not None:
			for param in self.deducer.unembed.parameters():
				param.requires_grad = False

		# randomize parameters
		for param in itertools.chain(self.deducer.text_encoder.text_model.encoder.parameters(), self.deducer.text_encoder.text_model.final_layer_norm.parameters()):
			if param.dim() > 1:
				nn.init.xavier_uniform_(param)


	def forward (self, batch):
		# mask to look backward only
		mask = batch['attention_mask']
		batch_size, n_seq = mask.shape
		mask = mask[:, None, :].expand(batch_size, n_seq, n_seq)
		triu = 1 - torch.triu(torch.ones(n_seq, n_seq), diagonal=1)
		mask = mask * triu[None, :, :]

		pred = self.deducer(batch['input_ids'], mask=mask)
		pred_ncs = pred.permute(0, 2, 1)
		target = batch['output_ids']

		#weight = batch['attention_mask'][:, None, :].expand(*pred_ncs.shape).float()
		#print('weight:', weight.shape)

		loss = F.cross_entropy(pred_ncs, target)

		pred_ids = torch.argmax(pred, dim=-1)
		acc = (pred_ids == target).float().mean()

		return loss, {'acc': acc.item()}
