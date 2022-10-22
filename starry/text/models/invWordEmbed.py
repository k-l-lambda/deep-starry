
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPTextModel

from .modules import InvWordEmbed



class InvWordEmbedLoss (nn.Module):
	def __init__ (self, tokenizer_pretrained_path):
		super().__init__()

		text_encoder = CLIPTextModel.from_pretrained(tokenizer_pretrained_path, subfolder='text_encoder')
		self.embed = text_encoder.text_model.embeddings.token_embedding

		self.vocab_size = text_encoder.config.vocab_size

		self.unembed = InvWordEmbed(text_encoder.config.hidden_size, text_encoder.config.vocab_size)

		self.deducer = self.unembed


	def forward (self, id):
		#onehot0 = F.one_hot(id, num_classes=self.vocab_size)
		embedding = self.embed(id)
		onehot1 = self.unembed(embedding)

		loss = F.cross_entropy(onehot1, id)

		id1 = torch.argmax(onehot1, dim=-1)
		acc = (id1 == id).float().mean()

		return loss, {'acc': acc}
