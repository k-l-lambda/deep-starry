
''' This module will handle the text generation. '''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import Transformer, get_pad_mask, get_subsequent_mask



class Generator (nn.Module):
	''' Load a trained model and generate sentences by multinomial sampling. '''

	def __init__ (self, model, max_seq_len, trg_bos_idx, trg_eos_idx):
		super().__init__()

		self.max_seq_len = max_seq_len
		self.trg_bos_idx = trg_bos_idx
		self.trg_eos_idx = trg_eos_idx

		self.model = model
		self.model.eval()


	def _model_decode (self, seq):
		mask = get_subsequent_mask(seq)
		dec_output, *_ = self.model.decoder(seq, mask)
		vocab_vec = self.model.trg_word_prj(dec_output)[:, -1, :]
		return vocab_vec


	def generate_sentence (self, temperature = 1):
		seq = torch.full((1, self.max_seq_len), 1, dtype=torch.long)
		seq[0][0] = self.trg_bos_idx

		step = 1

		with torch.no_grad():
			while True:
				dec_output = self._model_decode(seq[:, :step])
				dec_output /= temperature
				dec_output = F.softmax(dec_output, dim=-1).flatten()
				word = torch.multinomial(dec_output, 1)

				if word == self.trg_eos_idx:
					break

				seq[0][step] = word[0]

				step += 1
				if step >= self.max_seq_len:
					break

		return seq[0][1:step]
