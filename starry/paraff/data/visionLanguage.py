
import os
import torch
from torch.utils.data import IterableDataset
from fs import open_fs
from jinja2 import Environment, FileSystemLoader
import numpy as np
from transformers import AutoTokenizer
from typing import List

from ...utils.parsers import parseFilterStr, mergeArgs
import pandas as pd



class VisionLanguage (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		splits = splits.split(':')

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return (
			cls(root, split, device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)


	def __init__ (self, root, split, device, shuffle, prompt_template, sft_template, tokenizer, tags, num_image_tokens=576, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle
		self.tags = tags

		total_table = pd.read_csv(root + '.csv')

		phases, cycle = parseFilterStr(split)
		self.table = total_table.iloc[[i for i in range(len(total_table)) if i % cycle in phases]]

		self.vision_lib = open_fs(f'zip://{root}.zip')

		env = Environment(loader=FileSystemLoader('./assets'))
		self.prompt_template = env.get_template(prompt_template)
		self.sft_template = env.get_template(sft_template)

		self.tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(tokenizer['path']))
		self.image_id = self.tokenizer.vocab.get(tags['image'])
		self.image_start_id = self.tokenizer.vocab.get(tags['image_start'])
		self.image_end_id = self.tokenizer.vocab.get(tags['image_end'])
		self.pad_id = self.tokenizer.vocab.get(tags['pad'])

		self.num_image_tokens = num_image_tokens


	def __len__ (self):
		return len(self.table)


	def __iter__ (self):
		if self.shuffle:
			self.table = self.table.sample(frac=1).reset_index(drop=True)
		else:
			torch.manual_seed(0)
			np.random.seed(1)

		for _, row in self.table.iterrows():
			index = row['index']
			img_emb = torch.load(self.vision_lib.openbin(f'{index}.pt'), weights_only=True)

			prompt_seed = np.random.randint(0, 0x7fffffff) if self.shuffle else index
			question = self.prompt_template.render(seed=prompt_seed).strip()
			prompt = self.sft_template.render(question=question)

			prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)[0]
			image_token_mask = prompt_ids == self.image_id
			image_indices = image_token_mask.nonzero()
			prompt_ids, num_image_tokens = self.add_image_token(
				image_indices=image_indices,
				input_ids=prompt_ids,
			)

			yield prompt_ids, self.tokenizer.encode(row['sentence'] + self.tags['eos'], return_tensors='pt', add_special_tokens=False), img_emb


	def collateBatch (self, batch):
		prompt = torch.nn.utils.rnn.pad_sequence([ex[0].squeeze(0) for ex in batch], batch_first=True, padding_value=self.pad_id).to(self.device)
		sentence = torch.nn.utils.rnn.pad_sequence([ex[1].squeeze(0) for ex in batch], batch_first=True, padding_value=self.pad_id).to(self.device)
		img_emb = torch.stack([ex[2] for ex in batch], dim=0).to(self.device)

		return dict(prompt=prompt, sentence=sentence, img_emb=img_emb)


	def add_image_token(
		self,
		image_indices: List[int],
		input_ids: torch.LongTensor,
	):
		input_slices = []

		start = 0
		for index in image_indices:
			end = index

			# original text tokens
			input_slices.append(input_ids[start:end])

			# add boi, image tokens, eoi and set the mask as False
			input_slices.append(self.image_start_id * torch.ones((1), dtype=torch.long))
			input_slices.append(
				self.image_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
			)
			input_slices.append(self.image_end_id * torch.ones((1), dtype=torch.long))
			start = index + 1

		# the left part
		input_slices.append(input_ids[start:])

		# concat all slices
		input_ids = torch.cat(input_slices, dim=0)
		num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))

		return input_ids, num_image_tokens
