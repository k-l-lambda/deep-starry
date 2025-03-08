
import os
import torch
from torch.utils.data import IterableDataset
from fs import open_fs
from jinja2 import Environment, FileSystemLoader
import numpy as np
from transformers import AutoTokenizer

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


	def __init__ (self, root, split, device, shuffle, prompt_template, tokenizer, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle

		total_table = pd.read_csv(root + '.csv')

		phases, cycle = parseFilterStr(split)
		self.table = total_table.iloc[[i for i in range(len(total_table)) if i % cycle in phases]]

		self.vision_lib = open_fs(f'zip://{root}.zip')

		env = Environment(loader=FileSystemLoader('./assets'))
		self.prompt_template = env.get_template(prompt_template)

		self.tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(tokenizer['path']))


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
			prompt = self.prompt_template.render(seed=prompt_seed).strip()

			yield self.tokenizer.encode(prompt, return_tensors='pt'), self.tokenizer.encode(row['sentence'], return_tensors='pt'), img_emb


	def collateBatch (self, batch):
		prompt = torch.nn.utils.rnn.pad_sequence([ex[0].squeeze(0) for ex in batch], batch_first=True).to(self.device)
		sentence = torch.nn.utils.rnn.pad_sequence([ex[1].squeeze(0) for ex in batch], batch_first=True).to(self.device)
		img_emb = torch.stack([ex[2] for ex in batch], dim=0).to(self.device)

		return dict(prompt=prompt, sentence=sentence, img_emb=img_emb)
