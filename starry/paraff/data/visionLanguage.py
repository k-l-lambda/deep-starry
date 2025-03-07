
import torch
from torch.utils.data import IterableDataset
from fs import open_fs
from jinja2 import Environment, FileSystemLoader
import numpy as np
from transformers import AutoModelForCausalLM

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

		#self.tokenizer = AutoModelForCausalLM.from_pretrained(tokenizer['path'], trust_remote_code=True)#vocab_size=tokenizer['vocab_size'], max_position_embeddings=tokenizer['max_position_embeddings'])


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
			img_emb = torch.load(self.vision_lib.openbin(f'{index}.pt'))

			prompt_seed = np.random.randint(0, 0x7fffffff) if self.shuffle else index
			prompt = self.prompt_template.render(seed=prompt_seed).strip()

			# TODO: tokenize text
			#yield self.tokenizer.encode(prompt), self.tokenizer.encode(row['sentence']), img_emb
			yield prompt, row['sentence'], img_emb


	def collateBatch (self, batch):
		sentence = [ex for ex in batch]

		return dict(sentence=sentence)
