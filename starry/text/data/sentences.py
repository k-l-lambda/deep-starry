
import torch
from torch.utils.data import IterableDataset
from transformers import CLIPTokenizer

from ...utils.parsers import parseFilterStr, mergeArgs



tokenizer = None


class SentenceShift (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None):
		splits = splits.split(':')

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		tokenizer = CLIPTokenizer.from_pretrained(args['tokenizer_path'], subfolder='tokenizer')

		return (
			cls(root, tokenizer, split, device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)


	def __init__ (self, root, tokenizer, split, device, shuffle, **_):
		super().__init__()

		self.tokenizer = tokenizer
		self.device = device
		self.shuffle = shuffle

		phases, cycle = parseFilterStr(split)

		text = open(root, 'r').read()
		sentences = text.split('\n')
		sentences = [s for i, s in enumerate(sentences) if i % cycle in phases]

		self.entries = [tokenizer(sentence,
			padding="max_length",
			max_length=tokenizer.model_max_length,
			return_tensors="pt",
		) for sentence in sentences]

		for entry in self.entries:
			entry['output_ids'] = torch.zeros_like(entry['input_ids'])
			entry['output_ids'][:, :-1] = entry['input_ids'][:, 1:]
			entry['output_ids'][:, -1] = self.tokenizer.vocab_size - 1
