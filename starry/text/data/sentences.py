
import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import CLIPTokenizer

from ...utils.parsers import parseFilterStr, mergeArgs



tokenizer = None


class SentenceShift (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		splits = splits.split(':')

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		global tokenizer
		if tokenizer is None:
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

		sentence_length = tokenizer.model_max_length

		entries = [tokenizer(sentence,
			padding="max_length",
			max_length=sentence_length,
			return_tensors="pt",
		) for sentence in sentences]

		self.entries = [entry for entry in entries if entry['input_ids'].shape[1] == sentence_length]

		# clip long sentences
		long_entries = [entry for entry in entries if entry['input_ids'].shape[1] > sentence_length]
		SECTION_SIZE = sentence_length // 2
		for entry in long_entries:
			ids = entry['input_ids']
			entry_len = ids.shape[1]
			for i in range(0, entry_len, SECTION_SIZE):
				input_ids, mask = torch.zeros((1, sentence_length), dtype=torch.long), torch.zeros((1, sentence_length), dtype=torch.long)
				input_ids[:, :] = self.tokenizer.vocab_size - 1
				input_ids[:, :min(entry_len - i, sentence_length)] = ids[:, i:i + sentence_length]
				mask[:, :min(entry_len - i, sentence_length)] = entry['attention_mask'][:, i:i + sentence_length]

				self.entries.append({
					'input_ids': input_ids,
					'attention_mask': mask,
				})

		for entry in self.entries:
			entry['output_ids'] = torch.zeros_like(entry['input_ids'])
			entry['output_ids'][:, :-1] = entry['input_ids'][:, 1:]
			entry['output_ids'][:, -1] = self.tokenizer.vocab_size - 1


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.entries)

		for entry in self.entries:
			yield entry


	def __len__ (self):
		return len(self.entries)


	def collateBatch (self, batch):
		keys = [*batch[0].keys()]

		return {
			key: torch.cat([entry[key] for entry in batch], dim=0).to(self.device)
			for key in keys
		}
