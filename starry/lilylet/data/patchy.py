import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs


class LilyletPatchy(Dataset):
    @classmethod
    def load(cls, root, args, splits, device='cpu', args_variant=None, **_):
        splits = splits.split(':')

        def argi(i):
            if args_variant is None:
                return args
            return mergeArgs(args, args_variant.get(i))

        return tuple(
            cls(root, split, device=device, shuffle='*' in split, **argi(i))
            for i, split in enumerate(splits)
        )

    def __init__(self, root, split, device='cpu', shuffle=False, pad_id=0, **_):
        super().__init__()
        self.device = device
        self.shuffle = shuffle
        self.pad_id = pad_id
        self.artifact = torch.load(root, map_location='cpu')

        phases, cycle = parseFilterStr(split)
        self.indices = [
            i for i in range(len(self.artifact['items']))
            if i % cycle in phases
        ]

    def __len__(self):
        return len(self.indices)

    def _item(self, index):
        item = self.artifact['items'][index]
        patches = item['patches'].long()
        # The per-item mask is always all-ones; reconstruct it from the patch count.
        # Older artifacts may still carry a stored 'mask'; honor it if present.
        mask = item['mask'].long() if 'mask' in item else torch.ones(patches.shape[0], dtype=torch.long)
        return patches, mask

    def __getitem__(self, index):
        return self._item(self.indices[index])

    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle:
            order = torch.randperm(len(indices)).tolist()
            indices = [indices[i] for i in order]
        for index in indices:
            yield self._item(index)

    def collateBatch(self, batch):
        input_patches = [ex[0] for ex in batch]
        input_masks = [ex[1] for ex in batch]
        input_patches = pad_sequence(input_patches, batch_first=True, padding_value=self.pad_id)
        input_masks = pad_sequence(input_masks, batch_first=True, padding_value=0)
        return dict(
            input_patches=input_patches.to(self.device),
            input_masks=input_masks.to(self.device),
        )
