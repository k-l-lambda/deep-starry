
import numpy as np
import torch



def collateBatch (batch, trans, device, by_numpy=False):
	images = list()
	labels = list()

	for b in batch:
		im,lb = b
		images.append(im)
		labels.append(lb)

	images = np.stack(images, axis=0)
	labels = np.stack(labels, axis=0)
	if trans != None:
		images, labels = trans(images, labels)

	if by_numpy:
		return images, labels

	labels = torch.from_numpy(labels).to(device)
	images = torch.from_numpy(images).to(device)


def loadSplittedDatasets (dataset_cls, root, args, splits, device='cpu'):
	splits = splits.split(':')
	return tuple(map(lambda split: dataset_cls(root, split=split, shuffle=split.startswith('*'), device=device, **args), splits))
