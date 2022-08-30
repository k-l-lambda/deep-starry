
import os
import re
from copy import deepcopy
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


def deep_update (d, u):
	for k, v in u.items():
		if isinstance(v, dict):
			d[k] = deep_update(d.get(k, {}), v)
		else:
			d[k] = v
	return d


def loadSplittedDatasets (dataset_cls, root, args, args_variant, splits, labels=None, device='cpu'):
	splits = splits.split(':')

	def load (isplit):
		this_args = args
		i, split = isplit
		if args_variant:
			argv = args_variant.get(i)
			if argv:
				this_args = deep_update(deepcopy(args), argv)

		return dataset_cls(root, labels=labels, split=split, shuffle=split.startswith('*'), device=device, **this_args)

	return tuple(map(load, enumerate(splits)))


def parseFilterStr (filterStr):
	filterStr = filterStr[1:] if filterStr.startswith('*') else filterStr
	phases, cycle = filterStr.split('/')
	captures = re.match(r'(\d+)\.\.(\d+)', phases)
	if captures:
		phases = list(range(int(captures[1]), int(captures[2]) + 1))
	else:
		phases = list(map(int, phases.split(',')))
	cycle = int(cycle)

	return phases, cycle


def listAllImageNames (reader, filterStr, dir='/'):
	# split file name & ext name
	all_names = [os.path.splitext(name)[0] for name in reader.listFiles(dir)]

	if filterStr is None:
		return all_names

	phases, cycle = parseFilterStr(filterStr)

	return [name for i, name in enumerate(all_names) if (i % cycle) in phases]
