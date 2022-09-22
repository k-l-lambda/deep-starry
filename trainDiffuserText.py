
import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from starry.utils.config import Configuration
from starry.vision.data import PerisCaption



logger = get_logger(__name__)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	tokenizer = CLIPTokenizer.from_pretrained(config['trainer.pretrained_model_name_or_path'], subfolder='tokenizer')

	root = os.path.join(DATA_DIR, config['data.root'])
	labels = os.path.join(DATA_DIR, config['data.labels'])
	train_dataset = PerisCaption(root, labels, tokenizer, shuffle=True, **config['data.args'])

	it = iter(train_dataset)
	print(next(it))


if __name__ == "__main__":
	main()
