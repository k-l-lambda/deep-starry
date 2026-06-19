'''CLaMP 3 M3 (symbolic) patch encoder — model definition.

A trimmed, self-contained port of the M3 music encoder from CLaMP 2 / CLaMP 3
(sander-wood): the BERT-style `M3PatchEncoder` plus the config builder and a
loader that pulls the symbolic tower out of the cached `sander-wood/clamp3`
checkpoint. The token decoder and the text/audio towers are intentionally
omitted.

The combined CLaMP 3 state-dict stores the M3 tower under the `symbolic_model.*`
prefix, identical in structure to `M3PatchEncoder`, so `load_m3_encoder` strips
the prefix and loads it strictly into a fresh encoder.

Reference: NotaGen/clamp2/utils.py (M3PatchEncoder) and
NotaGen/clamp2/extract_clamp2.py:47-52 (the BertConfig).
'''

import torch
from transformers import BertConfig, BertModel, PreTrainedModel


# --- M3 hyper-parameters (CLaMP 2/3 symbolic tower) ---
PATCH_SIZE = 64        # characters (token ids) per patch
PATCH_LENGTH = 512     # max patches per segment
M3_HIDDEN_SIZE = 768   # encoder hidden size
PATCH_NUM_LAYERS = 12  # BERT encoder layers
NUM_CLASSES = 128      # character vocabulary size (one-hot classes per token id)

# special token ids (shared across patchilizer + encoder)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MASK_TOKEN_ID = 3

# default weights: the symbolic tower of CLaMP 3's main 3-modal (saas) checkpoint
_CLAMP3_REPO_ID = 'sander-wood/clamp3'
_CLAMP3_SAAS_FILENAME = (
	'weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base'
	'_t_length_128_a_size_768_a_layers_12_a_length_128'
	'_s_size_768_s_layers_12_p_size_64_p_length_512.pth'
)
_SYMBOLIC_PREFIX = 'symbolic_model.'


class M3PatchEncoder (PreTrainedModel):
	'''BERT-style patch encoder. One-hots each patch (128 classes), flattens to
	PATCH_SIZE*128, projects to hidden, then runs a BertModel over the patch
	sequence. Ported from NotaGen/clamp2/utils.py; the only change is the input
	dtype/device cast (upstream forced CPU float, which breaks GPU inference).
	'''

	def __init__ (self, config, num_classes=NUM_CLASSES):
		super().__init__(config)
		self.num_classes = num_classes
		self.patch_embedding = torch.nn.Linear(PATCH_SIZE * num_classes, M3_HIDDEN_SIZE)
		torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
		self.base = BertModel(config=config)
		self.pad_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.eos_token_id = EOS_TOKEN_ID
		self.mask_token_id = MASK_TOKEN_ID

	def forward (self, input_patches, input_masks):
		# input_patches: [batch, seq_length, PATCH_SIZE]; input_masks: [batch, seq_length]
		input_patches = torch.nn.functional.one_hot(input_patches.long(), num_classes=self.num_classes)
		input_patches = input_patches.reshape(len(input_patches), -1, PATCH_SIZE * self.num_classes)
		# cast to the embedding's dtype/device (upstream hard-coded torch.FloatTensor = CPU)
		weight = self.patch_embedding.weight
		input_patches = self.patch_embedding(input_patches.to(device=weight.device, dtype=weight.dtype))

		return self.base(inputs_embeds=input_patches, attention_mask=input_masks)


def build_m3_config ():
	'''BertConfig matching the CLaMP 2/3 symbolic tower (extract_clamp2.py:47-52).'''
	return BertConfig(
		vocab_size=1,
		hidden_size=M3_HIDDEN_SIZE,
		num_hidden_layers=PATCH_NUM_LAYERS,
		num_attention_heads=M3_HIDDEN_SIZE // 64,
		intermediate_size=M3_HIDDEN_SIZE * 4,
		max_position_embeddings=PATCH_LENGTH,
	)


def _resolve_weights_path (weights_path=None):
	'''Resolve the M3 weight file. An explicit local path wins; otherwise fetch
	the default saas checkpoint via the HF hub API (`hf_hub_download` returns the
	cached file path, downloading only if absent).'''
	if weights_path:
		return weights_path
	from huggingface_hub import hf_hub_download
	return hf_hub_download(repo_id=_CLAMP3_REPO_ID, filename=_CLAMP3_SAAS_FILENAME)


def load_m3_encoder (weights_path=None, device=None, num_classes=NUM_CLASSES):
	'''Build an M3PatchEncoder and load CLaMP 3's symbolic tower into it.

	The combined CLaMP 3 state-dict stores the M3 tower under `symbolic_model.*`;
	we strip that prefix and load strictly into a fresh encoder. `num_classes`
	must match the checkpoint's `patch_embedding` (128 for the released weights).
	'''
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'

	path = _resolve_weights_path(weights_path)
	checkpoint = torch.load(path, map_location='cpu', weights_only=True)
	state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
	sub = {k[len(_SYMBOLIC_PREFIX):]: v for k, v in state.items() if k.startswith(_SYMBOLIC_PREFIX)}
	if not sub:
		raise KeyError('no %s* keys found in %s' % (_SYMBOLIC_PREFIX, path))

	encoder = M3PatchEncoder(build_m3_config(), num_classes=num_classes)
	encoder.load_state_dict(sub, strict=True)
	encoder = encoder.to(device).eval()
	encoder._m3_weights_path = path
	return encoder
