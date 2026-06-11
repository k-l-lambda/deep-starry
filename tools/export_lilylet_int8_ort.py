"""Export INT8 ONNX-Runtime weights for a given LilyletNotaGen checkpoint.

Produces only the int8-quantized weights (fp32 is a throwaway intermediate that
quantize_dynamic needs, then is deleted):
  - patch_int8.onnx    : PatchNet   (patch ids -> patch hidden states, full recompute)
  - token_int8.onnx    : TokenNet   (inputs_embeds -> logits, the token-level decoder)
  - patch_kv_int8.onnx : PatchNetKV (patch ids + past KV -> hidden + present KV, for
                         incremental patch-level decoding with a KV cache)

Architecture is read from the run's own .state.yaml so it always matches the
weights. Uses the PatchNet/TokenNet transformer wrappers from
starry.lilylet.models.notagen (TokenNet wraps model.token_level_decoder).

Usage:
  python tools/export_lilylet_int8_ort.py \
    --run /home/camus/data/models/deep-starry-logs/lilylet/20260611-lilylet-notagenx-1m0611-llama
"""

import os
import sys
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch

from starry.utils.config import Configuration
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator
from starry.lilylet.models.notagen import PatchNet, TokenNet, PatchNetKV


def _rm (path):
	for p in (path, path + '.data'):
		if os.path.exists(p):
			os.remove(p)


def export_int8 (gen, hidden, out_dir):
	'''Export PatchNet/TokenNet to fp32 ONNX, quantize to int8, drop the fp32 temps.'''
	from onnxruntime.quantization import quantize_dynamic, QuantType

	os.makedirs(out_dir, exist_ok=True)
	model = gen.model
	patch_net = PatchNet(model).eval()
	token_net = TokenNet(model).eval()

	dummy_patches = torch.randint(0, 256, (1, 4, gen.patch_size), dtype=torch.long)
	dummy_embed = torch.randn(1, 5, hidden)

	specs = {
		'patch': (patch_net, (dummy_patches,), ['patches'], ['hidden'],
			{'patches': {1: 'T'}, 'hidden': {1: 'T'}}),
		'token': (token_net, (dummy_embed,), ['inputs_embeds'], ['logits'],
			{'inputs_embeds': {1: 'L'}, 'logits': {1: 'L'}}),
	}

	results = {}
	for name, (net, dummy, inames, onames, dyn) in specs.items():
		fp32 = os.path.join(out_dir, f'{name}_fp32_tmp.onnx')
		int8 = os.path.join(out_dir, f'{name}_int8.onnx')
		with torch.no_grad():
			torch.onnx.export(net, dummy, fp32, input_names=inames, output_names=onames,
				dynamic_axes=dyn, opset_version=17)
		quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
		_rm(fp32)
		results[f'{name}_int8'] = int8
	return results


def export_patch_kv_int8 (gen, hidden, out_dir):
	'''Export the KV-cache patch decoder (PatchNetKV) to int8 ONNX.

	Inputs:  patches [1,L,patch_size], past_k_{i}/past_v_{i} [1,NKV,P,HD] (i<num_layers)
	Outputs: hidden [1,L,hidden], new_k_{i}/new_v_{i} [1,NKV,P+L,HD]

	Uses dynamo export with dynamic_shapes (the dynamic_axes dict path fails under
	dynamo for the variadic past list), then quantize_dynamic to int8.
	'''
	from onnxruntime.quantization import quantize_dynamic, QuantType
	from torch.export import Dim

	os.makedirs(out_dir, exist_ok=True)
	net = PatchNetKV(gen.model).eval()
	base = gen.model.patch_level_decoder.base
	NL = base.config.num_hidden_layers
	NKV = base.config.num_key_value_heads
	HD = base.config.hidden_size // base.config.num_attention_heads

	# dummy: prefill-like inputs (L=2 new patches over a P=3 cache)
	dummy_patches = torch.randint(0, 256, (1, 2, gen.patch_size), dtype=torch.long)
	dummy_past = [torch.randn(1, NKV, 3, HD) for _ in range(2 * NL)]

	L = Dim('L', min=1, max=4096)
	P = Dim('P', min=1, max=4096)
	dynamic_shapes = ({1: L}, [{2: P} for _ in range(2 * NL)])

	in_names = ['patches'] + sum([[f'past_k_{i}', f'past_v_{i}'] for i in range(NL)], [])
	out_names = ['hidden'] + sum([[f'new_k_{i}', f'new_v_{i}'] for i in range(NL)], [])

	fp32 = os.path.join(out_dir, 'patch_kv_fp32_tmp.onnx')
	int8 = os.path.join(out_dir, 'patch_kv_int8.onnx')
	with torch.no_grad():
		torch.onnx.export(net, (dummy_patches, dummy_past), fp32,
			input_names=in_names, output_names=out_names,
			dynamic_shapes=dynamic_shapes, opset_version=18, dynamo=True)
	quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
	_rm(fp32)
	return {'patch_kv_int8': int8}


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--run', required=True, help='training run dir (holds best.chkpt + .state.yaml)')
	ap.add_argument('--checkpoint', default=None, help='checkpoint path (default: <run>/best.chkpt)')
	ap.add_argument('--output-dir', default=None, help='onnx output dir (default: <run>/onnx)')
	ap.add_argument('--threads', type=int, default=14)
	args = ap.parse_args()

	torch.set_num_threads(args.threads)

	run = args.run
	ckpt = args.checkpoint or os.path.join(run, 'best.chkpt')
	out_dir = args.output_dir or os.path.join(run, 'onnx')

	config = Configuration.createOrLoad(run, volatile=True)
	tk = config['data.args.tokenizer_path']
	tk = tk if os.path.isabs(tk) else os.path.join(REPO_ROOT, tk)
	hidden = config['model.args.hidden_size']
	print('run:', run)
	print('checkpoint:', ckpt)
	print('output dir:', out_dir)
	print('base_type:', config['model.args.base_type'], '| hidden:', hidden, '| building on CPU...')

	gen = LilyletPatchyGenerator.from_config(config, ckpt, tokenizer_path=tk, device='cpu')
	n_params = sum(p.numel() for p in gen.model.parameters())
	print('params: %.2fM | patch_size: %d' % (n_params / 1e6, gen.patch_size))

	paths = export_int8(gen, hidden, out_dir)
	paths.update(export_patch_kv_int8(gen, hidden, out_dir))

	print('\n=== exported int8 artifacts ===')
	for k, p in paths.items():
		print('  %-11s %8.1f MB  %s' % (k, os.path.getsize(p) / 1e6, p))
	print('\nint8 onnx ready in', out_dir)


if __name__ == '__main__':
	main()
