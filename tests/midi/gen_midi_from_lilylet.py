"""Lilylet -> MIDI translation via MidiBgptTransSelfAttn.

Conditions on a lilylet score (a READ-ONLY prefix, encoded once by the frozen lyl encoder)
and autoregressively generates MIDI event patches under the SAME measure-coupled
windowed/block joint attention mask the CondMidiPatchy feeder builds (build_vis). Each
generated midi patch is tagged with its own / lilylet-aligned measure (identity alignment:
midi measure i <- lilylet measure i, matching the no-repeat test corpus where
source_measure == index); an <eom> patch closes the current measure and advances it, an
<eos> patch stops generation.

Full-recompute decode (NO KV cache): the joint self-attention mask is windowed, not plain
causal, and the decoder is O(T^2), so this is meant for SHORT scores. (See
tools/midi/midiBgptOrtInfer.py for the KV-cache UNCONDITIONAL midi generator.)

KNOWN LIMITATION (checkpoint 20260702-...-selfattn-nota0701, verified by a teacher-forced
probe): the token-level decoder never learned to terminate an event patch. condPatchifier
pads midi event patches with the lilylet `pad_patch` (content -> <pad>, no in-patch <eos>),
but TokenLevelDecoder masks <pad> (== PAD_TOKEN_ID) to -100 in the loss, so the content->pad
terminator position is UNSUPERVISED. At a true event end the model gives <pad> rank ~34/41
(prob ~0) and instead emits a high-confidence content token, so free-running decode overruns
every event and degenerates. Event CONTENT is learned (leading tokens reproduce the ground
truth); only event LENGTH is not. Fix is in the DATA pipeline: use MidiTokenizer.pad_patch
(inserts a supervised <eos>, id 2, as the unconditional midi path does) for midi event
patches in condPatchifier, then retrain. This script is correct and will work once that lands.

Usage:
  python tests/midi/gen_midi_from_lilylet.py            # 3 shortest .lyl from the src dir
  python tests/midi/gen_midi_from_lilylet.py --files a.lyl b.lyl --device cuda
"""

import argparse
import glob
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModel
from starry.lilylet.data.patchifier import LilyletTokenizer
from starry.midi.tokenizer import MidiTokenizer
from starry.midi.data.condPatchifier import patchify_lilylet, PATCH_SIZE
from starry.midi.data.condPatchy import build_vis
from starry.lilylet.patchyGenerator import sample_next
from starry.bgpt import token_embedding_weight


RUN = os.path.expanduser('~/data/models/deep-starry-logs/midi/20260702-midi-measurewise-trans-selfattn-nota0701')
SRC_DIR = os.path.expanduser('~/work/lilylet/tests/output/notagenx-from-abc')
LYL_TOKENIZER = os.path.join(REPO_ROOT, 'assets', 'lilylet-tokenizer.json')
# Deterministic midi HEADER prompt (measure 0). In the corpus these two HEADER_EVENT lines
# are constant across all 215 songs; everything else (set_tempo / key_signature /
# control_change ...) is a FIELD event at tick 0 and thus a measure-1 event the MODEL emits.
DEFAULT_HEADER = ['ticks_per_beat 1e0', 'format_type 1']


def load_model (run, checkpoint, device):
	'''Build MidiBgptTransSelfAttn from the run's .state.yaml and load weights (eval).

	lyl_encoder_weights points at a remote training path that need not exist here; the
	encoder tensors come from THIS checkpoint anyway, so blank it to skip the load attempt.
	'''
	config = Configuration.createOrLoad(run, volatile=True)
	config['model.args.lyl_encoder_weights'] = None
	model = loadModel(config['model'], imports=config['imports'])
	blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
	state = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
	missing, unexpected = model.load_state_dict(state, strict=False)
	if missing or unexpected:
		print(f'[load] {len(missing)} missing, {len(unexpected)} unexpected keys')
	return config, model.to(device).eval()


class MidiFromLilyletGenerator:
	'''Autoregressive MIDI generation conditioned on a fixed lilylet score.

	The lilylet prefix is a read-only condition: its patches occupy joint positions
	[0, L) with modality 0 and RoPE positions 0..L-1. Generated midi patches occupy
	positions L.. with modality 1 and their OWN RoPE frame (0..Mp-1). At every step the
	full joint sequence is re-encoded (no KV cache) and the windowed/block mask is rebuilt
	from the per-patch own/src measure tags, so the coupling matches training exactly.
	'''

	def __init__ (self, model, lyl_tokenizer, midi_tokenizer, config, device='cpu'):
		self.model = model
		self.lt = lyl_tokenizer
		self.mt = midi_tokenizer
		self.device = torch.device(device)
		self.patch_size = model.patch_size
		self.w_midi = int(config['data.args.w_midi'] or 2)
		self.w_cross = int(config['data.args.w_cross'] or 2)
		# midi special ids (0..) — these live in the MIDI vocab, distinct from the lyl ids.
		self.m_pad = midi_tokenizer.pad_id
		self.m_bos = midi_tokenizer.bos_id
		self.m_eos = midi_tokenizer.eos_id
		self.m_eom = midi_tokenizer.eom_id

	@torch.no_grad()
	def _encode_lyl (self, lyl_patches, lyl_meas):
		'''Encode the lilylet prefix into projected memory [1, L, d_model] (done once).

		Mirrors MidiBgptTransSelfAttn.forward: frozen lyl_encoder over the real lyl patches
		(padding mask = all real here) with per-patch RoPE positions 0..L-1, then enc_proj.
		'''
		L = len(lyl_patches)
		patches = torch.tensor([lyl_patches], dtype=torch.long, device=self.device)	# [1,L,ps]
		real = torch.ones(1, L, dtype=torch.long, device=self.device)
		pos = torch.arange(L, device=self.device).unsqueeze(0)						# [1,L]
		memory = self.model.lyl_encoder(patches, real, position_ids=pos)['last_hidden_state']
		return self.model.enc_proj(memory)											# [1,L,d_model]

	@torch.no_grad()
	def _decode_patch (self, hidden, temperature, top_k, top_p, rep_penalty=1.0, rep_context=None,
		type_penalty=1.0, type_counts=None):
		'''Token-level decode of one midi event patch from its patch hidden state.

		Position 0 is the patch hidden state (not an embedding); positions 1.. are the
		embeddings of the tokens sampled so far.

		rep_penalty / rep_context: optional CTRL-style repetition penalty. rep_context is the
		set of token ids recently emitted (in the current measure); their logits are damped by
		rep_penalty (>1) at every within-patch step. 1.0 (default) disables it.

		TERMINATOR = <pad> (id 0), NOT <eos>. condPatchifier builds midi event patches with
		the lilylet `pad_patch` (content then <pad>, NO in-patch <eos>) — unlike the
		unconditional MidiTokenizer.pad_patch (which inserts <eos>). So this checkpoint's
		event patches end at the first <pad>; stopping on <eos> would overrun the event.
		The whole-patch boundary markers still work: an <eom> patch is [<eom>, <pad>...]
		(first token <eom>) and the <eos> patch is [<bos>, <eos>, ...] (caught by the
		first-token check in generate()); both are detected before padding matters.

		CAVEAT: condPatchifier masks post-content <pad> positions (-100) in the token loss,
		so the content->pad transition is UNSUPERVISED — the model has no strong learned
		signal to stop, and free-running decode can run long / degenerate. <pad> is the
		correct data-layout terminator, but generation quality is bounded by that gap.
		'''
		dec = self.model.token_level_decoder.base
		wte = token_embedding_weight(dec)
		generated = []
		enc = hidden.reshape(1, 1, -1)
		emb = enc
		while len(generated) < self.patch_size:
			logits = dec(inputs_embeds=emb).logits[0, -1]
			# Repetition penalty (CTRL-style): divide the logit of any token seen in the recent
			# measure history by `rep_penalty` (>1) — positive logits shrink, negatives grow more
			# negative, so recently-emitted tokens are discouraged. This counters the free-running
			# collapse (see gen diagnosis): under the windowed mask the model loses the long-range
			# context that would otherwise damp repetition, so greedy locks onto one event.
			if rep_penalty and rep_penalty != 1.0 and rep_context:
				logits = logits.clone()
				ids = torch.tensor(sorted(rep_context), device=self.device)
				sel = logits[ids]
				logits[ids] = torch.where(sel > 0, sel / rep_penalty, sel * rep_penalty)
			# FIRST-TOKEN (event-type) penalty, applied ONLY at position 0: the collapse locks on
			# an event TYPE (e.g. control_change 63% of patches) and dodges the content-level penalty
			# by varying only the value byte. `type_counts` maps event-type token id -> how many of
			# the recent patches started with it; penalise proportionally so a type that keeps
			# recurring is pushed down (freeing <eom> and other event types to win).
			if len(generated) == 0 and type_penalty and type_penalty > 1.0 and type_counts:
				logits = logits.clone()
				for tid, c in type_counts.items():
					pen = type_penalty ** c						# compounding with recurrence count
					logits[tid] = logits[tid] / pen if logits[tid] > 0 else logits[tid] * pen
			nxt = sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p)
			# first <pad> ends the event; right-pad the rest (matches the training layout).
			if nxt == self.m_pad and generated:
				generated += [self.m_pad] * (self.patch_size - len(generated))
				break
			generated.append(nxt)
			if len(generated) >= self.patch_size:
				break
			tok = torch.tensor([[nxt]], device=self.device)
			emb = torch.cat((emb, F.embedding(tok, wte)), dim=1)
		return generated

	@torch.no_grad()
	def generate (self, lyl_text, header=None, max_midi_patches=2048, max_measures=0,
		temperature=1.0, top_k=0, top_p=0.9, rep_penalty=1.0, rep_window=8,
		type_penalty=1.0, verbose=False):
		'''Generate a whole-song MidiText string conditioned on `lyl_text`.

		header: list of MidiText header lines (measure 0). max_measures: stop after this
		many <eom>-closed measures (0 = only stop on <eos> / max_midi_patches).
		rep_penalty: CTRL-style repetition penalty (>1 discourages recently-emitted content
		tokens); 1.0 disables. rep_window: how many recent PATCHES contribute to the penalised
		token set. type_penalty: separate penalty on the patch FIRST TOKEN (event type),
		compounding with how many recent patches shared that type — targets the event-type lock
		(e.g. control_change) that the content-level penalty can't reach.
		'''
		header = DEFAULT_HEADER if header is None else header

		# --- fixed lilylet prefix: patches + measure tags, encoded once ---
		lyl_patches, lyl_meas = patchify_lilylet(lyl_text, self.lt, patch_size=self.patch_size)
		L = len(lyl_patches)
		memory = self._encode_lyl(lyl_patches, lyl_meas)			# [1,L,d_model]

		# --- midi seed: deterministic header patches (measure 0), never re-predicted ---
		midi_patches = [self.mt.pad_patch(self.mt.encode_event(h)) for h in header]
		midi_own = [0] * len(midi_patches)							# header -> measure 0
		gen_lines = list(header)									# header lines echoed into output

		wte_dtype = self.model.enc_proj.weight.dtype
		cur_measure = 1												# first body patch is measure 1
		steps = 0
		stop = None
		while len(midi_patches) < max_midi_patches:
			# assemble the joint sequence: [lyl prefix ; midi (header + body so far)]
			joint_patches = lyl_patches + midi_patches
			T = len(joint_patches)
			patches = torch.tensor([joint_patches], dtype=torch.long, device=self.device).reshape(1, T, self.patch_size)
			positions = torch.zeros(1, T, dtype=torch.long, device=self.device)
			positions[0, :L] = torch.arange(L, device=self.device)
			positions[0, L:] = torch.arange(T - L, device=self.device)
			modality = torch.zeros(T, dtype=torch.long, device=self.device)
			modality[L:] = 1
			own = torch.tensor(lyl_meas + midi_own, dtype=torch.long, device=self.device)
			# identity alignment (source_measure == index on the no-repeat corpus): src == own.
			vis = build_vis(modality, own, own, self.w_midi, self.w_cross)			# [T,T]
			attn_mask = vis.unsqueeze(0).unsqueeze(0)								# [1,1,T,T]

			# joint embedding: midi embed everywhere, then overwrite the lyl prefix with memory.
			embeds = self.model._midi_embed(patches)								# [1,T,d_model]
			embeds[:, :L] = memory.to(wte_dtype)
			dec = self.model.decoder(inputs_embeds=embeds, attention_mask=attn_mask,
				position_ids=positions)['last_hidden_state']
			# recent-token set for the content repetition penalty: content tokens of the last
			# rep_window generated patches (drop pad/bos/eos so structural markers aren't penalised).
			recent = midi_patches[-rep_window:]
			rep_context = None
			if rep_penalty and rep_penalty != 1.0:
				rep_context = {t for p in recent for t in p
					if t not in (self.m_pad, self.m_bos, self.m_eos)}
			# event-type counts over the same window: first token of each recent patch -> count.
			# Drives the first-token penalty against the dominant event-type lock (never counts
			# <eom>/<eos>, so the structural markers are free to fire).
			type_counts = None
			if type_penalty and type_penalty > 1.0:
				type_counts = {}
				for p in recent:
					t0 = p[0]
					if t0 in (self.m_eom, self.m_eos, self.m_bos, self.m_pad):
						continue
					type_counts[t0] = type_counts.get(t0, 0) + 1
			patch_ids = self._decode_patch(dec[0, -1], temperature, top_k, top_p,
				rep_penalty=rep_penalty, rep_context=rep_context,
				type_penalty=type_penalty, type_counts=type_counts)
			steps += 1

			# <eos> patch [bos, eos, ...] -> stop
			if patch_ids[0] == self.m_bos and patch_ids[1] == self.m_eos:
				stop = 'eos'
				break
			# <eom> patch [eom, eos?, pad...] -> close the current measure, advance
			if patch_ids[0] == self.m_eom:
				midi_patches.append(patch_ids)
				midi_own.append(cur_measure)
				cur_measure += 1
				if verbose:
					print('  [eom -> measure %d]' % cur_measure)
				if max_measures and cur_measure > max_measures:
					stop = 'max_measures'
					break
				continue
			# normal event patch at the current measure
			midi_patches.append(patch_ids)
			midi_own.append(cur_measure)
			line = self.mt.decode_event(patch_ids)
			if line:
				gen_lines.append(line)
				if verbose:
					print('  ' + line)
		if stop is None:
			stop = 'max_patches'
		return '\n'.join(gen_lines) + '\n', dict(
			stop=stop, steps=steps, midi_patches=len(midi_patches),
			measures=cur_measure - 1, lyl_patches=L)


def _pick_sources (files, src_dir, num):
	'''Resolve explicit --files or default to the `num` SHORTEST .lyl in src_dir (short =
	fewer patches = a smaller O(T^2) joint sequence, so the CPU test stays quick).'''
	if files:
		out = []
		for f in files:
			out.append(f if os.path.isabs(f) or os.path.exists(f) else os.path.join(src_dir, f))
		return out
	allf = glob.glob(os.path.join(src_dir, '*.lyl'))
	allf.sort(key=lambda p: os.path.getsize(p))
	return allf[:num]


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--run', default=RUN, help='training run dir (best.chkpt + .state.yaml)')
	ap.add_argument('--checkpoint', default=None, help='checkpoint path (default: <run>/best.chkpt)')
	ap.add_argument('--src-dir', default=SRC_DIR, help='dir of source .lyl files')
	ap.add_argument('--files', nargs='*', default=None,
		help='specific .lyl files (abs, or names under --src-dir); default: shortest N')
	ap.add_argument('--num', type=int, default=3, help='how many shortest .lyl to use when --files omitted')
	ap.add_argument('--max-midi-patches', type=int, default=4096)
	ap.add_argument('--max-measures', type=int, default=0, help='stop after N measures (0 = until <eos>)')
	ap.add_argument('--temperature', type=float, default=1.0)
	ap.add_argument('--top-k', type=int, default=0)
	ap.add_argument('--top-p', type=float, default=0.9)
	ap.add_argument('--rep-penalty', type=float, default=1.0,
		help='CTRL-style repetition penalty (>1 discourages recently-emitted tokens; 1.0 = off)')
	ap.add_argument('--rep-window', type=int, default=8,
		help='number of recent patches contributing to the repetition-penalty token set')
	ap.add_argument('--type-penalty', type=float, default=1.0,
		help='event-type (patch first-token) penalty, compounding with recurrence count in the '
		'rep-window; >1 breaks the dominant-event-type lock (e.g. control_change). 1.0 = off')
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--threads', type=int, default=14)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--out-dir', default=os.path.join(REPO_ROOT, 'tests', 'output', 'midi_from_lilylet'))
	ap.add_argument('--verbose', action='store_true')
	args = ap.parse_args()

	torch.set_num_threads(args.threads)
	device = args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
	if device != args.device:
		print('[warn] cuda unavailable, falling back to cpu')

	ckpt = args.checkpoint or os.path.join(args.run, 'best.chkpt')
	assert os.path.isfile(ckpt), f'checkpoint not found: {ckpt}'
	config, model = load_model(args.run, ckpt, device)
	n_params = sum(p.numel() for p in model.parameters())
	print('run:', args.run)
	print('checkpoint:', ckpt)
	print('params: %.2fM | d_model %d | patch_size %d | w_midi %s w_cross %s | device %s'
		% (n_params / 1e6, config['model.args.d_model'], model.patch_size,
			config['data.args.w_midi'], config['data.args.w_cross'], device))

	lt = LilyletTokenizer(LYL_TOKENIZER)
	mt = MidiTokenizer(model.patch_size)
	gen = MidiFromLilyletGenerator(model, lt, mt, config, device=device)

	sources = _pick_sources(args.files, args.src_dir, args.num)
	assert sources, 'no source .lyl files found'
	os.makedirs(args.out_dir, exist_ok=True)
	print('\n--- %d source(s) (temp=%.2f top_k=%d top_p=%.2f max_patches=%d max_measures=%s) ---'
		% (len(sources), args.temperature, args.top_k, args.top_p, args.max_midi_patches, args.max_measures))

	for i, src in enumerate(sources):
		torch.manual_seed(args.seed)
		with open(src, encoding='utf-8') as f:
			lyl_text = f.read()
		name = os.path.splitext(os.path.basename(src))[0]
		print('\n===== [%d/%d] %s (%d bytes) =====' % (i + 1, len(sources), name, len(lyl_text)))
		t0 = time.perf_counter()
		text, info = gen.generate(lyl_text, max_midi_patches=args.max_midi_patches,
			max_measures=args.max_measures, temperature=args.temperature,
			top_k=args.top_k, top_p=args.top_p,
			rep_penalty=args.rep_penalty, rep_window=args.rep_window,
			type_penalty=args.type_penalty, verbose=args.verbose)
		dt = time.perf_counter() - t0
		out = os.path.join(args.out_dir, name + '.midi.txt')
		with open(out, 'w', encoding='utf-8') as f:
			f.write(text)
		# sanity: the generated MidiText should re-encode cleanly
		reenc, dropped = mt.encode_patches(text, add_special_patches=False)
		print('[%s] %.1fs | lyl_patches %d | midi_patches %d | measures %d | %d event lines'
			% (info['stop'], dt, info['lyl_patches'], info['midi_patches'], info['measures'],
				text.count('\n')))
		print('  re-encode: %d patches, %d dropped -> %s' % (len(reenc), len(dropped), out))


if __name__ == '__main__':
	main()

