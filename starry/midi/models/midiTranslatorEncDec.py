'''MidiTranslatorEncDec — an ENCODER-DECODER translator over the same midiseq2 corpus.

The sibling of `midiTranslator.MidiTranslator`, and deliberately its minimal contrast. Both read
`starry.midi.data.seq2seq2.Seq2Seq2`, both use one shared embedding table over one vocabulary, both
are supervised on exactly the same labels; the ONLY difference is where the source goes:

	MidiTranslator         one causal stack, source as a prefix in the same sequence
	MidiTranslatorEncDec   a BIDIRECTIONAL encoder over the source, a causal decoder over the target
	                       reading it through cross-attention

So the two runs answer one question: does the source half benefit from bidirectional attention (an
encoder sees the whole source at every position) and from being freed of the causal stack's length
budget, enough to pay for the cross-attention parameters?

The feeder supplies the split shape via `pack: 'split'` (see its module docstring). That option cuts
the SAME assembled sequence at its recorded <sep>, so alignment, wrappers, positions and the
unsupervised `skip` head are untouched, and the supervised label set is bit-identical to the flat
form's. `tests/midi/seq2seq2_pack_check.py` asserts that equality — without it, a loss difference
between the two architectures could just as easily be a difference in what they were asked to predict.

Four decisions worth stating, because each one is a place this could quietly go wrong:

<sep> is the decoder's start token. The target half's own <bos> is conditional (it appears only when
the crop reaches the start of the piece), so it cannot be a start token; <sep> is unconditional,
already means "the target begins here" in the flat form, and is already in the vocabulary.

RoPE applies to self-attention only, never to cross-attention. The two halves sit on different
position axes — under `pos_style: 'absolute'` deliberately so, each half carrying where in its own
file it came from — so a query-key distance ACROSS them is not a distance in any axis. The encoder
states reach the decoder through the mask and the content, positionally unrotated.

That split also decides which library each sublayer comes from. Self-attention is built from llama's
components (RMSNorm pre-norm, SwiGLU MLP, RoPE), matching MidiTranslator's backbone block-for-block so
a quality difference between the two runs is architecture and not block details. Cross-attention has no
llama counterpart to reuse — `LlamaAttention` takes a single `hidden_states` and its own position
embeddings, with no key/value argument — so it is the repo's own
`starry.transformer.sub_layers.MultiHeadAttention`, which already takes separate q/k/v and is what
starry.transformer.layers.DecoderLayer and midiBgptTrans use. It is post-norm and owns its residual,
so a decoder layer is deliberately mixed; DecoderLayer documents the consequences.

The masks are not symmetric between the two stacks, and `_attn_mask` / `_keep_mask` document why: the
encoder needs its key-padding mask (bidirectional attention would otherwise read the pad tail), while
the decoder's self-attention deliberately takes the causal mask ALONE. Causality plus right padding
already stops every real query from reading a pad key, so the key-padding mask changes nothing that
survives: real query rows come out bit-identical, and the only rows it does change are pad QUERY rows,
which target_mask discards. It is omitted as redundant, not as dangerous — see `_attn_mask` for why the
usual all -inf/nan hazard does not arise here. Cross-attention takes the source key-padding mask, since
the decoder is not causal w.r.t. the encoder.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
	LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding, apply_rotary_pos_emb)

from ...transformer.sub_layers import MultiHeadAttention
from ...utils.registry import register_model
from .midiTranslator import MidiTranslatorLoss


def _block_config (d_model, n_head, d_inner, dropout, max_seq_len, rope_theta):
	'''One LlamaConfig, reused for the MLP and the rotary embedding.

	Built from llama's own components rather than hand-rolled so this model's blocks are the same
	arithmetic as MidiTranslator's backbone (RMSNorm pre-norm, SwiGLU MLP, RoPE) — otherwise a
	quality difference between the two runs would confound architecture with block details.
	'''
	return LlamaConfig(
		hidden_size=d_model,
		num_attention_heads=n_head,
		num_key_value_heads=n_head,
		intermediate_size=d_inner or d_model * 4,
		max_position_embeddings=max_seq_len,
		attention_dropout=dropout,
		rope_theta=rope_theta,
		mlp_bias=False,
	)


def _default_positions (ids, position_ids):
	"""Fall back to 0..T-1 when the caller passes no positions.

	Needed explicitly: LlamaModel derives this internally, but LlamaRotaryEmbedding — which this model
	drives directly — requires real position_ids and raises on None. 0..T-1 is exactly `pos_style:
	'flat'`, so a batch built without positions behaves as the flat style rather than crashing.
	"""
	if position_ids is not None:
		return position_ids
	return torch.arange(ids.shape[1], device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)


def _keep_mask (key_padding):
	"""[B, Lk] 1=real -> [B, 1, Lk] boolean keep-mask for the repo's MultiHeadAttention, or None.

	That module takes a keep-mask (`masked_fill(mask == 0, -1e9)`) and unsqueezes a head axis itself,
	so [B, 1, Lk] arrives as [B, 1, 1, Lk] and broadcasts across heads AND across query positions —
	which is exactly right for cross-attention, where what gets masked is the source key, the same for
	every decoder position.
	"""
	return None if key_padding is None else key_padding.bool().unsqueeze(1)


def _attn_mask (query_len, key_padding, causal, dtype, device):
	'''Additive [B, 1, Lq, Lk] mask, or None when nothing needs masking.

	`key_padding` is a [B, Lk] 1=real mask or None. `causal` adds the upper-triangular block, which
	assumes Lq == Lk (true for self-attention; cross-attention is never causal here).

	NOTE the asymmetry, which is the subtle part: a caller passes key_padding for BIDIRECTIONAL
	self-attention, but NOT for causal self-attention over a right-padded batch (cross-attention does
	not come through here at all — it masks via `_keep_mask`, the boolean form the repo's
	MultiHeadAttention takes). Under causal self-attention, causality already stops every real query
	from reading a pad key, so the key-padding mask changes no number that survives target_mask: it
	alters only pad QUERY rows, whose outputs are discarded. Redundant, so not passed.

	Two notes on the fill, because the usual "all -inf row -> softmax nan" worry does NOT apply here
	and it is worth recording why rather than re-deriving it:

	- the fill is `finfo(dtype).min`, which is FINITE. A row masked everywhere therefore softmaxes to
	  uniform (finite garbage) rather than nan. Summing the two blocks does saturate to -inf where both
	  apply, so -inf does appear in the tensor — just never across a whole row.
	- the causal block leaves the DIAGONAL at 0, so under `causal` every query row keeps at least one
	  unmasked key whatever the key-padding says.

	Verified exhaustively over all 32 key-padding layouts of a width-5 row, causal and not: no row is
	ever entirely -inf and no softmax is ever nan.
	'''
	mask = None
	minimum = torch.finfo(dtype).min
	if key_padding is not None:
		mask = torch.zeros(key_padding.shape[0], 1, 1, key_padding.shape[1], dtype=dtype, device=device)
		mask = mask.masked_fill(~key_padding.bool().unsqueeze(1).unsqueeze(1), minimum)
		mask = mask.expand(-1, 1, query_len, -1)
	if causal:
		tri = torch.ones(query_len, query_len, dtype=torch.bool, device=device).triu(1)
		block = torch.zeros(query_len, query_len, dtype=dtype, device=device).masked_fill(tri, minimum)
		mask = block.expand(1, 1, -1, -1) if mask is None else mask + block
	return mask


class SelfAttention (nn.Module):
	'''RoPE multi-head SELF-attention, used by both stacks (bidirectional in the encoder, causal in
	the decoder — the difference is entirely in the mask the caller passes).

	Self-attention only, by construction: there is no key/value argument to pass a second sequence
	through. Cross-attention is the repo's `starry.transformer.sub_layers.MultiHeadAttention` instead,
	which already takes separate q/k/v — llama has no cross-attention module to reuse (LlamaAttention
	takes one `hidden_states` and its own position embeddings), and RoPE would be meaningless across
	the two halves' separate position axes anyway. So the split falls out of what each library offers:
	llama's blocks where RoPE applies, the repo's where it does not.
	'''

	def __init__ (self, config, dropout, rotary):
		super().__init__()
		self.n_head = config.num_attention_heads
		self.d_head = config.hidden_size // self.n_head
		if self.d_head * self.n_head != config.hidden_size:
			raise ValueError(f'd_model {config.hidden_size} is not divisible by n_head {self.n_head}')
		self.rotary = rotary
		self.dropout = dropout
		self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
		self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
		self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
		self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

	def _heads (self, x):
		b, t, _ = x.shape
		return x.view(b, t, self.n_head, self.d_head).transpose(1, 2)

	def forward (self, x, mask=None, position_ids=None):
		'''x: [B, T, D]. mask: additive [B, 1, T, T] from `_attn_mask`. Returns [B, T, D].'''
		q = self._heads(self.q_proj(x))
		k = self._heads(self.k_proj(x))
		v = self._heads(self.v_proj(x))

		cos, sin = self.rotary(x, position_ids)
		q, k = apply_rotary_pos_emb(q, k, cos, sin)

		out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
			dropout_p=self.dropout if self.training else 0.0)
		b, _, t, _ = out.shape
		out = out.transpose(1, 2).contiguous().view(b, t, self.n_head * self.d_head)
		return self.o_proj(out)


class EncoderLayer (nn.Module):
	'''Pre-norm bidirectional self-attention + MLP.'''

	def __init__ (self, config, dropout, rotary):
		super().__init__()
		self.self_attn = SelfAttention(config, dropout, rotary)
		self.mlp = LlamaMLP(config)
		self.attn_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.mlp_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward (self, x, mask, position_ids):
		x = x + self.self_attn(self.attn_norm(x), mask, position_ids)
		return x + self.mlp(self.mlp_norm(x))


class DecoderLayer (nn.Module):
	'''Causal self-attention -> cross-attention over the encoder -> MLP.

	Cross-attention sits between the two, the standard placement: the token first integrates what the
	target has emitted so far, then reads the source conditioned on that.

	The cross sublayer is the repo's `MultiHeadAttention` (starry/transformer/sub_layers.py), the same
	module starry.transformer.layers.DecoderLayer and midiBgptTrans build on. Two consequences, since
	it is not a drop-in twin of the self-attention above:

	  * it is POST-norm and owns its own residual (`q += residual` then `layer_norm`), so it is applied
	    WHOLE rather than wrapped in a pre-norm residual here — wrapping it would double the residual
	    and put the norm in the wrong place. Self-attention and the MLP stay pre-norm llama blocks, so
	    a decoder layer is deliberately mixed: llama's blocks where RoPE applies, the repo's where it
	    does not.
	  * its mask is a BOOLEAN keep-mask (0 = masked, via `masked_fill(mask == 0, -1e9)`), not the
	    additive float mask `_attn_mask` builds — see `_keep_mask`. It returns the attention weights
	    for free, which is what the cross-attention alignment inspection reads.

	It also materializes the full [B, H, U, S] weight matrix instead of going through SDPA, so a
	long-source batch costs more memory here than the self-attention does. That is the same cost every
	other repo model built on this module already pays.
	'''

	def __init__ (self, config, dropout, rotary):
		super().__init__()
		d_model = config.hidden_size
		d_head = d_model // config.num_attention_heads
		self.self_attn = SelfAttention(config, dropout, rotary)
		self.cross_attn = MultiHeadAttention(config.num_attention_heads, d_model, d_head, d_head,
			dropout=dropout)
		self.mlp = LlamaMLP(config)
		self.attn_norm = LlamaRMSNorm(d_model, eps=config.rms_norm_eps)
		self.mlp_norm = LlamaRMSNorm(d_model, eps=config.rms_norm_eps)

	def forward (self, x, memory, self_mask, cross_keep, position_ids):
		'''`cross_keep`: boolean keep-mask broadcastable to [B, n_head, U, S], or None.

		Returns (x, cross_attention_weights). The weights come out of MultiHeadAttention regardless, so
		they are always returned rather than gated behind a flag.
		'''
		x = x + self.self_attn(self.attn_norm(x), self_mask, position_ids)
		x, weights = self.cross_attn(x, memory, memory, mask=cross_keep)
		return x + self.mlp(self.mlp_norm(x)), weights


@register_model
class MidiTranslatorEncDec (nn.Module):
	'''Encoder-decoder over the midiseq2 vocabulary: (source, decoder_input) -> logits [B, U, vocab].

	One embedding table is SHARED by both stacks. The two halves are the same vocabulary — the same
	pitch, elapse and channel tokens, differing only in which rendering of the piece they describe —
	so separate tables would learn the same thing twice from half the data each.

	`logits[:, j]` predicts `labels[:, j]`: the shift lives in the feeder's split collate, so this
	model's output aligns with the labels position-for-position and needs no further offset.
	'''

	# vocab_size's default is the current asset's size, kept only so a bare constructor builds for
	# probes; a real run has it injected by the vocabulary asset. Same contract as MidiTranslator.
	def __init__ (self, vocab_size=582, d_model=512, n_layer=None, n_layer_encoder=6,
		n_layer_decoder=6, n_head=8, d_inner=None, max_seq_len=4096, dropout=0.1,
		rope_theta=10000.0, tie_embedding=False, sep_id=5, eos_id=2, pad_id=0, **_):
		super().__init__()

		# n_layer is a convenience: one number for a symmetric stack. The two explicit counts win when
		# given, so a config can be asymmetric (a deeper decoder over a shallow encoder is a real
		# option here — the source is read-only and the target is what gets generated).
		if n_layer is not None:
			n_layer_encoder = n_layer_decoder = n_layer

		self.vocab_size = vocab_size
		self.max_seq_len = max_seq_len
		self.sep_id = sep_id
		self.eos_id = eos_id
		self.pad_id = pad_id

		config = _block_config(d_model, n_head, d_inner, dropout, max_seq_len, rope_theta)
		# One rotary embedding instance shared by every self-attention: it holds no parameters, only
		# the inv_freq buffer, and every stack is on the same theta.
		self.rotary = LlamaRotaryEmbedding(config)
		self.embed_tokens = nn.Embedding(vocab_size, d_model)
		self.encoder = nn.ModuleList(
			EncoderLayer(config, dropout, self.rotary) for _ in range(n_layer_encoder))
		self.decoder = nn.ModuleList(
			DecoderLayer(config, dropout, self.rotary) for _ in range(n_layer_decoder))
		self.encoder_norm = LlamaRMSNorm(d_model, eps=config.rms_norm_eps)
		self.decoder_norm = LlamaRMSNorm(d_model, eps=config.rms_norm_eps)
		self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

		nn.init.normal_(self.embed_tokens.weight, std=0.02)
		if tie_embedding:
			self.lm_head.weight = self.embed_tokens.weight
		else:
			nn.init.normal_(self.lm_head.weight, std=0.02)

	def parameters_trainable (self):
		'''Every parameter, NOT filtered on the live requires_grad flag.

		The distributed validator calls `model.requires_grad_(False)` before its per-epoch parameter
		broadcast, so a flag-based filter returns an empty list on the validator while the trainer
		returns everything — broadcastParam then pairs mismatched sizes and gloo aborts. Enumerating
		all parameters gives the SAME list in the SAME order on both ranks. See
		MidiTranslator.parameters_trainable for the full account.
		'''
		return list(self.parameters())

	def encode (self, source_ids, source_masks=None, source_position_ids=None):
		'''Source -> memory [B, S, D]. Separate from `forward` because generation encodes ONCE and
		then steps the decoder; calling forward per step would re-encode the source every token.

		The self-attention here is FULL (bidirectional): every source position attends to every other
		one, which is the whole point of the encoder arm. The key-padding mask is therefore
		load-bearing rather than redundant as it is under a causal stack.
		'''
		x = self.embed_tokens(source_ids)
		mask = _attn_mask(source_ids.shape[1], source_masks, False, x.dtype, x.device)
		positions = _default_positions(source_ids, source_position_ids)
		for layer in self.encoder:
			x = layer(x, mask, positions)
		return self.encoder_norm(x)

	def decode (self, memory, decoder_input_ids, source_masks=None, decoder_position_ids=None,
		need_weights=False):
		'''(memory, decoder inputs) -> logits [B, U, vocab], and the per-layer cross-attention
		weights when asked (for the alignment plots in tools/midi/translateMidiseq2.py --inspect).

		`decoder_masks` is deliberately NOT a parameter: causal self-attention over a right-padded
		batch must not take a key-padding mask (see `_attn_mask`), and cross-attention masks the
		SOURCE, not the target. Accepting one would invite passing it where it breaks things.
		'''
		x = self.embed_tokens(decoder_input_ids)
		self_mask = _attn_mask(decoder_input_ids.shape[1], None, True, x.dtype, x.device)
		cross_keep = _keep_mask(source_masks)
		positions = _default_positions(decoder_input_ids, decoder_position_ids)
		weights = []
		for layer in self.decoder:
			x, w = layer(x, memory, self_mask, cross_keep, positions)
			if need_weights:
				weights.append(w)
		logits = self.lm_head(self.decoder_norm(x))
		return (logits, weights) if need_weights else logits

	def forward (self, source_ids, decoder_input_ids, source_masks=None, source_position_ids=None,
		decoder_position_ids=None):
		'''
		source_ids:           LongTensor [B, S]
		decoder_input_ids:    LongTensor [B, U]  — <sep> ++ target[:-1] from the split collate
		source_masks:         LongTensor [B, S] or None — 1 = real token
		source_position_ids:  LongTensor [B, S] or None — the source half of the feeder's pos_style
		decoder_position_ids: LongTensor [B, U] or None
		Returns: FloatTensor [B, U, vocab] — logits[:, j] predicts labels[:, j], NO further shift.

		Position values may be negative and may exceed max_seq_len ('sep' and 'absolute' both produce
		negatives): RoPE computes sin/cos from the value itself, so nothing indexes a table and
		neither case is out of range.
		'''
		memory = self.encode(source_ids, source_masks, source_position_ids)
		return self.decode(memory, decoder_input_ids, source_masks, decoder_position_ids)

	@torch.no_grad()
	def generate (self, source_ids, max_new_tokens=512, eos_id=None, temperature=0.0,
		source_masks=None, source_position_ids=None, decoder_start_position=None):
		'''Free-run the target for ONE source. Returns the generated ids WITHOUT the leading <sep>.

		temperature 0 = greedy, else sample from the softmax. Stops at eos_id, or the configured
		target EOS when eos_id is None.

		`decoder_start_position` is <sep>'s position on the feeder's axis; each generated token
		continues that run (+1 per step), matching what the target half does under every pos_style.
		It defaults to `source_position_ids[-1] + 1` — correct for 'flat' — and must be passed as -1
		for 'sep' and 'absolute', where <sep> sits at -1 by construction. None with no source
		positions leaves the decoder at 0..U-1, i.e. 'flat'.

		The source is encoded ONCE. The decoder still recomputes its own prefix every step (no KV
		cache), so this is O(U^2) in the target and is meant for inspection, not bulk decoding — but
		the source, which is the longer half, costs one pass regardless.
		'''
		ids = source_ids if source_ids.dim() == 2 else source_ids.unsqueeze(0)
		if ids.shape[0] != 1:
			raise ValueError('generate handles one source at a time')
		masks = None
		if source_masks is not None:
			masks = source_masks if source_masks.dim() == 2 else source_masks.unsqueeze(0)
		positions = None
		if source_position_ids is not None:
			positions = source_position_ids if source_position_ids.dim() == 2 else source_position_ids.unsqueeze(0)

		memory = self.encode(ids, masks, positions)

		start = decoder_start_position
		if start is None and positions is not None:
			start = int(positions[0, -1]) + 1
		decoder_ids = torch.full((1, 1), self.sep_id, dtype=torch.long, device=ids.device)
		decoder_positions = None
		if start is not None:
			decoder_positions = torch.full((1, 1), int(start), dtype=torch.long, device=ids.device)

		out = []
		stop = self.eos_id if eos_id is None else eos_id
		for _ in range(max_new_tokens):
			window = decoder_ids[:, -self.max_seq_len:]
			position_window = None if decoder_positions is None else decoder_positions[:, -self.max_seq_len:]
			logits = self.decode(memory, window, masks, position_window)[:, -1, :]
			if temperature and temperature > 0:
				nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
			else:
				nxt = logits.argmax(dim=-1, keepdim=True)
			token = int(nxt.item())
			out.append(token)
			decoder_ids = torch.cat((decoder_ids, nxt), dim=1)
			if decoder_positions is not None:
				decoder_positions = torch.cat((decoder_positions, decoder_positions[:, -1:] + 1), dim=1)
			if token == stop:
				break
		return torch.tensor(out, dtype=torch.long, device=ids.device)


@register_model
class MidiTranslatorEncDecLoss (MidiTranslatorLoss):
	'''Training wrapper. Inherits everything about the VOCABULARY and the METRICS from
	MidiTranslatorLoss and overrides only what the architecture changes.

	That split is the point: the type map, the per-type CE weights, the grouped error rates and
	`stat`'s aggregation are identical to the decoder-only run's, so the two runs' TensorBoard panels
	measure the same quantities and their `loss` curves are directly comparable. Only three hooks
	differ — `DEDUCER`, `_logits`, `_shift`.

	Named `MidiTranslatorEncDec` + 'Loss' because both trainers construct
	`loadModel(config['model'], postfix='Loss')` from `type: MidiTranslatorEncDec`. Only
	`self.deducer` is checkpointed.
	'''

	DEDUCER = MidiTranslatorEncDec

	def __init__ (self, loss_type_weights=None, vocab_path=None, **kw_args):
		super().__init__(loss_type_weights=loss_type_weights, vocab_path=vocab_path, **kw_args)
		# <sep> is the decoder's start token and must be the number the FEEDER's vocabulary uses. Taken
		# from the tokenizer the base class already resolved rather than from the config, so a run
		# cannot pin one vocabulary for the data and a different token numbering for the model.
		self.deducer.sep_id = self.sep_id
		self.deducer.pad_id = self.pad_id

	def _logits (self, batch):
		return self.deducer(batch['source_ids'], batch['decoder_input_ids'],
			batch.get('source_masks'), batch.get('source_position_ids'),
			batch.get('decoder_position_ids'))

	def _shift (self, batch, logits):
		'''NO shift — the split collate already did it.

		`logits[:, j]` predicts `labels[:, j]`, so this only selects the supervised positions and
		flattens, returning the same ([N, vocab], [N]) pair the base class's version does. N itself is
		equal to the flat form's for the same crop, which is what makes the two losses comparable;
		tests/midi/seq2seq2_pack_check.py asserts the label sets match element for element.
		'''
		sel = batch['target_mask'].bool()
		return logits[sel], batch['labels'][sel]
