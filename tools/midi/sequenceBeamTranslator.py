"""Shared-window translators backed by the additive sequence beam search."""

from dataclasses import dataclass, replace
from collections import Counter
import math
import time

import torch
import torch.nn.functional as F

from starry.midi.align import AlignState, soft_delta, soft_indices
from starry.midi.beam import BranchState
from starry.midi.sequenceBeam import sequence_beam
from translateMidiseq2 import (SlidingTranslator, SlidingEncDecTranslator,
                             encode_lines, note_on_events)


def choose_candidate(source, greedy, beam, greedy_done, beam_done, margin=0.05):
    """Conservative whole-file selection using input pitches, never score labels.

    A greedy candidate is generated independently, so a bad beam primer cannot
    contaminate its fallback. Small source/target differences are expected under
    augmentation; only a five-point loss of source pitch F1 triggers that guard.
    Completion is checked separately from pitch overlap.
    """
    source = Counter(source)
    def fidelity(pitches):
        pitches = Counter(pitches)
        size = sum(source.values()) + sum(pitches.values())
        return 2.0 * sum((source & pitches).values()) / size if size else 1.0
    gs, bs = fidelity(greedy), fidelity(beam)
    reason = ('completion' if greedy_done and not beam_done else
              'source_fidelity' if bs < gs - margin else None)
    return dict(selected='greedy' if reason else 'beam', reason=reason,
                greedy_source_f1=gs, beam_source_f1=bs, margin=margin,
                greedy_done=greedy_done, beam_done=beam_done)


@dataclass(frozen=True)
class EventState:
    align: object
    tick: int = 0
    keyword: str = ''
    previous_onset: object = None
    softindex: float = 0.0


def advance_event(state, token, keywords):
    """Charge each placed onset once; never forecast a partial elapse run.

    Alignment objects are shared for structural/argument tokens and cloned only
    when a note is placed. A miss is a soft cost, not a veto. Timing uses the
    parent's prediction (before observing the note), with a quarter-beat floor
    on uncertainty. Each note costs at most two units, independently of history.
    """
    if token == '<eos>':
        # EOS ends a decoding window but is omitted from the continuous output.
        # Keep a pending event resumable when the next window supplies its fields.
        return state, 0.0
    if token.startswith('E'):
        return replace(state, tick=state.tick + int(token[1:], 16), keyword=''), 0.0
    if token.startswith('<'):
        return replace(state, keyword=''), 0.0
    if token in keywords:
        return replace(state, keyword=token), 0.0
    if state.keyword != 'note_on' or not token.startswith('#'):
        return state, 0.0
    si = state.softindex
    if state.previous_onset is not None:
        si += soft_delta(state.tick - state.previous_onset)
    align = state.align.clone()
    detail = align.observe(int(token[1:], 16), state.tick, si)
    # Small residuals are normal under rubato and reordered chords. A dead band
    # avoids trading a correct model choice for tiny, noisy alignment differences.
    cost = (1.0 if detail['src'] is None
            else min(1.0, max(0.0, detail['self_cost'] - 0.5) * 2.0))
    if detail['src'] is not None and len(state.align.pairs) >= 8:
        predicted = state.align.predict_tick(state.align.src_events[detail['src']]['onset'])
        if predicted is not None and state.align.residual is not None:
            uncertainty = max(120.0, 2.0 * state.align.residual)
            error = max(0.0, abs(state.tick - predicted) - uncertainty) / uncertainty
            cost += error / (1.0 + error)
    return replace(state, align=align, previous_onset=state.tick, softindex=si), cost


class SequenceBeamMixin:
    """Only generation changes. Primer/source advancement remains the greedy code."""

    def __init__(self, *args, beam_size=4, branch_k=4, alignment_weight=0.0,
                 length_alpha=0.7, logprob_margin=2.0, inspect=False, guard=True,
                 rescue_alignment_weight=0.5, **kwargs):
        self._baseline_args = args
        self._baseline_kwargs = dict(kwargs)
        super().__init__(*args, **kwargs)
        if beam_size < 1 or branch_k < 1:
            raise ValueError('beam_size and branch_k must be positive')
        for name, value in [('alignment_weight', alignment_weight), ('length_alpha', length_alpha),
                            ('logprob_margin', logprob_margin),
                            ('rescue_alignment_weight', rescue_alignment_weight)]:
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be finite and nonnegative')
        self.beam_size = beam_size
        self.branch_k = branch_k
        self.alignment_weight = alignment_weight
        self.length_alpha = length_alpha
        self.logprob_margin = logprob_margin
        self.inspect = inspect
        self.guard = guard
        self.rescue_alignment_weight = rescue_alignment_weight
        self.search_report = []
        self.search_windows = []
        self.event_root = None

    def translate(self, lines, **kwargs):
        # A translator can be reused for files: no lineage from the preceding one
        # may leak into the next. Width one is the actual greedy implementation.
        self.search_report = []
        self.search_windows = []
        self.event_root = None
        self.greedy_output = None
        self.candidate_output = None
        self.greedy_stats = None
        self.candidate_stats = None
        self.rescue_output = None
        self.rescue_stats = None
        self.rescue_search_report = []
        self.rescue_windows = []
        started = time.monotonic()
        if self.beam_size > 1 and self.guard:
            baseline = self.greedy_class(*self._baseline_args, **self._baseline_kwargs)
            self.greedy_output, self.greedy_stats = baseline.translate(lines, **kwargs)
        if self.beam_size > 1 and self.alignment_weight:
            ids = encode_lines(lines, self.tk, self.source_eom)
            events, _, _ = note_on_events(ids, self.tk, self.keywords)
            for event, si in zip(events, soft_indices([e['onset'] for e in events])):
                event['softIndex'] = si
            self.event_root = EventState(AlignState(events, seed_offset=0.0))
        output, stats = super().translate(lines, **kwargs)
        if self.beam_size > 1:
            stats['kv_cache'] = bool(self.search_report) and all(r['kv_cache'] for r in self.search_report)
        self.candidate_output, self.candidate_stats = output, dict(stats)
        if self.greedy_output is not None:
            def pitches(ids):
                events, _, _ = note_on_events(ids, self.tk, self.keywords)
                return [e['pitch'] for e in events]
            decision = choose_candidate(pitches(encode_lines(lines, self.tk, self.source_eom)),
                                        pitches(self.greedy_output), pitches(output),
                                        self.greedy_stats['done'], stats['done'])
            if decision['selected'] == 'greedy':
                output, stats = self.greedy_output, dict(self.greedy_stats)
                if self.rescue_alignment_weight and not self.alignment_weight:
                    rescue = type(self)(*self._baseline_args, **self._baseline_kwargs,
                                        beam_size=self.beam_size, branch_k=self.branch_k,
                                        alignment_weight=self.rescue_alignment_weight,
                                        length_alpha=self.length_alpha, logprob_margin=self.logprob_margin,
                                        inspect=self.inspect, guard=False, rescue_alignment_weight=0.)
                    self.rescue_output, self.rescue_stats = rescue.translate(lines, **kwargs)
                    self.rescue_search_report = rescue.search_report
                    self.rescue_windows = rescue.search_windows
                    rescue_decision = choose_candidate(
                        pitches(encode_lines(lines, self.tk, self.source_eom)),
                        pitches(self.greedy_output), pitches(self.rescue_output),
                        self.greedy_stats['done'], self.rescue_stats['done'])
                    decision['rescue'] = rescue_decision
                    if rescue_decision['selected'] == 'beam':
                        output, stats = self.rescue_output, dict(self.rescue_stats)
                        decision['selected'] = 'alignment'
            decision.update(greedy_seconds=self.greedy_stats['elapsed'],
                            beam_seconds=self.candidate_stats['elapsed'])
            if self.rescue_stats is not None:
                decision['rescue_seconds'] = self.rescue_stats['elapsed']
            stats['guard'] = decision
            stats['elapsed'] = time.monotonic() - started
        return output, stats

    @torch.no_grad()
    def generate(self, prefix_ids, prefix_positions, temperature, top_k, top_p,
                 n_source=None, next_position=None):
        if self.beam_size <= 1:
            return super().generate(prefix_ids, prefix_positions, temperature, top_k, top_p,
                                    n_source=n_source, next_position=next_position)
        if temperature:
            raise ValueError('beam search requires temperature=0')
        start = time.monotonic()
        seed = BranchState()
        split = n_source if n_source is not None else prefix_ids.index(self.tk.sep_id)
        for tid in prefix_ids[split:]:
            seed.feed(self.tk.tokens[tid], self.keywords)
        positions = lambda n: [(next_position if next_position is not None
                               else prefix_positions[-1] + 1) + i for i in range(n)]
        decoder = self.make_step(prefix_ids, prefix_positions, split, positions)
        record = []

        def observe(position, live, candidates, selected):
            selected_ids = {h.uid for h, _, _ in selected}
            record.append(dict(position=position, candidates=[dict(
                uid=h.uid, parent=parent, token=self.tk.tokens[tid], kept=h.uid in selected_ids,
                finished=h.finished, logprob=h.logprob, penalty=h.penalty,
                score=h.score(self.length_alpha, self.alignment_weight))
                for h, parent, tid in candidates]))

        advance = (lambda state, tid: advance_event(state, self.tk.tokens[tid], self.keywords)
                   ) if self.event_root is not None else None
        winner, forced, report = sequence_beam(
            lambda rows: F.log_softmax(decoder(rows).float(), dim=-1),
            self.tk.tokens, self.keywords, self.tk.eos_id,
            self.max_token - len(prefix_ids), beam_size=self.beam_size,
            branch_k=self.branch_k, length_alpha=self.length_alpha,
            logprob_margin=self.logprob_margin, alignment_weight=self.alignment_weight,
            state=self.event_root, advance=advance, seed=seed,
            observer=observe if self.inspect else None)
        self.event_root = winner.state
        report['kv_cache'] = self._beam_cache
        self.search_report.append(report)
        if self.inspect:
            self.search_windows.append(dict(best_uid=winner.uid, positions=record))
        self.decode_seconds += time.monotonic() - start
        self.decode_tokens += len(winner.ids)
        return list(winner.ids), forced


class SequenceBeamTranslator(SequenceBeamMixin, SlidingTranslator):
    """Decoder-only beam with batched cache selection, including sibling copies."""

    greedy_class = SlidingTranslator

    def make_step(self, prefix, prefix_positions, n_source, positions):
        cached = self.kv_cache and self.max_token <= self.model.max_seq_len
        self._beam_cache = cached
        if not cached:
            def step(rows):
                ids = torch.tensor([list(prefix) + r for r in rows], device=self.device)
                pos = torch.tensor([list(prefix_positions) + positions(len(r)) for r in rows],
                                   device=self.device)
                return self.model(ids, torch.ones_like(ids), pos)[:, -1, :]
            return step
        from transformers import DynamicCache
        cache = DynamicCache()
        previous = []
        length = len(prefix)

        def step(rows):
            nonlocal previous, length
            if not previous:
                if rows != [[]]:
                    raise ValueError('beam cache must begin with one empty continuation')
                ids = torch.tensor([prefix], device=self.device)
                pos = torch.tensor([prefix_positions], device=self.device)
                slots = torch.arange(length, device=self.device)
            else:
                parents = {tuple(row): i for i, row in enumerate(previous)}
                indices = torch.tensor([parents[tuple(row[:-1])] for row in rows],
                                       device=self.device)
                cache.batch_select_indices(indices)
                ids = torch.tensor([[row[-1]] for row in rows], device=self.device)
                pos = torch.tensor([[positions(len(row))[-1]] for row in rows], device=self.device)
                slots = torch.tensor([length], device=self.device)
                length += 1
            logits = self.model(ids, None, pos, past_key_values=cache,
                                cache_position=slots, use_cache=True)
            previous = rows
            return logits[:, -1, :]
        return step


class SequenceBeamEncDecTranslator(SequenceBeamMixin, SlidingEncDecTranslator):
    """Encoder is shared; decoder recomputation is the reference implementation."""

    greedy_class = SlidingEncDecTranslator

    def make_step(self, prefix, prefix_positions, n_source, positions):
        self._beam_cache = False
        ids = torch.tensor([prefix[:n_source]], device=self.device)
        masks = torch.ones_like(ids)
        pos = torch.tensor([prefix_positions[:n_source]], device=self.device)
        memory = self.model.encode(ids, masks, pos)

        def step(rows):
            n = len(rows)
            ids = torch.tensor([list(prefix[n_source:]) + r for r in rows], device=self.device)
            pos = torch.tensor([list(prefix_positions[n_source:]) + positions(len(r)) for r in rows],
                               device=self.device)
            return self.model.decode(memory.expand(n, *memory.shape[1:]),
                                     ids[:, -self.model.max_seq_len:], masks.expand(n, -1),
                                     pos[:, -self.model.max_seq_len:])[:, -1, :]
        return step
