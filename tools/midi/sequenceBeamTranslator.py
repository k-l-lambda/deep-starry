"""Shared-window translators backed by the additive sequence beam search."""

from dataclasses import dataclass, replace
from collections import Counter
from bisect import bisect_right
import math
import time

import torch
import torch.nn.functional as F

from starry.midi.align import AlignState, soft_delta, soft_indices
from starry.midi.beam import BranchState
from starry.midi.sequenceBeam import sequence_beam
from starry.midi.translationControl import TranslationControl
from translateMidiseq2 import (SlidingTranslator, SlidingEncDecTranslator,
                             encode_lines, note_on_events, line_token_offsets)


def choose_candidate(source, beam, rescue, beam_done, rescue_done, margin=0.05,
                     beam_stopped=False, rescue_stopped=False):
    """Only a complete, unstopped beam retry may replace the primary beam.

    A truncated primary is a valid partial result, never a reason to return an
    unprotected greedy stream. Input-pitch fidelity is a secondary guard, not
    evidence that a stopped candidate is complete.
    """
    source = Counter(source)
    def fidelity(pitches):
        pitches = Counter(pitches)
        size = sum(source.values()) + sum(pitches.values())
        return 2.0 * sum((source & pitches).values()) / size if size else 1.0
    bs, rs = fidelity(beam), fidelity(rescue)
    reason = ('primary_complete' if beam_done and not beam_stopped else
              'rescue_early_stop' if rescue_stopped else
              'rescue_incomplete' if not rescue_done else
              'source_fidelity' if rs < bs - margin else None)
    return dict(selected='beam' if reason else 'alignment', reason=reason,
                beam_source_f1=bs, rescue_source_f1=rs, margin=margin,
                beam_done=beam_done, rescue_done=rescue_done)


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
    """Shared sliding loop, with optional committed-output quality/source control."""

    def __init__(self, *args, beam_size=4, branch_k=4, alignment_weight=0.0,
                 length_alpha=0.7, logprob_margin=2.0, inspect=False, guard=True,
                 rescue_alignment_weight=0.5, align_advance=False, quality_stop=True,
                 quality_window=32, quality_patience=2, quality_miss_rate=.8,
                 quality_fresh_rate=.25, align_stall_windows=3, **kwargs):
        self._translator_args = args
        self._translator_kwargs = dict(kwargs)
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
        # Each beam attempt has its own quality controller and window history.
        self.control_align_advance = align_advance
        self.control_options = dict(quality=quality_stop, window=quality_window,
                                    patience=quality_patience, miss_rate=quality_miss_rate,
                                    fresh_rate=quality_fresh_rate, stall_windows=align_stall_windows)
        TranslationControl([], [], **self.control_options)  # validate before inference
        self.control = None

    def check_generated_output(self, output, out_base, src_ids):
        if self.control is None:
            return None
        # The shared loop will discard tokens after an honored end_of_track.
        # Such tokens must not accuse an otherwise valid output of bad quality.
        eot = self.tk.id_by_token.get('end_of_track')
        end = len(output)
        if eot in src_ids and eot in output[out_base:]:
            end = out_base + output[out_base:].index(eot) + 1
        self._control_boundary = max((i + 1 for i in range(out_base, end)
                                      if output[i] == self.tk.eom_id),
                                     default=self._control_boundary)
        events, self._control_tick, self._control_walk = note_on_events(
            output[out_base:end], self.tk, self.keywords, tick0=self._control_tick,
            state=self._control_walk, index0=out_base)
        for event in events:
            self.control.observe(event)
            if self.control.stop:
                # Keep complete bars only; never synthesize a successful EOT.
                cut = max((i + 1 for i, tid in enumerate(output[:event['pitch_index']])
                           if tid == self.tk.eom_id), default=0)
                self.control.stop.update(cut=cut, generated_tokens=len(output),
                                         detection='window_end')
                return dict(self.control.stop)
        return None

    def advance_window_source(self, lines, cursor, rolled, src_events, src_line_of,
                              index0, next_cursor):
        if not self.control_align_advance or self.control is None:
            return super().advance_window_source(lines, cursor, rolled, src_events,
                                                  src_line_of, index0, next_cursor)
        # Windows containing only metadata/rests have no note correspondence to
        # preserve. Consuming them is safe; zero TARGET onsets alone is not.
        visible_notes = any(cursor <= line < next_cursor for line in self.control.source_lines)
        new_cursor, stop = self.control.advance(cursor, index0 + len(rolled), next_cursor)
        if not visible_notes:
            self.control.stalls = self.control.stall_notes = self.control.empty_steps = 0
            self.control.steps[-1].update(after=max(cursor, next_cursor), mode='non_note_source')
            return max(cursor, next_cursor), None
        if stop:
            stop['cut'] = self._control_boundary
            self.control.stop = stop
        return new_cursor, stop

    def translate(self, lines, **kwargs):
        # A translator can be reused for files: no lineage from the preceding one
        # may leak into the next. Width one is the actual greedy implementation.
        self.search_report = []
        self.search_windows = []
        self.event_root = None
        self.control = None
        self._control_tick, self._control_walk = 0, None
        self._control_boundary = 0
        self.decode_seconds, self.decode_tokens = 0., 0
        self.greedy_output = None
        self.candidate_output = None
        self.greedy_stats = None
        self.candidate_stats = None
        self.rescue_output = None
        self.rescue_stats = None
        self.rescue_search_report = []
        self.rescue_windows = []
        started = time.monotonic()
        if self.control_align_advance or self.control_options['quality']:
            ids = encode_lines(lines, self.tk, self.source_eom)
            events, _, _ = note_on_events(ids, self.tk, self.keywords)
            offsets = line_token_offsets(lines, self.tk, self.source_eom)
            source_lines = [bisect_right(offsets, e['pitch_index']) - 1 for e in events]
            for event, si in zip(events, soft_indices([e['onset'] for e in events])):
                event['softIndex'] = si
            self.control = TranslationControl(events, source_lines, **self.control_options)
        if self.beam_size > 1 and self.alignment_weight:
            ids = encode_lines(lines, self.tk, self.source_eom)
            events, _, _ = note_on_events(ids, self.tk, self.keywords)
            for event, si in zip(events, soft_indices([e['onset'] for e in events])):
                event['softIndex'] = si
            self.event_root = EventState(AlignState(events, seed_offset=0.0))
        output, stats = super().translate(lines, **kwargs)
        if self.control is not None:
            stats['control'] = self.control.report()
        if self.beam_size > 1:
            stats['kv_cache'] = bool(self.search_report) and all(r['kv_cache'] for r in self.search_report)
        self.candidate_output, self.candidate_stats = output, dict(stats)
        if self.beam_size > 1:
            primary_reason = ('early_stop' if stats.get('early_stop') else
                              'incomplete' if not stats['done'] else None)
            decision = dict(selected='beam', reason=primary_reason,
                            policy='beam_only', beam_seconds=self.candidate_stats['elapsed'])
            if (primary_reason and self.guard and self.rescue_alignment_weight
                    and not self.alignment_weight):
                rescue = type(self)(*self._translator_args, **self._translator_kwargs,
                                    beam_size=self.beam_size, branch_k=self.branch_k,
                                    alignment_weight=self.rescue_alignment_weight,
                                    length_alpha=self.length_alpha, logprob_margin=self.logprob_margin,
                                    inspect=self.inspect, guard=False, rescue_alignment_weight=0.,
                                    align_advance=self.control_align_advance,
                                    quality_stop=self.control_options['quality'],
                                    quality_window=self.control_options['window'],
                                    quality_patience=self.control_options['patience'],
                                    quality_miss_rate=self.control_options['miss_rate'],
                                    quality_fresh_rate=self.control_options['fresh_rate'],
                                    align_stall_windows=self.control_options['stall_windows'])
                self.rescue_output, self.rescue_stats = rescue.translate(lines, **kwargs)
                self.rescue_search_report = rescue.search_report
                self.rescue_windows = rescue.search_windows
                def pitches(ids):
                    events, _, _ = note_on_events(ids, self.tk, self.keywords)
                    return [e['pitch'] for e in events]
                retry = choose_candidate(
                    pitches(encode_lines(lines, self.tk, self.source_eom)),
                    pitches(output), pitches(self.rescue_output),
                    stats['done'], self.rescue_stats['done'],
                    beam_stopped=bool(stats.get('early_stop')),
                    rescue_stopped=bool(self.rescue_stats.get('early_stop')))
                decision['rescue'] = retry
                decision['rescue_seconds'] = self.rescue_stats['elapsed']
                if retry['selected'] == 'alignment':
                    output, stats = self.rescue_output, dict(self.rescue_stats)
                    decision['selected'] = 'alignment'
            # Copy: aggregate timing/selection must not mutate candidate stats.
            stats = dict(stats)
            decision['partial'] = not stats['done'] or bool(stats.get('early_stop'))
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
