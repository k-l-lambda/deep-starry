"""Search optimality, structural tokens, observation and cache ancestry regressions."""

import itertools
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/midi')]

import torch

from starry.midi.sequenceBeam import sequence_beam
from starry.midi.beam import BranchState
from starry.midi.align import AlignState
from starry.midi.models.midiTranslator import MidiTranslator
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer
from sequenceBeamTranslator import SequenceBeamTranslator, EventState, advance_event, choose_candidate
from translateMidiseq2 import SlidingTranslator, keyword_tokens
from beam_quality_check import score_output


class SearchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tokens = ['<eos>', '<eom>', '<A>']

    def test_objective_matches_exhaustive_paths(self):
        def step(rows):
            return torch.log_softmax(torch.tensor([
                [0.7 * len(r), 0.8 if not r or r[-1] == 2 else -0.4, 0.3]
                for r in rows]), dim=-1)
        def advance(state, tid):
            return state, 0.6 if tid == 2 else 0.0
        best = None
        # Enumerate every completed sequence and every budget-limited prefix.
        for n in range(1, 5):
            for path in itertools.product((1, 2), repeat=n):
                score = sum(float(step([list(path[:i])])[0, tid]) for i, tid in enumerate(path))
                cost = sum(advance(None, tid)[1] for tid in path)
                if n == 4:
                    cand = ((score - cost) / n ** .7, path)
                    best = cand if best is None or cand[0] > best[0] else best
                else:
                    cand = ((score + float(step([list(path)])[0, 0]) - cost) / (n + 1) ** .7, path)
                    best = cand if best is None or cand[0] > best[0] else best
        h, _, _ = sequence_beam(step, self.tokens, set(), 0, 4, beam_size=64,
                                branch_k=3, logprob_margin=None, alignment_weight=1,
                                advance=advance)
        self.assertAlmostEqual(h.score(.7, 1), best[0], places=6)
        self.assertEqual(h.ids, best[1])

    def test_final_selection_uses_alignment_too(self):
        h, _, _ = sequence_beam(lambda rows: torch.log_softmax(torch.tensor([[-100., 2., 1.9]]), -1),
                                self.tokens, set(), 0, 1, beam_size=2, branch_k=2,
                                alignment_weight=1, advance=lambda s, t: (s, float(t == 1)))
        self.assertEqual(h.ids, (2,))

    def test_eom_keeps_lm_confidence(self):
        h, _, _ = sequence_beam(lambda rows: torch.log_softmax(torch.tensor([[-100., 0., -10.]]), -1),
                                self.tokens, set(), 0, 1, alignment_weight=1,
                                advance=lambda s, t: (s, 0.0))
        self.assertEqual(h.ids, (1,))

    def test_uncertain_meter_is_searched(self):
        tokens = ['<eos>', 'time_signature', '2', '6', '<eom>']
        seed = BranchState()
        seed.feed('time_signature', {'time_signature'})
        def step(rows):
            return torch.log_softmax(torch.tensor([
                [-20., -20., 0., .8, -20.] if not r else
                ([0., -20., -20., -20., -20.] if r[-1] == 2 else [0.] * 5)
                for r in rows]), -1)
        h, _, _ = sequence_beam(step, tokens, {'time_signature'}, 0, 2,
                                beam_size=2, branch_k=2, seed=seed, length_alpha=0)
        self.assertTrue(h.finished)
        self.assertEqual(h.ids, (2,))

    def test_finished_count_does_not_end_search(self):
        def step(rows):
            if not rows[0]:
                row = [-100., -.1, -.2]
            elif len(rows[0]) == 1:
                row = [-1., -1.1, -100.]
            else:
                row = [-100., 0., -100.]
            return torch.log_softmax(torch.tensor([row for _ in rows]), -1)
        a, _, report = sequence_beam(step, self.tokens, set(), 0, 20, beam_size=4, branch_k=2)
        b, _, _ = sequence_beam(step, self.tokens, set(), 0, 20, beam_size=4, branch_k=2,
                                early_stop=False)
        self.assertGreaterEqual(report['finished'], 2)
        self.assertEqual(len(a.ids), 20)
        self.assertEqual(a.ids, b.ids)
        self.assertAlmostEqual(a.score(.7, 0), b.score(.7, 0))

    def test_forced_start_preserves_greedy_recovery(self):
        def step(rows):
            values = []
            for row in rows:
                values.append([10., 0., -.1] if not row else
                              ([0., -20., -20.] if row[0] == 1 else [-20., -20., 0.]))
            return torch.log_softmax(torch.tensor(values), -1)
        h, forced, _ = sequence_beam(step, self.tokens, set(), 0, 20)
        self.assertTrue(forced)
        self.assertTrue(h.finished)
        self.assertEqual(h.ids, (1,))

    def test_observer_is_not_part_of_selection(self):
        step = lambda rows: torch.log_softmax(torch.tensor([[.2, .1, .0] for _ in rows]), -1)
        records = []
        a, _, ra = sequence_beam(step, self.tokens, set(), 0, 8)
        b, _, rb = sequence_beam(step, self.tokens, set(), 0, 8,
                                observer=lambda *args: records.append(args[0]))
        self.assertEqual(a.ids, b.ids)
        self.assertEqual(ra, rb)
        self.assertTrue(records)

    def test_nonnegative_penalty_contract(self):
        with self.assertRaises(ValueError):
            sequence_beam(lambda rows: torch.tensor([[-100., 0., -1.]]), self.tokens,
                          set(), 0, 1, advance=lambda s, t: (s, -1.0))


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.tk = Midiseq2Tokenizer()
        self.kw = keyword_tokens(self.tk)

    def test_cache_siblings_reorder_and_absolute_positions(self):
        model = MidiTranslator(vocab_size=self.tk.vocab_size, d_model=32, n_layer=2,
                               n_head=2, max_seq_len=128, dropout=0.0).eval()
        tr = SequenceBeamTranslator(model, self.tk, max_token=64, alignment_weight=0)
        prefix = [self.tk.bos_id, self.tk.sep_id, self.tk.bos_id]
        positions = [-900, -1, 0]
        tail = lambda n: list(range(7, 7 + n))
        fast = tr.make_step(prefix, positions, 1, tail)
        tr.kv_cache = False
        slow = tr.make_step(prefix, positions, 1, tail)
        # Expand siblings, reorder them, expand one parent twice and drop another.
        batches = [[[]], [[10], [11], [12]], [[12, 13], [10, 14], [12, 15]],
                   [[12, 15, 16], [12, 13, 17]]]
        with torch.no_grad():
            for batch in batches:
                torch.testing.assert_close(fast(batch), slow(batch), atol=1e-5, rtol=1e-5)

    def test_structural_tokens_do_not_copy_or_penalize_alignment(self):
        root = EventState(AlignState([dict(pitch=60, onset=0, softIndex=0.)], seed_offset=0.))
        state = root
        for tok in ['note_on', '#3c', '$50', '<eom>']:
            old = state
            state, cost = advance_event(state, tok, self.kw)
            if tok != '#3c':
                self.assertIs(state.align, old.align)
                self.assertEqual(cost, 0)
        self.assertEqual(root.align.matched, 0)
        self.assertEqual(state.align.matched, 1)

    def test_window_eos_does_not_erase_a_pending_output_event(self):
        root = EventState(AlignState([dict(pitch=60, onset=0, softIndex=0.)], seed_offset=0.))
        pending, _ = advance_event(root, 'note_on', self.kw)
        carried, cost = advance_event(pending, '<eos>', self.kw)
        resumed, _ = advance_event(carried, '#3c', self.kw)
        self.assertIs(pending, carried)
        self.assertEqual(cost, 0.)
        self.assertEqual(resumed.align.matched, 1)

    def test_window_advancement_is_the_greedy_policy(self):
        tr = SequenceBeamTranslator(None, self.tk, alignment_weight=0, prime_window=32)
        base = SlidingTranslator(None, self.tk, prime_window=32)
        output = [self.tk.id_by_token[t] for t in ['note_on', '#3c', '$50', '<eom>', 'E010']]
        p = tr.trim_prime(output, tr.advance_output(output, 0))
        q = base.trim_prime(output, base.advance_output(output, 0))
        self.assertEqual(p, q)
        lines = ['note_on #3c $50', 'E010', 'note_on #3d $50']
        self.assertEqual(tr.advance_source_by_onsets(lines, 0, 1),
                         base.advance_source_by_onsets(lines, 0, 1))

    def test_width_one_delegates_and_observation_does_not_change_ids(self):
        model = MidiTranslator(vocab_size=self.tk.vocab_size, d_model=32, n_layer=2,
                               n_head=2, max_seq_len=128, dropout=0.0).eval()
        prefix = [self.tk.bos_id, self.tk.sep_id, self.tk.bos_id]
        pos = [-2, -1, 0]
        args = dict(max_token=20, prime_window=16)
        base = SlidingTranslator(model, self.tk, **args)
        one = SequenceBeamTranslator(model, self.tk, beam_size=1, **args)
        self.assertEqual(base.generate(prefix, pos, 0, 0, 1, n_source=1),
                         one.generate(prefix, pos, 0, 0, 1, n_source=1))
        a = SequenceBeamTranslator(model, self.tk, alignment_weight=0, **args)
        b = SequenceBeamTranslator(model, self.tk, alignment_weight=0, inspect=True, **args)
        self.assertEqual(a.generate(prefix, pos, 0, 0, 1, n_source=1),
                         b.generate(prefix, pos, 0, 0, 1, n_source=1))

    def test_missing_output_is_not_dropped_from_evaluation(self):
        reference = ['note_on #3c $50', 'E1e0', 'note_off #3c',
                     '@measure 2', 'note_on #3e $50', 'E1e0', 'note_off #3e']
        empty = score_output([], reference)
        partial = score_output(reference[:3], reference)
        self.assertEqual(empty['onset_f1'], 0)
        self.assertAlmostEqual(partial['onset_f1'], 2 / 3)
        self.assertEqual(partial['notes_ref'], 2)

    def test_guard_tolerates_noise_and_rejects_gross_source_divergence(self):
        source = [60, 62] * 50
        good = source[:-1]
        small_difference = source[:-2]
        self.assertEqual(choose_candidate(source, good, small_difference, True, True)['selected'], 'beam')
        bad = choose_candidate(source, good, [60] * 300, True, True)
        self.assertEqual(bad['selected'], 'greedy')
        self.assertEqual(bad['reason'], 'source_fidelity')

    def test_guard_uses_an_independent_greedy_translation(self):
        tr = SequenceBeamTranslator(None, self.tk, rescue_alignment_weight=0.)
        good = [self.tk.id_by_token[t] for t in ['note_on', '#3c', '$50', 'end_of_track']]
        bad = [self.tk.id_by_token[t] for t in ['note_on', '#3e', '$50']]
        calls = []
        def translate(instance, lines, **kwargs):
            calls.append(instance)
            greedy = type(instance) is SlidingTranslator
            return (good if greedy else bad), dict(done=greedy, elapsed=.1, kv_cache=False)
        with patch.object(SlidingTranslator, 'translate', translate):
            output, stats = tr.translate(['note_on #3c $50', 'end_of_track'])
        self.assertEqual(len(calls), 2)
        self.assertIsNot(calls[0], tr)
        self.assertEqual(output, good)
        self.assertEqual(stats['guard']['reason'], 'completion')
        self.assertTrue(stats['done'])
        self.assertEqual(tr.candidate_output, bad)

    def test_alignment_rescue_is_conditional_and_cannot_recurse(self):
        good = [self.tk.id_by_token[t] for t in ['note_on', '#3c', '$50', 'end_of_track']]
        bad = [self.tk.id_by_token[t] for t in ['note_on', '#3e', '$50']]
        for primary_ok, rescue_ok in [(True, True), (False, True), (False, False)]:
            with self.subTest(primary_ok=primary_ok, rescue_ok=rescue_ok):
                tr = SequenceBeamTranslator(None, self.tk)
                calls = []
                def translate(instance, lines, **kwargs):
                    calls.append(instance)
                    is_greedy = type(instance) is SlidingTranslator
                    is_rescue = not is_greedy and instance.alignment_weight > 0
                    ok = is_greedy or (rescue_ok if is_rescue else primary_ok)
                    return (good if ok else bad), dict(done=ok, elapsed=.1, kv_cache=False)
                with patch.object(SlidingTranslator, 'translate', translate):
                    output, stats = tr.translate(['note_on #3c $50', 'end_of_track'])
                self.assertEqual(output, good)
                self.assertEqual(len(calls), 2 if primary_ok else 3)
                self.assertEqual(stats['guard']['selected'],
                                 'beam' if primary_ok else 'alignment' if rescue_ok else 'greedy')
                if not primary_ok:
                    self.assertFalse(calls[2].guard)
                    self.assertEqual(calls[2].rescue_alignment_weight, 0.)
                    self.assertIsNot(calls[2], tr)


if __name__ == '__main__':
    unittest.main()
