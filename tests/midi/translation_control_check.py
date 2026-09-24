"""Online stop safety, committed-primer ancestry and conservative source progress."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/midi')]
from starry.midi.translationControl import TranslationControl
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer
from starry.midi.align import soft_indices
from sequenceBeamTranslator import SequenceBeamTranslator, choose_candidate
from translateMidiseq2 import encode_lines, note_on_events, keyword_tokens


def source(n=100, pitch=60):
    result = [dict(pitch=pitch, onset=i * 120) for i in range(n)]
    for e, si in zip(result, soft_indices([e['onset'] for e in result])):
        e['softIndex'] = si
    return result


def event(i, pitch=60):
    return dict(pitch=pitch, onset=i * 120, pitch_index=i * 10)


class ControlTests(unittest.TestCase):
    def make(self, src=None, **kw):
        src = source() if src is None else src
        return TranslationControl(src, list(range(len(src))), window=8, **kw)

    def test_transient_misses_recover_without_stop(self):
        c = self.make()
        for i in range(8): c.observe(event(i, 20))
        for i in range(8, 16): c.observe(event(i, 60))
        for i in range(16, 24): c.observe(event(i, 20))
        self.assertIsNone(c.stop)
        for i in range(24, 32): c.observe(event(i, 20))
        self.assertEqual(c.stop['reason'], 'quality_miss')
        self.assertEqual(c.stop['notes'], 32)

    def test_alignment_reuse_alone_must_abstain(self):
        c = self.make()
        c.align = Mock(observe=Mock(return_value=dict(src=0, self_cost=0.)))
        for i in range(32): c.observe(event(i))
        self.assertIsNone(c.stop)
        self.assertEqual(c.blocks[-1]['fresh_rate'], 0.)
        self.assertEqual(c.blocks[-1]['excess_rate'], 0.)

    def test_lost_alignment_with_valid_pitch_inventory_must_abstain(self):
        c = self.make()
        c.align = Mock(observe=Mock(return_value=dict(src=None, self_cost=None)))
        for i in range(32): c.observe(event(i))
        self.assertIsNone(c.stop)
        self.assertEqual(c.blocks[-1]['miss_rate'], 1.)

    def test_reuse_with_exhausted_pitch_inventory_stops(self):
        src = source(pitch=61)
        src[0]['pitch'] = 60
        c = self.make(src)
        c.align = Mock(observe=Mock(return_value=dict(src=0, self_cost=0.)))
        for i in range(16): c.observe(event(i))
        self.assertEqual(c.stop['reason'], 'quality_reuse')

    def test_count_budget_tolerates_noise_but_bounds_overgeneration(self):
        c = self.make()
        c.align = Mock(observe=Mock(return_value=dict(src=0, self_cost=0.)))
        for i in range(120): c.observe(event(i))
        self.assertIsNone(c.stop)
        for i in range(120, 151): c.observe(event(i))
        self.assertIsNotNone(c.stop)
        off = self.make(quality=False)
        for i in range(160): off.observe(event(i, 20))
        self.assertIsNone(off.stop)

    def ledger(self, indices):
        c = self.make(quality=False)
        c.align = Mock(observe=Mock(side_effect=[dict(src=i, self_cost=0.) for i in indices]))
        for i in range(len(indices)): c.observe(event(i))
        return c

    def test_one_far_match_cannot_skip_source(self):
        c = self.ledger([0, 1, 80, 2, 3])
        cursor, stop = c.advance(0, 50, 100)
        self.assertGreater(cursor, 0)
        self.assertLessEqual(cursor, 4)
        self.assertIsNone(stop)

    def test_invisible_matches_cannot_drive_source(self):
        c = self.ledger([70, 71, 72])
        cursor, _ = c.advance(0, 30, 10)
        self.assertEqual(cursor, 0)

    def test_old_primer_records_survive_and_are_observed_once(self):
        c = self.ledger([0, 1, 2, 3, 4, 5])
        self.assertEqual(c.advance(0, 10, 100)[0], 0)
        cursor, _ = c.advance(0, 40, 100)
        self.assertGreater(cursor, 0)
        self.assertEqual(c.align.observe.call_count, 6)
        self.assertEqual(c.retired, 4)
        self.assertEqual(len(c.records), 2)

    def test_unretired_chord_note_holds_frontier(self):
        c = self.ledger([1, 2, 3, 0])
        self.assertEqual(c.advance(0, 30, 100)[0], 0)
        self.assertGreater(c.advance(0, 40, 100)[0], 0)

    def test_stall_stops_without_skipping_input(self):
        c = self.make(quality=False)
        for _ in range(2):
            self.assertEqual(c.advance(0, 0, 50), (0, None))
        cursor, stop = c.advance(0, 0, 50)
        self.assertEqual(cursor, 0)
        self.assertEqual(stop['reason'], 'alignment_stall')

    def test_empty_source_does_not_accuse_quality(self):
        c = self.make([])
        for i in range(24): c.observe(event(i))
        self.assertIsNone(c.stop)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tk = Midiseq2Tokenizer()
        self.kw = keyword_tokens(self.tk)

    def scripted(self, bodies, **kw):
        tk = self.tk
        class Scripted(SequenceBeamTranslator):
            def generate(instance, *args, **kwargs):
                value = bodies[min(instance.calls, len(bodies) - 1)]
                instance.calls += 1
                return encode_lines(value, tk, True), False
        tr = Scripted(None, tk, guard=False, quality_window=8, **kw)
        tr.calls = 0
        return tr

    def test_quality_stop_precedes_eot_and_preserves_complete_bars(self):
        bad = ['note_on #14 $50', 'E078'] * 8 + ['@measure 2']
        tr = self.scripted([bad + bad + ['end_of_track']])
        ids, stats = tr.translate(['note_on #3c $50', 'end_of_track'])
        self.assertEqual(tr.calls, 1)
        self.assertFalse(stats['done'])
        self.assertEqual(stats['early_stop']['reason'], 'quality_miss')
        self.assertEqual(ids[-1], self.tk.eom_id)
        self.assertNotIn(self.tk.id_by_token['end_of_track'], ids)

    def test_ignored_post_eot_junk_cannot_trigger_quality_stop(self):
        tr = self.scripted([['note_on #3c $50', 'end_of_track'] + ['note_on #14 $50'] * 24])
        ids, stats = tr.translate(['note_on #3c $50', 'end_of_track'])
        self.assertTrue(stats['done'])
        self.assertNotIn('early_stop', stats)
        self.assertEqual(stats['control']['observed'], 1)

    def test_quality_stop_avoids_decoding_the_next_window(self):
        bad = ['note_on #14 $50', 'E078'] * 8 + ['@measure 2']
        frames = [bad + bad, ['note_on #3c $50', 'end_of_track']]
        lines = ['note_on #3c $50', 'E078'] * 100 + ['end_of_track']
        stopped = self.scripted(frames)
        _, stats = stopped.translate(lines)
        self.assertEqual(stopped.calls, 1)
        self.assertFalse(stats['done'])
        unchecked = self.scripted(frames, quality_stop=False)
        _, stats = unchecked.translate(lines)
        self.assertEqual(unchecked.calls, 2)
        self.assertTrue(stats['done'])

    def test_chunked_event_and_absolute_tick_survive_window_boundary(self):
        tr = self.scripted([['E120', 'note_on'], ['#3c $50']], src_window=4)
        tr.translate(['note_on #3c $50', 'E120'] * 10, max_steps=2)
        self.assertEqual(tr.control.observed, 1)
        self.assertEqual(tr.control.previous_onset, 0x120)

    def test_zero_target_onsets_do_not_skip_a_source_window_with_notes(self):
        tr = self.scripted([['time_signature 4 2']], align_advance=True, quality_stop=False)
        _, stats = tr.translate(['note_on #3c $50', 'E120'] * 10, max_steps=5)
        self.assertEqual(stats['early_stop']['reason'], 'alignment_stall')
        self.assertEqual(stats['consumed_lines'], 0)
        self.assertEqual(tr.calls, 3)

    def test_metadata_only_source_can_progress(self):
        tr = self.scripted([['time_signature 4 2']], src_window=1,
                           align_advance=True, quality_stop=False)
        _, stats = tr.translate(['E120', 'E120', 'note_on #3c $50'], max_steps=2)
        self.assertEqual(stats['consumed_lines'], 2)
        self.assertNotIn('early_stop', stats)

    def test_controller_is_reset_between_files(self):
        tr = self.scripted([['note_on #3c $50', 'end_of_track']])
        _, first = tr.translate(['note_on #3c $50', 'end_of_track'])
        _, second = tr.translate(['note_on #3c $50', 'end_of_track'])
        self.assertEqual(first['control']['observed'], second['control']['observed'])
        self.assertEqual(second['control']['observed'], 1)

    def test_stopped_candidate_never_passes_guard_on_pitch_f1_alone(self):
        d = choose_candidate([60], [60], [60], False, False,
                             beam_stopped=True, rescue_stopped=True)
        self.assertEqual(d['selected'], 'beam')
        self.assertEqual(d['reason'], 'rescue_early_stop')


if __name__ == '__main__':
    unittest.main()
