"""Online control of committed MIDI output, independent of beam branch scoring.

Every generated onset is aligned once. Its global output index stays in a ledger
until retired, so source advancement cannot lose old-primer events or follow the
end of a still-live primer. Quality decisions use disjoint blocks, not cumulative
alignment loss or many overlapping votes on the same bad notes.
"""

from collections import Counter, deque
import math

from .align import AlignState, soft_delta


class TranslationControl:
    def __init__(self, source, source_lines, *, quality=True, window=32, patience=2,
                 miss_rate=.8, fresh_rate=.25, stall_windows=3):
        if window < 8 or patience < 2 or stall_windows < 1:
            raise ValueError('quality window >= 8, patience >= 2 and stall windows >= 1 required')
        if not .5 <= miss_rate <= 1 or not 0 <= fresh_rate <= .5:
            raise ValueError('miss rate must be in [.5, 1], fresh rate in [0, .5]')
        if len(source) != len(source_lines):
            raise ValueError('source event/line mapping mismatch')
        self.source, self.source_lines = source, source_lines
        self.source_pitches = Counter(e['pitch'] for e in source)
        self.output_pitches = Counter()
        self.align = AlignState(source, seed_offset=0.)
        self.quality = quality
        self.window, self.patience = window, patience
        self.miss_rate, self.fresh_rate = miss_rate, fresh_rate
        self.stall_windows = stall_windows
        self.previous_onset = None
        self.softindex = 0.
        self.records = deque()
        self.anchors = deque(maxlen=8)
        self.used = set()
        self.block = []
        self.bad_runs = dict(miss=0, reuse=0)
        self.observed = self.retired = 0
        self.stalls = self.stall_notes = self.empty_steps = 0
        self.new_notes = 0
        self.stop = None
        self.blocks = []
        self.steps = []

    def observe(self, event):
        if self.previous_onset is not None:
            self.softindex += soft_delta(event['onset'] - self.previous_onset)
        self.previous_onset = event['onset']
        detail = self.align.observe(event['pitch'], event['onset'], self.softindex)
        src = detail['src']
        reliable = src is not None and detail['self_cost'] <= .5
        fresh = reliable and src not in self.used
        if reliable:
            self.used.add(src)
        self.records.append(dict(index=event['pitch_index'], src=src, reliable=reliable))
        self.observed += 1
        self.new_notes += 1
        pitch = event['pitch']
        self.output_pitches[pitch] += 1
        count = self.source_pitches[pitch]
        allowance = max(2, math.ceil(count * .25)) if count else 0
        excess = self.output_pitches[pitch] > count + allowance
        self.block.append((src is None, reliable, fresh, excess))
        if len(self.block) == self.window:
            misses = sum(x[0] for x in self.block) / self.window
            matched = sum(x[1] for x in self.block)
            fresh_share = sum(x[2] for x in self.block) / matched if matched else 1.
            excess_share = sum(x[3] for x in self.block) / self.window
            # Alignment alone is not a quality oracle: real repeats can map to
            # old notes, and long rubato passages can lose the anchor. Require
            # independent pitch-inventory evidence as well, with augmentation
            # slack. Otherwise record the suspicion but abstain from stopping.
            corroborated = excess_share >= .5
            bad = dict(miss=corroborated and misses >= self.miss_rate,
                       reuse=corroborated and matched >= self.window // 2
                       and fresh_share < self.fresh_rate)
            for key, value in bad.items():
                self.bad_runs[key] = self.bad_runs[key] + 1 if value else 0
            self.blocks.append(dict(notes=self.observed, miss_rate=misses,
                                    reliable=matched, fresh_rate=fresh_share,
                                    excess_rate=excess_share))
            self.block.clear()
            if self.quality and self.source and self.stop is None:
                cause = next((k for k in bad if self.bad_runs[k] >= self.patience), None)
                if cause:
                    self.stop = dict(reason='quality_' + cause, index=event['pitch_index'],
                                     notes=self.observed, blocks=self.patience,
                                     miss_rate=misses, fresh_rate=fresh_share,
                                     excess_rate=excess_share)
        # A very generous source-count budget also catches loops whose aligner
        # keeps finding different same-pitch notes. This is not an exact count
        # constraint: augmentation can legitimately insert/delete source notes.
        limit = max(len(self.source) * 1.5, len(self.source) + 2 * self.window)
        if self.quality and self.source and self.stop is None and self.observed > limit:
            self.stop = dict(reason='quality_note_budget', index=event['pitch_index'],
                             notes=self.observed, source_notes=len(self.source), limit=limit)

    def advance(self, cursor, end, visible_end):
        """Retire output < end; return a conservative source line and stop reason.

        Three distinct reliable matches must support the frontier. The third
        largest of the last eight anchors rejects isolated forward jumps and
        tolerates local chord reordering. Unretired matched notes can hold this
        frontier back, never push it forward. Never jump beyond the shown source.
        """
        retired = 0
        while self.records and self.records[0]['index'] < end:
            record = self.records.popleft()
            retired += 1
            src = record['src']
            if record['reliable'] and self.source_lines[src] < visible_end:
                if src in self.anchors:
                    self.anchors.remove(src)
                self.anchors.append(src)
        self.retired += retired
        proposed = cursor
        if len(self.anchors) >= 3:
            frontier = sorted(self.anchors)[-3]
            # Keep source notes represented by the still-live primer, including
            # chord inversions where output order differs from source order.
            pending = [r['src'] for r in self.records if r['reliable']
                       and self.source_lines[r['src']] >= cursor]
            if pending:
                frontier = min(frontier, min(pending) - 1)
            if frontier >= 0:
                proposed = max(cursor, min(visible_end, self.source_lines[frontier] + 1))
        if proposed > cursor:
            self.stalls = self.stall_notes = self.empty_steps = 0
        else:
            self.stalls += 1
            self.stall_notes += retired
            self.empty_steps = self.empty_steps + 1 if not self.new_notes else 0
        self.steps.append(dict(before=cursor, after=proposed, visible_end=visible_end,
                               retired=retired, new_notes=self.new_notes,
                               anchors=list(self.anchors), stalled_windows=self.stalls))
        self.new_notes = 0
        stop = None
        if (self.stalls >= self.stall_windows and self.stall_notes >= self.window) or (
                self.empty_steps >= self.stall_windows):
            stop = dict(reason='alignment_stall', source_line=cursor,
                        windows=self.stalls, retired_notes=self.stall_notes)
        return proposed, stop

    def report(self):
        return dict(observed=self.observed, retired=self.retired, distinct=len(self.used),
                    blocks=self.blocks, steps=self.steps, stop=self.stop)
