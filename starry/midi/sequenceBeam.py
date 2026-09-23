"""Beam search with one objective for pruning, completion and stopping.

Alignment is an optional nonnegative event penalty. It never creates candidates,
changes grammar, or treats an unscored structural token as an inferior choice.
The decoder callback receives equal-length token rows; it may reorder a KV cache
by the parents of those rows. Observation is strictly downstream of selection.
"""

from dataclasses import dataclass, field
import math

from .beam import BranchState
from .align import token_class, elapse_value, CLS_ELAPSE


@dataclass
class Hypothesis:
    ids: tuple = ()
    logprob: float = 0.0
    penalty: float = 0.0
    grammar: BranchState = field(default_factory=BranchState)
    state: object = None
    uid: int = 0
    finished: bool = False

    def score(self, alpha, weight):
        # EOS is a scored token and must count in the denominator too.
        length = max(1, len(self.ids) + int(self.finished))
        return (self.logprob - weight * self.penalty) / length ** alpha


def sequence_beam(step, tokens, keywords, eos_id, max_new, *, beam_size=4,
                  branch_k=4, length_alpha=0.7, logprob_margin=2.0,
                  alignment_weight=0.0, state=None, advance=None, seed=None,
                  observer=None, early_stop=True):
    """Return (winner, first_eos_forced, report).

    ``advance(state, token_id)`` returns an immutable child state and a finite,
    nonnegative incremental cost. Non-events return zero. Only model top-k
    candidates within the local logprob margin are searched; EOS and EOM receive
    exactly the same score treatment as other tokens.

    Stopping uses an optimistic upper bound: future log probabilities cannot be
    positive and future penalties cannot be negative. With alpha >= 0, the best
    a live path can possibly score is its current numerator / max_new**alpha.
    The number of completed paths alone is never a stopping criterion.
    """
    if beam_size < 1 or branch_k < 1 or max_new < 0:
        raise ValueError('beam_size/branch_k must be positive and max_new nonnegative')
    if not math.isfinite(length_alpha) or length_alpha < 0:
        raise ValueError('length_alpha must be finite and nonnegative')
    if not math.isfinite(alignment_weight) or alignment_weight < 0:
        raise ValueError('alignment_weight must be finite and nonnegative')
    if logprob_margin is not None and (not math.isfinite(logprob_margin) or logprob_margin < 0):
        raise ValueError('logprob_margin must be finite and nonnegative')
    live = [Hypothesis(grammar=seed.clone() if seed else BranchState(), state=state)]
    done = []
    uid = 0
    forced = False
    classes = [token_class(t, keywords) for t in tokens]
    values = [elapse_value(t) for t in tokens]
    report = dict(positions=0, expanded=0, finished=0, bounded_stop=False)

    for position in range(max_new):
        rows = step([list(h.ids) for h in live])
        if tuple(rows.shape) != (len(live), len(tokens)):
            raise ValueError(f'logit/vocabulary mismatch: {tuple(rows.shape)}, vocab={len(tokens)}')
        # Copy to CPU once, rather than synchronizing a device for each candidate.
        rows = rows.tolist()
        candidates = []
        for parent, row in zip(live, rows):
            if any(math.isnan(x) or x > 1e-5 for x in row):
                raise ValueError('step must return nonpositive log probabilities without NaNs')
            forced_start = position == 0 and max(range(len(row)), key=row.__getitem__) == eos_id
            if forced_start:
                forced = True
            grammar = parent.grammar.grammar
            legal = [i for i, lp in enumerate(row) if math.isfinite(lp)
                     and not (position == 0 and i == eos_id)
                     and (grammar.admits(values[i]) if classes[i] == CLS_ELAPSE
                          else grammar.admits_class(classes[i]))]
            legal.sort(key=lambda i: (-row[i], i))
            if not legal:
                continue
            # Meter/header arguments can change every subsequent bar. Do not
            # collapse those decisions to argmax just because they are neither
            # pitches nor elapses. The probability margin limits confident rows.
            # The shared sliding loop masks an immediate EOS as a recovery step.
            # Preserve its greedy fallback here: searching alternatives to that
            # artificial start can turn an end_of_track into a long hallucinated
            # continuation favored by length normalization.
            for tid in legal[:1 if forced_start else branch_k]:
                if logprob_margin is not None and row[tid] < row[legal[0]] - logprob_margin:
                    continue
                child_state, cost = (advance(parent.state, tid) if advance
                                     else (parent.state, 0.0))
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError('event penalties must be finite and nonnegative')
                uid += 1
                finished = tid == eos_id
                branch = parent.grammar.clone()
                branch.feed(tokens[tid], keywords)
                child = Hypothesis(parent.ids if finished else parent.ids + (tid,),
                                   parent.logprob + row[tid], parent.penalty + cost,
                                   branch, child_state, uid, finished)
                candidates.append((child, parent.uid, tid))
        if not candidates:
            raise RuntimeError('no finite legal continuation in beam search')
        # Completed children competed with live children at this same position.
        candidates.sort(key=lambda c: (-c[0].score(length_alpha, alignment_weight), c[0].uid))
        selected = candidates[:beam_size]
        next_live = [h for h, _, _ in selected if not h.finished]
        done.extend(h for h, _, _ in selected if h.finished)
        done.sort(key=lambda h: (-h.score(length_alpha, alignment_weight), h.uid))
        done = done[:beam_size]
        report['positions'] += 1
        report['expanded'] += len(candidates)
        report['finished'] += sum(h.finished for h, _, _ in selected)
        if observer:
            observer(position, live, candidates, selected)
        live = next_live
        if not live:
            break
        if early_stop and done:
            bound = max((h.logprob - alignment_weight * h.penalty) / max_new ** length_alpha
                        for h in live)
            if done[0].score(length_alpha, alignment_weight) >= bound:
                report['bounded_stop'] = True
                break
    finalists = done + live
    winner = max(finalists, key=lambda h: (h.score(length_alpha, alignment_weight), -h.uid))
    report.update(best_uid=winner.uid, logprob=winner.logprob, penalty=winner.penalty,
                  score=winner.score(length_alpha, alignment_weight), complete=winner.finished)
    return winner, forced, report
