"""
decode.py

Turns the detector's per-frame score curve into a list of move onsets, and
scores those onsets against BLE ground truth.

Peak-picking, not thresholding
------------------------------
The old MotionGate asked "is the delta above a threshold" and used the
falling edge as a move boundary, which needs the cube to come to rest
between moves. This asks "where does the score peak", which does not. Two
moves 200ms apart produce two peaks in one continuous hump; a state
machine sees one uninterrupted MOVING run.

Refractory period
-----------------
MIN_SEP suppresses peaks closer together than N frames. At 30fps a 1-frame
gap is 33ms and a 2-frame gap is 67ms, which is the regime fast two-handed
solving actually produces — 28% of held-out onsets have a neighbour under
150ms.

MIN_SEP was 3 until 2026-07-22, picked from p1/p5 inter-move gaps measured
on the 2026-07-21 sessions alone. Those sessions were slow (2.5% of pairs
under 100ms), and against faster footage that setting was discarding turns
the model had already found: every single sub-150ms miss bucketed as
"suppressed", with a peak score of 0.80-0.98 against a 0.40 threshold. The
model saw them; the decoder threw them away. MIN_SEP=2 recovers sub-150ms
recall 73.8% -> 78.6% for 1.9 points of precision.

MIN_SEP=1 measures identically to 2, because peak_pick requires a strict
local maximum and two onsets one frame apart can never both be one. That
is this decoder's hard floor, and it is a property of peak-picking rather
than of the data.

What is actually irreducible
----------------------------
An earlier version of this note claimed sub-100ms different-face pairs are
two-handed simultaneous turns that "no temporal model at any frame rate"
can separate, and predicted a recall ceiling near 98%. Both halves were
wrong, and wrong in a way that steered work away from a recoverable
problem — the ceiling was measured on data that contained almost no fast
moves, and the impossibility claim was never tested.

Fraction of crowded (<150ms) pairs whose score curve is a single merged
hump rather than two resolvable peaks, measured on held-out sessions:

    sigma=2.0 targets    45%
    sigma=1.0 targets    34%

Most crowded pairs are separable, and how many depends on the training
target rather than on physics. At sigma=2 the Gaussian targets for two
onsets 1-2 frames apart overlap into a single blob, so the supervision
never asks for two peaks and the model cannot learn to emit them.
Sharpening to sigma=1 moved sub-150ms recall 78.6% -> 83.5% with no loss
above 150ms and no cost in timing error. See dataset.SIGMA.

A residual floor is real — 14 of the 17 remaining sub-150ms misses are on
different faces, consistent with genuinely overlapping two-handed turns —
but it is smaller than the 2% claimed above and it is not fixed. Recover
the remainder downstream regardless: a group-theoretic decode against the
scanned start and end states can insert a move the detector never saw,
which per-move scoring alone cannot.

Metrics
-------
Onset F1 with a tolerance, not frame-level accuracy. Frame-level accuracy
is dominated by the ~60% of frames that are trivially negative and would
read ~90% for a model that never fires at all.
"""

import sys
from pathlib import Path

import numpy as np

_BLE_DIR = Path(__file__).resolve().parents[1]
if str(_BLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BLE_DIR))

# Reuse the training-time window geometry rather than restating it — the
# squeeze rule for closely spaced moves lives there.
from postprocess_session import (move_window, WINDOWS,  # noqa: E402
                                 HALF_FRAME)

MIN_SEP   = 2      # frames; refractory period between accepted peaks
THRESHOLD = 0.5    # score above which a peak counts
TOLERANCE = 2      # frames; |pred - gt| this small counts as a hit (67ms)
BETA      = 2.0    # recall:precision weight when tuning THRESHOLD — see
                   # sweep_threshold for why misses cost more than extras


def peak_pick(scores: np.ndarray, threshold: float = THRESHOLD,
              min_sep: int = MIN_SEP) -> np.ndarray:
    """
    Local maxima above `threshold`, greedily thinned so no two accepted
    peaks sit within `min_sep` frames. Strongest peak wins a conflict.
    Returns frame indices, ascending.
    """
    if len(scores) < 3:
        return np.array([], dtype=int)

    # Strictly greater on the right handles plateaus without double-firing
    cand = np.where((scores[1:-1] >= scores[:-2]) &
                    (scores[1:-1] > scores[2:]) &
                    (scores[1:-1] >= threshold))[0] + 1
    if len(cand) == 0:
        return np.array([], dtype=int)

    accepted = []
    for i in cand[np.argsort(-scores[cand])]:
        if all(abs(i - j) >= min_sep for j in accepted):
            accepted.append(int(i))
    return np.array(sorted(accepted), dtype=int)


def match_onsets(pred: np.ndarray, gt: np.ndarray, scores: np.ndarray | None = None,
                 tolerance: int = TOLERANCE) -> dict:
    """
    Greedy one-to-one matching of predicted onsets to ground truth.
    Strongest predictions claim their ground truth first, so a spurious
    low-score peak cannot steal a match from a confident one.

    Also returns WHICH ground-truth onsets went unmatched ("missed_gt"),
    not just how many. Diagnosing a recall problem needs the frame indices
    so the score curve can be inspected around each miss — and deriving
    them by re-running the match elsewhere would let the diagnostic drift
    out of sync with the metric it is explaining.
    """
    order = np.argsort(-scores[pred]) if scores is not None and len(pred) \
        else np.arange(len(pred))

    unmatched = [int(g) for g in gt]
    errors, tp, matched = [], 0, []
    for p in np.asarray(pred)[order]:
        if not unmatched:
            break
        best = min(unmatched, key=lambda g: abs(g - p))
        if abs(best - p) <= tolerance:
            unmatched.remove(best)
            matched.append(int(best))
            errors.append(int(p) - int(best))
            tp += 1

    fp, fn = len(pred) - tp, len(gt) - tp
    precision = tp / len(pred) if len(pred) else 0.0
    recall    = tp / len(gt)   if len(gt)   else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "median_err": float(np.median(np.abs(errors))) if errors else float("nan"),
        "bias": float(np.mean(errors)) if errors else float("nan"),
        "matched_gt": sorted(matched),
        "missed_gt": sorted(unmatched),
    }


def fbeta(m: dict, beta: float = BETA) -> float:
    """F-beta from a match_onsets result. beta > 1 weights recall higher."""
    p, r, b2 = m["precision"], m["recall"], beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def sweep_threshold(scores: np.ndarray, gt: np.ndarray,
                    thresholds=np.arange(0.10, 0.91, 0.05),
                    min_sep: int = MIN_SEP, tolerance: int = TOLERANCE,
                    plateau: float = 0.002, beta: float = BETA
                    ) -> tuple[float, dict]:
    """
    Best-F-beta threshold on this data. Tune on val, then fix it.

    Selection is by F-beta, not F1, because the two error types are not
    equally expensive downstream. A false positive is a spurious onset the
    group-theoretic decode can DELETE — its position is known, so the fix
    is to try dropping it. A false negative is a move that was never seen,
    which the decode has to INVENT at an unknown position from ~18
    candidates. Deleting is far cheaper than inserting, so recall is worth
    more than precision here and beta=2 says so explicitly.

    On the 2026-07-21 session holdout this moves the operating point from
    0.65 to 0.50: recall 95.7% -> 97.7% (11 misses -> 6) for 7 extra false
    positives. 97.7% is essentially the ~98% ceiling this module's header
    predicts from simultaneous two-handed turns, so there is little left to
    win by pushing further — and beta=3 would pick 0.10, which buys 3 more
    onsets for 33 more false positives. That is the wrong end of the knee.

    F-beta is usually flat across a range of thresholds rather than peaked.
    Taking the first argmax would park the operating point at the edge of
    that plateau, where a small shift in score calibration on new data
    falls straight off it. So: find the best score, take every threshold
    within `plateau` of it, and return the MIDDLE of that run.
    """
    results = []
    for t in thresholds:
        pred = peak_pick(scores, threshold=float(t), min_sep=min_sep)
        results.append((float(t), match_onsets(pred, gt, scores, tolerance)))

    best = max(fbeta(m, beta) for _, m in results)
    good = [t for t, m in results if fbeta(m, beta) >= best - plateau]
    chosen = good[len(good) // 2]
    return chosen, dict(next(m for t, m in results if t == chosen))


def format_metrics(m: dict, label: str = "") -> str:
    return (f"{label:<12} F1 {m['f1']*100:5.1f}%   "
            f"P {m['precision']*100:5.1f}%  R {m['recall']*100:5.1f}%   "
            f"tp {m['tp']:4d} fp {m['fp']:3d} fn {m['fn']:3d}   "
            f"|err| {m['median_err']:.1f}f  bias {m['bias']:+.1f}f")


def align_sequences(truth: list[str], pred: list[str]) -> list[tuple]:
    """
    Needleman-Wunsch alignment of the predicted sequence against the
    prescribed one. Returns [(op, truth_move, pred_move), ...] where op is
    one of "ok", "sub", "miss", "phantom".

    Alignment rather than index-by-index comparison, because a single
    missed move shifts every later index — position i of the prediction
    stops corresponding to position i of the truth. Compared positionally,
    one dropped move at the start reads as ~25 errors instead of 1, and the
    whole measurement becomes noise.

    The three non-"ok" ops map onto the three failure modes:
      miss     detector never fired          -> recall problem
      phantom  detector fired spuriously     -> precision problem
      sub      detected, named wrong         -> classifier problem
    """
    n, m = len(truth), len(pred)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(
                dp[i - 1, j - 1] + (truth[i - 1] != pred[j - 1]),
                dp[i - 1, j] + 1,      # truth move absent from pred = miss
                dp[i, j - 1] + 1,      # extra pred move = phantom
            )

    ops: list[tuple] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and
                dp[i, j] == dp[i - 1, j - 1] + (truth[i - 1] != pred[j - 1])):
            same = truth[i - 1] == pred[j - 1]
            ops.append(("ok" if same else "sub", truth[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("miss", truth[i - 1], None))
            i -= 1
        else:
            ops.append(("phantom", None, pred[j - 1]))
            j -= 1
    return ops[::-1]


# ---------------------------------------------------------------------------
# Handoff to the move classifier
# ---------------------------------------------------------------------------

def window_from_anchor(anchor: int, offsets: dict, fps: float, n_frames: int,
                       frame_times: np.ndarray | None = None
                       ) -> dict[str, int]:
    """
    The 5 frame indices for one move, given its anchor frame and the window
    offsets in seconds.

    Two ways to answer, and they are not the same answer:

    `frame_times` given (PREFERRED) — pick the frame NEAREST IN TIME to
        anchor_ts + offset, constrained to the move's sandwich. This is
        postprocess_session's own rule, so the window is selected the way
        the classifier's training windows were selected.

    `frame_times` None (legacy) — index arithmetic, round(anchor +
        offset*fps), against one global average fps.

    WHICH frames land in the window genuinely matters, and window_audit.py
    measured how much on 2456 moves (2026-07-26). Grouping moves by how many
    of the 5 slots differ from the trainer's own choice, against the SAME
    moves scored on the trainer's window:

        differing   0      1      2      3      4      5
        trainer   98.9%  98.6%  99.5%  98.2%  97.6% 100.0%
        inference 98.9%  97.8%  95.8%  95.8%  91.9%  91.9%

    Dose-response, and the trainer row is flat — so these are not harder
    moves, the window is what costs the accuracy.

    What this timestamp path does NOT do is close that gap. It raises exact
    agreement with the trainer's window from 11% of moves to 33% (mean slots
    differing 1.78 -> 1.38) and buys +0.3 points, which is not significant
    (McNemar p=0.40, and the end-to-end rung moved -0.4, p=0.31). It was
    predicted to be worth ~2 points and it is not; keep it because it is the
    construction training used and because it is a prerequisite for
    sub-frame anchoring, not because it measured as a win.

    The residual is the ANCHOR, and it is irreducible without a change to the
    detector: training anchored on the BLE timestamp itself, while inference
    only knows which FRAME the onset peaked on. That quantises the anchor by
    up to half a frame, which flips the nearest-frame choice for any target
    sitting near a midpoint — and the mid offsets (+30/+60/+100ms) do sit
    near midpoints at 30fps. Closing it needs either a sub-frame onset
    regression head, or a classifier trained to be insensitive to the shift.
    """
    if frame_times is None:
        return {key: int(np.clip(round(anchor + offsets[key] * fps),
                                 0, n_frames - 1))
                for key in WINDOWS}

    ts = frame_times
    t0 = float(ts[anchor])
    lo_ts = t0 + offsets["before"] - HALF_FRAME
    hi_ts = t0 + offsets["after"] + HALF_FRAME
    # Frames inside this move's sandwich. postprocess_session dropped a move
    # whose window had no frame in range; inference cannot drop a detected
    # onset (that is a lost move), so it falls back to the nearest frame.
    lo_i = int(np.searchsorted(ts, lo_ts, side="left"))
    hi_i = int(np.searchsorted(ts, hi_ts, side="right")) - 1

    out = {}
    for key in WINDOWS:
        target = t0 + offsets[key]
        i = int(np.searchsorted(ts, target))
        if i > 0 and (i >= len(ts) or
                      abs(ts[i - 1] - target) < abs(ts[i] - target)):
            i -= 1
        i = int(np.clip(i, 0, n_frames - 1))
        if lo_i <= hi_i:
            i = int(np.clip(i, lo_i, min(hi_i, n_frames - 1)))
        out[key] = i
    return out


def onset_windows(onsets: np.ndarray, fps: float, n_frames: int,
                  frame_times: np.ndarray | None = None
                  ) -> list[dict[str, int]]:
    """
    Frame indices of the 5-frame window the move classifier expects, for
    every detected onset: {before, mid_00, mid_01, mid_02, after}.

    Geometry comes from postprocess_session.move_window(), the same
    function that carved the training windows — including its rule that a
    move crowded by its neighbors gets a squeezed window. Applying that
    rule here too keeps inference windows distributed like training ones;
    a fixed window would bleed a neighboring move's motion into the frames
    the classifier is told are STILL.

    Neighbor gaps come from the DETECTED onsets, since at inference there
    are no BLE timestamps to measure against. That substitution is not free
    and it does not fail where you would expect: a MISSED move makes the gap
    between the two moves either side of it look twice as long, so both of
    them skip a squeeze training would have applied and take in the motion
    of the turn between them. Measured over 2456 moves, 4% of windows came
    out >25% wider than their training counterpart for this reason, and the
    classifier scored 72.5% on them against 90.1% with the true gaps. The
    detector's error lands on its NEIGHBOURS' inputs, which is why no
    per-onset metric can see it.

    Pass `frame_times` (seconds, one per frame, ascending) whenever they are
    available — see window_from_anchor for what it buys.
    """
    ts = None if frame_times is None else np.asarray(frame_times, dtype=float)
    out = []
    for i, o in enumerate(onsets):
        # Gaps in real time when timestamps are known, frames/fps otherwise.
        if ts is None:
            gap_prev = (o - onsets[i - 1]) / fps if i > 0 else None
            gap_next = (onsets[i + 1] - o) / fps if i < len(onsets) - 1 else None
        else:
            gap_prev = float(ts[o] - ts[onsets[i - 1]]) if i > 0 else None
            gap_next = float(ts[onsets[i + 1]] - ts[o]) \
                if i < len(onsets) - 1 else None
        offsets = move_window(gap_prev, gap_next)
        out.append(window_from_anchor(int(o), offsets, fps, n_frames, ts))
    return out
