"""
coach/moves.py — L1 analytics that read move IDENTITY but not move ORDER.

Companion to `timing.py`. Everything here is a share or a count over the
whole move list — deliberately, because that is the shape measured to
survive our decode error. `metric_robustness.py` (2026-08-06, both seeds)
found that what predicts an analytic's accuracy is not whether it needs
move identity but what KIND of statistic it is: aggregates over every move
land at 1-6% error even when they read identity, while counts of
thresholded events land at 25-27% even when they read only timing.

So the rule this module follows: **every function here returns a sum, a
mean or a share.** Nothing counts events defined by a threshold, takes a
max, or divides one noisy estimate by another. Anything tempting in those
shapes was measured and rejected — see TODO §7C.

The one deliberate exception is `distinct_face_runs`, which is an
adjacent-pair (`local`) statistic and would normally be suspect. It is
kept because it measured 1.8-2.7% worst-case on both seeds — the only
local-kind metric that did — and because it is the closest thing available
to a regrip count. It is flagged `provisional` in the report so a future
corpus can retire it if that number was luck.

UNITS: everything is QTM. A half turn is two moves.
"""

from __future__ import annotations

import numpy as np

#: Canonical face order. Matches the U/D/L/R/F/B convention used by the
#: solver and the 54-facelet string, not the decoder's class order.
FACES = "UDLRFB"

#: R and U are the faces a right-handed CFOP solver executes with finger
#: tricks and no regrip. A high share here is the signature of efficient
#: execution; heavy D/B use usually means an F2L pair was solved the hard
#: way or a rotation went untracked. This is the most coaching-legible
#: thing in the module.
EASY_FACES = "RU"
#: The two faces that cannot be reached without a regrip or a rotation.
#: Their combined share was BUILT AND REJECTED (2026-08-06, both seeds):
#: 5.6% median error daytime but 30.5% worst-session, and 26-42% evening
#: with a 100% worst case. The cause is instructive and generalises — D+B
#: is only ~6% of moves, and a share whose numerator is a small fraction
#: of the total does NOT inherit the robustness of a mean, because there
#: is nothing for the errors to average against. **A share needs a big
#: denominator to be safe.** Kept as a constant because `face_share`
#: exposes the same information safely per-face; do not resurrect the
#: combined fraction without re-measuring it.
AWKWARD_FACES = "DB"


def move_report(words: list[str]) -> dict:
    """
    WCA move words (e.g. ["R", "U'", "F", ...]) -> identity aggregates.

    `words` must be quarter turns from the decoder's 12-class alphabet.
    Returns {} for an empty list rather than raising, so a failed decode
    degrades to "no move analysis" instead of taking the report down.
    """
    if not words:
        return {}

    n = len(words)
    faces = [w[0] for w in words]
    counts = {f: faces.count(f) for f in FACES}
    share = {f: counts[f] / n for f in FACES}

    top_face = max(share, key=share.get)
    ccw = sum(w.endswith("'") for w in words)

    # Adjacent-pair structure. See the module docstring for why only this
    # one survives from the `local` family.
    runs = 1 + sum(a[0] != b[0] for a, b in zip(faces, faces[1:]))

    return {
        # -- volume ---------------------------------------------------------
        "n_moves_qtm": n,

        # -- turn direction -------------------------------------------------
        #: Counter-clockwise share. A solver far from 0.5 is avoiding one
        #: direction, which is almost always a finger-trick gap rather than
        #: a deliberate choice.
        "ccw_fraction": round(ccw / n, 4),
        "cw_fraction": round((n - ccw) / n, 4),

        # -- face usage -----------------------------------------------------
        "face_counts": counts,
        "face_share": {f: round(v, 4) for f, v in share.items()},
        "top_face": top_face,
        "top_face_share": round(share[top_face], 4),
        #: The headline of this module: how much of the solve stayed on the
        #: two faces that need no regrip.
        "easy_face_fraction": round(
            sum(share[f] for f in EASY_FACES), 4),
        #: Spread of face usage, as normalised entropy over the 6 faces.
        #: 1.0 is perfectly even, 0 is one face only. Reported instead of a
        #: distance-from-uniform because uniform is NOT the target — real
        #: CFOP is correctly R/U-heavy, and a metric implying otherwise
        #: would coach people away from good technique. This is a
        #: descriptive number for trend views, not a score.
        "face_entropy": round(_norm_entropy(list(share.values())), 4),

        # -- execution shape (provisional, see docstring) -------------------
        "distinct_face_runs": round(runs / n, 4),
        "moves_per_face_run": round(n / runs, 3),
    }


def _norm_entropy(p: list[float]) -> float:
    a = np.asarray([x for x in p if x > 0], dtype=np.float64)
    if a.size <= 1:
        return 0.0
    return float(-(a * np.log(a)).sum() / np.log(len(p)))
