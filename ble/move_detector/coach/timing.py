"""
coach/timing.py — L1 timing analytics over a list of onset timestamps.

This is the bottom layer of the coach (LAUNCH_ROADMAP §3.2 L1): it needs
only *when* moves happened, never *which* move it was, so it is the one
layer that survives the evening-lighting cliff and a failed endpoint check
intact. Pure functions over a float array, no model, no I/O — the offline
harness and the server worker import the same code.

THE CENTRAL METRIC: EXECUTION TPS
---------------------------------
"Moves per second while not hesitating." A solve is not a uniform stream
of turns; it is bursts of execution separated by recognition pauses, and
those two things are different skills. Overall TPS averages them together
and is therefore the one number that improves when you get better at
*either* and tells you which at neither. Splitting them gives:

    execution TPS      how fast your hands are, once you know what to do
    hesitation time    how long you spend deciding
    burst structure    how much you execute per decision

WHERE THE THRESHOLD COMES FROM, AND THE HONEST CAVEAT
-----------------------------------------------------
A pause is an inter-onset interval above a threshold. It would be much
nicer if that threshold were discovered rather than chosen, and it is
worth recording that **it cannot be, on this corpus.** The pooled
inter-onset distribution over the 6 held-out solves (716 intervals) is a
single heavy-tailed mode peaking at 50-150ms and decaying monotonically
once bin widths are normalised — there is no valley between "executing"
and "thinking" to put a line in. Execution and hesitation overlap
continuously.

So the threshold is a **convention**, and the honest response to a
convention is to state it and measure how much the answer depends on it
(`sweep_threshold` below, and `execution_tps.py` prints it). It is
relative to the solver's own median rather than absolute because solve
speed drifts — 1.30 to 2.83 moves/s in 12 days — so a fixed constant
would silently change meaning as the user improves.

UNITS
-----
Every rate here is **QTM**: the decoder emits 12 quarter-turn classes and
a half turn arrives as two moves. Cubers quote HTM and every other timer
will show a lower number, so any UI reading these must say QTM.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

#: Pause = IOI above max(FACTOR * median IOI, FLOOR). See the caveat above:
#: chosen, not discovered. The floor stops the rule degenerating on a very
#: fast solve where 3x a tiny median is still inside execution noise; it is
#: also comfortably above the ~40ms timing jitter measured in
#: results/2026-08-06/onset_timing_s*.json, so a pause is never an artifact
#: of onset noise.
PAUSE_FACTOR = 3.0
PAUSE_FLOOR_S = 0.25

#: A burst needs this many moves before its own TPS is quoted. Two onsets
#: give one interval, and one interval is a sample of size one — a 60ms
#: pair would report 16 TPS and be meaningless.
MIN_BURST_FOR_RATE = 4


@dataclass
class Burst:
    """A run of consecutive onsets with no pause between them."""

    start: float            #: first onset, seconds
    end: float              #: last onset, seconds
    n_moves: int            #: onsets in the burst
    first_index: int        #: index of `start` in the onset list

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def tps(self) -> float | None:
        """Rate inside the burst, or None if it is too short to quote.

        `(n - 1) / span`, not `n / span`: n onsets bound n-1 intervals, and
        a rate estimated from event times is intervals over elapsed time.
        Using n/span would inflate every burst, most severely the short
        ones. (`Solve.tps` in the server divides moves by the full timer
        duration, which includes the lead-in before the first move — the
        two converge for long streams and are not meant to match exactly.)
        """
        if self.n_moves < MIN_BURST_FOR_RATE or self.seconds <= 0:
            return None
        return (self.n_moves - 1) / self.seconds


def inter_onset(times) -> np.ndarray:
    """Inter-onset intervals. Empty for fewer than two onsets."""
    t = np.asarray(times, dtype=np.float64)
    return np.diff(t) if t.size >= 2 else np.array([])


def pause_threshold(times, factor: float = PAUSE_FACTOR,
                    floor: float = PAUSE_FLOOR_S) -> float:
    """
    Threshold derived from the SAME stream it will be applied to.

    Deliberate: at inference there is no ground truth to borrow a median
    from, so a threshold fitted on anything else would measure something
    the product cannot reproduce.
    """
    ioi = inter_onset(times)
    if ioi.size < 2:
        return floor
    return max(factor * float(np.median(ioi)), floor)


def pause_spans(times, thr: float | None = None
                ) -> tuple[float, list[tuple[float, float]]]:
    """(threshold, [(start, end)]) for every interval above threshold."""
    t = np.asarray(times, dtype=np.float64)
    ioi = inter_onset(t)
    if ioi.size == 0:
        return (thr if thr is not None else PAUSE_FLOOR_S), []
    if thr is None:
        thr = pause_threshold(t)
    spans = [(float(t[k]), float(t[k + 1]))
             for k in range(ioi.size) if ioi[k] > thr]
    return thr, spans


def bursts(times, thr: float | None = None) -> list[Burst]:
    """
    Split the onset stream wherever a pause occurs.

    This is the "clusters" view: the solver decides what to do, executes a
    run of moves, and stops to decide again. Each Burst is one such run.
    A solve with no pauses at all is a single burst, which is correct — it
    means the solver never stopped.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return []
    if t.size == 1:
        return [Burst(float(t[0]), float(t[0]), 1, 0)]
    if thr is None:
        thr = pause_threshold(t)
    ioi = inter_onset(t)

    out, start_i = [], 0
    for k in range(ioi.size):
        if ioi[k] > thr:
            out.append(Burst(float(t[start_i]), float(t[k]),
                             k - start_i + 1, start_i))
            start_i = k + 1
    out.append(Burst(float(t[start_i]), float(t[-1]),
                     t.size - start_i, start_i))
    return out


def timing_report(times, thr: float | None = None) -> dict:
    """
    The L1 payload for one solve. All rates QTM.

    `execution_tps` is the headline the user asked for: total moves
    executed inside bursts over the time actually spent executing, with
    every hesitation interval removed. Algebraically it is
    `1 / mean(non-pause IOI)`, which is worth knowing because it makes
    clear that it is an average over intervals and not over bursts — a
    long burst weighs more than a short one, which is what you want.
    """
    t = np.asarray(times, dtype=np.float64)
    n = int(t.size)
    if n < 2:
        return {"n_moves": n, "usable": False}

    if thr is None:
        thr = pause_threshold(t)
    ioi = inter_onset(t)
    is_pause = ioi > thr

    span = float(t[-1] - t[0])
    hesitation = float(ioi[is_pause].sum())
    execution = float(ioi[~is_pause].sum())
    n_exec_intervals = int((~is_pause).sum())

    bs = bursts(t, thr)
    sizes = [b.n_moves for b in bs]
    rates = [b.tps for b in bs if b.tps is not None]

    return {
        "usable": True,
        "n_moves": n,
        "pause_threshold_s": round(thr, 3),
        "median_ioi_s": round(float(np.median(ioi)), 3),

        # -- the split ------------------------------------------------------
        "span_seconds": round(span, 3),
        "execution_seconds": round(execution, 3),
        "hesitation_seconds": round(hesitation, 3),
        "hesitation_fraction": round(hesitation / span, 4) if span > 0 else None,

        # -- rates ----------------------------------------------------------
        #: What the user asked for: hesitation removed.
        "execution_tps": (round(n_exec_intervals / execution, 3)
                          if execution > 0 else None),
        #: The same quantity estimated WITHOUT a threshold at all, as the
        #: reciprocal of the median interval. A median is already immune to
        #: the long tail that hesitation lives in, so no pause rule is
        #: needed to exclude it.
        #:
        #: This exists to anchor the number above. `execution_tps` moves
        #: ~30% across a 2x-5x threshold sweep, which on its own would mean
        #: the metric is substantially a re-parameterisation of a
        #: convention. It is rescued by the fact that at the default 3x the
        #: two agree to ~3% on all six held-out solves — so 3x is not an
        #: arbitrary pick, it is the threshold at which the pause-based
        #: metric reproduces the threshold-free one. Report `execution_tps`
        #: (it decomposes into bursts, which the robust form cannot), but
        #: if the two ever diverge, trust this one and re-derive the factor.
        "execution_tps_robust": (round(1.0 / float(np.median(ioi)), 3)
                                 if np.median(ioi) > 0 else None),
        #: The same span WITHOUT removing hesitation, so the two can be
        #: compared. This is the number a plain moves/seconds reports.
        "span_tps": round((n - 1) / span, 3) if span > 0 else None,
        #: Fastest sustained run. "How fast can your hands actually go."
        "peak_burst_tps": round(max(rates), 3) if rates else None,
        "median_burst_tps": round(float(np.median(rates)), 3) if rates else None,

        # -- move duration --------------------------------------------------
        #
        # The reciprocal view of the rates above. Worth returning explicitly
        # even though it carries no new information (1/tps), because "your
        # average move takes 205ms" is a far more legible sentence than
        # "4.9 TPS" and the UI should not be doing arithmetic.
        #
        # `move_duration_cv` is the exception — it IS new information, and
        # arguably the most interesting thing in this block. Coefficient of
        # variation over execution intervals is turning *consistency*: two
        # solvers can share an execution TPS while one turns metronomically
        # and the other alternates bursts with micro-stutters. Low CV is the
        # signature of a well-drilled algorithm; high CV at the same mean
        # rate means the hands know the moves but not the transitions.
        "mean_move_duration_s": (round(execution / n_exec_intervals, 4)
                                 if n_exec_intervals else None),
        "median_move_duration_s": round(float(np.median(ioi)), 4),
        "mean_move_duration_incl_pauses_s": (round(span / (n - 1), 4)
                                             if n > 1 else None),
        "move_duration_cv": (round(float(ioi[~is_pause].std())
                                   / float(ioi[~is_pause].mean()), 3)
                             if n_exec_intervals > 1
                             and ioi[~is_pause].mean() > 0 else None),

        # -- rate over the solve ---------------------------------------------
        #: The shape of the solve rather than one number for it. Measured
        #: 2026-08-10, both seeds, 14 held-out solves: 6.8% median daytime
        #: (9.9% worst) and 10.2% evening (14.0% worst) — the only rate
        #: metric here that stays reportable in BOTH regimes. Its rejected
        #: alternative, `slowdown_ratio`, re-measured 10.5-15.9% daytime and
        #: 12.2-21.2% evening on the same run, so the curve beats the ratio
        #: in every cell. See tps_curve.
        "tps_curve": tps_curve(t),

        # -- cluster structure ----------------------------------------------
        "n_bursts": len(bs),
        "burst_sizes": sizes,
        "median_burst_size": int(np.median(sizes)) if sizes else 0,
        "largest_burst": max(sizes) if sizes else 0,
        "moves_per_decision": round(n / len(bs), 2) if bs else None,

        # -- hesitation detail ----------------------------------------------
        "n_pauses": int(is_pause.sum()),
        "longest_pause_s": round(float(ioi[is_pause].max()), 3)
        if is_pause.any() else 0.0,
        "median_pause_s": round(float(np.median(ioi[is_pause])), 3)
        if is_pause.any() else 0.0,
    }


#: Moves per window of the TPS curve. Small enough to resolve structure
#: inside a solve (an F2L pair is ~7-10 QTM, so k=5 sits below one algorithm
#: and can show the shape *within* it) and large enough that one interval
#: cannot dominate a point. Not tuned: 5 is what TODO.md §7D specified, and
#: the accuracy measurement below is at that value, so changing it
#: invalidates the number.
TPS_CURVE_K = 5


def tps_curve(times, k: int = TPS_CURVE_K) -> list[dict]:
    """
    Turn rate over the course of the solve: a sliding k-move window.

    WHY A CURVE AND NOT A SLOWDOWN NUMBER. The obvious summary — "how much
    slower is the last third than the first" — was built and measured and is
    the wrong shape. `slowdown_ratio` is one noisy estimate divided by
    another, which §7C measured as the second-worst statistic kind in the
    table (15-18% median error), so it is deliberately NOT returned here.
    Each point of this curve is a `mean` over its window, which is the kind
    that survives.

    WHY IT DOES NOT COLLAPSE THROUGH A PAUSE. Rate-per-window is
    `(k-1) / span`, so a window straddling a 3-second think reads as a low
    rate, not a zero and not a gap. A fixed-time window would have to report
    zero moves per second there, which is both wrong (the solver did not
    stop being a solver) and undrawable. This also makes the curve
    scale-free: it has the same number of points for a 40-move solve and a
    70-move one.

    Returns one dict per window:

        t        midpoint of the window in seconds, the x-axis
        t_start  first onset in the window
        t_end    last onset in the window
        i        index of the first move, for cross-referencing the move list
        tps      (k-1) / (t_end - t_start), QTM

    Empty for fewer than `k` onsets: a window shorter than k moves is a
    different statistic with a different variance, and quietly returning one
    would put two incomparable things on the same axis.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.size < k or k < 2:
        return []
    out = []
    for i in range(t.size - k + 1):
        span = float(t[i + k - 1] - t[i])
        if span <= 0:
            continue
        out.append({"i": int(i),
                    "t": round(float(t[i] + span / 2), 3),
                    "t_start": round(float(t[i]), 3),
                    "t_end": round(float(t[i + k - 1]), 3),
                    "tps": round((k - 1) / span, 3)})
    return out


def tps_curve_at(times, grid, k: int = TPS_CURVE_K):
    """
    The curve sampled at arbitrary times — how two curves get compared.

    Truth and decode produce different numbers of windows (they disagree
    about how many moves there were), so the curves cannot be compared point
    for point by index. They do share a wall clock, so resampling both onto
    one time grid is the comparison that means something. Held flat outside
    the curve's own span rather than extrapolated, since a rate estimated
    off the end of the evidence is not an estimate.
    """
    c = tps_curve(times, k)
    if not c:
        return None
    xs = np.array([p["t"] for p in c], dtype=np.float64)
    ys = np.array([p["tps"] for p in c], dtype=np.float64)
    return np.interp(np.asarray(grid, dtype=np.float64), xs, ys)


def sweep_threshold(times, factors=(2.0, 2.5, 3.0, 4.0, 5.0)) -> list[dict]:
    """
    How much does the answer depend on the convention?

    Reported rather than hidden, because the threshold is chosen and not
    discovered (see the module docstring). If execution TPS swings wildly
    across this sweep the metric is really a re-parameterisation of the
    threshold and should not be shown to a user as a fact about them.
    """
    out = []
    for f in factors:
        thr = pause_threshold(times, factor=f)
        r = timing_report(times, thr=thr)
        if not r.get("usable"):
            continue
        out.append({"factor": f, "threshold_s": r["pause_threshold_s"],
                    "execution_tps": r["execution_tps"],
                    "hesitation_fraction": r["hesitation_fraction"],
                    "n_bursts": r["n_bursts"],
                    "median_burst_size": r["median_burst_size"]})
    return out


def burst_dicts(bs: list[Burst]) -> list[dict]:
    """Bursts as plain dicts, for JSON output and the API payload."""
    return [{**asdict(b), "seconds": round(b.seconds, 3),
             "tps": round(b.tps, 3) if b.tps is not None else None}
            for b in bs]
