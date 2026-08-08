"""
coach/report.py — the shipped solve analysis, and the accuracy it claims.

This is the single entry point the API and the worker call:

    solve_report(onset_times, move_words, evening=False)

It assembles `timing.py` and `moves.py` into one payload, and — the part
that makes it a product rather than a pile of numbers — **attaches each
metric's measured accuracy and suppresses the ones that do not hold in the
current lighting regime.**

WHY ACCURACY TRAVELS WITH THE VALUE
-----------------------------------
Every number here is derived from a decode that is ~96% accurate daytime
and ~73% evening. Two failure modes follow if accuracy is not carried
alongside the value:

  * the UI shows an evening hesitation figure that is 22% wrong with the
    same confidence as a solve duration that is 0.1% wrong; and
  * a future contributor adds a metric by analogy ("it's just another
    average") without measuring it, and it silently ships broken.

So `MEASURED` below is not documentation. It is the gate. A metric absent
from it cannot be reported, and `metric_robustness.py` is what fills it
in. Numbers are the WORSE of the two seeds' median relative error on the
held-out solves (results/2026-08-06/metric_robustness_s0.json / _s1.json).

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Measured and rejected, each with its number, so nobody rebuilds them:

  n_pauses               25-27%  count of thresholded events
  median_burst_size      37-63%  same
  longest_pause_s        48-72% worst-case; a max is one bad move wide
  slowdown_ratio         15-18%  ratio of two noisy estimates
  same_face_pair_rate    27-75%  adjacent-pair, one insertion breaks two
  awkward_face_fraction  27-42%  share with a ~6% denominator
  half_turn_rate         18-24% worst-case; adjacent-pair

The design rule they encode: **report sums, means and shares over the
whole solve. Never a count of thresholded events, never a max, never a
ratio of two estimates, and never a share with a small denominator.**
"""

from __future__ import annotations

from dataclasses import dataclass

from .moves import move_report
from .timing import timing_report

#: Above this measured error a metric is not shown at all in that regime.
#: 15% is a judgement, but a defensible one: below it the number moves a
#: displayed figure by less than its own rounding in most cases, above it
#: the sign of a week-over-week change can flip on noise alone.
SUPPRESS_ABOVE_PCT = 15.0
#: Between these, the value ships flagged rather than plain.
CAUTION_ABOVE_PCT = 8.0


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    #: mean | local — see metric_robustness.STAT_KIND. Only these two kinds
    #: appear here; the other three were all rejected.
    kind: str
    #: Worse-of-two-seeds median relative error, percent.
    day_err: float
    eve_err: float
    #: Worse-of-two-seeds WORST-SESSION relative error, percent. Carried
    #: because the median flatters structurally: on the held-out set one
    #: daytime solve (20260803_095533) runs 12-16% wrong on several
    #: metrics whose median is 3%. The median is what the product is on
    #: average; this is what a user meets on a bad solve, and a single-solve
    #: view should be honest about it.
    day_worst: float
    eve_worst: float
    #: False for anything not yet independently measured.
    measured: bool = True


#: THE REGISTRY. Every metric the product is allowed to show.
MEASURED: tuple[Metric, ...] = (
    #                key                   label                 unit   kind    dayerr everr daywst evewst
    # -- headline timing ---------------------------------------------------
    Metric("span_seconds", "Solve time", "s", "mean", 0.1, 1.0, 0.2, 2.0),
    Metric("n_moves_qtm", "Moves", "QTM", "mean", 5.3, 2.8, 12.1, 14.3),
    Metric("span_tps", "Turns per second", "TPS", "mean", 5.4, 2.8, 12.3, 16.2),
    Metric("execution_tps", "Execution speed", "TPS", "mean",
           2.7, 8.6, 13.8, 17.9),
    Metric("mean_move_duration_s", "Average move", "s", "mean",
           2.8, 7.9, 16.0, 15.2),
    Metric("median_move_duration_s", "Typical move", "s", "mean",
           6.0, 11.2, 13.0, 23.1),
    Metric("move_duration_cv", "Turn consistency", "cv", "mean",
           5.1, 16.2, 9.1, 19.6),

    # -- hesitation --------------------------------------------------------
    Metric("hesitation_seconds", "Time spent thinking", "s", "mean",
           4.8, 20.8, 6.5, 24.2),
    Metric("hesitation_fraction", "Share of solve thinking", "frac", "mean",
           4.6, 21.7, 6.5, 22.0),

    # -- move identity -----------------------------------------------------
    Metric("ccw_fraction", "Counter-clockwise share", "frac", "mean",
           5.6, 12.9, 9.0, 15.2),
    Metric("face_share", "Face usage", "frac/face", "mean",
           2.5, 4.5, 6.4, 7.9),
    Metric("top_face_share", "Most-used face share", "frac", "mean",
           3.5, 10.9, 13.8, 16.7),
    Metric("easy_face_fraction", "R/U share (no regrip)", "frac", "mean",
           1.5, 5.6, 5.9, 7.1),
    Metric("face_entropy", "Face-usage spread", "0-1", "mean",
           0.5, 4.9, 4.5, 6.6),

    # -- execution shape (local-kind; the only two that survived) ----------
    Metric("distinct_face_runs", "Face changes per move", "frac", "local",
           1.9, 7.9, 2.7, 8.5),
    Metric("moves_per_face_run", "Moves per face run", "moves", "local",
           1.9, 8.6, 2.7, 9.3),
)

BY_KEY = {m.key: m for m in MEASURED}


def confidence(m: Metric, evening: bool | None) -> str:
    """high | caution | suppressed, from the measured error in this regime.

    `evening=None` means the lighting regime is unknown. That is treated as
    evening, not as daytime: the whole point of the gate is that an
    unverified regime must not borrow daytime's accuracy claim.
    """
    err = m.eve_err if (evening or evening is None) else m.day_err
    if err > SUPPRESS_ABOVE_PCT:
        return "suppressed"
    return "caution" if err > CAUTION_ABOVE_PCT else "high"


def solve_report(times, words: list[str], evening: bool | None = None,
                 include_suppressed: bool = False) -> dict:
    """
    Onset times + move words -> the analysis payload.

    `times` are seconds relative to timer start (coach t=0), `words` the
    decoded WCA quarter turns. The two must be the same length; they come
    from the same decode.

    Nothing here trusts a client: the caller supplies data the server
    derived. `evening` should come from `lighting_check.py`, not from a
    clock the client controls.
    """
    t_rep = timing_report(times)
    if not t_rep.get("usable"):
        return {"usable": False, "reason": "fewer than 2 onsets"}
    m_rep = move_report(words) if words else {}

    source = {**t_rep, **m_rep}
    # timing_report calls it span_seconds; the registry uses the same name.
    metrics, suppressed = {}, []
    for m in MEASURED:
        if m.key not in source or source[m.key] is None:
            continue
        conf = confidence(m, evening)
        if conf == "suppressed" and not include_suppressed:
            suppressed.append(m.key)
            continue
        metrics[m.key] = {
            "value": source[m.key],
            "label": m.label,
            "unit": m.unit,
            "confidence": conf,
            "accuracy_pct": (m.eve_err if (evening or evening is None)
                             else m.day_err),
        }

    return {
        "usable": True,
        "regime": ("evening" if evening else
                   "daytime" if evening is False else "unknown"),
        "metrics": metrics,
        "suppressed": suppressed,
        #: The raw layer outputs, unfiltered. Kept so the solve-detail view
        #: and any future analysis can reach past the gate deliberately —
        #: but nothing in `metrics` above ever comes from here ungated.
        "raw": {"timing": t_rep, "moves": m_rep},
    }


def registry_table() -> list[dict]:
    """The shipped metric inventory, for docs and for the API's /meta."""
    return [{"key": m.key, "label": m.label, "unit": m.unit, "kind": m.kind,
             "daytime_err_pct": m.day_err, "evening_err_pct": m.eve_err,
             "daytime_worst_pct": m.day_worst,
             "evening_worst_pct": m.eve_worst,
             "daytime": confidence(m, False), "evening": confidence(m, True)}
            for m in MEASURED]
