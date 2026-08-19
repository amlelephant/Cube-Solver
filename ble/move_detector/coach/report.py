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
held-out solves.

RE-MEASURED 2026-08-10 ON A HOLDOUT THAT MORE THAN DOUBLED
-----------------------------------------------------------
The figures below came from 6 held-out solves (2026-08-06). The corpus has
since grown and the same checkpoints now hold out **14** — 9 daytime, 5
evening — so every number here has been re-derived on that larger set
(results/2026-08-10/metric_robustness_s0.json / _s1.json).

Before replacing them, the harness was re-run restricted to the ORIGINAL
six sessions and reproduced the 2026-08-06 table exactly, all 25 metrics in
both regimes. That check is the point: without it, "the numbers moved"
could equally mean the measurement code drifted, and there would be no way
to tell which. They moved because there is more data, and nothing else.

Most moved toward being *better characterised* rather than uniformly
better — `hesitation_seconds` fell 4.8% -> 2.1% daytime while its worst
case rose 6.5% -> 17.4%, which is what a bigger sample does: the median
settles and the tail finally shows up. Read the worst column.

    ONE CONSEQUENCE NEEDS A DECISION, NOT JUST A NUMBER. Evening
    hesitation went 20.8% -> 8.4% and evening `move_duration_cv` 16.2% ->
    8.1%, which drops both under SUPPRESS_ABOVE_PCT. **Nothing in the
    registry is suppressed in either regime any more** — the three metrics
    the /analytics page shows as withheld are now shown. That is what the
    measurement says, and the rule is that the measurement decides. But it
    rests on FIVE evening solves (was three), and a median over five is
    still weak; the worst column for those same metrics is 24.2% and
    19.6%, i.e. a bad evening solve is still bad. If the intent is that
    evening hesitation stays hidden until the evening corpus is real
    (LAUNCH_ROADMAP B5), that is a policy change — lower
    SUPPRESS_ABOVE_PCT, or gate evening on the worst column instead of the
    median — and not a matter of editing these numbers.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Measured and rejected, each with its number (2026-08-10, worse of two
seeds, daytime/evening median unless noted), so nobody rebuilds them:

  n_pauses               5.9-9.5% day, 11.8-22.2% evening; a count of
                         thresholded events, and the evening cell alone
                         disqualifies it
  median_burst_size      14-33%, worst 200%   same kind
  longest_pause_s        median 0.4-2.1% but worst 48-72%; a max is one
                         bad move wide, and the median flatters it
                         structurally
  slowdown_ratio         10.5-15.9% day, 12.2-21.2% evening; a ratio of
                         two noisy estimates. SUPERSEDED — `tps_curve`
                         carries the same "did they slow down" signal as a
                         sequence of means and measures 6.8/10.2%, better
                         in every cell.
  same_face_pair_rate    10-16% day, 20-29% evening; adjacent-pair
  awkward_face_fraction  4.9-8.6% day, 27-28% evening, worst 100%; a share
                         with a ~6% denominator
  half_turn_rate         14-17% day, 9-20% evening; adjacent-pair

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
    #: True when the value is a SEQUENCE (a curve over the solve) rather
    #: than a scalar. The gate is identical either way — a curve still has
    #: one measured error and one confidence — but a client cannot render a
    #: list of points into a stat tile, so it has to be told which it is
    #: rather than sniffing the JSON type.
    series: bool = False


#: THE REGISTRY. Every metric the product is allowed to show.
#: Errors: worse of two seeds, 14 held-out solves (9 daytime / 5 evening),
#: results/2026-08-10/metric_robustness_s{0,1}.json.
MEASURED: tuple[Metric, ...] = (
    #                key                   label                 unit   kind    dayerr everr daywst evewst
    # -- headline timing ---------------------------------------------------
    Metric("span_seconds", "Solve time", "s", "mean", 0.2, 0.1, 1.9, 2.0),
    Metric("n_moves_qtm", "Moves", "QTM", "mean", 5.3, 6.5, 13.6, 14.3),
    Metric("span_tps", "Turns per second", "TPS", "mean", 5.4, 6.5, 12.3, 16.2),
    Metric("execution_tps", "Execution speed", "TPS", "mean",
           5.0, 8.6, 13.8, 17.9),
    Metric("mean_move_duration_s", "Average move", "s", "mean",
           5.3, 7.9, 16.0, 15.2),
    Metric("median_move_duration_s", "Typical move", "s", "mean",
           7.3, 12.9, 18.2, 27.3),
    Metric("move_duration_cv", "Turn consistency", "cv", "mean",
           7.7, 8.1, 11.3, 19.6),
    #: The rate over the course of the solve, sliding 5-move window
    #: (timing.tps_curve). Scored by resampling truth and decode onto one
    #: shared time grid and taking the median disagreement across it, so it
    #: is a `mean`-kind statistic 24 times over — which is why its WORST
    #: case (9.9 / 14.0%) is tighter than most scalars here, the opposite
    #: of how `extreme`-kind metrics behave. Replaces `slowdown_ratio`,
    #: which is the same question asked as a ratio and measures worse in
    #: every cell.
    Metric("tps_curve", "Speed through the solve", "TPS", "mean",
           6.8, 10.2, 9.9, 14.0, series=True),

    # -- hesitation --------------------------------------------------------
    Metric("hesitation_seconds", "Time spent thinking", "s", "mean",
           2.1, 8.4, 17.4, 24.2),
    Metric("hesitation_fraction", "Share of solve thinking", "frac", "mean",
           2.2, 8.4, 19.7, 22.0),

    # -- move identity -----------------------------------------------------
    Metric("ccw_fraction", "Counter-clockwise share", "frac", "mean",
           4.4, 7.0, 10.5, 15.2),
    Metric("face_share", "Face usage", "frac/face", "mean",
           2.4, 3.6, 6.4, 7.9),
    Metric("top_face_share", "Most-used face share", "frac", "mean",
           2.4, 5.0, 13.8, 16.7),
    Metric("easy_face_fraction", "R/U share (no regrip)", "frac", "mean",
           1.8, 3.0, 6.7, 7.1),
    Metric("face_entropy", "Face-usage spread", "0-1", "mean",
           0.7, 4.0, 15.1, 6.6),

    # -- execution shape (local-kind; the only two that survived) ----------
    Metric("distinct_face_runs", "Face changes per move", "frac", "local",
           2.7, 3.9, 6.8, 8.5),
    Metric("moves_per_face_run", "Moves per face run", "moves", "local",
           2.7, 4.0, 7.3, 9.3),
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
            "worst_pct": (m.eve_worst if (evening or evening is None)
                          else m.day_worst),
            "series": m.series,
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
             "series": m.series,
             "daytime_err_pct": m.day_err, "evening_err_pct": m.eve_err,
             "daytime_worst_pct": m.day_worst,
             "evening_worst_pct": m.eve_worst,
             "daytime": confidence(m, False), "evening": confidence(m, True)}
            for m in MEASURED]
