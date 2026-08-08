"""
metric_robustness.py — which analytics can we actually ship, given our error?

The question this answers is a product question, not a modelling one:
**of everything extractable from a decoded move list, which metrics are
accurate enough to show a user, and which are noise wearing a number?**

It is answered by measurement rather than argument. Every candidate metric
is computed twice per session — once from BLE ground truth, once from the
CTC decode the product would actually have — and ranked by the median
absolute relative disagreement. A metric that lands under a few percent is
shippable; one at 20% is a plot, whatever its intuitive appeal.

WHY A BATTERY AND NOT A DERIVATION
----------------------------------
It is tempting to reason from the known error rates (~4% word error
daytime, ~40ms onset jitter) to which metrics survive. That reasoning is
unreliable in both directions, and this project has been burned by it:

  * Errors **cancel** more than expected in some aggregates. A miss and a
    phantom in the same solve leave the move count untouched; onset bias
    cancels completely in any interval (results/2026-08-06/onset_timing_*).
  * Errors **compound** worse than expected in others. Pause counts are a
    classification over intervals, so a single miss merges two intervals
    and manufactures a pause — 4% word error became 32% pause precision
    loss in evening light.

Neither is predictable from the headline rate. So: measure.

METRIC CLASSES, WHICH IS THE REAL FINDING
-----------------------------------------
Candidates are tagged by what they need, because that tag is what
generalises to metrics not yet invented:

  timing    onset timestamps only. Immune to substitutions entirely, and
            survives the evening cliff better than anything else.
  count     how many moves, not which. Miss/phantom partially cancel.
  identity  which move. Exposed to the full word error rate.
  order     which move, in sequence, over a span. Error compounds with
            span length — the most fragile class.

Usage:
    python metric_robustness.py --ctc checkpoints/move_ctc_spd_s0.pt \\
        --out results/2026-08-06/metric_robustness_s0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from algorithm_gate import posteriorgram
from coach.moves import FACES, move_report
from coach.timing import bursts, inter_onset, pause_threshold, timing_report
from ctc_decode import prefix_beam_decode
from eval_lighting import DAY_END, DAY_START, held_out_sessions, take_hour
from model import build_joint_from_ckpt
from onset_timing import frame_time_axis
from reconstruct import WCA12

SKIP_SUFFIX = "_scramble"

#: metric -> what it needs. Drives the grouping in the report, and is the
#: part that generalises beyond this specific list.
NEEDS = {
    "median_move_duration_ms": "timing",
    "mean_move_duration_ms": "timing",
    "move_duration_cv": "timing",
    "execution_tps": "timing",
    "execution_tps_robust": "timing",
    "span_tps": "timing",
    "hesitation_fraction": "timing",
    "hesitation_seconds": "timing",
    "longest_pause_s": "timing",
    "n_pauses": "timing",
    "median_burst_size": "timing",
    "slowdown_ratio": "timing",
    "solve_seconds": "timing",
    "n_moves": "count",
    "moves_per_second_overall": "count",
    "ccw_fraction": "identity",
    "face_share_L1": "identity",
    "top_face_share": "identity",
    "easy_face_fraction": "identity",
    "awkward_face_fraction": "identity",
    "face_entropy": "identity",
    "same_face_pair_rate": "order",
    "half_turn_rate": "order",
    "distinct_face_runs": "order",
    "moves_per_face_run": "order",
}

#: What KIND of statistic it is — and this, not `NEEDS`, turned out to be
#: what predicts robustness (measured 2026-08-06, both seeds). The `NEEDS`
#: taxonomy above was the a-priori guess and it is wrong on the main axis:
#: "timing" contains both the most robust metrics here and the worst tail
#: of any class (72%), because it contains both raw means and thresholded
#: counts. The distinction that actually holds:
#:
#:   mean      a sum or average over every move. Errors average out; these
#:             are the robust ones regardless of needing identity or not.
#:   threshold a COUNT of events defined by crossing a threshold. A small
#:             continuous error becomes a discrete miscount, so the error
#:             is amplified rather than averaged.
#:   extreme   a max or min. One bad move IS the statistic; no averaging
#:             is possible even in principle. Median error flatters these
#:             badly — read their worst column.
#:   ratio     one noisy estimate divided by another; the two errors
#:             compound instead of cancelling.
#:   local     adjacent-pair structure. A single insertion corrupts two
#:             pairs at once, so error is roughly doubled.
STAT_KIND = {
    "median_move_duration_ms": "mean", "mean_move_duration_ms": "mean",
    "execution_tps": "mean", "execution_tps_robust": "mean",
    "span_tps": "mean", "hesitation_seconds": "mean",
    "hesitation_fraction": "mean", "solve_seconds": "mean",
    "n_moves": "mean", "moves_per_second_overall": "mean",
    "ccw_fraction": "mean", "face_share_L1": "mean",
    "top_face_share": "mean", "move_duration_cv": "mean",
    "easy_face_fraction": "mean", "awkward_face_fraction": "mean",
    "face_entropy": "mean",
    "moves_per_face_run": "local",
    "n_pauses": "threshold", "median_burst_size": "threshold",
    "longest_pause_s": "extreme",
    "slowdown_ratio": "ratio",
    "same_face_pair_rate": "local", "half_turn_rate": "local",
    "distinct_face_runs": "local",
}


def battery(times: np.ndarray, labels: list[int]) -> dict:
    """Every candidate metric for one solve, from onsets + move labels."""
    t = np.asarray(times, dtype=np.float64)
    out: dict[str, float | None] = {}
    if t.size < 8 or len(labels) != t.size:
        return out

    rep = timing_report(t)
    ioi = inter_onset(t)
    thr = pause_threshold(t)
    exec_ioi = ioi[ioi <= thr]

    # -- timing ------------------------------------------------------------
    out["median_move_duration_ms"] = float(np.median(ioi)) * 1e3
    out["mean_move_duration_ms"] = (float(exec_ioi.mean()) * 1e3
                                    if exec_ioi.size else None)
    out["move_duration_cv"] = rep["move_duration_cv"]
    out["execution_tps"] = rep["execution_tps"]
    out["execution_tps_robust"] = rep["execution_tps_robust"]
    out["span_tps"] = rep["span_tps"]
    out["hesitation_fraction"] = rep["hesitation_fraction"]
    out["hesitation_seconds"] = rep["hesitation_seconds"]
    out["longest_pause_s"] = rep["longest_pause_s"]
    out["n_pauses"] = float(rep["n_pauses"])
    out["median_burst_size"] = float(rep["median_burst_size"])
    out["solve_seconds"] = rep["span_seconds"]

    # Does the solver slow down? Median interval over the last third vs the
    # first third. A ratio, so the solver's absolute speed divides out and
    # only the shape of the solve remains.
    third = max(t.size // 3, 2)
    first = np.median(inter_onset(t[:third]))
    last = np.median(inter_onset(t[-third:]))
    out["slowdown_ratio"] = float(last / first) if first > 0 else None

    # -- count -------------------------------------------------------------
    out["n_moves"] = float(t.size)
    span = float(t[-1] - t[0])
    out["moves_per_second_overall"] = (t.size / span) if span > 0 else None

    # -- identity ----------------------------------------------------------
    # Routed through the SHIPPED implementation rather than a local copy, so
    # these numbers describe the code the product runs. A private
    # reimplementation here would be measuring a thing nobody uses.
    words = [WCA12[c] for c in labels]
    mv = move_report(words)
    out["ccw_fraction"] = mv["ccw_fraction"]
    out["top_face_share"] = mv["top_face_share"]
    out["easy_face_fraction"] = mv["easy_face_fraction"]
    out["awkward_face_fraction"] = mv["awkward_face_fraction"]
    out["face_entropy"] = mv["face_entropy"]
    out["moves_per_face_run"] = mv["moves_per_face_run"]
    #: Stashed so the L1 distance below can be computed against the other
    #: stream rather than against a constant.
    out["_face_share"] = [mv["face_share"][f] for f in FACES]

    # -- order -------------------------------------------------------------
    # Adjacent-pair structure. Same face twice in a row is either a half
    # turn (identical move repeated) or a regrip/setup; both are execution
    # signatures a coach would want, and both are the most fragile class
    # here because a single insertion breaks two pairs at once.
    pairs = list(zip(words, words[1:]))
    out["same_face_pair_rate"] = (sum(a[0] == b[0] for a, b in pairs)
                                  / max(len(pairs), 1))
    out["half_turn_rate"] = (sum(a == b for a, b in pairs)
                             / max(len(pairs), 1))
    out["distinct_face_runs"] = mv["distinct_face_runs"]
    return out


def compare(truth: dict, pred: dict) -> dict:
    """Relative disagreement per metric, plus the face-distribution L1."""
    err: dict[str, float | None] = {}
    for k, tv in truth.items():
        if k.startswith("_"):
            continue
        pv = pred.get(k)
        if tv is None or pv is None or tv == 0:
            err[k] = None
        else:
            err[k] = (pv - tv) / abs(tv)
    ts, ps = truth.get("_face_share"), pred.get("_face_share")
    if ts and ps:
        # Total variation between the two face distributions: an absolute
        # error already on a 0-1 scale, so it is stored as-is rather than
        # relativised.
        err["face_share_L1"] = float(np.abs(np.array(ts)
                                            - np.array(ps)).sum() / 2)
    return err


def score_session(d: Path, model, device, tag: str, beam: int,
                  refresh: bool) -> dict | None:
    z = np.load(d / "detector_stream_color.npz", allow_pickle=True)
    true_t = z["onset_ts"].astype(np.float64)
    true_c = z["onset_class"].astype(int).tolist()
    if true_t.size < 8 or len(true_c) != true_t.size:
        return None

    class_prob, _ = posteriorgram(d, model, device, refresh, tag)
    fts = frame_time_axis(d, class_prob.shape[0])
    if fts is None:
        return None
    labels, frames = prefix_beam_decode(np.log(np.maximum(class_prob, 1e-12)),
                                        beam=beam)
    if len(labels) < 8:
        return None
    pred_t = fts[np.clip(np.asarray(frames, dtype=int), 0, len(fts) - 1)]

    tb, pb = battery(true_t, true_c), battery(pred_t, labels)
    if not tb or not pb:
        return None

    hour = take_hour(d)
    return {
        "session": d.name, "hour": hour,
        "evening": not (hour is not None and DAY_START <= hour < DAY_END),
        "truth": {k: v for k, v in tb.items() if not k.startswith("_")},
        "decoded": {k: v for k, v in pb.items() if not k.startswith("_")},
        "error": compare(tb, pb),
    }


def rank(rows: list[dict]) -> list[dict]:
    names = sorted({k for r in rows for k in r["error"]},
                   key=lambda n: NEEDS.get(n, "zzz"))
    out = []
    for n in names:
        vals = [abs(r["error"][n]) for r in rows
                if r["error"].get(n) is not None]
        signed = [r["error"][n] for r in rows if r["error"].get(n) is not None]
        truths = [r["truth"][n] for r in rows
                  if r["truth"].get(n) is not None]
        if not vals:
            continue
        out.append({
            "metric": n, "needs": NEEDS.get(n, "?"),
            "kind": STAT_KIND.get(n, "?"),
            "n": len(vals),
            "abs_err": float(np.median(vals)),
            "bias": float(np.median(signed)),
            "worst": float(max(vals)),
            "truth_median": float(np.median(truths)) if truths else None,
        })
    return sorted(out, key=lambda r: r["abs_err"])


def print_table(title: str, ranked: list[dict]) -> None:
    print(f"\n  {title}")
    print(f"    {'metric':<28} {'needs':<9} {'truth':>10} {'|err|':>8} "
          f"{'bias':>8} {'worst':>8}  verdict")
    print(f"    {'-' * 88}")
    for r in ranked:
        e, w = r["abs_err"] * 100, r["worst"] * 100
        verdict = ("ship" if e < 5 and w < 15 else
                   "ship w/ care" if e < 10 and w < 25 else
                   "aggregate only" if e < 20 else "NOT SHIPPABLE")
        tv = r["truth_median"]
        tvs = f"{tv:>10.3g}" if tv is not None else f"{'--':>10}"
        print(f"    {r['metric']:<28} {r['needs']:<9} {tvs} {e:>7.1f}% "
              f"{r['bias'] * 100:>+7.1f}% {w:>7.1f}%  {verdict}")


def _rollup(ranked: list[dict], field: str, order: tuple[str, ...]) -> None:
    print(f"    {'class':<11} {'metrics':>8} {'median |err|':>14} "
          f"{'worst |err|':>13}")
    for cls in order:
        sel = [r for r in ranked if r[field] == cls]
        if not sel:
            continue
        print(f"    {cls:<11} {len(sel):>8} "
              f"{np.median([r['abs_err'] for r in sel]) * 100:>13.1f}% "
              f"{max(r['worst'] for r in sel) * 100:>12.1f}%")


def by_class(rows: list[dict], ranked: list[dict]) -> None:
    print("\n  Rolled up by what the metric NEEDS (the a-priori guess):")
    _rollup(ranked, "needs", ("timing", "count", "identity", "order"))
    print("\n  Rolled up by what KIND of statistic it is — this is the one "
          "that\n  actually predicts robustness, and it cuts across the "
          "table above:")
    _rollup(ranked, "kind",
            ("mean", "local", "ratio", "threshold", "extreme"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctc", default="checkpoints/move_ctc_spd_s0.pt")
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--include-scrambles", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ctc, map_location=device)
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    tag = f"{Path(args.ctc).stem}_e{ckpt['epoch']}"
    print(f"  model {args.ctc} (epoch {ckpt['epoch']}) on {device}")

    dirs = ([Path(s) for s in args.sessions] if args.sessions
            else held_out_sessions([Path(args.ctc)]))
    dirs = [d for d in dirs if (d / "detector_stream_color.npz").exists()]
    if not args.include_scrambles:
        dirs = [d for d in dirs if not d.name.endswith(SKIP_SUFFIX)]

    rows = []
    for d in dirs:
        try:
            r = score_session(d, model, device, tag, args.beam, args.refresh)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {d.name}: {type(exc).__name__}: {exc} — skipped")
            continue
        if r is not None:
            rows.append(r)
    if not rows:
        raise SystemExit("Nothing scored.")
    print(f"  {len(rows)} held-out solve(s)")

    day = [r for r in rows if not r["evening"]]
    eve = [r for r in rows if r["evening"]]
    r_day, r_eve, r_all = rank(day), rank(eve), rank(rows)

    print(f"\n{'=' * 92}")
    print("  METRIC ROBUSTNESS — decoded vs BLE truth, ranked by "
          "disagreement")
    print(f"{'=' * 92}")
    print("  |err| is the median absolute relative error over sessions; "
          "'worst' is the worst\n  single session, which is what a user "
          "actually experiences on a bad solve.")
    print_table(f"DAYTIME ({len(day)} solves)", r_day)
    print_table(f"EVENING ({len(eve)} solves)", r_eve)
    by_class(rows, r_all)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"model": args.ctc, "epoch": int(ckpt["epoch"]),
             "daytime": r_day, "evening": r_eve, "pooled": r_all,
             "sessions": rows}, indent=2, default=float))
        print(f"\n  Written to {p}")


if __name__ == "__main__":
    main()
