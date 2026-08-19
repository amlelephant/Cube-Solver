"""
coach_report.py — run the shipped coach analysis on real solves.

Where `metric_robustness.py` measures how far the decode is from truth,
this runs `coach/report.py` exactly as the product will and prints what a
user would actually see, on real held-out solves. It is the "does this
look like a product" check, not a measurement.

Two views:

  --inventory   the metric registry: every metric the coach is allowed to
                report, its measured error in each lighting regime, and
                whether it ships, ships flagged, or is suppressed there.

  (default)     per-solve analysis for each held-out session, from the
                decoded move stream, with the ground-truth value beside it
                so the printed numbers can be trusted at a glance.

Usage:
    python coach_report.py --ctc checkpoints/move_ctc_spd_s0.pt
    python coach_report.py --inventory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from algorithm_gate import posteriorgram
from coach.report import BY_KEY, registry_table, solve_report
from ctc_decode import prefix_beam_decode
from eval_lighting import DAY_END, DAY_START, held_out_sessions, take_hour
from model import build_joint_from_ckpt
from onset_timing import frame_time_axis
from reconstruct import WCA12

#: Session name suffixes that are NOT solves and must never be scored as
#: one. `_scan` is verify_solve.py phase 3 - the post-timer verification
#: window, which is deliberately move-free; counted as a solve it would
#: read as a legitimate attempt with ~0 moves and drag every aggregate
#: here down. Suffix classification is load-bearing, so it lives in one
#: tuple per file rather than in an inline endswith().
SKIP_SUFFIXES = ("_scramble", "_scan")
SKIP_SUFFIX = SKIP_SUFFIXES[0]   # kept: existing single-suffix callers

#: How each metric is rendered. Percent-style fractions read far better as
#: percentages, and seconds want more precision than turns per second.
FMT = {
    "frac": lambda v: f"{v * 100:.1f}%",
    "frac/face": lambda v: ", ".join(f"{k} {x * 100:.0f}%"
                                     for k, x in v.items() if x > 0),
    "s": lambda v: f"{v:.3f}s" if v < 1 else f"{v:.2f}s",
    "TPS": lambda v: f"{v:.2f}",
    "QTM": lambda v: f"{v:.0f}",
    "cv": lambda v: f"{v:.3f}",
    "0-1": lambda v: f"{v:.3f}",
    "moves": lambda v: f"{v:.2f}",
}

MARK = {"high": " ", "caution": "~", "suppressed": "x"}


def fmt(unit: str, value):
    try:
        return FMT.get(unit, str)(value)
    except (TypeError, ValueError):
        return str(value)


#: What the registry's accuracy figures were measured on. Stated here rather
#: than typed into the footnote so the two cannot drift apart the next time
#: metric_robustness.py is re-run on a bigger corpus — which is exactly what
#: happened between 2026-08-06 (6 solves) and 2026-08-10 (14).
HOLDOUT_DATE = "2026-08-10"
HOLDOUT_DAY, HOLDOUT_EVE = 9, 5
HOLDOUT_N = HOLDOUT_DAY + HOLDOUT_EVE


def print_inventory() -> None:
    rows = registry_table()
    print(f"\n{'=' * 90}")
    print("  COACH METRIC INVENTORY — everything the product may report")
    print(f"{'=' * 90}")
    print(f"  {'metric':<24} {'label':<26} {'kind':<7} "
          f"{'day':>13} {'evening':>13}  {'day':<10} {'evening'}")
    print(f"  {'':<24} {'':<26} {'':<7} "
          f"{'med / worst':>13} {'med / worst':>13}")
    print(f"  {'-' * 100}")
    for r in rows:
        print(f"  {r['key']:<24} {r['label']:<26} {r['kind']:<7} "
              f"{r['daytime_err_pct']:>5.1f}/{r['daytime_worst_pct']:<6.1f} "
              f"{r['evening_err_pct']:>5.1f}/{r['evening_worst_pct']:<6.1f}  "
              f"{r['daytime']:<10} {r['evening']}")
    n_day = sum(r["daytime"] != "suppressed" for r in rows)
    n_eve = sum(r["evening"] != "suppressed" for r in rows)
    print(f"\n  {len(rows)} metrics shipped: {n_day} usable in daytime, "
          f"{n_eve} in evening.")
    print(f"  Both columns are the worse of two seeds vs BLE truth on "
          f"{HOLDOUT_N} held-out solves")
    print(f"  ({HOLDOUT_DAY} daytime / {HOLDOUT_EVE} evening, "
          f"re-measured {HOLDOUT_DATE}):")
    print("  'med' is the median session, 'worst' the worst — read 'worst' "
          "for what a\n  user meets on a bad solve. The evening column "
          f"rests on n={HOLDOUT_EVE}, so its\n  medians are still weak and "
          "its worst column is the honest one.")


def analyse(d: Path, model, device, tag: str, beam: int):
    """(decoded report, truth report, session meta) for one session."""
    z = np.load(d / "detector_stream_color.npz", allow_pickle=True)
    true_t = z["onset_ts"].astype(np.float64)
    true_w = [WCA12[c] for c in z["onset_class"].astype(int)]
    if true_t.size < 8:
        return None

    class_prob, _ = posteriorgram(d, model, device, False, tag)
    fts = frame_time_axis(d, class_prob.shape[0])
    if fts is None:
        return None
    labels, frames = prefix_beam_decode(np.log(np.maximum(class_prob, 1e-12)),
                                        beam=beam)
    if len(labels) < 8:
        return None
    pred_t = fts[np.clip(np.asarray(frames, dtype=int), 0, len(fts) - 1)]

    hour = take_hour(d)
    evening = not (hour is not None and DAY_START <= hour < DAY_END)
    # Onsets relative to the first move. The product uses the timer start;
    # here there is no timer, and every metric in the registry is
    # translation-invariant, so the choice of origin does not affect them.
    return (solve_report(pred_t - pred_t[0], [WCA12[c] for c in labels],
                         evening=evening),
            solve_report(true_t - true_t[0], true_w, evening=evening,
                         include_suppressed=True),
            {"session": d.name, "hour": hour, "evening": evening})


def trained_sessions(ckpt: dict) -> set[str]:
    """Every session name this checkpoint has already seen, train or val.

    Only interesting when `--sessions` is passed: the default path is
    `held_out_sessions()`, which is that set's complement by construction.
    Given an explicit list it is the one thing that decides whether a
    printed agreement figure means anything, so it travels with the report
    rather than being reconstructed later from the checkpoint.
    """
    return (set(ckpt.get("train_session_names") or [])
            | set(ckpt.get("val_session_names") or []))


def print_solve(rep: dict, truth: dict, meta: dict) -> None:
    seen = "" if meta.get("held_out", True) else "  TRAINED ON"
    print(f"\n  {meta['session']}   "
          f"[{rep['regime']}, {meta['hour']:02d}:00]{seen}")
    print(f"  {'-' * 84}")
    print(f"    {'metric':<26} {'decoded':>22} {'truth':>22}   err")
    for key, m in rep["metrics"].items():
        tv = truth["metrics"].get(key, {}).get("value")
        dv = m["value"]
        if m.get("series"):
            # A curve has no single value to print here, and forcing one
            # would be inventing a scalar the registry deliberately does not
            # have. Its agreement with truth is measured properly by
            # metric_robustness.py, on a shared time grid; this line just
            # says the series is present and how it starts and ends.
            pts = dv or []
            span = (f"{pts[0]['tps']:.2f} -> {pts[-1]['tps']:.2f} TPS"
                    if pts else "--")
            tpts = tv or []
            tspan = (f"{tpts[0]['tps']:.2f} -> {tpts[-1]['tps']:.2f} TPS"
                     if tpts else "--")
            print(f"    {MARK[m['confidence']]}{m['label']:<25} "
                  f"{span:>22} {tspan:>22}   {len(pts):>4} pts")
            continue
        if isinstance(dv, dict) or tv in (None, 0):
            err = ""
        else:
            err = f"{(dv - tv) / abs(tv) * 100:+6.1f}%"
        print(f"    {MARK[m['confidence']]}{m['label']:<25} "
              f"{fmt(m['unit'], dv):>22} "
              f"{fmt(m['unit'], tv) if tv is not None else '--':>22}   {err}")
    if rep["suppressed"]:
        print(f"    (suppressed in this regime: "
              f"{', '.join(rep['suppressed'])})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctc", default="checkpoints/move_ctc_spd_s0.pt")
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--inventory", action="store_true",
                    help="print the metric registry and exit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.inventory:
        print_inventory()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ctc, map_location=device)
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    tag = f"{Path(args.ctc).stem}_e{ckpt['epoch']}"
    print(f"  model {args.ctc} (epoch {ckpt['epoch']}) on {device}")

    dirs = ([Path(s) for s in args.sessions] if args.sessions
            else held_out_sessions([Path(args.ctc)]))
    dirs = [d for d in dirs
            if (d / "detector_stream_color.npz").exists()
            and not d.name.endswith(SKIP_SUFFIX)]
    seen = trained_sessions(ckpt)

    print(f"\n{'=' * 90}")
    print("  COACH ANALYSIS — real solves, as the product would render them")
    print(f"{'=' * 90}")
    print("  '~' = shipped but flagged (measured 8-15% error);  "
          "'x' = suppressed in this regime")
    n_seen = sum(d.name in seen for d in dirs)
    if n_seen:
        print(f"  {n_seen} of {len(dirs)} were TRAINED ON — their agreement "
              "with truth is not a measurement.\n  The registry's "
              "accuracy_pct figures below are the held-out ones and stand.")

    out = []
    for d in dirs:
        try:
            got = analyse(d, model, device, tag, args.beam)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {d.name}: {type(exc).__name__}: {exc} — skipped")
            continue
        if got is None:
            continue
        rep, truth, meta = got
        meta["held_out"] = d.name not in seen
        print_solve(rep, truth, meta)
        out.append({**meta, "decoded": rep, "truth": truth})

    print_inventory()

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model": args.ctc, "solves": out},
                                indent=2, default=float))
        print(f"\n  Written to {p}")


if __name__ == "__main__":
    main()
