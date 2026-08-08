"""
execution_tps.py — measure "moves per second while not hesitating".

Builds the L1 timing report (`coach/timing.py`) two ways for each held-out
session — once from BLE ground-truth onsets, once from the CTC-decoded
onsets the product would actually have — and reports the disagreement.
The metric is only worth shipping to the extent those two agree.

Three things it answers:

  1. **Is execution TPS accurate?** Decoded vs truth, per session, split
     by lighting regime (GAMEPLAN §1: never pooled).

  2. **How much does it depend on the threshold?** A pause threshold is a
     convention on this corpus — the inter-onset distribution has no
     valley between executing and thinking (see `coach/timing.py`). So the
     sweep is printed alongside the headline. A number that swings across
     the sweep is a re-parameterisation of the threshold, not a fact about
     the solver.

  3. **Do moves really come in clusters?** The burst-size distribution is
     the evidence for or against, and it is the entry point to phase and
     algorithm work (TODO §7D) — F2L pairs run ~7-10 moves, OLL ~9-11,
     PLL ~10-13, so a burst histogram piling up in that range is the
     hypothesis surviving a first contact with data.

Usage:
    python execution_tps.py --ctc checkpoints/move_ctc_spd_s0.pt \\
        --out results/2026-08-06/execution_tps_s0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from algorithm_gate import posteriorgram
from coach.timing import (
    PAUSE_FACTOR, burst_dicts, bursts, inter_onset, sweep_threshold,
    timing_report,
)
from ctc_decode import prefix_beam_decode
from eval_lighting import DAY_END, DAY_START, held_out_sessions, take_hour
from model import build_joint_from_ckpt
from onset_timing import frame_time_axis

#: Scramble takes are excluded by default. A scramble is read off a card
#: and turned at a steady, deliberate pace with no recognition pauses at
#: all — the earlier onset-timing run found 0 pauses in all three, and
#: their median IOI is 5-10x a solve's. Including them would drag every
#: aggregate here toward a regime the coach never analyses.
SKIP_SUFFIX = "_scramble"


def decoded_onsets(d: Path, model, device, tag: str, beam: int,
                   refresh: bool) -> np.ndarray | None:
    """Decoded onset times in the same wall clock as the BLE truth."""
    class_prob, _ = posteriorgram(d, model, device, refresh, tag)
    fts = frame_time_axis(d, class_prob.shape[0])
    if fts is None:
        return None
    _, frames = prefix_beam_decode(np.log(np.maximum(class_prob, 1e-12)),
                                   beam=beam)
    if len(frames) < 4:
        return None
    return fts[np.clip(np.asarray(frames, dtype=int), 0, len(fts) - 1)]


def rel_err(pred, true) -> float | None:
    if pred is None or true is None or not true:
        return None
    return (pred - true) / true


def score_session(d: Path, model, device, tag: str, beam: int,
                  refresh: bool) -> dict | None:
    z = np.load(d / "detector_stream_color.npz", allow_pickle=True)
    true_t = z["onset_ts"].astype(np.float64)
    if true_t.size < 8:
        return None

    pred_t = decoded_onsets(d, model, device, tag, beam, refresh)
    if pred_t is None:
        return None

    truth = timing_report(true_t)
    pred = timing_report(pred_t)
    if not (truth.get("usable") and pred.get("usable")):
        return None

    hour = take_hour(d)
    return {
        "session": d.name,
        "hour": hour,
        "evening": not (hour is not None and DAY_START <= hour < DAY_END),
        "truth": truth,
        "decoded": pred,
        "error": {
            "execution_tps": rel_err(pred["execution_tps"],
                                     truth["execution_tps"]),
            "execution_tps_robust": rel_err(pred["execution_tps_robust"],
                                            truth["execution_tps_robust"]),
            #: How far the threshold-based headline sits from the
            #: threshold-free anchor, on truth. Large means the 3x default
            #: has drifted and should be re-derived.
            "anchor_gap": rel_err(truth["execution_tps"],
                                  truth["execution_tps_robust"]),
            "span_tps": rel_err(pred["span_tps"], truth["span_tps"]),
            "hesitation_seconds": rel_err(pred["hesitation_seconds"],
                                          truth["hesitation_seconds"]),
            "hesitation_fraction_abs": (pred["hesitation_fraction"]
                                        - truth["hesitation_fraction"]),
            "n_bursts": pred["n_bursts"] - truth["n_bursts"],
        },
        "sweep_truth": sweep_threshold(true_t),
        "sweep_decoded": sweep_threshold(pred_t),
        "_true_burst_sizes": truth["burst_sizes"],
        "_true_ioi": inter_onset(true_t).tolist(),
        "_true_bursts": burst_dicts(bursts(true_t)),
    }


def pct(x) -> str:
    return "   --  " if x is None else f"{x * 100:+6.1f}%"


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n_sessions": 0}

    def med(key):
        vals = [r["error"][key] for r in rows if r["error"][key] is not None]
        return round(float(np.median(vals)), 4) if vals else None

    def absmed(key):
        vals = [abs(r["error"][key]) for r in rows
                if r["error"][key] is not None]
        return round(float(np.median(vals)), 4) if vals else None

    return {
        "n_sessions": len(rows),
        "execution_tps_bias": med("execution_tps"),
        "execution_tps_abs_err": absmed("execution_tps"),
        "execution_tps_robust_abs_err": absmed("execution_tps_robust"),
        "anchor_gap_abs": absmed("anchor_gap"),
        "hesitation_seconds_bias": med("hesitation_seconds"),
        "hesitation_seconds_abs_err": absmed("hesitation_seconds"),
        "hesitation_fraction_abs_err": absmed("hesitation_fraction_abs"),
        "n_bursts_bias": med("n_bursts"),
        "truth_execution_tps": round(float(np.median(
            [r["truth"]["execution_tps"] for r in rows])), 3),
        "truth_span_tps": round(float(np.median(
            [r["truth"]["span_tps"] for r in rows])), 3),
        "truth_hesitation_fraction": round(float(np.median(
            [r["truth"]["hesitation_fraction"] for r in rows])), 3),
    }


def print_regime(title: str, rows: list[dict], s: dict) -> None:
    print(f"\n  {title} — {s.get('n_sessions', 0)} session(s)")
    if not s.get("n_sessions"):
        print("    (none)")
        return
    print(f"    {'session':<32} {'exec TPS':>18} {'span TPS':>9} "
          f"{'hesit%':>7} {'exec err':>9} {'hesit s err':>11} {'burst d':>8}")
    for r in rows:
        t, p, e = r["truth"], r["decoded"], r["error"]
        print(f"    {r['session']:<32} "
              f"{t['execution_tps']:>7.2f} -> {p['execution_tps']:<6.2f} "
              f"{t['span_tps']:>9.2f} "
              f"{t['hesitation_fraction'] * 100:>6.1f}% "
              f"{pct(e['execution_tps']):>9} {pct(e['hesitation_seconds']):>11} "
              f"{e['n_bursts']:>+8}")
    print(f"    {'-' * 92}")
    print(f"    truth median: execution {s['truth_execution_tps']:.2f} TPS "
          f"vs span {s['truth_span_tps']:.2f} TPS, "
          f"{s['truth_hesitation_fraction'] * 100:.0f}% of the solve spent "
          f"hesitating")
    print(f"    decode error (median |rel|): execution TPS "
          f"{s['execution_tps_abs_err'] * 100:.1f}%, hesitation seconds "
          f"{s['hesitation_seconds_abs_err'] * 100:.1f}%, "
          f"burst count bias {s['n_bursts_bias']:+.0f}")
    print(f"    threshold-free anchor (1/median IOI): decode error "
          f"{s['execution_tps_robust_abs_err'] * 100:.1f}%, and it sits "
          f"{s['anchor_gap_abs'] * 100:.1f}% from the 3x headline on truth")


def print_sweep(rows: list[dict]) -> None:
    print(f"\n{'=' * 78}\n  THRESHOLD SENSITIVITY (ground truth)\n{'=' * 78}")
    print("  The pause threshold is a convention, not a discovered boundary —")
    print("  the inter-onset distribution has no valley. So: how much of the")
    print("  answer is the convention?\n")
    print(f"    {'x median':>9} {'thresh':>9} {'exec TPS':>10} "
          f"{'hesit%':>9} {'bursts':>8} {'burst size':>11}")
    factors = [s["factor"] for s in rows[0]["sweep_truth"]]
    for i, f in enumerate(factors):
        cells = [r["sweep_truth"][i] for r in rows]
        print(f"    {f:>8.1f}x {np.median([c['threshold_s'] for c in cells]):>8.2f}s"
              f" {np.median([c['execution_tps'] for c in cells]):>9.2f} "
              f"{np.median([c['hesitation_fraction'] for c in cells]) * 100:>8.1f}% "
              f"{np.median([c['n_bursts'] for c in cells]):>7.0f} "
              f"{np.median([c['median_burst_size'] for c in cells]):>10.0f}")
    print(f"\n  Read the spread, not the row: execution TPS moving little "
          f"across\n  the sweep means the metric is about the solver; moving "
          f"a lot means\n  it is about the threshold.")


def print_bursts(rows: list[dict]) -> None:
    sizes = [n for r in rows for n in r["_true_burst_sizes"]]
    if not sizes:
        return
    print(f"\n{'=' * 78}\n  BURST STRUCTURE — do moves come in clusters?"
          f"\n{'=' * 78}")
    arr = np.array(sizes)
    print(f"  {len(arr)} bursts over {len(rows)} solves at the default "
          f"{PAUSE_FACTOR:g}x threshold")
    print(f"  size: median {np.median(arr):.0f}  mean {arr.mean():.1f}  "
          f"p90 {np.percentile(arr, 90):.0f}  max {arr.max()}\n")
    edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 21, 26, 1000]
    hist, _ = np.histogram(arr, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        label = f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}"
        bar = "#" * int(round(c * 40 / max(hist.max(), 1)))
        print(f"    {label:>6} moves  {c:>4}  {bar}")
    print("\n  Reference lengths: F2L pair ~7-10 QTM, OLL ~9-11, PLL ~10-13.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctc", default="checkpoints/move_ctc_spd_s0.pt")
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--include-scrambles", action="store_true",
                    help="score scramble takes too; off by default because a "
                         "scramble has no recognition pauses at all")
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
    if not dirs:
        raise SystemExit("No sessions to score.")
    print(f"  {len(dirs)} held-out solve(s)")

    rows = []
    for d in dirs:
        try:
            r = score_session(d, model, device, tag, args.beam, args.refresh)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {d.name}: {type(exc).__name__}: {exc} — skipped")
            continue
        if r is None:
            print(f"  {d.name}: unusable — skipped")
            continue
        rows.append(r)

    if not rows:
        raise SystemExit("Nothing scored.")

    day = [r for r in rows if not r["evening"]]
    eve = [r for r in rows if r["evening"]]
    out = {
        "model": args.ctc, "epoch": int(ckpt["epoch"]), "beam": args.beam,
        "pause_factor": PAUSE_FACTOR,
        "daytime": summarise(day), "evening": summarise(eve),
        "sessions": [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in rows],
    }

    print(f"\n{'=' * 78}\n  EXECUTION TPS — moves per second, hesitation "
          f"removed (QTM)\n{'=' * 78}")
    print_regime("DAYTIME", day, out["daytime"])
    print_regime("EVENING", eve, out["evening"])
    print_sweep(rows)
    print_bursts(rows)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f"\n  Written to {p}")


if __name__ == "__main__":
    main()
