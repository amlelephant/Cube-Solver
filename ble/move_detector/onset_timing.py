"""
onset_timing.py — how accurate are the decoded move TIMESTAMPS?

WHY THIS EXISTS
---------------
Every number this pipeline has ever been scored on measures move
**identity**: MER, word accuracy, sub/ins/del. Timing has never been
measured at all. That was fine while the product was verification (the
count gate reads a count, not a clock) and it stops being fine the moment
the coach ships, because every L1 analytic — TPS curve, hesitation, pause
map, thinking-vs-turning — is a **difference of timestamps**.

So this file answers one question with three numbers:

  1. **Bias** — the systematic lag between a decoded onset and the BLE
     event it corresponds to. A constant offset is harmless for every
     metric here: all of them are differences, and a constant cancels.
     Measured anyway, because "it is constant" is a claim, not a given.

  2. **Jitter** — the spread around that bias. This is the number that
     matters, and it is worth being precise about why it does not simply
     average away. For two independent onset errors with variance s^2,
     the interval between them has variance 2*s^2: differencing *adds*
     noise, it does not cancel it. Averaging helps only across many
     solves, and only for aggregate statistics — it does nothing for the
     per-solve question "was there a pause here", which is a decision on
     a single interval.

  3. **The interval error itself** — measured directly rather than
     predicted from 1 and 2, so no assumption about independence or
     normality is load-bearing. This is the honest read.

And then the product question, which is a classification and not a
regression: taking the pause rule the coach would actually use, do the
pauses found in the decoded stream land where the BLE stream says they
are? Precision/recall on pause spans is the accept gate. A metric can
have a fine RMS error and still put the pauses in the wrong places.

WHAT THE TRUTH IS, AND ITS OWN FLOOR
------------------------------------
Truth is `onset_ts` from the prepared stream — the raw BLE wall-clock
timestamps, the same clock `frames.jsonl` stamps frames with. Two limits
on it, both already documented in GROUND_TRUTH_ARTIFACTS.md and both
reported below rather than assumed away:

  * **Frame quantisation.** A decoded onset can only ever name a frame,
    so its resolution is one frame interval (~33-60ms here). That is a
    floor on measurable timing error, not an error of the model.
  * **BLE tick collisions.** The cube reports on a 30ms tick and 10.1% of
    moves share a tick with their neighbour. Those intervals are not
    resolvable in the truth either, so they are reported separately.

Sessions are the honest holdout (`eval_lighting.held_out_sessions`), and
results are split morning/evening and never pooled, per GAMEPLAN §1 — a
pooled mean here would hide the same bimodality it hides for accuracy.

Usage:
    python onset_timing.py --ctc checkpoints/move_ctc_spd_s0.pt \\
        --out results/2026-08-06/onset_timing_s0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from algorithm_gate import posteriorgram
# The pause rule lives in coach/timing.py and is imported, never restated:
# it is the definition the shipped metric uses, and a second copy here
# would let the thing being validated drift away from the thing validated.
from coach.timing import PAUSE_FACTOR, PAUSE_FLOOR_S, pause_spans
from ctc_decode import prefix_beam_decode
from eval_lighting import DAY_END, DAY_START, held_out_sessions, take_hour
from model import build_joint_from_ckpt

#: A decoded pause counts as finding a true pause when it covers at least
#: this fraction of it. Overlap rather than midpoint distance: a pause is
#: an interval, and the coach's claim is about when the solver was idle.
PAUSE_MIN_OVERLAP = 0.5

#: Intervals at or below this in the TRUTH are inside the BLE tick's own
#: resolution and are excluded from the "resolvable" split.
BLE_TICK_S = 0.030


def frame_time_axis(d: Path, n_expected: int) -> np.ndarray | None:
    """
    Real capture timestamp per frame, in the npz's index space.

    NOT `frame / fps`. `fps` is a session average (prepare_data computes
    n/duration) and webcam frame intervals genuinely jitter — a spot check
    shows 39.6ms and 60.1ms adjacent in the same session. Using nominal fps
    would inject that jitter straight into the metric being measured here
    and then attribute it to the model.

    prepare_data drops indexed frames whose file is missing from disk, so
    that filter is mirrored here — but only when the lengths actually
    disagree, since stat()-ing thousands of files per session is not free.
    """
    f = d / "frames.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(line) for line in open(f) if line.strip()]
    if len(recs) != n_expected:
        frames_dir = d / "frames"
        recs = [r for r in recs if (frames_dir / r["file"]).exists()]
    if len(recs) != n_expected:
        return None
    return np.array([r["ts"] for r in recs], dtype=np.float64)


def align(pred: list[int], true: list[int]) -> list[tuple[int, int, bool]]:
    """
    Levenshtein alignment with backtrace -> [(pred_i, true_j, exact)].

    Only *matched* moves have a defined timing error: an insertion
    corresponds to no real event and a deletion has no decoded onset to
    time. Substitutions are kept and flagged, because the interesting
    question is whether a move called wrong was still detected at the
    right instant — for a timing metric it usually was, and dropping them
    would throw away real timing evidence.

    Diagonal wins ties, which keeps the alignment maximally paired rather
    than sliding through equal-cost indel runs.
    """
    n, m = len(true), len(pred)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ti = true[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ti == pred[j - 1] else 1
            d[i, j] = min(d[i - 1, j - 1] + cost, d[i - 1, j] + 1,
                          d[i, j - 1] + 1)

    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        cost = 0 if true[i - 1] == pred[j - 1] else 1
        if d[i, j] == d[i - 1, j - 1] + cost:
            pairs.append((j - 1, i - 1, cost == 0))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1          # deletion: a move we missed
        else:
            j -= 1          # insertion: a phantom
    pairs.reverse()
    return pairs


def match_spans(true_spans, pred_spans) -> int:
    """Greedy one-to-one overlap match; returns the true-positive count."""
    used: set[int] = set()
    tp = 0
    for a, b in true_spans:
        dur = b - a
        best, best_frac = None, 0.0
        for k, (c, e) in enumerate(pred_spans):
            if k in used:
                continue
            overlap = min(b, e) - max(a, c)
            frac = (overlap / dur) if dur > 0 else 0.0
            if frac > best_frac:
                best, best_frac = k, frac
        if best is not None and best_frac >= PAUSE_MIN_OVERLAP:
            used.add(best)
            tp += 1
    return tp


def describe(errs: np.ndarray) -> dict:
    """The summary applied to every error population here."""
    if errs.size == 0:
        return {"n": 0}
    med = float(np.median(errs))
    return {
        "n": int(errs.size),
        "median_ms": round(med * 1e3, 1),
        "mean_ms": round(float(errs.mean()) * 1e3, 1),
        "std_ms": round(float(errs.std(ddof=1)) * 1e3, 1) if errs.size > 1
        else 0.0,
        # Median absolute deviation about the median, rescaled to a normal
        # sigma. Reported next to std because a handful of misaligned pairs
        # can dominate std while leaving the bulk of the distribution
        # untouched, and the two disagreeing is itself the finding.
        "mad_ms": round(float(np.median(np.abs(errs - med))) * 1.4826e3, 1),
        "iqr_ms": round(float(np.percentile(errs, 75)
                              - np.percentile(errs, 25)) * 1e3, 1),
        "p5_ms": round(float(np.percentile(errs, 5)) * 1e3, 1),
        "p95_ms": round(float(np.percentile(errs, 95)) * 1e3, 1),
        "abs_median_ms": round(float(np.median(np.abs(errs))) * 1e3, 1),
    }


def score_session(d: Path, model, device, tag: str, beam: int,
                  refresh: bool) -> dict | None:
    class_prob, stream = posteriorgram(d, model, device, refresh, tag)
    z = np.load(d / "detector_stream_color.npz", allow_pickle=True)
    onset_idx = z["onset_idx"].astype(int)
    onset_ts = z["onset_ts"].astype(np.float64)
    onset_class = z["onset_class"].astype(int)
    if len(onset_ts) < 4 or len(onset_ts) != len(onset_class):
        return None

    fts = frame_time_axis(d, class_prob.shape[0])
    if fts is None:
        return None

    labels, frames = prefix_beam_decode(np.log(np.maximum(class_prob, 1e-12)),
                                        beam=beam)
    if len(labels) < 4:
        return None
    pred_t = fts[np.clip(np.asarray(frames, dtype=int), 0, len(fts) - 1)]

    pairs = align(labels, list(onset_class))
    if not pairs:
        return None

    # -- 1 & 2: per-onset error --------------------------------------------
    errs = np.array([pred_t[pi] - onset_ts[tj] for pi, tj, _ in pairs])
    exact = np.array([ex for _, _, ex in pairs], dtype=bool)

    # -- 3: interval error, on pairs consecutive in BOTH streams. An
    #       insertion or deletion between two matches splits or merges the
    #       interval, so those are a different error mode (measured below
    #       as pause precision/recall) and would contaminate this one.
    ioi_err, ioi_true_all = [], []
    for k in range(len(pairs) - 1):
        (p0, t0, _), (p1, t1, _) = pairs[k], pairs[k + 1]
        if p1 != p0 + 1 or t1 != t0 + 1:
            continue
        true_ioi = onset_ts[t1] - onset_ts[t0]
        ioi_err.append((pred_t[p1] - pred_t[p0]) - true_ioi)
        ioi_true_all.append(true_ioi)
    ioi_err = np.array(ioi_err)
    ioi_true_all = np.array(ioi_true_all)
    resolvable = ioi_true_all > BLE_TICK_S

    # -- the product question: pauses ---------------------------------------
    true_thr, true_spans = pause_spans(onset_ts)
    pred_thr, pred_spans = pause_spans(pred_t)
    tp = match_spans(true_spans, pred_spans)

    frame_dt = float(np.median(np.diff(fts))) if len(fts) > 1 else 0.0
    hour = take_hour(d)
    return {
        "session": d.name,
        "hour": hour,
        "evening": not (hour is not None and DAY_START <= hour < DAY_END),
        "n_true": len(onset_ts),
        "n_pred": len(labels),
        "n_matched": len(pairs),
        "n_exact": int(exact.sum()),
        "frame_dt_ms": round(frame_dt * 1e3, 1),
        "onset_err": describe(errs),
        "onset_err_exact_only": describe(errs[exact]),
        "ioi_err": describe(ioi_err),
        "ioi_err_resolvable": describe(ioi_err[resolvable]
                                       if ioi_err.size else ioi_err),
        "ioi_collisions": int((~resolvable).sum()),
        "pause": {
            "true_thr_ms": round(true_thr * 1e3, 1),
            "pred_thr_ms": round(pred_thr * 1e3, 1),
            "n_true": len(true_spans), "n_pred": len(pred_spans),
            "tp": tp,
            "fp": len(pred_spans) - tp, "fn": len(true_spans) - tp,
            "true_pause_s": round(sum(b - a for a, b in true_spans), 2),
            "pred_pause_s": round(sum(b - a for a, b in pred_spans), 2),
        },
        # Kept raw so the pooled distributions below are computed from the
        # data, not from per-session summaries of it.
        "_errs": errs.tolist(),
        "_errs_exact": errs[exact].tolist(),
        "_ioi_err": ioi_err.tolist(),
        "_ioi_err_resolvable": (ioi_err[resolvable].tolist()
                                if ioi_err.size else []),
    }


def pool(rows: list[dict], key: str) -> dict:
    return describe(np.array([v for r in rows for v in r[key]]))


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n_sessions": 0}
    tp = sum(r["pause"]["tp"] for r in rows)
    fp = sum(r["pause"]["fp"] for r in rows)
    fn = sum(r["pause"]["fn"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_sessions": len(rows),
        "onset_err": pool(rows, "_errs"),
        "onset_err_exact_only": pool(rows, "_errs_exact"),
        "ioi_err": pool(rows, "_ioi_err"),
        "ioi_err_resolvable": pool(rows, "_ioi_err_resolvable"),
        "median_frame_dt_ms": round(float(np.median(
            [r["frame_dt_ms"] for r in rows])), 1),
        "pause": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "true_pause_s": round(sum(r["pause"]["true_pause_s"]
                                      for r in rows), 1),
            "pred_pause_s": round(sum(r["pause"]["pred_pause_s"]
                                      for r in rows), 1),
        },
    }


def print_block(title: str, s: dict) -> None:
    print(f"\n  {title}  ({s.get('n_sessions', 0)} sessions)")
    if not s.get("n_sessions"):
        print("    (none)")
        return
    print(f"    {'population':<26} {'n':>5} {'median':>9} {'MAD':>8} "
          f"{'std':>8} {'IQR':>8} {'p5..p95':>16}")
    for label, key in (("per-onset error", "onset_err"),
                       ("  correctly-ID'd only", "onset_err_exact_only"),
                       ("interval (IOI) error", "ioi_err"),
                       ("  IOI > 30ms BLE tick", "ioi_err_resolvable")):
        e = s[key]
        if not e.get("n"):
            continue
        print(f"    {label:<26} {e['n']:>5} {e['median_ms']:>8.1f}m "
              f"{e['mad_ms']:>7.1f}m {e['std_ms']:>7.1f}m {e['iqr_ms']:>7.1f}m "
              f"{e['p5_ms']:>7.0f}..{e['p95_ms']:<7.0f}")
    print(f"    frame interval (quantisation floor): "
          f"{s['median_frame_dt_ms']:.1f}ms")
    p = s["pause"]
    print(f"    pauses: precision {p['precision']:.3f}  recall "
          f"{p['recall']:.3f}  F1 {p['f1']:.3f}   "
          f"(tp {p['tp']}  fp {p['fp']}  fn {p['fn']})")
    print(f"    total pause seconds: true {p['true_pause_s']}  "
          f"decoded {p['pred_pause_s']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctc", default="checkpoints/move_ctc_spd_s0.pt",
                    help="CTC checkpoint; defaults to the one verify_solve "
                         "ships with, so this measures the deployed timing")
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="explicit session dirs; default is the honest "
                         "holdout for --ctc")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute cached posteriorgrams")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ctc, map_location=device)
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    tag = f"{Path(args.ctc).stem}_e{ckpt['epoch']}"
    print(f"  model {args.ctc} (epoch {ckpt['epoch']}) on {device}")

    if args.sessions:
        dirs = [Path(s) for s in args.sessions]
    else:
        dirs = held_out_sessions([Path(args.ctc)])
    dirs = [d for d in dirs if (d / "detector_stream_color.npz").exists()]
    if not dirs:
        raise SystemExit("No sessions to score.")
    print(f"  {len(dirs)} held-out session(s)\n")

    rows = []
    for d in dirs:
        try:
            r = score_session(d, model, device, tag, args.beam, args.refresh)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {d.name}: {type(exc).__name__}: {exc} — skipped")
            continue
        if r is None:
            print(f"  {d.name}: unusable (frames/labels mismatch) — skipped")
            continue
        rows.append(r)
        e, i = r["onset_err"], r["ioi_err"]
        print(f"  {r['session']:<34} {'EVE' if r['evening'] else 'day'} "
              f"{r['n_true']:>3}mv  onset med {e['median_ms']:>7.1f}ms "
              f"MAD {e['mad_ms']:>6.1f}ms | IOI med "
              f"{i.get('median_ms', 0):>6.1f}ms MAD {i.get('mad_ms', 0):>6.1f}ms"
              f" | pause tp{r['pause']['tp']}/fp{r['pause']['fp']}"
              f"/fn{r['pause']['fn']}")

    if not rows:
        raise SystemExit("Nothing scored.")

    day = [r for r in rows if not r["evening"]]
    eve = [r for r in rows if r["evening"]]
    out = {
        "model": args.ctc, "epoch": int(ckpt["epoch"]), "beam": args.beam,
        "pause_rule": {"factor": PAUSE_FACTOR, "floor_s": PAUSE_FLOOR_S,
                       "min_overlap": PAUSE_MIN_OVERLAP},
        "daytime": summarise(day), "evening": summarise(eve),
        "sessions": [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in rows],
    }

    print(f"\n{'=' * 78}")
    print("  ONSET TIMING — split by lighting regime, never pooled "
          "(GAMEPLAN §1)")
    print(f"{'=' * 78}")
    print_block("DAYTIME", out["daytime"])
    print_block("EVENING", out["evening"])
    print("\n  Reading it: median is bias (cancels in any interval); MAD/IQR "
          "\n  is jitter (does NOT cancel — differencing two independent "
          "\n  errors adds their variances). The IOI row is the direct "
          "\n  measurement and is what the hesitation metric actually eats.")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f"\n  Written to {p}")


if __name__ == "__main__":
    main()
