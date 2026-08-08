"""
speed_ceiling.py — where does move detection STRUCTURALLY stop working as
solves get faster, independent of how good any model is?

The distinction from speed_sim.py
---------------------------------
`speed_sim.py` drops frames from real footage and asks a real checkpoint how
many moves it still counts. That measures THIS MODEL at speed, and its answer
moves whenever the model is retrained (aug44 -> spd moved every level).

This file asks the complementary question and its answer never moves: given
the corpus's real inter-onset timing, at what speed do two moves stop being
REPRESENTABLE — by the sampling grid, by CTC's collapse rule, by peak-picking's
local-max rule? That is a property of the frame rate and the decoder algebra,
not of any weights. It is the number that says when to stop training and start
buying a faster camera.

The three structural limits, in increasing order of severity
------------------------------------------------------------
  1. SAMPLING. Two onsets landing in the SAME frame are one event. No
     decoder, no architecture and no amount of data recovers them. Only fps
     moves this.

  2. CTC COLLAPSE. Two onsets of the SAME class need an intervening blank
     frame or they collapse to one (`R R` -> `R`). Different-class pairs are
     unaffected — `R U` at one frame apart is fine. So CTC's floor is driven
     by REPEATED FACES specifically, which is why half turns keep coming up.

  3. PEAK-PICKING. `decode.peak_pick` needs a strict local maximum, and
     `MIN_SEP` rejects anything closer. Both members of a pair closer than
     MIN_SEP can never be reported, whatever their classes. This is the
     harshest of the three and it is why the CTC arm exists.

Two speed models, because they disagree and the disagreement is the point
------------------------------------------------------------------------
Scaling every interval by 1/s is the WRONG model of a cuber getting faster.
Real improvement removes hesitation first — `coach/timing.py` measures
hesitation at ~53% of a solve — and only compresses the execution bursts once
the pauses are gone. Since collisions live entirely inside the bursts, the two
models give very different ceilings at the same nominal TPS:

  uniform          every interval / s. Compresses bursts immediately.
  hesitation_first speed_sim.py's two stages: drain the pause slack down to
                   the pause threshold first, and only then compress
                   everything. Bursts stay at their recorded tightness until
                   the hesitation budget is spent.

`hesitation_first` is the honest model of a human getting faster and is the
one to quote. `uniform` is the pessimistic bound and is what you would get by
literally speeding up a video.

GROUND TRUTH CAVEAT, and it binds at exactly the speeds in question
--------------------------------------------------------------------
Onset times come from the BLE move log, whose notification tick is ~30ms —
about ONE frame at 30fps. So a measured gap of 0 could be a true gap of up to
30ms, and 10.1% of corpus moves already share a tick
(GROUND_TRUTH_ARTIFACTS.md). The collision counts below are therefore an
UPPER bound on collisions at the tight end: some pairs recorded as
simultaneous were not. This inflates the pessimism of every row, and it cannot
be corrected from this data — it needs a truth source finer than the cube's
own tick.

Usage:
    python speed_ceiling.py
    python speed_ceiling.py --fps 30 60 120 --out results/2026-08-08/ceiling.json
    python speed_ceiling.py --include-scrambles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coach.timing import PAUSE_FACTOR, PAUSE_FLOOR_S, inter_onset

SESSION_ROOT = Path("../training_data")
STREAM_FILE = "detector_stream_color.npz"

#: Matches decode.MIN_SEP. Peak-picking needs `abs(i-j) >= MIN_SEP`, so a gap
#: of MIN_SEP-1 frames or less is unreportable. Imported as a literal rather
#: than from decode.py so this file stays torch-free and runs anywhere.
MIN_SEP = 2

#: TPS levels. The first block matches speed_sim.TPS_LEVELS so the two files'
#: rows line up; the tail extends past human ability to locate the wall.
TPS_LEVELS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

#: Reference points for a ~55-move CFOP solve, for reading the tables:
#:   15s -> 3.7 TPS   12s -> 4.6   10s -> 5.5   8s -> 6.9   6s -> 9.2 (WR pace)

SKIP_SUFFIX = "_scramble"      # same exclusion rule as execution_tps.py


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_sessions(root: Path, include_scrambles: bool) -> list[dict]:
    """Every prepared colour stream with at least 2 onsets."""
    out = []
    for npz in sorted(root.glob(f"*/{STREAM_FILE}")):
        name = npz.parent.name
        if not include_scrambles and name.endswith(SKIP_SUFFIX):
            continue
        d = np.load(npz, allow_pickle=True)
        ts = np.asarray(d["onset_ts"], dtype=np.float64)
        cls = np.asarray(d["onset_class"], dtype=np.int64)
        if ts.size < 2:
            continue
        order = np.argsort(ts)
        ts, cls = ts[order], cls[order]
        span = float(ts[-1] - ts[0])
        if span <= 0:
            continue
        out.append({
            "name": name,
            "ts": ts - ts[0],
            "cls": cls,
            "n": int(ts.size),
            "span": span,
            "tps": float(ts.size) / span,
            # Adjacent pairs the BLE log records as EXACTLY simultaneous:
            # both moves arrived in one 30ms notification packet, so their
            # true separation is unknown in [0, 30ms). Measured at 0.99% of
            # corpus pairs. They read as unresolvable at every fps, which is
            # a fact about the TRUTH SOURCE, not about the camera — see
            # --exclude-tick-collisions.
            "tick_collision": np.diff(ts) <= 0,
        })
    return out


# ---------------------------------------------------------------------------
# Speed models
# ---------------------------------------------------------------------------

def pause_threshold(ts: np.ndarray) -> float:
    """coach.timing.pause_threshold, inlined on the raw array."""
    ioi = inter_onset(ts)
    if ioi.size < 2:
        return PAUSE_FLOOR_S
    return max(PAUSE_FACTOR * float(np.median(ioi)), PAUSE_FLOOR_S)


def retime(ts: np.ndarray, target_tps: float, model: str) -> np.ndarray | None:
    """
    Re-time an onset sequence to `target_tps`. Returns None if the target is
    slower than the recording (we never pad time — that would invent
    hesitation the solver did not have).
    """
    n = ts.size
    ioi = inter_onset(ts)
    total = float(ts[-1] - ts[0])
    want = n / float(target_tps)
    if want >= total:
        return None                       # already faster than the target

    if model == "uniform":
        new_ioi = ioi * (want / total)

    elif model == "hesitation_first":
        thr = pause_threshold(ts)
        slack = np.maximum(ioi - thr, 0.0)      # removable hesitation
        floor_total = total - float(slack.sum())  # all pauses drained to thr
        if want >= floor_total:
            # Stage 1 only: remove the fraction of slack that gets us there.
            f = (total - want) / max(float(slack.sum()), 1e-12)
            new_ioi = ioi - f * slack
        else:
            # Stage 1 exhausted; Stage 2 compresses what is left uniformly.
            drained = ioi - slack
            new_ioi = drained * (want / max(floor_total, 1e-12))
    else:
        raise ValueError(model)

    return np.concatenate([[0.0], np.cumsum(new_ioi)])


# ---------------------------------------------------------------------------
# Structural collision counting
# ---------------------------------------------------------------------------

def count_losses(ts: np.ndarray, cls: np.ndarray, fps: float,
                 exclude: np.ndarray | None = None) -> dict:
    """
    Moves that CANNOT be reported, by mechanism, at this timing and fps.

    Every mechanism is counted as adjacent violating pairs. For a run of k
    mutually colliding onsets that yields k-1, which is exactly the number
    lost when only one of the run can be emitted — so this generalises past
    pairs without special-casing them.

    `exclude` masks out adjacent pairs whose ground-truth separation is
    unknown (the BLE tick collisions). Those pairs are dropped from the
    numerator AND the denominator: counting them as losses measures the
    smart cube's packet rate, and counting them as successes would assume a
    separation the data does not contain. Neither is defensible, so they are
    removed from the question.
    """
    frames = np.round(np.asarray(ts) * fps).astype(np.int64)
    gaps = np.diff(frames)
    same = cls[:-1] == cls[1:]
    keep = np.ones(gaps.shape, dtype=bool) if exclude is None else ~exclude
    n_excluded = int((~keep).sum())

    same_frame = (gaps <= 0) & keep                    # 1. sampling
    ctc_strict = same & (gaps <= 1) & keep             # 2. needs a blank
    ctc_conservative = same & (gaps <= 2) & keep       # 2b. GAMEPLAN's <=2
    peak = (gaps <= (MIN_SEP - 1)) & keep              # 3. closer than MIN_SEP

    n = int(cls.size) - n_excluded
    return {
        "n_onsets": n,
        "n_excluded": n_excluded,
        "lost_sampling": int(same_frame.sum()),
        "lost_ctc": int(ctc_strict.sum()),
        "lost_ctc_conservative": int(ctc_conservative.sum()),
        "lost_peakpick": int(peak.sum()),
        "ceil_sampling": 1.0 - same_frame.sum() / n,
        "ceil_ctc": 1.0 - ctc_strict.sum() / n,
        "ceil_ctc_conservative": 1.0 - ctc_conservative.sum() / n,
        "ceil_peakpick": 1.0 - peak.sum() / n,
    }


def sweep(sessions: list[dict], fps_levels: list[float],
          models: list[str], exclude_ticks: bool = False) -> dict:
    """Per-session ceilings at every (model, tps, fps) combination."""
    results: dict = {}
    for model in models:
        for fps in fps_levels:
            rows = []
            for tps in TPS_LEVELS:
                per_session = []
                for s in sessions:
                    ts = retime(s["ts"], tps, model)
                    if ts is None:
                        continue
                    m = count_losses(ts, s["cls"], fps,
                                     s["tick_collision"] if exclude_ticks
                                     else None)
                    m["name"] = s["name"]
                    per_session.append(m)
                if not per_session:
                    continue
                # MEAN OF PER-SESSION ceilings, matching the ship metric's
                # footing (GAMEPLAN §1: a user experiences their own solve,
                # not a pooled onset-weighted average).
                rows.append({
                    "tps": tps,
                    "n_sessions": len(per_session),
                    **{k: float(np.mean([p[k] for p in per_session]))
                       for k in ("ceil_sampling", "ceil_ctc",
                                 "ceil_ctc_conservative", "ceil_peakpick")},
                    "worst_ctc": float(min(p["ceil_ctc"] for p in per_session)),
                    "worst_peakpick": float(
                        min(p["ceil_peakpick"] for p in per_session)),
                })
            results[f"{model}@{fps:g}fps"] = rows
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n  {title}")
    print("  " + "-" * 76)
    print(f"  {'TPS':>4} {'n':>4} | {'sampling':>9} {'CTC':>9} "
          f"{'CTC(<=2)':>9} {'peak-pick':>10} | {'worst CTC':>10}")
    print("  " + "-" * 76)
    for r in rows:
        print(f"  {r['tps']:>4} {r['n_sessions']:>4} | "
              f"{r['ceil_sampling']*100:>8.2f}% {r['ceil_ctc']*100:>8.2f}% "
              f"{r['ceil_ctc_conservative']*100:>8.2f}% "
              f"{r['ceil_peakpick']*100:>9.2f}% | "
              f"{r['worst_ctc']*100:>9.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=SESSION_ROOT)
    ap.add_argument("--fps", type=float, nargs="+", default=[30.0, 60.0, 120.0])
    ap.add_argument("--models", nargs="+",
                    default=["hesitation_first", "uniform"])
    ap.add_argument("--include-scrambles", action="store_true")
    ap.add_argument("--exclude-tick-collisions", action="store_true",
                    help="Drop adjacent pairs the BLE log records as exactly "
                         "simultaneous (0.99%% of corpus pairs). Their true "
                         "separation is unknown, so with them included every "
                         "ceiling carries a fixed ~1%% floor that no camera "
                         "can lift. Use this to see the physics.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sessions = load_sessions(args.root, args.include_scrambles)
    if not sessions:
        raise SystemExit(f"No prepared colour streams under {args.root}. "
                         "Run `prepare_data.py --color` first.")

    tps_all = np.array([s["tps"] for s in sessions])
    n_all = sum(s["n"] for s in sessions)
    print(f"\n  Corpus: {len(sessions)} sessions, {n_all} onsets")
    print(f"  Recorded TPS: median {np.median(tps_all):.2f}, "
          f"range {tps_all.min():.2f}-{tps_all.max():.2f}")
    print(f"  Scramble takes: "
          f"{'included' if args.include_scrambles else 'excluded'}")

    n_ticks = sum(int(s["tick_collision"].sum()) for s in sessions)
    print(f"  BLE tick collisions: {n_ticks} adjacent pairs "
          f"({100*n_ticks/max(n_all-len(sessions),1):.2f}%) — "
          f"{'EXCLUDED' if args.exclude_tick_collisions else 'included'}")

    results = sweep(sessions, args.fps, args.models,
                    args.exclude_tick_collisions)
    for key, rows in results.items():
        print_table(key, rows)

    print("\n  Columns are MAX ACHIEVABLE move accuracy — a perfect model's "
          "score.\n  'sampling' bounds every possible decoder at that fps.\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "corpus": {"n_sessions": len(sessions), "n_onsets": n_all,
                           "median_tps": float(np.median(tps_all)),
                           "include_scrambles": args.include_scrambles},
                "min_sep": MIN_SEP,
                "results": results,
                "sessions": [{"name": s["name"], "n": s["n"],
                              "tps": s["tps"]} for s in sessions],
            }, f, indent=2)
        print(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
