"""
compare_speed_sim.py — two speed_sim.py runs, side by side.

Exists because the question "did speed augmentation work" is not answerable
from val MER (the val split is slow footage, so an improvement at speed
scores as noise there at best). The answer is in the RETENTION curve:
predicted/true move count as simulated solve speed rises.

The number to read is the WORST session at each TPS level, not the mean.
The anticheat count floor is calibrated against the worst case — a mean that
improves while the worst case does not moves no verdict
(anticheat_gate.MIN_OBSERVED_MOVES).

    python compare_speed_sim.py --base speed_sim_move_ctc_aug44_s0_blur.json \\
                                --new  speed_sim_move_ctc_spd_s0_blur.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def curve(path: Path) -> dict:
    rows = json.loads(Path(path).read_text())
    by_tps = defaultdict(list)
    for r in rows:
        by_tps[r["target_tps"]].append(r)
    out = {}
    for tps, sel in by_tps.items():
        ret = np.array([r["pred"] / max(r["true"], 1) for r in sel])
        out[tps] = {"worst": ret.min(), "mean": ret.mean(), "n": len(sel),
                    "sessions": {r["session"]: r["pred"] / max(r["true"], 1)
                                 for r in sel}}
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="baseline run's json")
    ap.add_argument("--new", required=True, help="new run's json")
    args = ap.parse_args()

    a, b = curve(args.base), curve(args.new)
    print(f"base = {args.base}\nnew  = {args.new}\n")
    print(f"{'TPS':>5} {'n':>3} | {'worst base':>10} {'worst new':>10} "
          f"{'delta':>7} | {'mean base':>10} {'mean new':>9} {'delta':>7}")
    print("-" * 76)
    for tps in sorted(set(a) & set(b)):
        wa, wb = a[tps]["worst"], b[tps]["worst"]
        ma, mb = a[tps]["mean"], b[tps]["mean"]
        print(f"{tps:5.1f} {a[tps]['n']:3d} | {wa:10.3f} {wb:10.3f} "
              f"{wb - wa:+7.3f} | {ma:10.3f} {mb:9.3f} {mb - ma:+7.3f}")

    # Per-session at the levels that decide the gate's usable band. Session
    # counts here are tiny (4 held-out), so a single session swinging is the
    # difference between "moved" and "noise" — show them rather than let a
    # mean hide it.
    for tps in (6, 8, 10):
        if tps not in a or tps not in b:
            continue
        print(f"\nper-session retention at {tps} TPS:")
        for s in sorted(a[tps]["sessions"]):
            if s in b[tps]["sessions"]:
                x, y = a[tps]["sessions"][s], b[tps]["sessions"][s]
                print(f"  {s:38s} {x:.3f} -> {y:.3f}  {y - x:+.3f}")

    print("\nRetention is predicted/true move count. The count gate is "
          "calibrated on the WORST session, so that is the column that can "
          "move a verdict; the mean cannot.")
    print("One seed. Seed spread on this pipeline has repeatedly been >= the "
          "effect being measured — a curve that moves here needs seed 1 "
          "before it is a result.")


if __name__ == "__main__":
    main()
