"""
falsifiability_batch.py

Phase 0 of MODEL_REWORK_PLAN.md: the batch false-accept harness that
PATH_TO_VERIFICATION.md Sec7 called mandatory and left undone (only run in
depth on one session there). No new decode logic - pure orchestration
around verify_solve.run_session(), which already runs one falsifiability
sweep per session whose true claim verifies. This script aggregates those
sweeps into the number Sec7 asked for: a Clopper-Pearson upper bound on the
false-accept rate, broken out by decoy type, plus the cost-margin
distribution (not just accept/reject - a lever that erodes decoy
separation without flipping any single verdict would be invisible to a
bare accept/reject count).

    python falsifiability_batch.py --sessions ../training_data/solve_*/

Every decoder-facing change (D1-D4, any future model swap) should re-run
this and compare the bound and the margin distribution, not just whether
any individual decoy flipped.
"""

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

import reconstruct as RC
import verify_solve as VS


# ---------------------------------------------------------------------------
# Clopper-Pearson upper bound, no scipy (not in requirements.txt)
# ---------------------------------------------------------------------------

def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """
    One-sided (1-alpha) upper confidence bound on a binomial rate: the
    largest p such that P(X <= k | n, p) >= alpha. P(X<=k|p) is monotone
    decreasing in p, so this is a plain bisection.
    """
    if n == 0:
        return float("nan")
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------
# Decoy bucketing
# ---------------------------------------------------------------------------

def _bucket(row: dict) -> str:
    d = row["distance"]
    if d is not None:
        return f"{d} move{'s' if d > 1 else ''} off"
    if "never" in row["name"]:
        return "never scrambled"
    return "scrambled by something else"


def aggregate(results: list[dict], alpha: float = 0.05) -> dict:
    """
    Pool every session's falsifiability_sweep() rows (skipping the true
    claim itself, distance 0) into per-bucket and overall accept counts,
    plus the margin distribution for every decoy that DID decode.
    """
    swept = [r for r in results if r.get("sweep")]
    unswept = [r for r in results if r["res"]["solved"] and not r.get("sweep")]
    not_verified = [r for r in results if not r["res"]["solved"]]

    buckets: dict[str, list[dict]] = defaultdict(list)
    all_rows = []
    for res in swept:
        true_cost = next((s["cost"] for s in res["sweep"] if s["distance"] == 0),
                         None)
        for s in res["sweep"]:
            if s["distance"] == 0:
                continue
            row = {**s, "session": res["session"],
                   "margin": (None if s["cost"] is None or true_cost is None
                              else s["cost"] - true_cost)}
            buckets[_bucket(s)].append(row)
            all_rows.append(row)

    overall_k = sum(r["verified"] for r in all_rows)
    overall_n = len(all_rows)
    margins = [r["margin"] for r in all_rows
               if r["verified"] and r["margin"] is not None]

    cheaper_sessions = [res["session"] for res in swept
                        if any(s["verified"] and s["distance"] not in (0, None)
                               and s["cost"] is not None
                               and s["cost"] < next(
                                   (t["cost"] for t in res["sweep"]
                                    if t["distance"] == 0), 1e18) - 1e-6
                               for s in res["sweep"])]

    return {
        "n_sessions_total": len(results),
        "n_sessions_verified": len(swept) + len(unswept),
        "n_sessions_swept": len(swept),
        "n_sessions_not_verified": len(not_verified),
        "n_sessions_unswept_no_reason": len(unswept),
        "overall_k": overall_k, "overall_n": overall_n,
        "overall_bound": clopper_pearson_upper(overall_k, overall_n, alpha),
        "buckets": {
            name: {
                "k": sum(r["verified"] for r in rows),
                "n": len(rows),
                "bound": clopper_pearson_upper(
                    sum(r["verified"] for r in rows), len(rows), alpha),
            }
            for name, rows in sorted(buckets.items())
        },
        "margins": margins,
        "cheaper_sessions": cheaper_sessions,
        "alpha": alpha,
    }


def print_report(agg: dict):
    print(f"\n{'='*70}")
    print(f"  FALSIFIABILITY BATCH — {agg['n_sessions_swept']} sessions swept "
          f"of {agg['n_sessions_total']} total")
    print(f"{'='*70}")
    print(f"  {agg['n_sessions_not_verified']} session(s) did not verify "
          f"(no sweep possible — no baseline cost)")
    if agg["n_sessions_unswept_no_reason"]:
        print(f"  {agg['n_sessions_unswept_no_reason']} verified session(s) "
              f"had no sweep recorded (--no-sweep?) — excluded")

    a = int(round((1 - agg["alpha"]) * 100))
    print(f"\n  OVERALL: {agg['overall_k']}/{agg['overall_n']} decoys "
          f"accepted  ->  {a}% upper bound = "
          f"{agg['overall_bound']*100:.2f}%")

    print(f"\n  By decoy type:")
    print(f"    {'type':<28} {'accepted/n':>12} {a}% bound")
    for name, b in agg["buckets"].items():
        print(f"    {name:<28} {b['k']:>5}/{b['n']:<6} "
              f"{b['bound']*100:6.2f}%")

    if agg["margins"]:
        m = np.array(agg["margins"])
        print(f"\n  Cost margin of ACCEPTED decoys vs the true claim "
              f"(n={len(m)}):")
        print(f"    min {m.min():+.2f}  median {np.median(m):+.2f}  "
              f"max {m.max():+.2f}")
        print(f"    (only decoys the beam actually solved contribute a "
              f"margin; a shrinking\n     median across re-runs is the "
              f"early warning a lever is eroding separation)")
    else:
        print(f"\n  No accepted decoys produced a cost margin (either none "
              f"accepted, or none\n  of the accepted ones decoded a finite "
              f"cost).")

    if agg["cheaper_sessions"]:
        print(f"\n  WARNING — a WRONG claim was CHEAPER than the true claim "
              f"on {len(agg['cheaper_sessions'])}\n  session(s): "
              f"{', '.join(agg['cheaper_sessions'])}")
        print(f"  Those sessions' VERIFIED verdicts are not evidence as-is "
              f"— see verify_solve.py's\n  falsifiability_sweep docstring.")
    else:
        print(f"\n  No session had a wrong claim cheaper than the true one.")
    print(f"{'='*70}")


def main():
    p = argparse.ArgumentParser(
        description="Aggregate verify_solve.py's falsifiability sweep "
                    "across every recorded session into a Clopper-Pearson "
                    "false-accept bound (MODEL_REWORK_PLAN.md Phase 0).")
    p.add_argument("--sessions", nargs="+",
                   default=["../training_data/solve_*/"])
    p.add_argument("--detector", type=str, default=VS.LD.DETECTOR_PATH)
    p.add_argument("--classifier", type=str, default=VS.LD.CLASSIFIER_PATH)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="1-alpha is the confidence level (default 0.05 -> "
                        "95%% upper bound)")
    p.add_argument("--out", type=str, default=None,
                   help="Also write the raw aggregate as JSON here")
    # Decoder knobs — identical names/defaults to verify_solve.py so any
    # D-lever combination can be re-swept with the same flags.
    p.add_argument("--beam", type=int, default=RC.BEAM)
    p.add_argument("--retry-beam", type=int, default=4 * RC.BEAM)
    p.add_argument("--del-cost", type=float, default=RC.C_DEL)
    p.add_argument("--ins-cost", type=float, default=RC.C_INS)
    p.add_argument("--rot-cost", type=float, default=RC.C_ROT)
    p.add_argument("--max-end-ins", type=int, default=RC.MAX_END_INS)
    p.add_argument("--candidate-threshold", type=float, default=None)
    p.add_argument("--del-floor", type=float, default=RC.DEL_FLOOR)
    p.add_argument("--blend-inv", type=float, default=RC.BLEND_INV)
    p.add_argument("--blend-unif", type=float, default=RC.BLEND_UNIF)
    p.add_argument("--blend-adj", type=float, default=RC.BLEND_ADJ)
    p.add_argument("--rel-weight", type=float, default=RC.REL_WEIGHT)
    p.add_argument("--rotations", action="store_true")
    p.add_argument("--bidir", action="store_true")
    p.add_argument("--meet", type=int, default=None)
    p.add_argument("--meet-sweep", action="store_true")
    args = p.parse_args()

    args.session = args.sessions
    args.no_sweep = False   # the whole point of this script

    tables = RC.build_tables()
    results = VS.run_session(args, tables)   # prints per-session detail too

    agg = aggregate(results, alpha=args.alpha)
    print_report(agg)

    if args.out:
        Path(args.out).write_text(json.dumps(agg, indent=2, default=str))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
