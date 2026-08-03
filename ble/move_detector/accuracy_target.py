"""
accuracy_target.py

Answers the question TODO.md item 4 calls the first goal: what per-move
accuracy does the pipeline need before a given fraction of solves VERIFY?

Why this needs simulating rather than measuring
-----------------------------------------------
The recorded corpus cannot answer it. Only 27 sessions are solve-length, and
they cluster at the extremes — 15 sit above 98% raw accuracy and verify
15/15, four sit in the 95-98% band and verify 0/4. That is a cliff with
almost no samples on it, and no way to add samples except by recording
hundreds more solves at accuracies we cannot dial in on purpose.

Simulation can dial it in. The generator here is the one `run_synthetic`
already uses (it produces onset posteriors with the measured confidence
profile), driven at error rates swept across the region of interest, decoded
by the real decoder against a real start state.

Calibration
-----------
The error MIX is not invented. Measured over the 27 solve-length sessions of
the seed-0 sweep, total error mass splits:

    substitutions  19%      miss  60%      phantom  21%

so 81% of what goes wrong is an insertion/deletion problem, not a naming
problem. `--mix measured` uses that split; `--mix` also takes `sub`, `miss`
and `phantom` to put ALL error mass through one channel, which is what
answers "which error type actually costs us verifications".

Reading the output
------------------
Decodes run at the production first-pass beam with no retry, so the verified
fractions here are a floor: production retries once at 4x beam. The shape of
the curve and the ranking of the error channels are the results; the exact
threshold is a lower bound on accuracy needed, not an upper one.

Usage:
    python accuracy_target.py                       # main curve
    python accuracy_target.py --channels            # which error type hurts
    python accuracy_target.py --lengths             # vs solve length
"""

import argparse
import json
import time

import numpy as np

import reconstruct as RC

# Measured composition of total error mass, seed-0 sweep, 27 solve-length
# sessions. See module docstring.
MIX_MEASURED = {"sub": 0.19, "miss": 0.60, "phantom": 0.21}
MIX_PURE = {
    "sub":     {"sub": 1.0, "miss": 0.0, "phantom": 0.0},
    "miss":    {"sub": 0.0, "miss": 1.0, "phantom": 0.0},
    "phantom": {"sub": 0.0, "miss": 0.0, "phantom": 1.0},
}

# Of substitutions, the share that is the same-face temporal inverse rather
# than a different face. Measured 2026-07-30 over 41 real substitutions:
# 29% same-face wrong-direction, the rest other faces.
INV_SHARE = 0.29


def simulate(rng, n_moves, rate, mix, tables, beam, slices=False):
    """One synthetic solve at the given total error rate. Returns a dict."""
    gt_cube = RC._random_gt(rng, n_moves)
    r_sub = rate * mix["sub"]
    r_mis = rate * mix["miss"]
    r_pha = rate * mix["phantom"]

    obs, scores, n_ok, n_sub, n_mis, n_pha = [], [], 0, 0, 0, 0
    for k in gt_cube:
        if rng.random() < r_mis:
            n_mis += 1
            continue
        u = rng.random()
        if u < r_sub:
            n_sub += 1
            pred = RC.INV12[k] if rng.random() < INV_SHARE \
                else int(rng.integers(12))
            conf = rng.uniform(0.5, 0.98) if pred == RC.INV12[k] \
                else rng.uniform(0.3, 0.8)
        else:
            n_ok += 1
            pred = k
            # Matches the measured profile on real replays.
            conf = rng.uniform(0.9, 0.999) if rng.random() > 0.15 \
                else rng.uniform(0.55, 0.9)
        obs.append(RC._fake_probs(rng, k, pred, conf))
        scores.append(rng.uniform(0.75, 1.0))
        if rng.random() < r_pha:
            n_pha += 1
            obs.append(RC._fake_probs(rng, None, int(rng.integers(12)),
                                      rng.uniform(0.3, 0.9)))
            scores.append(rng.uniform(0.3, 0.6))

    if not obs:
        return None
    gt_names = [RC.WCA12[k] for k in gt_cube]
    start = RC.start_from_gt(gt_names)
    cost_rows = [RC.onset_costs(p) for p in obs]
    del_costs = RC.score_del_costs(scores, 0.25)
    kw = {}
    if slices:
        kw = dict(slices=True,
                  slice_rows=[RC.slice_mask(p) for p in obs])
    res = RC.decode(start, cost_rows, beam=beam, del_costs=del_costs,
                    tables=tables, **kw)
    return {"acc": n_ok / len(gt_cube), "solved": bool(res["solved"]),
            "exact": bool(res["solved"] and res["moves"] == gt_names),
            "sub": n_sub, "miss": n_mis, "phantom": n_pha}


def sweep(label, rows, trials, n_moves, mix, beam, seed, tables):
    """Run one sweep and print a table. `rows` is a list of (name, rate)."""
    print(f"\n  {label}   ({trials} trials x {n_moves} moves, beam {beam})")
    print(f"  {'setting':<16}{'raw acc':>9}{'verified':>10}{'exact':>8}"
          f"{'95% CI':>16}")
    out = []
    for name, rate, m in rows:
        rng = np.random.default_rng(seed)
        res = [r for _ in range(trials)
               if (r := simulate(rng, n_moves, rate, m, tables, beam))]
        acc = np.mean([r["acc"] for r in res])
        ver = np.mean([r["solved"] for r in res])
        exa = np.mean([r["exact"] for r in res])
        # Wilson interval, so a 0/25 or 25/25 cell still reports a range
        n, p = len(res), ver
        z = 1.96
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        print(f"  {name:<16}{acc*100:8.1f}%{ver*100:9.0f}%{exa*100:7.0f}%"
              f"     [{max(0,c-h)*100:3.0f}%,{min(1,c+h)*100:4.0f}%]")
        out.append({"setting": name, "rate": rate, "acc": acc,
                    "verified": ver, "exact": exa, "n": n})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--moves", type=int, default=120,
                   help="median solve length in the real corpus")
    p.add_argument("--beam", type=int, default=RC.BEAM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--channels", action="store_true",
                   help="also sweep each error channel in isolation")
    p.add_argument("--lengths", action="store_true",
                   help="also sweep solve length at a fixed error rate")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    tables = RC.build_tables()
    t0 = time.time()
    result = {}

    rates = [0.002, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.045, 0.060]
    result["curve"] = sweep(
        "VERIFICATION vs ACCURACY (measured error mix 19/60/21)",
        [(f"err {r*100:.2f}%", r, MIX_MEASURED) for r in rates],
        args.trials, args.moves, MIX_MEASURED, args.beam, args.seed, tables)

    if args.channels:
        # Same total error mass, routed entirely through one channel. If the
        # three curves separate, error TYPE matters more than error RATE and
        # the pipeline should be optimised against the worst channel.
        rows = []
        for r in (0.005, 0.010, 0.020, 0.030):
            for ch in ("sub", "miss", "phantom"):
                rows.append((f"{ch} {r*100:.1f}%", r, MIX_PURE[ch]))
        result["channels"] = sweep("ERROR CHANNEL ISOLATION", rows,
                                   args.trials, args.moves, MIX_MEASURED,
                                   args.beam, args.seed, tables)

    if args.lengths:
        rows = [(f"{n} moves", 0.010, MIX_MEASURED) for n in (60, 90, 120, 160, 200)]
        print("\n  SOLVE LENGTH at 1.0% error (measured mix)")
        result["lengths"] = []
        for name, rate, m in rows:
            n_moves = int(name.split()[0])
            rng = np.random.default_rng(args.seed)
            res = [r for _ in range(args.trials)
                   if (r := simulate(rng, n_moves, rate, m, tables, args.beam))]
            ver = np.mean([r["solved"] for r in res])
            print(f"    {name:<12} verified {ver*100:3.0f}%")
            result["lengths"].append({"moves": n_moves, "verified": ver})

    print(f"\n  ({time.time()-t0:.0f}s total)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
