"""
m6_stats.py — uncertainty for the headline numbers and the ablation ladder.

Two things, both of which the report otherwise lacks:

  1. Bootstrap confidence intervals over SESSIONS (not over moves). The
     quantity being estimated is "what would this model score on a new
     session from this solver", so the resampling unit has to be the
     session. Resampling moves would give intervals several times too
     narrow, because moves within a session are strongly dependent — the
     16x per-session spread in miss rate IS that dependence.

  2. A paired test for each ablation rung. Sessions are paired across
     rungs (the same 14 sessions throughout), so the right test is on the
     per-session differences, not on two independent samples. Reported as
     a bootstrap CI on the mean paired difference plus an exact sign test,
     which needs no distributional assumption at n=14.

Seeds are averaged within a session before resampling: the two seeds are
not independent evidence about a session, they are two draws of the same
estimator on it.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

import common as C

B = 20000
RNG = np.random.default_rng(0)


def boot_ci(x, stat=np.mean, b=B, alpha=0.05):
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), size=(b, len(x)))
    d = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(d, 100 * alpha / 2)), \
        float(np.percentile(d, 100 * (1 - alpha / 2)))


def sign_test(diffs):
    """Exact two-sided sign test on paired differences (ties dropped)."""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    pos = sum(1 for x in d if x > 0)
    k = min(pos, n - pos)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0), pos, n - pos


def per_session(rows, key="acc", filt=lambda r: True):
    """session -> mean over seeds."""
    out = {}
    for r in rows:
        if not filt(r):
            continue
        out.setdefault(r["session"], []).append(r[key])
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    meta = {m["session"]: m for m in C.load("holdout_meta.json")}
    m1 = [r for r in C.load("m1_recognition.json")
          if r["session"].endswith("_solve")]
    out = {}

    print(f"\n{'='*78}\n  BOOTSTRAP CIs OVER SESSIONS "
          f"({B:,} resamples, 95%)\n{'='*78}")
    print(f"  {'quantity':<34}{'n':>4}{'mean':>10}{'95% CI':>20}")
    for lab, f in (("raw accuracy, all held-out solves", lambda r: True),
                   ("raw accuracy, daytime",
                    lambda r: not meta[r["session"]]["evening"]),
                   ("raw accuracy, evening",
                    lambda r: meta[r["session"]]["evening"])):
        d = per_session(m1, "acc", f)
        m, lo, hi = boot_ci(list(d.values()))
        out[lab] = {"n": len(d), "mean": m, "lo": lo, "hi": hi}
        print(f"  {lab:<34}{len(d):>4}{m*100:>9.1f}%"
              f"   [{lo*100:5.1f}, {hi*100:5.1f}]")

    # ---- ablation rungs, paired -----------------------------------------
    m4 = C.load("m4_ablation.json")
    rungs = []
    for r in m4:
        if (r["rung"], r["tag"]) not in rungs:
            rungs.append((r["rung"], r["tag"]))
    print(f"\n{'='*78}\n  ABLATION RUNGS, PAIRED OVER THE SAME SESSIONS"
          f"\n{'='*78}")
    print(f"  {'step':<44}{'delta':>8}{'95% CI':>18}{'sign p':>9}")
    steps = []
    for (l0, t0), (l1, t1) in zip(rungs, rungs[1:]):
        a = per_session([r for r in m4 if r["tag"] == t0])
        b = per_session([r for r in m4 if r["tag"] == t1])
        keys = sorted(set(a) & set(b))
        diffs = [b[k] - a[k] for k in keys]
        m, lo, hi = boot_ci(diffs)
        p, npos, nneg = sign_test(diffs)
        name = f"{l0}  ->  {l1}"
        steps.append({"step": name, "n": len(keys), "delta": m,
                      "lo": lo, "hi": hi, "p_sign": p,
                      "better": npos, "worse": nneg})
        print(f"  {name[:43]:<44}{m*100:>+7.1f}"
              f"   [{lo*100:+5.1f}, {hi*100:+5.1f}]{p:>9.3f}")
    # first vs last
    a = per_session([r for r in m4 if r["tag"] == rungs[0][1]])
    b = per_session([r for r in m4 if r["tag"] == rungs[-1][1]])
    keys = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in keys]
    m, lo, hi = boot_ci(diffs)
    p, npos, nneg = sign_test(diffs)
    steps.append({"step": "first rung -> last rung", "n": len(keys),
                  "delta": m, "lo": lo, "hi": hi, "p_sign": p,
                  "better": npos, "worse": nneg})
    print(f"  {'FIRST RUNG -> LAST RUNG':<44}{m*100:>+7.1f}"
          f"   [{lo*100:+5.1f}, {hi*100:+5.1f}]{p:>9.3f}")

    out["rungs"] = steps
    C.dump("m6_stats.json", out)

    print(f"\n  A sign test at n={len(keys)} cannot go below p={2/2**len(keys):.4f}, "
          f"so a unanimous\n  rung is reported at that floor rather than as "
          f"something smaller.")
    print("  CIs are over sessions, which is the unit a new user "
          "corresponds to.")


if __name__ == "__main__":
    main()
