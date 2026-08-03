"""
threshold_sweep.py

Re-tunes the joint model's onset threshold against the MISS/PHANTOM balance
that actually predicts verification, instead of against onset F1.

Why
---
The deployed threshold was picked to maximise onset F1. F1 weights a miss and
a phantom equally *by construction*, which is only the right call if they cost
the same downstream. `accuracy_target.py` measured what they actually cost:
at 3% error through one channel, miss -> 45% of solves verify, phantom -> 40%.
Close enough that equal weighting is defensible.

But the deployed operating point is not balanced. Measured over the 27
solve-length sessions: miss 1.50% / phantom 0.72% (medians). More than 2:1
skewed toward misses. If the two cost about the same, that is the wrong place
on the curve — lowering the threshold converts misses into detections faster
than it invents phantoms, right up until it doesn't.

This sweeps the threshold over the cached posteriorgrams and reports the
per-channel rates, with no decoding at all. Decoding is the expensive part
and it is not needed to find the balance point; run verify_joint at the
chosen threshold afterwards to confirm it converts into verifications.

Scoring matches metric_audit.score_by_time (greedy, most-confident-first,
one-to-one, within TOLERANCE frames) so the numbers are comparable to every
other miss/phantom figure in the repo.

Usage:
    python threshold_sweep.py --model checkpoints/move_joint_seed0.pt
    python threshold_sweep.py --model checkpoints/move_joint_seed0.pt --min-sep 1 2 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import reconstruct as RC
from decode import peak_pick, onset_collisions, TOLERANCE, MIN_SEP
from model import build_joint_from_ckpt, score_stream_joint
from dataset import JointSessionStream
from joint_decode import posteriorgram_to_moves


def score_against_gt(moves, gt_onset, gt_labels, tolerance=TOLERANCE):
    """Greedy one-to-one time match — same policy as metric_audit."""
    un = list(range(len(gt_onset)))
    pairs = {}
    for pi in sorted(range(len(moves)), key=lambda i: -moves[i]["conf"]):
        if not un:
            break
        f = moves[pi]["frame"]
        best = min(un, key=lambda j: abs(int(gt_onset[j]) - f))
        if abs(int(gt_onset[best]) - f) <= tolerance:
            un.remove(best)
            pairs[best] = pi
    ok = sum(1 for j, pi in pairs.items() if moves[pi]["move"] == gt_labels[j])
    return {"n_gt": len(gt_onset), "n_pred": len(moves), "ok": ok,
            "sub": len(pairs) - ok, "miss": len(gt_onset) - len(pairs),
            "phantom": len(moves) - len(pairs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sessions", nargs="+",
                   default=["../training_data/solve_*/"])
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                            0.50, 0.60, 0.70])
    p.add_argument("--min-sep", type=int, nargs="+", default=[MIN_SEP],
                   dest="min_seps")
    p.add_argument("--min-moves", type=int, default=60,
                   help="only score solve-length sessions")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()

    dirs = sorted(d for pat in args.sessions
                  for d in Path(".").glob(pat) if d.is_dir())
    print(f"\n  model {args.model} (epoch {ckpt['epoch']}, deployed threshold "
          f"{ckpt.get('threshold', 0.5)})")

    # Score every session ONCE; the sweep is then pure post-processing.
    cached = []
    for d in dirs:
        sp = d / "detector_stream_color.npz"
        mj = d / "moves.jsonl"
        if not sp.exists() or not mj.exists():
            continue
        gt = [json.loads(l)["wca_notation"]
              for l in open(mj) if l.strip()]
        if len(gt) < args.min_moves:
            continue
        stream = JointSessionStream(sp, sigma=1.0)
        onset_prob, class_prob, count_prob = score_stream_joint(model, stream, device)
        gt_onset = stream.onset_idx.astype(int)
        # onset_class drops unresolved moves, so re-derive labels from it
        labels = [RC.WCA12[i] for i in stream.onset_class.astype(int)]
        cached.append((d.name, onset_prob, class_prob, gt_onset, labels,
                       float(stream.fps)))
        print(f"    scored {d.name} ({len(gt_onset)} onsets)")
    print(f"\n  {len(cached)} solve-length session(s) scored\n")

    # A pair closer than MIN_SEP can have AT MOST one of its two members
    # reported, so it forces at least one miss. The floor on the miss RATE is
    # therefore about half the fraction of onsets sitting in such pairs.
    unres = sum(len(onset_collisions(g)[0]) for _, _, _, g, _, _ in cached)
    tot_gt = sum(len(g) for _, _, _, g, _, _ in cached)
    floor = unres / 2 / tot_gt
    print(f"  {unres} of {tot_gt} GT onsets ({unres/tot_gt*100:.2f}%) sit in "
          f"pairs closer than MIN_SEP.\n  At most one member of each can ever "
          f"be reported, so the miss rate has a\n  STRUCTURAL FLOOR of "
          f"~{floor*100:.2f}% that no threshold can go below.\n")

    rows = []
    print(f"  {'min_sep':>7}{'thresh':>8}{'miss%':>8}{'phantom%':>10}"
          f"{'sub%':>7}{'recall%':>9}{'F1':>7}   {'miss:phantom':>12}")
    for ms in args.min_seps:
        for th in args.thresholds:
            agg = {"n_gt": 0, "n_pred": 0, "ok": 0, "sub": 0,
                   "miss": 0, "phantom": 0}
            for _, op, cp, g, lab, fps in cached:
                mv = posteriorgram_to_moves(op, cp, th, ms, fps)
                s = score_against_gt(mv, g, lab)
                for k in agg:
                    agg[k] += s[k]
            n = agg["n_gt"]
            miss, pha, sub = (agg["miss"] / n, agg["phantom"] / n,
                              agg["sub"] / n)
            det = n - agg["miss"]
            prec = det / agg["n_pred"] if agg["n_pred"] else 0.0
            rec = det / n
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            ratio = miss / pha if pha else float("inf")
            print(f"  {ms:>7}{th:>8.2f}{miss*100:>7.2f}%{pha*100:>9.2f}%"
                  f"{sub*100:>6.2f}%{rec*100:>8.2f}%{f1:>7.3f}"
                  f"   {ratio:>11.2f}")
            # counts first — the rate keys must win, they share names
            rows.append({**{f"n_{k}": v for k, v in agg.items()},
                         "min_sep": ms, "threshold": th, "miss": miss,
                         "phantom": pha, "sub": sub, "recall": rec, "f1": f1})

    # The operating point this study is looking for: miss and phantom rates
    # roughly equal, since they cost about the same downstream.
    best = min(rows, key=lambda r: abs(r["miss"] - r["phantom"]))
    bf1 = max(rows, key=lambda r: r["f1"])
    tot = min(rows, key=lambda r: r["miss"] + r["phantom"])
    for lbl, r in (("balanced (miss==phantom)", best), ("best F1", bf1),
                   ("min miss+phantom", tot)):
        print(f"  {lbl:26} min_sep {r['min_sep']} threshold "
              f"{r['threshold']:.2f}  miss {r['miss']*100:.2f}% "
              f"phantom {r['phantom']*100:.2f}%  "
              f"sum {(r['miss']+r['phantom'])*100:.2f}%")
    print(f"\n  miss floor from collisions: {floor*100:.2f}%   "
          f"(target from ACCURACY_TARGET.md: <= 1.00%)")
    if floor > 0.01:
        print(f"  ** The floor is ABOVE the target. No threshold reaches it; "
              f"the collision\n     mechanism itself has to change. **")
    print(f"\n  These are DETECTION rates only. Confirm with:"
          f"\n    python verify_joint.py --model {args.model} --slices "
          f"--threshold <t>\n")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
