"""
eval_lighting.py

Score any number of Stage A checkpoints on the held-out lighting test set,
split by time of day — the measurement the widened photometric
augmentation (dataset.AUG_*, 2026-07-31) exists to move.

The test set is every prepared session OUTSIDE the 42-session split the
baseline checkpoints were trained and validated on, which as of 2026-07-31
is eight takes across three sittings:

    07-29 22:19   evening
    07-30 11:19   morning
    07-30 11:31   morning
    07-31 21:10   evening, no extra light
    07-31 21:36   evening, lamp added

No checkpoint has trained on any of them, so old and new are measured on
equal footing and the morning/evening split is the contrast of interest.

Scoring is END-TO-END per-move accuracy by word alignment against the BLE
move log — live_detect.align_sequences, the same arithmetic
report_scramble prints live, so these numbers sit on the same axis as the
ones quoted from actual sittings. MER is reported alongside because the
two can disagree in sign, which has happened in this project before.

Input comes from the prepared detector_stream_color.npz rather than the
raw JPEGs: it is the same crop code the live path runs, and it makes a
four-checkpoint sweep minutes instead of an hour. The comparison that
matters is old-vs-new on IDENTICAL input, which this guarantees by
construction.

Usage:
    python eval_lighting.py --models checkpoints/move_ctc_s0.pt checkpoints/move_ctc_aug_s0.pt
    python eval_lighting.py --models move_ctc_*.pt --out results/2026-08-01/lighting_eval.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import live_detect as LD
import reconstruct as RC
from model import build_joint_from_ckpt, score_stream_joint
from dataset import JointSessionStream
from joint_decode import posteriorgram_to_moves
from ctc_decode import prefix_beam_decode, ctc_to_moves, move_error_rate
from decode import MIN_SEP

SESSION_ROOT = Path("../training_data")
DAY_START, DAY_END = 9, 18


def held_out_sessions(models: list[Path]) -> list[Path]:
    """
    Prepared sessions no listed checkpoint has ever seen.

    Intersected across ALL models rather than taken from the first: a take
    that one checkpoint trained on is not a fair test for that checkpoint
    even if another never saw it, and quietly scoring it would flatter
    whichever model had the advantage.
    """
    seen = set()
    for m in models:
        c = torch.load(m, map_location="cpu")
        seen |= set(c.get("train_session_names") or [])
        seen |= set(c.get("val_session_names") or [])
    out = [p.parent for p in
           sorted(SESSION_ROOT.glob("solve_*/detector_stream_color.npz"))
           if p.parent.name not in seen and (p.parent / "moves.jsonl").exists()]
    return out


def take_hour(d: Path) -> int | None:
    f = d / "frames.jsonl"
    if not f.exists():
        return None
    with open(f) as fh:
        ts = json.loads(fh.readline())["ts"]
    return time.localtime(ts).tm_hour


def truth_word(d: Path) -> list[str] | None:
    gt = [json.loads(l).get("wca_notation")
          for l in open(d / "moves.jsonl") if l.strip()]
    return None if not gt or any(g is None for g in gt) else gt


def decode_arm(ckpt, model, stream, device, beam: int) -> list[dict]:
    """CTC checkpoints go through the prefix beam; everything else peak-picks."""
    onset_prob, class_prob, count_prob = score_stream_joint(model, stream,
                                                            device)
    if ckpt.get("model_type") == "ctc":
        labels, frames = prefix_beam_decode(
            np.log(np.maximum(class_prob, 1e-12)), beam=beam)
        return ctc_to_moves(class_prob, labels, frames, fps=stream.fps)
    return posteriorgram_to_moves(
        onset_prob, class_prob, threshold=ckpt.get("threshold", 0.5),
        min_sep=ckpt.get("min_sep", MIN_SEP), fps=stream.fps,
        count_prob=count_prob, count_radius=ckpt.get("count_radius", 2))


def score(gt: list[str], moves: list[dict]) -> dict:
    pred = [m["move"] for m in moves]
    ops = LD.align_sequences(gt, pred)
    c = {k: sum(1 for o, _, _ in ops if o == k)
         for k in ("ok", "sub", "miss", "phantom")}
    mer, parts = move_error_rate([RC.WCA12.index(p) for p in pred],
                                 [RC.WCA12.index(g) for g in gt])
    return {"n": len(gt), "n_pred": len(pred), "e2e": c["ok"] / max(len(gt), 1),
            "mer": mer, **c}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--beam-ctc", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    paths = sorted({Path(q) for m in args.models
                    for q in (Path(".").glob(m) if "*" in m else [Path(m)])})
    paths = [q for q in paths if q.exists()]
    if not paths:
        sys.exit("No checkpoints matched --models.")

    dirs = held_out_sessions(paths)
    if not dirs:
        sys.exit("No held-out prepared sessions — every prepared session is "
                 "in some checkpoint's split.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  {len(paths)} checkpoint(s), {len(dirs)} held-out session(s)")
    for d in dirs:
        h = take_hour(d)
        print(f"    {d.name:<36} {h:02d}:xx  "
              f"{'morning' if h is not None and DAY_START <= h < DAY_END else 'EVENING'}")

    rows = []
    for mp in paths:
        ckpt = torch.load(mp, map_location=device)
        model = build_joint_from_ckpt(ckpt, device)
        model.eval()
        arm = ckpt.get("model_type", "joint")
        print(f"\n{'='*78}\n  {mp.name}  (epoch {ckpt.get('epoch','?')}, seed "
              f"{ckpt.get('seed','?')}, arm {arm}, aug_strength "
              f"{ckpt.get('aug_strength', 'pre-07-31 narrow')})\n{'='*78}")
        print(f"  {'session':<36} {'hour':>5} {'n':>4} {'pred':>5} "
              f"{'e2e':>7} {'MER':>7} {'miss':>5} {'phan':>5}")
        for d in dirs:
            gt = truth_word(d)
            if gt is None or any(g not in RC.WCA12 for g in gt):
                continue
            stream = JointSessionStream(d / "detector_stream_color.npz")
            moves = decode_arm(ckpt, model, stream, device, args.beam_ctc)
            s = score(gt, moves)
            h = take_hour(d)
            s.update({"model": mp.name, "session": d.name, "hour": h,
                      "evening": not (DAY_START <= h < DAY_END),
                      "arm": arm})
            rows.append(s)
            print(f"  {d.name:<36} {h:02d}:xx {s['n']:>4} {s['n_pred']:>5} "
                  f"{s['e2e']*100:>6.1f}% {s['mer']*100:>6.1f}% "
                  f"{s['miss']:>5} {s['phantom']:>5}")

    print(f"\n\n{'='*78}\n  SUMMARY — end-to-end by lighting regime\n{'='*78}")
    print(f"  {'checkpoint':<26} {'morning':>18} {'EVENING':>18} "
          f"{'all':>12}")
    for mp in paths:
        mine = [r for r in rows if r["model"] == mp.name]
        if not mine:
            continue
        cells = []
        for sel in (lambda r: not r["evening"], lambda r: r["evening"],
                    lambda r: True):
            g = [r for r in mine if sel(r)]
            ok = sum(r["ok"] for r in g)
            n = sum(r["n"] for r in g)
            cells.append(f"{ok/max(n,1)*100:5.1f}% ({len(g)} takes)"
                         if sel is not (lambda r: True) else
                         f"{ok/max(n,1)*100:5.1f}%")
        print(f"  {mp.name:<26} {cells[0]:>18} {cells[1]:>18} "
              f"{cells[2].split()[0]:>12}")
    print(f"\n  Pooled over moves, not averaged over takes, so a long solve "
          f"is not\n  diluted by a short one. 'EVENING' is the regime the "
          f"augmentation targets.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, default=float))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
