"""
m4_ablation.py — the checkpoint ladder, every rung scored on the SAME
never-seen sessions.

Five checkpoints per seed exist, and they differ along four axes that were
each introduced one at a time. Scoring them all on one holdout is the only
way to read what each axis was worth, because the repo's historical numbers
were each measured on whatever holdout existed that week.

    move_joint_base   joint onset+class head, PEAK-PICKED       38 train
    move_ctc          same trunk, CTC loss + prefix beam        38 train
    move_ctc_aug      + widened photometric augmentation        38 train
    move_ctc_aug44    + 6 more (faster) training sessions       44 train
    move_ctc_spd      + speed/time-warp augmentation            44 train

The holdout is intersected across ALL of them, so every rung is scored on
sessions none of them has seen — including the earlier rungs, which saw
strictly fewer sessions and would otherwise be flattered by a holdout
derived from the last rung alone.

    python m4_ablation.py
"""

from __future__ import annotations

import numpy as np
import torch

import common as C

LADDER = [
    ("peak-pick (joint)", "move_joint_base"),
    ("CTC", "move_ctc"),
    ("CTC + photo-aug", "move_ctc_aug"),
    ("CTC + photo-aug + 6 sessions", "move_ctc_aug44"),
    ("CTC + photo + speed aug", "move_ctc_spd"),
]
SEEDS = ("s0", "s1")


def main():
    from model import build_joint_from_ckpt, score_stream_joint
    from dataset import JointSessionStream
    from ctc_decode import prefix_beam_decode, ctc_to_moves
    from joint_decode import posteriorgram_to_moves
    from decode import MIN_SEP

    dev = C.device()
    paths = [C.MD / "checkpoints" / f"{t}_{s}.pt"
             for _, t in LADDER for s in SEEDS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    dirs = C.holdout(paths, kind="solve")
    scrambles = C.holdout(paths, kind="scramble")
    print(f"\n  holdout intersected over {len(paths)} checkpoints: "
          f"{len(dirs)} solves + {len(scrambles)} scrambles\n")

    meta = {m["session"]: m for m in C.load("holdout_meta.json")}
    rows = []
    for label, tag in LADDER:
        for seed in SEEDS:
            cp = C.MD / "checkpoints" / f"{tag}_{seed}.pt"
            ck = torch.load(cp, map_location=dev, weights_only=False)
            model = build_joint_from_ckpt(ck, dev)
            model.eval()
            is_ctc = ck.get("model_type") == "ctc"
            for d in dirs:
                gt = C.truth_word(d)
                if gt is None:
                    continue
                stream = JointSessionStream(d / "detector_stream_color.npz")
                onset_prob, class_prob, count_prob = score_stream_joint(
                    model, stream, dev)
                if is_ctc:
                    lab, fr = prefix_beam_decode(
                        np.log(np.maximum(class_prob, 1e-12)), beam=16)
                    moves = ctc_to_moves(class_prob, lab, fr, fps=stream.fps)
                else:
                    moves = posteriorgram_to_moves(
                        onset_prob, class_prob,
                        threshold=ck.get("threshold", 0.5),
                        min_sep=ck.get("min_sep", MIN_SEP), fps=stream.fps,
                        count_prob=count_prob,
                        count_radius=ck.get("count_radius", 2))
                pred = [m["move"] for m in moves]
                ch = C.channel_split(gt, pred)
                m_, _ = C.mer(gt, pred)
                rows.append({"rung": label, "tag": tag, "seed": seed,
                             "session": d.name,
                             "evening": bool(meta[d.name]["evening"]),
                             **ch, "mer": m_})
            done = [r for r in rows if r["tag"] == tag and r["seed"] == seed]
            print(f"  {label:<32} {seed}  "
                  f"acc {np.mean([r['acc'] for r in done])*100:5.1f}%  "
                  f"MER {np.mean([r['mer'] for r in done])*100:5.1f}%  "
                  f"miss {sum(r['miss'] for r in done):3d}  "
                  f"phantom {sum(r['phantom'] for r in done):3d}  "
                  f"sub {sum(r['sub'] for r in done):3d}")

    C.dump("m4_ablation.json", rows)

    print(f"\n{'='*92}\n  LADDER — mean of per-session accuracy on "
          f"{len(dirs)} never-seen solves, both seeds\n{'='*92}")
    print(f"  {'rung':<32}{'daytime':>20}{'evening':>20}{'all':>16}")
    summ = []
    for label, tag in LADDER:
        cells = {}
        for regime in ("daytime", "evening", "all"):
            vals = []
            for seed in SEEDS:
                g = [r for r in rows if r["tag"] == tag and r["seed"] == seed
                     and (regime == "all"
                          or (regime == "evening") == r["evening"])]
                vals.append(np.mean([r["acc"] for r in g]))
            cells[regime] = (float(np.mean(vals)), float(np.ptp(vals)))
        summ.append({"rung": label, "tag": tag,
                     **{k: v[0] for k, v in cells.items()},
                     **{k + "_spread": v[1] for k, v in cells.items()}})
        print(f"  {label:<32}"
              + "".join(f"{cells[r][0]*100:>14.1f}% ±{cells[r][1]*100/2:>3.1f}"
                        if r != "all" else f"{cells[r][0]*100:>15.1f}%"
                        for r in ("daytime", "evening", "all")))
    C.dump("m4_summary.json", summ)
    print("\n  ± is half the two-seed spread, i.e. the seed noise on that "
          "cell. A rung whose gain is inside it is not a result.")


if __name__ == "__main__":
    main()
