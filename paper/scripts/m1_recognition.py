"""
m1_recognition.py — raw recognition accuracy of the joint CTC model.

Measures, on every session no evaluated checkpoint has seen:

  * per-move accuracy against the camera-frame BLE word
  * move error rate (Levenshtein / |truth|) and its sub/ins/del split
  * the ok/sub/miss/phantom channel decomposition
  * greedy (best-path) vs prefix-beam decoding, same posteriorgram
  * what the substitutions actually are (inverse / adjacent / opposite)

Caches each session's posteriorgram to paper/data/post/ so the figure
scripts and the decode measurement do not re-run the network.

    python m1_recognition.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import common as C

CKPTS = ["move_ctc_spd_s0.pt", "move_ctc_spd_s1.pt"]
POST = C.DATA / "post"


def posteriorgram(model, stream, device, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)["class_prob"]
    from model import score_stream_joint
    _, class_prob, _ = score_stream_joint(model, stream, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, class_prob=class_prob.astype(np.float32))
    return class_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    from model import build_joint_from_ckpt
    from dataset import JointSessionStream
    from ctc_decode import prefix_beam_decode, greedy_decode, ctc_to_moves

    dev = C.device()
    paths = [C.MD / "checkpoints" / c for c in CKPTS]
    dirs = C.holdout(paths)
    seen = C.ckpt_seen(paths)

    print(f"\n  checkpoints : {', '.join(CKPTS)}")
    print(f"  sessions the checkpoints have seen : {len(seen)}")
    print(f"  held-out prepared sessions         : {len(dirs)}"
          f"  ({sum(1 for d in dirs if d.name.endswith('_solve'))} solves,"
          f" {sum(1 for d in dirs if d.name.endswith('_scramble'))} scrambles)\n")

    metas = {d.name: C.session_meta(d) for d in dirs}
    C.dump("holdout_meta.json", list(metas.values()))

    rows = []
    for cp in paths:
        ck = torch.load(cp, map_location=dev, weights_only=False)
        model = build_joint_from_ckpt(ck, dev)
        model.eval()
        tag = cp.stem
        print(f"{'='*86}\n  {tag}  epoch {ck['epoch']}  seed {ck['seed']}  "
              f"(trained on {len(ck['train_session_names'])} sessions)\n{'='*86}")
        print(f"  {'session':<38}{'n':>5}{'acc':>8}{'MER':>8}"
              f"{'miss':>6}{'sub':>5}{'phan':>6}{'greedy':>9}")
        for d in dirs:
            gt = C.truth_word(d)
            if gt is None:
                print(f"  {d.name:<38}  unresolved truth — skipped")
                continue
            stream = JointSessionStream(d / "detector_stream_color.npz")
            cache = POST / f"{tag}__{d.name}.npz"
            if args.refresh and cache.exists():
                cache.unlink()
            cp_ = posteriorgram(model, stream, dev, cache)
            lp = np.log(np.maximum(cp_, 1e-12))

            lab_b, fr_b = prefix_beam_decode(lp, beam=args.beam)
            lab_g, fr_g = greedy_decode(lp)
            mv_b = ctc_to_moves(cp_, lab_b, fr_b, fps=stream.fps)
            mv_g = ctc_to_moves(cp_, lab_g, fr_g, fps=stream.fps)
            pred_b = [m["move"] for m in mv_b]
            pred_g = [m["move"] for m in mv_g]

            ch = C.channel_split(gt, pred_b)
            m, parts = C.mer(gt, pred_b)
            chg = C.channel_split(gt, pred_g)
            mg, _ = C.mer(gt, pred_g)

            from decode import align_sequences
            subs = [(w, p) for o, w, p in align_sequences(gt, pred_b)
                    if o == "sub"]
            kinds = {k: sum(1 for a, b in subs if C.sub_kind(a, b) == k)
                     for k in ("inverse", "adjacent", "opposite")}

            row = {"model": tag, "seed": ck["seed"], **metas[d.name],
                   **ch, "mer": m, "mer_parts": parts,
                   "acc_greedy": chg["acc"], "mer_greedy": mg,
                   "miss_greedy": chg["miss"], "phantom_greedy": chg["phantom"],
                   "sub_kinds": kinds,
                   "pred": pred_b, "gt": gt}
            rows.append(row)
            print(f"  {d.name:<38}{ch['n_gt']:>5}{ch['acc']*100:>7.1f}%"
                  f"{m*100:>7.1f}%{ch['miss']:>6}{ch['sub']:>5}"
                  f"{ch['phantom']:>6}{chg['acc']*100:>8.1f}%")

    C.dump("m1_recognition.json", rows)

    # ---- summary, split the way the ship metric is defined -------------
    print(f"\n{'='*86}\n  SUMMARY — mean of per-session accuracies "
          f"(never pooled over moves)\n{'='*86}")
    print(f"  {'model':<22}{'kind':<10}{'regime':<10}{'n':>4}"
          f"{'acc':>9}{'MER':>9}{'floor':>9}")
    summary = []
    for tag in [Path(c).stem for c in CKPTS]:
        for kind in ("solve", "scramble"):
            for regime in ("daytime", "evening", "all"):
                g = [r for r in rows if r["model"] == tag
                     and r["session"].endswith("_" + kind)
                     and (regime == "all"
                          or (regime == "evening") == bool(r["evening"]))]
                if not g:
                    continue
                s = {"model": tag, "kind": kind, "regime": regime,
                     "n_sessions": len(g),
                     "acc_mean": float(np.mean([r["acc"] for r in g])),
                     "acc_min": float(np.min([r["acc"] for r in g])),
                     "mer_mean": float(np.mean([r["mer"] for r in g])),
                     "acc_pooled": float(sum(r["ok"] for r in g)
                                         / sum(r["n_gt"] for r in g)),
                     "ctc_ceiling": float(np.mean([1 - r["ctc_floor"]
                                                   for r in g])),
                     "n_moves": int(sum(r["n_gt"] for r in g))}
                summary.append(s)
                print(f"  {tag:<22}{kind:<10}{regime:<10}{len(g):>4}"
                      f"{s['acc_mean']*100:>8.1f}%{s['mer_mean']*100:>8.1f}%"
                      f"{s['ctc_ceiling']*100:>8.1f}%")
    C.dump("m1_summary.json", summary)


if __name__ == "__main__":
    main()
