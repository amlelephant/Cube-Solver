"""
count_check.py — how accurately can we count moves, ignoring which moves
they were?

ANTICHEAT.md issue 1, "new option": use the onset/move stack as the swap
detector by asking only "were roughly enough moves made to have legitimately
solved this cube". This measures the one number that idea rests on.

Why counting is the right ask of this model
-------------------------------------------
Move-by-move VERIFICATION is dead (ACCURACY_TARGET.md: a cliff at ~98-99%
per-move accuracy, decode ceiling measured below it). But that cliff is
about IDENTIFYING moves — which face, which direction. Counting discards
substitutions entirely and is sensitive only to insertions and deletions,
so it uses the model's strong axis and throws away its weak one. MER is
therefore an OVER-estimate of counting error, and the number that matters
here is the signed count error, measured below.

And the separation it needs to support is not marginal. A legit solve is
tens of moves; a substituted, already-solved cube needs none. It also
catches two attacks the appearance detector provably misses:
  * partial-solve-then-swap (solve one face, swap in a solved cube) —
    small appearance change, but far too few moves;
  * running the inverse scramble from a solver instead of solving (~20
    moves against a human's ~50+).

Honesty: sessions the checkpoint trained on are reported separately from
sessions it never saw. Only the unseen ones bound real-world behaviour.

Run from inside ble/move_detector/:
    python count_check.py --ctc checkpoints/move_ctc_aug44_s0.pt
    python count_check.py --ctc checkpoints/move_ctc_aug44_s1.pt --greedy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ctc_decode import BLANK, greedy_decode, prefix_beam_decode
from dataset import JointSessionStream
from model import build_joint_from_ckpt, score_stream_joint

SESSION_ROOT = Path("../training_data")


def load_truth_count(d: Path):
    """True move count from the session's own BLE metadata."""
    mp = d / "metadata.json"
    if not mp.is_file():
        return None
    m = json.loads(mp.read_text())
    if isinstance(m.get("move_count"), int):
        return m["move_count"]
    seq = m.get("wca_sequence")
    return len(seq) if isinstance(seq, list) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctc", required=True)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--greedy", action="store_true",
                    help="greedy CTC instead of prefix-beam (much faster; "
                         "counting may not need the beam — worth knowing)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ctc, map_location=device, weights_only=False)
    model = build_joint_from_ckpt(ck, device)
    model.eval()

    trained_on = set(ck.get("train_session_names") or [])
    print(f"checkpoint {args.ctc}")
    print(f"  trained on {len(trained_on)} sessions; "
          f"decode = {'greedy' if args.greedy else f'prefix-beam {args.beam}'}\n")

    sessions = sorted(d for d in SESSION_ROOT.iterdir()
                      if (d / "detector_stream_color.npz").is_file()
                      and load_truth_count(d) is not None)
    if args.limit:
        sessions = sessions[:args.limit]

    rows = []
    for i, d in enumerate(sessions):
        true_n = load_truth_count(d)
        s = JointSessionStream(d / "detector_stream_color.npz")
        _, cp, _ = score_stream_joint(model, s, device)
        lp = np.log(np.maximum(cp, 1e-12))
        if args.greedy:
            labels, _ = greedy_decode(lp)
        else:
            labels, _ = prefix_beam_decode(lp, beam=args.beam)
        pred_n = len(labels)
        seen = d.name in trained_on
        rows.append({"session": d.name, "true": true_n, "pred": pred_n,
                     "err": pred_n - true_n, "seen": seen})
        print(f"[{i+1}/{len(sessions)}] {d.name:38s} true={true_n:4d} "
              f"pred={pred_n:4d} err={pred_n - true_n:+4d} "
              f"{'(trained)' if seen else 'HELD-OUT'}")

    for label, sel in (("HELD-OUT (honest)", [r for r in rows if not r["seen"]]),
                       ("trained-on", [r for r in rows if r["seen"]])):
        if not sel:
            continue
        err = np.array([r["err"] for r in sel])
        rel = np.array([r["err"] / max(r["true"], 1) for r in sel])
        true = np.array([r["true"] for r in sel])
        pred = np.array([r["pred"] for r in sel])
        print(f"\n=== {label}: {len(sel)} sessions ===")
        print(f"  true moves      : min {true.min()} median {np.median(true):.0f} max {true.max()}")
        print(f"  predicted moves : min {pred.min()} median {np.median(pred):.0f} max {pred.max()}")
        print(f"  signed error    : mean {err.mean():+.1f}  median {np.median(err):+.0f}"
              f"  range [{err.min():+d}, {err.max():+d}]")
        print(f"  relative error  : mean {100*rel.mean():+.1f}%  "
              f"worst under-count {100*rel.min():+.1f}%")
        print(f"  LOWEST predicted count on a legit solve: {pred.min()}")
        print(f"    -> a substitution must be claimed below this to be "
              f"separable with zero false accusations.")

    out = Path(f"count_check_{Path(args.ctc).stem}.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {out}")
    print("\nThis measures COUNTING only. It says nothing about whether the "
          "moves were the RIGHT moves — that is the verification cliff, and "
          "is deliberately not what this defends.")


if __name__ == "__main__":
    main()
