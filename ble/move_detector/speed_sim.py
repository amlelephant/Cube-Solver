"""
speed_sim.py — simulate faster solves from the recorded corpus by dropping
frames, to find where move COUNTING breaks down.

Motivation: the corpus is slow (~2-3 turns per second). The anticheat idea
in ANTICHEAT.md issue 1 rests on counting moves, and counting has only been
measured at corpus speed. A sub-15s solve is ~3.7 TPS and a world-class
solve is ~9-11 TPS, so the question is whether counting survives there.

Two stages, in the order a real cuber actually gets faster:

  Stage 1 — HESITATION REMOVAL. Drop frames BETWEEN turns. The turns
            themselves are untouched, so each turn looks exactly as it did;
            only the pauses shrink. This is a cuber with better lookahead,
            and it raises TPS without changing turn dynamics at all.
  Stage 2 — TURN COMPRESSION. Once the pauses are gone, drop frames INSIDE
            the turns. Each turn now spans fewer frames, i.e. the hands move
            faster. Only entered when Stage 1 has run out of slack, and the
            report says which stage each TPS level needed — crossing into
            Stage 2 is the point where the simulation stops being a
            relabelling of real footage.

Onset frames are ALWAYS kept, so ground truth remains exact and the
remapped onset indices never collide. That also caps the achievable TPS at
`fps` (30), far above any human.

WHAT THIS SIMULATION IS NOT
---------------------------
Dropping frames compresses time; it does not add motion blur. A real fast
turn is blurred in proportion to angular velocity, whereas these frames
carry the blur of the SLOW turn they were recorded at. So the frames are
sharper than genuinely fast footage and results are OPTIMISTIC — this
isolates TEMPORAL CROWDING (moves closer together in frame space), which is
the hypothesised failure, and does not test blur. `--blur` averages the
dropped frames into the frame that survives them, which approximates a
longer exposure and recovers part of the difference, but real fast
recordings remain the arbiter.

Expected failure mode, worth naming in advance: CTC collapses repeated
labels unless a blank separates them, so a half turn (R2 = R then R) needs a
clear gap between its two quarter turns. Crowding removes that gap first,
which should show up as systematic UNDER-counting.

Run from inside ble/move_detector/:
    python speed_sim.py --ctc checkpoints/move_ctc_aug44_s0.pt
    python speed_sim.py --ctc checkpoints/move_ctc_aug44_s0.pt --blur --sessions all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ctc_decode import greedy_decode, prefix_beam_decode
from dataset import JointArrayStream, JointSessionStream
from model import build_joint_from_ckpt, score_stream_joint

SESSION_ROOT = Path("../training_data")
TURN_HALFWIDTH = 2      # frames either side of an onset counted as "the turn"
                        #   (~5 frames = 165ms at 30fps, matching a quarter
                        #   turn; everything else is hesitation)
TPS_LEVELS = [2, 3, 4, 5, 6, 8, 10]
# reference points, ~55-move CFOP solve: 15s->3.7  12s->4.6  10s->5.5
#                                         8s->6.9   6s->9.2 (world record)


def _even_subset(idx, k):
    """k evenly-spaced elements of idx (all of it if k >= len)."""
    idx = np.asarray(idx)
    if k >= len(idx):
        return idx
    if k <= 0:
        return idx[:0]
    return idx[np.linspace(0, len(idx) - 1, k).round().astype(int)]


def warp(frames, onset_idx, fps, target_tps, turn_hw=TURN_HALFWIDTH,
         blur=False):
    """Resample a stream to `target_tps`. Returns (frames, onset_idx, info)."""
    n = len(frames)
    onset_idx = np.asarray(onset_idx, dtype=int)
    n_moves = len(onset_idx)
    if n_moves == 0:
        return frames, onset_idx, {"stage": 0, "orig_tps": 0.0, "tps": 0.0}
    orig_tps = n_moves * fps / n
    n_keep = int(round(n_moves * fps / target_tps))
    n_keep = max(n_keep, n_moves)          # onsets are never dropped

    is_turn = np.zeros(n, dtype=bool)
    for oi in onset_idx:
        is_turn[max(0, oi - turn_hw): oi + turn_hw + 1] = True
    turn_idx = np.flatnonzero(is_turn)
    hes_idx = np.flatnonzero(~is_turn)

    if n_keep >= len(turn_idx):
        # Stage 1: every turn frame survives; thin the pauses only.
        keep = np.union1d(turn_idx, _even_subset(hes_idx, n_keep - len(turn_idx)))
        stage = 1
    else:
        # Stage 2: pauses are gone; thin the turns, protecting the onsets.
        non_onset_turn = np.setdiff1d(turn_idx, onset_idx)
        keep = np.union1d(onset_idx,
                          _even_subset(non_onset_turn, n_keep - n_moves))
        stage = 2

    keep = np.unique(keep)
    if blur:
        # approximate a longer exposure: each surviving frame becomes the
        # mean of the frames it absorbed (nearest-kept assignment)
        assign = np.searchsorted(keep, np.arange(n))
        assign = np.clip(assign, 0, len(keep) - 1)
        left = np.clip(assign - 1, 0, len(keep) - 1)
        take_left = np.abs(np.arange(n) - keep[left]) < np.abs(np.arange(n) - keep[assign])
        assign = np.where(take_left, left, assign)
        acc = np.zeros((len(keep),) + frames.shape[1:], dtype=np.float64)
        cnt = np.zeros(len(keep), dtype=np.int64)
        np.add.at(acc, assign, frames.astype(np.float64))
        np.add.at(cnt, assign, 1)
        new_frames = (acc / np.maximum(cnt, 1)[:, None, None, None]).astype(frames.dtype)
    else:
        new_frames = frames[keep]

    new_onsets = np.searchsorted(keep, onset_idx)
    info = {"stage": stage, "orig_tps": float(orig_tps),
            "tps": float(n_moves * fps / len(keep)),
            "n_frames": int(len(keep)), "n_moves": int(n_moves),
            "collisions": int(n_moves - len(np.unique(new_onsets)))}
    return new_frames, new_onsets, info


def true_count(d: Path):
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
    ap.add_argument("--sessions", choices=["heldout", "all"], default="heldout")
    ap.add_argument("--blur", action="store_true",
                    help="average dropped frames into survivors (approximate "
                         "motion blur; closer to real fast footage)")
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--beam", type=int, default=0,
                    help="use prefix-beam with this width instead of greedy")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ctc, map_location=device, weights_only=False)
    model = build_joint_from_ckpt(ck, device)
    model.eval()
    trained = set(ck.get("train_session_names") or [])

    sessions = [d for d in sorted(SESSION_ROOT.iterdir())
                if (d / "detector_stream_color.npz").is_file()
                and true_count(d) is not None]
    if args.sessions == "heldout":
        sessions = [d for d in sessions if d.name not in trained]
    if args.limit:
        sessions = sessions[:args.limit]
    print(f"{args.ctc}\n{len(sessions)} sessions ({args.sessions}), "
          f"blur={args.blur}\n")

    rows = []
    for d in sessions:
        base = JointSessionStream(d / "detector_stream_color.npz")
        raw = np.load(d / "detector_stream_color.npz", allow_pickle=True)
        frames, onsets = raw["frames"], raw["onset_idx"]
        oclass = raw["onset_class"] if "onset_class" in raw else None
        fps = float(raw["fps"])
        tn = true_count(d)
        orig_tps = len(onsets) * fps / len(frames)
        print(f"{d.name}  true={tn}  native {orig_tps:.2f} TPS")
        for tps in TPS_LEVELS:
            if tps < orig_tps - 1e-6:
                continue                     # cannot slow a session down
            nf, no, info = warp(frames, onsets, fps, tps,
                                blur=args.blur)
            st = JointArrayStream(nf, name=d.name, fps=fps, onset_idx=no,
                                  onset_class=oclass)
            _, cp, _ = score_stream_joint(model, st, device)
            lp = np.log(np.maximum(cp, 1e-12))
            labels, _ = (prefix_beam_decode(lp, beam=args.beam) if args.beam
                         else greedy_decode(lp))
            pred = len(labels)
            err = pred - tn
            rows.append({"session": d.name, "target_tps": tps,
                         "tps": info["tps"], "stage": info["stage"],
                         "true": tn, "pred": pred, "err": err})
            print(f"    {tps:4.1f} TPS (stage {info['stage']}, "
                  f"{info['n_frames']:4d} frames)  pred={pred:4d} "
                  f"err={err:+4d}  {100*err/max(tn,1):+5.1f}%")

    print(f"\n{'TPS':>5} {'stage':>6} {'n':>3} {'mean err':>9} {'worst under':>12} "
          f"{'mean rel':>9}")
    for tps in TPS_LEVELS:
        sel = [r for r in rows if r["target_tps"] == tps]
        if not sel:
            continue
        e = np.array([r["err"] for r in sel])
        rel = np.array([r["err"] / max(r["true"], 1) for r in sel])
        stages = {r["stage"] for r in sel}
        print(f"{tps:5.1f} {'/'.join(map(str, sorted(stages))):>6} {len(sel):3d} "
              f"{e.mean():+9.1f} {e.min():+12d} {100*rel.mean():+8.1f}%")

    out = Path(f"speed_sim_{Path(args.ctc).stem}{'_blur' if args.blur else ''}.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {out}")
    print("Stage 1 = pauses removed only (turns untouched, most trustworthy).")
    print("Stage 2 = turns themselves compressed (extrapolation — verify "
          "against real fast recordings).")


if __name__ == "__main__":
    main()
