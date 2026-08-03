"""
compare_onset_arms.py

Scores the three onset-representation arms against each other on one
identical set of sessions, through one identical downstream:

  base    joint onset+class model, peak-picked          (train_joint.py)
  count   same + the 0/1/2 count head splitting merged
          peaks                                (train_joint.py --count-head)
  ctc     same trunk trained with CTC, prefix-beam decoded   (train_ctc.py)

Two metrics, deliberately both:

  MER   move error rate — Levenshtein distance from the true move sequence
        over its length, split into sub / ins / del. This is the only
        metric all three arms can share: a CTC model emits no aligned
        onsets, so onset F1 is not computable for it at all. `ins` is
        phantoms and `del` is misses, which is exactly the pair this
        experiment exists to move, and reporting them split matters
        because an aggregate rate hides one rising as the other falls.

  decode  verified / gt_path_cost through reconstruct.py, via
        verify_joint.decode_moves — literally the same function
        verify_joint.py uses, so these numbers sit on the same axis as
        every other verification number in this project.

MER is the fast read and decode is the ship-gate; a change that moves MER
but not verification has happened before in this project and is worth
seeing as such rather than being reported as a win.

The baseline is a checkpoint RETRAINED alongside the other two on the same
38/4 split (run_onset_arms.sh), not checkpoints/move_joint_seed0.pt — that one saw 36
sessions, and comparing across a data difference would attribute it to the
architecture.

Usage:
    python compare_onset_arms.py --base checkpoints/move_joint_base_s0.pt \\
        --count checkpoints/move_joint_count_s0.pt --ctc checkpoints/move_ctc_s0.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import reconstruct as RC
import verify_joint as VJ
from model import build_joint_from_ckpt, score_stream_joint
from joint_decode import posteriorgram_to_moves
from ctc_decode import prefix_beam_decode, ctc_to_moves, move_error_rate
from move_lm import MoveLM
from dataset import JointSessionStream
from decode import MIN_SEP


def load_arm(path: str, device):
    ckpt = torch.load(path, map_location=device)
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    model._ckpt_tag = f"{Path(path).stem}_e{ckpt['epoch']}"
    return model, ckpt


def moves_for(arm: str, model, ckpt, stream, device, args, lm=None
              ) -> list[dict]:
    """Produce the moves list for one arm on one session. This is the ONLY
    place the arms differ; everything downstream is shared."""
    onset_prob, class_prob, count_prob = score_stream_joint(model, stream,
                                                            device)
    if arm in ("ctc", "ctclm"):
        log_probs = np.log(np.maximum(class_prob, 1e-12))
        fuse = arm == "ctclm"
        labels, frames = prefix_beam_decode(
            log_probs, beam=args.beam_ctc,
            lm=lm if fuse else None,
            alpha=args.alpha if fuse else 0.0,
            beta=args.beta if fuse else 0.0)
        return ctc_to_moves(class_prob, labels, frames, fps=stream.fps)

    return posteriorgram_to_moves(
        onset_prob, class_prob,
        threshold=ckpt.get("threshold", 0.5),
        min_sep=ckpt.get("min_sep", MIN_SEP), fps=stream.fps,
        count_prob=count_prob if arm == "count" else None,
        count_radius=ckpt.get("count_radius", 2))


def true_sequence(d: Path) -> list[str] | None:
    records = [json.loads(l) for l in open(d / "moves.jsonl") if l.strip()]
    gt = [r.get("wca_notation") for r in records]
    return None if not gt or any(g is None for g in gt) else gt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--count", default=None)
    p.add_argument("--ctc", default=None)
    p.add_argument("--ctclm", default=None,
                   help="same checkpoint as --ctc, decoded with n-gram LM "
                        "fusion; pass the ctc checkpoint path again")
    p.add_argument("--lm-order", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.9,
                   help="LM weight, chosen on the unseen 07-29/30 dev "
                        "sessions (tune_lm_fusion.py)")
    p.add_argument("--beta", type=float, default=4.0,
                   help="per-symbol insertion bonus, chosen with alpha")
    p.add_argument("--sessions", nargs="+",
                   default=["../training_data/solve_*/"])
    p.add_argument("--beam-ctc", type=int, default=16)
    p.add_argument("--honest-only", action="store_true", default=True,
                   help="score only sessions unseen by the checkpoints "
                        "(default; --all-sessions to override)")
    p.add_argument("--all-sessions", dest="honest_only", action="store_false")
    p.add_argument("--no-decode", action="store_true",
                   help="MER only; skip the reconstruct.py decode (fast)")
    p.add_argument("--out", default=None)
    # reconstruct knobs, defaulted identically to verify_joint.py
    p.add_argument("--beam", type=int, default=RC.BEAM)
    p.add_argument("--retry-beam", type=int, default=4 * RC.BEAM)
    p.add_argument("--del-cost", type=float, default=RC.C_DEL)
    p.add_argument("--ins-cost", type=float, default=RC.C_INS)
    p.add_argument("--rot-cost", type=float, default=RC.C_ROT)
    p.add_argument("--max-end-ins", type=int, default=RC.MAX_END_INS)
    p.add_argument("--del-floor", type=float, default=RC.DEL_FLOOR)
    p.add_argument("--blend-inv", type=float, default=RC.BLEND_INV)
    p.add_argument("--blend-unif", type=float, default=RC.BLEND_UNIF)
    p.add_argument("--blend-adj", type=float, default=RC.BLEND_ADJ)
    p.add_argument("--rel-weight", type=float, default=RC.REL_WEIGHT)
    p.add_argument("--slices", action="store_true")
    p.add_argument("--c-slice", type=float, default=RC.C_SLICE, dest="c_slice")
    p.add_argument("--slice-gate", type=float, default=RC.SLICE_GATE,
                   dest="slice_gate")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = {}
    for name, path in (("base", args.base), ("count", args.count),
                       ("ctc", args.ctc), ("ctclm", args.ctclm)):
        if path:
            arms[name] = load_arm(path, device)
            print(f"  {name:<6} {path}  (epoch {arms[name][1]['epoch']})")

    lm = None
    if args.ctclm:
        # Fit on the acoustic model's OWN training sessions only. The
        # holdout must not appear here, or the decode would be reading its
        # own answer out of the prior.
        lm = MoveLM.from_sessions(list(arms["ctclm"][1]["train_session_names"]),
                                  order=args.lm_order)
        print(f"  LM     order {args.lm_order}, {lm.n_sequences} sessions / "
              f"{lm.n_moves} moves, alpha={args.alpha} beta={args.beta}")

    dirs = VJ.resolve_sessions(args.sessions)
    if args.honest_only:
        base_ckpt = arms["base"][1]
        dirs = [d for d in dirs if VJ.joint_unseen(d.name, base_ckpt)]
    if not dirs:
        sys.exit("No sessions to score.")
    print(f"\n  {len(dirs)} session(s): "
          f"{', '.join(d.name for d in dirs)}")

    tables = None if args.no_decode else RC.build_tables()
    results = {a: [] for a in arms}

    for d in dirs:
        gt = true_sequence(d)
        if gt is None:
            print(f"  {d.name}: unresolved wca_notation — skipping")
            continue
        gt_idx = [RC.WCA12.index(g) for g in gt if g in RC.WCA12]
        if len(gt_idx) != len(gt):
            print(f"  {d.name}: non-quarter-turn ground truth — skipping")
            continue

        stream = JointSessionStream(d / "detector_stream_color.npz")
        for arm, (model, ckpt) in arms.items():
            moves = moves_for(arm, model, ckpt, stream, device, args, lm)
            pred_idx = [RC.WCA12.index(m["move"]) for m in moves]
            mer, parts = move_error_rate(pred_idx, gt_idx)
            row = {"session": d.name, "arm": arm, "mer": mer, **parts}
            if not args.no_decode and moves:
                dec = VJ.decode_moves(d, moves, stream.onset_idx, ckpt, args,
                                      tables)
                if dec:
                    row.update({"solved": dec["solved"],
                                "gt_path_cost": dec["gt_path_cost"],
                                "raw_acc": dec["raw_acc"]})
            results[arm].append(row)

        line = "  ".join(
            f"{a}: MER {results[a][-1]['mer']*100:5.1f}% "
            f"(i{results[a][-1]['ins']}/d{results[a][-1]['del']}"
            f"/s{results[a][-1]['sub']})" for a in arms)
        print(f"\n  {d.name}  ({len(gt)} true moves)\n    {line}")

    print(f"\n{'='*78}\n  AGGREGATE over {len(dirs)} held-out session(s)\n{'='*78}")
    print(f"  {'arm':<8} {'MER':<9} {'sub':<7} {'ins(phantom)':<14} "
          f"{'del(miss)':<11} {'verified':<10} {'median gtc'}")
    print(f"  {'-'*76}")
    for arm, rows in results.items():
        if not rows:
            continue
        n_true = sum(r["n_true"] for r in rows)
        s, i, dl = (sum(r[k] for r in rows) for k in ("sub", "ins", "del"))
        mer = (s + i + dl) / max(n_true, 1)
        gtc = [r["gt_path_cost"] for r in rows if "gt_path_cost" in r]
        ver_s = f"{sum(bool(r.get('solved')) for r in rows)}/{len(rows)}" \
            if gtc else "-"
        gtc_s = f"{np.median(gtc):.1f}" if gtc else "-"
        print(f"  {arm:<8} {mer*100:<8.1f}% {s:<7} {i:<14} {dl:<11} "
              f"{ver_s:<10} {gtc_s}")
    any_rows = next(iter(results.values()))
    print(f"\n  MER is over {sum(r['n_true'] for r in any_rows)} true moves. "
          f"ins = phantoms, del = misses.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=float))
        print(f"\n  Written to {args.out}")


if __name__ == "__main__":
    main()
