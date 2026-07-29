"""
verify_joint.py

Stage A measurement block (MODEL_REWORK_PLAN.md): decode the joint
model's own posteriorgram (joint_decode.py) through the IDENTICAL
reconstruct.py machinery every other measurement in this project uses —
same cost model, same beam/retry policy, same falsifiability sweep — so
the result is directly comparable to oracle_attribution.py's "real"
column (the deployed detector+classifier baseline) on the exact same
sessions, and to PATH_TO_VERIFICATION.md's headline numbers.

Reports, per session and aggregated over the honest (checkpoint-unseen)
set:
  - raw per-move accuracy (sub/miss/phantom breakdown, reconstruct.
    score_vs_gt) — the pre-decode number, comparable to "raw pipeline"
    elsewhere in this project
  - decode: verified / cost / gt_path_cost (reconstruct.decode_between +
    costs_from_moves) — comparable to oracle_attribution.py's "real"
    column and PATH_TO_VERIFICATION.md's honest-subset headline
  - falsifiability sweep on sessions that verify (verify_solve.
    falsifiability_sweep, unchanged)

Usage:
    python verify_joint.py --model move_joint_seed0.pt
    python verify_joint.py --model move_joint_seed0.pt --model2 move_joint_seed1.pt
    python verify_joint.py --model move_joint_seed0.pt \\
        --sessions ../training_data/solve_20260721_102711/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import reconstruct as RC
import verify_solve as VS
from model import build_joint_model
from joint_decode import load_joint_replay


def resolve_sessions(patterns: list[str]) -> list[Path]:
    dirs = [Path(p) for pattern in patterns
            for p in (Path(".").glob(pattern) if "*" in pattern
                      else [Path(pattern)])]
    return sorted(d for d in dirs if d.is_dir()
                 and (d / "detector_stream_color.npz").exists())


def joint_unseen(name: str, ckpt: dict) -> bool:
    """
    Unlike reconstruct.classifier_unseen (which reads the DEPLOYED
    classifier's checkpoint), this reads the JOINT checkpoint's own
    val_session_names/train_session_names directly — both are always
    written by train_joint.py, so there is no legacy-checkpoint fallback
    to reproduce here.
    """
    if name in set(ckpt.get("val_session_names") or []):
        return True
    return name not in set(ckpt.get("train_session_names") or [])


def decode_session(d: Path, model, device, ckpt, args, tables) -> dict | None:
    replay = load_joint_replay(d, model, device,
                               sigma=ckpt.get("sigma", 1.0),
                               threshold=ckpt.get("threshold", 0.5),
                               min_sep=ckpt.get("min_sep", 2),
                               refresh_cache=args.refresh_cache)
    if replay is None or not replay["moves"]:
        return None

    gt_records = [json.loads(l) for l in open(d / "moves.jsonl") if l.strip()]
    gt = [r.get("wca_notation") for r in gt_records]
    if not gt or any(g is None for g in gt):
        print(f"  {d.name}: unresolved wca_notation — skipping")
        return None

    start = RC.start_from_gt(gt)
    end = RC.SOLVED.copy()
    threshold = ckpt.get("threshold", 0.5)

    pred_names, cost_rows, del_costs = RC.costs_from_moves(
        replay["moves"], threshold, args.blend_inv, args.blend_unif,
        args.del_cost, None, args.del_floor, args.blend_adj)

    raw = RC.score_vs_gt(gt, pred_names)

    kw = dict(c_del=args.del_cost, c_ins=args.ins_cost, c_rot=args.rot_cost,
             max_end_ins=args.max_end_ins, rel_weight=args.rel_weight,
             del_costs=del_costs, rotations=False, tables=tables)
    t0 = time.time()
    res = RC.decode_between(start, end, cost_rows, beam=args.beam, **kw)
    if not res["solved"] and args.retry_beam > args.beam:
        res = RC.decode_between(start, end, cost_rows, beam=args.retry_beam,
                                **kw)
        res["retried"] = True
    dt = time.time() - t0

    gtc = RC.gt_path_cost(gt, pred_names, cost_rows, del_costs, args.ins_cost)
    out = {"session": d.name, "n_gt": len(gt), "n_pred": len(pred_names),
          "raw_acc": raw["acc"], "raw_sub": raw["sub"], "raw_miss": raw["miss"],
          "raw_phantom": raw["phantom"], "solved": res["solved"],
          "cost": res.get("cost"), "gt_path_cost": gtc,
          "decode_seconds": dt,
          "cost_rows": cost_rows, "del_costs": del_costs,
          "beam_used": args.retry_beam if res.get("retried") else args.beam}
    if res["solved"]:
        rec = RC.score_vs_gt(gt, res["moves"])
        out["dec_acc"] = rec["acc"]
        out["exact"] = rec["exact"]
    return out


def print_session(o: dict, unseen: bool):
    tag = " [unseen]" if unseen else " [trained on]"
    print(f"\n  {o['session']}{tag}  ({o['n_gt']} true moves, "
          f"{o['n_pred']} joint-model candidates)")
    print(f"    raw     {o['raw_acc']*100:5.1f}%   ({o['raw_sub']} sub, "
          f"{o['raw_miss']} miss, {o['raw_phantom']} phantom)")
    if o["solved"]:
        print(f"    decode  VERIFIED  cost {o['cost']:.2f}  vs gt_path_cost "
              f"{o['gt_path_cost']:.2f}  ({o['decode_seconds']:.1f}s"
              f", beam {o['beam_used']})")
        print(f"    decoded {o['dec_acc']*100:5.1f}%  "
              f"{'EXACT' if o['exact'] else ''}")
    else:
        print(f"    decode  NOT VERIFIED  gt_path_cost {o['gt_path_cost']:.2f}"
              f"  ({o['decode_seconds']:.1f}s, beam {o['beam_used']})")


def print_summary(results: list[dict], unseen_names: set, label: str):
    results = [r for r in results if r is not None]
    if not results:
        return
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    for subset_name, rows in (("ALL", results),
                              ("UNSEEN", [r for r in results
                                         if r["session"] in unseen_names])):
        if not rows:
            continue
        n = len(rows)
        ver = sum(r["solved"] for r in rows)
        exact = sum(r.get("exact", False) for r in rows)
        raw = np.mean([r["raw_acc"] for r in rows])
        eff = np.mean([r.get("dec_acc", r["raw_acc"]) for r in rows])
        print(f"  {subset_name} ({n} sessions): verified {ver}/{n}, exact "
              f"{exact}/{n}, raw {raw*100:.1f}% -> system {eff*100:.1f}%")
        gtc = [r["gt_path_cost"] for r in rows]
        print(f"    gt_path_cost: min {min(gtc):.1f}  median "
              f"{np.median(gtc):.1f}  max {max(gtc):.1f}")


def main():
    p = argparse.ArgumentParser(
        description="Stage A measurement: decode the joint model through "
                    "reconstruct.py, comparable to oracle_attribution.py's "
                    "'real' column.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--sessions", nargs="+", default=["../training_data/solve_*/"])
    p.add_argument("--refresh-cache", action="store_true")
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
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = build_joint_model(device, in_channels=ckpt.get("in_channels", 4),
                              n_classes=ckpt.get("n_classes", 13))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model._ckpt_tag = f"{Path(args.model).stem}_e{ckpt['epoch']}"

    dirs = resolve_sessions(args.sessions)
    if not dirs:
        sys.exit("No color-prepped session directories found.")

    unseen_names = {d.name for d in dirs if joint_unseen(d.name, ckpt)}
    print(f"\n  Model: {args.model}  (epoch {ckpt['epoch']}, seed "
          f"{ckpt.get('seed','?')}, val F1 {ckpt.get('val_f1',0)*100:.1f}%, "
          f"at_onset {ckpt.get('val_at_onset',0)*100:.1f}%)")
    print(f"  {len(dirs)} session(s), {len(unseen_names)} unseen by this "
          f"checkpoint")

    tables = RC.build_tables()
    results = []
    for d in dirs:
        o = decode_session(d, model, device, ckpt, args, tables)
        if o is None:
            continue
        print_session(o, d.name in unseen_names)

        if o["solved"] and not args.no_sweep:
            vs_args = argparse.Namespace(
                beam=args.beam, retry_beam=args.retry_beam,
                del_cost=args.del_cost, ins_cost=args.ins_cost,
                rot_cost=args.rot_cost, max_end_ins=args.max_end_ins,
                rel_weight=args.rel_weight, rotations=False, bidir=False)
            gt_records = [json.loads(l) for l in open(d / "moves.jsonl")
                         if l.strip()]
            gt = [r["wca_notation"] for r in gt_records]
            start = RC.start_from_gt(gt)
            o["sweep"] = VS.falsifiability_sweep(
                start, RC.SOLVED.copy(), o["cost_rows"], o["del_costs"],
                vs_args, tables, seed=0, beam=o["beam_used"])
        results.append(o)

    print_summary(results, unseen_names,
                  f"STAGE A — {Path(args.model).name}")

    if args.out:
        clean = [{k: v for k, v in r.items()
                 if k not in ("cost_rows", "del_costs")} for r in results]
        Path(args.out).write_text(json.dumps(clean, indent=2, default=str))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
