"""
crossday_verify.py

Verification on the CROSS-DAY sessions (07-29 / 07-30) — the regime the
pipeline actually has to work in — for base / ctc / ctc+LM, with
leave-one-out tuning of the fusion weights.

Why this file exists
--------------------
Every verification number in this project so far comes from the 4-session
"honest" holdout, and those sessions are same-DAY as training. Unfused CTC
scores 5.1% MER there and 17.3% on these sessions: the holdout is not the
deployment regime, and a decoder aimed at the hard regime was being graded
on the easy one.

Two methodological points, both learned the hard way:

  * **Scrambles are excluded.** A scramble is a RANDOM sequence; a
    solve-structure n-gram prior does not apply to it and including the
    three scramble sessions in an alpha/beta sweep drags alpha down toward
    zero. Only the three solve sessions are used.

  * **Leave-one-out, not a fixed split.** With three sessions any fixed
    tune/test split wastes most of the data and its result is one session
    wide. Here each session is verified using weights tuned on the OTHER
    TWO only, so every session is a test session and none is ever tuned
    on. `alpha`/`beta` therefore differ per fold, which is the honest
    representation of "weights chosen without seeing this session".

The base and ctc arms have no tuned parameters, so LOO changes nothing for
them; they are folded through the identical loop anyway so all three arms
meet the decoder in exactly the same way.

Usage:
    python crossday_verify.py --base checkpoints/move_joint_base_s0.pt \\
        --ctc checkpoints/move_ctc_s0.pt --out results/2026-07-31/crossday_s0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import reconstruct as RC
import verify_joint as VJ
from model import build_joint_from_ckpt, score_stream_joint
from joint_decode import posteriorgram_to_moves
from ctc_decode import prefix_beam_decode, ctc_to_moves, move_error_rate
from move_lm import MoveLM, load_truth, SESSION_ROOT
from dataset import JointSessionStream
from decode import MIN_SEP

CROSSDAY_SOLVES = ["solve_20260729_221809_solve",
                   "solve_20260730_111941_solve",
                   "solve_20260730_113054_solve"]


def tune_on(cached, names, lm, alphas, betas, beam):
    """Grid-search alpha/beta by MER over the named sessions only."""
    best = None
    for a in alphas:
        for b in betas:
            s = i = d = n = 0
            for nm, lp, gt in cached:
                if nm not in names:
                    continue
                labels, _ = prefix_beam_decode(lp, beam=beam, lm=lm,
                                               alpha=a, beta=b)
                _, p = move_error_rate(labels, gt)
                s += p["sub"]; i += p["ins"]; d += p["del"]; n += p["n_true"]
            mer = (s + i + d) / max(n, 1)
            if best is None or mer < best[0]:
                best = (mer, a, b)
    return best[1], best[2], best[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--ctc", required=True)
    p.add_argument("--lm-order", type=int, default=4)
    p.add_argument("--beam-ctc", type=int, default=16)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.5, 0.7, 0.9, 1.2])
    p.add_argument("--betas", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0, 4.0, 5.0])
    p.add_argument("--out", default=None)
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
    base_ck = torch.load(args.base, map_location=device)
    ctc_ck = torch.load(args.ctc, map_location=device)
    base_m = build_joint_from_ckpt(base_ck, device); base_m.eval()
    ctc_m = build_joint_from_ckpt(ctc_ck, device); ctc_m.eval()

    for nm, ck in (("base", base_ck), ("ctc", ctc_ck)):
        seen = set(ck.get("train_session_names") or []) | \
               set(ck.get("val_session_names") or [])
        overlap = seen & set(CROSSDAY_SOLVES)
        if overlap:
            raise SystemExit(f"{nm} checkpoint has SEEN {sorted(overlap)} — "
                             f"these sessions are not cross-day for it.")
    print(f"  base {args.base}\n  ctc  {args.ctc}")
    print(f"  neither checkpoint has seen any cross-day session — verified\n")

    lm = MoveLM.from_sessions(list(ctc_ck["train_session_names"]),
                              order=args.lm_order)
    print(f"  LM order {args.lm_order}, {lm.n_sequences} sessions / "
          f"{lm.n_moves} moves (training sessions only)\n")

    # score once per session per model
    cached_ctc, streams = [], {}
    for n in CROSSDAY_SOLVES:
        d = SESSION_ROOT / n
        s = JointSessionStream(d / "detector_stream_color.npz")
        streams[n] = s
        _, cp, _ = score_stream_joint(ctc_m, s, device)
        cached_ctc.append((n, np.log(np.maximum(cp, 1e-12)), load_truth(d)))
    logp_by_session = {n: lp for n, lp, _ in cached_ctc}

    tables = RC.build_tables()
    rows = {"base": [], "ctc": [], "ctclm": []}

    for n in CROSSDAY_SOLVES:
        d = SESSION_ROOT / n
        s = streams[n]
        gt = load_truth(d)
        others = [x for x in CROSSDAY_SOLVES if x != n]
        a, b, dev_mer = tune_on(cached_ctc, set(others), lm, args.alphas,
                                args.betas, args.beam_ctc)
        print(f"\n  {n}  ({len(gt)} true moves)")
        print(f"    LOO weights from {', '.join(x[-19:] for x in others)}: "
              f"alpha={a} beta={b} (their MER {dev_mer*100:.1f}%)")

        lp = logp_by_session[n]

        op_b, cp_b, cnt_b = score_stream_joint(base_m, s, device)
        variants = {
            "base": posteriorgram_to_moves(
                op_b, cp_b, threshold=base_ck.get("threshold", 0.5),
                min_sep=base_ck.get("min_sep", MIN_SEP), fps=s.fps),
            "ctc": None, "ctclm": None,
        }
        cp_ctc = np.exp(lp)
        for arm, (al, be) in (("ctc", (0.0, 0.0)), ("ctclm", (a, b))):
            labels, frames = prefix_beam_decode(
                lp, beam=args.beam_ctc, lm=lm if al else None,
                alpha=al, beta=be)
            variants[arm] = ctc_to_moves(cp_ctc, labels, frames, fps=s.fps)

        for arm, moves in variants.items():
            pred = [RC.WCA12.index(m["move"]) for m in moves]
            mer, parts = move_error_rate(pred, gt)
            ck = ctc_ck if arm != "base" else base_ck
            dec = VJ.decode_moves(d, moves, s.onset_idx, ck, args, tables)
            row = {"session": n, "arm": arm, "mer": mer, **parts,
                   "alpha": a if arm == "ctclm" else None,
                   "beta": b if arm == "ctclm" else None}
            if dec:
                row.update(solved=dec["solved"],
                           gt_path_cost=dec["gt_path_cost"])
            rows[arm].append(row)
            print(f"      {arm:<7} MER {mer*100:5.1f}%  "
                  f"(i{parts['ins']}/d{parts['del']}/s{parts['sub']})  "
                  f"{'VERIFIED' if row.get('solved') else 'not verified'}  "
                  f"gtc {row.get('gt_path_cost', float('nan')):.1f}")

    print(f"\n{'='*76}\n  CROSS-DAY AGGREGATE ({len(CROSSDAY_SOLVES)} solve "
          f"sessions)\n{'='*76}")
    print(f"  {'arm':<8} {'MER':<9} {'sub':<6} {'ins':<6} {'del':<6} "
          f"{'verified':<10} {'median gtc'}")
    print(f"  {'-'*74}")
    for arm, rs in rows.items():
        n_true = sum(r["n_true"] for r in rs)
        s_, i_, d_ = (sum(r[k] for r in rs) for k in ("sub", "ins", "del"))
        gtc = [r["gt_path_cost"] for r in rs if "gt_path_cost" in r]
        ver = sum(bool(r.get("solved")) for r in rs)
        print(f"  {arm:<8} {(s_+i_+d_)/n_true*100:<8.1f}% {s_:<6} {i_:<6} "
              f"{d_:<6} {f'{ver}/{len(rs)}':<10} "
              f"{np.median(gtc) if gtc else float('nan'):.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, default=float))
        print(f"\n  Written to {args.out}")


if __name__ == "__main__":
    main()
