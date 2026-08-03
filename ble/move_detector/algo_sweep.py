"""
algo_sweep.py

Does OP_ALGO actually fill algorithms in, on sessions the model has never
seen — and does it buy verifications without buying credulity?

`algorithm_gate.py` answered the prior question (can a last-layer algorithm
be identified from the posteriorgram: yes, 91-97% held out). This answers
the only one that decides anything: wired into `reconstruct.py`'s beam, does
the decode change, in the right direction, on unseen data.

Three things are measured, and the third is not optional:

  fired    how many OP_ALGO transitions the WINNING story took, how many
           moves they emitted, and — the actual question — how many
           ground-truth moves the raw CTC decode had MISSED inside those
           onset spans and the prior therefore filled in. A prior that
           fires but recovers nothing is a prior that is being paid for
           and doing no work.
  decode   verified / exact / gt_path_cost, baseline and --algo run BACK TO
           BACK in the same sitting. No recalled baselines: this project
           has drawn a wrong conclusion that way before.
  decoys   the same claims re-decoded from a start state 1/2/4 moves off,
           and from a cube that was never scrambled, under the IDENTICAL
           cost model. A cost model more generous to the true claim than to
           its decoys inflates every verdict without making any of them
           more true. The prior makes the true story cheaper; it makes
           near-decoys cheaper too, since they share most of their moves.

Usage (from inside move_detector/):

    python algo_sweep.py --ctc checkpoints/move_ctc_s0.pt
    python algo_sweep.py --ctc checkpoints/move_ctc_s1.pt --out results/2026-08-01/algo_sweep_s1.json
    python algo_sweep.py --ctc checkpoints/move_ctc_s0.pt --all-sessions
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import algorithm_gate as AG
import reconstruct as RC
import verify_joint as VJ
from ctc_decode import prefix_beam_decode, ctc_to_moves
from decode import align_sequences
from model import build_joint_from_ckpt


def truth(d: Path) -> list[str] | None:
    recs = [json.loads(l) for l in open(d / "moves.jsonl") if l.strip()]
    w = [r.get("wca_notation") for r in recs]
    return None if not w or any(x is None or x not in RC.WCA12
                                for x in w) else w


def missed_gt_indices(gt: list[str], pred: list[str]) -> list[int]:
    """Ground-truth positions the raw decode never reported."""
    out, i = [], 0
    for op, want, _ in align_sequences(gt, pred):
        if want is None:
            continue
        if op == "miss":
            out.append(i)
        i += 1
    return out


def onset_to_gt(gt: list[str], pred: list[str]) -> dict[int, int]:
    """Predicted-onset index -> the ground-truth index it aligned to."""
    m, gi, pi = {}, 0, 0
    for op, want, got in align_sequences(gt, pred):
        if want is not None and got is not None:
            m[pi] = gi
        if got is not None:
            pi += 1
        if want is not None:
            gi += 1
    return m


def fired_report(res: dict, gt: list[str], pred: list[str]) -> dict:
    """What the OP_ALGO transitions in the winning story actually did."""
    fired = res.get("algo_fired") or []
    if not fired:
        return {"n_algo": 0, "algo_moves": 0, "recovered": 0, "words": []}
    o2g = onset_to_gt(gt, pred)
    missed = set(missed_gt_indices(gt, pred))
    recovered = 0
    for f in fired:
        # Ground-truth window this span covers, bracketed by whatever the
        # aligner could pin down inside it.
        gis = [o2g[o] for o in range(f["onset"], f["onset"] + f["n_onsets"])
               if o in o2g]
        if not gis:
            continue
        lo, hi = min(gis), max(gis)
        recovered += sum(1 for g in missed if lo <= g <= hi)
    return {"n_algo": len(fired),
            "algo_moves": sum(len(f["word"]) for f in fired),
            "recovered": recovered,
            "words": [" ".join(f["word"]) for f in fired]}


def post_acc(gt: list[str], res: dict,
             field: str = "best_effort_moves") -> float | None:
    """
    Accuracy of the decoded word against ground truth, on the MAP story
    whether or not it verified.

    Deliberately separate from `solved`. A decode that misses the binary
    check by one move scores ~99% here and 0 there, and on a corpus where
    most held-out sessions currently fail the binary check, the graded
    number is the only one that can show movement. It is NOT a verification
    claim — a best-effort story that does not reach solved proves nothing
    about the cube — but it is what a move-recording product is judged on.
    """
    moves = res.get(field) or res.get("best_effort_moves")
    if not moves:
        return None
    return RC.score_vs_gt(gt, moves)["acc"]


DECOYS = [("start 1 move off", 1), ("start 2 moves off", 2),
          ("start 4 moves off", 4)]


def decoy_starts(start: np.ndarray, rng) -> list[tuple[str, np.ndarray]]:
    out = []
    for name, dist in DECOYS:
        s = start.copy()
        for _ in range(dist):
            s = RC.compose(s, RC.CLASS_VECS[int(rng.integers(12))])
        out.append((name, s))
    out.append(("never scrambled", RC.SOLVED.copy()))
    return out


def decode_arm(start, cost_rows, del_costs, args, tables, cands):
    kw = dict(c_del=args.del_cost, c_ins=args.ins_cost,
              max_end_ins=args.max_end_ins, rel_weight=args.rel_weight,
              del_costs=del_costs, rotations=False, tables=tables,
              slices=args.slices, c_slice=args.c_slice,
              algo_cands=cands)
    res = RC.decode(start, cost_rows, beam=args.beam, **kw)
    if not res["solved"] and args.retry_beam > args.beam:
        res = RC.decode(start, cost_rows, beam=args.retry_beam, **kw)
        res["retried"] = True
    return res


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ctc", required=True)
    p.add_argument("--sessions", nargs="+",
                   default=["../training_data/solve_*/"])
    p.add_argument("--library-glob", default="../training_data/solve_*/",
                   help="corpus the library is mined from (training "
                        "sessions within it only); independent of "
                        "--sessions on purpose")
    p.add_argument("--honest-only", action="store_true", default=True)
    p.add_argument("--all-sessions", dest="honest_only", action="store_false")
    p.add_argument("--beam", type=int, default=RC.BEAM)
    p.add_argument("--retry-beam", type=int, default=4 * RC.BEAM)
    p.add_argument("--beam-ctc", type=int, default=16)
    p.add_argument("--anchor-beam", type=int, default=1000,
                   help="beam for the anchored prefix decode (both "
                        "endpoints pinned, ~half the onsets)")
    p.add_argument("--max-anchors", type=int, default=40)
    p.add_argument("--algo-gate", type=float, default=AG.ALGO_GATE)
    p.add_argument("--c-algo", type=float, default=AG.C_ALGO)
    p.add_argument("--trust-nll", type=float, default=AG.TRUST_NLL,
                   help="worst per-link identification cost at which a peel "
                        "is trusted to replace the plain decode's tail when "
                        "nothing verifies (ALGORITHM_PRIOR §9b). OFF by "
                        "default: measured at -0.6 daytime, §9d. Pass a "
                        "value to re-measure the arm.")
    p.add_argument("--del-cost", type=float, default=RC.C_DEL)
    p.add_argument("--ins-cost", type=float, default=RC.C_INS)
    p.add_argument("--max-end-ins", type=int, default=RC.MAX_END_INS)
    p.add_argument("--rel-weight", type=float, default=RC.REL_WEIGHT)
    p.add_argument("--slices", action="store_true", default=True)
    p.add_argument("--no-slices", dest="slices", action="store_false")
    p.add_argument("--c-slice", type=float, default=RC.C_SLICE)
    p.add_argument("--no-decoys", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ctc, map_location=device, weights_only=False)
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    tag = f"{Path(args.ctc).stem}_e{ckpt['epoch']}"
    train_names = set(ckpt.get("train_session_names", []))

    dirs = [d for d in VJ.resolve_sessions(args.sessions)
            if (d / "detector_stream_color.npz").exists()]
    if args.honest_only:
        dirs = [d for d in dirs if d.name not in train_names]
    if not dirs:
        sys.exit("No sessions to score.")

    # Library from TRAINING sessions only — mining it from the sessions
    # being scored would be reading the answer out of the prior. Mined over
    # the whole corpus, NOT over --sessions: narrowing the scored set must
    # not silently narrow the prior as well.
    lib_dirs = [d for d in VJ.resolve_sessions([args.library_glob])
                if d.name in train_names]
    library = [w for w, _ in AG.build_library(lib_dirs).most_common()]
    print(f"  {args.ctc} (epoch {ckpt['epoch']}); library {len(library)} "
          f"words from {len(lib_dirs)} training sessions")
    print(f"  {len(dirs)} session(s) to score"
          f"{' (held out only)' if args.honest_only else ''}\n")

    tables = RC.build_tables()
    rows = []
    for d in dirs:
        gt = truth(d)
        if gt is None:
            print(f"  {d.name}: unresolved ground truth — skipping")
            continue
        class_prob, stream = AG.posteriorgram(d, model, device,
                                              args.refresh, tag)
        log_probs = np.log(np.maximum(class_prob, 1e-12))
        labels, frames = prefix_beam_decode(log_probs, beam=args.beam_ctc)
        moves = ctc_to_moves(class_prob, labels, frames, fps=stream.fps)
        if not moves:
            print(f"  {d.name}: empty decode — skipping")
            continue

        pred, cost_rows, del_costs = RC.costs_from_moves(
            moves, ckpt.get("threshold", 0.5), RC.BLEND_INV, RC.BLEND_UNIF,
            args.del_cost, None, RC.DEL_FLOOR, RC.BLEND_ADJ)
        start = RC.start_from_gt(gt)
        raw = RC.score_vs_gt(gt, pred)

        t0 = time.time()
        cands = AG.candidates_for_session(moves, class_prob, library,
                                          gate=args.algo_gate,
                                          c_algo=args.c_algo)
        t_cand = time.time() - t0

        # Back to back, same sitting, identical everything but the arm.
        base = decode_arm(start, cost_rows, del_costs, args, tables, None)
        algo = decode_arm(start, cost_rows, del_costs, args, tables, cands)
        # Backward-first: peel the last layer off the END (where the state
        # is known exactly), then decode the remainder against the anchor
        # that exposes. Unlike the forward arm this needs no correct prefix.
        # The prefix sub-problem has both endpoints pinned and roughly half
        # the onsets, so it does not need the full-sequence beam — that is
        # the point of splitting it. A narrower beam here is what makes
        # trying many anchors affordable.
        bwd = AG.decode_backward_first(
            start, cost_rows, del_costs, cands,
            max_anchors=args.max_anchors, trust_nll=args.trust_nll,
            beam=args.anchor_beam, c_del=args.del_cost, c_ins=args.ins_cost,
            max_end_ins=args.max_end_ins, rel_weight=args.rel_weight,
            rotations=False, tables=tables, slices=args.slices,
            c_slice=args.c_slice)
        # The do-nothing anchor IS the baseline decode, and the method's
        # never-worse-than-baseline property depends on it being decoded at
        # the same budget. It is not: anchors are screened at --anchor-beam
        # so that trying forty of them is affordable, which left the trivial
        # anchor decoded more weakly than the baseline and produced a
        # spurious regression on seed 1. Fold the full-budget baseline back
        # in explicitly rather than paying full beam forty times.
        if base["solved"] and (not bwd["solved"]
                               or base["cost"] < bwd["cost"]):
            bwd = {**base, "anchor_onset": len(cost_rows),
                   "anchor_cost": 0.0, "anchor_algos": 0, "anchor_moves": 0}

        gtc = RC.gt_path_cost(gt, pred, cost_rows, del_costs, args.ins_cost,
                              args.slices, args.c_slice)
        row = {"session": d.name, "n_gt": len(gt), "n_pred": len(pred),
               "raw_acc": raw["acc"], "raw_miss": raw["miss"],
               "raw_sub": raw["sub"], "raw_phantom": raw["phantom"],
               "gt_path_cost": gtc, "n_cands": len(cands),
               "cand_seconds": t_cand,
               "base_solved": base["solved"], "base_cost": base.get("cost"),
               "algo_solved": algo["solved"], "algo_cost": algo.get("cost"),
               "base_exact": bool(base["solved"]
                                  and base["moves"] == gt),
               "algo_exact": bool(algo["solved"]
                                  and algo["moves"] == gt),
               "algo_offered": algo.get("algo_offered", 0),
               "algo_gate_onsets": len(algo.get("algo_gate_onsets", [])),
               # Post-decode accuracy against ground truth. This is the
               # graded metric: verification is a cliff and moves rarely,
               # but "how close did the decoded word get to the truth" moves
               # continuously and is what a move-recording product is
               # actually judged on. Scored on the MAP story whether or not
               # it verified — a failed decode can still be nearly right.
               **{f"acc_{a}": post_acc(gt, r)
                  for a, r in (("base", base), ("algo", algo),
                               ("bwd", bwd))},
               # The trusted-anchor arm shares every decode with `bwd` and
               # differs only in which best-effort story is reported when
               # nothing verified — so this column isolates the selection
               # rule of ALGORITHM_PRIOR §9b and nothing else.
               "acc_trust": post_acc(gt, bwd, "best_effort_moves_trusted"),
               "trusted_fired": bool(bwd.get("trusted_fired")),
               "trusted_onset": bwd.get("trusted_onset"),
               "trusted_algos": bwd.get("trusted_algos", 0),
               "bwd_solved": bwd["solved"], "bwd_cost": bwd.get("cost"),
               "bwd_exact": bool(bwd["solved"] and bwd["moves"] == gt),
               "bwd_anchor_onset": bwd.get("anchor_onset"),
               "bwd_anchor_algos": bwd.get("anchor_algos", 0),
               "bwd_anchor_moves": bwd.get("anchor_moves", 0),
               **{f"fired_{k}": v
                  for k, v in fired_report(algo, gt, pred).items()}}

        # Falsifiability, under the SAME cost model as the true claim.
        if not args.no_decoys:
            rng = np.random.default_rng(abs(hash(d.name)) % (2 ** 31))
            for name, s in decoy_starts(start, rng):
                for arm, c in (("base", None), ("algo", cands)):
                    r = decode_arm(s, cost_rows, del_costs, args, tables, c)
                    row[f"decoy_{arm}_{name}"] = bool(r["solved"])

        rows.append(row)
        flag = ""
        if row["algo_solved"] and not row["base_solved"]:
            flag = "  <- NEWLY VERIFIED"
        elif row["base_solved"] and not row["algo_solved"]:
            flag = "  <- REGRESSION"
        if row["bwd_solved"] and not row["base_solved"]:
            flag = "  <- BACKWARD VERIFIES, base does not"
        print(f"  {d.name:<32} acc {100*raw['acc']:5.1f}%  "
              f"gtc {gtc:6.1f}  cands {len(cands):>4}  "
              f"base {'Y' if row['base_solved'] else 'n'} "
              f"algo {'Y' if row['algo_solved'] else 'n'} "
              f"bwd {'Y' if row['bwd_solved'] else 'n'}  "
              f"anchor@{row['bwd_anchor_onset']} "
              f"({row['bwd_anchor_algos']} algs, "
              f"{row['bwd_anchor_moves']} mv){flag}")

    report(rows, args)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1, default=str)
        print(f"\n  wrote {args.out}")


def report(rows, args):
    if not rows:
        return
    n = len(rows)
    print(f"\n{'='*70}\n  DECODE  ({n} sessions)\n{'='*70}")
    for k in ("solved", "exact"):
        b = sum(r[f"base_{k}"] for r in rows)
        a = sum(r[f"algo_{k}"] for r in rows)
        w = sum(r[f"bwd_{k}"] for r in rows)
        print(f"  {k.upper():<8} base {b}/{n}    algo(fwd) {a}/{n} "
              f"({a-b:+d})    backward {w}/{n} ({w-b:+d})")
    for arm in ("algo", "bwd"):
        gained = [r["session"] for r in rows
                  if r[f"{arm}_solved"] and not r["base_solved"]]
        lost = [r["session"] for r in rows
                if r["base_solved"] and not r[f"{arm}_solved"]]
        label = "forward" if arm == "algo" else "backward"
        print(f"  {label:<9} newly verified: "
              f"{', '.join(gained) if gained else 'none'}")
        print(f"  {label:<9} regressions   : "
              f"{', '.join(lost) if lost else 'none'}")
    # The graded metric. Ship bar is stated on this, not on verification.
    ARMS = (("base", "baseline"), ("algo", "forward"), ("bwd", "backward"),
            ("trust", "trusted"))
    print(f"\n{'='*70}\n  POST-DECODE ACCURACY vs GROUND TRUTH\n{'='*70}")
    print(f"  {'session':<32} {'raw':>7} {'base':>7} {'fwd':>7} {'bwd':>7}"
          f" {'trust':>7}  {'best-raw':>9}")
    for r in sorted(rows, key=lambda r: -(r.get("acc_bwd") or 0)):
        cells = []
        for a, _ in ARMS:
            v = r.get(f"acc_{a}")
            cells.append(f"{100*v:6.1f}%" if v is not None else "     - ")
        best = max((r.get(f"acc_{a}") or 0) for a, _ in ARMS)
        print(f"  {r['session']:<32} {100*r['raw_acc']:6.1f}% "
              + " ".join(f"{c:>7}" for c in cells)
              + f"  {100*(best - r['raw_acc']):+8.1f}")

    # NEVER quote the pooled mean: it is bimodal and the split is lighting
    # ([[post-decode-accuracy-ship-bar]]). Daytime is the regime the ship
    # bar is stated on; evening is a model problem no decoder change
    # touches, and averaging them hides which one moved.
    def regime(r):
        stamp = r["session"].split("_")[2] if len(
            r["session"].split("_")) > 2 else "000000"
        try:
            return "evening" if int(stamp[:2]) >= 18 else "daytime"
        except ValueError:
            return "daytime"

    for label, sel in (("DAYTIME (<18h)", lambda r: regime(r) == "daytime"),
                       ("EVENING (>=18h)", lambda r: regime(r) == "evening"),
                       ("pooled (do not quote)", lambda r: True)):
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        print(f"\n  {label}  n={len(sub)}")
        rawv = [r["raw_acc"] for r in sub]
        print(f"    {'raw (pre)':<10} mean {100*np.mean(rawv):5.1f}%   "
              f"median {100*np.median(rawv):5.1f}%   "
              f">=95%: {sum(x >= 0.95 for x in rawv)}/{len(rawv)}")
        for a, lab in ARMS:
            v = [r[f"acc_{a}"] for r in sub if r.get(f"acc_{a}") is not None]
            if not v:
                continue
            print(f"    {lab:<10} mean {100*np.mean(v):5.1f}%   "
                  f"median {100*np.median(v):5.1f}%   "
                  f">=95%: {sum(x >= 0.95 for x in v)}/{len(v)}   "
                  f">=97%: {sum(x >= 0.97 for x in v)}/{len(v)}")

    fired = [r for r in rows if r.get("trusted_fired")]
    print(f"\n  trusted-anchor rule fired on {len(fired)}/{n} sessions"
          + (f": {', '.join(r['session'] for r in fired)}" if fired else ""))
    moved = [(r["session"], 100 * ((r.get("acc_trust") or 0)
                                   - (r.get("acc_bwd") or 0)))
             for r in rows
             if abs((r.get("acc_trust") or 0) - (r.get("acc_bwd") or 0)) > 1e-9]
    for s, d in sorted(moved, key=lambda x: -x[1]):
        print(f"      {s:<34} {d:+6.1f} pts vs backward")

    anch = [r for r in rows if r.get("bwd_anchor_algos", 0) > 0]
    print(f"\n  backward peel found >=1 algorithm in {len(anch)}/{n} "
          f"sessions; {sum(r.get('bwd_anchor_moves', 0) for r in rows)} "
          f"moves came from the peel")

    print(f"\n{'='*70}\n  DID THE PRIOR ACTUALLY FILL ALGORITHMS IN\n{'='*70}")
    fired = [r for r in rows if r["fired_n_algo"] > 0]
    print(f"  sessions where OP_ALGO fired in the winning story: "
          f"{len(fired)}/{n}")
    print(f"  OP_ALGO transitions taken     : "
          f"{sum(r['fired_n_algo'] for r in rows)}")
    print(f"  moves emitted by them         : "
          f"{sum(r['fired_algo_moves'] for r in rows)} of "
          f"{sum(r['n_gt'] for r in rows)} ground-truth moves")
    print(f"  raw-decode MISSES inside them : "
          f"{sum(r['fired_recovered'] for r in rows)} of "
          f"{sum(r['raw_miss'] for r in rows)} total misses")
    print(f"  candidate generation          : "
          f"{np.mean([r['cand_seconds'] for r in rows]):.1f}s/session, "
          f"{np.mean([r['n_cands'] for r in rows]):.0f} candidates")
    # The distinction that decides the next move: a transition that was
    # never OFFERED means no hypothesis ever had the first two layers
    # right, which is a prefix problem the prior cannot touch. One offered
    # and not taken would be a pricing problem instead.
    offered = [r for r in rows if r.get("algo_offered", 0) > 0]
    print(f"  sessions where the gate admitted anyone: {len(offered)}/{n}")
    for r in rows:
        if r.get("n_cands", 0) and not r.get("algo_offered", 0):
            print(f"      {r['session']:<32} {r['n_cands']:>4} candidates, "
                  f"gate never admitted a hypothesis")

    if not args.no_decoys:
        print(f"\n{'='*70}\n  FALSIFIABILITY  (accepted decoys — lower is "
              f"better)\n{'='*70}")
        print(f"  {'decoy':<22} {'base':>8} {'algo':>8}")
        names = [nm for nm, _ in DECOYS] + ["never scrambled"]
        for nm in names:
            b = sum(r.get(f"decoy_base_{nm}", False) for r in rows)
            a = sum(r.get(f"decoy_algo_{nm}", False) for r in rows)
            print(f"  {nm:<22} {b:>4}/{n:<3} {a:>4}/{n:<3}")
        print("\n  A prior that buys verifications by also accepting decoys "
              "has bought nothing.")


if __name__ == "__main__":
    main()
