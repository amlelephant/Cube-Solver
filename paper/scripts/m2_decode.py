"""
m2_decode.py — the group-theoretic decode, and the falsifiability sweep
that is the only thing that makes a VERIFIED verdict mean anything.

For every held-out free solve, both seeds:

  1. Build the start state the truth implies (inverse of the truth word's
     product) — this stands in for a scanned scramble.
  2. Beam-decode the CTC posteriorgram's move list against start -> solved
     under the noisy-channel cost model (reconstruct.decode).
  3. Score the resulting word against ground truth: post-decode accuracy,
     whether it verified (reached solved), whether it is exactly right.
  4. Re-decode the IDENTICAL onsets against deliberately FALSE claims:
     a start state 1, 2 or 4 quarter turns off the truth, and a cube that
     was never scrambled at all. A verifier that accepts these is worthless,
     and one that has only ever been run on true claims has not been tested.

Everything runs back to back in one sitting; no baseline is recalled from
a document.

    python m2_decode.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import common as C

CKPTS = ["move_ctc_spd_s0.pt", "move_ctc_spd_s1.pt"]
POST = C.DATA / "post"

DECOYS = [("off1", 1), ("off2", 2), ("off4", 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--retry-beam", type=int, default=None)
    ap.add_argument("--no-decoys", action="store_true")
    ap.add_argument("--retry", action="store_true",
                    help="also retry unsolved decodes at 4x beam "
                         "(production behaviour; ~3x the compute)")
    ap.add_argument("--only", default=None,
                    help="score only this checkpoint stem (parallel runs)")
    ap.add_argument("--out", default="m2_decode.json")
    args = ap.parse_args()

    import reconstruct as RC
    from ctc_decode import prefix_beam_decode, ctc_to_moves

    beam = args.beam or RC.BEAM
    retry = args.retry_beam or 4 * RC.BEAM

    # The holdout is always derived from BOTH checkpoints, even when only one
    # is scored — otherwise a parallel per-seed run would silently widen it.
    paths = [C.MD / "checkpoints" / c for c in CKPTS]
    dirs = C.holdout(paths, kind="solve")
    tables = RC.build_tables()
    if args.only:
        paths = [p for p in paths if p.stem == args.only]

    print(f"\n  {len(dirs)} held-out solves x {len(CKPTS)} seeds; "
          f"beam {beam} (retry {retry})")
    print(f"  cost model: C_DEL={RC.C_DEL} C_INS={RC.C_INS} "
          f"C_SLICE={RC.C_SLICE} REL_WEIGHT={RC.REL_WEIGHT} "
          f"MAX_END_INS={RC.MAX_END_INS}\n")

    rows = []
    for cp in paths:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        tag = cp.stem
        print(f"{'='*94}\n  {tag}\n{'='*94}")
        print(f"  {'session':<34}{'raw':>7}{'post':>7}{'ver':>5}{'exact':>7}"
              f"{'cost':>8}{'gtcost':>8}{'decoys accepted':>18}")
        for d in dirs:
            cache = POST / f"{tag}__{d.name}.npz"
            if not cache.exists():
                print(f"  {d.name}: no cached posteriorgram — run m1 first")
                continue
            class_prob = np.load(cache)["class_prob"]
            fps = float(np.load(d / "detector_stream_color.npz",
                                allow_pickle=True)["fps"])
            lab, fr = prefix_beam_decode(
                np.log(np.maximum(class_prob, 1e-12)), beam=16)
            moves = ctc_to_moves(class_prob, lab, fr, fps=fps)
            if not moves:
                continue

            # The decoder's cube model is centre-relative, so its truth and
            # its start state are both CUBE-frame. The model emits
            # camera-frame. They coincide except on the 7 corpus sessions
            # containing a middle slice — flagged, not silently pooled.
            gt_cube = C.cube_word(d)
            gt_cam = C.truth_word(d)
            slice_session = gt_cube != gt_cam

            pred, cost_rows, del_costs = RC.costs_from_moves(
                moves, ck.get("threshold", 0.5), RC.BLEND_INV, RC.BLEND_UNIF,
                RC.C_DEL, None, RC.DEL_FLOOR, RC.BLEND_ADJ)
            start = RC.start_from_gt(gt_cube)
            raw = RC.score_vs_gt(gt_cube, pred)

            kw = dict(c_del=RC.C_DEL, c_ins=RC.C_INS,
                      max_end_ins=RC.MAX_END_INS, rel_weight=RC.REL_WEIGHT,
                      del_costs=del_costs, rotations=False, tables=tables,
                      slices=True, c_slice=RC.C_SLICE)
            t0 = time.time()
            res = RC.decode(start, cost_rows, beam=beam, **kw)
            # Production retries once at 4x beam. Omitted by default here:
            # every held-out session's true story costs far more than the
            # ~10 envelope the beam can carry (gt_path_cost is reported per
            # session so this is checkable, not assumed), so the retry
            # cannot change a verdict and it triples the sweep's cost.
            retried = False
            if args.retry and not res["solved"]:
                res = RC.decode(start, cost_rows, beam=retry, **kw)
                retried = True
            secs = time.time() - t0

            post = RC.score_vs_gt(gt_cube, res.get("best_effort_moves")
                                  or res.get("moves") or [])
            gtc = RC.gt_path_cost(gt_cube, pred, cost_rows, del_costs,
                                  RC.C_INS, True, RC.C_SLICE)

            row = {"model": tag, "seed": ck["seed"], "session": d.name,
                   "slice_session": slice_session,
                   "n_gt": len(gt_cube), "n_pred": len(pred),
                   "raw_acc": raw["acc"], "raw_miss": raw["miss"],
                   "raw_sub": raw["sub"], "raw_phantom": raw["phantom"],
                   "post_acc": post["acc"], "post_exact": bool(post["exact"]),
                   "verified": bool(res["solved"]),
                   "cost": res.get("cost"), "gt_path_cost": gtc,
                   "retried": retried, "seconds": secs}

            accepted = []
            if not args.no_decoys:
                rng = np.random.default_rng(abs(hash(d.name)) % (2 ** 31))
                for name, dist in DECOYS:
                    s = start.copy()
                    for _ in range(dist):
                        s = RC.compose(s, RC.CLASS_VECS[int(rng.integers(12))])
                    r = RC.decode(s, cost_rows, beam=beam, **kw)
                    row[f"decoy_{name}"] = bool(r["solved"])
                    row[f"decoy_{name}_cost"] = r.get("cost")
                    if r["solved"]:
                        accepted.append(name)
                r = RC.decode(RC.SOLVED.copy(), cost_rows, beam=beam, **kw)
                row["decoy_unscrambled"] = bool(r["solved"])
                row["decoy_unscrambled_cost"] = r.get("cost")
                if r["solved"]:
                    accepted.append("unscrambled")

            rows.append(row)
            print(f"  {d.name:<34}{raw['acc']*100:>6.1f}%{post['acc']*100:>6.1f}%"
                  f"{'Y' if res['solved'] else '.':>5}"
                  f"{'Y' if post['exact'] else '.':>7}"
                  f"{(res.get('cost') or float('nan')):>8.2f}{gtc:>8.2f}"
                  f"{(','.join(accepted) or 'none'):>18}"
                  f"{'   [slice]' if slice_session else ''}")

    C.dump(args.out, rows)

    print(f"\n{'='*94}\n  SUMMARY\n{'='*94}")
    summ = []
    for tag in [Path(c).stem for c in CKPTS]:
        for regime in ("daytime", "evening", "all"):
            meta = {m["session"]: m for m in C.load("holdout_meta.json")}
            g = [r for r in rows if r["model"] == tag
                 and (regime == "all"
                      or (regime == "evening") == bool(meta[r["session"]]["evening"]))]
            if not g:
                continue
            s = {"model": tag, "regime": regime, "n": len(g),
                 "raw_mean": float(np.mean([r["raw_acc"] for r in g])),
                 "post_mean": float(np.mean([r["post_acc"] for r in g])),
                 "post_min": float(np.min([r["post_acc"] for r in g])),
                 "verified": int(sum(r["verified"] for r in g)),
                 "exact": int(sum(r["post_exact"] for r in g)),
                 "decoys_accepted": int(sum(
                     sum(bool(r.get(f"decoy_{k}")) for k in
                         ("off1", "off2", "off4", "unscrambled")) for r in g)),
                 "decoys_tried": 4 * len(g)}
            summ.append(s)
            print(f"  {tag:<20}{regime:<10}n={s['n']:<3} raw {s['raw_mean']*100:5.1f}%"
                  f"  post {s['post_mean']*100:5.1f}%  verified {s['verified']}/{s['n']}"
                  f"  exact {s['exact']}/{s['n']}"
                  f"  false accepts {s['decoys_accepted']}/{s['decoys_tried']}")
    C.dump(args.out.replace("decode","summary"), summ)


if __name__ == "__main__":
    main()
