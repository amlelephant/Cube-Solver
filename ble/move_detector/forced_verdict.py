"""
forced_verdict.py — what does the last layer look like when we REFUSE to abstain?

The backward peel currently only commits to an anchor whose prefix decode
verifies, and falls back to the do-nothing anchor otherwise
(ALGORITHM_PRIOR §9a cause 1). This asks the opposite question, which is
the one worth answering before any more selection tuning:

    the solve ENDS in an algorithm, by construction. So force the chain:
    take the deepest anchor the peel can build, whatever it costs, and
    measure how far the forced last-layer word is from the truth.

If forcing lands close, the peel's problem is selection and worth fixing.
If forcing lands far even with no budget constraint, the last layer is not
being read well enough for ANY selection rule, and the lever is the model.

Reads cached posteriorgrams only — no model load. Run from move_detector/:

    python forced_verdict.py --tag move_ctc_spd_s0_e54
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import algorithm_gate as AG
import reconstruct as RC
from ctc_decode import prefix_beam_decode, ctc_to_moves

DATA = Path(__file__).resolve().parent.parent / "training_data"


def lev(a, b):
    """Levenshtein distance between two move words."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def align_index(gt, pred, k):
    """Index in `pred` that aligns to gt[k], via Levenshtein backtrace."""
    N, M = len(gt), len(pred)
    D = np.zeros((N + 1, M + 1), dtype=np.int32)
    D[:, 0] = np.arange(N + 1)
    D[0, :] = np.arange(M + 1)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            D[i, j] = min(D[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]),
                          D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j, out = N, M, M
    while i > 0 or j > 0:
        if i == k:
            out = j
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]):
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def forced_anchor(anchors):
    """
    The verdict we are forced to give: the anchor that peels the MOST
    algorithms, cheapest among those. No cost threshold, no abstain — the
    point of the experiment is that budget does not get a vote.
    """
    def n_algo(a):
        return sum(1 for k, _ in a["ops"] if k == "algorithm")
    return max(anchors, key=lambda a: (n_algo(a), -a["cost"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="move_ctc_spd_s0_e54")
    ap.add_argument("--ckpt", default="checkpoints/move_ctc_spd_s0.pt")
    ap.add_argument("--beam-ctc", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sessions = [d for d in sorted(DATA.glob("*_solve"))
                if (d / f"ctc_post_{args.tag}.npz").exists()
                and AG.session_chunks(d)]
    unseen = {d.name: RC.classifier_unseen(d.name, args.ckpt) for d in sessions}

    # Library mined from TRAINING sessions only — never the held-out ones.
    lib_dirs = [d for d in sessions if not unseen[d.name]]
    library = [w for w, _ in AG.build_library(lib_dirs).most_common()]
    print(f"library: {len(library)} words from {len(lib_dirs)} training sessions")

    from dataset import JointSessionStream
    rows = []
    for d in sessions:
        chunks = AG.session_chunks(d)
        gt = AG.session_word(d)
        ll = chunks[0]["ll_start"]
        gt_ll = gt[ll:]

        cp = np.load(d / f"ctc_post_{args.tag}.npz")["class_prob"]
        stream = JointSessionStream(d / "detector_stream_color.npz")
        lp = np.log(np.maximum(cp, 1e-12))
        labels, frames = prefix_beam_decode(lp, beam=args.beam_ctc)
        moves = ctc_to_moves(cp, labels, frames, fps=stream.fps)
        if not moves:
            continue
        pred = [m["move"] for m in moves]

        _, cost_rows, del_costs = RC.costs_from_moves(
            moves, 0.5, RC.BLEND_INV, RC.BLEND_UNIF,
            RC.C_DEL, None, RC.DEL_FLOOR, RC.BLEND_ADJ)

        cands = AG.candidates_for_session(moves, cp, library)
        anchors = AG.peel_backward(cost_rows, del_costs, cands)
        a = forced_anchor(anchors)
        forced_ll = AG.anchor_moves(a["ops"])
        n_algo = sum(1 for k, _ in a["ops"] if k == "algorithm")

        # Alternative forcing rules, all committing (never abstaining), plus
        # the ORACLE: the best anchor in the set by hindsight. The oracle is
        # the number that decides everything — if the true last layer is not
        # in the anchor set at all, no selection rule can find it.
        def nalg(x):
            return sum(1 for k, _ in x["ops"] if k == "algorithm")
        withalg = [x for x in anchors if nalg(x) > 0] or anchors
        cheapest = min(withalg, key=lambda x: x["cost"])
        scored = [(lev(AG.anchor_moves(x["ops"]), gt_ll), x) for x in anchors]
        best_lev, best_a = min(scored, key=lambda t: t[0])
        rank_oracle = sorted(x["cost"] for x in anchors).index(best_a["cost"]) + 1

        j = align_index(gt, pred, ll)
        raw_ll = pred[j:]

        # did the forced anchor land on the true F2L-boundary state?
        states = AG.trajectory(AG.paired_start(d), gt)
        true_state = states[ll]
        hit = bool((a["state"] == true_state).all())

        rows.append(dict(
            session=d.name, unseen=unseen[d.name], n=len(gt), ll=ll,
            n_gt_ll=len(gt_ll), n_cands=len(cands), n_anchors=len(anchors),
            n_algo=n_algo, anchor_onset=a["onset"], anchor_hit=hit,
            lev_raw=lev(raw_ll, gt_ll), lev_forced=lev(forced_ll, gt_ll),
            lev_cheap=lev(AG.anchor_moves(cheapest["ops"]), gt_ll),
            lev_oracle=best_lev, oracle_rank=rank_oracle,
            oracle_algos=nalg(best_a), oracle_onset=best_a["onset"],
            n_raw_ll=len(raw_ll), n_forced_ll=len(forced_ll),
            gt_ll=" ".join(gt_ll), raw_ll=" ".join(raw_ll),
            forced_ll=" ".join(forced_ll)))
        r = rows[-1]
        print(f"{d.name:34} {'UNSEEN' if r['unseen'] else 'train':>6} "
              f"LL={r['n_gt_ll']:3d} cands={r['n_cands']:4d} "
              f"algo={n_algo} anchor@{a['onset']:3d} hit={'Y' if hit else 'n'} "
              f"lev raw={r['lev_raw']:3d} forced={r['lev_forced']:3d}")

    def block(name, sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return
        tot = sum(r["n_gt_ll"] for r in rs)
        lr = sum(r["lev_raw"] for r in rs)
        lf = sum(r["lev_forced"] for r in rs)
        print(f"\n=== {name} (n={len(rs)}, {tot} last-layer moves) ===")
        print(f"  raw    LL edit distance {lr:4d}  -> accuracy {100*(1-lr/tot):5.1f}%"
              f"   exact {sum(r['lev_raw']==0 for r in rs)}/{len(rs)}")
        print(f"  FORCED LL edit distance {lf:4d}  -> accuracy {100*(1-lf/tot):5.1f}%"
              f"   exact {sum(r['lev_forced']==0 for r in rs)}/{len(rs)}")
        for key, name in (("lev_cheap", "cheapest w/ algo"), ("lev_oracle", "ORACLE best")):
            v = sum(r[key] for r in rs)
            print(f"  {name:16} {v:4d}  -> accuracy {100*(1-v/tot):5.1f}%"
                  f"   exact {sum(r[key]==0 for r in rs)}/{len(rs)}")
        print(f"  oracle anchor's rank by cost: median "
              f"{np.median([r['oracle_rank'] for r in rs]):.0f}  "
              f"max {max(r['oracle_rank'] for r in rs)}")
        print(f"  forced anchor == true F2L-boundary state: "
              f"{sum(r['anchor_hit'] for r in rs)}/{len(rs)}")
        print(f"  forced peel took >=1 algorithm: "
              f"{sum(r['n_algo'] > 0 for r in rs)}/{len(rs)}"
              f"   median algos {np.median([r['n_algo'] for r in rs]):.0f}")
        better = sum(r["lev_forced"] < r["lev_raw"] for r in rs)
        worse = sum(r["lev_forced"] > r["lev_raw"] for r in rs)
        print(f"  forced better / worse / same: {better} / {worse} / "
              f"{len(rs)-better-worse}")

    block("ALL", lambda r: True)
    block("CLASSIFIER-UNSEEN", lambda r: r["unseen"])

    out = args.out or f"results/forced_verdict_{args.tag}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
