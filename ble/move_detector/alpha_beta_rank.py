"""
alpha_beta_rank.py — does forward evidence rank the backward peel's anchors?

`forced_verdict.py` established the shape of the problem: the peel generates
a median ~10,400 anchors per session, the TRUE one is exact in 17/24, and it
sits at median cost-rank 505. Cost cannot find it, and ALGORITHM_PRIOR §9b
explains why structurally — the peel scores only BACKWARD evidence

    beta(s, j)  = cost of explaining onsets [j, n) from state s to solved

and beta is trivially ~0 for the do-nothing anchor, which has explained
nothing. That is the classic stack-decoder normalisation failure: partial
hypotheses covering different amounts of evidence are not comparable.

The A* fix (f = g + h) is CLOSED here, and it is worth writing down why so
nobody tries it: `onset_costs` returns `log(q.max()) - log(q)` clamped at 0,
so costs are argmax-RELATIVE and explaining any prefix by accepting the
model's own reading costs exactly 0. The tightest admissible h is therefore
0 and A* degenerates to the uniform-cost search we already have.

What is NOT degenerate is the forward mass

    alpha(s, j) = cost of getting from the known scramble start to state s
                  over onsets [0, j)

because reaching a SPECIFIC cube state is a hard algebraic constraint: most
anchors are simply not reachable at cost 0, so the forward pass must pay to
bend the observation toward them. Ranking by alpha+beta (a product of
probabilities, a sum of costs) is posterior decoding rather than Viterbi.

One forward beam gives alpha for EVERY anchor at once, because the beam is
already a lattice over states — that is what `_run_beam`'s `on_onset` hook
is for. Run from move_detector/:

    python alpha_beta_rank.py --tag move_ctc_spd_s0_e54
"""
import argparse
import json
from pathlib import Path

import numpy as np

import algorithm_gate as AG
import reconstruct as RC
from ctc_decode import prefix_beam_decode, ctc_to_moves
from forced_verdict import lev

DATA = Path(__file__).resolve().parent.parent / "training_data"


def forward_alpha(start, cost_rows, del_costs, beam, protect=None):
    """
    alpha[j] = {state_bytes: cost} after onsets [0, j) have been consumed.

    use_bounds=False for the same reason decode_bidirectional uses it: the
    ranking heuristics are all calibrated around a FIXED target of solved,
    and a partial forward beam has no such target, so they would bias it
    toward states that merely look close to solved.

    `protect` (hash keys of the peel's anchor states) makes those states
    immune to beam truncation. That is the mandatory-survivor arm: it
    separates "alpha is undefined because the beam PRUNED the anchor" from
    "because the forward search never generated it at all".
    """
    snaps: dict[int, dict] = {0: {start.tobytes(): 0.0}}

    def hook(j, bm):
        d = {}
        for st, c in zip(bm.states, bm.costs):
            k = st.tobytes()
            c = float(c)
            if k not in d or c < d[k]:
                d[k] = c
        snaps[j + 1] = d

    RC._run_beam(start, cost_rows, del_costs, beam=beam,
                 use_bounds=False, on_onset=hook, protect=protect)
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="move_ctc_spd_s0_e54")
    ap.add_argument("--ckpt", default="checkpoints/move_ctc_spd_s0.pt")
    ap.add_argument("--beam", type=int, default=2000)
    ap.add_argument("--beam-ctc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--protect", action="store_true",
                    help="make every peel anchor immune to beam truncation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sessions = [d for d in sorted(DATA.glob("*_solve"))
                if (d / f"ctc_post_{args.tag}.npz").exists()
                and AG.session_chunks(d)]
    if args.limit:
        sessions = sessions[:args.limit]
    unseen = {d.name: RC.classifier_unseen(d.name, args.ckpt) for d in sessions}
    lib_dirs = [d for d in sessions if not unseen[d.name]]
    library = [w for w, _ in AG.build_library(lib_dirs).most_common()]
    print(f"library: {len(library)} words from {len(lib_dirs)} training "
          f"sessions; forward beam {args.beam}")

    from dataset import JointSessionStream
    rows = []
    for d in sessions:
        chunks = AG.session_chunks(d)
        gt = AG.session_word(d)
        ll = chunks[0]["ll_start"]
        gt_ll = gt[ll:]
        start = AG.paired_start(d)

        cp = np.load(d / f"ctc_post_{args.tag}.npz")["class_prob"]
        stream = JointSessionStream(d / "detector_stream_color.npz")
        lab, fr = prefix_beam_decode(np.log(np.maximum(cp, 1e-12)),
                                     beam=args.beam_ctc)
        moves = ctc_to_moves(cp, lab, fr, fps=stream.fps)
        if not moves:
            continue
        _, cost_rows, del_costs = RC.costs_from_moves(
            moves, 0.5, RC.BLEND_INV, RC.BLEND_UNIF,
            RC.C_DEL, None, RC.DEL_FLOOR, RC.BLEND_ADJ)

        cands = AG.candidates_for_session(moves, cp, library)
        anchors = AG.peel_backward(cost_rows, del_costs, cands)

        protect = None
        if args.protect:
            protect = RC.beam_state_keys(
                np.stack([a["state"] for a in anchors]))
        alpha = forward_alpha(start, cost_rows, del_costs, args.beam, protect)

        scored = []
        for i, a in enumerate(anchors):
            j = a["onset"]
            al = alpha.get(j, {}).get(a["state"].tobytes())
            scored.append({
                "i": i, "a": a, "beta": a["cost"], "alpha": al,
                "lev": lev(AG.anchor_moves(a["ops"]), gt_ll)})

        reach = [s for s in scored if s["alpha"] is not None]
        best_lev = min(s["lev"] for s in scored)
        # Among equally-good anchors, the one the CURRENT rule would find
        # first — so the reported rank is the honest best case for beta.
        oracle = min((s for s in scored if s["lev"] == best_lev),
                     key=lambda s: s["beta"])

        by_beta = sorted(scored, key=lambda s: s["beta"])
        rank_beta = [s["i"] for s in by_beta].index(oracle["i"]) + 1
        top_beta = by_beta[0]

        if reach:
            by_ab = sorted(reach, key=lambda s: s["alpha"] + s["beta"])
            order = [s["i"] for s in by_ab]
            rank_ab = (order.index(oracle["i"]) + 1
                       if oracle["i"] in order else None)
            top_ab = by_ab[0]
        else:
            by_ab, rank_ab, top_ab = [], None, None

        rows.append(dict(
            session=d.name, unseen=unseen[d.name], n_ll=len(gt_ll),
            n_anchors=len(anchors), n_reachable=len(reach),
            oracle_lev=best_lev,
            oracle_reachable=oracle["alpha"] is not None,
            rank_beta=rank_beta, rank_ab=rank_ab,
            lev_top_beta=top_beta["lev"],
            lev_top_ab=(top_ab["lev"] if top_ab else None),
            algos_top_ab=(sum(1 for k, _ in top_ab["a"]["ops"]
                              if k == "algorithm") if top_ab else None)))
        r = rows[-1]
        print(f"{d.name:34} LL={r['n_ll']:3d} anch={r['n_anchors']:6d} "
              f"reach={r['n_reachable']:5d} | oracle lev={best_lev:3d} "
              f"rank beta={rank_beta:6d} -> a+b="
              f"{str(r['rank_ab']):>6} | lev top-beta={r['lev_top_beta']:3d} "
              f"top-a+b={str(r['lev_top_ab']):>3}")

    def block(name, sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return
        tot = sum(r["n_ll"] for r in rs)
        ok = [r for r in rs if r["rank_ab"] is not None]
        print(f"\n=== {name} (n={len(rs)}, {tot} last-layer moves) ===")
        print(f"  anchors per session: median "
              f"{np.median([r['n_anchors'] for r in rs]):.0f}; "
              f"forward-reachable: median "
              f"{np.median([r['n_reachable'] for r in rs]):.0f}")
        print(f"  oracle anchor forward-reachable: "
              f"{sum(r['oracle_reachable'] for r in rs)}/{len(rs)}")
        print(f"  oracle rank by beta   : median "
              f"{np.median([r['rank_beta'] for r in rs]):.0f}")
        if ok:
            print(f"  oracle rank by alpha+beta: median "
                  f"{np.median([r['rank_ab'] for r in ok]):.0f}   (n={len(ok)})")
        lb = sum(r["lev_top_beta"] for r in rs)
        print(f"  committed LL, top by beta   : {lb:4d} edits -> "
              f"{100*(1-lb/tot):5.1f}%   exact "
              f"{sum(r['lev_top_beta']==0 for r in rs)}/{len(rs)}")
        rr = [r for r in rs if r["lev_top_ab"] is not None]
        if rr:
            t2 = sum(r["n_ll"] for r in rr)
            la = sum(r["lev_top_ab"] for r in rr)
            print(f"  committed LL, top by alpha+beta: {la:4d} edits -> "
                  f"{100*(1-la/t2):5.1f}%   exact "
                  f"{sum(r['lev_top_ab']==0 for r in rr)}/{len(rr)}")
        lo = sum(r["oracle_lev"] for r in rs)
        print(f"  ORACLE                      : {lo:4d} edits -> "
              f"{100*(1-lo/tot):5.1f}%   exact "
              f"{sum(r['oracle_lev']==0 for r in rs)}/{len(rs)}")

    block("ALL", lambda r: True)
    block("CLASSIFIER-UNSEEN", lambda r: r["unseen"])

    out = args.out or f"results/alpha_beta_{args.tag}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
