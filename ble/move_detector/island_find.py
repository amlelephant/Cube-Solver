"""
island_find.py — can the FORWARD DECODE find the milestones by itself?

`islands.py` measured what milestone anchors would be worth if you had
them: cross + 4 F2L slots, 5 per solve, ~13-move segments, and a
verification ceiling of 13/24 against today's 2/24. Every one of those
numbers used milestones replayed from BLE ground truth, so they bound the
idea rather than test it.

This is the deployability half. It runs the ordinary forward beam from the
known scramble state and asks, at every onset, whether any live hypothesis
satisfies each milestone predicate — and whether the one that does is the
TRUE state the solve passed through.

Three numbers decide it:

  generated   is the true milestone state ever in the beam at all? This is
              the question that killed the backward peel
              ([[alpha-beta-frontiers-dont-meet]]): on failing sessions the
              forward search never generated the anchor. A milestone sits
              ~13 moves deep instead of ~60, so it should survive where
              that did not — but "should" is the whole thing under test.
  pickable    at the first onset where SOME hypothesis claims the
              milestone, is the cheapest such hypothesis the true state?
              That is the deployable rule, with no hindsight in it.
  timing      how far that first claim sits from the true milestone onset.
              An island pinned in the wrong place is worse than none.

Reads cached posteriorgrams only, no model load. From move_detector/:

    python island_find.py --tag move_ctc_spd_s0_e54
"""
import argparse
import json
from pathlib import Path

import numpy as np

import algorithm_gate as AG
import reconstruct as RC
import error_phase as EP
from islands import slot_sets, level, settle, IN_FLIGHT
from forced_verdict import align_index
from ctc_decode import prefix_beam_decode, ctc_to_moves

DATA = Path(__file__).resolve().parent.parent / "training_data"
BT = {f: RC.seq_to_state([f]) for f in AG.FACES}


def levels_batch(states: np.ndarray, cross, pairs, bottom) -> np.ndarray:
    """
    `islands.level` for a whole beam at once — (N,) ladder level.

    The per-state version is 4 composes and a Python loop each; at 4000
    hypotheses x ~120 onsets that is 2M calls, so this has to be batched
    or the test does not finish.
    """
    n = len(states)
    best = np.zeros(n, dtype=np.int16)
    cross = np.asarray(cross)
    t = states
    for _ in range(4):
        ep, eo = t[:, RC._EP], t[:, RC._EO]
        cp, co = t[:, RC._CP], t[:, RC._CO]
        ok = ((ep[:, cross] == cross[None, :]).all(1)
              & (eo[:, cross] == 0).all(1))
        k = np.zeros(n, dtype=np.int16)
        for c, e in pairs:
            k += ((cp[:, c] == c) & (co[:, c] == 0)
                  & (ep[:, e] == e) & (eo[:, e] == 0)).astype(np.int16)
        best = np.maximum(best, np.where(ok, 1 + k, 0))
        t = RC.apply_batch(t, BT[bottom])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="move_ctc_spd_s0_e54")
    ap.add_argument("--ckpt", default="checkpoints/move_ctc_spd_s0.pt")
    ap.add_argument("--beam", type=int, default=2000)
    ap.add_argument("--beam-ctc", type=int, default=8)
    ap.add_argument("--tol", type=int, default=3,
                    help="onsets of slack allowed on a milestone's position")
    args = ap.parse_args()

    from dataset import JointSessionStream
    rows = []
    for d in sorted(DATA.glob("*_solve")):
        if not (d / f"ctc_post_{args.tag}.npz").exists():
            continue
        chunks = AG.session_chunks(d)
        if not chunks:
            continue
        gt = AG.session_word(d)
        start = AG.paired_start(d)
        states = AG.trajectory(start, gt)
        top, ll = chunks[0]["top"], chunks[0]["ll_start"]
        bottom = AG.OPP[top]
        cross, pairs = slot_sets(top)

        lv = [level(s, cross, pairs, bottom) for s in states]
        marks = []
        for m in range(1, 2 + len(pairs)):
            i = settle(lv, ll, m)
            if i is not None and (not marks or i > marks[-1][1]):
                marks.append((m, i))
        if not marks:
            continue

        cp = np.load(d / f"ctc_post_{args.tag}.npz")["class_prob"]
        stream = JointSessionStream(d / "detector_stream_color.npz")
        lab, fr = prefix_beam_decode(np.log(np.maximum(cp, 1e-12)),
                                     beam=args.beam_ctc)
        moves = ctc_to_moves(cp, lab, fr, fps=stream.fps)
        pred = [m["move"] for m in moves]
        if not pred:
            continue
        _, cost_rows, del_costs = RC.costs_from_moves(
            moves, 0.5, RC.BLEND_INV, RC.BLEND_UNIF,
            RC.C_DEL, None, RC.DEL_FLOOR, RC.BLEND_ADJ)

        # true milestone states, and where they sit in ONSET space
        want = []
        for m, i in marks:
            want.append({"m": m, "gt_idx": i, "state": states[i],
                         "onset": align_index(gt, pred, i)})

        # one forward pass; per onset, record each milestone's status
        seen = {w["m"]: {"gen": False, "gen_onset": None,
                         "first": None, "first_true": None} for w in want}

        def hook(j, bm):
            lvs = levels_batch(bm.states, cross, pairs, bottom)
            for w in want:
                s, m = seen[w["m"]], w["m"]
                hit = np.where((bm.states == w["state"][None, :]).all(1))[0]
                if len(hit) and not s["gen"]:
                    s["gen"], s["gen_onset"] = True, j + 1
                claim = np.where(lvs >= m)[0]
                if len(claim) and s["first"] is None:
                    cheapest = claim[np.argmin(bm.costs[claim])]
                    s["first"] = j + 1
                    s["first_true"] = bool(
                        (bm.states[cheapest] == w["state"]).all())

        RC._run_beam(start, cost_rows, del_costs, beam=args.beam,
                     use_bounds=False, on_onset=hook)

        ops, dist = EP.align(gt, pred)
        rec = dict(session=d.name, dist=dist, n=len(gt),
                   unseen=RC.classifier_unseen(d.name, args.ckpt),
                   milestones=[])
        for w in want:
            s = seen[w["m"]]
            off = None if s["first"] is None else s["first"] - w["onset"]
            rec["milestones"].append(dict(
                m=w["m"], true_onset=w["onset"], gen=s["gen"],
                first=s["first"], first_true=s["first_true"], off=off,
                in_tol=(off is not None and abs(off) <= args.tol)))
        rows.append(rec)
        g = sum(x["gen"] for x in rec["milestones"])
        t = sum(bool(x["first_true"]) for x in rec["milestones"])
        print(f"{d.name:34} err={dist:3d} milestones={len(want)} "
              f"generated={g} pickable={t} "
              f"offsets={[x['off'] for x in rec['milestones']]}")

    def block(name, sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return
        ms = [x for r in rs for x in r["milestones"]]
        print(f"\n=== {name} (n={len(rs)} sessions, {len(ms)} milestones) ===")
        g = sum(x["gen"] for x in ms)
        t = sum(bool(x["first_true"]) for x in ms)
        c = sum(x["first"] is not None for x in ms)
        tol = sum(x["in_tol"] for x in ms)
        print(f"  true milestone state GENERATED in beam: {g}/{len(ms)} "
              f"= {100*g/len(ms):.1f}%")
        print(f"  some hypothesis CLAIMS the milestone   : {c}/{len(ms)} "
              f"= {100*c/len(ms):.1f}%")
        print(f"  cheapest claimer IS the true state     : {t}/{len(ms)} "
              f"= {100*t/len(ms):.1f}%   <- the deployable rule")
        print(f"  claim within +/-{3} onsets of the truth  : {tol}/{len(ms)} "
              f"= {100*tol/len(ms):.1f}%")
        offs = [x["off"] for x in ms if x["off"] is not None]
        if offs:
            print(f"  claim offset (onsets): median {np.median(offs):+.0f}, "
                  f"IQR {np.percentile(offs,25):+.0f}..{np.percentile(offs,75):+.0f}")
        allp = sum(1 for r in rs
                   if all(bool(x["first_true"]) for x in r["milestones"]))
        print(f"  sessions where EVERY milestone is pickable: {allp}/{len(rs)}")

    block("ALL", lambda r: True)
    block("CLEAN (<7 edits)", lambda r: r["dist"] < 7)
    block("BAD (>=7 edits)", lambda r: r["dist"] >= 7)

    out = "results/2026-08-09/island_find.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
