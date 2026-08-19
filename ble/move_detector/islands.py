"""
islands.py — are there interior structural anchors, and do they cut the
error up into segments the decoder can actually repair?

The backward peel anchors at ONE interior point (the F2L boundary) and can
only help the last layer. `results/2026-08-09/alpha_beta_*.json` showed the
forward and backward searches never meet on a failing session, and that
pre-last-layer errors veto verification on 16/24 sessions regardless.

This asks the follow-up: a layer-by-layer solve passes through a LADDER of
structural milestones — cross done, then each F2L pair slotted, then the
last layer. Each is a predicate on the cube state, so it is checkable
without knowing whether the prefix was decoded correctly, which is what
makes it usable as an island (the same argument that made the last-layer
abstract gate work: a wrong prefix lands on a state that does not satisfy
it).

Three things decide whether island-driven search is worth building:

  ladder    how many milestones a real solve actually passes through, and
            how far apart they are
  repair    once the solve is cut at those milestones, how many segments
            contain 0 or 1 errors -- the decoder's measured envelope is
            about ONE insertion-unit of cost, so a 0-1-error segment with
            both ends pinned is inside it and a 5-error segment is not
  null      how often a WRONG state satisfies a milestone predicate. An
            island that fires on garbage is worse than no island.

Reads cached posteriorgrams only, no model load. From move_detector/:

    python islands.py --tag move_ctc_spd_s0_e54
"""
import argparse
import json
from pathlib import Path

import numpy as np

import algorithm_gate as AG
import reconstruct as RC
import error_phase as EP
from ctc_decode import prefix_beam_decode, ctc_to_moves

DATA = Path(__file__).resolve().parent.parent / "training_data"
BT = {f: RC.seq_to_state([f]) for f in AG.FACES}
IN_FLIGHT = 8          # an F2L insertion pulls a piece out and puts it back


def slot_sets(top: str):
    """(cross edges, [(corner, edge)] for the 4 F2L slots) for a top face."""
    bottom = AG.OPP[top]
    cross = [i for i, f in enumerate(AG.EDGE_FACES) if bottom in f]
    corners = [i for i, f in enumerate(AG.CORNER_FACES) if bottom in f]
    mids = [i for i, f in enumerate(AG.EDGE_FACES)
            if top not in f and bottom not in f]
    # Pair each bottom corner with the middle edge sharing both its side
    # faces — that is the F2L slot they occupy together.
    pairs = []
    for c in corners:
        cf = set(AG.CORNER_FACES[c]) - {bottom}
        for e in mids:
            if set(AG.EDGE_FACES[e]) == cf:
                pairs.append((c, e))
                break
    return cross, pairs


def level(state, cross, pairs, bottom):
    """
    How far up the ladder this state is: 0 = cross not done, 1 = cross
    done, 1+k = cross plus k slots. Maximised over a turn of the bottom
    face, since AUF on the bottom does not undo anything.
    """
    best = 0
    t = state
    for _ in range(4):
        cp, co, ep, eo = t[RC._CP], t[RC._CO], t[RC._EP], t[RC._EO]
        if not all(ep[i] == i and eo[i] == 0 for i in cross):
            t = RC.compose(t, BT[bottom])
            continue
        k = sum(1 for c, e in pairs
                if cp[c] == c and co[c] == 0 and ep[e] == e and eo[e] == 0)
        best = max(best, 1 + k)
        t = RC.compose(t, BT[bottom])
    return best


def settle(levels, upto, m, w=IN_FLIGHT):
    """First index after which `levels` stays >= m, tolerating short dips."""
    for i in range(upto + 1):
        if levels[i] < m:
            continue
        run = worst = 0
        for v in levels[i:upto + 1]:
            run = 0 if v >= m else run + 1
            worst = max(worst, run)
        if worst <= w:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="move_ctc_spd_s0_e54")
    ap.add_argument("--ckpt", default="checkpoints/move_ctc_spd_s0.pt")
    ap.add_argument("--beam-ctc", type=int, default=8)
    ap.add_argument("--null-trials", type=int, default=200)
    args = ap.parse_args()

    from dataset import JointSessionStream
    rows, null_hit, null_n = [], 0, 0
    rng = np.random.default_rng(0)

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

        levels = [level(s, cross, pairs, bottom) for s in states]
        marks = []
        for m in range(1, 2 + len(pairs)):
            i = settle(levels, ll, m)
            if i is not None and (not marks or i > marks[-1][1]):
                marks.append((m, i))
        bounds = [0] + [i for _, i in marks] + [len(gt)]
        bounds = sorted(set(bounds))

        cp = np.load(d / f"ctc_post_{args.tag}.npz")["class_prob"]
        stream = JointSessionStream(d / "detector_stream_color.npz")
        lab, fr = prefix_beam_decode(np.log(np.maximum(cp, 1e-12)),
                                     beam=args.beam_ctc)
        moves = ctc_to_moves(cp, lab, fr, fps=stream.fps)
        pred = [m["move"] for m in moves]
        if not pred:
            continue
        ops, dist = EP.align(gt, pred)
        errs = [i for _, i in ops]

        segs = []
        for a, b in zip(bounds, bounds[1:]):
            segs.append({"lo": a, "hi": b, "len": b - a,
                         "err": sum(1 for i in errs if a <= i < b)})

        # NULL: does a state reached by a WRONG prefix satisfy a milestone?
        # Perturb the truth prefix by one substitution and re-check.
        for _ in range(args.null_trials // 20):
            k = int(rng.integers(10, max(11, ll)))
            w = list(gt[:k])
            j = int(rng.integers(0, k))
            w[j] = RC.WCA12[int(rng.integers(0, 12))]
            if w[j] == gt[j]:
                continue
            s = RC.compose(start, RC.seq_to_state(w))
            null_n += 1
            null_hit += level(s, cross, pairs, bottom) >= 1

        rows.append(dict(session=d.name, n=len(gt), ll=ll, dist=dist,
                         marks=marks, segs=segs,
                         unseen=RC.classifier_unseen(d.name, args.ckpt)))
        r = rows[-1]
        print(f"{d.name:34} n={r['n']:3d} err={dist:3d} "
              f"milestones={len(marks)} at {[i for _, i in marks]}")

    def block(name, sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return
        segs = [s for r in rs for s in r["segs"]]
        print(f"\n=== {name} (n={len(rs)} sessions, {len(segs)} segments) ===")
        print(f"  milestones per solve : median "
              f"{np.median([len(r['marks']) for r in rs]):.0f}  "
              f"range {min(len(r['marks']) for r in rs)}-"
              f"{max(len(r['marks']) for r in rs)}")
        print(f"  segment length (moves): median "
              f"{np.median([s['len'] for s in segs]):.0f}  "
              f"max {max(s['len'] for s in segs)}")
        z = sum(1 for s in segs if s["err"] == 0)
        o = sum(1 for s in segs if s["err"] <= 1)
        print(f"  segments with 0 errors: {z}/{len(segs)} = {100*z/len(segs):.1f}%")
        print(f"  segments with <=1 err : {o}/{len(segs)} = {100*o/len(segs):.1f}%")
        bad = [s for s in segs if s["err"] >= 2]
        print(f"  segments with >=2 err : {len(bad)}"
              f"  (worst {max((s['err'] for s in bad), default=0)} errors)")
        # A session is repairable only if EVERY segment is inside envelope.
        allok = sum(1 for r in rs if all(s["err"] <= 1 for s in r["segs"]))
        print(f"  sessions where EVERY segment has <=1 error: "
              f"{allok}/{len(rs)}")

    block("ALL", lambda r: True)
    block("CLEAN (<7 edits)", lambda r: r["dist"] < 7)
    block("BAD (>=7 edits)", lambda r: r["dist"] >= 7)
    print(f"\nNULL: a one-substitution-wrong prefix still satisfies "
          f"'cross done': {null_hit}/{null_n} = {100*null_hit/max(null_n,1):.1f}%")

    out = "results/2026-08-09/islands.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
