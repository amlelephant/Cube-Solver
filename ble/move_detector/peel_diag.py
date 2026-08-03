"""
peel_diag.py

Why does the backward peel find no algorithms?

`algo6_s0/s1.json` say the peel exposes a real anchor in 1 of 24
session-runs: everywhere else it stops 1-2 onsets from the end having
peeled nothing, despite 57-513 candidates being available. ALGORITHM_PRIOR
§8e attributes this to "one unidentifiable algorithm blocks the whole
chain", which would be a failure at link 3 or 4 — but an anchor one onset
from the end is a failure at link 1, and those need different fixes.

This decides which it is, per session, without running the model (it reads
the cached `ctc_post_*.npz` written by `algorithm_gate.posteriorgram`):

  truth     the F2L-boundary anchor the solve actually passed through, its
            ground-truth move index, and the last-layer chunks
  library   is each true chunk's word in the mined library at all
  cands     does a candidate exist whose labels ARE a true chunk's word,
            and at which onset span — the peel cannot take a transition
            that was never generated
  reach     run peel_backward and report the deepest anchor found, whether
            the TRUE anchor is among the anchors at all, and at what cost
            and rank — an anchor that exists but ranks behind 400 cheaper
            do-nothing anchors is a ranking failure, not a generation one
  trace     per backward step: frontier size, cost range, how many
            algorithm peels were taken and how many were merely offered

Usage (from inside move_detector/):

    python peel_diag.py --ctc checkpoints/move_ctc_s0.pt
    python peel_diag.py --ctc checkpoints/move_ctc_s0.pt --session solve_20260730_111941_solve
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import algorithm_gate as AG
import reconstruct as RC
import verify_joint as VJ
from ctc_decode import prefix_beam_decode, ctc_to_moves


def load_cached(d: Path, tag: str):
    """Cached posteriorgram + the prepared stream, no model needed."""
    cache = d / f"ctc_post_{tag}.npz"
    if not cache.exists():
        return None, None
    from dataset import JointSessionStream
    stream = JointSessionStream(d / "detector_stream_color.npz")
    return np.load(cache)["class_prob"], stream


def true_anchor(d: Path):
    """
    (anchor state, ground-truth move index, chunks) at the F2L boundary.

    This is the state the peel is trying to find. Derived from the paired
    scramble take, never by walking back from solved — see
    algorithm_gate.paired_start for why that distinction matters.
    """
    chunks = AG.session_chunks(d)
    if not chunks:
        return None, None, None
    start, word = AG.paired_start(d), AG.session_word(d)
    states = AG.trajectory(start, word)
    ll = chunks[0]["ll_start"]
    return states[ll], ll, chunks


def peel_trace(cost_rows, del_costs, cands, beam=AG.PEEL_BEAM,
               max_peel=AG.MAX_PEEL, target=None):
    """
    peel_backward, instrumented. Same transitions and same order; the only
    additions are per-step counters and a check for `target` (the true
    anchor state) so we can see the step it is lost at, if it is ever
    reached at all.
    """
    n = len(cost_rows)
    by_end: dict[int, list[dict]] = {}
    for c in cands:
        by_end.setdefault(c["start"] + c["k"], []).append(c)

    tgt = target.tobytes() if target is not None else None
    solved_key = RC.SOLVED.tobytes()
    frontier = {(n, solved_key): (0.0, RC.SOLVED.copy(), [], 0)}
    anchors = {(n, solved_key): (0.0, RC.SOLVED.copy(), [])}
    steps = []

    for it in range(max_peel * 3):
        nxt: dict = {}
        n_algo_taken = n_algo_offered = 0
        target_in_frontier = False

        def push(j, state, cost, ops, depth):
            if j < 0:
                return
            key = (j, state.tobytes())
            cur = nxt.get(key)
            if cur is None or cost < cur[0]:
                nxt[key] = (cost, state, ops, depth)
            a = anchors.get(key)
            if a is None or cost < a[0]:
                anchors[key] = (cost, state, ops)

        for (j, sk), (cost, state, ops, depth) in frontier.items():
            if tgt is not None and sk == tgt:
                target_in_frontier = True
            faces = AG._f2l_faces(state)
            if not faces.any():
                continue
            if depth < max_peel:
                for c in by_end.get(j, []):
                    n_algo_offered += 1
                    vec = RC.SOLVED.copy()
                    for lab in c["labels"]:
                        vec = RC.compose(vec, RC.CLASS_VECS[lab])
                    prev = RC.compose(state, RC.inverse(vec))
                    if not (faces & AG._f2l_faces(prev)).any():
                        continue
                    n_algo_taken += 1
                    push(c["start"], prev, cost + c["cost"],
                         [("algorithm", list(c["labels"]))] + ops, depth + 1)
            if j == 0:
                continue
            for fi in np.where(faces)[0]:
                face = RC.TP_NAME[fi]
                for k in (RC.WCA12.index(face), RC.WCA12.index(face + "'")):
                    prev = RC.compose(state, RC.inverse(RC.CLASS_VECS[k]))
                    if not (faces & AG._f2l_faces(prev)).any():
                        continue
                    push(j - 1, prev, cost + float(cost_rows[j - 1][k]),
                         [("move", k)] + ops, depth)
            push(j - 1, state, cost + float(del_costs[j - 1]),
                 [("delete", None)] + ops, depth)

        costs = [v[0] for v in nxt.values()]
        steps.append({
            "step": it, "frontier_in": len(frontier),
            "generated": len(nxt),
            "algo_offered": n_algo_offered, "algo_taken": n_algo_taken,
            "cost_min": min(costs) if costs else None,
            "cost_max": max(costs) if costs else None,
            "kept_max": (sorted(costs)[:beam][-1] if costs else None),
            "target_in_frontier": target_in_frontier,
            "max_depth": max((v[3] for v in nxt.values()), default=0),
            "min_onset": min((k[0] for k in nxt), default=None),
        })
        if not nxt:
            break
        frontier = dict(sorted(nxt.items(), key=lambda kv: kv[1][0])[:beam])

    out = [{"onset": j, "state": st, "cost": c, "ops": ops}
           for (j, _), (c, st, ops) in anchors.items()]
    out.sort(key=lambda a: (a["cost"], -a["onset"]))
    return out, steps


def anchor_word(res: dict, a: dict) -> list[str] | None:
    """The full reconstruction an anchor implies: prefix story + peel tail."""
    pre = res["moves"] if res.get("solved") else res.get("best_effort_moves")
    if pre is None:
        return None
    return pre + AG.anchor_moves(a["ops"])


def oracle(d: Path, tag: str, args):
    """
    What would the BEST available anchor have scored?

    `decode_backward_first` only uses an anchor whose prefix decode reaches
    it (`solved`); when none does it falls back to the first shortlisted
    anchor, which is the zero-cost do-nothing one — so on a failing session
    the peel's work is discarded entirely and the arm reduces to the plain
    decode. 8 of 12 held-out sessions are failing sessions, so that path is
    the common case, not the exception.

    This scores EVERY shortlisted anchor's implied reconstruction against
    ground truth and reports the ceiling alongside what several selection
    rules would have picked. An oracle gap of ~0 means the anchor set has
    nothing better in it and D2 is not worth building; a large gap means
    the anchors are fine and only the selection rule is wrong.
    """
    gt = AG.session_word(d)
    class_prob, stream = load_cached(d, tag)
    if class_prob is None or gt is None:
        return None
    log_probs = np.log(np.maximum(class_prob, 1e-12))
    labels, frames = prefix_beam_decode(log_probs, beam=args.beam_ctc)
    moves = ctc_to_moves(class_prob, labels, frames, fps=stream.fps)
    if not moves:
        return None
    pred, cost_rows, del_costs = RC.costs_from_moves(
        moves, args.threshold, RC.BLEND_INV, RC.BLEND_UNIF,
        RC.C_DEL, None, RC.DEL_FLOOR, RC.BLEND_ADJ)
    start = RC.start_from_gt(gt)
    cands = AG.candidates_for_session(moves, class_prob, args.library)
    anchors = AG.shortlist_anchors(
        AG.peel_backward(cost_rows, del_costs, cands))

    tables = RC.build_tables()
    kw = dict(beam=args.anchor_beam, del_costs=del_costs, rotations=False,
              tables=tables, slices=False)
    rows = []
    for a in anchors[:args.max_anchors]:
        res = RC.decode_between(start, a["state"], cost_rows[:a["onset"]],
                                del_costs=del_costs[:a["onset"]],
                                **{k: v for k, v in kw.items()
                                   if k != "del_costs"})
        w = anchor_word(res, a)
        if w is None:
            continue
        n_alg = sum(1 for k, _ in a["ops"] if k == "algorithm")
        rows.append({
            "onset": a["onset"], "peel_cost": a["cost"], "n_algo": n_alg,
            # Worst per-move identification cost among this anchor's links.
            # The whole point of dumping these rows is that a selection rule
            # can then be tuned offline, without paying 40 decodes a session
            # to re-measure it.
            "max_nll": float(a.get("max_nll", 0.0)),
            "tail": len(AG.anchor_moves(a["ops"])),
            "solved": bool(res.get("solved")),
            "pre_cost": float(res.get("cost") if res.get("solved")
                              else res.get("best_effort_cost", 1e9)),
            "acc": RC.score_vs_gt(gt, w)["acc"]})
    if not rows:
        return None

    base = max((r for r in rows if r["n_algo"] == 0),
               key=lambda r: r["onset"], default=None)
    base_acc = base["acc"] if base else None
    best = max(rows, key=lambda r: r["acc"])

    def pick(name, key, sel=None):
        pool = [r for r in rows if sel(r)] if sel else rows
        if not pool:
            return name, None
        return name, min(pool, key=key)

    rules = [
        pick("cheapest total", lambda r: r["pre_cost"] + r["peel_cost"]),
        pick("deepest w/ algo", lambda r: r["onset"],
             lambda r: r["n_algo"] > 0),
        pick("deepest verified", lambda r: r["onset"],
             lambda r: r["solved"]),
        pick("cheapest verified", lambda r: r["pre_cost"] + r["peel_cost"],
             lambda r: r["solved"]),
    ]
    print(f"\n  {d.name}   {len(rows)} anchors scored")
    print(f"    {'rule':<22} {'acc':>7}  {'onset':>6} {'algs':>5} {'slv':>4}")
    print(f"    {'do-nothing (baseline)':<22} "
          f"{100*base_acc:6.1f}%  {base['onset']:>6} {0:>5} "
          f"{'Y' if base['solved'] else 'n':>4}" if base else "    no baseline")
    for name, r in rules:
        if r is None:
            print(f"    {name:<22}      -")
            continue
        print(f"    {name:<22} {100*r['acc']:6.1f}%  {r['onset']:>6} "
              f"{r['n_algo']:>5} {'Y' if r['solved'] else 'n':>4}")
    print(f"    {'ORACLE best':<22} {100*best['acc']:6.1f}%  "
          f"{best['onset']:>6} {best['n_algo']:>5} "
          f"{'Y' if best['solved'] else 'n':>4}"
          f"   (headroom {100*(best['acc'] - (base_acc or 0)):+.1f} pts)")
    return {"session": d.name, "base": base_acc, "oracle": best["acc"],
            "base_onset": base["onset"] if base else None,
            "n_onsets": len(pred), "n_gt": len(gt), "anchors": rows,
            "rules": {n: (r["acc"] if r else None) for n, r in rules}}


def analyse(d: Path, tag: str, args):
    gt = AG.session_word(d)
    if gt is None:
        return None
    class_prob, stream = load_cached(d, tag)
    if class_prob is None:
        print(f"  {d.name}: no cached posteriorgram for {tag} — skipping")
        return None

    log_probs = np.log(np.maximum(class_prob, 1e-12))
    labels, frames = prefix_beam_decode(log_probs, beam=args.beam_ctc)
    moves = ctc_to_moves(class_prob, labels, frames, fps=stream.fps)
    if not moves:
        return None
    pred, cost_rows, del_costs = RC.costs_from_moves(
        moves, args.threshold, RC.BLEND_INV, RC.BLEND_UNIF,
        RC.C_DEL, None, RC.DEL_FLOOR, RC.BLEND_ADJ)

    anc_state, ll_start, chunks = true_anchor(d)
    cands = AG.candidates_for_session(moves, class_prob, args.library)

    print(f"\n{'='*74}\n  {d.name}   {len(gt)} gt moves, {len(pred)} onsets, "
          f"{len(cands)} candidates\n{'='*74}")

    if chunks is None:
        print("  not scramble-paired / does not solve — no ground-truth "
              "last layer to find")
        return None

    print(f"  last layer starts at gt move {ll_start}; "
          f"{len(chunks)} chunk(s), {len(gt) - ll_start} moves")
    for ch in chunks:
        w = tuple(ch["word"])
        in_lib = w in args.library
        # Does a candidate carry EXACTLY this word? If not, the peel can
        # never take it, whatever the search does.
        labs = [RC.WCA12.index(m) for m in w]
        hits = [c for c in cands if c["labels"] == labs]
        span = f"{ch['span'][0]}-{ch['span'][1]}"
        print(f"    gt[{span:>9}] len {len(w):>2}  "
              f"{'in-lib' if in_lib else 'NOT-IN-LIB':<10} "
              f"{len(hits):>3} candidate(s) carry it"
              + (f"  onsets {sorted(set((h['start'], h['start']+h['k']) for h in hits))[:4]}"
                 if hits else "")
              + f"   {' '.join(w)}")

    # Where do candidates END? The peel starts at onset n and walks back one
    # onset per cheap step, so a candidate ending far from n is only
    # reachable after that many AUF/delete steps.
    ends = sorted({c["start"] + c["k"] for c in cands})
    print(f"  candidate END onsets (peel starts at {len(pred)}): "
          f"{ends[-12:] if ends else 'none'}")

    anchors, steps = peel_trace(cost_rows, del_costs, cands, target=anc_state)
    short = AG.shortlist_anchors(anchors)

    print(f"\n  {'step':>4} {'front':>6} {'gen':>7} {'algOff':>7} "
          f"{'algTake':>8} {'minCost':>8} {'cutoff':>8} {'depth':>6} "
          f"{'minOnset':>9} {'TRUE':>5}")
    for s in steps[:args.max_steps]:
        print(f"  {s['step']:>4} {s['frontier_in']:>6} {s['generated']:>7} "
              f"{s['algo_offered']:>7} {s['algo_taken']:>8} "
              f"{(s['cost_min'] if s['cost_min'] is not None else 0):>8.2f} "
              f"{(s['kept_max'] if s['kept_max'] is not None else 0):>8.2f} "
              f"{s['max_depth']:>6} {str(s['min_onset']):>9} "
              f"{'yes' if s['target_in_frontier'] else '':>5}")

    hit = [a for a in anchors if (a["state"] == anc_state).all()]
    deepest = min((a["onset"] for a in anchors), default=None)
    n_alg = max((sum(1 for k, _ in a["ops"] if k == "algorithm")
                 for a in anchors), default=0)
    print(f"\n  anchors {len(anchors)} (shortlist {len(short)}); deepest at "
          f"onset {deepest}; max algorithms peeled {n_alg}")
    if hit:
        best = min(hit, key=lambda a: a["cost"])
        rank = [i for i, a in enumerate(anchors)
                if (a["state"] == anc_state).all()][0]
        in_short = any((a["state"] == anc_state).all() for a in short)
        print(f"  TRUE ANCHOR FOUND: onset {best['onset']}, cost "
              f"{best['cost']:.2f}, rank {rank}/{len(anchors)}, "
              f"{'IN' if in_short else 'NOT IN'} shortlist")
    else:
        print("  TRUE ANCHOR NOT REACHED — the peel never generated it")
    return {"session": d.name, "found": bool(hit), "n_cands": len(cands),
            "deepest": deepest, "max_algos": n_alg,
            "n_chunks": len(chunks)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ctc", required=True)
    p.add_argument("--sessions", nargs="+",
                   default=["../training_data/solve_*/"])
    p.add_argument("--session", default=None, help="just this one")
    p.add_argument("--library-glob", default="../training_data/solve_*/")
    p.add_argument("--beam-ctc", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--oracle", action="store_true",
                   help="score every shortlisted anchor against ground "
                        "truth: the ceiling D2's selection rule aims at")
    p.add_argument("--anchor-beam", type=int, default=1000)
    p.add_argument("--max-anchors", type=int, default=40)
    p.add_argument("--out", default=None,
                   help="--oracle: dump every scored anchor to JSON")
    args = p.parse_args()

    import torch
    ckpt = torch.load(args.ctc, map_location="cpu", weights_only=False)
    tag = f"{Path(args.ctc).stem}_e{ckpt['epoch']}"
    train_names = set(ckpt.get("train_session_names", []))
    args.threshold = ckpt.get("threshold", 0.5)

    lib_dirs = [d for d in VJ.resolve_sessions([args.library_glob])
                if d.name in train_names]
    args.library = [w for w, _ in AG.build_library(lib_dirs).most_common()]
    print(f"  {args.ctc} (epoch {ckpt['epoch']}); library "
          f"{len(args.library)} words from {len(lib_dirs)} train sessions")

    dirs = [d for d in VJ.resolve_sessions(args.sessions)
            if d.name not in train_names]
    if args.session:
        dirs = [d for d in dirs if d.name == args.session]
    if not dirs:
        sys.exit("no held-out sessions matched")

    if args.oracle:
        res = [r for r in (oracle(d, tag, args) for d in dirs) if r]
        if not res:
            return
        print(f"\n{'='*74}\n  ANCHOR SELECTION CEILING\n{'='*74}")
        names = list(res[0]["rules"])
        print(f"  {'session':<32} {'base':>7} "
              + " ".join(f"{n[:13]:>14}" for n in names) + f" {'ORACLE':>8}")
        for r in res:
            cells = " ".join(
                (f"{100*r['rules'][n]:13.1f}%" if r["rules"][n] is not None
                 else f"{'-':>14}") for n in names)
            print(f"  {r['session']:<32} {100*r['base']:6.1f}% {cells} "
                  f"{100*r['oracle']:7.1f}%")
        print(f"\n  {'mean':<32} {100*np.mean([r['base'] for r in res]):6.1f}% "
              + " ".join(
                  f"{100*np.mean([x['rules'][n] for x in res if x['rules'][n] is not None]):13.1f}%"
                  for n in names)
              + f" {100*np.mean([r['oracle'] for r in res]):7.1f}%")
        if args.out:
            json.dump(res, open(args.out, "w"), indent=1)
            print(f"\n  wrote {args.out} — every scored anchor, so a "
                  f"selection rule can be tuned without re-decoding")
        return

    out = [r for r in (analyse(d, tag, args) for d in dirs) if r]
    if out:
        print(f"\n{'='*74}\n  SUMMARY\n{'='*74}")
        print(f"  true anchor reached: "
              f"{sum(r['found'] for r in out)}/{len(out)} sessions")
        for r in out:
            print(f"    {r['session']:<34} {'FOUND' if r['found'] else 'lost':<6} "
                  f"deepest onset {str(r['deepest']):>5}, "
                  f"max {r['max_algos']} algs peeled, "
                  f"{r['n_chunks']} true chunks")


if __name__ == "__main__":
    main()
