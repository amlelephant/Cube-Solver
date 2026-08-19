"""
error_phase.py — WHERE is the error, by phase of the solve?

Levenshtein-aligns the raw CTC decode against BLE truth, attributes every
edit (substitution / miss / phantom) to a TRUTH move index, and buckets
those indices into the four phases the solve actually has:

    cross     [0, settle)            improvised, D-invariant cross predicate
    f2l       [settle, ll_start)     pair insertions
    ll_alg    inside a last-layer algorithm chunk
    ll_gap    inside the last layer but BETWEEN chunks (AUF / recognition)

Reads cached posteriorgrams only -- no model load.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

MD = Path(__file__).resolve().parent
sys.path.insert(0, str(MD))
import algorithm_gate as AG
import reconstruct as RC
from ctc_decode import prefix_beam_decode, ctc_to_moves

DATA = MD.parent / "training_data"
BT = {f: RC.seq_to_state([f]) for f in AG.FACES}


def cross_settle(states, top, ll, w=8):
    idx = [i for i, f in enumerate(AG.EDGE_FACES) if AG.OPP[top] in f]
    bottom = AG.OPP[top]

    def done(s):
        t = s
        for _ in range(4):
            ep, eo = t[RC._EP], t[RC._EO]
            if all(ep[i] == i and eo[i] == 0 for i in idx):
                return True
            t = RC.compose(t, BT[bottom])
        return False

    ok = [done(s) for s in states]
    for i in range(ll + 1):
        if not ok[i]:
            continue
        run = best = 0
        for v in ok[i:ll + 1]:
            run = 0 if v else run + 1
            best = max(best, run)
        if best <= w:
            return i
    return ll


def align(gt, pred):
    """Levenshtein backtrace -> list of (op, gt_index) for every edit."""
    N, M = len(gt), len(pred)
    D = np.zeros((N + 1, M + 1), dtype=np.int32)
    D[:, 0] = np.arange(N + 1)
    D[0, :] = np.arange(M + 1)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            c = 0 if gt[i - 1] == pred[j - 1] else 1
            D[i, j] = min(D[i - 1, j - 1] + c, D[i - 1, j] + 1, D[i, j - 1] + 1)
    ops, i, j = [], N, M
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]):
            if gt[i - 1] != pred[j - 1]:
                ops.append(("sub", i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("miss", i - 1)); i -= 1
        else:
            ops.append(("phantom", min(i, N - 1))); j -= 1
    return ops[::-1], int(D[N, M])


def phase_of(i, settle, ll, chunk_spans):
    if i < settle:
        return "cross"
    if i < ll:
        return "f2l"
    for p, q in chunk_spans:
        if p <= i < q:
            return "ll_alg"
    return "ll_gap"


PHASES = ["cross", "f2l", "ll_alg", "ll_gap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="move_ctc_spd_s0_e54")
    ap.add_argument("--ckpt", default="checkpoints/move_ctc_spd_s0.pt")
    ap.add_argument("--beam-ctc", type=int, default=8)
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for d in sorted(DATA.glob("*_solve")):
        cache = d / f"ctc_post_{args.tag}.npz"
        if not cache.exists():
            continue
        chunks = AG.session_chunks(d)
        if not chunks:
            continue
        gt = AG.session_word(d)
        start = AG.paired_start(d)
        states = AG.trajectory(start, gt)
        top, ll = chunks[0]["top"], chunks[0]["ll_start"]
        spans = [c["span"] for c in chunks]
        settle = cross_settle(states, top, ll)

        from dataset import JointSessionStream
        stream = JointSessionStream(d / "detector_stream_color.npz")
        cp = np.load(cache)["class_prob"]
        lp = np.log(np.maximum(cp, 1e-12))
        labels, frames = prefix_beam_decode(lp, beam=args.beam_ctc)
        moves = ctc_to_moves(cp, labels, frames, fps=stream.fps)
        pred = [m["move"] for m in moves]
        if not pred:
            continue

        ops, dist = align(gt, pred)
        cnt = {p: {"sub": 0, "miss": 0, "phantom": 0} for p in PHASES}
        tot = {p: 0 for p in PHASES}
        for i in range(len(gt)):
            tot[phase_of(i, settle, ll, spans)] += 1
        for op, i in ops:
            cnt[phase_of(i, settle, ll, spans)][op] += 1

        rows.append(dict(session=d.name, n=len(gt), settle=settle, ll=ll,
                         dist=dist, cnt=cnt, tot=tot,
                         unseen=RC.classifier_unseen(d.name, args.ckpt),
                         gt=gt, pred=pred, spans=spans))

    def block(name, sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return
        print(f"\n=== {name}  (n={len(rs)} sessions, "
              f"{sum(r['n'] for r in rs)} truth moves) ===")
        print(f"{'phase':8} {'moves':>6} {'share':>6} | {'sub':>5} {'miss':>5} "
              f"{'phan':>5} {'ERR':>5} | {'err/move':>9} {'of all err':>11}")
        grand = sum(sum(r["cnt"][p][k] for k in ("sub", "miss", "phantom"))
                    for r in rs for p in PHASES)
        gmov = sum(r["tot"][p] for r in rs for p in PHASES)
        for p in PHASES:
            mv = sum(r["tot"][p] for r in rs)
            s = sum(r["cnt"][p]["sub"] for r in rs)
            m = sum(r["cnt"][p]["miss"] for r in rs)
            ph = sum(r["cnt"][p]["phantom"] for r in rs)
            e = s + m + ph
            print(f"{p:8} {mv:6d} {100*mv/gmov:5.1f}% | {s:5d} {m:5d} {ph:5d} "
                  f"{e:5d} | {e/max(mv,1):9.3f} {100*e/max(grand,1):10.1f}%")
        print(f"{'TOTAL':8} {gmov:6d} {100:5.1f}% | "
              f"{sum(r['cnt'][p]['sub'] for r in rs for p in PHASES):5d} "
              f"{sum(r['cnt'][p]['miss'] for r in rs for p in PHASES):5d} "
              f"{sum(r['cnt'][p]['phantom'] for r in rs for p in PHASES):5d} "
              f"{grand:5d} | {grand/gmov:9.3f} {100:10.1f}%")

    block("ALL cached sessions", lambda r: True)
    block("CLASSIFIER-UNSEEN only", lambda r: r["unseen"])

    for r in rows[:args.dump]:
        print(f"\n--- {r['session']}  n={r['n']} settle={r['settle']} "
              f"ll={r['ll']} lev={r['dist']}")
        print("  gt  :", " ".join(r["gt"]))
        print("  pred:", " ".join(r["pred"]))
        print("  LL  :", " | ".join(" ".join(r["gt"][p:q]) for p, q in r["spans"]))

    json.dump([{k: v for k, v in r.items() if k not in ("gt", "pred", "spans")}
               for r in rows], open(MD / "results" / "error_phase.json", "w"), indent=1)


if __name__ == "__main__":
    main()
