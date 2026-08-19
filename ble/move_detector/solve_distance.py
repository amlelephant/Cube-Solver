"""
solve_distance.py — can the DECODED MOVE IDENTITIES, replayed from the known
scramble, tell a real solve from a plausible-looking flail?

The question this answers
-------------------------
The count gate (`anticheat_gate.py`) reads only HOW MANY moves happened, and
that is deliberate: counting runs on the model's strong axis, while
move-by-move verification sits behind an accuracy cliff at ~98-99% per-move
(`ACCURACY_TARGET.md`, and `verification-accuracy-cliff`). So the attack that
survives it is "make a full solve's worth of real moves without solving".

`solved_check.py` closes that visually. This file asks whether the MOVE MODEL
can close it too, on a much weaker use of identity than verification needs:

    replay the decoded moves from the server-issued scramble.
    How far from solved did they land? DQ beyond a hard maximum.

This is not verification. Verification asks "is there a consistent story",
needs a repair search, and is measured dead on held-out data (0/28,
`paper-holdout-remeasurement`). This asks only for a SCALAR distance and
puts a ceiling on it, which is a strictly easier question — a solve that
lands near solved passes without anyone having to reconstruct which moves
were misread.

Two statistics, and the second is the one that matters
------------------------------------------------------
  n_solved  cubies home and oriented at the replayed end state (0-20).
  h_full    pruning-table lower bound on the moves STILL NEEDED from there.
            A true lower bound, so it cannot flatter a cheat.

`n_solved` is the intuitive one and it is fragile: a single decode error in
the MIDDLE of a solve does not shift the end state slightly, it CONJUGATES
it, and a conjugated quarter turn scatters. Measured on the first session
tried: edit distance 1 against BLE truth, and n_solved fell 20 -> 12. So the
honest side does not sit at 20 and any threshold pretending otherwise would
DQ everyone.

The null models
---------------
Three, because the answer depends entirely on what a cheat actually does:

  random    a same-length sequence of uniform random quarter turns. The
            "flailing" attack literally as described.
  partial   the first k moves of the session's own true solve, k drawn so
            the cheat stops before finishing. This is the HARD null and the
            honest one: someone who solves the cross and F2L and then stops
            has 12 of 20 cubies home — the same score the honest 1-error
            decode above produced. If the statistic cannot separate this, it
            cannot separate the attack that matters.
  decoy     the session's own decoded moves replayed from a DIFFERENT
            scramble. Models "these moves do not solve THIS cube", which is
            the server-side version of the same question.

MEASURED 2026-08-10 — the DQ direction is dead, the verify direction is not
---------------------------------------------------------------------------
24 sessions (`move_ctc_spd_s0_e54`), 200 nulls each of 3 kinds = 4800 per
model. Replaying BLE truth scores 20/20 on every session, so the ground truth
and the replay are sound and the numbers below are about the decode.

**As a hard maximum / DQ rule: it catches nothing.** 4 of 24 honest solves
replay to `n_solved = 0` — as scrambled as a uniformly random sequence — so
a threshold that never false-DQs sits at 0 and DQs 0/4800 of every null.

The mechanism is exact and worth stating, because it kills the idea cleanly
rather than merely disappointing: **`n_solved` is very nearly a deterministic
function of the decode's edit distance.**

    edit distance | 0  | 1  | 2   | 3 | 6-7      | 10-11 | 13+
    n_solved      | 20 | 12 | 7,9 | 6 | 2,3,3,3,4| 3,5   | 0,0,0,2

All seven 1-error sessions land on exactly 12. A mid-sequence error does not
nudge the end state, it CONJUGATES it, and a conjugated quarter turn
scatters. The first error costs ~8 cubies and the signal is inside the random
floor (null median 1, best 5) by ~5 errors — while daytime word error already
puts a typical solve at 1-3 errors and the tail at 30+.

**As positive evidence it survives, because the direction is not symmetric.**
Landing NEAR solved cannot happen by accident: it requires the moves to
genuinely carry that scramble to that state. So pin the threshold above every
null and read it as a proof rather than an accusation:

    | null model      | best of 4800 | honest solves verified, 0 false accept |
    |-----------------|--------------|----------------------------------------|
    | random flail    | 5            | **13/24 = 54%**                        |
    | decoy scramble  | 6            | 12/24 = 50%                            |
    | partial solve   | 17           | 2/24 = 8%                              |

**The partial-solve null is the weak cell, and it is exactly what
`solved_check.py` is strongest against** — someone who stops before the last
layer has a visibly unsolved cube at the timer stop. The two arms are
complementary in precisely the right way, which is the useful finding here.

**Both seeds agree.** `move_ctc_spd_s1_e54` has only 6 sessions cached, and
they are the 6 HARDEST for s0 (edit 7-33). On that common subset the seeds
track per-session (0/0, 3/2, 3/2, 2/1, 0/1, 0/1) and both give a median of
1.0. So this is not seed luck — but note what it implies: on the most recent
and hardest sessions the statistic is dead for BOTH seeds, and s0's 54% is
carried by the easier ones. Quote 54% with that caveat attached or not at all.

Not wired into `anticheat_gate.adjudicate()`. A 54% fast path that collapses
to 0% on the hardest sessions is a measurement, not yet an arm.

Run from move_detector/ (needs cached posteriorgrams):

    python solve_distance.py --tag move_ctc_spd_s0_e54
"""
import argparse
import json
from pathlib import Path

import numpy as np

import algorithm_gate as AG
import error_phase as EP
import reconstruct as RC
from ctc_decode import prefix_beam_decode, ctc_to_moves

DATA = Path(__file__).resolve().parent.parent / "training_data"


def replay(start, word):
    """End state after applying `word` to `start`."""
    return RC.compose(start, RC.seq_to_state(word))


def score(state, tables):
    st = np.atleast_2d(state)
    return {
        "n_solved": int(RC.n_solved(st)[0]),
        "h_full": int(RC.h_full(np.asarray(state), tables)),
    }


def session_rows(d, tag, beam_ctc, rng, n_null, tables):
    """One session -> the honest row plus its null rows."""
    gt = AG.session_word(d)
    if not gt:
        return None
    start = RC.start_from_gt(gt)

    cp = np.load(d / f"ctc_post_{tag}.npz")["class_prob"]
    from dataset import JointSessionStream
    stream = JointSessionStream(d / "detector_stream_color.npz")
    lab, fr = prefix_beam_decode(np.log(np.maximum(cp, 1e-12)), beam=beam_ctc)
    moves = ctc_to_moves(cp, lab, fr, fps=stream.fps)
    pred = [m["move"] for m in moves]
    if not pred:
        return None
    _, dist = EP.align(gt, pred)

    out = {
        "session": d.name,
        "n_gt": len(gt),
        "n_pred": len(pred),
        "edit_distance": int(dist),
        # Sanity: replaying TRUTH must give a solved cube by construction
        # (start_from_gt is its inverse). If this is not 20 the session's
        # ground truth is broken and nothing below means anything.
        "truth": score(replay(start, gt), tables),
        "honest": score(replay(start, pred), tables),
        "null_random": [],
        "null_partial": [],
        "null_decoy": [],
    }

    for _ in range(n_null):
        w = [RC.WCA12[i] for i in rng.integers(0, 12, len(pred))]
        out["null_random"].append(score(replay(start, w), tables))

        # Stop somewhere in the last two thirds: before that it is not a
        # plausible "full solve's worth of moves" and the count gate has it.
        k = int(rng.integers(int(len(gt) * 0.55), max(int(len(gt) * 0.95), 2)))
        out["null_partial"].append(score(replay(start, gt[:k]), tables))

        # A decoy start: the true start permuted by a random scramble, which
        # is what "the same moves against a different cube" means.
        sc = [RC.WCA12[i] for i in rng.integers(0, 12, 25)]
        out["null_decoy"].append(score(replay(RC.seq_to_state(sc), pred),
                                       tables))
    return out


def summarise(rows, key):
    """Distribution of a statistic over honest rows and each null."""
    honest = np.array([r["honest"][key] for r in rows], float)
    truth = np.array([r["truth"][key] for r in rows], float)
    nulls = {}
    for name in ("null_random", "null_partial", "null_decoy"):
        nulls[name] = np.array([s[key] for r in rows for s in r[name]], float)
    return honest, truth, nulls


def report(rows, key, higher_is_solved):
    honest, truth, nulls = summarise(rows, key)
    print(f"\n=== {key} "
          f"({'higher' if higher_is_solved else 'lower'} = closer to solved) "
          f"===")
    print(f"  truth replay      med {np.median(truth):6.1f}   "
          f"(must be {'20' if key == 'n_solved' else '0'} — sanity)")
    print(f"  honest decode     med {np.median(honest):6.1f}   "
          f"worst {(honest.min() if higher_is_solved else honest.max()):6.1f}   "
          f"n={len(honest)}")
    for name, v in nulls.items():
        print(f"  {name:<17} med {np.median(v):6.1f}   "
              f"best {(v.max() if higher_is_solved else v.min()):6.1f}   "
              f"n={len(v)}")

    # The operating point a DQ rule would actually use: pinned by the WORST
    # honest solve, because this rule must never DQ a real one.
    print("\n  zero-false-DQ threshold (set by the worst honest solve):")
    thr = honest.min() if higher_is_solved else honest.max()
    for name, v in nulls.items():
        caught = (v < thr).sum() if higher_is_solved else (v > thr).sum()
        print(f"    vs {name:<15} DQ {caught}/{len(v)} = {100*caught/len(v):5.1f}%"
              f"   (threshold {'<' if higher_is_solved else '>'} {thr:.0f})")


def report_verify(rows, key="n_solved"):
    """The OTHER direction, and the one that survives.

    A hard maximum is a DQ rule: it needs the worst honest solve to be
    cleaner than the best cheat, and it is not — decode errors scatter the
    replayed end state until a noisy honest solve is indistinguishable from
    a random one. But the converse is not symmetric. Landing NEAR solved
    cannot happen by accident: it requires the moves to genuinely take that
    scramble to that state. So the statistic is worthless as an accusation
    and strong as a PROOF.

    This reports it that way: pin the threshold above every null (zero false
    ACCEPTS) and ask how many honest solves clear it.
    """
    honest = np.array([r["honest"][key] for r in rows], float)
    _, _, nulls = summarise(rows, key)

    print(f"\n=== {key} as POSITIVE evidence (fast-path verify, not DQ) ===")
    for name, v in nulls.items():
        thr = v.max()
        passed = int((honest > thr).sum())
        print(f"  vs {name:<13} threshold >{thr:4.0f} (best of {len(v)} nulls)"
              f"   verifies {passed:2d}/{len(honest)} honest "
              f"= {100*passed/len(honest):5.1f}%")
    allmax = max(v.max() for v in nulls.values())
    passed = int((honest > allmax).sum())
    print(f"  vs ALL nulls    threshold >{allmax:4.0f}"
          f"                     verifies {passed:2d}/{len(honest)} "
          f"= {100*passed/len(honest):5.1f}%")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="move_ctc_spd_s0_e54")
    ap.add_argument("--beam-ctc", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tables = RC.build_tables()
    rng = np.random.default_rng(0)

    sessions = [d for d in sorted(DATA.glob("*_solve"))
                if (d / f"ctc_post_{args.tag}.npz").exists()]
    if args.limit:
        sessions = sessions[:args.limit]
    if not sessions:
        raise SystemExit(f"no sessions with ctc_post_{args.tag}.npz")

    rows = []
    for i, d in enumerate(sessions):
        r = session_rows(d, args.tag, args.beam_ctc, rng, args.n_null, tables)
        if r is None:
            continue
        rows.append(r)
        print(f"[{i+1}/{len(sessions)}] {d.name}: edit={r['edit_distance']:3d} "
              f"n_solved={r['honest']['n_solved']:2d} "
              f"h={r['honest']['h_full']:2d} "
              f"(truth n_solved={r['truth']['n_solved']})")

    bad = [r for r in rows if r["truth"]["n_solved"] != 20]
    if bad:
        print(f"\nWARNING: {len(bad)} sessions where replaying TRUTH does not "
              f"solve — their ground truth is broken, excluded:")
        for r in bad:
            print(f"    {r['session']}")
        rows = [r for r in rows if r["truth"]["n_solved"] == 20]

    print(f"\n{len(rows)} sessions, {args.n_null} nulls each")
    report(rows, "n_solved", higher_is_solved=True)
    report(rows, "h_full", higher_is_solved=False)
    report_verify(rows, "n_solved")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tag": args.tag, "rows": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
