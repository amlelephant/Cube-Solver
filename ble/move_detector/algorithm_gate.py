"""
algorithm_gate.py

Gate for the *generative* last-layer algorithm prior: can a whole OLL/PLL
algorithm be IDENTIFIED from the CTC posteriorgram without that stretch
having been read correctly first?

Why this is not ALGORITHM_PRIOR.md again
----------------------------------------
`algorithms.py` is a RECOGNISER. It reads the predicted move string and
looks for a known algorithm inside it, which requires the neighbourhood to
have been read cleanly (`cost/move <= 0.25`) — and a classifier error is
precisely where it was not. §6e measured that ceiling exactly: reach 19%,
structural, and it self-selects onto the parts of the solve that were
already right.

This asks the opposite question. The decoder's beam already carries a
CONCRETE CUBE STATE per hypothesis, so the abstraction

    alpha : cube state -> (cross done, F2L complete, LL oriented, solved)

is computable for free at every step, with no perception involved. At any
hypothesis whose abstract state says "F2L complete", the small set of
library words that move the abstract state upward can be GENERATED and each
scored as a whole against the posteriorgram. Nothing has to be recognised,
so there is no reach: coverage of the last layer is 100% by construction,
and a span containing dropped onsets is scored no differently from a clean
one.

The measured facts this is built on (2026-08-01, 20 scramble-paired
sessions, BLE ground truth replayed through the cube group):

  * every solve is one 49-78 quarter-turn F2L-building stretch followed by
    2-7 last-layer chunks, each of which breaks the F2L predicate and
    restores it within 5-20 quarter turns. The two do not overlap in
    length, so the boundary is unambiguous.
  * the 95 last-layer chunks collapse to 14 distinct words; the top 8 cover
    90.5%. The structure holds on the cross-day sessions (07-29/30/31) that
    no checkpoint has seen.
  * the last layer is 46% of moves but 78% of all MISSES, and misses are
    the channel that verification actually dies on (ACCURACY_TARGET.md:
    3% substitutions still verifies 100% of the time, 3% misses verifies
    45%) and the one with a structural floor no threshold reaches.

Why CTC is what makes this scorable
-----------------------------------
The first matcher's fixed-offset alignment broke on exactly one dropped
onset (see `[[algorithm-prior-blocked-by-reach]]`). The CTC forward
algorithm gives log P(word | frames) marginalised over EVERY alignment, so
a candidate word costs nothing extra for containing onsets the detector
never fired and needs no alignment assumption at all. That is the whole
reason this is worth re-asking now and was not before `train_ctc.py`.

What this file does and does not do
-----------------------------------
It changes no decoder. It is a measurement, in the same spirit as
`algorithms.py --step1`, and it exists to be failed cheaply. It reports:

  ident     is the true word the argmax of the candidate set, on spans the
            plain CTC decode got WRONG (the only interesting case)
  jitter    does identification survive the span being placed wrong, which
            is the operating assumption: the decoder will propose a chunk
            boundary from a hypothesis, not from an oracle
  null      the same candidate set on F2L spans and on frame-shuffled
            spans — without separation here the test has no precision
  budget    ground-truth moves collapsible into one transition, and the
            insertion cost that removes from `gt_path_cost`

Accept / reject, decided before running (ALGORITHM_PRIOR.md §5 discipline):

  PASS   >= 80% top-1 identification on dirty spans, held-out sessions,
         with the F2L-decoy null separated on nll/move, and identification
         surviving +/-10 frames of span jitter.
  FAIL   identification only where the plain decode was already right —
         that is the 19%-reach ceiling wearing a new hat, and it dies here.

Usage (from inside move_detector/):

    python algorithm_gate.py --ctc checkpoints/move_ctc_s0.pt
    python algorithm_gate.py --ctc checkpoints/move_ctc_s1.pt --out results/2026-08-01/gate_s1.json
    python algorithm_gate.py --selftest
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import reconstruct as RC
from decode import align_sequences

# --------------------------------------------------------------------------
# 1. The abstraction: cube state -> solve progress
# --------------------------------------------------------------------------
#
# Kociemba cubie naming, which is what reconstruct.py's state vector uses.
CORNER_FACES = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]
EDGE_FACES = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB",
              "FR", "FL", "BL", "BR"]
FACES = "URFDLB"
OPP = {"U": "D", "D": "U", "L": "R", "R": "L", "F": "B", "B": "F"}

# Longest last-layer algorithm we expect, in quarter turns. Measured range
# is 6-21; F2L building is 49-78. Any threshold in between separates them,
# which is why the boundary rule below needs no tuning.
MAX_ALG = 30


def piece_sets(top: str) -> tuple[list, list, list, list]:
    """(top corners, top edges, F2L corners, F2L edges) for a given top face."""
    tc = [i for i, f in enumerate(CORNER_FACES) if top in f]
    te = [i for i, f in enumerate(EDGE_FACES) if top in f]
    return (tc, te,
            [i for i in range(8) if i not in tc],
            [i for i in range(12) if i not in te])


def all_solved(state: np.ndarray, corners, edges) -> bool:
    """Are these cubies home and oriented?"""
    cp, co = state[RC._CP], state[RC._CO]
    ep, eo = state[RC._EP], state[RC._EO]
    return (all(cp[i] == i and co[i] == 0 for i in corners)
            and all(ep[i] == i and eo[i] == 0 for i in edges))


def trajectory(start: np.ndarray, word: list[str]) -> list[np.ndarray]:
    """Cube state after each prefix of `word`; index i = after i moves."""
    states, s = [start.copy()], start.copy()
    for m in word:
        s = RC.compose(s, RC.seq_to_state([m]))
        states.append(s)
    return states


def broken_runs(done: list[bool]) -> list[tuple[int, int]]:
    """Maximal [p, q) intervals where the F2L predicate does not hold."""
    out, i, n = [], 0, len(done)
    while i < n:
        if not done[i]:
            j = i
            while j < n and not done[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def pick_top_face(states: list[np.ndarray]) -> str:
    """
    Which face did the solver build the last layer on?

    Chosen by which top face has its F2L complete most often over the back
    half of the solve — that is where a layer-by-layer solve is finishing,
    and the wrong face is essentially never complete there.
    """
    half = len(states) // 2
    best, top = -1, "U"
    for f in FACES:
        _, _, fc, fe = piece_sets(f)
        n = sum(all_solved(s, fc, fe) for s in states[half:])
        if n > best:
            best, top = n, f
    return top


def ll_chunks(states: list[np.ndarray], top: str
              ) -> tuple[int | None, list[tuple[int, int]]]:
    """
    (index where the last layer begins, algorithm spans as move indices).

    A chunk is returned as a half-open move-index span [p, q): the moves
    that break the F2L predicate and the one that restores it. Cut points
    between chunks — where F2L holds — are AUF and recognition pauses, and
    are deliberately NOT part of any chunk.
    """
    _, _, fc, fe = piece_sets(top)
    done = [all_solved(s, fc, fe) for s in states]
    runs = broken_runs(done)
    if not runs:
        return None, []
    p, q = max(runs, key=lambda r: r[1] - r[0])          # the F2L build
    if q - p < MAX_ALG or q >= len(done) or not done[q]:
        return None, []                                   # no clean boundary
    ll_start = q
    # A run of broken STATES [a, b) means move a-1 broke the predicate and
    # move b-1 restored it, so the algorithm is moves [a-1, b).
    spans = [(ll_start + a - 1, ll_start + b)
             for a, b in broken_runs(done[ll_start:])]
    return ll_start, spans


# --------------------------------------------------------------------------
# 2. The library, mined from training sessions only
# --------------------------------------------------------------------------

def session_word(d: Path) -> list[str] | None:
    recs = [json.loads(l) for l in open(d / "moves.jsonl") if l.strip()]
    w = [r.get("wca_notation") for r in recs]
    if not w or any(x is None or x not in RC.WCA12 for x in w):
        return None
    return w


def paired_start(d: Path) -> np.ndarray | None:
    """
    The state the solve started from, built from its scramble take.

    Deliberately NOT derived by walking backwards from solved: that assumes
    the answer (the recording ending solved) and silently produces a
    garbage trajectory when the assumption fails, which it does on the
    unpaired early sessions.
    """
    sc = d.parent / d.name.replace("_solve", "_scramble")
    if sc == d or not (sc / "moves.jsonl").exists():
        return None
    w = session_word(sc)
    if w is None:
        return None
    return RC.compose(RC.SOLVED, RC.seq_to_state(w))


def session_chunks(d: Path) -> list[dict] | None:
    """Last-layer algorithm spans of one session, or None if not analysable."""
    start, word = paired_start(d), session_word(d)
    if start is None or word is None:
        return None
    states = trajectory(start, word)
    if not (states[-1] == RC.SOLVED).all():
        return None                                    # take does not solve
    top = pick_top_face(states)
    ll_start, spans = ll_chunks(states, top)
    if ll_start is None:
        return None
    return [{"span": (p, q), "word": word[p:q], "top": top,
             "ll_start": ll_start, "n": len(word)} for p, q in spans]


def build_library(dirs: list[Path]) -> Counter:
    """Distinct last-layer words and their counts, over the given sessions."""
    lib = Counter()
    for d in dirs:
        for ch in session_chunks(d) or []:
            lib[tuple(ch["word"])] += 1
    return lib


# --------------------------------------------------------------------------
# 3. Scoring a candidate word against the posteriorgram (CTC forward)
# --------------------------------------------------------------------------

NEG_INF = -1e30


def ctc_logp(log_probs: np.ndarray, labels: list[int],
             blank: int = 12) -> float:
    """
    log P(labels | frames) under CTC, summed over every alignment.

    This is the property the whole approach rests on. A fixed-offset
    matcher scores position j of the word against onset i+j and so is
    destroyed by one dropped onset; this marginalises over alignments, so
    a word whose onsets the detector never fired is scored on the evidence
    that IS there rather than penalised for the evidence that is not.

    `log_probs` is (T, C) log-softmax rows. Returns -inf when the span is
    too short to emit the word at all.
    """
    T = log_probs.shape[0]
    L = len(labels)
    if L == 0:
        return float(log_probs[:, blank].sum())
    # Extended sequence: blank between every pair of labels, and at the ends.
    ext = np.full(2 * L + 1, blank, dtype=int)
    ext[1::2] = labels
    S = len(ext)
    # Minimum frames needed: every label, plus a blank between equal
    # neighbours (CTC cannot emit two identical labels back to back).
    need = L + sum(1 for i in range(1, L) if labels[i] == labels[i - 1])
    if T < need:
        return float("-inf")

    # Which extended positions may be reached by a skip of 2 (i.e. the
    # blank between them is optional).
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    a = np.full(S, NEG_INF)
    a[0] = log_probs[0, blank]
    if S > 1:
        a[1] = log_probs[0, ext[1]]
    for t in range(1, T):
        prev = a
        shift1 = np.concatenate(([NEG_INF], prev[:-1]))
        cur = np.logaddexp(prev, shift1)
        shift2 = np.concatenate(([NEG_INF, NEG_INF], prev[:-2]))
        cur = np.where(skip, np.logaddexp(cur, shift2), cur)
        a = cur + log_probs[t, ext]

    return float(np.logaddexp(a[-1], a[-2]) if S > 1 else a[-1])


def ctc_logp_local(log_probs: np.ndarray, labels: list[int],
                   blank: int = 12) -> float:
    """
    Best-scoring placement of `labels` ANYWHERE inside the span.

    `ctc_logp` is global: it requires the word to explain every frame of the
    window, so a span that brackets one extra AUF turn scores the true word
    as badly wrong. That is the same trap `algorithms.py` fell into and
    documented — "a banded DP for 'does X appear somewhere near here' needs
    LOCAL alignment, not global; global silently taxes the search for not
    already knowing where the match starts."

    Frames before the match and after it are simply not scored (they are
    explained by an unspecified model, contributing 0), so this is the
    formulation to use whenever the span boundary comes from a hypothesis
    rather than from ground truth — i.e. always, in deployment. Maximised
    over the end frame rather than summed, so no path is counted twice.
    """
    T = log_probs.shape[0]
    L = len(labels)
    if L == 0 or T == 0:
        return 0.0
    ext = np.full(2 * L + 1, blank, dtype=int)
    ext[1::2] = labels
    S = len(ext)
    need = L + sum(1 for i in range(1, L) if labels[i] == labels[i - 1])
    if T < need:
        return float("-inf")

    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    a = np.full(S, NEG_INF)
    best = NEG_INF
    for t in range(T):
        prev = a.copy()
        prev[0] = 0.0                       # the match may start at frame t
        shift1 = np.concatenate(([NEG_INF], prev[:-1]))
        cur = np.logaddexp(prev, shift1)
        shift2 = np.concatenate(([NEG_INF, NEG_INF], prev[:-2]))
        cur = np.where(skip, np.logaddexp(cur, shift2), cur)
        a = cur + log_probs[t, ext]
        end = np.logaddexp(a[-1], a[-2]) if S > 1 else a[-1]
        best = max(best, float(end))        # the match may end at frame t
    return best


def score_candidates(log_probs: np.ndarray, lo: int, hi: int,
                     library: list[tuple], blank: int = 12,
                     local: bool = False) -> list[dict]:
    """Every library word scored against frames [lo, hi), best first."""
    window = log_probs[max(0, lo):max(hi, 0)]
    fn = ctc_logp_local if local else ctc_logp
    out = []
    for w in library:
        labels = [RC.WCA12.index(m) for m in w]
        lp = fn(window, labels, blank)
        out.append({"word": w, "logp": lp,
                    "nll_per_move": (-lp / len(w)) if np.isfinite(lp)
                                    else float("inf")})
    out.sort(key=lambda r: -r["logp"])
    return out


# --------------------------------------------------------------------------
# 3b. Candidate generation for the decoder
# --------------------------------------------------------------------------
#
# What the beam consumes. Everything below produces plain numpy/python data
# so `reconstruct.py` never imports this module (or torch): candidates are
# passed in, not fetched.

# Accept a span as a possible algorithm at or below this cost per move.
# Measured 2026-08-01 over both seeds: a single threshold near here accepts
# ~98% of real last-layer chunks and 0.4% of nulls (F2L spans of equal
# length, and true spans with their frames shuffled). Do NOT loosen it
# without re-running `--selftest`-adjacent nulls: the failure mode is a
# near-free 15-move transition offered where no algorithm happened, which
# is the OP_SLICE beam flood at fifteen times the scale.
ALGO_GATE = 2.3

# Prior against using an algorithm transition at all, in the same -log
# units as C_DEL / C_INS. Small, because the likelihood term carries the
# discrimination; non-zero, so that where the per-onset reading is already
# clean the transition is inert rather than a free lunch.
C_ALGO = 2.0

# How far the onset count may differ from the word length. A chunk of L
# moves may arrive as fewer onsets (misses, BLE collisions) or more
# (phantoms); measured miss/phantom rates put almost everything inside
# these, and each extra value costs a full re-scoring pass.
MAX_MISS, MAX_PHANTOM = 6, 3


def _free_logp(window: np.ndarray) -> float:
    """
    Best-path log-probability of a window under no constraint.

    Used as the baseline the algorithm hypothesis is priced against, so the
    cost is a likelihood RATIO — "how much worse than the model's own
    favourite explanation is 'this span is exactly this word'".

    It is NOT a bound on `ctc_logp_local`, and the ratio must be clamped at
    zero by the caller. Two reasons it can come out negative: the CTC score
    SUMS over alignments, which can exceed any single path; and the local
    scorer leaves frames outside the match unscored while this sums the
    (negative) best log-prob of every frame in the window. A negative
    transition cost would let the beam lower its cost by taking algorithms,
    breaking the monotonicity decode()'s tail search relies on.
    """
    return float(window.max(axis=1).sum())


def build_candidates(onset_frames: list[int], top_classes: list[int],
                     log_probs: np.ndarray, library: list[tuple],
                     pad: int = 8, gate: float = ALGO_GATE,
                     c_algo: float = C_ALGO,
                     overlap_frac: float = 0.5) -> list[dict]:
    """
    Every (start onset, onset count, library word) the beam may consider.

    Returned dicts are self-contained: `start` and `k` say which onsets the
    transition consumes, `labels` are WCA12 class indices in order, `cost`
    is already in the decoder's -log units.

    The scorer is `ctc_logp_local`, never the global one. The span here is
    derived from onset frames, which in deployment are the DETECTOR's
    onsets, not ground truth — they drift, they miss, and the local scorer
    is flat under exactly that (ALGORITHM_PRIOR.md §7b) where the global one
    collapses.

    `overlap_frac` is a cheap pre-screen: a word whose moves barely appear
    among the span's argmax classes cannot survive scoring, and skipping it
    avoids the dominant cost of this function.
    """
    n = len(onset_frames)
    T = log_probs.shape[0]
    out = []
    for i in range(n):
        for word in library:
            L = len(word)
            labels = [RC.WCA12.index(m) for m in word]
            want = Counter(labels)
            for k in range(max(2, L - MAX_MISS), L + MAX_PHANTOM + 1):
                if i + k > n:
                    break
                got = Counter(top_classes[i:i + k])
                shared = sum(min(want[c], got[c]) for c in want)
                if shared < overlap_frac * L:
                    continue
                lo = max(0, onset_frames[i] - pad)
                hi = min(T, onset_frames[i + k - 1] + pad + 1)
                if hi - lo < L:
                    continue
                window = log_probs[lo:hi]
                lp = ctc_logp_local(window, labels)
                if not np.isfinite(lp):
                    continue
                nll = -lp / L
                if nll > gate:
                    continue
                ratio = max(0.0, _free_logp(window) - lp)
                # An onset consumed by the span that produced no move is a
                # PHANTOM, and must be priced as one. Without this the cost
                # is independent of k — the local scorer skips leading and
                # trailing frames for free, so every k in range ties at
                # exactly c_algo and the choice among them is arbitrary.
                # Measured consequence: an 8-move word was selected at k=11,
                # so the transition swallowed three real onsets and declared
                # them silent, which corrupts the reference story the
                # ranking ladder is built from and cost a session that
                # previously verified. Words with FEWER onsets than moves
                # are not penalised — filling in misses for free is the
                # entire point.
                extra = max(0, k - L)
                out.append({"start": i, "k": k, "labels": labels,
                            "nll_per_move": float(nll),
                            "cost": float(c_algo + ratio + extra * RC.C_DEL),
                            "word": " ".join(word)})
    # Keep the cheapest reading of each (start, k) — the same span scored by
    # two different words is a real ambiguity, but two costs for the same
    # word/span pair is not.
    best: dict = {}
    for c in out:
        key = (c["start"], c["k"])
        if key not in best or c["cost"] < best[key]["cost"]:
            best[key] = c
    return sorted(best.values(), key=lambda c: (c["start"], c["cost"]))


def candidates_for_session(moves: list[dict], class_prob: np.ndarray,
                           library: list[tuple], **kw) -> list[dict]:
    """`build_candidates` straight off the standard moves list."""
    frames = [int(m["frame"]) for m in moves]
    top = [RC.WCA12.index(m["move"]) for m in moves]
    log_probs = np.log(np.maximum(class_prob, 1e-12))
    return build_candidates(frames, top, log_probs, library, **kw)


# --------------------------------------------------------------------------
# 3c. Backward peel: identify the last layer FIRST, from the end
# --------------------------------------------------------------------------
#
# Why backwards. Offered as a left-to-right transition, OP_ALGO can only fire
# from a hypothesis whose first two layers are already correct — so it needs
# the prefix decoded before it can help, and the prefix is where a measured
# median 60% of the true story's cost sits. Gating on the abstract state is
# right; requiring the decoder to EARN that state first is what made it
# inert (measured: 1 transition taken across 12 held-out sessions, 0 misses
# recovered).
#
# The end of a solve is known exactly — it is solved. So peel from there:
# identify the last algorithm, undo it, identify the one before, and keep
# going while the abstract state stays inside the last layer. What falls out
# is a concrete cube state at the F2L boundary, and THAT is a mid-solve
# anchor the ordinary decoder can be run against — the thing
# PATH_TO_VERIFICATION.md rev 1 wanted from scans, obtained for free from
# structure the solve already has.
#
# The decode then splits in two: a generatively-identified last layer, and a
# cross+F2L segment with BOTH endpoints pinned and roughly half the onsets.
# That is the same argument D3 (bidirectional) validated — splitting the
# true story's cost across two halves roughly doubles the surviving budget
# at the same beam.
#
# The empty peel (anchor = solved, at the last onset) is always among the
# hypotheses, so the baseline decode is always available and this can only
# lower the best total cost, never raise it.

MAX_PEEL = 12          # last layers deeper than this are not LBL solves
PEEL_BEAM = 400


def _f2l_faces(state: np.ndarray) -> np.ndarray:
    return RC.f2l_complete(state[None, :])[0]


def peel_backward(cost_rows: list[np.ndarray], del_costs: np.ndarray,
                  cands: list[dict], beam: int = PEEL_BEAM,
                  max_peel: int = MAX_PEEL) -> list[dict]:
    """
    Anchors: cube states the solve plausibly passed through at the moment
    the last layer began, each with the onset index it sits at, the cost of
    getting from there to solved, and the moves that do it.

    Search state is (onsets remaining, cube state) walking backwards from
    SOLVED. Three transitions, all of which must keep the abstract state
    inside the last layer:

      * peel a whole library algorithm off the tail (the point);
      * peel one ordinary turn, but ONLY a turn of the face whose first two
        layers are complete — that is AUF and cut-point regripping, and it
        is what keeps the peel from degenerating into a second general
        decoder (at most 2 options, not 12);
      * delete a phantom onset.

    Every state visited is emitted as an anchor, including the initial one,
    so the caller always has the do-nothing option.

    Each anchor also carries `n_algo` (how many library algorithms its chain
    peeled) and `max_nll` (the WORST per-move identification cost among
    them). The aggregate is a max rather than a mean because the chain is
    all-or-nothing: every link has to be the right word for the anchor state
    to be the right state, so one badly-identified link invalidates the
    anchor no matter how clean its neighbours are. Selection rules need this
    — peel COST cannot serve, since it mixes identification quality with how
    much was peeled and is dominated by the latter.
    """
    n = len(cost_rows)
    by_end: dict[int, list[dict]] = {}
    for c in cands:
        by_end.setdefault(c["start"] + c["k"], []).append(c)

    solved_key = RC.SOLVED.tobytes()
    frontier = {(n, solved_key): (0.0, RC.SOLVED.copy(), [], 0, 0.0)}
    anchors = {(n, solved_key): (0.0, RC.SOLVED.copy(), [], 0, 0.0)}

    for _ in range(max_peel * 3):
        nxt: dict = {}

        def push(j, state, cost, ops, depth, worst):
            if j < 0:
                return
            key = (j, state.tobytes())
            cur = nxt.get(key)
            if cur is None or cost < cur[0]:
                nxt[key] = (cost, state, ops, depth, worst)
            a = anchors.get(key)
            if a is None or cost < a[0]:
                anchors[key] = (cost, state, ops, depth, worst)

        for (j, _), (cost, state, ops, depth, worst) in frontier.items():
            faces = _f2l_faces(state)
            if not faces.any():
                continue          # left the last layer; nothing more to peel

            if depth < max_peel:
                for c in by_end.get(j, []):
                    vec = RC.SOLVED.copy()
                    for lab in c["labels"]:
                        vec = RC.compose(vec, RC.CLASS_VECS[lab])
                    prev = RC.compose(state, RC.inverse(vec))
                    if not (faces & _f2l_faces(prev)).any():
                        continue
                    push(c["start"], prev, cost + c["cost"],
                         [("algorithm", list(c["labels"]))] + ops, depth + 1,
                         max(worst, float(c.get("nll_per_move", 0.0))))

            if j == 0:
                continue
            # AUF / regrip: a quarter turn of a face whose F2L is complete.
            for fi in np.where(faces)[0]:
                face = RC.TP_NAME[fi]
                for k in (RC.WCA12.index(face), RC.WCA12.index(face + "'")):
                    prev = RC.compose(state, RC.inverse(RC.CLASS_VECS[k]))
                    if not (faces & _f2l_faces(prev)).any():
                        continue
                    push(j - 1, prev, cost + float(cost_rows[j - 1][k]),
                         [("move", k)] + ops, depth, worst)
            push(j - 1, state, cost + float(del_costs[j - 1]),
                 [("delete", None)] + ops, depth, worst)

        if not nxt:
            break
        frontier = dict(sorted(nxt.items(), key=lambda kv: kv[1][0])[:beam])

    out = [{"onset": j, "state": st, "cost": c, "ops": ops,
            "n_algo": na, "max_nll": mn}
           for (j, _), (c, st, ops, na, mn) in anchors.items()]
    out.sort(key=lambda a: (a["cost"], -a["onset"]))
    return out


def shortlist_anchors(anchors: list[dict], per_onset: int = 3) -> list[dict]:
    """
    Which anchors are worth paying a prefix decode for.

    Ranking the raw list by peel cost does not work, and the reason is
    structural rather than a tuning miss: peeling nothing costs nothing, and
    peeling one already-argmax turn also costs nothing (acceptance costs are
    relative to the argmax), so thousands of do-almost-nothing anchors sort
    ahead of every real one. Measured on a held-out session, the true
    F2L-boundary anchor sat at rank 42 by cost while the first 24 slots went
    to anchors that peeled zero algorithms.

    What the ordering has to reflect is that a DEEPER anchor leaves fewer
    onsets for the prefix decode, which is the entire benefit and is invisible
    in the peel cost. So: keep the cheapest few at each depth, drop anchors
    that some other anchor beats on both depth and cost, and let the totals
    arbitrate from there. The do-nothing anchor survives this by construction
    (nothing is shallower), which preserves the never-worse-than-baseline
    guarantee.
    """
    by_onset: dict[int, list[dict]] = {}
    for a in anchors:
        by_onset.setdefault(a["onset"], []).append(a)
    kept = []
    for j, group in by_onset.items():
        kept.extend(sorted(group, key=lambda a: a["cost"])[:per_onset])

    # Pareto frontier: drop `a` if some `b` is at least as DEEP and at least
    # as cheap. Deeper means a smaller onset index, so the scan must run
    # deepest-first with a running minimum. Running it shallowest-first
    # inverts the domination and collapses the whole frontier onto the
    # zero-cost do-nothing anchor — nothing deeper can ever beat 0.0.
    kept.sort(key=lambda a: (a["onset"], a["cost"]))
    front, best_cost = [], float("inf")
    for a in kept:
        if a["cost"] <= best_cost + 1e-9:
            front.append(a)
            best_cost = min(best_cost, a["cost"])
    front.sort(key=lambda a: (a["cost"], a["onset"]))
    return front


def anchor_moves(ops: list[tuple]) -> list[str]:
    """The WCA word an anchor's op list emits, chronologically."""
    out = []
    for kind, arg in ops:
        if kind == "algorithm":
            out.extend(RC.WCA12[k] for k in arg)
        elif kind == "move":
            out.append(RC.WCA12[arg])
    return out


# Worst per-move identification cost a peel chain may carry and still be
# trusted to REPLACE the plain decode's reading of the tail when nothing
# verified. Far tighter than ALGO_GATE (2.3, the "could this be an
# algorithm at all" screen), because the question is different: the gate
# asks whether a span may be offered to a search that will weigh it against
# everything else, this asks whether a span should be committed to with no
# further evidence. §7a measured real chunks at median nll/move 0.00-0.04
# against F2L decoys at 10-11, which suggested a wide safe interval.
#
# DEFAULT None = OFF, because it was measured and it LOSES (§9c/§9d).
# Wired up 2026-08-02, run as a fourth arm on seed 0's 12 held-out
# sessions, and the result was a 0.6-point daytime REGRESSION: it fired on
# 2 of 12 sessions, gained +0.9 on one evening take and lost 5.2 points on
# `solve_20260721_102711` — a session whose oracle headroom is exactly
# zero, so any anchor it picks there can only do harm. That is the +0.65
# ceiling and the -6.4 ungated rule of §9c showing up in deployment
# exactly as predicted.
#
# The machinery is kept because it is the honest measurement harness for
# any future selection rule (`best_effort_moves_trusted` is computed and
# scored whenever a value is passed), not because a better threshold is
# expected. Anything above ~0.4 fires on 102711. Do not re-enable without
# a two-seed sweep showing daytime non-negative.
TRUST_NLL = None


def _best_effort_anchor(efforts: list[tuple], trust_nll: float) -> tuple:
    """
    Which anchor's story to report when NOTHING verified — see §9b.

    Cost cannot arbitrate this and the reason is structural, not a tuning
    miss. On the verified path both stories explain every onset and reach
    solved, so costs are comparable. A best-effort story is under no such
    obligation: it accepts the argmax wherever it likes at cost 0 and is
    never charged for being wrong, so the do-nothing anchor is always
    cheapest and any cost-ranked rule re-derives the behaviour this is
    replacing.

    What arbitrates instead is the peel's own identification confidence —
    the evidence the verified path gets from reaching the anchor and this
    path does not have. Take the DEEPEST anchor whose every link identified
    cleanly; deeper means more of the sequence comes from the generative
    tail, which is the reliable half. Returns (anchor, result) or None,
    in which case the caller keeps the do-nothing story.
    """
    if trust_nll is None:                 # rule disabled: keep the old story
        return None
    trusted = [(a, r) for a, r in efforts
               if a.get("n_algo", 0) > 0
               and a.get("max_nll", 1e9) <= trust_nll]
    if not trusted:
        return None
    return min(trusted, key=lambda ar: ar[0]["onset"])


def decode_backward_first(start: np.ndarray, cost_rows: list[np.ndarray],
                          del_costs: np.ndarray, cands: list[dict],
                          max_anchors: int = 40,
                          trust_nll: float | None = TRUST_NLL,
                          **decode_kw) -> dict:
    """
    Peel the last layer generatively, then decode the remainder against the
    anchor it exposes. Returns a decode() result dict plus anchor diagnostics.

    Anchors are tried cheapest-first and the best TOTAL (peel + prefix) wins.
    The do-nothing anchor is always present, so this is never worse than
    decoding the whole sequence directly — that guarantee is the reason to
    prefer it over a heuristic that commits to one segmentation.

    When NOTHING verifies, two best-effort stories are returned rather than
    one, because they differ only in a selection rule and re-running the
    whole peel to compare them would double the cost for nothing:

      best_effort_moves          the deployed choice — the cheapest anchor,
                                 which in practice discards the peel and
                                 reduces to the plain decode (§9b)
      best_effort_moves_trusted  the deepest anchor whose every link
                                 identified below `trust_nll`, falling back
                                 to the same story when none qualifies

    Verified sessions are unaffected: both fields are the verified word.
    Neither best-effort story is a verification claim — the result carries
    `solved: False` and `reconstruct.py`'s endpoint check never ran on it.
    """
    anchors = shortlist_anchors(peel_backward(cost_rows, del_costs, cands))
    best, tried, efforts = None, 0, []
    for a in anchors:
        if tried >= max_anchors:
            break
        if best is not None and a["cost"] >= best["cost"]:
            break             # peel cost alone already exceeds the best total
        tried += 1
        # The PREFIX runs start -> anchor; the peel already covers
        # anchor -> solved. Getting this direction backwards decodes a
        # different problem that still "solves" and still asserts clean.
        res = RC.decode_between(start, a["state"],
                                cost_rows[:a["onset"]],
                                del_costs=del_costs[:a["onset"]],
                                **decode_kw)
        if not res["solved"]:
            # Collected in the same pass rather than by a second loop that
            # re-ran every decode: the fallback needs the identical results.
            if best is None and res.get("best_effort_moves") is not None:
                efforts.append((a, res))
            continue
        total = res["cost"] + a["cost"]
        if best is None or total < best["cost"]:
            tail = anchor_moves(a["ops"])
            full = res["moves"] + tail
            # `res` carries the PREFIX decode's own best_effort_moves, which
            # stop at the anchor. Overwriting "moves" without overwriting
            # that too silently scores the arm on a truncated word — it read
            # as a 59.7% "false verification" on a session that had in fact
            # decoded correctly, purely because 80 of 134 moves were being
            # compared against all 134.
            best = {**res, "cost": total, "moves": full,
                    "best_effort_moves": full, "best_effort_cost": total,
                    "best_effort_moves_trusted": full,
                    "anchor_onset": a["onset"], "anchor_cost": a["cost"],
                    "anchor_ops": [k for k, _ in a["ops"]],
                    "anchor_algos": sum(1 for k, _ in a["ops"]
                                        if k == "algorithm"),
                    "anchor_moves": len(tail)}
    if best is not None:
        return best

    # Nothing verified. Still produce the MAP story so post-decode accuracy
    # is measurable. `efforts` is in shortlist order — cheapest first, ties
    # broken deepest-first — so efforts[0] is whatever the previous
    # unconditional fallback would have picked; `trust_nll=None` disables
    # the new rule and reproduces the old behaviour exactly.
    fallback = {"solved": False, "cost": None, "moves": None,
                "n_anchors": len(anchors), "anchors_tried": tried}
    if not efforts:
        return fallback

    def story(ar):
        a, res = ar
        return res["best_effort_moves"] + anchor_moves(a["ops"])

    a, res = efforts[0]
    chosen = _best_effort_anchor(efforts, trust_nll)
    fallback.update({
        "best_effort_moves": story((a, res)),
        "best_effort_cost": res.get("best_effort_cost"),
        "best_effort_moves_trusted": story(chosen if chosen else (a, res)),
        "anchor_onset": a["onset"], "anchor_cost": a["cost"],
        "anchor_algos": sum(1 for k, _ in a["ops"] if k == "algorithm"),
        "anchor_moves": len(anchor_moves(a["ops"])),
        # Diagnostics for the trusted arm, so a flat result can be read as
        # "the rule never fired" vs "it fired and did not help".
        "trusted_fired": chosen is not None,
        "trusted_onset": chosen[0]["onset"] if chosen else a["onset"],
        "trusted_algos": (chosen[0]["n_algo"] if chosen else 0),
        "trusted_max_nll": (chosen[0].get("max_nll") if chosen else None)})
    return fallback


# --------------------------------------------------------------------------
# 4. The measurement
# --------------------------------------------------------------------------

def posteriorgram(d: Path, model, device, refresh: bool, tag: str):
    """Per-frame class posterior for a session, cached beside the session."""
    from model import score_stream_joint
    from dataset import JointSessionStream

    cache = d / f"ctc_post_{tag}.npz"
    stream = JointSessionStream(d / "detector_stream_color.npz")
    if cache.exists() and not refresh:
        z = np.load(cache)
        return z["class_prob"], stream
    _, class_prob, _ = score_stream_joint(model, stream, device)
    np.savez_compressed(cache, class_prob=class_prob.astype(np.float32))
    return class_prob, stream


def dirty_map(gt: list[str], pred: list[str]) -> list[bool]:
    """Per ground-truth move: did the plain decode get it wrong?"""
    bad, i = [False] * len(gt), 0
    for op, want, _ in align_sequences(gt, pred):
        if want is None:                      # phantom — blame the neighbour
            if i < len(gt):
                bad[i] = True
            continue
        if op != "ok":
            bad[i] = True
        i += 1
    return bad


def run(args):
    import torch
    import verify_joint as VJ
    from ctc_decode import prefix_beam_decode
    from model import build_joint_from_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ctc, map_location=device, weights_only=False)
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    tag = f"{Path(args.ctc).stem}_e{ckpt['epoch']}"
    train_names = set(ckpt.get("train_session_names", []))
    print(f"  checkpoint {args.ctc}  (epoch {ckpt['epoch']}, "
          f"{len(train_names)} train sessions)")

    dirs = [d for d in VJ.resolve_sessions(args.sessions)
            if (d / "detector_stream_color.npz").exists()]
    analysable = [(d, session_chunks(d)) for d in dirs]
    analysable = [(d, c) for d, c in analysable if c]
    if not analysable:
        sys.exit("No scramble-paired, solving sessions to analyse.")

    # The library is mined ONLY from sessions the acoustic model trained on.
    # Mining from the holdout would be reading the answer out of the prior —
    # the same circularity ALGORITHM_PRIOR.md §4 records.
    lib_dirs = [d for d, _ in analysable if d.name in train_names]
    lib_counts = build_library(lib_dirs)
    library = [w for w, _ in lib_counts.most_common()]
    print(f"  library mined from {len(lib_dirs)} training session(s): "
          f"{len(library)} distinct words, "
          f"{sum(lib_counts.values())} executions")
    for w, c in lib_counts.most_common(8):
        print(f"    {c:>3}x  ({len(w):>2}) {' '.join(w)}")

    rng = random.Random(args.seed)
    rows, decoys = [], []
    for d, chunks in analysable:
        gt = session_word(d)
        held_out = d.name not in train_names
        class_prob, stream = posteriorgram(d, model, device, args.refresh, tag)
        log_probs = np.log(np.maximum(class_prob, 1e-12))
        onset_idx = stream.onset_idx
        if len(onset_idx) != len(gt):
            print(f"  {d.name}: {len(onset_idx)} prepared onsets vs "
                  f"{len(gt)} truth moves — skipping")
            continue

        labels, _ = prefix_beam_decode(log_probs, beam=args.beam)
        pred = [RC.WCA12[c] for c in labels]
        bad = dirty_map(gt, pred)
        ll_start = chunks[0]["ll_start"]

        for ch in chunks:
            p, q = ch["span"]
            lo = int(onset_idx[p]) - args.pad
            hi = int(onset_idx[q - 1]) + args.pad + 1
            true_word = tuple(ch["word"])
            ranked = score_candidates(log_probs, lo, hi, library)
            in_lib = true_word in library
            rank = (next(i for i, r in enumerate(ranked)
                         if r["word"] == true_word) if in_lib else None)
            margin = (ranked[0]["logp"] - ranked[1]["logp"]
                      if len(ranked) > 1 else float("inf"))

            # Does identification survive the span being placed wrong? This
            # is the operating assumption, not a robustness nicety: in
            # deployment the boundary comes from a beam hypothesis, never
            # from onset_idx.
            jit, wide = {}, {}
            for dj in args.jitter:
                rj = score_candidates(log_probs, lo + dj, hi + dj, library)
                jit[str(dj)] = bool(in_lib and rj[0]["word"] == true_word)
            for k in args.widen:
                for mode, use_local in (("strict", False), ("local", True)):
                    rw = score_candidates(log_probs, lo - k, hi + k, library,
                                          local=use_local)
                    wide[f"{mode}{k}"] = bool(in_lib
                                              and rw[0]["word"] == true_word)

            rows.append({
                "session": d.name, "held_out": held_out,
                "span": [p, q], "len": q - p,
                "frames": [lo, hi], "in_library": in_lib,
                "rank": rank, "top1": bool(in_lib and rank == 0),
                "dirty": bool(any(bad[p:q])),
                "n_wrong": int(sum(bad[p:q])),
                "n_wrong_session": int(sum(bad)),
                "n_moves_session": len(gt),
                "nll_per_move": ranked[0]["nll_per_move"],
                "margin": margin, "jitter": jit, "widen": wide,
                "word": " ".join(true_word),
                "best": " ".join(ranked[0]["word"]),
            })

            # Null 1 — the same candidate set on an F2L span of equal length.
            for _ in range(args.decoys):
                if ll_start < 12:
                    break
                a = rng.randrange(2, ll_start - 2)
                b = min(a + (q - p), ll_start)
                if b - a < 3:
                    continue
                dlo = int(onset_idx[a]) - args.pad
                dhi = int(onset_idx[b - 1]) + args.pad + 1
                rk = score_candidates(log_probs, dlo, dhi, library)
                decoys.append({"kind": "f2l", "session": d.name,
                               "nll_per_move": rk[0]["nll_per_move"],
                               "margin": (rk[0]["logp"] - rk[1]["logp"]
                                          if len(rk) > 1 else float("inf"))})

            # Null 2 — the true span with its frames shuffled. Destroys move
            # ORDER while preserving every per-frame posterior exactly.
            perm = rng.sample(range(max(0, lo), hi), hi - max(0, lo))
            shuffled = log_probs[perm]
            rk = score_candidates(shuffled, 0, len(perm), library)
            decoys.append({"kind": "shuffled", "session": d.name,
                           "nll_per_move": rk[0]["nll_per_move"],
                           "margin": (rk[0]["logp"] - rk[1]["logp"]
                                      if len(rk) > 1 else float("inf"))})

        n_ll = sum(q - p for p, q in (c["span"] for c in chunks))
        print(f"  {d.name:<32} {'HELD OUT' if held_out else 'train':<9} "
              f"{len(chunks)} chunks, {n_ll} of {len(gt)} moves")

    report(rows, decoys, args)
    if args.out:
        json.dump({"rows": rows, "decoys": decoys,
                   "library": [" ".join(w) for w in library]},
                  open(args.out, "w"), indent=1)
        print(f"\n  wrote {args.out}")
    return rows, decoys


def _pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "    - "


def report(rows, decoys, args):
    def block(name, sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            print(f"  {name:<26}       -")
            return
        inlib = [r for r in sub if r["in_library"]]
        t1 = sum(r["top1"] for r in sub)
        print(f"  {name:<26} {len(sub):>4} {_pct(len(inlib), len(sub))} "
              f"{_pct(t1, len(sub))} {_pct(t1, max(len(inlib), 1))}")

    print(f"\n{'='*72}\n  IDENTIFICATION\n{'='*72}")
    print(f"  {'set':<26} {'n':>4} {'inLib':>6} {'top1':>6} {'top1|inLib':>10}")
    block("all chunks", lambda r: True)
    block("  dirty (decode wrong)", lambda r: r["dirty"])
    block("  clean (decode right)", lambda r: not r["dirty"])
    block("held-out sessions", lambda r: r["held_out"])
    block("  dirty, held out", lambda r: r["held_out"] and r["dirty"])
    block("  clean, held out", lambda r: r["held_out"] and not r["dirty"])

    print(f"\n{'='*72}\n  SPAN JITTER  (top-1 identification vs frames of "
          f"misplacement)\n{'='*72}")
    hdr = "  ".join(f"{d:>+5}" for d in args.jitter)
    print(f"  {'set':<26} {hdr}")
    for name, sel in (("all chunks", lambda r: True),
                      ("dirty", lambda r: r["dirty"]),
                      ("held out", lambda r: r["held_out"])):
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        cells = "  ".join(
            _pct(sum(r["jitter"][str(d)] for r in sub), len(sub))[:5].rjust(5)
            for d in args.jitter)
        print(f"  {name:<26} {cells}")

    print(f"\n{'='*72}\n  NULL  (does the candidate set fire on things that "
          f"are not algorithms)\n{'='*72}")
    real = np.array([r["nll_per_move"] for r in rows
                     if np.isfinite(r["nll_per_move"])])
    print(f"  {'population':<26} {'n':>4} {'median':>8} {'p90':>8} "
          f"{'min':>8}")
    print(f"  {'real LL chunks':<26} {len(real):>4} "
          f"{np.median(real):>8.2f} {np.percentile(real, 90):>8.2f} "
          f"{real.min():>8.2f}")
    for kind in ("f2l", "shuffled"):
        v = np.array([d["nll_per_move"] for d in decoys
                      if d["kind"] == kind
                      and np.isfinite(d["nll_per_move"])])
        if len(v):
            print(f"  {'null: ' + kind:<26} {len(v):>4} "
                  f"{np.median(v):>8.2f} {np.percentile(v, 90):>8.2f} "
                  f"{v.min():>8.2f}")
    if len(real):
        # An operating point is only usable if some threshold separates the
        # two populations; report the best achievable split rather than
        # asserting one.
        nulls = np.array([d["nll_per_move"] for d in decoys
                          if np.isfinite(d["nll_per_move"])])
        if len(nulls):
            best = max(((float(t),
                         (real <= t).mean(), (nulls <= t).mean())
                        for t in np.unique(np.round(
                            np.concatenate([real, nulls]), 2))),
                       key=lambda x: x[1] - x[2])
            print(f"\n  best nll/move threshold {best[0]:.2f}: "
                  f"accepts {100*best[1]:.1f}% of real chunks, "
                  f"{100*best[2]:.1f}% of nulls")

    print(f"\n{'='*72}\n  SPAN WIDENING  (span bracketing extra moves; "
          f"strict vs local CTC)\n{'='*72}")
    hdr = "  ".join(f"{k:>5}" for k in args.widen)
    print(f"  {'set / frames added each side':<28} {hdr}")
    for name, sel in (("all, strict", lambda r: True),
                      ("all, local", lambda r: True),
                      ("held out, strict", lambda r: r["held_out"]),
                      ("held out, local", lambda r: r["held_out"])):
        sub = [r for r in rows if sel(r)]
        mode = "local" if name.endswith("local") else "strict"
        if not sub:
            continue
        cells = "  ".join(
            _pct(sum(r["widen"][f"{mode}{k}"] for r in sub),
                 len(sub))[:5].rjust(5) for k in args.widen)
        print(f"  {name:<28} {cells}")

    print(f"\n{'='*72}\n  BUDGET  (what an identified chunk removes from the "
          f"decode)\n{'='*72}")
    for name, sel in (("all", lambda r: True),
                      ("held out", lambda r: r["held_out"])):
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        ident = [r for r in sub if r["top1"]]
        mv = sum(r["len"] for r in ident)
        wrong = sum(r["n_wrong"] for r in ident)
        # Session totals, counted once per session rather than per chunk —
        # the denominator that matters is the whole solve, not the part of
        # it this method already covers.
        sess = {r["session"]: r for r in sub}
        tot_mv = sum(r["n_moves_session"] for r in sess.values())
        tot_wrong = sum(r["n_wrong_session"] for r in sess.values())
        # Is error actually concentrated in the last layer under THIS
        # decoder? Measured on the peak-picking arm it was (78% of misses);
        # that must not be assumed to carry over to CTC, which specifically
        # removed phantoms. Re-derived here rather than recalled.
        in_mv = sum(r["len"] for r in sub)
        in_wr = sum(r["n_wrong"] for r in sub)
        out_mv, out_wr = tot_mv - in_mv, tot_wrong - in_wr
        print(f"  {name:<10} {len(ident):>3}/{len(sub):<3} chunks identified")
        print(f"  {'':<10} error rate inside LL chunks "
              f"{100*in_wr/max(in_mv,1):>5.1f}%  vs outside "
              f"{100*out_wr/max(out_mv,1):>5.1f}%")
        print(f"  {'':<10} {mv:>4} of {tot_mv:>5} ground-truth moves "
              f"collapsed into single transitions ({100*mv/tot_mv:.0f}%)")
        print(f"  {'':<10} {wrong:>4} of {tot_wrong:>5} mis-read moves fall "
              f"inside an identified chunk ({100*wrong/max(tot_wrong,1):.0f}%"
              f", ~{4.0 * wrong:.0f} cost units at C_INS)")


# --------------------------------------------------------------------------
# 5. Self-test
# --------------------------------------------------------------------------

def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # The abstraction must hold at cut points and break inside algorithms.
    _, _, fc, fe = piece_sets("U")
    sune = ["R", "U", "R'", "U", "R", "U", "U", "R'"]
    check("Sune restores F2L(U)", all_solved(RC.seq_to_state(sune), fc, fe))
    check("U preserves F2L(U)", all_solved(RC.seq_to_state(["U"]), fc, fe))
    check("R breaks F2L(U)", not all_solved(RC.seq_to_state(["R"]), fc, fe))
    states = trajectory(RC.SOLVED.copy(), sune)
    done = [all_solved(s, fc, fe) for s in states]
    check("Sune breaks F2L in flight",
          done[0] and done[-1] and not all(done))

    # CTC scoring.
    T, C = 40, 13
    lp = np.full((T, C), -20.0)
    lp[:, 12] = -0.001                      # confidently blank everywhere
    lp[10, 6], lp[20, 0] = -0.001, -0.001   # an R at t=10, a U at t=20
    lp = lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))
    ru = ctc_logp(lp, [6, 0])
    ur = ctc_logp(lp, [0, 6])
    check("CTC scores the right order higher", ru > ur)
    check("CTC prefers the observed word to a longer one",
          ru > ctc_logp(lp, [6, 0, 6]))
    check("too-short span is impossible",
          ctc_logp(lp[:1], [6, 0]) == float("-inf"))
    # Two identical labels need a blank between them.
    check("repeat needs a blank frame",
          ctc_logp(lp[:2], [6, 6]) == float("-inf"))
    # Marginalisation: a word whose second move was never emitted still
    # scores finitely, which is the property the whole approach needs.
    check("a dropped onset is not fatal", np.isfinite(ctc_logp(lp, [6, 0, 3])))

    # -- backward peel ---------------------------------------------------
    # A cube needing AUF + T-perm, with the detector dropping four onsets
    # inside the algorithm. Peeling backwards must recover the anchor
    # exactly, without ever seeing a correct prefix.
    tperm = ["R", "U", "R'", "U'", "R'", "F", "R", "R",
             "U'", "R'", "U'", "R", "U", "R'", "F'"]
    word = ["U"] + tperm
    labels = [RC.WCA12.index(m) for m in word]
    anchor_true = RC.inverse(RC.seq_to_state(word))
    kept = [i for i in range(len(word)) if i not in (3, 6, 9, 12)]
    rng = np.random.default_rng(5)

    def row(k):
        p = np.full(12, 0.02 / 11)
        p[k] = 0.98
        return RC.onset_costs(p)

    rows = [row(labels[i]) for i in kept]
    dcs = RC.score_del_costs([0.9] * len(kept), 0.25)
    cand = [{"start": 1, "k": len(kept) - 1,
             "labels": [RC.WCA12.index(m) for m in tperm], "cost": 2.0}]
    anchors = peel_backward(rows, dcs, cand)
    check("peel returns the do-nothing anchor",
          any(a["onset"] == len(rows) and (a["state"] == RC.SOLVED).all()
              for a in anchors))
    hit = [a for a in anchors if (a["state"] == anchor_true).all()]
    check("peel recovers the true F2L-boundary anchor", bool(hit))
    check("the true anchor sits at onset 0",
          bool(hit) and min(a["onset"] for a in hit) == 0)
    check("peel emits the dropped moves",
          bool(hit) and anchor_moves(
              min(hit, key=lambda a: a["cost"])["ops"]) == word)

    # It must not wander outside the last layer: a state with F2L broken
    # can neither be an anchor nor be reached by an AUF peel.
    broken = RC.compose(anchor_true, RC.CLASS_VECS[RC.WCA12.index("R")])
    check("peel never anchors on an F2L-broken state",
          not any((a["state"] == broken).all() for a in anchors))

    print(f"\n  {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ctc", help="CTC checkpoint, e.g. checkpoints/move_ctc_s0.pt")
    p.add_argument("--sessions", nargs="+",
                   default=["../training_data/solve_*/"])
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--pad", type=int, default=8,
                   help="frames of context added each side of a chunk span")
    p.add_argument("--jitter", type=int, nargs="+",
                   default=[-40, -20, -10, -5, 0, 5, 10, 20, 40])
    p.add_argument("--widen", type=int, nargs="+",
                   default=[0, 10, 20, 40, 80],
                   help="frames added to BOTH ends, so the span brackets "
                        "neighbouring moves — the realistic deployment case")
    p.add_argument("--decoys", type=int, default=2,
                   help="F2L decoy spans drawn per real chunk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refresh", action="store_true",
                   help="recompute cached posteriorgrams")
    p.add_argument("--out", default=None)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.ctc:
        sys.exit("--ctc is required (or --selftest)")
    run(args)


if __name__ == "__main__":
    main()
