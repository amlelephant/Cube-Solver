"""
ctc_decode.py

CTC (Connectionist Temporal Classification) decoding over the joint
model's per-frame posteriorgram, as the alternative to peak-picking.

What this replaces, and why
---------------------------
joint_decode.posteriorgram_to_moves calls decode.peak_pick, which commits
to a discrete candidate list BEFORE reconstruct.py's beam search ever
runs. Every phantom and every miss in the pipeline is created at that one
line: peak_pick needs a strict local maximum above a threshold, so a
regrip hump that clears the threshold becomes a phantom that the decoder
must then pay a deletion to remove, and two onsets within ~MIN_SEP frames
of each other can never both be reported at all (34% of sub-150ms pairs
arrive as a single merged hump — decode.py).

CTC removes the step rather than improving its input. The model emits per
frame a distribution over {12 quarter turns} + {blank}; a frame-level path
collapses to a move sequence by merging repeats then dropping blanks
(R R _ R U -> R R U); and the probability of a SEQUENCE is the sum over
every path that collapses to it. Decoding is a beam search over FRAMES,
not over candidates, so at each frame the search may extend the hypothesis
or stay in blank. "Is there a move here" is therefore decided INSIDE the
search, weighed against everything else the search knows, instead of being
decided greedily beforehand by a threshold.

Two properties that matter for this project specifically:

  * CTC never asks WHERE a move is, only what order the moves came in.
    ble-timestamp collisions (10.1% of moves share a 30ms BLE tick and are
    structurally unmatchable to a distinct frame) cost it nothing, where
    they are a hard defect for alignment-supervised training.
  * The collapse rule cannot emit two identical consecutive labels without
    an intervening blank frame: a true `R R` at 33ms separation decodes as
    a single `R`. This is a real and known CTC failure mode, not a bug
    here. Measured on the prepared sessions it is small — 19 of 3640
    onsets are same-face pairs within 2 frames — but it is a floor this
    decoder has and peak-picking does not.

Interchange format is unchanged: this produces the same list of
{"frame", "move", "conf", "probs", "score"} dicts live_detect.analyse()
and joint_decode.posteriorgram_to_moves produce, so reconstruct.py,
verify_solve.py and verify_joint.py consume it without modification.
"""

import numpy as np

from decode import move_time
from reconstruct import WCA12

BLANK = 12          # last column of the 13-way class head
NEG_INF = -1e30


def greedy_decode(log_probs: np.ndarray, blank: int = BLANK
                  ) -> tuple[list[int], list[int]]:
    """
    Best-path decode: per-frame argmax, then collapse. Returns
    (labels, frames). Fast, and the right thing for a training-loop metric;
    prefix_beam_decode is what should be used for a reported number, since
    best-path maximises path probability rather than SEQUENCE probability
    and those differ exactly when the model is unsure — which is the
    regime this whole exercise is about.
    """
    arg = log_probs.argmax(axis=1)
    labels, frames, prev = [], [], -1
    for t, c in enumerate(arg):
        if c != prev and c != blank:
            labels.append(int(c))
            frames.append(t)
        prev = int(c)
    return labels, frames


def prefix_beam_decode(log_probs: np.ndarray, blank: int = BLANK,
                       beam: int = 16, prune: float = -9.0,
                       lm=None, alpha: float = 0.0, beta: float = 0.0
                       ) -> tuple[list[int], list[int]]:
    """
    CTC prefix beam search, optionally with shallow LM fusion. Returns
    (labels, frames).

    LM fusion (move_lm.MoveLM), when `lm` is given:

        rank(prefix) = log P_ctc(prefix) + alpha * log P_lm(prefix)
                       + beta * |prefix|

    `alpha` weights the move-sequence prior; `beta` is a per-symbol
    insertion bonus. beta exists because any LM systematically prefers
    shorter sequences (every extra symbol multiplies in another
    probability < 1), so without it fusion trades phantoms for misses
    rather than reducing error. It is the direct knob on the
    insertion/deletion balance.

    The CTC accumulators (`pb`/`pnb`) stay PURE — the LM contribution
    enters only the ranking and the final argmax, never the forward
    recursion. Folding it into the recursion would make the stored
    quantity no longer a CTC marginal, and the repeat/blank algebra below
    depends on it being one.

    Why fusion here rather than as post-hoc repair: ALGORITHM_PRIOR.md §6b
    found the curated matcher reaches only 21% of errors because it must
    recognise an algorithm inside an already-corrupted sequence. Here the
    prior is consulted while competing prefixes are still alive, so its
    job is to make the corruption lose rather than to detect it after the
    fact.

    Scores label SEQUENCES by summing over the alignments that produce
    them, which is the thing best-path decoding approximates badly. Each
    beam entry carries the log-probability of its prefix split by whether
    the last frame was blank (`pb`) or not (`pnb`) — the split is what
    makes the repeat rule expressible: extending by a symbol equal to the
    prefix's last symbol only creates a NEW emission from the blank
    branch.

    Everything is in log space; over a 1600-frame session linear
    probabilities underflow to zero within a few hundred frames and the
    beam silently degenerates to whatever ties at 0.0.

    `frames` are approximate. A prefix is reached by many alignments and
    they disagree about which frame emitted which symbol, so each prefix
    keeps the frame list of the single largest contribution seen. Downstream
    (reconstruct.py) uses frames only for ordering and for reporting, never
    for cost, so this approximation does not enter any score.

    `prune` drops symbols whose log-probability at this frame is below it,
    which is what keeps the inner loop over 13 symbols from being 13x
    slower than it needs to be on the ~90% of frames that are confidently
    blank.
    """
    T = log_probs.shape[0]
    # prefix -> [log p(ends in blank), log p(ends in symbol), frames, lm]
    beams = {(): [0.0, NEG_INF, (), 0.0]}

    for t in range(T):
        row = log_probs[t]
        cand = np.where(row > prune)[0]
        if blank not in cand:
            cand = np.append(cand, blank)

        nxt: dict = {}          # prefix -> [lp_blank, lp_nonblank]
        src: dict = {}          # prefix -> (best single contribution, frames)
        lms: dict = {}          # prefix -> fused LM score (path-independent)

        def put(prefix, slot, contrib, frames, lm_score):
            """Accumulate one contribution, and remember the frame list of
            the largest SINGLE contribution seen for this prefix. Tracking
            frames separately from the accumulator matters: a prefix
            carried forward by a blank must keep its own emission frames,
            and it is reached after a fresh extension may already have
            created the entry with different ones."""
            e = nxt.get(prefix)
            if e is None:
                e = nxt[prefix] = [NEG_INF, NEG_INF]
                lms[prefix] = lm_score   # depends only on the prefix itself
            e[slot] = np.logaddexp(e[slot], contrib)
            b = src.get(prefix)
            if b is None or contrib > b[0]:
                src[prefix] = (contrib, frames)

        for prefix, (pb, pnb, frames, lm_score) in beams.items():
            ptot = np.logaddexp(pb, pnb)
            last = prefix[-1] if prefix else -1

            for c in cand:
                p = float(row[c])

                if c == blank:
                    put(prefix, 0, p + ptot, frames, lm_score)
                    continue

                c = int(c)
                if c == last:
                    # Same symbol with no blank between: still one emission.
                    put(prefix, 1, p + pnb, frames, lm_score)
                    contrib = p + pb          # a genuine repeat needs the blank
                else:
                    contrib = p + ptot

                ext_lm = lm_score
                if lm is not None:
                    ext_lm += alpha * lm.logp(c, prefix) + beta
                put(prefix + (c,), 1, contrib, frames + (t,), ext_lm)

        ranked = sorted(nxt.items(),
                        key=lambda kv: -(np.logaddexp(kv[1][0], kv[1][1])
                                         + lms[kv[0]]))[:beam]
        beams = {pfx: [e[0], e[1], src[pfx][1], lms[pfx]] for pfx, e in ranked}

    prefix, (pb, pnb, frames, lm_score) = max(
        beams.items(),
        key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]) + kv[1][3])
    return list(prefix), list(frames)


def ctc_to_moves(class_prob: np.ndarray, labels: list[int],
                 frames: list[int], fps: float = 30.0,
                 frame_times=None) -> list[dict]:
    """
    (labels, frames) from either decoder -> the standard moves list.

    `probs` on each move is the model's own renormalised 12-way posterior
    at the emitting frame, NOT a one-hot of the decoded label: the decoded
    label is this decoder's argmax, but reconstruct.costs_from_moves prices
    substitutions off the full distribution and flattening it here would
    throw away the only thing that lets the cube-state prior overrule a
    wrong call.

    `frame_times` (seconds, one per frame, ascending, index-aligned with
    `class_prob`) is the REAL capture clock and is preferred for `time`
    whenever the caller has it. Without it `time` falls back to
    `frame / fps`, where `fps` is a session mean — and webcam intervals
    jitter around that mean, so the two clocks disagree.

    See decode.move_time for the definition and for the measurement of how
    much the two clocks differ (short version: it cancels under differencing
    in the typical case and reaches 12% on hesitation seconds in the worst
    session, so it is a tail fix rather than a correction to the headline).
    """
    moves = []
    for c, f in zip(labels, frames):
        row = class_prob[f, :12].astype(np.float64)
        total = row.sum()
        probs = (row / total) if total > 1e-9 else np.full(12, 1 / 12)
        moves.append({"frame": int(f), "time": move_time(f, fps, frame_times),
                      "move": WCA12[c], "conf": float(probs[c]),
                      "probs": [float(p) for p in probs],
                      "score": float(1.0 - class_prob[f, BLANK])})
    return moves


def move_error_rate(pred: list[int], true: list[int]) -> tuple[float, dict]:
    """
    Levenshtein distance between two move sequences over the sequence
    length — the move-level analogue of word error rate, and the natural
    metric here because it is the only one that scores CTC and peak-picking
    on the same footing (CTC produces no aligned onsets, so onset F1 is not
    computable for it at all).

    Also returns the sub/ins/del split, because the whole question this
    experiment asks is whether phantoms (insertions) and misses (deletions)
    move, and an aggregate rate can hide one rising as the other falls.
    """
    n, m = len(true), len(pred)
    # dp[i][j] = (cost, sub, ins, del) for true[:i] vs pred[:j]
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)   # 0=match 1=sub 2=ins 3=del
    d[:, 0] = np.arange(n + 1)
    bt[1:, 0] = 3
    d[0, :] = np.arange(m + 1)
    bt[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if true[i - 1] == pred[j - 1]:
                d[i, j], bt[i, j] = d[i - 1, j - 1], 0
                continue
            opts = (d[i - 1, j - 1] + 1, d[i, j - 1] + 1, d[i - 1, j] + 1)
            k = int(np.argmin(opts))
            d[i, j] = opts[k]
            bt[i, j] = (1, 2, 3)[k]

    i, j = n, m
    sub = ins = dele = 0
    while i > 0 or j > 0:
        op = bt[i, j]
        if op == 0:
            i, j = i - 1, j - 1
        elif op == 1:
            sub += 1; i, j = i - 1, j - 1
        elif op == 2:
            ins += 1; j -= 1
        else:
            dele += 1; i -= 1
    return (int(d[n, m]) / max(n, 1),
            {"sub": sub, "ins": ins, "del": dele, "n_true": n, "n_pred": m})
