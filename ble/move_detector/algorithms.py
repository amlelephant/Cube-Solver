"""
algorithms.py

Recognise known cube algorithms inside a detected move sequence, at every
orientation. Step 1 of the plan in ALGORITHM_PRIOR.md.

What this is for
----------------
`reconstruct.py` decodes against the start and end states, and that
constraint is GLOBAL and TERMINAL — it discriminates nothing until the last
onset, which is why the beam has to be 4000 wide. An algorithm match is a
LOCAL anchor: evidence about moves 40-52 that arrives at move 40.

Two payoffs, deliberately kept separate so they can be ablated separately:

  correction    a stretch the classifier read as `R U R' F'` with `U'` a
                close second on the last onset matches `R U R' U'` — the
                prior pushes that onset toward U' without ever forcing it.
  orientation   if matches across a session prefer one sigma, that is the
                camera-to-cube orientation; if the preference CHANGES
                partway through, that is an unrecorded whole-cube rotation,
                which the pipeline is otherwise blind to (see
                verify_solve.py's caveat).

Nothing here decides anything. The matcher's output is intended to reshape
`reconstruct.costs_from_moves`'s cost rows as a capped, uncertainty-scaled
prior — the same shape as the existing BLEND_INV blend — never to rewrite
the sequence before decoding. A pre-pass that hard-corrects moves commits
greedily and double-counts the classifier's evidence. That hook is step 2+;
this file is the matcher and its measurement.

Matching
--------
Everything runs off COST ROWS, one (12,) array per onset giving the
acceptance cost of each class (`reconstruct.onset_costs`). For algorithm
`a` at orientation `g` starting at onset `i`:

    score(i, a, g) = sum_j  cost_rows[i+j][ camera_class_for(g, a_j) ]

which is the negative log-likelihood, under the classifier's own softmax,
of having observed that algorithm there. Using the full softmax rather than
the argmax is the whole point — a near-miss with the right class in second
place is exactly the case worth catching.

Ground-truth words are matched through the same code by turning each word
into one-hot cost rows (`word_cost_rows`), so step 1 measures the real
matcher rather than a simplified stand-in.

Orientation and mirror
----------------------
The 24 orientations come from `reconstruct`'s own sigma registry
(`camera_class_for`), so sigma ids here mean the same thing they mean in
the decoder. Mirrors are NOT orientations — a left-handed F2L insert is a
different algorithm, not a rotated one — so they are a separate library
expansion (reflect L<->R, invert every turn; identical to
train_move_classifier's WCA_FLIP, and asserted against it below).

Half turns: the classifier's alphabet is the 12 quarter turns, and the
smart cube reports R2 as two events. Library entries are therefore expanded
to quarter turns, BOTH ways (R2 -> `R R` and `R' R'`), since either is a
legal execution and they are different observations.

Usage
-----
    python algorithms.py --step1 --sessions ../training_data/solve_*/
    python algorithms.py --step1 --sessions ../training_data/solve_*/ --library mined
    python algorithms.py --show-library

Run from inside move_detector/, same convention as the rest of the repo.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import reconstruct as RC

WCA12 = RC.WCA12
IDX = {n: i for i, n in enumerate(WCA12)}

# Mirror through the L-R plane: face positions swap L<->R, and every turn
# reverses handedness. Same map as train_move_classifier.WCA_FLIP (asserted
# in the selftest) — it is written out here so this module does not depend
# on the trainer, and because the derivation matters: a mirror preserves
# temporal ORDER, it only reflects space.
MIRROR = {"U": "U'", "U'": "U", "D": "D'", "D'": "D",
          "L": "R'", "R'": "L", "R": "L'", "L'": "R",
          "F": "F'", "F'": "F", "B": "B'", "B'": "B"}
MIRROR12 = np.array([IDX[MIRROR[n]] for n in WCA12])

# Cost charged for a class the cost row says is impossible. Only used when
# building one-hot rows from a known word; real rows come from the softmax.
BIG = 1.0


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
#
# Restricted to OUTER FACE turns, because that is the classifier's alphabet:
# no wide moves, no slice moves, no rotations. Algorithms normally written
# with r/M/x are either given in a face-only variant or left out — a library
# entry the pipeline can never observe is worse than no entry, since it only
# adds false-match surface.
#
# Entries are deduplicated modulo the 24 rotations and the mirror (see
# canonical_form), so `R U R' U'` and `L' U' L U` are one entry, not two.
# Redundant entries would not improve recall and would inflate the
# false-match rate.

CURATED: dict[str, str] = {
    # -- triggers and F2L ---------------------------------------------------
    "sexy":              "R U R' U'",
    "sexy-inv":          "U R U' R'",
    "sledgehammer":      "R' F R F'",
    "hedgeslammer":      "F R' F' R",
    "sexy-x2":           "R U R' U' R U R' U'",
    "sexy-x3":           "R U R' U' R U R' U' R U R' U'",
    "F2L-basic-right":   "U R U' R' U' F' U F",
    "F2L-basic-left":    "U' L' U L U F U' F'",
    "F2L-split-right":   "R U' R' U R U' R'",
    "F2L-3move":         "U R U' R' U R U' R'",

    # -- OLL (2-look and the common full cases that are face-only) ----------
    "OLL-cross-line":    "F R U R' U' F'",
    "OLL-cross-dot":     "F U R U' R' F'",
    "sune":              "R U R' U R U2 R'",
    "antisune":          "R U2 R' U' R U' R'",
    "double-sune":       "R U R' U R U' R' U R U2 R'",
    "pi":                "R U2 R2 U' R2 U' R2 U2 R",
    "headlights":        "R2 D R' U2 R D' R' U2 R'",
    "bowtie":            "F' L F L' U' L' U L",
    "corner-orient":     "R' D' R D",
    "corner-orient-x2":  "R' D' R D R' D' R D",
    "corner-orient-x4":  "R' D' R D R' D' R D R' D' R D R' D' R D",

    # -- PLL (face-only variants; M-slice perms like H and Z are omitted) ---
    "T-perm":            "R U R' U' R' F R2 U' R' U' R U R' F'",
    "Y-perm":            "F R U' R' U' R U R' F' R U R' U' R' F R F'",
    "Ja-perm":           "R' U L' U2 R U' R' U2 R L",
    "Jb-perm":           "R U R' F' R U R' U' R' F R2 U' R' U'",
    "Ra-perm":           "R U' R' U' R U R D R' U' R D' R' U2 R'",
    "Rb-perm":           "R2 F R U R U' R' F' R U2 R' U2 R",
    "Ua-perm":           "R U' R U R U R U' R' U' R2",
    "Ub-perm":           "R2 U R U R' U' R' U' R' U R'",
    "Aa-perm":           "R' F R' B2 R F' R' B2 R2",
    "Ab-perm":           "R2 B2 R F R' B2 R F' R",
    "F-perm":            "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R",
    "V-perm":            "R' U R' U' R D' R' D R' U D' R2 U' R2 D R2",
    "Na-perm":           "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'",
    "Ga-perm":           "R2 U R' U R' U' R U' R2 U' D R' U R D'",
    "niklas":            "U R U' L' U R' U' L",
    "corner-3cycle":     "R B' R F2 R' B R F2 R2",
}

# Which library the deployed prior should use. The mined library (see
# --library mined) is strictly for measuring the ceiling: it is derived from
# the recorded solves, so evaluating on those same solves measures nothing.
# Mining is restricted to non-holdout sessions for exactly that reason.
DEFAULT_LIBRARY = "curated"

# The cross-environment holdout used for every classifier comparison in this
# repo. Kept out of any mining so a mined-library number stays honest.
HOLDOUT_SESSIONS = {
    "solve_20260720_142006",
    "solve_20260721_102711",
    "solve_20260722_101225",
    "solve_20260723_105530_solve",
    "solve_20260724_100120_solve",
}


def parse_word(word: str, max_expansions: int = 16) -> list[list[int]]:
    """
    A WCA algorithm string -> every quarter-turn expansion, as class indices.

    `R2` is ambiguous as an OBSERVATION: the solver may perform it as two
    CW turns or two CCW turns, and the smart cube reports two events either
    way. Both are legal readings of the same algorithm, so both are
    generated. An algorithm with h half turns yields 2^h expansions, capped
    (nothing in the library has enough half turns to hit the cap).
    """
    expansions: list[list[int]] = [[]]
    for tok in word.split():
        face, suffix = tok[0], tok[1:]
        if suffix == "2":
            cw, ccw = IDX[face], IDX[face + "'"]
            expansions = [e + [cw, cw] for e in expansions] + \
                         [e + [ccw, ccw] for e in expansions]
            expansions = expansions[:max_expansions]
        else:
            k = IDX[tok]
            expansions = [e + [k] for e in expansions]
    return expansions


def all_sigmas() -> list[int]:
    """The 24 orientation ids, from reconstruct's own registry."""
    seen, frontier = {0}, [0]
    while frontier:
        nxt = []
        for s in frontier:
            for r in RC.ROT_NAMES:
                t = RC.rotate_sigma(s, r)
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
        frontier = nxt
    return sorted(seen)


_SIGMA_MAPS: dict[int, np.ndarray] | None = None


def sigma_maps() -> dict[int, np.ndarray]:
    """sigma id -> (12,) array taking a cube-frame class to a camera class."""
    global _SIGMA_MAPS
    if _SIGMA_MAPS is None:
        _SIGMA_MAPS = {
            s: np.array([RC.camera_class_for(s, k) for k in range(12)])
            for s in all_sigmas()
        }
    return _SIGMA_MAPS


def canonical_form(classes: list[int]) -> tuple:
    """
    Smallest representative of a word over all 24 rotations x 2 mirrors.

    Two library entries sharing a canonical form are the same algorithm seen
    from different angles. Carrying both cannot improve recall — the matcher
    already searches every orientation — and each redundant entry adds
    false-match surface, so the library is deduplicated on this.
    """
    arr = np.asarray(classes)
    best = None
    for m in (arr, MIRROR12[arr]):
        for cmap in sigma_maps().values():
            cand = tuple(int(x) for x in cmap[m])
            if best is None or cand < best:
                best = cand
    return best


@dataclass(frozen=True)
class Variant:
    """One library algorithm at one orientation, possibly mirrored."""
    name: str
    classes: tuple          # camera-frame class indices
    sigma: int
    mirrored: bool


def build_library(source: str = DEFAULT_LIBRARY,
                  session_words: dict[str, list[str]] | None = None,
                  min_len: int = 4,
                  mine_lengths=None) -> dict[str, list[int]]:
    """
    name -> cube-frame class sequence, deduplicated modulo rotation/mirror.

    Multi-expansion entries (half turns) become `name#0`, `name#1`, ... —
    they are genuinely different observations, so they are not deduplicated
    against each other.

    `mine_lengths` overrides `mine_library`'s n-gram length range (its
    default excludes length 4-5, e.g. the "sexy move", so a mined-vs-curated
    comparison at `min_len=4` is apples-to-oranges unless this is widened to
    match).
    """
    if source == "curated":
        raw = {}
        for name, word in CURATED.items():
            for e, classes in enumerate(parse_word(word)):
                raw[name if e == 0 else f"{name}#{e}"] = classes
    elif source == "mined":
        if not session_words:
            sys.exit("--library mined needs sessions to mine from.")
        raw = mine_library(session_words, **(
            {"lengths": mine_lengths} if mine_lengths is not None else {}))
    else:
        sys.exit(f"unknown library source {source!r}")

    out, seen = {}, {}
    for name, classes in raw.items():
        if len(classes) < min_len:
            continue
        canon = canonical_form(classes)
        if canon in seen:
            continue          # same algorithm from another angle
        seen[canon] = name
        out[name] = classes
    return out


def mine_library(session_words: dict[str, list[str]], min_count: int = 6,
                 max_entries: int = 40, lengths=range(12, 5, -1)
                 ) -> dict[str, list[int]]:
    """
    Recurring n-grams from the given sessions, longest first.

    This is the CEILING measurement, not a deployable library: it is derived
    from recorded solves and describes one person's habits. The caller is
    responsible for passing only non-holdout sessions — see HOLDOUT_SESSIONS
    and the honesty note in ALGORITHM_PRIOR.md §4.
    """
    from collections import Counter
    cands = []
    for n in lengths:
        c: Counter = Counter()
        for w in session_words.values():
            for i in range(len(w) - n + 1):
                c[tuple(w[i:i + n])] += 1
        cands += [g for g, v in c.items() if v >= min_count]

    chosen: list[tuple] = []
    for g in cands:
        # Skip anything already contained in a longer accepted entry.
        if any(_contains(long, g) for long in chosen):
            continue
        chosen.append(g)
        if len(chosen) >= max_entries:
            break
    return {f"mined-{i:02d}": [IDX[m] for m in g]
            for i, g in enumerate(chosen)}


def _contains(hay: tuple, needle: tuple) -> bool:
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def expand(library: dict[str, list[int]], mirrors: bool = True
           ) -> list[Variant]:
    """Every library entry at all 24 orientations (x2 with mirrors)."""
    maps = sigma_maps()
    out, seen = [], set()
    for name, classes in library.items():
        arr = np.asarray(classes)
        forms = [(arr, False)] + ([(MIRROR12[arr], True)] if mirrors else [])
        for base, mirrored in forms:
            for sid, cmap in maps.items():
                cls = tuple(int(x) for x in cmap[base])
                key = (name, cls)
                if key in seen:
                    continue    # a symmetric algorithm hits the same word twice
                seen.add(key)
                out.append(Variant(name, cls, sid, mirrored))
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class Match:
    start: int
    length: int
    name: str
    sigma: int
    mirrored: bool
    cost: float          # total acceptance cost over the window
    subs: int            # onsets where the algorithm disagrees with the argmax

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def mean_cost(self) -> float:
        return self.cost / self.length


def word_cost_rows(word: list[str]) -> np.ndarray:
    """
    A known move word -> (n, 12) one-hot cost rows.

    Lets ground-truth evaluation run through the identical matcher rather
    than a simplified stand-in: cost 0 for the move that happened, BIG for
    everything else, so a match's `cost` is exactly its substitution count.
    """
    rows = np.full((len(word), 12), BIG, dtype=np.float32)
    for i, m in enumerate(word):
        rows[i, IDX[m]] = 0.0
    return rows


# Substitution budget. Measured 2026-07-25 (see step1): a FLAT budget is
# the wrong shape, because it means completely different things at different
# lengths. Two tolerated errors in a 4-move trigger is a 50% error rate and
# matches almost any random word — at min_len=4 a flat budget of 1 takes the
# scramble-legal null from 3.8% to 27%. The same two errors in a 14-move
# T-perm is 14% and is exactly the case worth catching. So the budget scales
# with length and is capped:
#
#     budget(L) = min(MAX_SUBS, floor(L * SUB_RATE))
#
# giving 0 for L<7, 1 for 7<=L<14, 2 for L>=14.
MAX_SUBS = 2
SUB_RATE = 0.15


def sub_budget(L: int, max_subs: int = MAX_SUBS,
               sub_rate: float = SUB_RATE) -> int:
    return min(max_subs, int(L * sub_rate))


def match_algorithms(cost_rows: np.ndarray, variants: list[Variant],
                     max_subs: int = MAX_SUBS, sub_rate: float = SUB_RATE,
                     max_mean_cost: float | None = None) -> list[Match]:
    """
    Every window where a library variant explains the observations.

    Substitutions are onsets where the variant disagrees with the row's
    argmax; the budget per match is `sub_budget(L)` — see above for why it
    scales with length rather than being flat. Pass `sub_rate=0` for exact
    matching only.

    `max_mean_cost` additionally bounds the average -log-likelihood per
    move. That is the knob that matters once rows come from a real softmax
    rather than a known word: it lets a match spend its budget on onsets the
    classifier was UNSURE about instead of onsets it was confident about,
    which counting substitutions cannot distinguish.

    Vectorised over start positions, grouped by variant length.
    """
    C = np.asarray(cost_rows, dtype=np.float32)
    n = len(C)
    argmax = C.argmin(axis=1)
    out: list[Match] = []

    by_len: dict[int, list[Variant]] = {}
    for v in variants:
        by_len.setdefault(len(v.classes), []).append(v)

    for L, group in by_len.items():
        if L > n:
            continue
        cls = np.array([v.classes for v in group])            # (V, L)
        starts = n - L + 1
        score = np.zeros((starts, len(group)), dtype=np.float32)
        nsub = np.zeros((starts, len(group)), dtype=np.int32)
        for j in range(L):
            score += C[j:j + starts][:, cls[:, j]]
            nsub += (argmax[j:j + starts][:, None] != cls[None, :, j])

        ok = nsub <= sub_budget(L, max_subs, sub_rate)
        if max_mean_cost is not None:
            ok &= score <= max_mean_cost * L
        for i, v_i in zip(*np.where(ok)):
            v = group[v_i]
            out.append(Match(int(i), L, v.name, v.sigma, v.mirrored,
                             float(score[i, v_i]), int(nsub[i, v_i])))
    return out


def select_cover(matches: list[Match], n: int) -> list[Match]:
    """
    Greedy non-overlapping cover: cheapest first, longest as the tiebreak.

    Coverage is reported over a NON-OVERLAPPING selection deliberately. The
    raw match list double-counts — a T-perm window also contains a sexy move
    and several sub-triggers — so "fraction of moves explained" is only
    meaningful once each move is claimed at most once.
    """
    taken = np.zeros(n, dtype=bool)
    chosen = []
    for m in sorted(matches, key=lambda m: (m.mean_cost, -m.length, m.start)):
        if taken[m.start:m.end].any():
            continue
        taken[m.start:m.end] = True
        chosen.append(m)
    return sorted(chosen, key=lambda m: m.start)


def coverage(matches: list[Match], n: int) -> float:
    taken = np.zeros(n, dtype=bool)
    for m in matches:
        taken[m.start:m.end] = True
    return float(taken.mean()) if n else 0.0


# ---------------------------------------------------------------------------
# Step 1 — matcher precision on ground truth, no decoder involved
# ---------------------------------------------------------------------------
#
# The question this answers, and the one that gates everything after it:
# does the matcher find real structure at a false-match rate near zero?
#
# Two nulls, because they fail differently:
#   shuffled   the session's own moves in random order. Preserves the move
#              distribution exactly, so it controls for "this solver turns R
#              a lot". Can produce physically silly words (R R' R R').
#   random     a fresh scramble-legal word of the same length (no immediate
#              face repeats, no three-on-an-axis). Never silly, so it is the
#              HARDER null and the one to quote.

def _random_legal(n: int, rng: random.Random) -> list[str]:
    faces, axis = "UDRLFB", {"U": 0, "D": 0, "R": 1, "L": 1, "F": 2, "B": 2}
    out: list[str] = []
    while len(out) < n:
        f = rng.choice(faces)
        if out and f == out[-1][0]:
            continue
        if len(out) >= 2 and axis[f] == axis[out[-1][0]] == axis[out[-2][0]]:
            continue
        out.append(f + rng.choice(("", "'")))
    return out


def load_session_words(patterns: list[str]) -> dict[str, list[str]]:
    """session name -> BLE ground-truth WCA move list."""
    dirs = [Path(p) for pat in patterns
            for p in (Path(".").glob(pat) if "*" in pat else [Path(pat)])]
    out = {}
    for d in sorted(dirs):
        f = d / "moves.jsonl"
        if not f.is_file():
            continue
        w = [json.loads(l)["wca_notation"] for l in open(f) if l.strip()]
        if w and all(w):
            out[d.name] = w
    return out


def step1(args) -> None:
    words = load_session_words(args.sessions)
    if not words:
        sys.exit("No sessions with a usable moves.jsonl matched --sessions.")

    solves = {k: w for k, w in words.items() if len(w) >= args.min_solve}
    print(f"\n{'='*72}")
    print(f"  STEP 1 — matcher precision on BLE ground truth (no decoder)")
    print(f"{'='*72}")
    print(f"  sessions: {len(words)} loaded, {len(solves)} of solve length "
          f"(>= {args.min_solve} moves)")

    if args.library == "mined":
        train = {k: w for k, w in solves.items() if k not in HOLDOUT_SESSIONS}
        print(f"  library:  MINED from {len(train)} non-holdout session(s) — "
              f"a ceiling, not a\n            deployable library. "
              f"{len(HOLDOUT_SESSIONS & set(solves))} holdout session(s) "
              f"excluded from mining.")
        lib = build_library("mined", train, min_len=args.min_len)
        eval_on = {k: w for k, w in solves.items() if k in HOLDOUT_SESSIONS} \
            or solves
    else:
        print(f"  library:  CURATED ({len(CURATED)} algorithms, face turns "
              f"only) — independent of\n            these recordings, so it "
              f"can be evaluated on all of them.")
        lib = build_library("curated", min_len=args.min_len)
        eval_on = solves

    variants = expand(lib, mirrors=not args.no_mirrors)
    lengths = sorted({len(c) for c in lib.values()})
    print(f"            {len(lib)} entries after dedup modulo rotation/mirror,"
          f" lengths {lengths[0]}-{lengths[-1]}")
    print(f"            {len(variants)} variants "
          f"({'24 orientations x 2 mirrors' if not args.no_mirrors else '24 orientations'})")
    print(f"  evaluating on {len(eval_on)} session(s), "
          f"{sum(len(w) for w in eval_on.values())} moves total")

    rng = random.Random(args.seed)
    print(f"\n  {'max_subs':>8}  {'real':>7}  {'shuffled':>9}  {'random':>7}   "
          f"{'margin':>7}   {'algs/solve':>10}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*7}   {'-'*7}   {'-'*10}")

    rows = []
    for max_subs in args.max_subs:
        real, shuf, rand, counts = [], [], [], []
        for name, w in eval_on.items():
            n = len(w)
            sel = select_cover(
                match_algorithms(word_cost_rows(w), variants, max_subs), n)
            real.append(coverage(sel, n))
            counts.append(len(sel))

            s = list(w)
            rng.shuffle(s)
            shuf.append(coverage(select_cover(
                match_algorithms(word_cost_rows(s), variants, max_subs), n), n))

            r = _random_legal(n, rng)
            rand.append(coverage(select_cover(
                match_algorithms(word_cost_rows(r), variants, max_subs), n), n))

        m_real, m_shuf, m_rand = np.mean(real), np.mean(shuf), np.mean(rand)
        print(f"  {max_subs:>8}  {m_real*100:>6.1f}%  {m_shuf*100:>8.1f}%  "
              f"{m_rand*100:>6.1f}%   {(m_real-m_rand)*100:>+6.1f}   "
              f"{np.mean(counts):>10.1f}")
        rows.append({"max_subs": max_subs, "real": m_real,
                     "shuffled": m_shuf, "random": m_rand,
                     "per_session": dict(zip(eval_on, real))})

    _step1_verdict(rows, eval_on, variants, args)


def _step1_verdict(rows, eval_on, variants, args) -> None:
    best = max(rows, key=lambda r: r["real"] - r["random"])
    print(f"\n  Per-session coverage at max_subs={best['max_subs']}:")
    for name, cov in sorted(best["per_session"].items(),
                            key=lambda kv: -kv[1])[:args.show]:
        bar = "#" * int(cov * 40)
        print(f"    {name:<34} {cov*100:>5.1f}%  {bar}")
    zero = [k for k, v in best["per_session"].items() if v == 0.0]
    if zero:
        print(f"    ({len(zero)} session(s) at 0% — {', '.join(zero[:3])}"
              f"{' ...' if len(zero) > 3 else ''})")

    print(f"\n{'-'*72}")
    print(f"  VERDICT")
    print(f"{'-'*72}")
    fp = best["random"]
    print(f"  Best operating point: max_subs={best['max_subs']}, "
          f"coverage {best['real']*100:.1f}% vs {fp*100:.1f}% on the\n"
          f"  harder (scramble-legal) null — a margin of "
          f"{(best['real']-fp)*100:+.1f} points.")
    if fp > 0.05:
        print(f"\n  The false-match floor is NOT near zero. A prior built on "
              f"this would fire on\n  {fp*100:.0f}% of a random word, which "
              f"is noise injected straight into the cost\n  model. Tighten "
              f"the library (drop the short entries) or lower max_subs "
              f"before\n  going near step 2.")
    elif best["real"] < 0.15:
        print(f"\n  The floor is clean but coverage is too low to move the "
              f"decoder: at\n  {best['real']*100:.0f}% of moves the prior "
              f"touches too little of the sequence to pay\n  for its "
              f"complexity. Widen the library before step 2.")
    else:
        print(f"\n  PASSES. The floor is near zero and coverage is "
              f"material, so a capped prior\n  built on these matches can "
              f"only help where it fires. Proceed to step 2\n  "
              f"(ALGORITHM_PRIOR.md): reshape reconstruct.costs_from_moves's "
              f"rows and measure\n  raw per-move accuracy on the "
              f"cross-environment holdout.")
    print(f"\n  Reminder: this measures the matcher on GROUND-TRUTH words. "
          f"Real cost rows are\n  a softmax, so step 2 must re-tune the "
          f"operating point against max_mean_cost,\n  not max_subs.")


# ---------------------------------------------------------------------------
# Window matching (seed-and-extend)
# ---------------------------------------------------------------------------
#
# `match_algorithms` above requires a WHOLE library entry to fit the
# substitution budget. Measured 2026-07-25 that reaches only 21% of the
# classifier's errors, because recognising a 15-move algorithm needs all 15
# read well and an error is exactly where they are not.
#
# This is the fix, and it is the standard shape for the problem (seed and
# extend, as in BLAST / read alignment):
#
#   SEED    find a contiguous run where the classifier's argmax agrees with
#           some library variant EXACTLY. A 6-move exact run has a measured
#           false-match rate of 0.0% against scramble-legal words, so a seed
#           is near-certain evidence of both WHICH algorithm is being
#           performed and WHERE it is aligned.
#   EXTEND  having anchored identity and offset on the clean stretch, score
#           the whole overlap and keep it if agreement clears a threshold.
#           The disagreeing positions are then exactly the onsets worth
#           proposing corrections for — the dirty part the seed reached
#           into, which whole-algorithm matching could never accept.
#
# The two conditions do different jobs and both are needed. Agreement rate
# alone is far too weak a filter at this scale: ~4700 variants x ~160
# offsets is ~750k candidate alignments per session, so even a 1-in-10k
# false rate yields ~75 spurious matches. The exact-seed requirement is what
# supplies the statistical anchor; the agreement threshold is what decides
# how far out from it to trust.
#
# Sub-window matching also subsumes the old behaviour: a whole-algorithm
# match is just the offset where the overlap is the entire entry.

SEED_LEN = 6         # run required to anchor a match
MIN_OVERLAP = 8      # shortest overlap with a library entry worth scoring
MIN_AGREE = 0.0      # see MAX_COST_PER_MOVE — agreement rate is the wrong test

# The acceptance test, measured 2026-07-25. Mean acceptance cost per move
# (-log of the classifier's probability for the algorithm's reading,
# relative to its own argmax) separates true identifications from false ones
# essentially perfectly: every window whose reading the solver ACTUALLY
# performed scored <= 0.07, every wrong one >= 0.27.
#
# Agreement RATE — "60% of the moves match" — does not work, and the reason
# is worth keeping. It counts disagreements without asking whether the
# classifier was confident. A 15-move T-perm with one disagreement where the
# classifier's second choice was the T-perm's move (cost/move 0.06) is a
# true match and a free correction; one with three disagreements the
# classifier was sure about (cost/move 1.11) is a false match. Agreement
# rate ranks the second ABOVE the first (0.80 vs 0.93). Cost ranks them
# correctly, because it is uncertainty-weighted by construction.
MAX_COST_PER_MOVE = 0.25

# A seed position is one the classifier does not actively contradict, rather
# than one it agrees with outright. Requiring exact argmax agreement makes
# the anchor unable to span the very onsets worth correcting; this threshold
# admits a class holding roughly half the argmax's probability.
SEED_COST = 0.7


@dataclass
class WindowMatch:
    start: int           # first detected onset covered
    length: int          # overlap length
    name: str
    sigma: int
    mirrored: bool
    offset: int          # index within the library entry that `start` aligns to
    agree: int           # positions matching the classifier argmax
    seed: int            # longest exact run inside the overlap
    cost: float          # total acceptance cost of the algorithm's reading
    classes: tuple       # the algorithm's predicted classes over the overlap

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def agree_frac(self) -> float:
        return self.agree / self.length


def match_windows(cost_rows: np.ndarray, variants: list[Variant], *,
                  seed_len: int = SEED_LEN, min_overlap: int = MIN_OVERLAP,
                  min_agree: float = MIN_AGREE,
                  max_cost_per_move: float | None = MAX_COST_PER_MOVE,
                  seed_cost: float = SEED_COST) -> list[WindowMatch]:
    """
    Every (variant, alignment offset) whose overlap with the detected
    sequence clears `min_agree` AND contains an exact run of `seed_len`.

    Partial overlaps are included — an algorithm may run off either end of
    the detected sequence, and more importantly a FRAGMENT of a long
    algorithm is exactly what survives when the classifier botched the rest.

    Substitution-only alignment (fixed offset). A detector miss or phantom
    inside an algorithm shifts the frame and will truncate the match at that
    point rather than realigning; the seed-and-extend structure means the
    clean side is still found, which is the part that matters here.

    Vectorised over offsets, grouped by variant length.
    """
    C = np.asarray(cost_rows, dtype=np.float32)
    n = len(C)
    argmax = C.argmin(axis=1)
    out: list[WindowMatch] = []

    by_len: dict[int, list[Variant]] = {}
    for v in variants:
        by_len.setdefault(len(v.classes), []).append(v)

    for L, group in by_len.items():
        if min(L, n) < min_overlap:
            continue
        cls = np.array([v.classes for v in group])              # (V, L)
        V = len(group)
        pad = L - min_overlap            # how far the entry may hang off each end

        # Sentinel -1 never equals a class, so padded positions count as
        # both non-overlap and non-agreement.
        a_pad = np.full(n + 2 * pad, -1, dtype=np.int64)
        a_pad[pad:pad + n] = argmax
        starts = len(a_pad) - L + 1
        if starts <= 0:
            continue

        win = np.lib.stride_tricks.sliding_window_view(a_pad, L)  # (starts, L)
        agree = win[:, None, :] == cls[None, :, :]                # (starts,V,L)
        overlap = (win != -1)[:, None, :]                         # (starts,1,L)

        n_over = overlap.sum(axis=2)                              # (starts,1)
        n_agree = (agree & overlap).sum(axis=2)                   # (starts,V)

        # Per-position acceptance cost of the algorithm's reading. Padded
        # positions are free so they cannot inflate a partial overlap's cost.
        c_pad = np.zeros((len(a_pad), 12), dtype=np.float32)
        c_pad[pad:pad + n] = C
        cwin = np.lib.stride_tricks.sliding_window_view(
            c_pad, L, axis=0)                                     # (starts,12,L)
        pos_cost = np.take_along_axis(
            cwin[:, None, :, :], cls[None, :, None, :], axis=2
        )[:, :, 0, :]                                             # (starts,V,L)
        pos_cost = np.where(overlap, pos_cost, 0.0)
        total = pos_cost.sum(axis=2)                              # (starts,V)

        # Longest SEED run, vectorised: one pass along the entry. A seed
        # position is one the classifier does not contradict (cost below
        # seed_cost), not one it agrees with outright — see SEED_COST.
        run = np.zeros((starts, V), dtype=np.int16)
        best = np.zeros((starts, V), dtype=np.int16)
        ok_pos = (pos_cost <= seed_cost) & overlap
        for j in range(L):
            run = np.where(ok_pos[:, :, j], run + 1, 0)
            best = np.maximum(best, run)

        keep = ((n_over >= min_overlap) & (best >= seed_len) &
                (n_agree >= min_agree * n_over))
        if max_cost_per_move is not None:
            keep &= total <= max_cost_per_move * n_over
        for s_i, v_i in zip(*np.where(keep)):
            v = group[v_i]
            lo = max(0, int(s_i) - pad)          # first detected index covered
            hi = min(n, int(s_i) - pad + L)
            off = lo - (int(s_i) - pad)          # index within the entry
            seg = tuple(int(x) for x in cls[v_i, off:off + (hi - lo)])
            out.append(WindowMatch(lo, hi - lo, v.name, v.sigma, v.mirrored,
                                   off, int(n_agree[s_i, v_i]),
                                   int(best[s_i, v_i]),
                                   float(total[s_i, v_i]), seg))
    return out


def select_windows(matches: list[WindowMatch], n: int) -> list[WindowMatch]:
    """
    Greedy non-overlapping selection, best evidence first.

    Ranked by seed length, then agreement fraction, then overlap length —
    NOT by cost. Cost ranking is what biased the whole-algorithm selection
    away from windows containing an error (a match that must explain a
    mistake necessarily costs more), which is the opposite of what this is
    for: here a window that reaches into a dirty stretch is the valuable
    one, provided its seed says it is real.
    """
    taken = np.zeros(n, dtype=bool)
    chosen = []
    for m in sorted(matches, key=lambda m: (-m.seed, -m.agree_frac,
                                            -m.length, m.start)):
        if taken[m.start:m.end].any():
            continue
        taken[m.start:m.end] = True
        chosen.append(m)
    return sorted(chosen, key=lambda m: m.start)


# ---------------------------------------------------------------------------
# Indel-tolerant extension
# ---------------------------------------------------------------------------
#
# `match_windows` extends a seed at a FIXED offset: variant position j
# always maps to predicted position (window_start + j). Measured directly
# 2026-07-26 by pulling real algorithm occurrences (verified by an EXACT
# match against BLE ground truth) and checking what the deployed classifier
# actually produced there: every occurrence containing a detector MISS
# failed to match (4/4); nearly every one without a miss succeeded (4/5).
# That is a bigger driver of the 21% reach ceiling than classifier accuracy
# — a single dropped onset shifts every later predicted index by one, so
# the fixed-offset scorer reads the (correctly classified, merely
# shifted) tail as confidently WRONG rather than as unlabelled, and the
# window's mean cost blows past threshold even though most of it was read
# fine. Misses cluster on fast execution (median gap at a miss: 210ms vs
# 298ms at a correct call), and algorithms ARE the fast part of a solve
# (case-study spans ran 120-150ms), which is the mechanism behind the
# user's "it's the speed" intuition — just not via elevated per-move error
# rate (that stayed flat in/out of algorithm spans); via a single dropped
# onset being able to veto an otherwise-clean 15-move window entirely.
#
# The fix: a banded edit-distance alignment between the variant and the
# predicted array, so the offset can RESYNC across a miss/phantom instead
# of staying fixed. Indels get a COUNT budget, like MAX_SUBS/SUB_RATE, not
# a cost — there is no classifier signal at a missing onset to weigh, so
# "how many indels a window may absorb" has to be answered from the null
# false-match rate, not from a probability. Tune via the same discipline as
# SUB_RATE: sweep against the scramble-legal null and stop where it stops
# being ~0%.
#
# Two-tier for cost: try the existing fixed-offset score first (cheap,
# vectorised, unchanged); only pay for the banded DP when the fixed offset
# fails the mean-cost test — which is exactly the signature a miss/phantom
# produces, so most non-indel-affected candidates never reach the DP at
# all.

MAX_INDELS = 3
INDEL_RATE = 0.20


def indel_budget(L: int, max_indels: int = MAX_INDELS,
                 indel_rate: float = INDEL_RATE) -> int:
    return min(max_indels, int(L * indel_rate))


@dataclass
class IndelMatch:
    """
    One accepted (variant, alignment) pair, possibly re-synced across
    tolerated misses/phantoms. `covered` is SPARSE — a deletion consumes a
    variant position but no predicted index (nothing to propose there); an
    insertion consumes a predicted index but isn't part of the algorithm
    (nothing to propose there either). Only true substitution positions —
    a predicted onset actually scored against the algorithm's expected
    class — appear, which is exactly the set `apply_prior` would have
    anything to reshape for.
    """
    name: str
    sigma: int
    mirrored: bool
    cost: float                # total substitution cost over `covered`
    n_sub: int                  # len(covered)
    n_del: int                   # variant positions with no predicted onset
    n_ins: int                    # predicted onsets not part of the algorithm
    seed: int                      # longest consecutive clean (sub) run
    covered: tuple              # ((pred_idx, proposed_class), ...), sorted

    @property
    def start(self) -> int:
        return self.covered[0][0] if self.covered else -1

    @property
    def end(self) -> int:
        return self.covered[-1][0] + 1 if self.covered else -1

    @property
    def mean_cost(self) -> float:
        return self.cost / self.n_sub if self.n_sub else float("inf")


def _dp_align(cost_rows: np.ndarray, v_classes, lo: int, hi: int,
             budget: int, seed_cost: float):
    """
    Minimum-cost monotone alignment of `v_classes` (len L) against
    predicted positions [lo, hi) of `cost_rows`, tolerating up to `budget`
    combined insertions/deletions (a delete consumes a variant position
    with no predicted match; an insert consumes a predicted position that
    isn't part of the algorithm). Both cost 0 — there is no classifier
    signal to charge them, only the budget gates how many are allowed.

    Small by construction (L <= ~15, budget <= ~3), so a plain DP with
    traceback is fast enough per candidate; this is only called for
    candidates that already passed the vectorised seed gate and failed the
    cheap fixed-offset score, not for every (variant, offset) pair.

    Returns (cost, covered, seed_run, n_del, n_ins) for the best-cost path
    reaching the end of the variant, or None if [lo, hi) can't fit it even
    with the full budget.
    """
    L = len(v_classes)
    M = hi - lo
    if M < 0 or M + budget < L:
        return None
    INF = float("inf")
    D = np.full((L + 1, M + 1, budget + 1), INF)
    # Local alignment: the variant may start matching at ANY predicted
    # position within [lo, hi) for free. Without this, the DP is forced to
    # treat everything before the true start as leading insertions, which
    # burns the whole indel budget just skipping unrelated moves before the
    # algorithm and never reaches the misses actually inside it — this was
    # the bug that made the first version worse than the non-indel matcher.
    D[0, :, 0] = 0.0
    parent: dict = {}
    for j in range(L + 1):
        for m in range(M + 1):
            for k in range(budget + 1):
                if j == 0 and k == 0:
                    continue          # free base case, already set above
                best, bp = INF, None
                if j > 0 and m > 0:
                    c = float(cost_rows[lo + m - 1, v_classes[j - 1]])
                    cand = D[j - 1, m - 1, k] + c
                    if cand < best:
                        best, bp = cand, (j - 1, m - 1, k, "sub", c)
                if j > 0 and k > 0:
                    cand = D[j - 1, m, k - 1]
                    if cand < best:
                        best, bp = cand, (j - 1, m, k - 1, "del", 0.0)
                if m > 0 and k > 0:
                    cand = D[j, m - 1, k - 1]
                    if cand < best:
                        best, bp = cand, (j, m - 1, k - 1, "ins", 0.0)
                if bp is not None:
                    D[j, m, k] = best
                    parent[(j, m, k)] = bp

    best_end, best_cost = None, INF
    for m in range(M + 1):
        for k in range(budget + 1):
            if D[L, m, k] < best_cost:
                best_cost, best_end = D[L, m, k], (L, m, k)
    if best_end is None or best_cost == INF:
        return None

    ops = []
    state = best_end
    while state[0] != 0:
        pj, pm, pk, op, c = parent[state]
        ops.append((op, c, pj, pm))
        state = (pj, pm, pk)
    ops.reverse()

    covered, run, best_run, n_del, n_ins = [], 0, 0, 0, 0
    for op, c, pj, pm in ops:
        if op == "sub":
            covered.append((lo + pm, int(v_classes[pj]), c))
            if c <= seed_cost:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
        else:
            run = 0
            if op == "del":
                n_del += 1
            else:
                n_ins += 1
    return best_cost, covered, best_run, n_del, n_ins


def match_windows_indel(cost_rows: np.ndarray, variants: list[Variant], *,
                        seed_len: int = SEED_LEN, min_overlap: int = MIN_OVERLAP,
                        max_cost_per_move: float | None = MAX_COST_PER_MOVE,
                        seed_cost: float = SEED_COST,
                        max_indels: int = MAX_INDELS,
                        indel_rate: float = INDEL_RATE) -> list[IndelMatch]:
    """
    Like `match_windows`, but when the cheap fixed-offset score fails,
    retries with a banded indel-tolerant alignment before giving up. See
    the module comment above for why this exists and what it measures.
    """
    C = np.asarray(cost_rows, dtype=np.float64)
    n = len(C)
    argmax = C.argmin(axis=1)
    out: list[IndelMatch] = []

    by_len: dict[int, list[Variant]] = {}
    for v in variants:
        by_len.setdefault(len(v.classes), []).append(v)

    for L, group in by_len.items():
        if min(L, n) < min_overlap:
            continue
        budget = indel_budget(L, max_indels, indel_rate)
        pad = L - min_overlap + budget
        a_pad = np.full(n + 2 * pad, -1, dtype=np.int64)
        a_pad[pad:pad + n] = argmax
        c_pad = np.zeros((len(a_pad), 12), dtype=np.float64)
        c_pad[pad:pad + n] = C
        starts = len(a_pad) - L + 1
        if starts <= 0:
            continue

        win = np.lib.stride_tricks.sliding_window_view(a_pad, L)
        cwin = np.lib.stride_tricks.sliding_window_view(c_pad, L, axis=0)
        overlap = win != -1
        cls = np.array([v.classes for v in group])              # (V, L)

        ok_pos = np.zeros((starts, len(group), L), dtype=bool)
        for v_i in range(len(group)):
            pos_cost = cwin[:, cls[v_i], np.arange(L)]           # (starts, L)
            ok_pos[:, v_i, :] = (pos_cost <= seed_cost) & overlap

        run = np.zeros((starts, len(group)), dtype=np.int16)
        best = np.zeros((starts, len(group)), dtype=np.int16)
        for j in range(L):
            run = np.where(ok_pos[:, :, j], run + 1, 0)
            best = np.maximum(best, run)

        cand_s, cand_v = np.where(best >= seed_len)
        for s_i, v_i in zip(cand_s.tolist(), cand_v.tolist()):
            v = group[v_i]
            lo_full = max(0, s_i - pad)
            hi_full = min(n, s_i - pad + L)
            off = lo_full - (s_i - pad)
            seg = [int(x) for x in cls[v_i, off:off + (hi_full - lo_full)]]
            over = np.arange(lo_full, hi_full)
            n_over = len(over)
            fixed_cost = float(C[over, seg].sum()) if n_over else 0.0

            if n_over >= min_overlap and (max_cost_per_move is None or
                                          fixed_cost <= max_cost_per_move * n_over):
                covered = list(zip(over.tolist(), seg))
                run_len = best_run = 0
                for i, c_i in covered:
                    if C[i, c_i] <= seed_cost:
                        run_len += 1
                        best_run = max(best_run, run_len)
                    else:
                        run_len = 0
                out.append(IndelMatch(v.name, v.sigma, v.mirrored, fixed_cost,
                                      n_over, 0, 0, best_run, tuple(covered)))
                continue

            if budget == 0:
                continue
            dp_lo = max(0, lo_full - budget)
            dp_hi = min(n, hi_full + budget)
            res = _dp_align(C, v.classes, dp_lo, dp_hi, budget, seed_cost)
            if res is None:
                continue
            cost, covered3, seed_run, n_del, n_ins = res
            n_sub = len(covered3)
            if (n_sub >= min_overlap and seed_run >= seed_len and
                    (max_cost_per_move is None or
                     cost <= max_cost_per_move * n_sub)):
                covered = tuple((i, c) for i, c, _ in covered3)
                out.append(IndelMatch(v.name, v.sigma, v.mirrored, cost,
                                      n_sub, n_del, n_ins, seed_run, covered))
    return out


def select_windows_indel(matches: list[IndelMatch], n: int) -> list[IndelMatch]:
    """Greedy non-overlapping selection over the SPARSE covered indices."""
    taken = np.zeros(n, dtype=bool)
    chosen = []
    for m in sorted(matches, key=lambda m: (-m.seed, -m.n_sub, m.start)):
        idxs = [i for i, _ in m.covered]
        if not idxs or any(taken[i] for i in idxs):
            continue
        for i in idxs:
            taken[i] = True
        chosen.append(m)
    return sorted(chosen, key=lambda m: m.start)


# ---------------------------------------------------------------------------
# Step 1b — correction potential against REAL classifier softmaxes
# ---------------------------------------------------------------------------
#
# Coverage-vs-null (step 1a) measures whether matches are coincidental. It
# does NOT measure whether the prior would help, and on ground-truth words
# it structurally cannot: with one-hot rows built from the truth, an exact
# match agrees with the truth by construction, so the correction rate is
# trivially 100% and the corruption rate trivially 0%.
#
# The question that actually gates step 2 is: on onsets where the DEPLOYED
# classifier is wrong, does the matcher point at the right move — and on
# onsets where it is right, how often does the matcher drag it off?
#
# That needs real softmaxes, which <session>/reconstruct_replay.json already
# caches (full per-onset distributions from the deployed classifier). The
# predicted sequence is aligned to the BLE ground truth first, because a
# single missed onset shifts every later index and would turn one detector
# error into a hundred phantom classifier errors.

def load_replays(patterns: list[str], classifier: str | None = None) -> dict:
    """
    session -> {"probs": (n,12), "pred": [...], "gt": [...]} from cached
    replays. `classifier` filters to one checkpoint — mixing checkpoints
    would average two different error profiles into one meaningless number.
    """
    dirs = [Path(p) for pat in patterns
            for p in (Path(".").glob(pat) if "*" in pat else [Path(pat)])]
    out = {}
    for d in sorted(dirs):
        f, g = d / "reconstruct_replay.json", d / "moves.jsonl"
        if not (f.is_file() and g.is_file()):
            continue
        data = json.loads(f.read_text())
        if classifier and data["meta"].get("classifier") != classifier:
            continue
        if data.get("class_names") != WCA12:
            continue
        gt = [json.loads(l)["wca_notation"] for l in open(g) if l.strip()]
        if not gt or not all(gt) or not data["moves"]:
            continue
        out[d.name] = {
            "probs": np.array([m["probs"] for m in data["moves"]],
                              dtype=np.float64),
            "pred": [m["move"] for m in data["moves"]],
            "gt": gt,
        }
    return out


def align_pred_to_gt(gt: list[str], pred: list[str]) -> list[str | None]:
    """
    For each PREDICTED onset, the ground-truth move it aligns to (None if
    the alignment calls it a phantom). Wraps decode.align_sequences, which
    returns ops in order but not indices.
    """
    from decode import align_sequences
    labels: list[str | None] = []
    for op, truth, p in align_sequences(gt, pred):
        if op in ("ok", "sub"):
            labels.append(truth)
        elif op == "phantom":
            labels.append(None)
        # "miss" consumes a truth move with no prediction — no pred slot
    return labels


def step1b(args) -> None:
    reps = load_replays(args.sessions, classifier=args.replay_classifier)
    if not reps:
        print(f"\n  No cached replays from {args.replay_classifier} found — "
              f"skipping step 1b.\n  Generate them with: python "
              f"reconstruct.py --session ../training_data/solve_*/")
        return

    if args.library == "mined":
        words = load_session_words(args.sessions)
        train = {k: w for k, w in words.items()
                 if len(w) >= args.min_solve and k not in HOLDOUT_SESSIONS}
        if not train:
            sys.exit("--library mined needs non-holdout sessions to mine "
                      "from (check --sessions and --min-solve).")
        lib = build_library("mined", train, min_len=args.min_len,
                             mine_lengths=range(12, args.min_len - 1, -1))
        lib_desc = (f"MINED from {len(train)} non-holdout session(s) — "
                     f"this IS what the solver actually\n            "
                     f"performs, not a textbook guess; still evaluated on "
                     f"{len(reps)} replay session(s)\n            "
                     f"(holdout ones included — mining excluded them, "
                     f"evaluation does not)")
    else:
        lib = build_library("curated", min_len=args.min_len)
        lib_desc = f"CURATED ({len(CURATED)} algorithms, face turns only)"
    variants = expand(lib, mirrors=not args.no_mirrors)

    print(f"\n{'='*72}")
    print(f"  STEP 1b — correction potential on REAL classifier softmaxes")
    print(f"{'='*72}")
    print(f"  replays:  {len(reps)} session(s) from "
          f"{args.replay_classifier}")
    print(f"  library:  {lib_desc}")
    if args.use_indel:
        print(f"  matcher:  indel-tolerant seed-and-extend windows — seed >= "
              f"{args.seed_len} positions at cost <= {args.seed_cost},\n"
              f"            accepted at mean cost <= {args.max_cost}/move, "
              f"up to indel_budget(L) missed/phantom\n            onsets "
              f"tolerated inside the window (max {args.max_indels}, rate "
              f"{args.indel_rate})")
    else:
        print(f"  matcher:  seed-and-extend windows — seed >= {args.seed_len} "
              f"positions at cost <= {args.seed_cost},\n            accepted at "
              f"mean cost <= {args.max_cost}/move over the overlap")

    print(f"\n  {'session':<32} {'cov':>6} {'clf':>6} "
          f"{'fix':>5} {'break':>6} {'net':>5}  {'held-out':>8}")
    print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*5}  {'-'*8}")

    tot = {"fix": 0, "break": 0, "cov": 0, "n": 0, "wrong": 0, "reach": 0}
    held = {"fix": 0, "break": 0, "wrong": 0, "n": 0}
    for name, r in sorted(reps.items()):
        probs, pred, gt = r["probs"], r["pred"], r["gt"]
        rows = np.stack([RC.onset_costs(p) for p in probs])
        labels = align_pred_to_gt(gt, pred)
        n = len(pred)
        wrong = sum(1 for p, l in zip(pred, labels) if l and p != l)

        if args.use_indel:
            all_matches = match_windows_indel(
                rows, variants, seed_len=args.seed_len,
                seed_cost=args.seed_cost, max_cost_per_move=args.max_cost,
                max_indels=args.max_indels, indel_rate=args.indel_rate)
            matches = select_windows_indel(all_matches, n)
            cov = sum(len(m.covered) for m in matches) / n if n else 0.0

            spans = np.zeros(n, dtype=bool)
            for m in all_matches:
                for i, _ in m.covered:
                    spans[i] = True

            fix = brk = 0
            for m in matches:
                for i, cls in m.covered:
                    truth = labels[i]
                    if truth is None:
                        continue
                    proposed = WCA12[cls]
                    if proposed == pred[i]:
                        continue
                    if proposed == truth:
                        fix += 1
                    elif pred[i] == truth:
                        brk += 1
        else:
            all_matches = match_windows(rows, variants, seed_len=args.seed_len,
                                        seed_cost=args.seed_cost,
                                        max_cost_per_move=args.max_cost)
            matches = select_windows(all_matches, n)
            cov = coverage(matches, n)

            # Reach is measured over ALL matches, not the selected cover.
            # select_cover is greedy by cost and therefore biased AWAY from
            # windows containing an error (a match that must spend
            # substitution budget costs more than one that need not), so
            # scoring reach on the cover alone would blame the selection for
            # a property of the data.
            spans = np.zeros(n, dtype=bool)
            for m in all_matches:
                spans[m.start:m.end] = True

            fix = brk = 0
            for m in matches:
                for j in range(m.length):
                    i = m.start + j
                    truth = labels[i]
                    if truth is None:
                        continue    # phantom onset: no truth to score against
                    proposed = WCA12[m.classes[j]]
                    if proposed == pred[i]:
                        continue    # prior agrees with the classifier
                    if proposed == truth:
                        fix += 1    # prior points at the right move
                    elif pred[i] == truth:
                        brk += 1    # prior drags a correct onset off
                    # else: both wrong, no change in error count

        reached = sum(1 for i, (p, l) in enumerate(zip(pred, labels))
                      if l and p != l and spans[i])
        tot["reach"] += reached
        acc = 1 - wrong / max(sum(l is not None for l in labels), 1)
        is_held = name in HOLDOUT_SESSIONS
        print(f"  {name:<32} {cov*100:>5.1f}% {acc*100:>5.1f}% "
              f"{fix:>5} {brk:>6} {fix-brk:>+5}  {'yes' if is_held else '':>8}")
        tot["fix"] += fix; tot["break"] += brk
        tot["cov"] += cov; tot["n"] += 1; tot["wrong"] += wrong
        if is_held:
            held["fix"] += fix; held["break"] += brk
            held["wrong"] += wrong; held["n"] += 1

    print(f"\n  {'-'*72}")
    print(f"  Across {tot['n']} session(s): mean coverage "
          f"{tot['cov']/tot['n']*100:.1f}%, "
          f"{tot['wrong']} classifier errors total")
    # Reach is the ceiling on everything else: an error no match spans
    # cannot be corrected at any weight, so this bounds the whole idea
    # before any tuning is considered.
    print(f"    errors any match spans: {tot['reach']:>4}   "
          f"({tot['reach']/max(tot['wrong'],1)*100:.0f}% — the CEILING on "
          f"correction)")
    print(f"    corrections proposed : {tot['fix']:>4}   "
          f"({tot['fix']/max(tot['wrong'],1)*100:.0f}% of all errors)")
    print(f"    corruptions proposed : {tot['break']:>4}")
    print(f"    NET                  : {tot['fix']-tot['break']:>+4}")
    if held["n"]:
        print(f"  Classifier-held-out only ({held['n']} session(s)): "
              f"{held['fix']} fix / {held['break']} break "
              f"= {held['fix']-held['break']:+d} of {held['wrong']} errors")

    _step1b_verdict(tot, held)


def _variant_class(m: Match, j: int, variants: list[Variant]) -> int:
    """The class the matched variant predicts at offset j of the window."""
    for v in variants:
        if (v.name == m.name and v.sigma == m.sigma
                and v.mirrored == m.mirrored and len(v.classes) == m.length):
            return v.classes[j]
    raise KeyError(m.name)


def _step1b_verdict(tot: dict, held: dict) -> None:
    net, fix, brk = tot["fix"] - tot["break"], tot["fix"], tot["break"]
    reach = tot["reach"] / max(tot["wrong"], 1)
    print(f"\n{'-'*72}")
    print(f"  VERDICT")
    print(f"{'-'*72}")
    ratio = fix / brk if brk else float("inf")

    if reach < 0.35:
        print(f"  BLOCKED BY REACH, not by tuning. Only {reach*100:.0f}% of "
              f"classifier errors fall inside\n  ANY match window, so "
              f"{100-reach*100:.0f}% of them cannot be corrected at any "
              f"prior weight.\n")
        print(f"  This is self-selection and it is structural: recognising a "
              f"15-move algorithm\n  needs its neighbourhood read correctly, "
              f"and an error is precisely where the\n  neighbourhood is not. "
              f"Loosening the budget to reach them destroys the signal —\n"
              f"  the library starts matching everywhere, including random "
              f"words (see step 1a,\n  where a flat budget of 2 takes the "
              f"null past 85%).\n")
        print(f"  Do NOT proceed to step 2 for the CORRECTION payoff. It is "
              f"bounded above by\n  {reach*100:.0f}% of the classifier half "
              f"of the error budget, and the classifier half is\n  itself "
              f"the smaller one — detector miss/phantom is the majority of "
              f"remaining\n  error. The orientation payoff is independent "
              f"and must be judged separately.")
    elif fix == 0:
        print(f"  The matcher spans {reach*100:.0f}% of errors but proposes "
              f"no corrections — matches are\n  landing on stretches the "
              f"classifier already reads correctly. Widen the\n  library "
              f"before spending step 2 on it.")
    elif net <= 0:
        print(f"  The matcher breaks at least as much as it fixes "
              f"({fix} vs {brk}). A prior built\n  on this would be noise. "
              f"Do NOT proceed to step 2 — tighten the operating\n  point "
              f"(lower --sub-rate, raise --min-len) until the ratio is "
              f"clearly > 1.")
    elif ratio < 2:
        print(f"  Marginal: {fix} fixes vs {brk} breaks ({ratio:.1f}:1). A "
              f"capped prior would still\n  net out positive, but the margin "
              f"is thin enough that the decoder's own cost\n  model may "
              f"already handle these. Tighten the operating point and "
              f"re-measure\n  before spending step 2 on it.")
    else:
        print(f"  PASSES: {fix} corrections vs {brk} corruptions "
              f"({ratio:.1f}:1), net {net:+d}, spanning\n  {reach*100:.0f}% "
              f"of errors.")
        print(f"  A CAPPED prior only shifts an onset when the algorithm and "
              f"the classifier\n  disagree, so this ratio is the quantity "
              f"that decides step 2's sign. Note the\n  prior is soft: it "
              f"does not apply these edits, it reprices them, and the "
              f"decoder\n  can still override any of them on endpoint "
              f"evidence.")
    print(f"\n  Caveat: corrections are counted where the matched algorithm "
          f"DISAGREES with the\n  classifier. Onsets it already got right "
          f"and the algorithm agrees with are not\n  counted either way — "
          f"but they do not help the decode either, because\n  "
          f"onset_costs is relative to the argmax, so reinforcing the argmax "
          f"moves nothing.")


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def selftest() -> None:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"    {'PASS' if cond else 'FAIL'}  {msg}")
        ok &= bool(cond)

    print("\n  algorithms.py selftest")

    check(len(all_sigmas()) == 24, "24 orientations enumerated")

    # The mirror map must agree with the trainer's augmentation map, or the
    # library's mirrored variants mean something different from the mirrored
    # training samples.
    try:
        from train_move_classifier import WCA_FLIP
        check(all(MIRROR12[k] == WCA_FLIP[k] for k in range(12)),
              "MIRROR12 agrees with train_move_classifier.WCA_FLIP")
    except Exception as e:
        print(f"    SKIP  WCA_FLIP cross-check ({e})")

    # A mirror is an involution and is not any of the 24 rotations.
    check(all(MIRROR12[MIRROR12[k]] == k for k in range(12)),
          "mirror is an involution")
    sexy = [IDX[m] for m in "R U R' U'".split()]
    check(not any(tuple(cmap[sexy]) == tuple(MIRROR12[sexy])
                  for cmap in sigma_maps().values()),
          "mirror of sexy is not reachable by rotation")

    # Rotating a word and matching at that orientation must recover it.
    lib = {"sexy": sexy}
    variants = expand(lib)
    sid = RC.rotate_sigma(0, "y")
    rotated = [int(sigma_maps()[sid][k]) for k in sexy]
    rows = word_cost_rows([WCA12[k] for k in rotated])
    ms = match_algorithms(rows, variants, max_subs=0)
    check(any(m.sigma == sid and m.start == 0 for m in ms),
          "a y-rotated sexy move matches at sigma=y")

    # The length-scaled budget: short entries must stay exact however
    # generous the rate, because that is the whole point of scaling it.
    check(sub_budget(4) == 0 and sub_budget(6) == 0, "L<7 budget is 0")
    check(sub_budget(7) == 1 and sub_budget(13) == 1, "7<=L<14 budget is 1")
    check(sub_budget(14) == 2 and sub_budget(40) == 2,
          "L>=14 budget is 2 (capped)")
    check(sub_budget(4, sub_rate=0.9) == 2 and sub_budget(4, max_subs=0) == 0,
          "budget respects both cap and rate")

    # A corrupted 4-move trigger must NOT match (budget 0)...
    broken = list(rotated)
    broken[2] = RC.INV12[broken[2]]
    rows = word_cost_rows([WCA12[k] for k in broken])
    check(not match_algorithms(rows, variants),
          "1 error in a 4-move trigger: no match (budget 0)")

    # ...but the same single error inside a 14-move T-perm must be found.
    tperm = parse_word(CURATED["T-perm"])[0]
    # 14 written tokens, one of which is R2 -> 15 quarter turns.
    check(len(tperm) == 15, "T-perm expands to 15 quarter turns")
    tvar = expand({"T-perm": tperm})
    bad = list(tperm)
    bad[5] = RC.INV12[bad[5]]
    rows = word_cost_rows([WCA12[k] for k in bad])
    check(any(m.subs == 1 for m in match_algorithms(rows, tvar)),
          "1 error in a 14-move T-perm: found")
    check(not match_algorithms(rows, tvar, sub_rate=0.0),
          "same error rejected under exact matching")

    # -- window (seed-and-extend) matching ---------------------------------
    # A FRAGMENT of a long algorithm must match even when the rest of it is
    # unreadable. This is the case whole-algorithm matching cannot express,
    # and the reason window matching exists.
    half = [WCA12[k] for k in tperm[:8]] + ["D"] * 7
    wm = match_windows(word_cost_rows(half), tvar, seed_len=6,
                       max_cost_per_move=None, min_agree=0.0)
    check(any(m.name == "T-perm" and m.start == 0 and m.seed >= 8 for m in wm),
          "a T-perm fragment matches with the tail destroyed")
    check(not match_algorithms(word_cost_rows(half), tvar),
          "whole-algorithm matching rejects that same fragment")

    # The cost test must reject a window the classifier confidently
    # contradicts, and accept one it merely hesitated on.
    clean = word_cost_rows([WCA12[k] for k in tperm])
    check(match_windows(clean, tvar, max_cost_per_move=0.25),
          "a perfectly-read T-perm is accepted on cost")
    conf_wrong = clean.copy()
    conf_wrong[3] = 1.0                     # every class expensive but one
    conf_wrong[3, RC.INV12[tperm[3]]] = 0.0  # ...and it is the WRONG one
    check(not match_windows(conf_wrong, tvar, max_cost_per_move=0.05),
          "a confidently-contradicted position blocks acceptance")

    # Half turns expand both ways.
    check(len(parse_word("R2")) == 2 and
          [IDX["R"], IDX["R"]] in parse_word("R2"),
          "R2 expands to both R R and R' R'")

    # Dedup must collapse an algorithm against its own rotation.
    dup = build_library("curated", min_len=4)
    check(len(dup) == len({canonical_form(c) for c in dup.values()}),
          "library is deduplicated modulo rotation/mirror")

    print(f"\n  {'ALL PASS' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Recognise known algorithms in a detected move sequence")
    p.add_argument("--step1", action="store_true",
                   help="Run both halves of ALGORITHM_PRIOR.md step 1: "
                        "coverage vs null on ground truth, then correction "
                        "potential on real classifier softmaxes")
    p.add_argument("--step1b", action="store_true",
                   help="Only the correction-potential half (needs cached "
                        "reconstruct_replay.json files)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--show-library", action="store_true")
    p.add_argument("--sessions", nargs="+",
                   default=["../training_data/solve_*/"])
    p.add_argument("--library", choices=["curated", "mined"],
                   default=DEFAULT_LIBRARY,
                   help="curated = independent of these recordings (honest); "
                        "mined = recurring n-grams from non-holdout sessions, "
                        "evaluated on the holdout (a ceiling)")
    p.add_argument("--max-subs", type=int, nargs="+", default=[0, 1, 2],
                   dest="max_subs",
                   help="Step 1a only: FLAT substitution budgets to sweep, "
                        "to show why a flat budget is the wrong shape")
    p.add_argument("--max-subs-cap", type=int, default=MAX_SUBS,
                   help="Step 1b: cap on the length-scaled budget")
    p.add_argument("--sub-rate", type=float, default=SUB_RATE,
                   help="Step 1b: substitutions tolerated per move of "
                        "algorithm length (0 = exact matching only)")
    p.add_argument("--seed-len", type=int, default=SEED_LEN,
                   help="Step 1b: run length required to anchor a window")
    p.add_argument("--seed-cost", type=float, default=SEED_COST,
                   help="Step 1b: per-move cost a seed position may reach")
    p.add_argument("--max-cost", type=float, default=MAX_COST_PER_MOVE,
                   help="Step 1b: mean cost per move for an accepted window "
                        "— the test that actually separates true "
                        "identifications from false ones")
    p.add_argument("--replay-classifier", type=str,
                   default="../../legacy/move_classifier_rnd/"
                           "move_classifier_all39_cropped.pt",
                   help="Step 1b: score only replays cached from this "
                        "checkpoint — mixing checkpoints averages two "
                        "different error profiles. Archived 2026-08-03; "
                        "this whole module is a measured-and-rejected "
                        "recogniser approach, kept for the record "
                        "(ALGORITHM_PRIOR.md §6-8)")
    p.add_argument("--min-len", type=int, default=4,
                   help="Shortest library entry to use. Short entries drive "
                        "the false-match rate — raise this if the null is "
                        "not near zero")
    p.add_argument("--min-solve", type=int, default=60,
                   help="Move count below which a session is a scramble, not "
                        "a solve")
    p.add_argument("--use-indel", action="store_true",
                   help="Step 1b: use the indel-tolerant matcher instead of "
                        "the fixed-offset one — tolerates detector "
                        "misses/phantoms inside a window instead of letting "
                        "one desync (and reject) the whole match. See the "
                        "module comment above match_windows_indel")
    p.add_argument("--max-indels", type=int, default=MAX_INDELS,
                   help="Step 1b --use-indel: cap on the length-scaled indel "
                        "budget")
    p.add_argument("--indel-rate", type=float, default=INDEL_RATE,
                   help="Step 1b --use-indel: misses/phantoms tolerated per "
                        "move of algorithm length")
    p.add_argument("--no-mirrors", action="store_true")
    p.add_argument("--show", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.selftest:
        selftest()
    elif args.show_library:
        lib = build_library("curated", min_len=args.min_len)
        print(f"\n  {len(lib)} entries (deduplicated modulo rotation/mirror):")
        for name, classes in sorted(lib.items(), key=lambda kv: len(kv[1])):
            print(f"    {name:<22} {len(classes):>2}  "
                  f"{' '.join(WCA12[k] for k in classes)}")
    elif args.step1:
        step1(args)
        step1b(args)
    elif args.step1b:
        step1b(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
