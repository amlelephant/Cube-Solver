"""
reconstruct.py

Group-theoretic decode of a noisy detected move sequence against known
start and end cube states — the step after decode.py in the pipeline:

    detector says WHEN  ->  classifier says WHICH  ->  this file says
    "which sequence is actually CONSISTENT with the cube getting solved"

Problem framing (noisy channel)
-------------------------------
We know the start state (scanned scramble), the end state (solved), and a
detected move sequence that may contain substitutions (classifier named
the wrong move — overwhelmingly the temporal inverse, U vs U'), phantoms
(spurious onset), and misses (onset never fired). The decode finds the
lowest-cost set of edits to the detected sequence such that applying the
edited sequence to the start state reaches the solved state. Costs are
-log probabilities, so "lowest cost" is a MAP estimate:

    accept class k at onset j   ln p'(top) - ln p'(k)   from the
                                classifier's full softmax, blended with a
                                small prior that concentrates on the
                                temporal inverse of the argmax (the
                                measured dominant confusion; the raw
                                softmax is overconfident exactly there)
    delete onset j (phantom)    C_DEL + DEL_SCORE_W * onset strength —
                                deleting a confidently-detected onset is
                                pricier than deleting one that barely
                                cleared the threshold. Without this the
                                beam drowns in "delete something, insert
                                something elsewhere" pairs: ~n^2*12 wrong
                                stories, parity-balanced, all cheaper
                                than a true multi-substitution story. Onset
                                score is real evidence against deletion;
                                using it prices that flood out (fp odds
                                at the F2-tuned operating point are ~7%
                                and concentrate at weak scores)
    insert a move (miss)        C_INS  (~ -ln of miss odds at ~98% recall,
                                plus the 12-way choice of which move)
    insert a rotation           C_ROT  (whole-cube rotations are invisible
                                to the camera labels; see below)

Why beam search and not exact IDA*
----------------------------------
At the measured classifier error rate (~6-50 wrong moves per session) an
exact IDA* over edit cost is intractable: the state constraint that kills
wrong hypotheses only pays off near the END of the sequence, so at the
budget needed for ~10 edits the tree explodes long before pruning bites.
Beam search over positions — the standard decoder for exactly this
noisy-channel shape in speech/OCR — keeps the K cheapest hypotheses per
onset, deduplicated by cube state, ranked by

    cost so far  +  admissible lookahead penalty

where the lookahead has three parts. Two are true lower bounds: a max
over pattern-database distances (BFS tables over the twist / flip /
udslice coordinates, quarter-turn metric) against the moves that can
still be applied, and a parity bound — every quarter turn flips
permutation parity, so a hypothesis whose parity cannot match any
achievable remaining move count provably needs a future insert/delete.

The third is a SUFFIX-CONSISTENCY ladder. Precompute the group product
S_j of the remaining detected moves (as classified); a hypothesis whose
story is consistent with the rest of the detections has rel = state*S_j
equal to the identity. Grading "almost consistent" with pattern
distances does NOT work — a conjugated error scrambles the coordinate
metrics, so h(rel) is ~20 +/- 3 for ANY hypothesis with >= 1 unfixed
error regardless of how many (measured with a truth-tracking harness
during development; feeding that noisy value into the ranking evicted
the true hypothesis by less than the noise amplitude). What does work is
EXACT algebra: rel is repairable by one future insertion of m at gap q
iff rel == S_q^-1 * m^-1 * S_q, by one future deletion of detection q
iff rel == S_{q+1}^-1 * d_q * S_{q+1}, and by one future substitution
m-for-d_q iff rel == S_{q+1}^-1 * (m^-1 d_q) * S_{q+1} — a set of ~25n
elements precomputed into a hash set. Ranking then adds rel_weight *
level, level 0 = consistent now, 1 = one future edit away, RESID_CAP =
everything else — noise-free, and the moment a hypothesis has repaired
everything except at most one remaining discrepancy it out-ranks the
cheap procrastinators that an absolute-cost beam would drown it under.
Never a hard prune.

The tail is exact: after the last onset, each candidate end state is
finished by IDDFS over words of length <= --max-end-ins, pruned by the
full pattern-database bound (including the corner-permutation table).
This covers end-of-solve moves the camera never saw. The insertion budget
being SMALL is what keeps the decode sound — every state is <= 20 moves
from solved, so with a big budget every branch "verifies" and the result
means nothing.

Whole-cube rotations
--------------------
The classifier emits camera-relative layers; an unrecorded y rotation
relabels every subsequent move. Rotation is therefore PERSISTENT search
state (one of 24 orientations), not a per-move edit: hypotheses carry a
camera-face -> cube-face mapping, and a single inserted rotation repairs
the whole suffix at one C_ROT instead of N substitutions. Off by default
(--rotations): the recorded sessions hold one orientation throughout, and
enabling it on rotation-free data only widens the search.

State is kept in the CUBE frame (centers define faces, as BLE reports);
camera moves are mapped through the orientation before being applied —
same convention as ble/orientation_tracker.py.

Modes
-----
    python reconstruct.py --selftest
    python reconstruct.py --synthetic 100 [--rotations] [--err-inv 0.06 ...]
    python reconstruct.py --session ../training_data/solve_*/

Session mode replays each recorded session through the deployed detector
+ classifier (cached to <session>/reconstruct_replay.json), builds the
start state from BLE ground truth (standing in for the scanned scramble),
decodes, and scores the reconstruction against the BLE move list. Run
from inside move_detector/, same convention as the rest of the repo.

Measured limits
---------------
On synthetic corruptions of a 40-move sequence (error onsets at conf
0.5-0.95, correct onsets 0.97+), a 512-wide beam recovers every mix of
up to ~6 errors tried — including 6 inverse flips, and a miss + inverse
+ phantom together. Two synthetic cases stay out of reach even at beam
32k: four errors of all three kinds interacting in one short window
(the true story loses an exact-tie among structurally identical
insertion placements that only the future can distinguish), and 8
substitutions in 40 moves (20% error rate — each fixed error re-doubles
the population of equally-priced alternatives). Those are information
limits of prefix decoding under this cost model, not bugs; if real
sessions hit them, the productive fix is a better classifier or
mid-solve state anchors, not more beam.

The binding quantity is the true story's total COST, not its edit count,
and synthetic noise flatters the decoder because its errors are cheap to
override. Measured 2026-07-22 at the deployed beam of 4000:

  * one missed move and nothing else, synthetic, recovered exactly at
    n = 20, 40, 60 and 90 — sequence length alone is not the limit;
  * that same missed move plus 1, 2 or 4 synthetic substitutions at
    n = 90, still recovered exactly;
  * but on the REAL solve_20260721_103149 replay, whose 4 genuine
    classifier errors give the true path a cost of 6.17, deleting one
    onset (true path 6.17 + C_INS = 10.17) is lost at beam 4000 AND at
    16000, and only found at 64000 — where it comes back at exactly
    10.17, confirming the beam, not the cost model, was the obstacle.

So: a true story costing ~6 survives 90 onsets at beam 4000; one costing
~10 does not. Since every insertion is a flat C_INS = 4.0, that is a
budget of roughly one insertion beyond whatever the classifier's mistakes
already cost. Quote the envelope in units of cost, and treat any envelope
figure measured on --synthetic as an upper bound on real performance.

Reading the results honestly
----------------------------
Two different failures look identical from the outside, so the report
separates them: if the decoder's solution costs MORE than the ground
truth path, the beam lost the truth (search failure — raise --beam); if
it costs LESS, the cost model itself prefers a wrong-but-cheaper story
(model failure — the fix is better calibration, not more search). And a
"solved" verdict alone is weak evidence: what matters is whether the
reconstruction matches the BLE move list, which is exactly what the
per-move accuracy column measures.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# The vendored pure-Python solver supplies the cubie-level group model.
_REPO = Path(__file__).resolve().parents[2]
_SOLVER_DIR = _REPO / "cv" / "solver"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

from twophase.cubes.cubiecube import CubieCube, MOVE_CUBE  # noqa: E402

from decode import align_sequences  # noqa: E402  (torch-free)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class order must match train_move_classifier.WCA_CLASS_NAMES; asserted
# against the classifier checkpoint in session mode. Kept local so that
# --selftest / --synthetic never import the torch stack.
WCA12 = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]
INV12 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]   # temporal inverse pairs

# twophase face indexing (pieces.Color): U R F D L B
TP_FACE = {"U": 0, "R": 1, "F": 2, "D": 3, "L": 4, "B": 5}
TP_NAME = "URFDLB"

# Edit costs — -log odds from the measured operating points (see decode.py
# sweep_threshold docstring: ~93% precision / ~98% recall at the deployed
# threshold). C_INS also carries the 12-way "which move" choice.
C_DEL = 2.6
C_INS = 4.0

# Cost of reading ONE onset as a middle-slice turn, i.e. as TWO layer moves
# instead of one (see SLICE_PARTNER below for why that is what a slice is).
#
# This is not an insertion and must not be priced like one. C_INS = 4.0 also
# carries the 12-way "which move" choice; a slice's second move is fully
# determined by its first, so that ~log(12) = 2.5 of the cost does not apply.
# What remains is the prior against a slice happening at all: 40 slice pairs
# in 4,641 recorded moves, so a slice reading should not be free either.
# Default 1.5 sits between those two arguments; --c-slice sweeps it.
C_SLICE = 1.5

# Minimum posterior mass on BOTH halves of a pair before that onset may be
# read as a slice at all.
#
# This gate is not an optimisation, it is what makes OP_SLICE usable. Allowed
# at every onset, a slice is 12 extra near-free transitions per step; the beam
# floods with equally-cheap nonsense and loses the true story even when the
# true story costs almost nothing (measured directly on
# solve_20260724_103307_solve: the intended path exists at cost 8.0 and the
# beam misses it at width 64000).
#
# The signal that fixes it comes from the training-label defect itself. A
# slice's two halves were labelled on heavily overlapping windows, so the
# classifier learned to hedge between them, and at a slice onset the
# posterior splits across the pair instead of concentrating. Measured
# 2026-07-30 over 3,527 matched onsets: partner-class mass has median 0.359
# at slice onsets against 0.00004 everywhere else — four orders of magnitude,
# p = 2e-23. At a 0.1 gate that is 33/33 slice onsets with 12/3494 false
# positives.
#
# Set to that measured operating point, NOT looser. An earlier 0.05 "for
# margin" cost a real regression: on solve_20260728_233139_scramble the gate
# opened at a merely UNCERTAIN onset (argmax F 0.539, partner B' 0.064) and
# the session stopped verifying. There is no margin to buy on the recall
# side — every observed slice onset sits at >= 0.1 with median 0.36 — so
# loosening only ever admits false positives.
SLICE_GATE = 0.10
C_ROT = 2.0
DEL_SCORE_W = 3.5    # extra deletion cost per unit of onset strength
DEL_FLOOR = 0.3      # D1 soft-onset lattice: near-free deletion cost for a
                     # sub-operating-point candidate near candidate_threshold
MAX_END_INS = 4      # end-of-solve moves the camera never saw
BEAM = 4000
MAX_CHAIN = 2        # consecutive mid-sequence insertions allowed per gap
REL_WEIGHT = 2.0     # ranking weight per unit of (capped) suffix residual
RESID_CAP = 3        # see module docstring: the residual is binary-ish;
                     # capping turns its garbage plateau into a constant

# Softmax blending. Raw -log p prices overriding an overconfident
# softmax like a miracle, so reserve some prior mass for the inverse of
# the argmax (the dominant measured confusion) and a little uniform mass
# for everything else. Both are scaled by the onset's UNCERTAINTY
# (1 - p_top): the measured error profile on real replays is that wrong
# calls sit at conf ~0.5-0.7 while correct ones sit at ~0.99+, so an
# uncertain onset gets a cheap inverse flip (~0.5-1.5) while a confident
# onset stays expensive to override (~5.5+). An earlier constant-mass
# version handed EVERY onset a ~2.2-cost inverse flip, which turned each
# confident correct onset into a decoy factory — combinatorially
# swamping the beam on hard cases.
BLEND_INV = 0.20
BLEND_UNIF = 0.04
BLEND_ADJ = 0.0      # D2 (PATH_TO_VERIFICATION.md §5): prior mass reserved
                     # for the 8 adjacent-face classes, scaled by the same
                     # onset uncertainty as BLEND_INV. 0.0 (off) until a
                     # session-measured value is passed via --blend-adj —
                     # see confusion_breakdown() and its --confusion CLI
                     # mode, which derives it from classifier-unseen
                     # sessions rather than guessing it.

# State vector layout: [cp(0:8) | co(8:16) | ep(16:28) | eo(28:40)], int8.
_CP = slice(0, 8)
_CO = slice(8, 16)
_EP = slice(16, 28)
_EO = slice(28, 40)

TABLES_CACHE = Path(__file__).parent / "cache" / "reconstruct_tables.npz"


# ---------------------------------------------------------------------------
# Cube group model (numpy, batched)
# ---------------------------------------------------------------------------

def _cubie_to_vec(c: CubieCube) -> np.ndarray:
    return np.array([int(x) for x in c.cp] + [int(x) for x in c.co] +
                    [int(x) for x in c.ep] + [int(x) for x in c.eo],
                    dtype=np.int8)


def _vec_to_cubie(v: np.ndarray) -> CubieCube:
    return CubieCube(cp=[int(x) for x in v[_CP]],
                     co=[int(x) for x in v[_CO]],
                     ep=[int(x) for x in v[_EP]],
                     eo=[int(x) for x in v[_EO]])


SOLVED = _cubie_to_vec(CubieCube())


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Group product "a then b" using the vendored solver's replaced-by
    convention: (a*b).cp[i] = a.cp[b.cp[i]] — see CubieCube.corner_multiply.
    """
    out = np.empty(40, dtype=np.int8)
    bcp, bep = b[_CP], b[_EP]
    out[_CP] = a[_CP][bcp]
    out[_CO] = (a[_CO][bcp] + b[_CO]) % 3
    out[_EP] = a[_EP][bep]
    out[_EO] = (a[_EO][bep] + b[_EO]) % 2
    return out


def apply_batch(states: np.ndarray, mv: np.ndarray) -> np.ndarray:
    """compose(each state, mv) vectorised over an (N, 40) batch."""
    out = np.empty_like(states)
    mcp, mep = mv[_CP], mv[_EP]
    out[:, _CP] = states[:, _CP][:, mcp]
    out[:, _CO] = (states[:, _CO][:, mcp] + mv[_CO]) % 3
    out[:, _EP] = states[:, _EP][:, mep]
    out[:, _EO] = (states[:, _EO][:, mep] + mv[_EO]) % 2
    return out


def inverse(v: np.ndarray) -> np.ndarray:
    return _cubie_to_vec(_vec_to_cubie(v).inverse_cubiecube())


def _build_move_vecs():
    """
    MOVE_FP[face][power] for power 1..3, and the 12 classifier classes.
    """
    move_fp = {}
    for f in range(6):
        m1 = _cubie_to_vec(MOVE_CUBE[f])
        m2 = compose(m1, m1)
        m3 = compose(m2, m1)
        move_fp[f] = {1: m1, 2: m2, 3: m3}
    class_vecs = np.stack([
        move_fp[TP_FACE[name[0]]][3 if name.endswith("'") else 1]
        for name in WCA12
    ])
    return move_fp, class_vecs


MOVE_FP, CLASS_VECS = _build_move_vecs()
CLASS_FACE = np.array([TP_FACE[n[0]] for n in WCA12])
CLASS_POWER = np.array([3 if n.endswith("'") else 1 for n in WCA12])

# Face-adjacency structure for D2's confusion-calibrated prior. URFDLB
# pairs opposite faces exactly 3 apart (U-D, R-L, F-B), so opposite(f) =
# (f+3)%6. For predicted class p, the 8 "adjacent-face" classes are the
# ones on neither p's face nor its opposite — the classifier's measured
# live error axis (classifier-error-axis-live memory), distinct from the
# 1 same-face inverse class BLEND_INV already covers.
ADJ_CLASSES = [np.array([k for k in range(12)
                         if CLASS_FACE[k] != CLASS_FACE[p]
                         and CLASS_FACE[k] != (CLASS_FACE[p] + 3) % 6])
              for p in range(12)]

# ---------------------------------------------------------------------------
# Middle-slice turns
# ---------------------------------------------------------------------------
#
# A slice turn (M/E/S) rotates the core, so the four centres on that axis move
# with it. The smart cube reports face rotation RELATIVE to the core, so what
# it emits for one M is the PAIR `R` + `L'` — the two outer layers, which did
# not move in space, appear to have turned backwards once the core rotated
# under them. That pair is a state-correct description of M relative to the
# centres, which is why every session containing slices still passes
# session_check.py's endpoint gate.
#
# Measured 2026-07-30 (GROUND_TRUTH_ARTIFACTS.md): 46 of 46 same-axis
# opposite-face pairs within 200ms carry exactly this signature — one primed,
# one not — and 31 of them share a BLE timestamp exactly. The camera sees ONE
# motion, so the detector can only ever produce ONE onset for them, and
# `decode.MIN_SEP` forbids two peaks that close in any case. Without the
# OP_SLICE transition below, the second half of every slice is an insertion
# the decoder must pay C_INS for, and that charge alone accounts for the
# entire gt_path_cost of several sessions.
#
# SLICE_PARTNER[k] is the class that accompanies k: same axis, opposite face,
# opposite handedness. The two commute (different layers about one axis), so
# the composition is the same whichever half the classifier happens to report.
SLICE_PARTNER = np.array([
    WCA12.index(TP_NAME[(TP_FACE[n[0]] + 3) % 6]
                + ("" if n.endswith("'") else "'"))
    for n in WCA12
])

# Cube-frame state change of reading class k as a slice: k then its partner.
SLICE_VECS = np.stack([compose(CLASS_VECS[k], CLASS_VECS[SLICE_PARTNER[k]])
                      for k in range(12)])


def slice_mask(probs: np.ndarray, gate: float = SLICE_GATE) -> np.ndarray:
    """
    (12,) bool: may this onset be read as a slice starting from class k?

    True only where the posterior puts real mass on BOTH halves of the pair,
    which is what a slice looks like to a classifier trained on overlapping
    windows for the two halves (see SLICE_GATE). A confident single turn puts
    ~0 on the opposite face and gates out entirely.
    """
    p = np.asarray(probs, dtype=np.float64)
    return (p > gate) & (p[SLICE_PARTNER] > gate)


def seq_to_state(names: list[str]) -> np.ndarray:
    """Product of a WCA move-name sequence, applied left to right."""
    s = SOLVED.copy()
    for n in names:
        s = compose(s, CLASS_VECS[WCA12.index(n)])
    return s


def start_from_gt(gt_names: list[str]) -> np.ndarray:
    """
    The state the cube must have been in so that the ground-truth sequence
    ends solved. Stands in for the scanned scramble in these evaluations —
    valid whether or not the physical session actually ended solved,
    because the GT moves transform it to solved BY CONSTRUCTION.
    """
    return inverse(seq_to_state(gt_names))


# ---------------------------------------------------------------------------
# Orientation (camera face -> cube face), for whole-cube rotations
# ---------------------------------------------------------------------------
#
# sigma[camera_face] = cube face currently at that camera position.
# After a rotation r, the face now at camera position c is the one that
# was at position RHO[r][c] before, so sigma' = sigma[RHO[r]].

ROT_NAMES = ["x", "x'", "y", "y'", "z", "z'"]


def _build_rotations():
    U, R, F, D, L, B = range(6)
    rho = {}
    x = np.arange(6)
    x[[F, U, B, D]] = [D, F, U, B]   # x: front comes from down
    y = np.arange(6)
    y[[F, R, B, L]] = [R, B, L, F]   # y: front comes from right
    z = np.arange(6)
    z[[U, R, D, L]] = [L, U, R, D]   # z: up comes from left
    for name, base in (("x", x), ("y", y), ("z", z)):
        rho[name] = base
        inv = np.empty(6, dtype=int)
        inv[base] = np.arange(6)
        rho[name + "'"] = inv
    return rho


RHO = _build_rotations()

# Orientation registry: id 0 is the identity mapping. Extended lazily as
# the search discovers new orientations.
_SIGMAS: list[tuple] = [tuple(range(6))]
_SIGMA_IDX: dict[tuple, int] = {tuple(range(6)): 0}


def sigma_id(t: tuple) -> int:
    if t not in _SIGMA_IDX:
        _SIGMA_IDX[t] = len(_SIGMAS)
        _SIGMAS.append(t)
    return _SIGMA_IDX[t]


def rotate_sigma(sid: int, rot: str) -> int:
    s = np.array(_SIGMAS[sid])
    return sigma_id(tuple(int(x) for x in s[RHO[rot]]))


def move_for(sid: int, k: int) -> np.ndarray:
    """Cube-frame move vector for camera-relative class k under orientation."""
    return MOVE_FP[_SIGMAS[sid][CLASS_FACE[k]]][CLASS_POWER[k]]


def camera_class_for(sid: int, cube_k: int) -> int:
    """Which camera class an observer reports for cube-frame move cube_k."""
    cam_face = _SIGMAS[sid].index(CLASS_FACE[cube_k])
    return int(np.where((CLASS_FACE == cam_face) &
                        (CLASS_POWER == CLASS_POWER[cube_k]))[0][0])


# ---------------------------------------------------------------------------
# Coordinates + pattern-database lower bounds (quarter-turn metric)
# ---------------------------------------------------------------------------

_POW3 = 3 ** np.arange(6, -1, -1)
_POW2 = 2 ** np.arange(10, -1, -1)
_CHOOSE = np.array([[math.comb(j, k) for k in range(5)] for j in range(12)])


def coord_twist(states: np.ndarray) -> np.ndarray:
    return states[:, _CO][:, :7].astype(np.int64) @ _POW3


def coord_flip(states: np.ndarray) -> np.ndarray:
    return states[:, _EO][:, :11].astype(np.int64) @ _POW2


def coord_udslice(states: np.ndarray) -> np.ndarray:
    """Vectorised CubieCube.udslice getter (verified in --selftest)."""
    ep = states[:, _EP]
    mask = ep >= 8
    seen = np.cumsum(mask, axis=1)
    take = (~mask) & (seen >= 1)
    j_idx = np.broadcast_to(np.arange(12), ep.shape)
    vals = _CHOOSE[j_idx, np.where(take, seen - 1, 0)]
    return (vals * take).sum(axis=1)


def coord_corner(states: np.ndarray) -> np.ndarray:
    """Vectorised CubieCube.corner getter (Lehmer code of cp)."""
    cp = states[:, _CP].astype(np.int64)
    c = np.zeros(len(states), dtype=np.int64)
    for j in range(7, 0, -1):
        s = (cp[:, :j] > cp[:, j:j + 1]).sum(axis=1)
        c = j * (c + s)
    return c


def corner_parity(states: np.ndarray) -> np.ndarray:
    cp = states[:, _CP]
    gt = cp[:, :, None] > cp[:, None, :]
    tri = np.triu(np.ones((8, 8), dtype=bool), 1)
    # inversion count over pairs i<j with cp[i] > cp[j]
    return (gt & tri).sum(axis=(1, 2)) % 2


def _bfs_table(size: int, seed_states: np.ndarray, coord_fn) -> np.ndarray:
    """
    BFS distance-to-solved over one coordinate, quarter-turn metric.

    The coordinate transition is a deterministic automaton (that is what
    makes Kociemba-style move tables possible), and the 12-move QTM
    generator set is inverse-closed, so a forward BFS from the solved
    coordinate gives the same distances as a backward one. Each table is
    an admissible lower bound on QTM distance to solved because it relaxes
    the cube to a single coordinate.
    """
    trans = np.empty((size, 12), dtype=np.int32)
    for k in range(12):
        trans[:, k] = coord_fn(apply_batch(seed_states, CLASS_VECS[k]))
    dist = np.full(size, -1, dtype=np.int8)
    start = int(coord_fn(SOLVED[None, :])[0])
    dist[start] = 0
    frontier = np.array([start])
    d = 0
    while len(frontier):
        nxt = np.unique(trans[frontier].ravel())
        nxt = nxt[dist[nxt] == -1]
        d += 1
        dist[nxt] = d
        frontier = nxt
    return dist


TABLE_NAMES = ("twist", "flip", "udslice", "corner")


def build_tables(cache: Path = TABLES_CACHE) -> dict:
    if cache.exists():
        z = np.load(cache)
        # A cache written before a table was added would KeyError deep
        # inside the ranking loop; rebuild instead.
        if all(k in z.files for k in TABLE_NAMES):
            return {k: z[k] for k in z.files}

    t0 = time.time()

    def enumerate_states(size, setter):
        c = CubieCube()
        out = np.empty((size, 40), dtype=np.int8)
        for i in range(size):
            setter(c, i)
            out[i] = _cubie_to_vec(c)
        return out

    tw = enumerate_states(2187, lambda c, i: setattr(c, "twist", i))
    fl = enumerate_states(2048, lambda c, i: setattr(c, "flip", i))
    ud = enumerate_states(495, lambda c, i: setattr(c, "udslice", i))
    co = enumerate_states(40320, lambda c, i: setattr(c, "corner", i))

    tables = {
        "twist": _bfs_table(2187, tw, coord_twist),
        "flip": _bfs_table(2048, fl, coord_flip),
        "udslice": _bfs_table(495, ud, coord_udslice),
        "corner": _bfs_table(40320, co, coord_corner),
    }
    np.savez_compressed(cache, **tables)
    print(f"  built QTM pattern databases in {time.time()-t0:.1f}s "
          f"(cached to {cache.name})")
    return tables


def h_light(states: np.ndarray, tables: dict) -> np.ndarray:
    """Cheap vectorised lower bound used to rank the beam."""
    return np.maximum.reduce([
        tables["twist"][coord_twist(states)],
        tables["flip"][coord_flip(states)],
        tables["udslice"][coord_udslice(states)],
    ]).astype(np.int32)




def h_full(state: np.ndarray, tables: dict) -> int:
    """Full lower bound (adds the corner table); scalar, for the tail."""
    s = state[None, :]
    return int(max(h_light(s, tables)[0],
                   tables["corner"][coord_corner(s)][0]))


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def onset_costs(probs: np.ndarray,
                blend_inv: float = BLEND_INV,
                blend_unif: float = BLEND_UNIF,
                blend_adj: float = BLEND_ADJ) -> np.ndarray:
    """
    Per-class acceptance costs for one onset, relative to the argmax
    (accepting the classifier's own answer costs 0). Prior mass is scaled
    by the onset's uncertainty — see the BLEND_INV comment.

    `blend_adj` (D2, PATH_TO_VERIFICATION.md §5) reserves additional prior
    mass, split uniformly over the 8 adjacent-face classes (ADJ_CLASSES —
    every class except the argmax's own face and its opposite face). The
    measured LIVE error axis is adjacent-face, not the temporal inverse
    BLEND_INV alone was calibrated against (classifier-error-axis-live
    memory) — this gives that confusion its own, separately-priced prior
    mass instead of forcing it through the 12-way uniform tail. 0.0 (the
    default) reproduces the exact prior behaviour.
    """
    p = np.asarray(probs, dtype=np.float64)
    u = 1.0 - float(p.max())
    argmax = int(np.argmax(p))
    reserved = blend_inv * u + blend_adj * u + blend_unif
    q = (1.0 - reserved) * p + blend_unif / 12.0
    q[INV12[argmax]] += blend_inv * u
    if blend_adj:
        adj = ADJ_CLASSES[argmax]
        q[adj] += (blend_adj * u) / len(adj)
    c = np.log(q.max()) - np.log(q)
    return np.maximum(c, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Beam decoder
# ---------------------------------------------------------------------------

# Trace op codes (for backtracking the winning hypothesis)
(OP_CARRY, OP_ACCEPT, OP_DELETE, OP_INSERT, OP_ROTATE, OP_SLICE,
 OP_ALGO, OP_ALGO_HOLD) = range(8)


# ---------------------------------------------------------------------------
# The abstract state: solve progress, as a function of the cube state
# ---------------------------------------------------------------------------
#
# ALGORITHM_PRIOR.md §7. The beam already carries a concrete cube state per
# hypothesis, so "how far through a layer-by-layer solve is this" is
# computable for free at every step, with no perception involved. That
# abstraction is what gates OP_ALGO: an algorithm may only be proposed
# where some face's first two layers are complete both before and after,
# and the transition must make strict progress. Random states essentially
# never satisfy that, which is why this gate is far tighter than
# SLICE_GATE's posterior heuristic could ever be.

# Which cubies belong to each face's layer, in the Kociemba naming the
# state vector uses.
_CORNER_FACES = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]
_EDGE_FACES = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB",
               "FR", "FL", "BL", "BR"]
# (6, 8) / (6, 12) masks: True where the cubie is NOT in that face's layer,
# i.e. the pieces that must be solved for "F2L complete under this top face".
_F2L_CORNERS = np.array([[f not in _CORNER_FACES[i] for i in range(8)]
                         for f in TP_NAME], dtype=bool)
_F2L_EDGES = np.array([[f not in _EDGE_FACES[i] for i in range(12)]
                       for f in TP_NAME], dtype=bool)
_HOME_C = np.arange(8, dtype=np.int8)
_HOME_E = np.arange(12, dtype=np.int8)


def piece_solved(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(m, 8) and (m, 12) booleans: is each cubie home and oriented?"""
    st = np.atleast_2d(states)
    return ((st[:, _CP] == _HOME_C) & (st[:, _CO] == 0),
            (st[:, _EP] == _HOME_E) & (st[:, _EO] == 0))


def f2l_complete(states: np.ndarray) -> np.ndarray:
    """
    (m, 6) booleans: for each candidate top face, is everything below it
    solved? Column order is TP_NAME (URFDLB).
    """
    cs, es = piece_solved(states)
    # (m, 1, 8) & (1, 6, 8) -> all over the pieces that must be solved
    ok_c = np.all(cs[:, None, :] | ~_F2L_CORNERS[None], axis=2)
    ok_e = np.all(es[:, None, :] | ~_F2L_EDGES[None], axis=2)
    return ok_c & ok_e


def n_solved(states: np.ndarray) -> np.ndarray:
    """(m,) count of cubies that are home and oriented."""
    cs, es = piece_solved(states)
    return cs.sum(axis=1) + es.sum(axis=1)


# A NOTE ON WHAT THE GATE DELIBERATELY DOES NOT REQUIRE.
#
# The obvious extra condition is that an algorithm must make PROGRESS — more
# solved cubies after than before. It was written that way first and it is
# wrong. Measured over the 95 ground-truth last-layer chunks:
#
#     strict increase in solved cubies                56%
#     strict increase in solved + correctly oriented  66%
#     merely non-decreasing in that same measure      79%
#
# No monotone rung exists, because an orientation algorithm reorients the
# pieces it is not aiming at: `F R U R' U' F'` fixes edge orientation and
# scrambles corner orientation on the way through. Requiring progress
# therefore rejects a fifth to a half of genuinely performed algorithms —
# it rejects exactly the OLL stage. The cut-point predicate alone (F2L
# complete before AND after) is the gate, and it is already tight: a
# hypothesis that has mis-decoded anything below the last layer cannot
# satisfy it at all.


# 40 state entries + orientation + OP_ALGO debt. _hash40 uses the first 40
# rows only, so the state-only hash is unaffected by the extra columns and
# the ladder's precomputed sets stay valid.
_HASH_MAT = np.random.default_rng(0xC0BE).integers(
    1, 2 ** 62, size=(42, 2), dtype=np.int64)


class _Beam:
    """The K best hypotheses plus the parent-pointer trace to rebuild one."""

    def __init__(self, start: np.ndarray, top_classes: list[int],
                 max_end_ins: int = MAX_END_INS,
                 rel_weight: float = REL_WEIGHT,
                 use_bounds: bool = True, slices: bool = False,
                 slice_ref: list[bool] | None = None,
                 algo_ref: list[tuple] | None = None,
                 algo_turns: int = 1):
        self.max_end_ins = max_end_ins
        self.rel_weight = rel_weight
        self.use_bounds = use_bounds
        # With OP_SLICE enabled an onset may contribute TWO quarter turns
        # instead of one, which invalidates both of _rank's admissible
        # bounds as written. See _rank.
        self.slices = slices
        # Which onsets the REFERENCE story reads as slices. The consistency
        # ladder measures every hypothesis against "apply the remaining
        # detections as-is"; if that reference applies a gated slice onset
        # as a single turn, then the true slice-using story's residual can
        # never be SOLVED and never lands in rep1, so the truth sits at
        # RESID_CAP for the whole decode while wrong-but-one-plain-edit
        # stories rank above it. Measured directly: the truth was evicted
        # at onset 25 of 88 on solve_20260724_103307_solve while doing a
        # conf-1.000 accept, purely on rank.
        self.slice_ref = slice_ref
        # Which onset spans the REFERENCE story reads as a whole algorithm,
        # as [(start, k, group element of the word)]. Exactly the same
        # problem slice_ref solves, one order of magnitude larger: a story
        # that consumed 15 onsets with one OP_ALGO differs from a reference
        # that applied 15 separate argmax turns, so its residual is never
        # SOLVED, never lands in rep1, and it sits at RESID_CAP for the
        # whole decode while cheap one-edit-away garbage ranks above it.
        self.algo_ref = algo_ref or []
        self._algo_at = {a: (k, v) for a, k, v in self.algo_ref}
        self._algo_inside = {i for a, k, _ in self.algo_ref
                             for i in range(a + 1, a + k)}
        # Max quarter turns one onset may supply, for _rank's capacity bound.
        self.algo_turns = algo_turns
        self.top_classes = top_classes      # detected argmax per onset
        self._suffix: dict[int, list[np.ndarray]] = {}
        self._rep1: dict[int, np.ndarray] = {}
        self.states = start[None, :].copy()
        self.sigmas = np.zeros(1, dtype=np.int32)
        self.costs = np.zeros(1, dtype=np.float32)
        # Onsets already consumed by an in-flight OP_ALGO. A hypothesis with
        # debt > 0 has ALREADY paid for the next `debt` onsets and must not
        # read them again; it is excluded from every other transition until
        # the debt clears. This is what lets one beam step consume many
        # onsets without breaking the one-stage-per-onset parent-pointer
        # structure the trace depends on.
        self.debt = np.zeros(1, dtype=np.int16)
        # Hash keys of states that must survive truncation however they
        # rank (set by _run_beam's `protect` argument). Diagnostic only:
        # it answers "is this state absent because it was PRUNED, or
        # because the search never generated it", which cost-ranked output
        # cannot distinguish. Never set on the shipping decode path.
        self.protect_keys: np.ndarray | None = None
        # arg of an OP_ALGO row indexes these; filled in by _run_beam.
        self.algo_words: list[list[int] | None] = []
        self.algo_spans: list[tuple[int, int]] = []
        # Diagnostics, not search state. "OP_ALGO did not appear in the
        # winning story" has two very different causes — the abstract gate
        # never admitted anyone, or it did and the story lost anyway — and
        # they point at opposite next moves (fix the prefix vs loosen the
        # gate). Counting admissions separates them.
        self.algo_offered = 0
        self.algo_onsets: set[int] = set()
        self.stages: list[dict] = [{
            "parent": np.array([-1], dtype=np.int32),
            "op": np.array([OP_CARRY], dtype=np.int8),
            "arg": np.array([0], dtype=np.int16),
        }]
        # Module-level constant, not a per-instance draw: decode_bidirectional
        # relies on a hash match across two different _Beam objects being a
        # genuine state match, and beam_state_keys has to reproduce it from
        # outside the class. Same seed and shape as before.
        self._hash_mat = _HASH_MAT

    def suffix(self, sid: int, j: int) -> np.ndarray:
        """
        Group product of the remaining detected argmax moves j..n-1, in
        the cube frame implied by orientation `sid`. S_n is the identity.
        """
        if sid not in self._suffix:
            n = len(self.top_classes)
            suf = [SOLVED.copy() for _ in range(n + 1)]
            for i in range(n - 1, -1, -1):
                suf[i] = compose(self._ref_move(sid, i), suf[i + 1])
            self._suffix[sid] = suf
        return self._suffix[sid][j]

    def _ref_move(self, sid: int, i: int) -> np.ndarray:
        """
        The reference story's contribution at onset i — the whole word where
        a gated algorithm span starts, nothing at all inside one, a slice
        where the slice gate fired, otherwise the plain argmax turn.
        """
        if i in self._algo_at:
            _, vec = self._algo_at[i]
            if sid == 0:
                return vec
            # Algorithm spans are camera-frame words; only sigma 0 is
            # exercised (OP_ALGO requires rotations=False), so any other
            # orientation falls back to the plain reading rather than
            # silently mis-rotating the reference.
        elif i in self._algo_inside and sid == 0:
            return SOLVED.copy()
        k = self.top_classes[i]
        mv = move_for(sid, k)
        if self.slice_ref and self.slice_ref[i]:
            mv = compose(mv, move_for(sid, int(SLICE_PARTNER[k])))
        return mv

    def _hash40(self, states: np.ndarray) -> np.ndarray:
        h = states.astype(np.int64) @ self._hash_mat[:40]
        return h[:, 0] ^ (h[:, 1] << 1)

    def rep1(self, sid: int) -> np.ndarray:
        """
        Hashes of every residual rel = state*S_j that ONE future edit can
        repair (see module docstring for the three closed forms). One
        global set over all positions — a slight superset for any given j
        (an "edit behind you" can't actually be taken), which can only
        award an undeserved +1-level bonus, never hide a real one.
        """
        if sid not in self._rep1:
            self.suffix(sid, 0)
            suf = self._suffix[sid]
            n = len(self.top_classes)
            els = []
            inv_cache = [inverse(s) for s in suf]
            for q in range(n + 1):
                s_inv, s = inv_cache[q], suf[q]
                for m in range(12):
                    ins = compose(compose(s_inv, move_for(sid, INV12[m])), s)
                    els.append(ins)
            for q in range(n):
                d = self._ref_move(sid, q)
                s_inv, s = inv_cache[q + 1], suf[q + 1]
                els.append(compose(compose(s_inv, d), s))          # delete
                for m in range(12):
                    if m == self.top_classes[q]:
                        continue
                    mid = compose(move_for(sid, INV12[m]), d)
                    els.append(compose(compose(s_inv, mid), s))    # sub
                # Toggling onset q between "one turn" and "a slice" is also
                # a single edit, so a story that differs from the reference
                # only there deserves level 1 rather than RESID_CAP.
                if self.slices:
                    k = self.top_classes[q]
                    alt = move_for(sid, k)
                    if not (self.slice_ref and self.slice_ref[q]):
                        alt = compose(alt, move_for(sid,
                                                    int(SLICE_PARTNER[k])))
                    mid = compose(inverse(alt), d)
                    els.append(compose(compose(s_inv, mid), s))
            # Reading a gated span the OTHER way — as its onsets rather than
            # as the algorithm — is also a single edit. Without this a
            # correct per-onset story inside a gated span is scored as
            # hopeless rather than one edit away, which is the same eviction
            # the slice toggle above exists to prevent, in the opposite
            # direction.
            for a, k_span, vec in self.algo_ref:
                if a + k_span > n:
                    continue
                alt = SOLVED.copy()
                for i in range(a, a + k_span):
                    alt = compose(alt, move_for(sid, self.top_classes[i]))
                s_inv, s = inv_cache[a + k_span], suf[a + k_span]
                mid = compose(inverse(alt), vec if sid == 0 else alt)
                els.append(compose(compose(s_inv, mid), s))
            self._rep1[sid] = np.unique(self._hash40(np.stack(els)))
        return self._rep1[sid]

    @staticmethod
    def _stratified_topk(costs: np.ndarray, rank: np.ndarray, k: int
                         ) -> np.ndarray:
        """
        Top-k by rank, but with beam slots reserved per COST band.

        A session with many real errors makes the true story's cumulative
        cost climb steadily, while wrong-but-cheap stories (delete+insert
        pairs and the like) sit at a permanently low cost and — through
        the long stretch where the consistency ladder is flat — would
        monopolise a plain top-k beam and evict the truth (observed on
        6-substitution synthetics). Reserving slots per cost band keeps
        the expensive strata populated: the cheap flood fills its own
        bands, and the truth only competes against stories of similar
        cost. Spare capacity from underfull bands is refilled globally by
        rank, so stratification never wastes beam width.
        """
        n_bands, width = 8, 2.5
        band = np.minimum((costs - costs.min()) / width,
                          n_bands - 1).astype(np.int32)
        chosen = np.zeros(len(rank), dtype=bool)
        # Half the beam is unconditional global top-by-rank: a hypothesis
        # that out-ranks the crowd must never die to a band cap (a pure
        # per-band split starved exactly that case — every band overfull,
        # the truth capped out of its own band while globally it ranked
        # in the top few thousand).
        top = np.argpartition(rank, k // 2)[:k // 2]
        chosen[top] = True
        per_band = k // (2 * n_bands)
        for b in range(n_bands):
            rows = np.where((band == b) & ~chosen)[0]
            if len(rows) > per_band:
                rows = rows[np.argpartition(rank[rows], per_band)[:per_band]]
            chosen[rows] = True
        spare = k - int(chosen.sum())
        if spare > 0:
            rest = np.where(~chosen)[0]
            if len(rest) > spare:
                rest = rest[np.argpartition(rank[rest], spare)[:spare]]
            chosen[rest] = True
        return np.where(chosen)[0]

    def _rank(self, states, sigmas, costs, remaining, c_ins, c_del, tables):
        """
        cost so far + future-cost estimate. `remaining` counts the onsets
        left to consume; max_end_ins free slots exist past the end.

        `use_bounds=False` (D3 bidirectional partial decodes — see
        decode_bidirectional) skips all three terms below and ranks by
        raw cost alone: every one of them is calibrated around a FIXED
        target of SOLVED/identity (the consistency ladder's `rel == SOLVED`
        check, the capacity bound's "distance to solved", the parity
        bound's "final parity must be even") via decode_between's
        frame-shift trick. A bidirectional partial beam has no such fixed
        target — its endpoint is whatever the OTHER half's search finds —
        so applying identity-calibrated heuristics here would not merely
        be uninformative, it would actively bias the search toward states
        that happen to look close to solved instead of toward the true
        (unknown in advance) meeting point. Pure cost ranking has no such
        bias; it is only less informed, never wrong.
        """
        if not self.use_bounds:
            return costs
        j = len(self.top_classes) - remaining
        # Suffix-consistency ladder (see module docstring): 0 = the rest
        # of the detections finish this story as-is, 1 = one future edit
        # away, RESID_CAP = anything else. Ranks, never prunes.
        rel = np.empty_like(states)
        level = np.full(len(states), float(RESID_CAP), dtype=np.float32)
        for sid in np.unique(sigmas):
            rows = sigmas == sid
            rel[rows] = apply_batch(states[rows], self.suffix(int(sid), j))
            hits = np.isin(self._hash40(rel[rows]), self.rep1(int(sid)))
            level[rows] = np.where(hits, 1.0, float(RESID_CAP))
        level[(rel == SOLVED).all(axis=1)] = 0.0
        pen = self.rel_weight * level

        # Admissible part: capacity and parity bounds. The capacity term
        # can only fire when the pattern bound (<= ~12) can exceed the
        # applicable-move budget, i.e. near the tail — skip it elsewhere.
        #
        # Both bounds below assume each remaining onset supplies exactly ONE
        # quarter turn. OP_SLICE breaks that assumption — an onset read as a
        # slice supplies two — so with slices enabled the capacity budget
        # doubles and the parity argument dissolves entirely (a slice is two
        # quarter turns, i.e. parity-neutral, so the remaining onsets can
        # produce either parity). Keeping either bound unmodified would
        # penalise exactly the slice stories this exists to find.
        #
        # OP_ALGO breaks it far harder in the same direction: one onset can
        # begin a 21-turn word. `algo_turns` is the largest turns-per-onset
        # ratio any live candidate has, so the budget stays an over-estimate
        # — over-estimating capacity can only UNDER-penalise, which is
        # admissible; under-estimating it would evict the truth. Parity
        # dissolves for the same reason it does with slices: a word of any
        # length may be odd or even.
        per_onset = max(2 if self.slices else 1,
                        self.algo_turns if self.algo_ref else 1)
        budget = remaining * per_onset + self.max_end_ins
        if budget < 14:
            need = h_light(states, tables)
            pen += np.maximum(need - budget, 0).astype(np.float32) * c_ins
        if not self.slices and not self.algo_ref:
            par_bad = corner_parity(states) != (remaining % 2)
            # Parity can only be repaired by a net-odd insert/delete (or an
            # odd number of end-insertions) — either way >= one edit.
            pen += par_bad.astype(np.float32) * min(c_ins, c_del)
        return costs + pen

    def merge(self, cand_states, cand_sigmas, cand_costs,
              cand_parent, cand_op, cand_arg,
              remaining, k, c_ins, c_del, tables, cand_debt=None):
        """Dedupe candidates by (state, orientation), keep top-k by rank."""
        if cand_debt is None:
            cand_debt = np.zeros(len(cand_costs), dtype=np.int16)
        # Cheapest-first so np.unique's first occurrence is the min cost.
        order = np.argsort(cand_costs, kind="stable")
        st = cand_states[order]
        sg = cand_sigmas[order]
        co = cand_costs[order]
        pa = cand_parent[order]
        op = cand_op[order]
        ar = cand_arg[order]
        db = cand_debt[order]

        # Single-int64 hash of (state, orientation): a 1-D unique is far
        # cheaper than a row-wise one, and a 64-bit collision (~1e-10 per
        # merge) at worst drops one duplicate hypothesis.
        #
        # Debt is part of the key, not a passenger. Two hypotheses at the
        # same cube state owing different numbers of onsets are genuinely
        # different futures — one still has onsets to read, the other has
        # already paid for them — and collapsing them onto the cheaper one
        # silently deletes the more expensive future.
        h2 = np.concatenate(
            [st.astype(np.int64),
             sg.astype(np.int64)[:, None],
             db.astype(np.int64)[:, None]], axis=1) @ self._hash_mat
        key = h2[:, 0] ^ (h2[:, 1] << 1)
        _, first = np.unique(key, return_index=True)

        st, sg, co, db = st[first], sg[first], co[first], db[first]
        pa, op, ar = pa[first], op[first], ar[first]

        # No cost-based pre-truncation before ranking: the consistency
        # ladder EXISTS to let an expensive-but-consistent hypothesis
        # out-rank hordes of cheap garbage, so cutting the cost tail
        # first would evict exactly the hypotheses the ladder protects
        # (observed directly: the true story's phantom-delete candidate
        # fell outside a 4k-cheapest cut while out-ranking the beam).
        rank = self._rank(st, sg, co, remaining, c_ins, c_del, tables)
        if len(rank) > k:
            keep = self._stratified_topk(co, rank, k)
        else:
            keep = np.arange(len(rank))

        # Protected states ride through truncation. They still have to be
        # GENERATED by an expansion from a surviving parent — this widens
        # what survives, it does not conjure states the search never
        # reached, which is exactly the distinction it exists to measure.
        if self.protect_keys is not None:
            forced = np.where(np.isin(key[first], self.protect_keys))[0]
            if len(forced):
                keep = np.union1d(keep, forced)

        self.states = st[keep]
        self.sigmas = sg[keep]
        self.costs = co[keep]
        self.debt = db[keep]
        self.stages.append({"parent": pa[keep], "op": op[keep],
                            "arg": ar[keep]})

    def backtrace(self, row: int) -> list[tuple[int, int]]:
        ops = []
        for stage in reversed(self.stages):
            op, arg = int(stage["op"][row]), int(stage["arg"][row])
            if op != OP_CARRY:
                ops.append((op, arg))
            row = int(stage["parent"][row])
            if row < 0:
                break
        return ops[::-1]


def beam_state_keys(states: np.ndarray, sigma: int = 0, debt: int = 0
                    ) -> np.ndarray:
    """
    `_Beam.merge`'s (state, orientation, debt) hash for a stack of states.

    Exposed so a caller can build the `protect` array for `_run_beam` with
    the SAME hash the merge uses — recomputing it independently is how the
    two silently drift and every protected state stops matching.
    """
    st = np.atleast_2d(np.asarray(states))
    n = len(st)
    h2 = np.concatenate(
        [st.astype(np.int64),
         np.full((n, 1), sigma, dtype=np.int64),
         np.full((n, 1), debt, dtype=np.int64)], axis=1) @ _HASH_MAT
    return h2[:, 0] ^ (h2[:, 1] << 1)


def _completion(state: np.ndarray, max_d: int, tables: dict
                ) -> list[int] | None:
    """
    Exact IDDFS for the cheapest word (<= max_d quarter turns) taking
    `state` to solved — the end-of-solve moves the camera never saw.
    Small enough to search exactly: branching 12, depth <= 4, and the full
    pattern-database bound prunes almost everything.
    """
    if (state == SOLVED).all():
        return []

    def dfs(s, depth, last_k, path):
        h = h_full(s, tables)
        if h > depth:
            return None
        if depth == 0:
            return list(path) if (s == SOLVED).all() else None
        for k in range(12):
            if last_k >= 0 and k == INV12[last_k]:
                continue  # m then m' is a no-op; skip the obvious cancel
            nxt = compose(s, CLASS_VECS[k])
            path.append(k)
            r = dfs(nxt, depth - 1, k, path)
            path.pop()
            if r is not None:
                return r
        return None

    for d in range(1, max_d + 1):
        r = dfs(state, d, -1, [])
        if r is not None:
            return r
    return None


def _run_beam(start: np.ndarray, cost_rows: list[np.ndarray],
             del_costs: np.ndarray, *,
             beam: int = BEAM, c_del: float = C_DEL, c_ins: float = C_INS,
             c_rot: float = C_ROT, max_end_ins: int = MAX_END_INS,
             rel_weight: float = REL_WEIGHT,
             rotations: bool = False, insertions: bool = True,
             max_chain: int = MAX_CHAIN, tables: dict | None = None,
             use_bounds: bool = True,
             slices: bool = False, c_slice: float = C_SLICE,
             slice_rows: list[np.ndarray] | None = None,
             algo_cands: list[dict] | None = None,
             on_onset=None, protect: np.ndarray | None = None) -> "_Beam":
    """
    Run the beam-search accumulation loop over `cost_rows`, starting from
    `start`, and return the live `_Beam` — no target, no tail completion.

    `on_onset(j, bm)`, if given, is called after onset j has been consumed
    and merged. It exists so a caller can snapshot the live beam — the
    forward mass alpha(state, j) — at every onset in ONE pass instead of
    re-running a truncated beam per onset. Purely observational: it is
    called after the merge and must not mutate `bm`.

    `protect` is an array of state hash keys (build them with
    `beam_state_keys`) that must survive beam truncation however they rank.
    Diagnostic only — it separates "pruned away" from "never generated".

    Factored out of decode() so decode_bidirectional's two partial passes
    (D3, PATH_TO_VERIFICATION.md §5) share the exact same loop rather than
    a parallel reimplementation that could drift from it. decode() itself
    is unchanged in behaviour: it calls this with use_bounds=True (the
    default) and immediately does what it always did with the result —
    see run_selftest, which re-verifies decode() end to end after this
    refactor.
    """
    if tables is None:
        tables = build_tables()
    n = len(cost_rows)
    # Per-onset MAP class under the BLENDED model, which on ~4% of onsets
    # (the low-confidence ones) is the temporal inverse of the classifier's
    # own argmax rather than the argmax itself. That is the more probable
    # single-onset reading, and it is only ever used to build the ranking
    # heuristic's reference story (suffix products + the REP1 ladder) —
    # never as a cost or a constraint, so it cannot bias the result.
    top_classes = [int(np.argmin(row)) for row in cost_rows]
    # The reference story reads a gated onset as a slice — that is what the
    # gate is asserting, and the ladder must measure against it. See
    # _Beam.slice_ref.
    slice_ref = None
    if slices and slice_rows is not None:
        slice_ref = [bool(slice_rows[i][top_classes[i]]) for i in range(n)]

    # -- OP_ALGO setup (ALGORITHM_PRIOR.md §7) ---------------------------
    algo_cands = algo_cands or []
    if algo_cands and rotations:
        raise ValueError("OP_ALGO and --rotations are not compatible: "
                         "library words are camera-frame, so a mid-solve "
                         "reorientation would silently mis-apply them")
    by_start: dict[int, list[dict]] = {}
    algo_vecs: list[np.ndarray] = []
    algo_turns = 1
    for c in algo_cands:
        if c["start"] + c["k"] > n:
            continue
        vec = SOLVED.copy()
        for lab in c["labels"]:
            vec = compose(vec, CLASS_VECS[lab])
        c = dict(c, vec=vec, idx=len(algo_vecs))
        algo_vecs.append(vec)
        by_start.setdefault(c["start"], []).append(c)
        algo_turns = max(algo_turns, -(-len(c["labels"]) // c["k"]))
    # The reference story needs ONE reading per onset, so the gated spans
    # must be a non-overlapping cover: cheapest first, greedily.
    algo_ref, taken = [], np.zeros(n, dtype=bool)
    for c in sorted((c for cs in by_start.values() for c in cs),
                    key=lambda c: c["cost"]):
        a, k_span = c["start"], c["k"]
        if taken[a:a + k_span].any():
            continue
        taken[a:a + k_span] = True
        algo_ref.append((a, k_span, c["vec"]))

    bm = _Beam(start, top_classes, max_end_ins, rel_weight, use_bounds,
               slices, slice_ref, algo_ref, algo_turns)
    if protect is not None and len(protect):
        bm.protect_keys = np.unique(np.asarray(protect, dtype=np.int64))

    for j in range(n):
        remaining = n - j
        # Rows mid-algorithm have already paid for onset j and must sit out
        # every transition below; they only tick their debt down.
        free = bm.debt <= 0
        held = ~free

        # -- expansion rounds: insertions / rotations, staying at onset j
        rounds = max_chain if (insertions or rotations) else 0
        for _ in range(rounds):
            m = len(bm.states)
            idx = np.arange(m, dtype=np.int32)
            fr = idx[bm.debt <= 0]          # every row when OP_ALGO is off
            f_st, f_sg, f_co = bm.states[fr], bm.sigmas[fr], bm.costs[fr]
            nf = len(fr)
            parts_s, parts_g, parts_c, parts_p, parts_o, parts_a, parts_d = \
                [bm.states], [bm.sigmas], [bm.costs], [idx], \
                [np.full(m, OP_CARRY, np.int8)], [np.zeros(m, np.int16)], \
                [bm.debt]
            if insertions and nf:
                for k in range(12):
                    if rotations:
                        new = np.empty_like(f_st)
                        for sid in np.unique(f_sg):
                            rows = f_sg == sid
                            new[rows] = apply_batch(f_st[rows],
                                                    move_for(int(sid), k))
                    else:
                        new = apply_batch(f_st, CLASS_VECS[k])
                    parts_s.append(new)
                    parts_g.append(f_sg)
                    parts_c.append(f_co + c_ins)
                    parts_p.append(fr)
                    parts_o.append(np.full(nf, OP_INSERT, np.int8))
                    parts_a.append(np.full(nf, k, np.int16))
                    parts_d.append(np.zeros(nf, np.int16))
            if rotations and nf:
                for r, rname in enumerate(ROT_NAMES):
                    new_sig = np.array([rotate_sigma(int(s), rname)
                                        for s in f_sg], dtype=np.int32)
                    parts_s.append(f_st)
                    parts_g.append(new_sig)
                    parts_c.append(f_co + c_rot)
                    parts_p.append(fr)
                    parts_o.append(np.full(nf, OP_ROTATE, np.int8))
                    parts_a.append(np.full(nf, r, np.int16))
                    parts_d.append(np.zeros(nf, np.int16))
            bm.merge(np.concatenate(parts_s), np.concatenate(parts_g),
                     np.concatenate(parts_c), np.concatenate(parts_p),
                     np.concatenate(parts_o), np.concatenate(parts_a),
                     remaining, beam, c_ins, c_del, tables,
                     np.concatenate(parts_d))

        # -- consume onset j: accept one of 12 classes, or delete it
        m = len(bm.states)
        idx = np.arange(m, dtype=np.int32)
        fr = idx[bm.debt <= 0]              # every row when OP_ALGO is off
        f_st, f_sg, f_co = bm.states[fr], bm.sigmas[fr], bm.costs[fr]
        nf = len(fr)
        costs_j = cost_rows[j]
        parts_s, parts_g, parts_c, parts_p, parts_o, parts_a, parts_d = \
            [], [], [], [], [], [], []

        # Rows mid-algorithm: tick the debt down, change nothing else. They
        # already paid for this onset when the OP_ALGO transition was taken.
        hr = idx[bm.debt > 0]
        if len(hr):
            parts_s.append(bm.states[hr])
            parts_g.append(bm.sigmas[hr])
            parts_c.append(bm.costs[hr])
            parts_p.append(hr)
            parts_o.append(np.full(len(hr), OP_ALGO_HOLD, np.int8))
            parts_a.append(np.zeros(len(hr), np.int16))
            parts_d.append(bm.debt[hr] - 1)

        for k in range(12):
            if rotations:
                new = np.empty_like(f_st)
                for sid in np.unique(f_sg):
                    rows = f_sg == sid
                    new[rows] = apply_batch(f_st[rows], move_for(int(sid), k))
            else:
                new = apply_batch(f_st, CLASS_VECS[k])
            parts_s.append(new)
            parts_g.append(f_sg)
            parts_c.append(f_co + costs_j[k])
            parts_p.append(fr)
            parts_o.append(np.full(nf, OP_ACCEPT, np.int8))
            parts_a.append(np.full(nf, k, np.int16))
            parts_d.append(np.zeros(nf, np.int16))

            # -- read this one onset as a middle slice: class k AND its
            # partner, from a single detected motion (see SLICE_PARTNER).
            # Priced off the SAME acceptance cost as the plain read, so a
            # slice is only ever preferred where the classifier already
            # liked k and the story needs the extra turn.
            if slices and (slice_rows is None or slice_rows[j][k]):
                if rotations:
                    sl = np.empty_like(f_st)
                    for sid in np.unique(f_sg):
                        rows = f_sg == sid
                        sl[rows] = apply_batch(
                            apply_batch(f_st[rows], move_for(int(sid), k)),
                            move_for(int(sid), int(SLICE_PARTNER[k])))
                else:
                    sl = apply_batch(f_st, SLICE_VECS[k])
                parts_s.append(sl)
                parts_g.append(f_sg)
                parts_c.append(f_co + costs_j[k] + c_slice)
                parts_p.append(fr)
                parts_o.append(np.full(nf, OP_SLICE, np.int8))
                parts_a.append(np.full(nf, k, np.int16))
                parts_d.append(np.zeros(nf, np.int16))
        parts_s.append(f_st)
        parts_g.append(f_sg)
        parts_c.append(f_co + float(del_costs[j]))
        parts_p.append(fr)
        parts_o.append(np.full(nf, OP_DELETE, np.int8))
        parts_a.append(np.zeros(nf, np.int16))
        parts_d.append(np.zeros(nf, np.int16))

        # -- OP_ALGO: consume onsets j..j+k-1 as one known algorithm.
        #
        # Gated by the ABSTRACT state, not by a posterior heuristic: the
        # SAME face's first two layers must be complete both before and
        # after, i.e. the hypothesis is at a genuine cut point and the word
        # preserves everything below the last layer. A hypothesis that has
        # mis-decoded anything earlier cannot satisfy that, which is what
        # keeps these multi-move jumps from flooding the beam the way
        # ungated slices did. See the note above n_solved for why there is
        # deliberately no progress requirement on top.
        if nf:
            for cand in by_start.get(j, []):
                if j + cand["k"] > n:
                    continue
                before = f2l_complete(f_st)
                if not before.any():
                    continue
                new = apply_batch(f_st, cand["vec"])
                after = f2l_complete(new)
                ok = ((before & after).any(axis=1)
                      & (new != f_st).any(axis=1))
                rows = np.where(ok)[0]
                if not len(rows):
                    continue
                bm.algo_offered += len(rows)
                bm.algo_onsets.add(j)
                parts_s.append(new[rows])
                parts_g.append(f_sg[rows])
                parts_c.append(f_co[rows] + cand["cost"])
                parts_p.append(fr[rows])
                parts_o.append(np.full(len(rows), OP_ALGO, np.int8))
                parts_a.append(np.full(len(rows), cand["idx"], np.int16))
                parts_d.append(np.full(len(rows), cand["k"] - 1, np.int16))

        bm.merge(np.concatenate(parts_s), np.concatenate(parts_g),
                 np.concatenate(parts_c), np.concatenate(parts_p),
                 np.concatenate(parts_o), np.concatenate(parts_a),
                 remaining - 1, beam, c_ins, c_del, tables,
                 np.concatenate(parts_d))

        if on_onset is not None:
            on_onset(j, bm)

    bm.algo_words = [None] * len(algo_vecs)
    bm.algo_spans = [(0, 0)] * len(algo_vecs)
    for cs in by_start.values():
        for c in cs:
            bm.algo_words[c["idx"]] = c["labels"]
            bm.algo_spans[c["idx"]] = (c["start"], c["k"])
    return bm


def _ops_to_moves(bm: "_Beam", row: int) -> tuple[list, list[str], list[dict]]:
    """
    backtrace(row) -> (ops, moves, algo_spans) named the way decode()'s
    result dict reports them. Shared by decode() and decode_bidirectional so
    the two never format a reconstruction differently.

    `algo_spans` records which OP_ALGO transitions the winning story
    actually took — the question "did the prior fire, and where" is not
    answerable from the move list alone, since OP_ALGO emits plain WCA
    moves indistinguishable from accepted ones.
    """
    ops: list[tuple[str, str | None]] = []
    moves: list[str] = []
    fired: list[dict] = []
    sid = 0

    def cube_name(k: int) -> str:
        return (TP_NAME[_SIGMAS[sid][CLASS_FACE[k]]]
                + ("'" if CLASS_POWER[k] == 3 else ""))

    for op, arg in bm.backtrace(row):
        if op == OP_ACCEPT or op == OP_INSERT:
            name = cube_name(arg)
            ops.append(("accept" if op == OP_ACCEPT else "insert", name))
            moves.append(name)
        elif op == OP_SLICE:
            # One onset, two layer turns. Both are emitted into `moves` so
            # the reconstruction stays a plain WCA word — every consumer
            # downstream (the replay assert in decode(), score_vs_gt,
            # verify_solve) keeps working without knowing slices exist.
            first, second = cube_name(arg), cube_name(int(SLICE_PARTNER[arg]))
            ops.append(("slice", f"{first}{second}"))
            moves.extend((first, second))
        elif op == OP_ALGO:
            # One transition, many onsets, a whole known word. Emitted move
            # by move so the reconstruction stays a plain WCA word and every
            # consumer downstream keeps working without knowing OP_ALGO
            # exists — same contract as OP_SLICE.
            word = [cube_name(k) for k in bm.algo_words[arg]]
            start, k_span = bm.algo_spans[arg]
            ops.append(("algorithm", " ".join(word)))
            fired.append({"onset": start, "n_onsets": k_span,
                          "move_index": len(moves), "word": word})
            moves.extend(word)
        elif op == OP_ALGO_HOLD:
            pass                    # onset already paid for by its OP_ALGO
        elif op == OP_DELETE:
            ops.append(("delete", None))
        elif op == OP_ROTATE:
            sid = rotate_sigma(sid, ROT_NAMES[arg])
            ops.append(("rotate", ROT_NAMES[arg]))
    return ops, moves, fired


def decode(start: np.ndarray, cost_rows: list[np.ndarray], *,
           beam: int = BEAM, c_del: float = C_DEL, c_ins: float = C_INS,
           c_rot: float = C_ROT, max_end_ins: int = MAX_END_INS,
           rel_weight: float = REL_WEIGHT,
           del_costs: np.ndarray | None = None,
           rotations: bool = False, insertions: bool = True,
           max_chain: int = MAX_CHAIN, tables: dict | None = None,
           slices: bool = False, c_slice: float = C_SLICE,
           slice_rows: list[np.ndarray] | None = None,
           algo_cands: list[dict] | None = None) -> dict:
    """
    Beam-decode the detected sequence against start -> solved.

    cost_rows: one (12,) array per detected onset — acceptance cost per
    class (see onset_costs). del_costs: optional per-onset deletion cost
    (from onset strength — see the module docstring); defaults to a flat
    c_del.

    `slices` enables OP_SLICE: one onset may be read as a middle-slice turn
    and emit two layer moves for c_slice on top of its acceptance cost.
    Defaults OFF, which reproduces the exact prior behaviour. `slice_rows`
    (from costs_from_moves) restricts WHICH onsets may be read that way —
    without it the transition is offered everywhere and the beam floods; see
    SLICE_GATE.

    `algo_cands` enables OP_ALGO: a span of onsets read as one known
    last-layer algorithm, applying the whole word for one cost (see
    ALGORITHM_PRIOR.md §7 and algorithm_gate.build_candidates, which
    produces these). Defaults None = off, reproducing prior behaviour
    exactly. Unlike the slice gate this is gated on the ABSTRACT CUBE STATE
    of each hypothesis, so it can only fire at a genuine cut point.

    Returns a dict with "solved", "cost", "ops" (chronological
    [(op_name, class/rotation name or None), ...]), "moves" (the
    reconstructed cube-frame layer turns), and diagnostics.
    """
    if tables is None:
        tables = build_tables()
    n = len(cost_rows)
    if del_costs is None:
        del_costs = np.full(n, c_del, dtype=np.float32)
    t0 = time.time()
    bm = _run_beam(start, cost_rows, del_costs, beam=beam, c_del=c_del,
                   c_ins=c_ins, c_rot=c_rot, max_end_ins=max_end_ins,
                   rel_weight=rel_weight, rotations=rotations,
                   insertions=insertions, max_chain=max_chain, tables=tables,
                   slices=slices, c_slice=c_slice, slice_rows=slice_rows,
                   algo_cands=algo_cands)

    # -- tail: finish each candidate end state exactly.
    # Exactly-solved detection is a vectorised scan of the WHOLE beam —
    # an earlier version only walked the 1024 cheapest rows, and on a
    # real session the true story (which had paid for its repairs) sat
    # behind >1024 cheap non-solving hypotheses and was never examined.
    # Only the per-state completion searches are capped.
    best = None
    solved_rows = np.where((bm.states == SOLVED).all(axis=1))[0]
    if len(solved_rows):
        row = solved_rows[np.argmin(bm.costs[solved_rows])]
        best = (float(bm.costs[row]), int(row), [])

    seen: set[bytes] = set()
    order = np.argsort(bm.costs, kind="stable")
    tried = 0
    for row in order:
        cost = float(bm.costs[row])
        if best is not None and cost >= best[0]:
            break  # completions only ADD cost; nothing later can win
        if tried >= 1024:
            break
        key = bm.states[row].tobytes()
        if key in seen:
            continue
        seen.add(key)
        state = bm.states[row]
        if (state == SOLVED).all():
            continue  # already covered by the vectorised scan
        tried += 1
        if h_full(state, tables) > max_end_ins:
            continue
        word = _completion(state, max_end_ins, tables)
        if word is None:
            continue
        total = cost + len(word) * c_ins
        if best is None or total < best[0]:
            best = (total, int(row), word)

    result = {
        "solved": best is not None,
        "n_onsets": n,
        "beam": beam,
        "decode_seconds": time.time() - t0,
        # Reported whether or not a story was found — on a failed decode
        # this is the only evidence about whether OP_ALGO was reachable.
        "algo_offered": bm.algo_offered,
        "algo_gate_onsets": sorted(bm.algo_onsets),
    }
    if best is None:
        # Diagnostics for the failure: how far is the closest end state?
        hmin = min(h_full(bm.states[r], tables) for r in order[:64])
        # The MAP story among hypotheses that consumed every onset, even
        # though it does not reach solved. This is NOT a verifiable claim
        # and must never be reported as one — but "how close to the truth
        # did we get" is a graded question, and a decode that fails the
        # binary check can still be nearly right. Scoring it is the only
        # way to see movement on sessions that are all currently failures.
        cheapest = int(order[0])
        _, be_moves, _ = _ops_to_moves(bm, cheapest)
        result.update({"cost": None, "ops": None, "moves": None,
                       "min_h_final": int(hmin),
                       "best_effort_moves": be_moves,
                       "best_effort_cost": float(bm.costs[cheapest])})
        return result

    total, row, end_word = best
    ops, moves, algo_fired = _ops_to_moves(bm, row)  # cube-frame turns
    for k in end_word:
        # _completion searches in the cube frame directly, so its word
        # needs no orientation mapping — mapping it through sigma again
        # would double-rotate it.
        name = WCA12[k]
        ops.append(("end-insert", name))
        moves.append(name)

    # The trace must replay to solved — a wrong reconstruction that
    # happens to score well is exactly the failure this asserts against.
    check = start.copy()
    for nm in moves:
        check = compose(check, CLASS_VECS[WCA12.index(nm)])
    assert (check == SOLVED).all(), "decoded sequence does not solve"

    counts = {}
    for op_name, _ in ops:
        counts[op_name] = counts.get(op_name, 0) + 1
    result.update({"cost": total, "ops": ops, "moves": moves,
                   "op_counts": counts, "algo_fired": algo_fired,
                   # Same field on the success path so callers scoring
                   # "how close to truth" never have to branch.
                   "best_effort_moves": moves,
                   "best_effort_cost": total})
    return result


def decode_between(start_state: np.ndarray, end_state: np.ndarray,
                   cost_rows: list[np.ndarray], **kw) -> dict:
    """
    Decode a story taking `start_state` to an arbitrary `end_state`.

    decode() targets the solved cube, which is the right target for a solve
    but not for a scramble (solved -> some scrambled state), a mid-solve
    segment, or a decoy claim. All of those are the same problem in a
    shifted frame: we want the word W with start*W = end, and decode finds
    the W with s*W = I for s = end^-1 * start. Every bound decode relies on
    (the pattern databases, the parity argument, the consistency ladder) is
    a property of a genuine cube state and survives the shift unchanged —
    only the target moves. The returned "moves" are W, in the cube frame,
    exactly as decode() reports them.

    OP_ALGO is the one thing that does NOT survive the shift: its gate asks
    whether a hypothesis' first two layers are solved, and under a shifted
    frame that predicate is about `end^-1 * actual`, not the actual cube.
    Refused loudly rather than silently mis-gated. (The decoy sweep is
    unaffected — a decoy claim still targets SOLVED, it just starts
    somewhere else, and the predicate stays meaningful.)
    """
    if kw.get("algo_cands") and not (end_state == SOLVED).all():
        raise ValueError("algo_cands requires end_state == SOLVED: the "
                         "abstract-state gate is not frame-shift invariant")
    return decode(compose(inverse(end_state), start_state), cost_rows, **kw)


_INV12_ARR = np.array(INV12, dtype=np.intp)


def _reverse_invert_rows(cost_rows: list[np.ndarray]) -> list[np.ndarray]:
    """
    cost_rows for the BACKWARD pass: reversed order, each row permuted so
    that accepting class c backward prices exactly what accepting class
    INV12[c] would have cost forward at the same (mirrored) onset — see
    decode_bidirectional's docstring for the group-theory derivation.
    """
    return [row[_INV12_ARR] for row in reversed(cost_rows)]


def decode_bidirectional(start_state: np.ndarray, end_state: np.ndarray,
                         cost_rows: list[np.ndarray],
                         del_costs: np.ndarray | None = None, *,
                         meet: int | None = None,
                         beam: int = BEAM, c_del: float = C_DEL,
                         c_ins: float = C_INS, c_rot: float = C_ROT,
                         max_end_ins: int = MAX_END_INS,
                         rel_weight: float = REL_WEIGHT,
                         rotations: bool = False,
                         max_chain: int = MAX_CHAIN,
                         tables: dict | None = None) -> dict:
    """
    D3 (PATH_TO_VERIFICATION.md §5): decode start_state -> end_state by
    searching FORWARD from start_state over onsets [0, meet) and BACKWARD
    from end_state over onsets [meet, n), meeting at the (unknown in
    advance) state in between — instead of one single-pass search that has
    to survive the true story's WHOLE cost against beam eviction.

    Why this works (group algebra)
    -------------------------------
    state_{i+1} = state_i * m_i (compose's "a then b" convention), so
    state_i = state_{i+1} * m_i^-1. Walking backward from state_n = end,
    applying m_{n-1}^-1, m_{n-2}^-1, ... reaches state_meet exactly. Since
    CLASS_VECS[INV12[k]] == inverse(CLASS_VECS[k]) (asserted in
    run_selftest: "every move times its inverse = I"), a beam search
    started at `end_state`, consuming the reversed cost rows with each
    row's classes permuted by INV12 (_reverse_invert_rows), naturally
    accepts class INV12[k] whenever the un-permuted row would have
    accepted k — which applies CLASS_VECS[INV12[k]] = m_i^-1, exactly the
    move needed to walk backward. So the backward beam's live states after
    consuming onsets [meet, n) are, like the forward beam's after [0,
    meet), candidates for the SAME quantity: state_meet. No frame-shift
    trick is needed (unlike decode_between) because both beams already
    have a natively-known anchor (start_state, end_state) — only the
    MEETING point is unknown, and that is what the join below resolves.

    This also means the backward half needs no separate "tail completion"
    for moves the camera missed near the end of the solve: those are
    ordinary mid-sequence insertions from the backward beam's own point of
    view, using the same c_ins/max_chain machinery as everywhere else.

    Ranking caveat
    --------------
    Both partial beams run with use_bounds=False (see _Beam._rank): the
    consistency ladder, capacity bound and parity bound are all calibrated
    around a FIXED target of solved/identity, which neither partial beam
    has (their target is whatever the other beam's search finds). They
    rank by raw accumulated cost instead — see _rank's docstring for why
    this is the safe default rather than a shortcut.

    Simplification
    --------------
    rotations must be False (asserted): sigma-bookkeeping for the join
    below assumes both beams share orientation 0 throughout, matching the
    single-orientation assumption already documented for --rotations
    elsewhere in this file. The join is EXACT (state hash match, verified
    by a full equality check on any hash collision, then a final
    replay-to-target assert) — no rep1-style one-edit seam bridge is
    attempted when the two halves fail to meet exactly; that is a known,
    documented simplification (PATH_TO_VERIFICATION.md §5, D3), not a
    silent gap: a failed join reports solved=False with both partial
    beams' cheapest costs, not a wrong answer.
    """
    if rotations:
        raise NotImplementedError(
            "decode_bidirectional: rotations=True needs sigma-aware "
            "joining, not implemented — see the module docstring caveat")
    if tables is None:
        tables = build_tables()
    n = len(cost_rows)
    if del_costs is None:
        del_costs = np.full(n, c_del, dtype=np.float32)
    meet = n // 2 if meet is None else int(np.clip(meet, 0, n))
    t0 = time.time()

    fwd_rows, fwd_del = cost_rows[:meet], del_costs[:meet]
    bwd_rows = _reverse_invert_rows(cost_rows[meet:])
    bwd_del = del_costs[meet:][::-1].copy()

    kw = dict(beam=beam, c_del=c_del, c_ins=c_ins, c_rot=c_rot,
             max_end_ins=max_end_ins, rel_weight=rel_weight,
             rotations=False, max_chain=max_chain, tables=tables,
             use_bounds=False)
    bm_f = _run_beam(start_state, fwd_rows, fwd_del, **kw)
    bm_b = _run_beam(end_state, bwd_rows, bwd_del, **kw)

    # Join: both beams' live states are candidates for state_meet. Beams
    # are deduped-unique within themselves (merge() dedupes every round),
    # and _hash_mat is seeded identically (0xC0BE) across every _Beam
    # instance, so a hash collision means a genuine state match.
    h_b = bm_b._hash40(bm_b.states)
    best_for_hash: dict[int, int] = {}
    for j, h in enumerate(h_b.tolist()):
        cur = best_for_hash.get(h)
        if cur is None or bm_b.costs[j] < bm_b.costs[cur]:
            best_for_hash[h] = j

    h_f = bm_f._hash40(bm_f.states)
    best = None
    for i, h in enumerate(h_f.tolist()):
        j = best_for_hash.get(h)
        if j is None or not (bm_f.states[i] == bm_b.states[j]).all():
            continue
        total = float(bm_f.costs[i] + bm_b.costs[j])
        if best is None or total < best[0]:
            best = (total, i, j)

    result = {"solved": best is not None, "n_onsets": n, "beam": beam,
             "meet": meet, "decode_seconds": time.time() - t0}
    if best is None:
        result.update({"cost": None, "ops": None, "moves": None,
                       "fwd_min_cost": float(bm_f.costs.min())
                       if len(bm_f.costs) else None,
                       "bwd_min_cost": float(bm_b.costs.min())
                       if len(bm_b.costs) else None})
        return result

    total, i, j = best
    fwd_ops, fwd_moves, _ = _ops_to_moves(bm_f, i)
    # Backward ops are in backward-chronological order (position 0 = the
    # LAST original onset); reverse for forward-time order, and map each
    # accepted/inserted class through INV12 back to the ORIGINAL move name
    # (see the docstring derivation — the backward beam's own class c
    # represents original move INV12[c]).
    bwd_ops_raw, _, _ = _ops_to_moves(bm_b, j)
    bwd_ops: list[tuple[str, str | None]] = []
    bwd_moves: list[str] = []
    for op, name in reversed(bwd_ops_raw):
        if op in ("accept", "insert"):
            orig = WCA12[INV12[WCA12.index(name)]]
            bwd_ops.append((op, orig))
            bwd_moves.append(orig)
        else:
            bwd_ops.append((op, name))

    ops = fwd_ops + bwd_ops
    moves = fwd_moves + bwd_moves

    check = start_state.copy()
    for nm in moves:
        check = compose(check, CLASS_VECS[WCA12.index(nm)])
    assert (check == end_state).all(), \
        "decode_bidirectional: joined sequence does not reach end_state"

    counts: dict[str, int] = {}
    for op_name, _ in ops:
        counts[op_name] = counts.get(op_name, 0) + 1
    result.update({"cost": total, "ops": ops, "moves": moves,
                   "op_counts": counts})
    return result


def decode_bidirectional_sweep(start_state: np.ndarray, end_state: np.ndarray,
                               cost_rows: list[np.ndarray],
                               del_costs: np.ndarray | None = None, *,
                               meets: list[int] | None = None,
                               **kw) -> dict:
    """
    decode_bidirectional at a few different meet points (default n/3, n/2,
    2n/3 — PATH_TO_VERIFICATION.md §5's "sweep the meet point" suggestion),
    keeping the cheapest SOLVED result. A miss-heavy or error-heavy half
    wants an off-centre meet, and this costs 2-3x the decode time of one
    fixed split to find it.
    """
    n = len(cost_rows)
    if meets is None:
        meets = sorted(set(int(np.clip(m, 1, max(n - 1, 1)))
                           for m in (n // 3, n // 2, (2 * n) // 3)))
    best = None
    for m in meets:
        res = decode_bidirectional(start_state, end_state, cost_rows,
                                   del_costs, meet=m, **kw)
        res["meet_swept"] = m
        if res["solved"] and (best is None or not best["solved"]
                              or res["cost"] < best["cost"]):
            best = res
        elif best is None:
            best = res
    return best


def costs_from_moves(moves: list[dict], threshold: float = 0.5,
                     blend_inv: float = BLEND_INV,
                     blend_unif: float = BLEND_UNIF,
                     c_del: float = C_DEL,
                     candidate_threshold: float | None = None,
                     del_floor: float = DEL_FLOOR,
                     blend_adj: float = BLEND_ADJ,
                     slice_gate: float = SLICE_GATE) -> tuple:
    """
    (pred_names, cost_rows, del_costs) from live_detect.analyse()'s move
    dicts — the one place the classifier softmax and the detector's onset
    strength are turned into edit costs, shared by session replay and the
    live verification so the two cannot drift apart.

    `candidate_threshold`/`del_floor` are D1's soft-onset lattice knobs —
    see score_del_costs. Pass the same `candidate_threshold` used to
    peak-pick `moves` (live_detect.analyse's `peak_threshold`), or leave it
    None if `moves` was peak-picked at `threshold` as usual. `blend_adj` is
    D2's confusion-calibrated prior — see onset_costs.
    """
    pred_names = [m["move"] for m in moves]
    cost_rows = [onset_costs(np.asarray(m["probs"]), blend_inv, blend_unif,
                             blend_adj)
                 for m in moves]
    del_costs = score_del_costs([m["score"] for m in moves], threshold, c_del,
                                candidate_threshold, del_floor)
    return pred_names, cost_rows, del_costs


def slice_rows_from_moves(moves: list[dict],
                          gate: float = SLICE_GATE) -> list[np.ndarray]:
    """
    Per-onset (12,) boolean mask of which classes may start a slice.

    Kept separate from costs_from_moves so its three-value return stays a
    stable interface for the callers that predate slices; pass the result to
    decode(slice_rows=...) alongside slices=True.
    """
    return [slice_mask(np.asarray(m["probs"]), gate) for m in moves]


# ---------------------------------------------------------------------------
# Scoring a reconstruction against ground truth
# ---------------------------------------------------------------------------

def score_vs_gt(gt: list[str], rec: list[str]) -> dict:
    ops = align_sequences(gt, rec)
    ok = sum(1 for o in ops if o[0] == "ok")
    return {
        "exact": rec == gt,
        "acc": ok / len(gt) if gt else 1.0,
        "sub": sum(1 for o in ops if o[0] == "sub"),
        "miss": sum(1 for o in ops if o[0] == "miss"),
        "phantom": sum(1 for o in ops if o[0] == "phantom"),
    }


def gt_path_cost(gt: list[str], pred_names: list[str],
                 cost_rows: list[np.ndarray],
                 del_costs: np.ndarray, c_ins: float,
                 slices: bool = False, c_slice: float = C_SLICE) -> float:
    """
    Cost of the ground-truth story under the same model: align GT to the
    detected argmax sequence, then price each aligned op. Approximate
    (the alignment is unit-cost) but tight enough to tell "beam lost the
    truth" from "the model prefers a cheaper wrong answer".

    With `slices`, a missed ground-truth move that is the slice partner of
    an adjacent matched one is priced at c_slice rather than c_ins: the
    decoder can reach it with a single OP_SLICE instead of an insertion, so
    charging the full insertion would overstate the truth's cost by exactly
    the amount OP_SLICE exists to save. Without this the metric would keep
    reporting a gap the decoder no longer has to close.
    """
    ops = align_sequences(gt, pred_names)
    cost, j = 0.0, 0
    for i, (op, truth, _pred) in enumerate(ops):
        if op == "ok" or op == "sub":
            cost += float(cost_rows[j][WCA12.index(truth)])
            j += 1
        elif op == "phantom":
            cost += float(del_costs[j])
            j += 1
        else:  # miss
            partnered = False
            if slices and truth in WCA12:
                want = WCA12[SLICE_PARTNER[WCA12.index(truth)]]
                for nb in (i - 1, i + 1):
                    if 0 <= nb < len(ops) and ops[nb][0] in ("ok", "sub") \
                            and ops[nb][1] == want:
                        partnered = True
                        break
            cost += c_slice if partnered else c_ins
    return cost


def score_del_costs(scores: list[float], threshold: float,
                    c_del: float = C_DEL,
                    candidate_threshold: float | None = None,
                    del_floor: float = DEL_FLOOR) -> np.ndarray:
    """
    Per-onset deletion costs from detector onset strength: an onset that
    barely cleared the peak-picking threshold deletes near c_del, one the
    detector was sure about costs c_del + DEL_SCORE_W.

    `candidate_threshold` (D1, PATH_TO_VERIFICATION.md's soft-onset
    lattice): when the caller peak-picked below the deployed operating
    point (live_detect.analyse's `peak_threshold`), onsets can arrive with
    score < `threshold` — previously impossible, since nothing below
    threshold was ever detected at all. Pricing those at the flat c_del
    above would tax the true story once per weak candidate for zero
    benefit (most are noise; the measured fp odds concentrate at weak
    scores — see the module docstring). Instead ramp their deletion cost
    from `del_floor` (near `candidate_threshold`, i.e. near free) up to
    c_del (at `threshold`, where the old flat cost takes back over) — a
    real subthreshold miss the detector actually saw still gets accepted
    near-free by its classifier softmax; a noise peak gets deleted
    near-free instead of taxing the beam. None (the default) reproduces
    the exact prior flat-c_del-below-threshold behaviour.
    """
    s = np.asarray(scores, dtype=np.float32)
    strength = np.clip((s - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    cost = c_del + DEL_SCORE_W * strength
    if candidate_threshold is not None and candidate_threshold < threshold:
        weak = s < threshold
        span = max(threshold - candidate_threshold, 1e-6)
        ramp = np.clip((s - candidate_threshold) / span, 0.0, 1.0)
        weak_cost = del_floor + (c_del - del_floor) * ramp
        cost = np.where(weak, weak_cost, cost)
    return cost.astype(np.float32)


# ---------------------------------------------------------------------------
# Session evaluation (replays through the real detector + classifier)
# ---------------------------------------------------------------------------

def _load_replay(session_dir: Path, args) -> dict | None:
    """
    Detected onsets + full per-onset softmax for a session, cached to
    <session>/reconstruct_replay.json (training_data/ is gitignored).

    Cache key includes `candidate_threshold` (D1's soft-onset lattice):
    flipping that flag changes which onsets even get peak-picked, so a
    cache written before/after the flag changed must not be reused across
    the flip — this is what stops the footgun the module docstring used to
    just warn about (a threshold change silently measuring a stale
    lattice). None (unset) round-trips through JSON as null.
    """
    cand_thresh = getattr(args, "candidate_threshold", None)
    cache = session_dir / "reconstruct_replay.json"
    meta = {"detector": Path(args.detector).name,
            "classifier": Path(args.classifier).name,
            "candidate_threshold": cand_thresh}
    if cache.exists() and not args.refresh_cache:
        data = json.loads(cache.read_text())
        m = data.get("meta", {})
        if (m.get("detector") == meta["detector"]
                and m.get("classifier") == meta["classifier"]
                and m.get("candidate_threshold") == meta["candidate_threshold"]):
            return data

    # Torch stack only loads when a replay actually has to run.
    import cv2
    import torch
    from model import build_model
    from crop_utils import load_detector
    from live_detect import analyse
    from decode import MIN_SEP

    global _REPLAY_CTX
    if "_REPLAY_CTX" not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.detector, map_location=device)
        det_model = build_model(device)
        det_model.load_state_dict(ckpt["state_dict"])
        det_model.eval()
        threshold = ckpt.get("threshold", 0.5)
        min_sep = ckpt.get("min_sep", MIN_SEP)
        detector = load_detector()
        print(f"  Detector:   {args.detector} (threshold {threshold}, "
              f"min_sep {min_sep})")
        print(f"  Classifier: {args.classifier}")
        print(f"  Device:     "
              f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
        _REPLAY_CTX = (detector, det_model, device, threshold, min_sep)
    detector, det_model, device, threshold, min_sep = _REPLAY_CTX

    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        print(f"  {session_dir.name}: no frames/ — skipping")
        return None
    recs = [json.loads(l) for l in open(session_dir / "frames.jsonl")
            if l.strip()]
    paths = [frames_dir / r["file"] for r in recs]
    keep = [i for i, p in enumerate(paths) if p.exists()]
    paths = [paths[i] for i in keep]
    ts = np.array([recs[i]["ts"] for i in keep])
    n = len(paths)
    fps = n / (ts[-1] - ts[0]) if n > 1 else 30.0

    cache_frames: dict[int, np.ndarray] = {}

    def load_color(i):
        if i not in cache_frames:
            if len(cache_frames) > 600:
                cache_frames.clear()
            cache_frames[i] = cv2.imread(str(paths[i]))
        return cache_frames[i]

    res = analyse(load_color, n, fps, detector, det_model, device,
                  threshold, min_sep, args.classifier, frame_times=ts,
                  peak_threshold=cand_thresh)
    data = {
        "meta": {**meta, "threshold": threshold, "min_sep": min_sep},
        "class_names": res["class_names"],
        "moves": [{"frame": m["frame"], "move": m["move"],
                   "conf": m["conf"], "score": m["score"],
                   "probs": [round(p, 6) for p in m["probs"]]}
                  for m in res["moves"]],
    }
    cache.write_text(json.dumps(data))
    return data


# Sessions the deployed classifier never trained on — the honest rows.
# Source: training_run_20260722_classifier_all23.log, whose four holdouts
# were named explicitly (--val-session-names) rather than picked at
# random, one per recording environment. Keep this in step with whatever
# CLASSIFIER_PATH points at: it shrank on 2026-07-22 when the three
# 20260722 sessions stopped being unseen and two of them became training
# data. A stale entry here silently reports a training session as an
# honest result, which is the one error this constant exists to prevent.
CLASSIFIER_UNSEEN = {
    "solve_20260720_142006",
    "solve_20260721_102711", "solve_20260721_110811",
    "solve_20260722_101225",
}


def classifier_unseen(session_name: str, classifier_path) -> bool:
    """
    Did the deployed classifier ever train on this session?

    Three sources, in order of authority:

    1. the checkpoint's own `val_session_names` — exact, written by the
       trainer, and immune to this file going stale;
    2. the session predating the checkpoint but not being held out — then
       it IS training data;
    3. the session being recorded AFTER the checkpoint was written, in
       which case it cannot possibly have been trained on.

    Rule 3 is the one the hardcoded set above could not express, and its
    absence had the opposite failure from the one that set was guarding
    against: every freshly recorded session — exactly the honest,
    out-of-distribution data worth measuring — was reported as training
    data and dropped from the honest row. Falls back to the literal set
    for checkpoints too old to carry metadata.
    """
    from datetime import datetime

    ckpt_path = Path(classifier_path)
    held_out, ckpt_time = None, None
    if ckpt_path.exists():
        ckpt_time = datetime.fromtimestamp(ckpt_path.stat().st_mtime)
        try:
            import torch
            meta = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
            names = meta.get("val_session_names")
            held_out = set(names) if names else None
        except Exception:
            held_out = None

    if held_out is None:
        held_out = CLASSIFIER_UNSEEN          # pre-metadata checkpoint
    if session_name in held_out:
        return True

    # Recorded after the weights were written => cannot be training data.
    if ckpt_time is not None:
        try:
            stamp = "_".join(session_name.split("_")[1:3])
            if datetime.strptime(stamp, "%Y%m%d_%H%M%S") > ckpt_time:
                return True
        except (ValueError, IndexError):
            pass
    return False


def confusion_breakdown(dirs: list[Path], args) -> dict:
    """
    D2's calibration measurement (PATH_TO_VERIFICATION.md §5): what kind of
    cube-relation does a classifier substitution error have, on
    classifier-UNSEEN sessions only (measuring on training sessions would
    understate real confusion — the model has already seen those errors).

    Pooled into 3 structural buckets rather than a full 12x12 confusion
    matrix — only ~4 unseen sessions exist, nowhere near enough to fit 144
    free cells without massively overfitting:

      inverse   same face, opposite direction — BLEND_INV's target already
      adjacent  one of the 4 faces sharing an edge with the true face —
                the measured LIVE error axis (classifier-error-axis-live
                memory), BLEND_ADJ's target
      opposite  the one face with no shared edge (U/D, L/R, F/B) — rare;
                not separately priced, folds into blend_unif's uniform tail

    Returns counts plus a SUGGESTED (blend_inv, blend_adj) split: the total
    reserved prior mass stays at the historically-tuned BLEND_INV (0.20),
    reallocated between the inverse and adjacent buckets in proportion to
    how often each was actually observed — this is a reallocation of
    already-calibrated mass, not new mass invented from nothing.
    """
    counts = {"inverse": 0, "adjacent": 0, "opposite": 0, "other": 0}
    total_sub = 0
    n_sessions = 0
    for d in dirs:
        if not classifier_unseen(d.name, args.classifier):
            continue
        gt_moves = [json.loads(l) for l in open(d / "moves.jsonl")
                    if l.strip()]
        gt = [m.get("wca_notation") for m in gt_moves]
        if not gt or any(g is None for g in gt):
            continue
        replay = _load_replay(d, args)
        if replay is None or not replay["moves"]:
            continue
        n_sessions += 1
        pred_names = [m["move"] for m in replay["moves"]]
        for op, truth, pred in align_sequences(gt, pred_names):
            if op != "sub":
                continue
            total_sub += 1
            tk, pk = WCA12.index(truth), WCA12.index(pred)
            if pk == INV12[tk]:
                counts["inverse"] += 1
            elif CLASS_FACE[pk] == (CLASS_FACE[tk] + 3) % 6:
                counts["opposite"] += 1
            elif CLASS_FACE[pk] != CLASS_FACE[tk]:
                counts["adjacent"] += 1
            else:
                counts["other"] += 1   # same face, same direction: impossible
                                        # unless truth==pred, i.e. never a sub

    print(f"\n  CONFUSION BREAKDOWN — {n_sessions} classifier-unseen "
          f"session(s), {total_sub} substitution errors")
    if total_sub == 0:
        print(f"    no substitution errors found — nothing to calibrate")
        return {"total_sub": 0, "counts": counts}
    for k in ("inverse", "adjacent", "opposite", "other"):
        print(f"    {k:<10} {counts[k]:>4}   ({counts[k]/total_sub*100:5.1f}%)")

    inv_frac = counts["inverse"] / total_sub
    adj_frac = counts["adjacent"] / total_sub
    reserve = BLEND_INV   # keep total structured mass at the tuned scale
    denom = inv_frac + adj_frac
    if denom > 0:
        sug_inv = reserve * inv_frac / denom
        sug_adj = reserve * adj_frac / denom
    else:
        sug_inv, sug_adj = BLEND_INV, 0.0
    print(f"\n    suggested: --blend-inv {sug_inv:.3f} --blend-adj "
          f"{sug_adj:.3f}\n    (reallocates the existing {BLEND_INV:.2f} "
          f"reserved mass by observed inverse:adjacent ratio;\n     "
          f"re-run the honest-subset sweep AND the falsifiability sweep "
          f"with these before trusting them)")
    return {"total_sub": total_sub, "counts": counts,
            "suggested_blend_inv": sug_inv, "suggested_blend_adj": sug_adj}


def _decode_dispatch(start: np.ndarray, end: np.ndarray,
                     cost_rows: list[np.ndarray], del_costs: np.ndarray,
                     args, tables: dict, beam: int,
                     slice_rows: list[np.ndarray] | None = None) -> dict:
    """
    Single-pass decode() (default) or D3 bidirectional (--bidir, optionally
    --meet-sweep) at the given `beam` — shared by run_sessions and any
    other caller that wants --bidir to be a drop-in swap for the existing
    decode() call, same args namespace, same retry-at-wider-beam pattern.
    """
    slices = bool(getattr(args, "slices", False))
    c_slice = float(getattr(args, "c_slice", C_SLICE))
    if not getattr(args, "bidir", False):
        return decode(start, cost_rows, beam=beam, c_del=args.del_cost,
                     c_ins=args.ins_cost, c_rot=args.rot_cost,
                     max_end_ins=args.max_end_ins,
                     rel_weight=args.rel_weight, del_costs=del_costs,
                     rotations=args.rotations, tables=tables,
                     slices=slices, c_slice=c_slice, slice_rows=slice_rows)
    if getattr(args, "rotations", False):
        sys.exit("--bidir does not support --rotations (see "
                 "decode_bidirectional's docstring)")
    kw = dict(beam=beam, c_del=args.del_cost, c_ins=args.ins_cost,
             c_rot=args.rot_cost, max_end_ins=args.max_end_ins,
             rel_weight=args.rel_weight, tables=tables,
             slices=slices, c_slice=c_slice, slice_rows=slice_rows)
    if getattr(args, "meet_sweep", False):
        return decode_bidirectional_sweep(start, end, cost_rows, del_costs,
                                          **kw)
    meet = getattr(args, "meet", None)
    return decode_bidirectional(start, end, cost_rows, del_costs,
                                meet=meet, **kw)


def run_sessions(args) -> None:
    dirs = [Path(p) for pattern in args.session
            for p in (Path(".").glob(pattern) if "*" in pattern
                      else [Path(pattern)])]
    dirs = sorted(d for d in dirs if d.is_dir())
    if not dirs:
        sys.exit("No session directories matched --session.")

    tables = build_tables()
    rows = []
    for d in dirs:
        gt_moves = [json.loads(l) for l in open(d / "moves.jsonl")
                    if l.strip()]
        gt = [m.get("wca_notation") for m in gt_moves]
        if not gt or any(g is None for g in gt):
            print(f"\n  {d.name}: missing wca_notation — skipping")
            continue

        replay = _load_replay(d, args)
        if replay is None:
            continue
        if not replay["moves"]:
            print(f"\n  {d.name}: no onsets detected — skipping")
            continue
        if replay["class_names"] != WCA12:
            sys.exit(f"classifier class order {replay['class_names']} != "
                     f"reconstruct WCA12 — refusing to decode")

        pred_names, cost_rows, del_costs = costs_from_moves(
            replay["moves"], replay["meta"].get("threshold", 0.5),
            args.blend_inv, args.blend_unif, args.del_cost,
            getattr(args, "candidate_threshold", None),
            getattr(args, "del_floor", DEL_FLOOR),
            getattr(args, "blend_adj", BLEND_ADJ))
        start = start_from_gt(gt)

        raw = score_vs_gt(gt, pred_names)
        raw_solved = bool((compose_seq(start, pred_names) == SOLVED).all())

        res = _decode_dispatch(start, SOLVED.copy(), cost_rows, del_costs,
                               args, tables, beam=args.beam)
        if not res["solved"] and args.retry_beam > args.beam:
            print(f"  {d.name}: not solved at beam {args.beam}, "
                  f"retrying at {args.retry_beam}")
            res = _decode_dispatch(start, SOLVED.copy(), cost_rows,
                                   del_costs, args, tables,
                                   beam=args.retry_beam)

        row = {"session": d.name,
               "unseen": classifier_unseen(d.name, args.classifier),
               "gt_n": len(gt), "onsets": len(pred_names),
               "raw_acc": raw["acc"], "raw_solved": raw_solved,
               "solved": res["solved"], "time": res["decode_seconds"]}
        if res["solved"]:
            rec = score_vs_gt(gt, res["moves"])
            gtc = gt_path_cost(gt, pred_names, cost_rows,
                               del_costs, args.ins_cost,
                               bool(getattr(args, "slices", False)),
                               float(getattr(args, "c_slice", C_SLICE)))
            row.update({"acc": rec["acc"], "exact": rec["exact"],
                        "cost": res["cost"], "gt_cost": gtc,
                        "ops": res["op_counts"]})
        else:
            row.update({"acc": 0.0, "exact": False, "cost": None,
                        "gt_cost": None, "min_h": res.get("min_h_final")})
        rows.append(row)
        _print_session_row(row)

    _print_summary(rows)


def compose_seq(state: np.ndarray, names: list[str]) -> np.ndarray:
    s = state.copy()
    for n in names:
        s = compose(s, CLASS_VECS[WCA12.index(n)])
    return s


def _print_session_row(r: dict) -> None:
    tag = "unseen" if r["unseen"] else "train "
    print(f"\n  {r['session']}  [{tag}]  "
          f"{r['gt_n']} GT moves, {r['onsets']} onsets")
    print(f"    raw:     acc {r['raw_acc']*100:5.1f}%   "
          f"solves cube: {'yes' if r['raw_solved'] else 'no'}")
    if r["solved"]:
        verdict = "exact" if r["exact"] else f"acc {r['acc']*100:.1f}%"
        gap = r["cost"] - r["gt_cost"]
        side = ("= GT path" if abs(gap) < 1e-6 else
                f"model prefers cheaper story (gap {gap:+.2f})" if gap < 0
                else f"beam lost truth (gap {gap:+.2f})")
        ops = r.get("ops", {})
        print(f"    decoded: {verdict}   cost {r['cost']:.2f} vs GT "
              f"{r['gt_cost']:.2f}  [{side}]")
        print(f"             ops: {ops}   ({r['time']:.1f}s)")
    else:
        print(f"    decoded: NOT SOLVED  (closest end state h="
              f"{r.get('min_h')}; raise --beam or --max-end-ins)  "
              f"({r['time']:.1f}s)")


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n{'='*70}")
    for subset, label in ((rows, "ALL SESSIONS"),
                          ([r for r in rows if r["unseen"]],
                           "CLASSIFIER-UNSEEN ONLY (honest number)")):
        if not subset:
            continue
        n = len(subset)
        solved = sum(r["solved"] for r in subset)
        exact = sum(r.get("exact", False) for r in subset)
        raw_acc = np.mean([r["raw_acc"] for r in subset])
        # When the decode fails the system keeps the raw sequence, so
        # that is the accuracy the pipeline actually delivers.
        eff_acc = np.mean([r["acc"] if r["solved"] else r["raw_acc"]
                           for r in subset])
        print(f"  {label}  ({n} sessions)")
        print(f"    verified solved:      {solved}/{n}")
        print(f"    exact reconstruction: {exact}/{n}")
        print(f"    per-move accuracy:    raw {raw_acc*100:.1f}%  ->  "
              f"system {eff_acc*100:.1f}%   ({(eff_acc-raw_acc)*100:+.1f} pts)")
        just = [r for r in subset if r["solved"]]
        if just:
            ra = np.mean([r["raw_acc"] for r in just])
            da = np.mean([r["acc"] for r in just])
            print(f"    on solved sessions:   raw {ra*100:.1f}%  ->  "
                  f"decoded {da*100:.1f}%   ({(da-ra)*100:+.1f} pts, "
                  f"{sum(r['exact'] for r in just)}/{len(just)} exact)")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Synthetic evaluation — controllable noise, exercises every edit type
# ---------------------------------------------------------------------------

def _random_gt(rng: np.random.Generator, n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        k = int(rng.integers(12))
        if out and CLASS_FACE[k] == CLASS_FACE[out[-1]]:
            continue
        out.append(k)
    return out


def _fake_probs(rng, true_k: int | None, pred_k: int, conf: float
                ) -> np.ndarray:
    p = np.full(12, (1 - conf) / 11)
    p[pred_k] = conf
    if true_k is not None and true_k != pred_k:
        # give truth a plausible share of the leftover mass
        extra = (1 - conf) * 0.6
        p *= (1 - conf - extra) / max(1 - conf, 1e-9)
        p[pred_k] = conf
        p[true_k] += extra
    return p / p.sum()


def run_synthetic(args) -> None:
    rng = np.random.default_rng(args.seed)
    tables = build_tables()
    n_solved = n_exact = 0
    accs = []
    t0 = time.time()
    for _ in range(args.synthetic):
        gt_cube = _random_gt(rng, args.moves)

        sid = 0
        obs: list[np.ndarray] = []
        scores: list[float] = []
        for k in gt_cube:
            if args.rotations and rng.random() < args.err_rot:
                sid = rotate_sigma(sid, ROT_NAMES[int(rng.integers(6))])
            if rng.random() < args.err_del:
                continue                              # detector missed it
            cam_k = camera_class_for(sid, k)
            r = rng.random()
            if r < args.err_inv:
                pred = INV12[cam_k]
                conf = rng.uniform(0.5, 0.98)
            elif r < args.err_inv + args.err_sub:
                pred = int(rng.integers(12))
                conf = rng.uniform(0.3, 0.8)
            else:
                pred = cam_k
                # Matches the measured profile on real replays: correct
                # calls are mostly very confident with a modest low tail
                # (conf_ok median ~0.93-0.999, p10 ~0.54-0.99).
                conf = rng.uniform(0.9, 0.999) if rng.random() > 0.15 \
                    else rng.uniform(0.55, 0.9)
            obs.append(_fake_probs(rng, cam_k, pred, conf))
            scores.append(rng.uniform(0.75, 1.0))     # real onsets score high
            if rng.random() < args.err_ins:           # phantom onset
                obs.append(_fake_probs(rng, None, int(rng.integers(12)),
                                       rng.uniform(0.3, 0.9)))
                scores.append(rng.uniform(0.3, 0.6))  # phantoms score weak

        gt_names = [WCA12[k] for k in gt_cube]
        start = start_from_gt(gt_names)
        cost_rows = [onset_costs(p, args.blend_inv, args.blend_unif)
                     for p in obs]
        del_costs = score_del_costs(scores, 0.25, args.del_cost)
        res = decode(start, cost_rows, beam=args.beam, c_del=args.del_cost,
                     c_ins=args.ins_cost, c_rot=args.rot_cost,
                     max_end_ins=args.max_end_ins, del_costs=del_costs,
                     rel_weight=args.rel_weight,
                     rotations=args.rotations, tables=tables)
        if res["solved"]:
            n_solved += 1
            s = score_vs_gt(gt_names, res["moves"])
            n_exact += s["exact"]
            accs.append(s["acc"])
        else:
            accs.append(0.0)

    n = args.synthetic
    print(f"\n  SYNTHETIC  {n} trials x {args.moves} moves   "
          f"(inv {args.err_inv} sub {args.err_sub} del {args.err_del} "
          f"ins {args.err_ins} rot {args.err_rot if args.rotations else 0})")
    print(f"    verified solved:      {n_solved}/{n}")
    print(f"    exact reconstruction: {n_exact}/{n}")
    print(f"    mean per-move acc:    {np.mean(accs)*100:.1f}%")
    print(f"    total time:           {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Selftest — pure logic, no models, no sessions
# ---------------------------------------------------------------------------

def run_selftest() -> None:
    failures = []

    def check(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            failures.append(name)

    # group identities
    check("R4 = I", (seq_to_state(["R"] * 4) == SOLVED).all())
    check("(R U R' U')^6 = I",
          (seq_to_state(["R", "U", "R'", "U'"] * 6) == SOLVED).all())
    check("every move times its inverse = I",
          all((compose(CLASS_VECS[k], CLASS_VECS[INV12[k]]) == SOLVED).all()
              for k in range(12)))

    # coordinates match the vendored getters on random states
    rng = np.random.default_rng(7)
    states = np.stack([seq_to_state([WCA12[k] for k in
                                     rng.integers(0, 12, size=30)])
                       for _ in range(50)])
    cubies = [_vec_to_cubie(s) for s in states]
    check("twist matches CubieCube",
          (coord_twist(states) ==
           np.array([c.twist for c in cubies])).all())
    check("flip matches CubieCube",
          (coord_flip(states) == np.array([c.flip for c in cubies])).all())
    check("udslice matches CubieCube",
          (coord_udslice(states) ==
           np.array([c.udslice for c in cubies])).all())
    check("corner matches CubieCube",
          (coord_corner(states) ==
           np.array([c.corner for c in cubies])).all())
    check("corner parity matches CubieCube",
          (corner_parity(states) ==
           np.array([c.corner_parity for c in cubies])).all())

    # pattern databases
    tables = build_tables()
    check("h(solved) = 0", h_full(SOLVED, tables) == 0)
    check("h(one move) <= 1 and > 0",
          all(0 < h_full(CLASS_VECS[k], tables) <= 1 for k in range(12)))
    superflip = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
    # expand half turns into quarter-turn pairs for our 12-move alphabet
    sf = []
    for tok in superflip.split():
        if tok.endswith("2"):
            sf += [tok[0], tok[0]]
        else:
            sf.append(tok)
    check("h(superflip) sensible (<= 24, >= 6)",
          6 <= h_full(seq_to_state(sf), tables) <= 24)

    # rotations form a group of order 24 and relabel moves correctly
    seen = {0}
    frontier = [0]
    while frontier:
        nxt = []
        for sid in frontier:
            for r in ROT_NAMES:
                t = rotate_sigma(sid, r)
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
        frontier = nxt
    check("rotation group has order 24", len(seen) == 24)
    y = rotate_sigma(0, "y")
    check("after y, camera F is cube R",
          (move_for(y, WCA12.index("F")) ==
           CLASS_VECS[WCA12.index("R")]).all())
    check("camera_class_for inverts move_for",
          all(camera_class_for(y, k) == c
              for c in range(12)
              for k in [next(kk for kk in range(12)
                             if (move_for(y, c) == CLASS_VECS[kk]).all())]))

    # end-completion finds a known 3-move finish
    tail = ["R", "U", "F'"]
    state = inverse(seq_to_state(tail))
    word = _completion(state, 4, tables)
    check("completion solves a 3-move tail",
          word is not None and
          (compose_seq(state, [WCA12[k] for k in word]) == SOLVED).all())

    # known corruptions are recovered exactly. Four errors of all three
    # kinds in one 40-move sequence is past the measured envelope (see the
    # module docstring's limits note), so the two cases here stay at
    # three errors each — both verified stable at this beam width.
    rng = np.random.default_rng(11)
    gt = [WCA12[k] for k in _random_gt(rng, 40)]
    start = start_from_gt(gt)

    def corrupt(miss=False, inv=(), phantom=False, miss_at=7):
        rng2 = np.random.default_rng(5)
        obs, scores = [], []
        for i, name in enumerate(gt):
            k = WCA12.index(name)
            if miss and i == miss_at:
                continue                               # a missed onset
            if i in inv:                               # inverse errors
                obs.append(_fake_probs(rng2, k, INV12[k],
                                       rng2.uniform(0.5, 0.95)))
            else:
                obs.append(_fake_probs(rng2, k, k,
                                       rng2.uniform(0.97, 0.999)))
            scores.append(rng2.uniform(0.75, 1.0))
            if phantom and i == 30:                    # a spurious onset
                obs.append(_fake_probs(rng2, None, 3, 0.6))
                scores.append(rng2.uniform(0.3, 0.6))
        return [onset_costs(p) for p in obs], score_del_costs(scores, 0.25)

    for label, kw in (("miss + 2 inverses", dict(miss=True, inv=(12, 25))),
                      ("miss + inverse + phantom",
                       dict(miss=True, inv=(12,), phantom=True)),
                      ("6 inverses",
                       dict(inv=(5, 9, 12, 25, 33, 37)))):
        rows, dc = corrupt(**kw)
        res = decode(start, rows, beam=512, del_costs=dc, tables=tables)
        check(f"recovers {label} exactly",
              res["solved"] and res["moves"] == gt)

    # --- OP_SLICE ---------------------------------------------------------
    # Table sanity first: the partner relation must be an involution, and
    # both halves must compose to the same slice (they commute).
    check("slice partner is an involution",
          all(SLICE_PARTNER[SLICE_PARTNER[k]] == k for k in range(12)))
    check("slice is the same from either half",
          all((SLICE_VECS[k] == SLICE_VECS[SLICE_PARTNER[k]]).all()
              for k in range(12)))
    check("slice^4 = I",
          all((seq_to_state([WCA12[k], WCA12[SLICE_PARTNER[k]]] * 4)
               == SOLVED).all() for k in range(12)))

    # The real case: a sequence containing a slice, where the camera saw
    # only ONE onset for it. Without --slices this is a forced insertion;
    # with it the decoder should recover the pair exactly.
    rng3 = np.random.default_rng(23)
    base = [WCA12[k] for k in _random_gt(rng3, 30)]
    slice_at, k_slice = 12, WCA12.index("R")
    gt_sl = (base[:slice_at] + ["R", "L'"] + base[slice_at:])
    start_sl = start_from_gt(gt_sl)
    obs_sl, sc_sl = [], []
    rng4 = np.random.default_rng(5)
    for i, name in enumerate(base):
        if i == slice_at:                       # ONE onset for the slice
            obs_sl.append(_fake_probs(rng4, k_slice, k_slice, 0.98))
            sc_sl.append(0.9)
        obs_sl.append(_fake_probs(rng4, WCA12.index(name),
                                  WCA12.index(name), 0.98))
        sc_sl.append(0.9)
    rows_sl = [onset_costs(p) for p in obs_sl]
    dc_sl = score_del_costs(sc_sl, 0.25)

    # The gate must admit a genuinely bimodal onset and reject a confident
    # one — this is what keeps OP_SLICE from being offered 12 ways at every
    # onset (see SLICE_GATE).
    bimodal = np.zeros(12)
    bimodal[k_slice] = 0.55
    bimodal[SLICE_PARTNER[k_slice]] = 0.40
    bimodal[0] = 0.05
    check("gate admits a bimodal slice onset",
          bool(slice_mask(bimodal)[k_slice]))
    check("gate rejects a confident single turn",
          not slice_mask(_fake_probs(np.random.default_rng(1), k_slice,
                                     k_slice, 0.98)).any())

    res_off = decode(start_sl, rows_sl, beam=512, del_costs=dc_sl,
                     tables=tables)
    res_on = decode(start_sl, rows_sl, beam=512, del_costs=dc_sl,
                    tables=tables, slices=True)
    check("slices=True recovers a one-onset slice exactly",
          res_on["solved"] and res_on["moves"] == gt_sl)
    check("slices=True is cheaper here than slices=False",
          res_on["solved"] and (not res_off["solved"]
                                or res_on["cost"] < res_off["cost"]))

    # Default-off must be bit-identical to the prior behaviour, on a case
    # with no slice in it at all — this is the regression guard that says
    # adding the transition changed nothing for existing callers.
    rows_c, dc_c = corrupt(miss=True, inv=(12, 25))
    a = decode(start, rows_c, beam=512, del_costs=dc_c, tables=tables)
    b = decode(start, rows_c, beam=512, del_costs=dc_c, tables=tables,
               slices=False)
    check("slices=False matches the prior decode exactly",
          a["solved"] == b["solved"] and a["moves"] == b["moves"]
          and a["cost"] == b["cost"])

    # -- OP_ALGO (ALGORITHM_PRIOR.md §7) ---------------------------------
    #
    # A cube needing only a T-perm: F2L complete, four last-layer pieces
    # out. The detector drops FIVE of the fifteen onsets, which is well
    # past what plain insertions can buy back at this beam — exactly the
    # miss-heavy last-layer regime the transition exists for.
    tperm = ["R", "U", "R'", "U'", "R'", "F", "R", "R",
             "U'", "R'", "U'", "R", "U", "R'", "F'"]
    t_labels = [WCA12.index(m) for m in tperm]
    start_al = inverse(seq_to_state(tperm))
    check("T-perm start has F2L complete", bool(f2l_complete(start_al).any()))
    check("T-perm start is not solved", not (start_al == SOLVED).all())

    kept = [i for i in range(len(tperm)) if i not in (2, 5, 8, 11, 13)]
    rng_al = np.random.default_rng(11)
    rows_al = [onset_costs(_fake_probs(rng_al, t_labels[i], t_labels[i],
                                       0.98)) for i in kept]
    dc_al = score_del_costs([0.9] * len(kept), 0.25)
    cand = [{"start": 0, "k": len(kept), "labels": t_labels, "cost": 2.0}]

    al_off = decode(start_al, rows_al, beam=512, del_costs=dc_al,
                    tables=tables)
    al_on = decode(start_al, rows_al, beam=512, del_costs=dc_al,
                   tables=tables, algo_cands=cand)
    check("OP_ALGO fills in the dropped onsets exactly",
          al_on["solved"] and al_on["moves"] == tperm)
    check("OP_ALGO actually fired",
          al_on["solved"] and al_on["op_counts"].get("algorithm", 0) == 1)
    check("OP_ALGO is cheaper than paying five insertions",
          al_on["solved"] and (not al_off["solved"]
                               or al_on["cost"] < al_off["cost"]))

    # The gate is the abstract state, not the candidate list. Offering the
    # same word from a cube whose F2L is broken must be refused outright —
    # this is what stops 14 near-free multi-move jumps from flooding the
    # beam the way ungated slices did.
    broken = compose(start_al, CLASS_VECS[WCA12.index("R")])
    check("F2L-broken state fails the abstract gate",
          not bool(f2l_complete(broken).any()))
    n_before = n_solved(start_al[None, :])[0]
    check("applying the word makes strict progress",
          n_solved(compose(start_al, seq_to_state(tperm))[None, :])[0]
          > n_before)

    # Default-off must be bit-identical, same guard as slices.
    c_off = decode(start, rows_c, beam=512, del_costs=dc_c, tables=tables,
                   algo_cands=None)
    check("algo_cands=None matches the prior decode exactly",
          a["solved"] == c_off["solved"] and a["moves"] == c_off["moves"]
          and a["cost"] == c_off["cost"])

    # A candidate whose word does NOT solve what it claims must not be
    # taken just because it is cheap: the state gate refuses it.
    sexy = ["R", "U", "R'", "U'", "R", "U", "R'"]
    bogus = [{"start": 0, "k": len(kept), "cost": 0.1,
              "labels": [WCA12.index(m) for m in sexy]}]
    al_bad = decode(start_al, rows_al, beam=512, del_costs=dc_al,
                    tables=tables, algo_cands=bogus)
    check("a wrong cheap word does not hijack the decode",
          (not al_bad["solved"]) or al_bad["moves"] == tperm
          or al_bad["op_counts"].get("algorithm", 0) == 0)

    # D3 bidirectional (PATH_TO_VERIFICATION.md §5): clean sequence first —
    # this exercises the INV12 class-remapping and the reversed-order join
    # with no error-repair pressure to mask a mapping bug. Beam is small
    # (256/side) precisely because each half only has to survive HALF the
    # true story's cost, which is the whole premise of D3.
    clean_rows = [onset_costs(_fake_probs(np.random.default_rng(3), k, k,
                                          0.98)) for k in
                 [WCA12.index(n) for n in gt]]
    clean_dc = score_del_costs([0.9] * len(gt), 0.25)
    bd = decode_bidirectional(start, SOLVED.copy(), clean_rows, clean_dc,
                              beam=256, tables=tables)
    check("bidirectional recovers a clean sequence exactly",
          bd["solved"] and bd["moves"] == gt)

    # bidirectional drops the consistency-ladder/capacity/parity guidance
    # (see _rank's use_bounds docstring — those are all identity-target
    # calibrated and wrong for a floating meet point), trading search
    # quality for correctness on an unknown target. That costs real beam
    # width: a miss + 2 inverses recovers at beam 2048/side here, not 512
    # (confirmed by direct sweep — not a logic bug, purely beam pressure
    # from the weaker, target-free ranking). Still far below what an
    # equivalent single-pass decode needs for comparable error mass on a
    # much longer real sequence — see PATH_TO_VERIFICATION.md §5's
    # beam-64000 acceptance test for the real comparison.
    for label, kw, beam in (("miss + 2 inverses",
                             dict(miss=True, inv=(12, 25)), 2048),
                            ("6 inverses",
                             dict(inv=(5, 9, 12, 25, 33, 37)), 512)):
        rows, dc = corrupt(**kw)
        bd = decode_bidirectional(start, SOLVED.copy(), rows, dc,
                                  beam=beam, tables=tables)
        check(f"bidirectional recovers {label} exactly",
              bd["solved"] and bd["moves"] == gt)

    # A miss in the BACKWARD half specifically — the forward-half cases
    # above only exercise the backward beam's ACCEPT-with-override
    # remapping (via the inv=(...,25) entries), never its INSERT
    # remapping. _ops_to_moves/the backward-remap loop treat accept and
    # insert identically, but this is exactly the kind of direction-flip
    # bug that is easy to get backwards without a dedicated case.
    rows, dc = corrupt(miss=True, miss_at=32, inv=(5,))
    bd = decode_bidirectional(start, SOLVED.copy(), rows, dc, beam=2048,
                              tables=tables)
    check("bidirectional recovers a backward-half miss exactly",
          bd["solved"] and bd["moves"] == gt)

    # Different meet points must all land on the same true story.
    rows, dc = corrupt(inv=(5, 9, 12))
    meets_ok = all(
        (lambda r: r["solved"] and r["moves"] == gt)(
            decode_bidirectional(start, SOLVED.copy(), rows, dc, beam=512,
                                 meet=m, tables=tables))
        for m in (10, 20, 30))
    check("bidirectional agrees across different meet points", meets_ok)

    print(f"\n  {'ALL PASS' if not failures else f'{len(failures)} FAILURE(S)'}")
    sys.exit(1 if failures else 0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Group-theoretic decode of detected moves against "
                    "known start/end states")
    p.add_argument("--session", nargs="+",
                   help="Session folder(s) to replay + decode "
                        "(globs ok: ../training_data/solve_*/)")
    p.add_argument("--detector", type=str, default="checkpoints/move_detector_all28.pt")
    p.add_argument("--classifier", type=str,
                   default="../../legacy/move_classifier_rnd/move_classifier_all39_jitter.pt",
                   help="the old ResNet-18 classifier track, archived "
                        "2026-08-03 — this CLI entry point predates the "
                        "joint/CTC models and is superseded by algo_sweep.py "
                        "/ review.py, which import reconstruct.py directly")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Re-run the replay even if a cache exists")
    p.add_argument("--beam", type=int, default=BEAM)
    p.add_argument("--retry-beam", type=int, default=4 * BEAM,
                   help="Wider beam to retry a session that did not solve "
                        "(session mode only; set <= --beam to disable)")
    p.add_argument("--del-cost", type=float, default=C_DEL)
    p.add_argument("--ins-cost", type=float, default=C_INS)
    p.add_argument("--slices", action="store_true",
                   help="Let one onset decode as a middle-slice turn, "
                        "emitting two layer moves. The smart cube reports a "
                        "slice as two same-timestamp face events but the "
                        "camera only ever sees one motion, so without this "
                        "the second half is an unavoidable insertion. Off by "
                        "default = exact prior behaviour.")
    p.add_argument("--c-slice", type=float, default=C_SLICE, dest="c_slice",
                   help="Cost of a slice reading, on top of the onset's own "
                        "acceptance cost (default %(default)s)")
    p.add_argument("--rot-cost", type=float, default=C_ROT)
    p.add_argument("--max-end-ins", type=int, default=MAX_END_INS)
    p.add_argument("--candidate-threshold", type=float, default=None,
                   help="D1 soft-onset lattice (PATH_TO_VERIFICATION.md "
                        "§5): peak-pick onsets down to this threshold "
                        "(below the deployed operating point) and feed "
                        "them to the decoder with a near-free deletion "
                        "cost instead of never detecting them at all. "
                        "Try 0.20-0.30. Default None reproduces the exact "
                        "prior behaviour.")
    p.add_argument("--del-floor", type=float, default=DEL_FLOOR,
                   help="Deletion cost floor for a --candidate-threshold "
                        "onset near the candidate threshold")
    p.add_argument("--blend-inv", type=float, default=BLEND_INV)
    p.add_argument("--blend-unif", type=float, default=BLEND_UNIF)
    p.add_argument("--blend-adj", type=float, default=BLEND_ADJ,
                   help="D2 confusion-calibrated prior: mass reserved for "
                        "the 8 adjacent-face classes (see --confusion to "
                        "measure a session-grounded value). Default 0.0: "
                        "unchanged behaviour.")
    p.add_argument("--confusion", action="store_true",
                   help="Measure the sub-error confusion structure "
                        "(inverse / adjacent-face / opposite-face) on "
                        "classifier-unseen sessions and suggest --blend-inv "
                        "/ --blend-adj values, then exit. Needs --session.")
    p.add_argument("--rel-weight", type=float, default=REL_WEIGHT,
                   help="Ranking weight per unit of suffix-consistency "
                        "residual (0 disables the lookahead)")
    p.add_argument("--rotations", action="store_true",
                   help="Track whole-cube rotations as search state")
    p.add_argument("--bidir", action="store_true",
                   help="D3 bidirectional meet-in-the-middle decode "
                        "(PATH_TO_VERIFICATION.md §5) instead of the "
                        "single-pass decoder. Not compatible with "
                        "--rotations.")
    p.add_argument("--meet", type=int, default=None,
                   help="--bidir: onset index to meet at (default n//2)")
    p.add_argument("--meet-sweep", action="store_true",
                   help="--bidir: try meet points at n/3, n/2, 2n/3 and "
                        "keep the cheapest solved result")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="Decode N synthetically corrupted sequences")
    p.add_argument("--moves", type=int, default=80,
                   help="Synthetic ground-truth length")
    # Defaults mirror the noise measured on the well-lit recorded
    # sessions (see the noise-profile survey in the module history):
    # subs dominate, misses are rare, phantoms were zero across all 12
    # frame-bearing sessions. Crank these up to explore the envelope.
    p.add_argument("--err-inv", type=float, default=0.03)
    p.add_argument("--err-sub", type=float, default=0.01)
    p.add_argument("--err-del", type=float, default=0.01)
    p.add_argument("--err-ins", type=float, default=0.005)
    p.add_argument("--err-rot", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.selftest:
        run_selftest()
    elif args.synthetic:
        run_synthetic(args)
    elif args.confusion:
        if not args.session:
            sys.exit("--confusion needs --session ../training_data/solve_*/")
        dirs = sorted(d for pattern in args.session
                     for d in (Path(".").glob(pattern) if "*" in pattern
                               else [Path(pattern)]) if d.is_dir())
        confusion_breakdown(dirs, args)
    elif args.session:
        run_sessions(args)
    else:
        p.print_help()
