"""
review.py

Per-move human review of a decoded solve, for turning video into training
labels without a smart cube.

Why per-move and not per-solve
------------------------------
At ~90% per-move accuracy a 110-move solve has ~11 wrong moves, so a
whole-sequence "is this right" judgement is almost always "no" and carries
almost no information. Reviewing move by move with the UI defaulting to
ACCEPT inverts that: the reviewer only touches the ~11 that are wrong, and
a solve costs a couple of minutes. This is `cv/labeling/autolabel.py`'s
pattern (confident proposals, review by exception) applied to a temporal
sequence instead of bounding boxes.

The flaw that pattern has here, and the three things that fix it
----------------------------------------------------------------
A reviewer who only sees DETECTIONS can reject a phantom and rename a
substitution, but a MISS renders as nothing at all — there is no widget to
click on where the model failed to fire. That is not a cosmetic gap.
Measured over this project's 12-session algorithm-prior evaluation set
(results/2026-08-02/algo8_s0.json / results/2026-08-03/algo8_s1.json, 1212 ground-truth moves, ~90.5% raw
accuracy, both seeds), the error mass splits:

                 seed 0        seed 1
    miss          74 (52%)     69 (42%)
    substitution  52 (37%)     68 (41%)
    phantom       15 (11%)     29 (17%)

So roughly half the error mass is invisible to a detection-only reviewer.
(An earlier informal estimate of 60/21/19 circulated for this split; the
artifacts above are what the corpus actually measures, and substitutions
are a much bigger share of it than that estimate had.)

And a miss is worse than merely uncorrected. For EVALUATION a miss is a
gap; for TRAINING LABELS a miss is poison — a label set that omits a real
move actively teaches the model not to fire there, reinforcing the single
largest term in the error budget. So misses have to be made reviewable
before the output of this tool is usable as supervision at all. Three
mechanisms here do that, in increasing order of leverage:

  1. SUB-THRESHOLD CANDIDATES (--candidate-threshold). Misses are mostly
     not "nothing there", they are peaks under the operating point.
     Dropping the peak-pick threshold trades precision for recall, which
     is a bad deal for deployment and a good one for review, because
     rejecting a phantom costs a reviewer half a second. These render as
     hollow ticks and default to REJECTED, so the extra false positives
     cost nothing unless the reviewer accepts one.

  2. A CONTINUOUS TIMELINE, not isolated onset snapshots. Isolated frames
     make absences invisible by construction. A scrubbable strip with the
     posteriorgram under it makes a miss visible as a gap: 800ms where the
     cube plainly turns and no column lights up.

  3. THE GROUP THEORY (the important one). Given the start state, the
     label sequence must take the cube to solved. When it does not,
     reconstruct.py's beam search already computes WHERE insertions are
     needed and WHICH move each must be. That converts "scan the whole
     video for something missing" into "check these four flagged spans",
     and it is the only one of the three that can find a move for which
     the model produced no evidence whatsoever.

The review head, and why mechanism 3 needs one
----------------------------------------------
Mechanism 3 does not work on a fresh session, and this is measured, not
anticipated. On held-out solve_20260730_111941_solve (106 true moves, 100
proposed by move_ctc_aug44_s0: 95 ok, 4 sub, 7 miss, 1 phantom) the
whole-sequence decode finds NO consistent story — at beam 4000 (39s), at
beam 16000 (150s), and at insertion costs of 4.0, 2.5 and 1.5 alike. It
ends 4 quarter turns short every time. So this is a search and information
limit, not a pricing one, and it is the same cliff verification hits
everywhere in this project: the decode is dependable at ~98% raw accuracy
and hopeless at ~90%, because a dozen simultaneous errors give the beam a
combinatorial number of equally-priced wrong stories. Shipping the flags
without dealing with that would mean shipping a feature that is blank
exactly when the session needs it most.

The fix is the REVIEW HEAD, and it falls out of how review actually works:
the reviewer moves left to right. Marking "I have reviewed up to here"
(`m`) asserts two things about everything before that point — every move
present is correctly named, and no move is missing. Together those pin the
CUBE STATE at the head, because state = start * (the reviewed prefix). The
decode then starts from that known mid-solve state and has to explain only
the unreviewed suffix, which shrinks in both length and error count with
every span reviewed. Same session, head advanced through it:

    head    suffix                    gt path cost   decode
      0     100 onsets, 12 errors        (global)    failed, 39s
     50      50 onsets,  5 errors                    failed, 20s
     75      25 onsets,  2 sub 3 miss       20.74    failed,  9s
     80      20 onsets,  1 sub 1 miss        7.65    SOLVED at 7.65, 7s
     88      12 onsets,  0 sub 1 miss        4.00    SOLVED at 4.00, 4s

Two things to read off that. The decode recovers the truth EXACTLY (its
cost equals the ground-truth path's) the moment the suffix is cheap
enough, and the threshold is a cost of roughly 8 — about two errors —
matching the envelope reconstruct.py's own docstring quotes. And the wall
at head 75 is not gradual: onsets 75-80 contain two missing moves and a
substitution inside one short window, which is precisely the interacting-
error case reconstruct.py documents as an information limit of prefix
decoding rather than a beam-width problem.

So the honest statement of what mechanism 3 buys: it does not find your
first ten errors, it finds your last two. Early in a session the timeline
and the sub-threshold candidates are what you have; the group theory
closes the tail, and the tail is exactly where a miss is hardest to spot
by eye because nothing is obviously wrong any more. The status bar always
says whether the flags are live, so this is visible rather than inferred.

Individual confirmations (`y`) push in the same direction more weakly: a
confirmed move is fed back as a LOCKED onset (acceptance cost 0 for its
class, --lock-cost for the other eleven, deletion likewise), which pins
one transition where the head pins an absolute state. The lock is finite
on purpose — the decoder may still overrule a human, and when it does that
is reported rather than silently discarded. A human and the group theory
disagreeing is a real event worth seeing.

The head is also the one control here that can be abused into meaning
nothing: dragging it to the end in one keystroke asserts a complete review
that never happened. That is deliberately not prevented and is instead
measured — the saved audit records how many labels lie inside the head
having never been individually confirmed.

Rubber-stamping, and how this measures it
-----------------------------------------
Default-accept invites confirming rather than checking, and a reviewer who
rubber-stamps produces labels that look reviewed and are not.
`--blind-rate` hides the model's answer on a random sample: the class
heatmap is blanked around the candidate and the proposed label reads `??`
until the reviewer commits an answer, which is then compared against what
the model said. Two agreement rates are reported at save time — blind and
sighted. If the model is ~90% accurate, blind agreement should land near
90%; sighted agreement of 99% against blind agreement of 90% means the
reviewer is confirming rather than checking, and the gap is the size of
the problem.

What this tool deliberately does NOT read
-----------------------------------------
`moves.jsonl` — the BLE ground-truth move list — is never loaded, even for
sessions that have one, because a review tool that can see the answer
measures nothing. The start state is a different matter: it is exactly
what a scan of the scramble gives you in deployment, so it may come from
BLE ground truth via --start-from-truth (which reads only the NET
transformation, never the sequence) for development on recorded sessions.
--start-moves is the deployment path: display a scramble, the reviewer
applies it, the start state follows from the word.

Usage
-----
For footage that is not already a recorded session, ingest_video.py turns a
video file into one and prepare_data.py --labels none scores it:

    python ingest_video.py --video take.mp4 --out ../captures --check
    python prepare_data.py --sessions ../captures/take/ --color --labels none

then:

    python review.py --session ../training_data/solve_20260721_103149 \\
        --model checkpoints/move_ctc_aug44_s0.pt --start-from-truth

    python review.py --session ../captures/take_07 \\
        --model checkpoints/move_ctc_aug44_s0.pt --start-moves "R U R' F2 D' L ..."

    python review.py --session ... --model ... --report      # no GUI
    python review.py --session ... --model ... --shot ui.png # render once

Run from inside move_detector/, same convention as the rest of the repo.
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import reconstruct as RC
from decode import peak_pick, MIN_SEP

WCA12 = RC.WCA12

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Acceptance cost charged for every class OTHER than the one a human
# confirmed, and the deletion cost of a confirmed onset. Finite, not
# infinite: see the module docstring on disputed flags. 8.0 is twice C_INS,
# so the decoder will insert two moves before it overrules one confirmation.
LOCK_COST = 8.0

# Peak-picking threshold for the extra sub-threshold candidate list. The
# deployed operating point is 0.5 (decode.THRESHOLD); this is the review
# lattice, not a deployment setting.
CANDIDATE_THRESHOLD = 0.10

# A candidate within this many frames of a decoder-emitted move is the same
# physical onset, not an extra one.
CLAIM_RADIUS = 2

# BGR face colours, matching the physical cube. Primes render dimmer.
FACE_BGR = {"U": (238, 238, 238), "D": (60, 220, 240), "L": (40, 140, 250),
            "R": (60, 60, 235), "F": (80, 200, 80), "B": (230, 150, 60)}


def class_bgr(name: str) -> tuple:
    b, g, r = FACE_BGR[name[0]]
    return (b // 2, g // 2, r // 2) if name.endswith("'") else (b, g, r)


# ---------------------------------------------------------------------------
# Candidates and flags
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One reviewable position in time."""
    frame: int
    probs: np.ndarray            # (12,) renormalised class posterior
    score: float                 # foreground strength (1 - blank) at `frame`
    proposed: str | None         # what the decoder said, None for weak/manual
    move: str                    # the current label
    origin: str                  # "decoder" | "weak" | "manual"
    accepted: bool               # in the label set?
    confirmed: bool = False      # a human has looked at this one
    blind: bool = False          # model's answer hidden until answered
    blind_answer: str | None = None   # what the reviewer said unaided
    # Whether the reviewer has typed anything into a blind candidate yet.
    # A blind item still holding the model's own proposal must not be
    # committable — that would score as agreement without the reviewer
    # having answered anything, which is the exact failure the blind audit
    # exists to detect.
    typed: bool = False

    @property
    def renamed(self) -> bool:
        return self.proposed is not None and self.move != self.proposed


@dataclass
class Flag:
    """Something the group-theoretic decode wants the reviewer to look at."""
    kind: str                    # insert | phantom | rename | slice | disputed
    lo: int                      # frame span to inspect (lo == hi for point)
    hi: int
    move: str | None             # the move the decoder says belongs there
    onset: int                   # index into the decoded onset list
    note: str = ""

    @property
    def at(self) -> int:
        return (self.lo + self.hi) // 2


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

class Session:
    """Frames, posteriorgram and decoded proposals for one recording."""

    def __init__(self, path: Path, model_path: Path, args):
        self.path = path
        recs = [json.loads(l) for l in
                open(path / "frames.jsonl") if l.strip()]
        # Index the frames that are actually ON DISK, exactly as
        # prepare_data.py does when it builds the stream. frames.jsonl can
        # list frames that were later deleted, and indexing the full list
        # would offset every posteriorgram lookup past the first gap — an
        # error that shows up as labels drifting off their onsets, with
        # nothing in any artifact to say so.
        present = [r for r in recs if (path / "frames" / r["file"]).exists()]
        if len(present) < len(recs):
            print(f"  {len(recs) - len(present)} indexed frames missing on "
                  f"disk; reviewing the {len(present)} present")
        self.frames_meta = present
        self.n_frames = len(self.frames_meta)
        self.frame_ts = np.array([f["ts"] for f in self.frames_meta])

        pg = load_posteriorgram(path, model_path, refresh=args.refresh_cache)
        self.onset_prob = pg["onset_prob"]
        self.class_prob = pg["class_prob"]
        self.crop = pg["crop"]
        self.fps = float(pg["fps"])
        self.ckpt_tag = str(pg["ckpt_tag"])
        self.model_type = str(pg["model_type"])
        # The class head's own foreground belief, which is what
        # joint_decode.posteriorgram_to_moves picks peaks on.
        self.fg = 1.0 - self.class_prob[:, 12]

        n = min(self.n_frames, len(self.fg))
        if n != self.n_frames:
            print(f"  note: {self.n_frames} jpegs but {len(self.fg)} stream "
                  f"frames — reviewing the first {n}")
            self.n_frames = n

        self._img_cache: dict[int, np.ndarray] = {}

    # -- imagery -----------------------------------------------------------

    def image(self, i: int):
        """The full camera frame (BGR), lazily decoded and cached."""
        import cv2
        i = int(np.clip(i, 0, self.n_frames - 1))
        if i not in self._img_cache:
            if len(self._img_cache) > 240:
                self._img_cache.clear()
            p = self.path / "frames" / self.frames_meta[i]["file"]
            self._img_cache[i] = cv2.imread(str(p))
        return self._img_cache[i]

    def model_view(self, i: int) -> np.ndarray:
        """The cropped frame the model actually saw at `i`."""
        return self.crop[int(np.clip(i, 0, len(self.crop) - 1))]

    def diff_view(self, i: int) -> np.ndarray:
        """Amplified inter-frame difference of the model's own input."""
        i = int(np.clip(i, 1, len(self.crop) - 1))
        d = np.abs(self.crop[i].astype(np.int16) -
                   self.crop[i - 1].astype(np.int16)) * 4
        return np.clip(d, 0, 255).astype(np.uint8)


def load_posteriorgram(session: Path, model_path: Path,
                       refresh: bool = False) -> dict:
    """
    Score a session with a joint/CTC checkpoint, cached to
    <session>/review_pg_<tag>.npz.

    Keyed by the checkpoint's own identity (stem + epoch), the same rule
    joint_decode.load_joint_replay uses, so a retrained checkpoint under an
    unchanged filename cannot silently serve a stale posteriorgram.
    """
    import torch
    from dataset import JointSessionStream
    from model import build_joint_from_ckpt, score_stream_joint

    stream_path = session / "detector_stream_color.npz"
    if not stream_path.exists():
        sys.exit(f"{session.name}: no detector_stream_color.npz — run "
                 f"prepare_data.py --color first.")

    ckpt = torch.load(model_path, map_location="cpu")
    tag = f"{model_path.stem}_e{ckpt.get('epoch', '?')}"
    cache = session / f"review_pg_{tag}.npz"

    stream = JointSessionStream(stream_path, sigma=ckpt.get("sigma", 1.0))
    if cache.exists() and not refresh:
        z = np.load(cache)
        return {"onset_prob": z["onset_prob"], "class_prob": z["class_prob"],
                "crop": stream.frames, "fps": stream.fps, "ckpt_tag": tag,
                "model_type": ckpt.get("model_type", "joint")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_joint_from_ckpt(ckpt, device)
    model.eval()
    t0 = time.time()
    onset_prob, class_prob, _ = score_stream_joint(model, stream, device)
    print(f"  scored {len(stream)} frames in {time.time()-t0:.1f}s "
          f"({device.type}) -> {cache.name}")
    np.savez_compressed(cache, onset_prob=onset_prob, class_prob=class_prob)
    return {"onset_prob": onset_prob, "class_prob": class_prob,
            "crop": stream.frames, "fps": stream.fps, "ckpt_tag": tag,
            "model_type": ckpt.get("model_type", "joint")}


def decode_proposals(sess: Session, args) -> list[dict]:
    """
    The model's own move list — the CTC prefix beam for a CTC checkpoint,
    peak-picking otherwise. Exactly eval_lighting.decode_arm's dispatch, so
    the proposals a reviewer sees are the ones the deployed decoder makes.
    """
    from ctc_decode import prefix_beam_decode, ctc_to_moves
    from joint_decode import posteriorgram_to_moves

    if sess.model_type == "ctc":
        labels, frames = prefix_beam_decode(
            np.log(np.maximum(sess.class_prob, 1e-12)), beam=args.beam_ctc)
        return ctc_to_moves(sess.class_prob, labels, frames, fps=sess.fps)
    return posteriorgram_to_moves(sess.onset_prob, sess.class_prob,
                                  threshold=args.threshold,
                                  min_sep=MIN_SEP, fps=sess.fps)


def build_candidates(sess: Session, proposals: list[dict], args
                     ) -> list[Candidate]:
    """
    Decoder proposals (default ACCEPT) plus sub-threshold peaks the decoder
    did not emit (default REJECT) — mechanism 1 in the module docstring.
    """
    cands = [Candidate(frame=int(m["frame"]),
                       probs=np.asarray(m["probs"], dtype=np.float64),
                       score=float(m["score"]), proposed=m["move"],
                       move=m["move"], origin="decoder", accepted=True)
             for m in proposals]

    claimed = {c.frame for c in cands}
    for f in range(-CLAIM_RADIUS, CLAIM_RADIUS + 1):
        claimed |= {c.frame + f for c in cands}

    if args.candidate_threshold is not None:
        weak = peak_pick(sess.fg, threshold=args.candidate_threshold,
                         min_sep=MIN_SEP)
        for o in weak:
            o = int(o)
            if o in claimed or o >= sess.n_frames:
                continue
            row = sess.class_prob[o, :12].astype(np.float64)
            row = row / row.sum() if row.sum() > 1e-9 else np.full(12, 1 / 12)
            cands.append(Candidate(frame=o, probs=row, score=float(sess.fg[o]),
                                   proposed=None, move=WCA12[int(row.argmax())],
                                   origin="weak", accepted=False))

    cands.sort(key=lambda c: c.frame)

    if args.blind_rate > 0:
        # Seeded by session name so re-opening the same session blinds the
        # same moves — otherwise a reviewer could reroll until nothing is
        # blind, and the audit would measure only their patience.
        rng = random.Random(sess.path.name)
        for c in cands:
            if c.origin == "decoder" and rng.random() < args.blind_rate:
                c.blind = True
    return cands


# ---------------------------------------------------------------------------
# Start state
# ---------------------------------------------------------------------------

def resolve_start_state(session: Path, args) -> np.ndarray | None:
    """
    The cube state the recording began in, in the CAMERA frame that
    reconstruct.py's classes live in.

    Three sources, and the difference between them matters:

    --start-moves   the scramble word, applied to a solved cube. This is
                    the deployment path (display a scramble, they apply
                    it), and it is camera-relative by construction.
    --start-state   a 54-char URFDLB facelet string in the CAMERA frame,
                    i.e. what cv/solver/state_finder.py scans. NOT the
                    smart cube's own snapshot_state() string, which is in
                    the cube's colour-fixed frame — see ble_truth.
                    snapshot_state's docstring.
    --start-from-truth
                    BLE ground truth, for development on recorded
                    sessions. Reads only the NET transformation (the
                    inverse of the whole move word's product), which is
                    the same information a scan gives; the sequence itself
                    is never surfaced to the reviewer.
    """
    if args.start_moves:
        word = args.start_moves.split()
        bad = [m for m in word if m not in WCA12]
        if bad:
            sys.exit(f"--start-moves: not quarter turns: {' '.join(bad)}")
        return RC.seq_to_state(word)

    if args.start_state:
        s = args.start_state.strip()
        if len(s) != 54:
            sys.exit(f"--start-state must be 54 facelets, got {len(s)}")
        from twophase.cubes.facecube import FaceCube
        return RC._cubie_to_vec(FaceCube(s).to_cubiecube())

    if args.start_from_truth:
        moves_path = session / "moves.jsonl"
        if not moves_path.exists():
            sys.exit(f"--start-from-truth: {moves_path} does not exist")
        gt = [json.loads(l).get("wca_notation")
              for l in open(moves_path) if l.strip()]
        if not gt or any(g not in WCA12 for g in gt):
            sys.exit("--start-from-truth: ground truth has non-quarter-turns")
        return RC.start_from_gt(gt)

    return None


# ---------------------------------------------------------------------------
# The review state
# ---------------------------------------------------------------------------

class Review:
    """The mutable label set, plus the decoder's flags on it."""

    def __init__(self, sess: Session, cands: list[Candidate],
                 start: np.ndarray | None, args):
        self.sess = sess
        self.cands = cands
        self.start = start
        self.args = args
        self.flags: list[Flag] = []
        self.decode_note = "not run"
        self.dirty = False
        self.tables = None
        self.applied_suggestions = 0
        # Frame up to which the reviewer asserts the labels are COMPLETE and
        # correct. See the module docstring: this is what pins the cube state
        # mid-solve and makes the flag decode tractable at all.
        self.head = 0
        # Bumped by every edit, so a decode that finishes against a label set
        # the reviewer has since changed is discarded instead of shown.
        self.version = 0
        self.decoding_since: float | None = None
        self._thread = None
        self._result = None

    # -- label set ---------------------------------------------------------

    @property
    def labels(self) -> list[Candidate]:
        return [c for c in self.cands if c.accepted]

    @property
    def word(self) -> list[str]:
        return [c.move for c in self.labels]

    @property
    def head_idx(self) -> int:
        """How many labels lie inside the reviewed prefix."""
        return sum(1 for c in self.labels if c.frame < self.head)

    def touch(self) -> None:
        self.dirty = True
        self.version += 1

    def state_after(self) -> np.ndarray | None:
        if self.start is None:
            return None
        return RC.compose_seq(self.start, self.word)

    def state_at_head(self) -> np.ndarray | None:
        """The cube state the reviewed prefix implies at the head."""
        if self.start is None:
            return None
        return RC.compose_seq(self.start, self.word[:self.head_idx])

    def distance_note(self) -> str:
        """
        Always-live completion signal: does the current label set take the
        cube from the start state to solved? Cheap enough to recompute every
        redraw (a 40-entry compose per move plus three table lookups), which
        is what lets it sit in the status bar rather than behind a keypress.
        """
        if self.start is None:
            return "no start state"
        s = self.state_after()
        if (s == RC.SOLVED).all():
            return "SOLVED"
        if self.tables is None:
            self.tables = RC.build_tables()
        return f"{RC.h_full(s, self.tables)}+ quarter turns from solved"

    # -- the group-theoretic decode ---------------------------------------

    def cost_model(self, labels: list[Candidate]
                   ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Per-onset acceptance and deletion costs over the given label set.

        Confirmed moves are locked (see LOCK_COST); unreviewed ones keep the
        model's own posterior, so the decoder spends its freedom exactly
        where the human has not yet spent theirs.
        """
        lock = self.args.lock_cost
        rows, dels = [], []
        for c in labels:
            if c.confirmed or c.origin == "manual":
                row = np.full(12, lock, dtype=np.float32)
                row[WCA12.index(c.move)] = 0.0
                rows.append(row)
                dels.append(lock)
            else:
                rows.append(RC.onset_costs(c.probs, self.args.blend_inv,
                                           self.args.blend_unif))
                dels.append(float(RC.score_del_costs(
                    [c.score], self.args.threshold, self.args.del_cost,
                    self.args.candidate_threshold, self.args.del_floor)[0]))
        return rows, np.asarray(dels, dtype=np.float32)

    def _decode_now(self) -> tuple[list[Flag], str]:
        """
        Decode the UNREVIEWED SUFFIX from the state the head pins, and turn
        the winning story's edits into flags — mechanism 3 in the module
        docstring. Returns (flags, note); mutates nothing, so it is safe to
        run on a worker thread.
        """
        if self.start is None:
            return [], "no start state — flags unavailable"
        labels = self.labels[self.head_idx:]
        if not labels:
            s = self.state_after()
            return [], ("reviewed to the end and SOLVED"
                        if (s == RC.SOLVED).all() else
                        "reviewed to the end but NOT solved — an error is "
                        "inside the reviewed span")
        if self.tables is None:
            self.tables = RC.build_tables()

        rows, dels = self.cost_model(labels)
        t0 = time.time()
        res = RC.decode(self.state_at_head(), rows, beam=self.args.flag_beam,
                        c_del=self.args.del_cost, c_ins=self.args.ins_cost,
                        max_end_ins=self.args.max_end_ins,
                        del_costs=dels, rotations=False,
                        max_chain=self.args.flag_max_chain,
                        tables=self.tables, slices=self.args.slices,
                        slice_rows=(RC.slice_rows_from_moves(
                            [{"probs": c.probs} for c in labels],
                            self.args.slice_gate)
                            if self.args.slices else None))
        dt = time.time() - t0

        if not res["solved"]:
            # An honest failure, not a silent one, and early in a review it
            # is the EXPECTED state — see the head table in the module
            # docstring. Inventing placements by inflating the budget until
            # something "verifies" is exactly what reconstruct.py warns
            # against, so the answer here is to advance the head, not to
            # loosen the decode.
            return [], (f"no consistent story over the {len(labels)} "
                        f"unreviewed onsets at beam {self.args.flag_beam} "
                        f"(closest end state {res.get('min_h_final','?')} "
                        f"turns off, {dt:.1f}s) — review further and press m "
                        f"to advance the head")

        frames = [c.frame for c in labels]
        names = [c.move for c in labels]
        flags = flags_from_ops(res["ops"], frames, names, self.sess.n_frames,
                               lo_bound=self.head)
        n_ins = sum(f.kind == "insert" for f in flags)
        return flags, (f"story found over {len(labels)} unreviewed onsets, "
                       f"cost {res['cost']:.1f} ({len(flags)} flags, "
                       f"{n_ins} missing, {dt:.1f}s)")

    def refresh_flags(self, verbose: bool = True) -> None:
        """Synchronous decode — used by --report and at startup."""
        self.flags, self.decode_note = self._decode_now()
        if verbose:
            print(f"  {self.decode_note}")

    def start_decode(self) -> None:
        """
        Run the decode on a worker thread. It takes 4-40s depending on how
        much suffix is left, which is far too long to block a scrubbing UI;
        the beam is numpy-heavy and spends most of that time with the GIL
        released.
        """
        import threading
        if self._thread is not None and self._thread.is_alive():
            return
        ver = self.version
        self.decoding_since = time.time()

        def work():
            try:
                out = self._decode_now()
            except Exception as exc:                      # noqa: BLE001
                out = ([], f"decode error: {exc}")
            self._result = (ver, out)
            self.decoding_since = None

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def poll_decode(self) -> bool:
        """Adopt a finished decode. Stale results are dropped, not shown."""
        if self._result is None:
            return False
        ver, (flags, note) = self._result
        self._result = None
        if ver != self.version:
            self.decode_note = "labels changed while decoding — press g"
            return True
        self.flags, self.decode_note = flags, note
        return True

    # -- edits -------------------------------------------------------------

    def insert_at(self, frame: int, move: str) -> Candidate:
        row = self.sess.class_prob[frame, :12].astype(np.float64)
        row = row / row.sum() if row.sum() > 1e-9 else np.full(12, 1 / 12)
        c = Candidate(frame=int(frame), probs=row,
                      score=float(self.sess.fg[frame]), proposed=None,
                      move=move, origin="manual", accepted=True,
                      confirmed=True)
        self.cands.append(c)
        self.cands.sort(key=lambda x: x.frame)
        self.touch()
        return c

    def apply_flag(self, flag: Flag, frame: int | None = None) -> str:
        """Take the decoder's suggestion for one flag."""
        self.applied_suggestions += 1
        self.touch()
        if flag.kind == "insert":
            self.insert_at(frame if frame is not None else flag.at, flag.move)
            return f"inserted {flag.move}"
        target = next((c for c in self.labels if c.frame == flag.lo), None)
        if target is None:
            return "flag no longer applies"
        if flag.kind == "phantom":
            target.accepted, target.confirmed = False, True
            return f"rejected {target.move}"
        if flag.kind in ("rename", "disputed"):
            old, target.move, target.confirmed = target.move, flag.move, True
            return f"{old} -> {flag.move}"
        return "nothing to apply"

    # -- audit -------------------------------------------------------------

    def audit(self) -> dict:
        blind = [c for c in self.cands if c.blind and c.blind_answer]
        blind_ok = sum(c.blind_answer == c.proposed for c in blind)
        sighted = [c for c in self.cands
                   if c.confirmed and not c.blind and c.origin == "decoder"]
        sighted_ok = sum(c.move == c.proposed for c in sighted)
        # Labels the head asserts were reviewed but which nobody ever looked
        # at individually — the measure of how much of this review was the
        # head being dragged forward. See the module docstring.
        in_head = [c for c in self.labels if c.frame < self.head]
        return {
            "n_blind_answered": len(blind),
            "blind_agreement": (blind_ok / len(blind)) if blind else None,
            "n_sighted_confirmed": len(sighted),
            "sighted_agreement": (sighted_ok / len(sighted))
                                 if sighted else None,
            "n_suggestions_applied": self.applied_suggestions,
            "head_frame": self.head,
            "n_in_head": len(in_head),
            "n_in_head_unconfirmed": sum(not c.confirmed for c in in_head),
        }


def flags_from_ops(ops: list[tuple], frames: list[int], names: list[str],
                   n_frames: int, lo_bound: int = 0) -> list[Flag]:
    """
    reconstruct.decode's chronological op list -> reviewable flags.

    The op list carries no onset indices, so they are recovered by walking
    it: accept / delete / slice each consume exactly one onset, insert and
    end-insert consume none. That is the only assumption here, it matches
    reconstruct._ops_to_moves exactly, and the assert below fails loudly
    rather than mis-attributing a flag if it ever stops holding. (OP_ALGO
    would break the walk, which is why algo_cands is not passed: an
    algorithm span consumes many onsets for one op.)
    """
    flags: list[Flag] = []
    j = 0
    for op, name in ops:
        if op == "accept":
            if name != names[j]:
                flags.append(Flag("rename", frames[j], frames[j], name, j,
                                  f"decoder prefers {name} over {names[j]}"))
            j += 1
        elif op == "slice":
            flags.append(Flag("slice", frames[j], frames[j], name, j,
                              "one motion, two layer turns"))
            j += 1
        elif op == "delete":
            flags.append(Flag("phantom", frames[j], frames[j], names[j], j,
                              "decoder says nothing turned here"))
            j += 1
        elif op == "insert":
            lo = frames[j - 1] if j > 0 else lo_bound
            hi = frames[j] if j < len(frames) else n_frames - 1
            flags.append(Flag("insert", lo, hi, name, j,
                              f"a move is missing here; it must be {name}"))
        elif op == "end-insert":
            lo = frames[-1] if frames else lo_bound
            flags.append(Flag("insert", lo, n_frames - 1, name, len(frames),
                              f"missing after the last onset: {name}"))
        elif op == "rotate":
            flags.append(Flag("disputed", frames[min(j, len(frames) - 1)],
                              frames[min(j, len(frames) - 1)], None, j,
                              f"whole-cube rotation {name}"))
    assert j == len(frames), (f"op walk consumed {j} onsets, expected "
                              f"{len(frames)} — reconstruct's op vocabulary "
                              f"changed")

    # A locked (confirmed) move the decoder still wanted to change is a
    # disagreement between a human and the group theory, not a suggestion.
    return flags


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

PAD = 8
W = 1500
PANE_H = 280
GLOBAL_H = 100
ZOOM_H = 210
STATUS_H = 250
H = PAD * 5 + PANE_H + GLOBAL_H + ZOOM_H + STATUS_H

# Move-list panel, occupying the right of the status band.
LIST_X = 950
LIST_ROWS = 9
LIST_ROW_H = 24

BG = (24, 24, 28)
FG = (225, 225, 230)
DIM = (130, 130, 140)
GRID = (55, 55, 62)

FLAG_BGR = {"insert": (60, 60, 235), "phantom": (0, 140, 255),
            "rename": (0, 200, 255), "slice": (200, 120, 60),
            "disputed": (200, 60, 220)}

KEY_HINTS = [
    "j/k move   ,/. frame   </> 15 frames   [/] flag   space play   -/= speed"
    "   z/Z zoom",
    "u d l r f b set face   ' prime   i insert here   y confirm   x reject"
    "   enter apply flag",
    "m reviewed-up-to-here   M reset head   g re-decode   s save   ? help"
    "   q quit",
]


class Renderer:
    """Draws the whole UI into one fixed-size canvas."""

    def __init__(self, rv: Review):
        import cv2
        self.cv2 = cv2
        self.rv = rv
        self.sess = rv.sess
        self.zoom_half = 90          # frames either side of the playhead
        self.show_help = False
        self.rects = {}              # name -> (x, y, w, h), for mouse hits

    # -- primitives --------------------------------------------------------

    def text(self, img, s, x, y, color=FG, scale=0.45, thick=1):
        self.cv2.putText(img, s, (x, y), self.cv2.FONT_HERSHEY_SIMPLEX,
                         scale, color, thick, self.cv2.LINE_AA)

    def fit(self, img, w, h):
        """Letterbox `img` into a w x h panel."""
        cv2 = self.cv2
        out = np.full((h, w, 3), (14, 14, 16), dtype=np.uint8)
        if img is None:
            self.text(out, "no image", 8, h // 2, DIM)
            return out
        s = min(w / img.shape[1], h / img.shape[0])
        r = cv2.resize(img, (max(1, int(img.shape[1] * s)),
                             max(1, int(img.shape[0] * s))),
                       interpolation=cv2.INTER_AREA if s < 1
                       else cv2.INTER_NEAREST)
        y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
        out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        return out

    # -- the strips --------------------------------------------------------

    def strip(self, w, h, lo, hi, playhead, sel_frame, heat: bool,
              blind_span: set) -> np.ndarray:
        """
        One timeline panel over frames [lo, hi).

        `heat` draws the 12-class posteriorgram as a heat map — the thing
        that makes a MISS legible, because a gap in it is visible where an
        absent detection marker is not. Rows are class-coloured so a
        substitution reads as the wrong colour rather than as unlabelled
        intensity.
        """
        cv2 = self.cv2
        rv, sess = self.rv, self.sess
        img = np.full((h, w, 3), (18, 18, 22), dtype=np.uint8)
        span = max(hi - lo, 1)

        def fx(f):
            return int((f - lo) / span * (w - 1))

        heat_h = 12 * 9 if heat else 0
        curve_h = h - heat_h - 26
        top = 0

        if heat:
            # (12, span) posterior block, nearest-resampled to the panel.
            #
            # Displayed through a gamma stretch, not linearly. A CTC
            # posterior is nearly one-hot — confidently blank on ~90% of
            # frames and saturated for 1-3 frames at an onset — so a linear
            # map renders as isolated dots on black and shows a reviewer
            # nothing they did not already get from the detection markers.
            # The evidence that marks a MISS is exactly the small mass the
            # linear map crushes: a hump at p ~ 0.05-0.2 where the model
            # nearly fired. gamma 0.4 lifts that into view while leaving a
            # confident call obviously brighter, which is the whole job of
            # mechanism 2 in the module docstring.
            idx = np.clip(np.linspace(lo, hi - 1, w).astype(int), 0,
                          sess.n_frames - 1)
            block = sess.class_prob[idx, :12].T            # (12, w)
            for k in range(12):
                y0 = top + k * 9
                col = np.asarray(class_bgr(WCA12[k]), dtype=np.float32)
                inten = np.clip(block[k], 0, 1)[:, None] ** 0.4
                bar = (col[None, :] * inten + 18 * (1 - inten)).astype(np.uint8)
                img[y0:y0 + 8, :] = bar[None, :, :]
                # Dark backing behind the row label: at 8px pitch the text
                # is otherwise drawn straight onto whatever the posterior
                # happens to be doing at frame `lo`, and reads as a smudge.
                img[y0:y0 + 8, 0:20] = (12, 12, 14)
                self.text(img, WCA12[k], 2, y0 + 7, (170, 170, 180), 0.32)
            # Blind candidates: blank the heat map so the model's answer is
            # genuinely hidden, not merely unlabelled.
            for f in blind_span:
                if not lo <= f < hi:
                    continue
                x = fx(f)
                pad = max(3, int(w / span * 3))
                x0, x1 = max(0, x - pad), min(w - 1, x + pad)
                img[top:top + 12 * 9, x0:x1 + 1] = (40, 30, 30)
                self.text(img, "BLIND", max(0, x - 18), top + 55,
                          (120, 120, 220), 0.4)
            top += heat_h + 2

        # foreground curve
        base = top + curve_h
        cv2.line(img, (0, base), (w, base), GRID, 1)
        thr_y = int(base - rv.args.threshold * curve_h)
        cv2.line(img, (0, thr_y), (w, thr_y), (60, 70, 60), 1)
        xs = np.clip(np.linspace(lo, hi - 1, w).astype(int), 0,
                     sess.n_frames - 1)
        # Both heads in the zoom strip, because they disagree exactly where
        # it matters: the class head can be confidently blank at a frame
        # where the ONSET head still bumps — "something turned here but I
        # cannot name it" — and that disagreement is a miss signature the
        # class heat map alone does not show. Kept dim and out of the global
        # strip, where on a CTC checkpoint it is noisy enough to bury the
        # curve that actually drives the decode.
        if heat:
            oy = (base - sess.onset_prob[xs] * curve_h).astype(int)
            cv2.polylines(img, [np.stack([np.arange(w), oy], axis=1)], False,
                          (70, 62, 100), 1, cv2.LINE_AA)
        ys = (base - sess.fg[xs] * curve_h).astype(int)
        pts = np.stack([np.arange(w), ys], axis=1)
        cv2.polylines(img, [pts], False, (150, 200, 150), 1, cv2.LINE_AA)

        # the reviewed span: everything left of the head is asserted complete
        if rv.head > lo:
            hx = min(w - 1, fx(rv.head))
            band = img[top:base, 0:max(1, hx)]
            if band.size:
                band[:] = (band * 0.55 + np.asarray((70, 55, 30)) * 0.45
                           ).astype(np.uint8)
            cv2.line(img, (hx, 0), (hx, h - 12), (255, 190, 90), 2)
            self.text(img, "reviewed", max(2, hx - 62), top + 10,
                      (255, 190, 90), 0.4)

        # flag bands, under the markers
        for fl in rv.flags:
            x0, x1 = fx(fl.lo), fx(fl.hi)
            if x1 < 0 or x0 > w:
                continue
            col = FLAG_BGR[fl.kind]
            if fl.kind == "insert":
                band = img[top:base, max(0, x0):max(1, x1 + 1)]
                if band.size:
                    band[:] = (band * 0.62 +
                               np.asarray(col) * 0.38).astype(np.uint8)
                cv2.line(img, (max(0, x0), top), (max(0, x0), base), col, 1)
                cv2.line(img, (min(w - 1, x1), top), (min(w - 1, x1), base),
                         col, 1)
            else:
                cv2.line(img, (x0, top), (x0, base), col, 1)

        # candidate markers. Full-height in the zoom strip; short ticks in
        # the global one, where 100+ markers would otherwise be a picket
        # fence with the score curve hidden behind it.
        mark_top = top if heat else base - 16
        for c in rv.cands:
            if not (lo <= c.frame < hi):
                continue
            x = fx(c.frame)
            y = base
            if c.accepted:
                col = class_bgr(c.move)
                cv2.line(img, (x, mark_top), (x, y), col,
                         2 if c.confirmed else 1)
                cv2.circle(img, (x, y), 4, col, -1 if c.confirmed else 1)
            else:
                cv2.circle(img, (x, y), 3, (90, 90, 100), 1)
            if c.frame == sel_frame:
                cv2.rectangle(img, (x - 6, mark_top - 1), (x + 6, y + 7),
                              (255, 255, 255), 1)
            if heat:
                lbl = "??" if (c.blind and not c.blind_answer) else c.move
                self.text(img, lbl, x - 7, y + 14,
                          class_bgr(c.move) if c.accepted else DIM, 0.42)

        # playhead + ruler
        px = fx(playhead)
        cv2.line(img, (px, 0), (px, h - 11), (255, 255, 255), 1)
        for f in np.linspace(lo, hi - 1, 8).astype(int):
            self.text(img, f"{f/self.sess.fps:5.1f}s", max(0, fx(f) - 16),
                      h - 2, DIM, 0.34)
        return img

    # -- the whole canvas --------------------------------------------------

    def render(self, playhead: int, sel: int, message: str) -> np.ndarray:
        cv2 = self.cv2
        rv, sess = self.rv, self.sess
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        cands = rv.cands
        cur = cands[sel] if 0 <= sel < len(cands) else None
        blind_span = {c.frame for c in cands if c.blind and not c.blind_answer}

        # -- row A: imagery + posterior bars
        y = PAD
        pw = 520
        canvas[y:y + PANE_H, PAD:PAD + pw] = self.fit(sess.image(playhead),
                                                      pw, PANE_H)
        x = PAD + pw + PAD
        canvas[y:y + PANE_H, x:x + PANE_H] = self.fit(
            sess.model_view(playhead), PANE_H, PANE_H)
        self.text(canvas, "model input", x + 6, y + 16, DIM, 0.4)
        x += PANE_H + PAD
        canvas[y:y + PANE_H, x:x + PANE_H] = self.fit(
            sess.diff_view(playhead), PANE_H, PANE_H)
        self.text(canvas, "frame diff x4", x + 6, y + 16, DIM, 0.4)

        # per-class posterior at the playhead
        x += PANE_H + PAD
        bw = W - PAD - x
        blinded = playhead in blind_span or (
            cur is not None and cur.blind and not cur.blind_answer
            and abs(cur.frame - playhead) <= 3)
        self.text(canvas, "posterior", x, y + 12, DIM, 0.4)
        for k in range(12):
            yy = y + 26 + k * 20
            p = 0.0 if blinded else float(sess.class_prob[playhead, k])
            cv2.rectangle(canvas, (x + 22, yy), (x + 22 + int(p * (bw - 30)),
                                                 yy + 13), class_bgr(WCA12[k]),
                          -1)
            self.text(canvas, WCA12[k], x, yy + 11, DIM, 0.4)
        if blinded:
            self.text(canvas, "BLIND", x + 30, y + 160, (120, 120, 220), 0.7, 2)
        y += PANE_H + PAD

        # -- row B: global strip
        self.rects["global"] = (PAD, y, W - 2 * PAD, GLOBAL_H)
        canvas[y:y + GLOBAL_H, PAD:W - PAD] = self.strip(
            W - 2 * PAD, GLOBAL_H, 0, sess.n_frames, playhead,
            cur.frame if cur else -1, heat=False, blind_span=blind_span)
        # viewport box for the zoom strip
        lo, hi = self.zoom_range(playhead)
        gx0 = PAD + int(lo / sess.n_frames * (W - 2 * PAD))
        gx1 = PAD + int(hi / sess.n_frames * (W - 2 * PAD))
        cv2.rectangle(canvas, (gx0, y), (gx1, y + GLOBAL_H), (120, 120, 130), 1)
        y += GLOBAL_H + PAD

        # -- row C: zoom strip
        self.rects["zoom"] = (PAD, y, W - 2 * PAD, ZOOM_H)
        canvas[y:y + ZOOM_H, PAD:W - PAD] = self.strip(
            W - 2 * PAD, ZOOM_H, lo, hi, playhead,
            cur.frame if cur else -1, heat=True, blind_span=blind_span)
        y += ZOOM_H + PAD

        # -- row D: status + the move list
        self.status(canvas, y, playhead, cur, message)
        self.move_list(canvas, y, sel)
        if self.show_help:
            self.help_overlay(canvas)
        return canvas

    def zoom_range(self, playhead: int) -> tuple[int, int]:
        n = self.sess.n_frames
        lo = int(np.clip(playhead - self.zoom_half, 0, max(0, n - 1)))
        hi = int(np.clip(lo + 2 * self.zoom_half, 1, n))
        lo = max(0, hi - 2 * self.zoom_half)
        return lo, hi

    def move_list(self, canvas, y, sel: int):
        """
        The sequence as a scrollable LIST, next to the timeline rather than
        instead of it.

        The timeline answers "is anything missing here", which is what it is
        for. It cannot answer "which move am I on and what did I do to the
        last few" — at one frame per pixel a hundred markers are a picket
        fence. Reviewing by exception means walking an ordered list and
        touching the odd entry, so the list is the primary control (j/k) and
        the timeline is the evidence next to it.
        """
        cv2 = self.cv2
        rv = self.rv
        x, w = LIST_X, W - PAD - LIST_X
        cv2.rectangle(canvas, (x, y), (x + w, y + STATUS_H - 4), (16, 16, 20), -1)

        # Move numbers count only ACCEPTED candidates — a rejected phantom
        # is not move 43, it is not a move.
        numbers, n = {}, 0
        for i, c in enumerate(rv.cands):
            if c.accepted:
                n += 1
                numbers[i] = n

        self.text(canvas, f"MOVES   {n} in the label set"
                          f"   (j/k to walk)", x + 8, y + 16, DIM, 0.42)
        half = LIST_ROWS // 2
        first = int(np.clip(sel - half, 0, max(0, len(rv.cands) - LIST_ROWS)))
        flagged = {f.lo for f in rv.flags if f.kind != "insert"}

        for row in range(LIST_ROWS):
            i = first + row
            if i >= len(rv.cands):
                break
            c = rv.cands[i]
            yy = y + 34 + row * LIST_ROW_H
            if i == sel:
                cv2.rectangle(canvas, (x + 4, yy - 13), (x + w - 4, yy + 6),
                              (48, 48, 58), -1)
                self.text(canvas, ">", x + 8, yy + 2, (255, 255, 255), 0.45)

            hidden = c.blind and not c.blind_answer
            num = f"{numbers[i]:>3}" if i in numbers else "  ."
            self.text(canvas, num, x + 20, yy + 2, DIM, 0.42)
            self.text(canvas, f"{c.frame/self.sess.fps:6.2f}s", x + 52,
                      yy + 2, DIM, 0.42)
            self.text(canvas, "??" if hidden else c.move, x + 122, yy + 2,
                      class_bgr(c.move) if c.accepted else (90, 90, 100),
                      0.55, 2)

            if not c.accepted:
                note, col = ("rejected" if c.origin != "weak"
                             else "sub-threshold"), (90, 90, 100)
            elif c.origin == "manual":
                note, col = "inserted by you", (140, 240, 140)
            elif c.renamed:
                note, col = f"renamed from {c.proposed}", (0, 200, 255)
            elif c.confirmed:
                note, col = "confirmed", (150, 200, 150)
            else:
                note, col = f"conf {c.probs.max():.2f}", DIM
            if c.frame in flagged:
                note, col = "FLAGGED", FLAG_BGR["rename"]
            self.text(canvas, note, x + 170, yy + 2, col, 0.42)

    def status(self, canvas, y, playhead, cur, message):
        rv = self.rv
        n_lab = len(rv.labels)
        n_conf = sum(c.confirmed and c.accepted for c in rv.cands)
        n_rej = sum(not c.accepted and c.confirmed for c in rv.cands)
        n_ren = sum(c.renamed for c in rv.cands if c.accepted)
        n_ins = sum(c.origin == "manual" for c in rv.cands if c.accepted)
        dist = rv.distance_note()

        self.text(canvas, f"{rv.sess.path.name}   {rv.sess.ckpt_tag}"
                          f" [{rv.sess.model_type}]", PAD, y + 14, DIM, 0.42)
        self.text(canvas, f"frame {playhead}/{rv.sess.n_frames-1}"
                          f"  {playhead/rv.sess.fps:6.2f}s", LIST_X - 190,
                  y + 14, DIM, 0.42)

        col = (140, 240, 140) if dist == "SOLVED" else (140, 190, 245)
        self.text(canvas, f"labels {n_lab}   confirmed {n_conf}   "
                          f"renamed {n_ren}   inserted {n_ins}   "
                          f"rejected {n_rej}", PAD, y + 38, FG, 0.5)
        self.text(canvas, f"cube: {dist}", PAD, y + 62, col, 0.55, 2)
        self.text(canvas, f"reviewed to frame {rv.head} "
                          f"({rv.head_idx}/{n_lab}, {n_lab - rv.head_idx} "
                          f"left)   [m]", LIST_X - 300, y + 62,
                  (255, 190, 90), 0.5)
        note = rv.decode_note
        if rv.decoding_since is not None:
            note = f"decoding... ({time.time() - rv.decoding_since:.0f}s)"
        self.text(canvas, f"decode: {note}", PAD, y + 86, DIM, 0.42)

        if cur is not None:
            hidden = cur.blind and not cur.blind_answer
            shown = "??" if hidden else cur.move
            state = ("accepted" if cur.accepted else "rejected") + \
                    (", confirmed" if cur.confirmed else "")
            self.text(canvas, f"selected  frame {cur.frame}   {shown}   "
                              f"conf {0.0 if hidden else cur.probs.max():.2f}"
                              f"   score {cur.score:.2f}   [{cur.origin}, "
                              f"{state}]", PAD, y + 112,
                      (250, 250, 250) if cur.accepted else DIM, 0.5)

        # nearest flag
        fl = self.nearest_flag(playhead)
        if fl is not None:
            self.text(canvas, f"FLAG [{fl.kind}] {fl.note}"
                              f"   frames {fl.lo}-{fl.hi}   [enter] to apply",
                      PAD, y + 136, FLAG_BGR[fl.kind], 0.5, 1)
        self.text(canvas, message, PAD, y + 160, (160, 220, 160), 0.45)

        # Permanent key hints. The help overlay still exists, but a tool
        # whose controls are only discoverable behind a keypress is a tool
        # you re-learn every time you open it.
        for i, line in enumerate(KEY_HINTS):
            self.text(canvas, line, PAD, y + 190 + i * 18, (120, 120, 130),
                      0.40)

    def nearest_flag(self, playhead: int) -> Flag | None:
        inside = [f for f in self.rv.flags if f.lo <= playhead <= f.hi]
        if inside:
            return min(inside, key=lambda f: abs(f.at - playhead))
        return None

    def help_overlay(self, canvas):
        cv2 = self.cv2
        lines = [
            "j / k   or up/down    previous / next MOVE  (walks the list)",
            ", / .   or left/right step one frame",
            "< / >   or pgup/pgdn  step 15 frames",
            "[ / ]                 previous / next FLAG",
            "space   play / pause          - / =   playback speed",
            "click or drag either timeline to scrub",
            "",
            "u d l r f b   set the face — or INSERT a move at the playhead",
            "'             toggle prime          i   insert at the playhead",
            "y             confirm this move     x   reject it (phantom)",
            "enter         apply the flagged suggestion here",
            "",
            "m   REVIEWED UP TO HERE — asserts the labels before this point",
            "    are complete and correct, which pins the cube state and is",
            "    what makes the group-theory flags findable at all.",
            "M   pull the review head back to the start",
            "g   re-run the decode (refresh flags)   z/Z  zoom out / in",
            "s   save        q  quit        ?  close this help",
        ]
        h = 24 * len(lines) + 30
        y0 = (H - h) // 2
        x0, x1 = 240, W - 240
        overlay = canvas[y0:y0 + h, x0:x1].copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], h), (12, 12, 14), -1)
        canvas[y0:y0 + h, x0:x1] = (
            canvas[y0:y0 + h, x0:x1] * 0.12 + overlay * 0.88).astype(np.uint8)
        for i, s in enumerate(lines):
            self.text(canvas, s, x0 + 22, y0 + 28 + i * 24, FG, 0.48)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def label_path(session: Path) -> Path:
    return session / "review_labels.json"


def save(rv: Review, reviewer: str) -> Path:
    """
    Write the corrected label set.

    Three ways to say WHERE each move is, on purpose:

      frame_file  the frame's filename, and the authoritative one.
                  prepare_data.py resolves labels by this, so nothing
                  breaks if frames are added or deleted between the review
                  and the preparation — an index would silently shift.
      frame       index into this review's frame list, for humans and for
                  re-opening the session here.
      timestamp   wall clock, so the file also converts into the
                  moves.jsonl shape the BLE path consumes.
    """
    sess = rv.sess
    moves = [{
        "move_num": i + 1,
        "frame": c.frame,
        "frame_file": sess.frames_meta[c.frame]["file"],
        "timestamp": float(sess.frame_ts[c.frame]),
        "wca_notation": c.move,
        "source": ("inserted" if c.origin == "manual" else
                   "renamed" if c.renamed else
                   "promoted" if c.origin == "weak" else "accepted"),
        "proposed": c.proposed,
        "confirmed": c.confirmed,
        "conf": float(c.probs.max()),
    } for i, c in enumerate(rv.labels)]

    out = {
        "session": sess.path.name,
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reviewer": reviewer,
        "model": sess.ckpt_tag,
        "decoder": sess.model_type,
        "n_frames": sess.n_frames,
        "fps": sess.fps,
        "review_head_frame": rv.head,
        "reaches_solved": rv.distance_note() == "SOLVED",
        "completion": rv.distance_note(),
        "decode_note": rv.decode_note,
        "sequence": rv.word,
        "moves": moves,
        "rejected": [{"frame": c.frame, "proposed": c.proposed,
                      "origin": c.origin}
                     for c in rv.cands if not c.accepted and c.confirmed],
        "audit": rv.audit(),
    }
    p = label_path(sess.path)
    p.write_text(json.dumps(out, indent=1))
    rv.dirty = False
    return p


def restore(rv: Review) -> bool:
    """Re-apply a saved review onto freshly built candidates."""
    p = label_path(rv.sess.path)
    if not p.exists():
        return False
    data = json.loads(p.read_text())
    by_frame = {c.frame: c for c in rv.cands}
    for m in data.get("moves", []):
        c = by_frame.get(m["frame"])
        if c is None:
            rv.insert_at(m["frame"], m["wca_notation"])
            continue
        c.accepted, c.move = True, m["wca_notation"]
        c.confirmed = bool(m.get("confirmed", False))
    for r in data.get("rejected", []):
        c = by_frame.get(r["frame"])
        if c is not None:
            c.accepted, c.confirmed = False, True
    rv.head = int(data.get("review_head_frame", 0))
    rv.dirty = False
    print(f"  resumed {p.name}: {len(data.get('moves', []))} labels, "
          f"{data.get('completion')}")
    return True


# ---------------------------------------------------------------------------
# Headless reporting
# ---------------------------------------------------------------------------

def report(rv: Review) -> None:
    sess = rv.sess
    print(f"\n  {sess.path.name}   {sess.n_frames} frames @ {sess.fps:.1f}fps"
          f"   model {sess.ckpt_tag} [{sess.model_type}]")
    n_weak = sum(c.origin == "weak" for c in rv.cands)
    print(f"  {len(rv.labels)} proposed moves, {n_weak} sub-threshold "
          f"candidates, {sum(c.blind for c in rv.cands)} blinded")
    print(f"  review head: frame {rv.head} "
          f"({rv.head_idx}/{len(rv.labels)} labels reviewed)")
    print(f"  cube: {rv.distance_note()}")
    print(f"  decode: {rv.decode_note}")

    if rv.flags:
        print(f"\n  {len(rv.flags)} flag(s) to review:")
        for f in rv.flags:
            span = (f"frames {f.lo}-{f.hi} "
                    f"({f.lo/sess.fps:.2f}-{f.hi/sess.fps:.2f}s)"
                    if f.hi > f.lo else
                    f"frame {f.lo} ({f.lo/sess.fps:.2f}s)")
            print(f"    [{f.kind:<8}] {span:<38} {f.note}")
    print(f"\n  sequence ({len(rv.word)}): {' '.join(rv.word)}")


# ---------------------------------------------------------------------------
# The interactive loop
# ---------------------------------------------------------------------------

# waitKeyEx codes for the keys that have no ASCII value (Windows).
K_LEFT, K_RIGHT, K_UP, K_DOWN = 2424832, 2555904, 2490368, 2621440
K_PGUP, K_PGDN, K_HOME, K_END = 2162688, 2228224, 2359296, 2293760
K_ENTER, K_ESC, K_DEL = 13, 27, 3014656

FACE_KEYS = {ord("u"): "U", ord("d"): "D", ord("l"): "L",
             ord("r"): "R", ord("f"): "F", ord("b"): "B"}


def run_ui(rv: Review, args) -> None:
    import cv2

    rend = Renderer(rv)
    win = f"review — {rv.sess.path.name}"
    cv2.namedWindow(win)

    state = {"playhead": rv.cands[0].frame if rv.cands else 0,
             "sel": 0, "playing": False, "speed": 1.0, "msg": "",
             "seek": None, "quit_armed": False}

    def on_mouse(event, x, y, flags, _):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            return
        if event == cv2.EVENT_MOUSEMOVE and not (flags & cv2.EVENT_FLAG_LBUTTON):
            return
        for name in ("global", "zoom"):
            rx, ry, rw, rh = rend.rects.get(name, (0, 0, 0, 0))
            if rx <= x < rx + rw and ry <= y < ry + rh:
                if name == "global":
                    lo, hi = 0, rv.sess.n_frames
                else:
                    lo, hi = rend.zoom_range(state["playhead"])
                state["seek"] = int(lo + (x - rx) / max(rw, 1) * (hi - lo))

    cv2.setMouseCallback(win, on_mouse)

    def select_nearest(frame):
        if not rv.cands:
            return
        state["sel"] = int(np.argmin([abs(c.frame - frame)
                                      for c in rv.cands]))

    def current():
        return rv.cands[state["sel"]] if 0 <= state["sel"] < len(rv.cands) \
            else None

    def set_face(face):
        """Face key: retarget the selected candidate, or insert a new move."""
        c = current()
        near = c is not None and abs(c.frame - state["playhead"]) <= CLAIM_RADIUS
        if near:
            c.move = face + ("'" if c.move.endswith("'") else "")
            c.typed = True
            if c.blind and not c.blind_answer:
                # Stay blind until `y`: the reviewer may still add a prime,
                # and committing on the first keystroke would record half an
                # answer and reveal the model's before they finished theirs.
                state["msg"] = f"blind: {c.move} — press y to commit"
                return
            c.accepted, c.confirmed = True, True
        else:
            c = rv.insert_at(state["playhead"], face)
            select_nearest(state["playhead"])
            state["msg"] = f"inserted {face} at frame {state['playhead']}"
        rv.touch()

    last_tick = time.time()
    while True:
        canvas = rend.render(state["playhead"], state["sel"], state["msg"])
        cv2.imshow(win, canvas)
        key = cv2.waitKeyEx(15)

        if rv.poll_decode():
            state["msg"] = rv.decode_note

        if state["seek"] is not None:
            state["playhead"] = int(np.clip(state["seek"], 0,
                                            rv.sess.n_frames - 1))
            select_nearest(state["playhead"])
            state["seek"] = None

        if state["playing"]:
            now = time.time()
            step = int((now - last_tick) * rv.sess.fps * state["speed"])
            if step:
                last_tick = now
                state["playhead"] = min(rv.sess.n_frames - 1,
                                        state["playhead"] + step)
                if state["playhead"] >= rv.sess.n_frames - 1:
                    state["playing"] = False
        else:
            last_tick = time.time()

        if key == -1:
            continue
        c = current()
        state["msg"] = ""
        if key not in (ord("q"), K_ESC):
            state["quit_armed"] = False

        if key in (ord("q"), K_ESC):
            if rv.dirty and not state["quit_armed"]:
                state["msg"] = ("unsaved changes — press s to save, or q "
                                "again to discard them")
                state["quit_armed"] = True
                continue
            break
        elif key == ord(" "):
            state["playing"] = not state["playing"]
            last_tick = time.time()
        elif key in (K_LEFT, ord(",")):
            state["playhead"] = max(0, state["playhead"] - 1)
        elif key in (K_RIGHT, ord(".")):
            state["playhead"] = min(rv.sess.n_frames - 1, state["playhead"] + 1)
        elif key in (K_PGUP, ord("<")):
            state["playhead"] = max(0, state["playhead"] - 15)
        elif key in (K_PGDN, ord(">")):
            state["playhead"] = min(rv.sess.n_frames - 1,
                                    state["playhead"] + 15)
        elif key == K_HOME:
            state["playhead"] = 0
        elif key == K_END:
            state["playhead"] = rv.sess.n_frames - 1
        elif key in (K_UP, K_DOWN, ord("j"), ord("k")):
            # Walking the move LIST is the primary motion, so it gets both
            # the arrows and a home-row pair — the arrow codes come back
            # from waitKeyEx and are platform-dependent, and a review tool
            # that silently loses its main control on some machine is worse
            # than one with redundant keys.
            back = key in (K_UP, ord("j"))
            if rv.cands:
                state["sel"] = int(np.clip(state["sel"] + (-1 if back else 1),
                                           0, len(rv.cands) - 1))
                state["playhead"] = rv.cands[state["sel"]].frame
        elif key in (ord("["), ord("]")):
            if rv.flags:
                order = sorted(rv.flags, key=lambda f: f.at)
                if key == ord("]"):
                    nxt = next((f for f in order if f.at > state["playhead"]),
                               order[0])
                else:
                    nxt = next((f for f in reversed(order)
                                if f.at < state["playhead"]), order[-1])
                state["playhead"] = nxt.at
                select_nearest(nxt.at)
                state["msg"] = f"flag: {nxt.note}"
            else:
                state["msg"] = "no flags — press g to run the decode"
        elif key in FACE_KEYS:
            set_face(FACE_KEYS[key])
        elif key == ord("'"):
            if c is not None:
                c.move = c.move[0] if c.move.endswith("'") else c.move + "'"
                c.typed = True
                if c.blind and not c.blind_answer:
                    state["msg"] = f"blind: {c.move} — press y to commit"
                else:
                    c.accepted, c.confirmed = True, True
                    rv.touch()
        elif key == ord("i"):
            fl = rend.nearest_flag(state["playhead"])
            mv = (fl.move if fl is not None and fl.kind == "insert" and fl.move
                  else WCA12[int(rv.sess.class_prob[state["playhead"],
                                                    :12].argmax())])
            rv.insert_at(state["playhead"], mv)
            select_nearest(state["playhead"])
            state["msg"] = f"inserted {mv} at frame {state['playhead']}"
        elif key == ord("y"):
            if c is not None:
                if c.blind and not c.blind_answer and not c.typed:
                    state["msg"] = ("blind: type the move you see (u d l r f "
                                    "b, then ' for a prime) before committing")
                elif c.blind and not c.blind_answer:
                    c.blind_answer = c.move
                    c.accepted, c.confirmed = True, True
                    rv.touch()
                    state["msg"] = (
                        f"you said {c.blind_answer}, model said {c.proposed}"
                        f" — {'AGREE' if c.blind_answer == c.proposed else 'DIFFER'}")
                else:
                    c.accepted, c.confirmed = True, True
                    rv.touch()
                    state["msg"] = f"confirmed {c.move}"
        elif key in (ord("x"), K_DEL):
            if c is not None:
                if c.origin == "manual" and not c.accepted:
                    rv.cands.remove(c)
                    state["sel"] = max(0, state["sel"] - 1)
                    state["msg"] = "removed inserted move"
                else:
                    c.accepted, c.confirmed = False, True
                    state["msg"] = f"rejected {c.move}"
                rv.touch()
        elif key == K_ENTER:
            fl = rend.nearest_flag(state["playhead"])
            if fl is None:
                state["msg"] = "no flag here"
            else:
                state["msg"] = rv.apply_flag(fl, state["playhead"])
                select_nearest(state["playhead"])
        elif key == ord("m"):
            behind = [c for c in rv.labels
                      if rv.head <= c.frame < state["playhead"]
                      and not c.confirmed]
            rv.head = state["playhead"]
            rv.touch()
            rv.start_decode()
            state["msg"] = (f"reviewed to frame {rv.head}"
                            + (f" ({len(behind)} labels in that span were "
                               f"never individually confirmed)" if behind
                               else "") + " — decoding")
        elif key == ord("M"):
            rv.head = 0
            rv.touch()
            state["msg"] = "review head reset to the start"
        elif key == ord("g"):
            rv.start_decode()
            state["msg"] = "decoding..."
        elif key in (ord("z"), ord("Z")):
            rend.zoom_half = int(np.clip(rend.zoom_half * (2 if key == ord("z")
                                                           else 0.5), 10, 4000))
        elif key == ord("-"):
            state["speed"] = max(0.1, state["speed"] / 2)
            state["msg"] = f"speed {state['speed']:.2f}x"
        elif key in (ord("="), ord("+")):
            state["speed"] = min(8.0, state["speed"] * 2)
            state["msg"] = f"speed {state['speed']:.2f}x"
        elif key == ord("s"):
            p = save(rv, args.reviewer)
            state["msg"] = f"saved {p.name}"
        elif key in (ord("?"), ord("/")):
            rend.show_help = not rend.show_help

    cv2.destroyAllWindows()
    a = rv.audit()
    print(f"\n  {len(rv.labels)} labels, {rv.distance_note()}")
    if a["blind_agreement"] is not None:
        print(f"  blind audit:   {a['blind_agreement']*100:.0f}% agreement "
              f"over {a['n_blind_answered']} hidden moves")
    if a["sighted_agreement"] is not None:
        print(f"  sighted:       {a['sighted_agreement']*100:.0f}% agreement "
              f"over {a['n_sighted_confirmed']} reviewed moves")
    if (a["blind_agreement"] is not None and a["sighted_agreement"] is not None
            and a["sighted_agreement"] - a["blind_agreement"] > 0.05):
        print("  NOTE: sighted agreement exceeds blind agreement — some of "
              "this review was confirmation rather than checking.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--reviewer", default="")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing review_labels.json")
    p.add_argument("--report", action="store_true",
                   help="print candidates and flags, then exit (no GUI)")
    p.add_argument("--shot", default=None,
                   help="render one UI frame to this PNG and exit")
    p.add_argument("--shot-frame", type=int, default=None,
                   help="which frame --shot renders (default: the first flag)")

    g = p.add_argument_group("start state (needed for the miss flags)")
    g.add_argument("--start-moves", default=None,
                   help="the scramble word, camera-relative WCA quarter turns")
    g.add_argument("--start-state", default=None,
                   help="54-char URFDLB facelets, CAMERA frame")
    g.add_argument("--start-from-truth", action="store_true",
                   help="derive the start state from moves.jsonl (dev only; "
                        "reads the net transformation, never the sequence)")

    g = p.add_argument_group("candidates")
    g.add_argument("--threshold", type=float, default=0.5,
                   help="deployed peak-pick operating point")
    g.add_argument("--candidate-threshold", type=float,
                   default=CANDIDATE_THRESHOLD,
                   help="surface sub-threshold peaks down to this score as "
                        "reviewable (default-rejected) candidates; "
                        "-1 disables")
    g.add_argument("--beam-ctc", type=int, default=16)
    g.add_argument("--blind-rate", type=float, default=0.0,
                   help="fraction of proposals whose label is hidden until "
                        "the reviewer answers (rubber-stamp audit)")

    g = p.add_argument_group("flag decode")
    g.add_argument("--head", type=int, default=None,
                   help="start with the review head at this FRAME (the UI "
                        "moves it with `m`); mostly for --report")
    g.add_argument("--flag-beam", type=int, default=RC.BEAM)
    g.add_argument("--flag-max-chain", type=int, default=RC.MAX_CHAIN,
                   help="consecutive missing moves the decoder may propose "
                        "in one gap; raising this makes flags hints rather "
                        "than proof (see reconstruct.py on budget soundness)")
    g.add_argument("--lock-cost", type=float, default=LOCK_COST)
    g.add_argument("--del-cost", type=float, default=RC.C_DEL)
    g.add_argument("--ins-cost", type=float, default=RC.C_INS)
    g.add_argument("--max-end-ins", type=int, default=RC.MAX_END_INS)
    g.add_argument("--del-floor", type=float, default=RC.DEL_FLOOR)
    g.add_argument("--blend-inv", type=float, default=RC.BLEND_INV)
    g.add_argument("--blend-unif", type=float, default=RC.BLEND_UNIF)
    g.add_argument("--slices", action="store_true")
    g.add_argument("--slice-gate", type=float, default=RC.SLICE_GATE)
    args = p.parse_args()

    if args.candidate_threshold is not None and args.candidate_threshold < 0:
        args.candidate_threshold = None

    session = Path(args.session)
    if not session.is_dir():
        sys.exit(f"{session} is not a directory")

    sess = Session(session, Path(args.model), args)
    proposals = decode_proposals(sess, args)
    cands = build_candidates(sess, proposals, args)
    start = resolve_start_state(session, args)
    if start is None:
        print("  no start state given — the group-theory miss flags are OFF. "
              "Pass --start-moves / --start-state / --start-from-truth.")

    rv = Review(sess, cands, start, args)
    if not args.no_resume:
        restore(rv)
    if args.head is not None:
        rv.head = int(np.clip(args.head, 0, sess.n_frames - 1))
    rv.refresh_flags()

    if args.report:
        report(rv)
        return
    if args.shot:
        import cv2
        rend = Renderer(rv)
        head = (args.shot_frame if args.shot_frame is not None else
                rv.flags[0].at if rv.flags else
                cands[0].frame if cands else 0)
        sel = int(np.argmin([abs(c.frame - head) for c in cands])) if cands \
            else 0
        cv2.imwrite(args.shot, rend.render(head, sel, "rendered by --shot"))
        print(f"  wrote {args.shot}")
        return

    run_ui(rv, args)


if __name__ == "__main__":
    main()
