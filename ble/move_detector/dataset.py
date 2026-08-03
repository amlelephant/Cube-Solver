"""
dataset.py

Clip sampling and target construction for the onset detector.

Targets
-------
Not a hard 0/1 per frame. Each BLE onset becomes a Gaussian bump of width
--sigma frames centered on the frame nearest that timestamp:

    y[i] = max over onsets o of  exp(-(i - o)^2 / (2 sigma^2))

Two reasons for the soft target rather than a single positive frame:

  * BLE timestamp -> frame index is only accurate to about half a frame
    (16ms at 30fps), and the physical turn spans ~5 frames anyway. A hard
    label on one frame would train the model to hit a boundary that is not
    actually that sharp in the pixels.
  * A single positive frame per ~13 makes the problem needlessly
    imbalanced. At sigma=2 the target averages ~0.4, so plain BCE works
    without any positive reweighting.

This is the standard onset-detection formulation (same shape as audio
onset detection), which is what makes peak-picking a sound decoder —
see decode.py.

Sigma has a cost the two reasons above do not mention: it sets how close
two onsets can be and still be *representable* as two peaks. At sigma=2
the bumps for onsets 1-2 frames apart merge into one blob, so the
supervision never asks for two peaks there and no amount of data teaches
the model to emit them. Measured on held-out sessions, 45% of pairs
closer than 150ms came out as a single merged hump at sigma=2 versus 34%
at sigma=1, and sub-150ms recall went 78.6% -> 83.5% with no regression
above 150ms.

SIGMA is 1.0 as of 2026-07-22, confirmed by two independent runs (34% and
31% merged, against 45% at sigma=2).

The positive mass roughly halves with it — measured 0.272 at sigma=2
against 0.148 at sigma=1 — which looked like it should need a pos_weight
of about 1.8 to compensate. It does not: pos_weight=1.8 at sigma=1 gives
byte-identical sub-150ms recall (83.5%, the same 17 misses) while costing
precision (92.5% -> 89.7%) and recall above 150ms (97.0% -> 96.2%). The
imbalance was never what was limiting this, which in hindsight sigma=1
already implied by winning despite it. train.py --pos-weight is kept,
defaulting to 1.0, so the knob is there if a future target change makes it
relevant.

Splitting
---------
Whole sessions are held out, never individual clips. Clips from one
session overlap and share lighting, grip and background; a random clip
split would report memorization. Same reasoning as --holdout session in
train_move_classifier.py.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

STREAM_FILE = "detector_stream.npz"
CLIP_LEN    = 96      # frames per training clip (~3.2s at 30fps)
SIGMA       = 1.0     # Gaussian target width, in frames — see Targets above;
                      # this also caps how close two onsets can be and still
                      # be representable as two peaks


class ArrayStream:
    """
    A frame array plus its onsets — the unit everything else consumes.

    Split out from SessionStream so live capture can build one in memory
    and feed it to model.score_stream() unchanged. Inference then runs on
    exactly the code path training used, rather than a parallel
    reimplementation that can silently drift out of sync.
    """

    def __init__(self, frames: np.ndarray, name: str = "live",
                 fps: float = 30.0, onset_idx: np.ndarray | None = None,
                 sigma: float = SIGMA):
        self.frames    = frames                          # (N, H, W) uint8
        self.name      = name
        self.fps       = fps
        self.onset_idx = np.asarray(onset_idx, dtype=int) \
            if onset_idx is not None else np.array([], dtype=int)
        self.sigma     = sigma
        self.target    = self._build_target()

    def __len__(self) -> int:
        return len(self.frames)

    def _build_target(self) -> np.ndarray:
        n = len(self.frames)
        y = np.zeros(n, dtype=np.float32)
        reach = max(1, int(np.ceil(3 * self.sigma)))
        for o in self.onset_idx:
            lo, hi = max(0, o - reach), min(n, o + reach + 1)
            d = np.arange(lo, hi) - o
            y[lo:hi] = np.maximum(y[lo:hi],
                                  np.exp(-(d ** 2) / (2 * self.sigma ** 2)))
        return y

    def clip_block(self, start: int, length: int) -> np.ndarray:
        """
        Raw uint8 block of (length + 1) frames starting one frame early, so
        a temporal diff can be taken for every frame of the clip. The
        lead-in frame is duplicated at the very start of a session.
        """
        lead = max(0, start - 1)
        block = self.frames[lead:start + length]
        if start == 0:
            block = np.concatenate([self.frames[:1], block], axis=0)
        return block


class SessionStream(ArrayStream):
    """One prepared session loaded from its detector_stream.npz."""

    def __init__(self, path: Path, sigma: float = SIGMA):
        data = np.load(path, allow_pickle=True)
        super().__init__(frames=data["frames"],
                         name=str(data["name"]),
                         fps=float(data["fps"]),
                         onset_idx=data["onset_idx"].astype(int),
                         sigma=sigma)
        # Written by prepare_data.py since 2026-07-25. Streams prepared
        # before that carry no field; "unknown" is reported as such rather
        # than assumed clean — the whole point is that a silent assumption
        # here is what cost the classifier a day.
        self.crop_mode = (str(data["crop_mode"]) if "crop_mode" in data
                          else "unknown")
        # Written by prepare_data.py since 2026-08-03 (--labels). Streams
        # prepared before that are all BLE-labelled — there was no other
        # source — so the fallback here is a fact, not an assumption.
        self.label_source = (str(data["label_source"])
                             if "label_source" in data else "ble")


def to_tensor(block: np.ndarray) -> torch.Tensor:
    """
    (T+1, H, W) uint8 -> (T, 2, H, W) float32.
    Channel 0 = grayscale, channel 1 = signed diff from the previous frame.
    BatchNorm in the encoder handles the remaining scale difference.
    """
    a = block.astype(np.float32) / 255.0
    gray = a[1:] - 0.5
    diff = a[1:] - a[:-1]
    return torch.from_numpy(np.stack([gray, diff], axis=1))


# ---------------------------------------------------------------------------
# Augmentation — all of it must be identical across every frame of a clip.
# A per-frame jitter would create global motion that looks exactly like the
# turn the model is supposed to detect.
# ---------------------------------------------------------------------------

def augment_block(block: np.ndarray, rng: random.Random) -> np.ndarray:
    out = block

    # Horizontal flip. Unlike the move CLASSIFIER (where a mirror turns CW
    # into CCW and the label must be remapped through WCA_FLIP), the
    # detector only answers "is a turn happening", which a mirror leaves
    # unchanged. Free doubling, no label surgery.
    if rng.random() < 0.5:
        out = out[:, :, ::-1]

    # Brightness / contrast, applied before diffing so the diff scales
    # consistently with the frames it came from.
    alpha = rng.uniform(0.85, 1.15)
    beta  = rng.uniform(-18, 18)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255)

    # Whole-clip translation: shifts where the cube sits, not how it moves.
    sx, sy = rng.randint(-6, 6), rng.randint(-6, 6)
    if sx or sy:
        out = np.roll(out, shift=(sy, sx), axis=(1, 2))

    return out.astype(np.uint8)


class OnsetClipDataset(Dataset):
    """
    Fixed-length clips drawn from a set of sessions.

    Training uses a dense stride so clips overlap heavily (cheap, and every
    onset is seen at many temporal offsets). Validation does NOT use this
    class — it scores whole sessions via model.score_stream(), which is how
    the detector will actually be run.
    """

    def __init__(self, streams: list[SessionStream], clip_len: int = CLIP_LEN,
                 stride: int = 24, augment: bool = True, seed: int = 0):
        self.streams  = streams
        self.clip_len = clip_len
        self.augment  = augment
        self.rng      = random.Random(seed)
        self.index    = []   # (stream_i, start)
        for si, s in enumerate(streams):
            last = len(s) - clip_len
            if last < 0:
                continue
            self.index += [(si, st) for st in range(0, last + 1, stride)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        si, start = self.index[i]
        s = self.streams[si]

        block = s.clip_block(start, self.clip_len)
        if self.augment:
            block = augment_block(block, self.rng)

        y = s.target[start:start + self.clip_len]
        return to_tensor(block), torch.from_numpy(y.copy())


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def load_streams(session_dirs: list[Path], sigma: float = SIGMA
                 ) -> list[SessionStream]:
    streams, missing = [], []
    for d in sorted(session_dirs):
        p = d / STREAM_FILE
        if p.exists():
            streams.append(SessionStream(p, sigma=sigma))
        else:
            missing.append(d.name)
    if missing:
        print(f"  {len(missing)} session(s) have no {STREAM_FILE} and were "
              f"skipped:")
        for name in missing[:6]:
            print(f"    {name}")
        if len(missing) > 6:
            print(f"    ... and {len(missing) - 6} more")
        print(f"  Run prepare_data.py first (sessions recorded without "
              f"--keep-frames cannot be prepared).")
    return streams


def check_label_regime(streams) -> str:
    """
    Report where each stream's onset labels came from.

    Deliberately NOT a refusal, unlike check_crop_regime. A mixed crop
    regime is always a defect — it trains on two input distributions and
    tests on one. A mixed LABEL regime is the intended growth path: a
    reviewed session (review.py, gated on the label set reaching solved)
    is supposed to sit alongside smart-cube sessions, and refusing the mix
    would refuse the only way the corpus grows past the hardware.

    What it must never be is invisible. The two sources have different
    failure modes and the difference shows up in the trained model rather
    than in any file: BLE labels carry ~1 frame of timestamp alignment
    noise and lose 10.1% of moves to 30ms-tick collisions, while reviewed
    labels are frame-exact but inherit whatever a human missed, and both
    inherit whatever the reviewer's model proposed. Knowing the split is
    what makes a later regression attributable.

    Returns "ble", "review", or "mixed".
    """
    src: dict[str, list[str]] = {}
    for s in streams:
        src.setdefault(getattr(s, "label_source", "ble"), []).append(s.name)

    # The one refusal. A `--labels none` stream was built to be SCORED, not
    # trained on: it has no onsets, so every frame of it is a negative and
    # it teaches the model that a whole session contains no moves. That is
    # the miss-poison failure at session scale, and it is silent — the
    # stream loads, trains, and merely makes the model worse. Not
    # overridable, because there is no case where it is what you meant.
    if "none" in src:
        names = src["none"]
        raise SystemExit(
            f"{len(names)} stream(s) have no labels at all (prepared with "
            f"--labels none, e.g. {names[0]}).\nThose exist only so review.py "
            f"can score an unlabelled video. Review them, then re-prepare:\n\n"
            f"    python prepare_data.py --sessions <session> --color "
            f"--labels review --force\n")

    if len(src) == 1:
        only = next(iter(src))
        label = {"ble": "smart-cube ground truth",
                 "review": "human review"}.get(only, only)
        print(f"  Labels:    all {len(streams)} stream(s) from {label}")
        return only

    print(f"\n  Label sources across {len(streams)} stream(s):")
    for mode, names in sorted(src.items()):
        print(f"    {mode:<14} {len(names):>3}   e.g. {names[0]}")
    print(f"  Mixed by design — reviewed sessions are how the corpus grows "
          f"past the\n  smart cube. Recorded here so a later regression can "
          f"be attributed.\n")
    return "mixed"


def check_crop_regime(streams: list[SessionStream],
                      allow_mixed: bool = False) -> str:
    """
    Report the crop regime across a set of streams; refuse a mixed one.

    Live inference (live_detect.analyse) crops every frame to the cube box.
    A stream prepared with --no-crop, or one where the cube was never
    detected, is a centered square instead — a completely different scale
    for the same 96x96 tensor. Training on both teaches the model to
    average over two input distributions and then be tested on one of them.
    This is the same failure the move CLASSIFIER shipped with until
    2026-07-24; it is checked here so the detector cannot repeat it.
    """
    modes: dict[str, list[str]] = {}
    for s in streams:
        modes.setdefault(s.crop_mode, []).append(s.name)

    if set(modes) == {"cropped"}:
        print(f"  Crops:     all {len(streams)} stream(s) cube-cropped")
        return "cropped"

    print(f"\n  Crop regime across {len(streams)} stream(s):")
    for mode, names in sorted(modes.items()):
        print(f"    {mode:<14} {len(names):>3}"
              + (f"   e.g. {names[0]}" if mode != "cropped" else ""))

    if "unknown" in modes and len(modes) == 1:
        print(f"\n  Every stream predates crop provenance. Re-run "
              f"prepare_data.py --force to\n  record it, or accept that "
              f"a --no-crop stream could be hiding in here.")
        return "unknown"

    bad = [n for m, names in modes.items() if m != "cropped" for n in names]
    print(f"\n  {len(bad)} stream(s) are not cube-cropped while live "
          f"inference always crops.\n  Re-prepare them with a working "
          f"detector:\n\n      python prepare_data.py --sessions "
          f"../training_data/solve_*/ --force\n")
    if not allow_mixed:
        raise SystemExit("Refusing to train on a mixed crop regime "
                         "(--allow-uncropped to override).")
    print(f"  --allow-uncropped given; proceeding on a mixed regime.\n")
    return "mixed"


def split_streams(streams: list[SessionStream], val_sessions: int | None = None,
                  seed: int = 42, val_names: list[str] | None = None
                  ) -> tuple[list, list]:
    """
    Hold out whole sessions. Returns (train_streams, val_streams).

    `val_names` names the validation sessions explicitly, overriding the
    random pick. That matters once the sessions span more than one recording
    environment: a random holdout can land entirely inside one of them, and
    then the reported number measures within-environment fit while looking
    exactly like a cross-environment result. Naming the holdout is how the
    split is made to answer the question actually being asked.
    """
    if len(streams) < 2:
        raise SystemExit(
            f"Need at least 2 prepared sessions to hold one out, found "
            f"{len(streams)}. Record and prepare more solves.")

    if val_names:
        wanted = set(val_names)
        known = {s.name for s in streams}
        missing = wanted - known
        if missing:
            raise SystemExit(
                f"--val-session-names not found among the loaded sessions: "
                f"{sorted(missing)}\nLoaded: {sorted(known)}")
        if len(wanted) >= len(streams):
            raise SystemExit("Every session was named as validation; "
                             "nothing left to train on.")
        return ([s for s in streams if s.name not in wanted],
                [s for s in streams if s.name in wanted])

    n_val = val_sessions if val_sessions is not None \
        else max(1, round(len(streams) * 0.2))
    n_val = min(n_val, len(streams) - 1)

    order = list(range(len(streams)))
    random.Random(seed).shuffle(order)
    val_ids = set(order[:n_val])
    return ([s for i, s in enumerate(streams) if i not in val_ids],
            [s for i, s in enumerate(streams) if i in val_ids])


def split_clips_pooled(dataset: "OnsetClipDataset", val_frac: float = 0.2,
                       seed: int = 42) -> tuple[list[int], list[int]]:
    """
    Pool every clip from every session and split them randomly val_frac/rest.

    Every session contributes to BOTH training and validation, which is the
    point — no solve is withheld from training.

    Read the number this produces with care. Clips are sampled at `stride`
    frames from a `clip_len`-frame window, so neighbouring clips overlap by
    (clip_len - stride) frames — at the defaults that is 72 of 96 frames,
    75%. A random split therefore puts clip[start=0] in train and
    clip[start=24] in val while they share three quarters of their pixels
    AND the onsets inside them. The resulting F1 measures how well the model
    reproduces frames it was optimised on, not whether it generalises.

    Use split_streams() (whole-session holdout) for a number that answers
    "will this work on the next solve". Use this one to squeeze every
    session into training once that question is already settled.

    overlap_frac() below reports exactly how contaminated a given
    configuration is; --stride >= --clip-len drives it to zero.
    """
    idx = list(range(len(dataset)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(idx) * val_frac))
    return idx[n_val:], idx[:n_val]


def overlap_frac(clip_len: int, stride: int) -> float:
    """Fraction of frames a clip shares with its immediate neighbour."""
    return max(0.0, (clip_len - stride) / clip_len)


# ---------------------------------------------------------------------------
# Stage A (MODEL_REWORK_PLAN.md): joint onset+class model — colour streams,
# dense per-frame 13-way targets. Everything above this line (the shipped
# onset-only detector) is untouched.
# ---------------------------------------------------------------------------

STREAM_FILE_COLOR = "detector_stream_color.npz"   # see prepare_data.py --color

# Horizontal-flip class remap, copied from train_move_classifier.WCA_FLIP
# (kept as a local literal rather than imported, so this module does not
# have to pull that file's heavier top-level in just for one table — MUST
# stay in sync if that table ever changes). U/D/F/B just reverse direction
# (prime toggle); L<->R' and L'<->R because mirroring swaps which
# PHYSICAL side is which, not just the turn's direction.
WCA_FLIP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5,
           8: 9, 9: 8, 10: 11, 11: 10}
WCA_FLIP_PERM = np.array([WCA_FLIP[i] for i in range(12)])


def build_dense_targets(onset_idx: np.ndarray, onset_class: np.ndarray,
                        n: int, sigma: float,
                        onset_target: np.ndarray) -> np.ndarray:
    """
    (n, 13) per-frame target: columns 0-11 are the 12 WCA quarter-turn
    classes, column 12 is background.

    Reuses `onset_target` (the SAME Gaussian-bump array the onset head
    trains against — ArrayStream._build_target) as the "how much move-mass
    is here" total, so the class head's background row always agrees with
    the onset head's own belief; only how that total mass splits across
    the 12 classes is new here. Per-class bumps use the identical
    sigma/reach formula, np.maximum'd WITHIN a class (same semantics as
    the onset target for repeated same-class onsets close together). A
    rare frame where two DIFFERENT classes' bumps overlap (near-
    simultaneous different-face turns) is rescaled proportionally so the
    row still sums to onset_target[i] exactly — every frame is a valid
    probability simplex over the 13 columns, which is what lets this train
    with plain per-frame cross-entropy.
    """
    reach = max(1, int(np.ceil(3 * sigma)))
    bumps = np.zeros((n, 12), dtype=np.float32)
    for o, c in zip(onset_idx, onset_class):
        lo, hi = max(0, o - reach), min(n, o + reach + 1)
        d = np.arange(lo, hi) - o
        b = np.exp(-(d ** 2) / (2 * sigma ** 2))
        bumps[lo:hi, c] = np.maximum(bumps[lo:hi, c], b)

    raw_sum = bumps.sum(axis=1)
    scale = np.divide(onset_target, raw_sum, out=np.zeros_like(raw_sum),
                      where=raw_sum > 1e-8)
    class_target = bumps * scale[:, None]
    bg = 1.0 - onset_target
    return np.concatenate([class_target, bg[:, None]], axis=1).astype(np.float32)


COUNT_RADIUS  = 2    # see build_count_target
COUNT_CLASSES = 3    # 0, 1, 2-or-more onsets in the bin


def build_count_target(onset_idx: np.ndarray, n: int,
                       radius: int = COUNT_RADIUS) -> np.ndarray:
    """
    (n,) int64 in {0, 1, 2}: how many onsets fall within +/-`radius` frames
    of frame i, clamped at 2.

    Why a BINNED count rather than a per-frame one
    ----------------------------------------------
    The motivating case is two-handed simultaneous turns, and those are
    real: of the 32 pairs of onsets that land on the SAME frame index
    across all 42 prepared sessions, 26 are opposite-face (R with L', U
    with D', ...) — physically one two-handed motion, exactly as expected.

    But 32 frames in 70192 is 0.05% positive mass, which is not a
    trainable class. Widening the bin is what makes this learnable, and
    the measured trade is:

        radius 0  (33ms bin)      32 frames with count>=2   0.05%
        radius 1  (100ms bin)    301                        0.43%
        radius 2  (167ms bin)   1455                        2.00%
        radius 3  (233ms bin)   3286                        4.68%

    radius=2 is chosen because it is also the regime the DECODER fails in,
    which is the whole point of this head. decode.MIN_SEP is 2, and
    peak_pick needs a strict local maximum, so onsets within ~2 frames of
    each other are precisely the ones it structurally cannot both report —
    decode.py measures 34% of sub-150ms pairs arriving as a single merged
    hump even at sigma=1. A count head asks the model to LABEL that frame
    "2" instead of having to render two resolvable peaks into a smoothed
    curve, so recovering the second move becomes reading an integer rather
    than resolving a shape.

    Flip-invariant (mirroring an image does not change how many turns are
    in it), so unlike class_target this needs no WCA_FLIP_PERM treatment.
    """
    cnt = np.zeros(n, dtype=np.int64)
    for o in onset_idx:
        lo, hi = max(0, o - radius), min(n, o + radius + 1)
        cnt[lo:hi] += 1
    return np.minimum(cnt, COUNT_CLASSES - 1)


class JointArrayStream(ArrayStream):
    """
    Stage A's colour counterpart of ArrayStream: (N, H, W, 3) BGR frames,
    the same onset target as the parent (reused verbatim — see
    build_dense_targets), plus a (N, 13) class_target and a (N,) integer
    count_target (build_count_target).
    """

    def __init__(self, frames: np.ndarray, name: str = "live",
                 fps: float = 30.0, onset_idx: np.ndarray | None = None,
                 onset_class: np.ndarray | None = None, sigma: float = SIGMA,
                 count_radius: int = COUNT_RADIUS):
        self.onset_class = np.asarray(onset_class, dtype=int) \
            if onset_class is not None else np.array([], dtype=int)
        super().__init__(frames=frames, name=name, fps=fps,
                         onset_idx=onset_idx, sigma=sigma)
        self.class_target = build_dense_targets(
            self.onset_idx, self.onset_class, len(self.frames), self.sigma,
            self.target)
        self.count_radius = count_radius
        self.count_target = build_count_target(
            self.onset_idx, len(self.frames), count_radius)


class JointSessionStream(JointArrayStream):
    """One prepared session loaded from its detector_stream_color.npz."""

    def __init__(self, path: Path, sigma: float = SIGMA,
                 count_radius: int = COUNT_RADIUS):
        data = np.load(path, allow_pickle=True)
        super().__init__(frames=data["frames"], name=str(data["name"]),
                         fps=float(data["fps"]),
                         onset_idx=data["onset_idx"].astype(int),
                         onset_class=data["onset_class"].astype(int),
                         sigma=sigma, count_radius=count_radius)
        self.crop_mode = (str(data["crop_mode"]) if "crop_mode" in data
                          else "unknown")
        # Written by prepare_data.py since 2026-08-03 (--labels). Streams
        # prepared before that are all BLE-labelled — there was no other
        # source — so the fallback here is a fact, not an assumption.
        self.label_source = (str(data["label_source"])
                             if "label_source" in data else "ble")


def to_tensor_color(block: np.ndarray) -> torch.Tensor:
    """
    (T+1, H, W, 3) uint8 BGR -> (T, 4, H, W) float32.

    Channels: B, G, R (OpenCV's native order — arbitrary but must match
    prepare_data.build_color_stream and live inference), each centred to
    [-0.5, 0.5] like to_tensor's grayscale channel, plus a diff-LUMA
    channel built with the IDENTICAL formula to_tensor uses for its diff
    channel (signed luma difference from the previous frame) — the onset
    half of the shared trunk sees the same signal it always has; only
    colour is new.
    """
    a = block.astype(np.float32) / 255.0                  # (T+1, H, W, 3)
    color = a[1:] - 0.5                                     # (T, H, W, 3)
    luma = a[..., 2] * 0.299 + a[..., 1] * 0.587 + a[..., 0] * 0.114
    diff = (luma[1:] - luma[:-1])[..., None]                # (T, H, W, 1)
    stacked = np.concatenate([color, diff], axis=-1)        # (T, H, W, 4)
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(stacked, -1, 1)))


# --- photometric augmentation ranges ------------------------------------
#
# Widened 2026-07-31 after a live take at 21:11 scored 56.0% end-to-end
# against ~91% on two morning takes, and adding a lamp to the SAME evening
# sitting recovered it to 76.9%. Lighting is causal; the corpus is 54 of 62
# sessions between 09:00 and 18:00.
#
# What the previous ranges could and could not express matters, because it
# is not obvious from reading them. They were one alpha in (0.85, 1.15) and
# one beta in (-18, 18) applied to the WHOLE clip, all three channels
# together. Against the diff-luma channel — which is what the onset half of
# the trunk actually consumes — a per-clip constant beta cancels EXACTLY in
# the frame difference, and a per-clip constant alpha only rescales it. So
# the diff channel received +/-15% of augmentation and nothing else, while
# the measured morning-vs-evening shift in its baseline is 5x (noise floor
# 5.5-5.8 by day, 1.18-1.30 at night: long exposure plus in-camera temporal
# denoising).
#
# Hence the split below: PER-CLIP terms move the appearance (gain, offset,
# gamma, white balance, saturation), PER-FRAME terms move the temporal
# statistics (gain wobble, sensor noise, exposure-like temporal blur) and
# are the only ones the diff channel can see at all.
#
# Honest caveat, so this is not over-read: raising the evening diff
# baseline back to daytime levels AT INFERENCE was measured and came out
# flat (56.0/57.1/56.0/54.8% at noise sigma 0/4/8/12). The causal channel
# is NOT isolated. These ranges are deliberately broad-spectrum for that
# reason, rather than aimed at the one statistic that is known to differ.
AUG_ALPHA    = (0.55, 1.60)   # per-clip gain          (was 0.85, 1.15)
AUG_BETA     = (-30.0, 30.0)  # per-clip offset        (was -18, 18)
AUG_GAMMA    = (0.60, 1.65)   # per-clip gamma — nonlinear, so it does what
                              # no alpha/beta pair can
AUG_WB       = (0.86, 1.16)   # per-CHANNEL gain: white balance. Daylight
                              # and tungsten differ here and there was no
                              # augmentation of it at all.
AUG_SAT      = (0.65, 1.35)   # saturation scale
AUG_FRAME_A  = 0.030          # per-FRAME gain wobble, std
AUG_NOISE    = (0.0, 5.0)     # per-frame gaussian noise sigma
AUG_TBLUR_P  = 0.15           # P(3-frame temporal box blur) — simulates the
                              # long exposure that flattens the evening diff
                              # baseline. Kept low and at radius 1: it smears
                              # an onset by ~1 frame, which CTC (order only)
                              # does not care about but the joint model's
                              # dense sigma=1 targets mildly do.

_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)   # BGR weights
# Added to a uint8 clip so channel c indexes its own 256-entry slice of the
# flat LUT below. int16, so the add promotes out of uint8 rather than
# wrapping at 255.
_CHAN_OFFSET = np.array([0, 256, 512], dtype=np.int16)


def _lerp(identity: float, value: float, s: float) -> float:
    """Scale one sampled parameter toward identity by `strength`."""
    return identity + s * (value - identity)


def augment_block_color(block: np.ndarray, rng: random.Random,
                        strength: float = 1.0) -> tuple[np.ndarray, bool]:
    """Colour counterpart of augment_block; also returns whether it
    flipped, since (unlike the onset-only detector) the CLASS target is
    not flip-invariant and the caller must permute it through
    WCA_FLIP_PERM to match.

    `strength` scales every photometric parameter toward its identity
    value, so strength=0 reproduces geometry-only augmentation and 1.0 is
    the full range above. It exists to make the widening A/B-able against
    the existing checkpoints rather than a change that can only be
    accepted or reverted wholesale.
    """
    out = block
    flipped = rng.random() < 0.5
    if flipped:
        out = out[:, :, ::-1, :]

    s = float(strength)

    # -- per-clip appearance, folded into one uint8 LUT per channel.
    # gamma, white balance, gain and offset all commute into a single
    # 256-entry table, so the whole clip costs three numpy takes instead of
    # several full-size float passes.
    gamma = _lerp(1.0, rng.uniform(*AUG_GAMMA), s)
    alpha = _lerp(1.0, rng.uniform(*AUG_ALPHA), s)
    beta = _lerp(0.0, rng.uniform(*AUG_BETA), s)
    ramp = np.arange(256, dtype=np.float32) / 255.0
    sat = _lerp(1.0, rng.uniform(*AUG_SAT), s)
    # The LUT is float32, not uint8: quantising to 256 integer levels here
    # and then immediately going back to float for the per-frame terms
    # would throw away a bit of precision for nothing, and the float table
    # lets saturation fold into the same single pass below.
    # Three per-channel tables laid end to end in ONE flat table, so the
    # whole clip is a single contiguous gather. Indexing each channel
    # separately writes into a stride-3 slice of the output, which numpy
    # does several times slower than one flat take.
    luts = np.empty(3 * 256, dtype=np.float32)
    for c in range(3):
        wb = _lerp(1.0, rng.uniform(*AUG_WB), s)
        luts[c * 256:(c + 1) * 256] = np.clip(
            np.power(ramp, gamma) * (255.0 * alpha * wb) + beta, 0, 255)
    out = luts[out + _CHAN_OFFSET]

    # -- saturation, as a blend toward luma. Cheaper than a round trip
    # through HSV on 97 frames and identical in effect for a linear scale.
    if abs(sat - 1.0) > 1e-3:
        luma = (out * _LUMA).sum(axis=-1, keepdims=True)
        out *= sat
        out += (1.0 - sat) * luma

    # -- per-FRAME terms. Everything above is constant across the clip and
    # therefore invisible (beta) or merely rescaling (alpha) in the diff
    # channel; these are the only terms that change its distribution.
    #
    # standard_normal(dtype=float32) rather than normal(): the float64
    # default allocates and converts 2.7M doubles per clip and dominated
    # the whole augmentation's cost when this was first written.
    if s > 0:
        gen = np.random.default_rng(rng.getrandbits(32))
        out *= (1.0 + AUG_FRAME_A * s
                * gen.standard_normal((out.shape[0], 1, 1, 1),
                                      dtype=np.float32))
        noise = _lerp(0.0, rng.uniform(*AUG_NOISE), s)
        if noise > 1e-3:
            # One noise field shared across B/G/R rather than three
            # independent ones: a third of the random draws, and the
            # channel this has to perturb is LUMA (the diff channel is
            # built from it), which common-mode noise moves correctly.
            out += noise * gen.standard_normal(
                (out.shape[0], out.shape[1], out.shape[2], 1),
                dtype=np.float32)
        if rng.random() < AUG_TBLUR_P * s and out.shape[0] >= 3:
            pad = np.concatenate([out[:1], out, out[-1:]], axis=0)
            out = (pad[:-2] + pad[1:-1] + pad[2:]) * (1.0 / 3.0)

    np.clip(out, 0, 255, out=out)

    sx, sy = rng.randint(-6, 6), rng.randint(-6, 6)
    if sx or sy:
        out = np.roll(out, shift=(sy, sx), axis=(1, 2))

    return out.astype(np.uint8), flipped


class JointClipDataset(Dataset):
    """
    Stage A counterpart of OnsetClipDataset: same fixed-length clip
    sampling, but each sample also carries a (clip_len, 13) class target,
    permuted through WCA_FLIP_PERM whenever the image is mirrored — a
    mirrored R turn looks exactly like a real L' turn, so the label must
    move with the pixels or the model is trained on the wrong class half
    the time (same reasoning as train_move_classifier.py's WCA_FLIP).
    """

    def __init__(self, streams: list[JointSessionStream],
                clip_len: int = CLIP_LEN, stride: int = 24,
                augment: bool = True, seed: int = 0,
                aug_strength: float = 1.0):
        self.streams  = streams
        self.clip_len = clip_len
        self.augment  = augment
        self.aug_strength = aug_strength
        self.rng      = random.Random(seed)
        self.index    = []
        for si, s in enumerate(streams):
            last = len(s) - clip_len
            if last < 0:
                continue
            self.index += [(si, st) for st in range(0, last + 1, stride)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        si, start = self.index[i]
        s = self.streams[si]

        block = s.clip_block(start, self.clip_len)
        y_onset = s.target[start:start + self.clip_len].copy()
        y_class = s.class_target[start:start + self.clip_len].copy()
        y_count = s.count_target[start:start + self.clip_len].copy()

        if self.augment:
            block, flipped = augment_block_color(block, self.rng,
                                                 self.aug_strength)
            if flipped:
                y_class = np.concatenate(
                    [y_class[:, WCA_FLIP_PERM], y_class[:, 12:13]], axis=1)
                # y_count needs no permutation — see build_count_target.

        return (to_tensor_color(block), torch.from_numpy(y_onset),
                torch.from_numpy(y_class), torch.from_numpy(y_count))


class CTCClipDataset(JointClipDataset):
    """
    Same clips and same augmentation as JointClipDataset, but the target is
    the ORDERED SEQUENCE of move classes whose onsets fall inside the clip
    — no per-frame alignment at all (see ctc_decode.py).

    Collated to (x, labels, label_len) with `labels` zero-padded to the
    batch's longest sequence; nn.CTCLoss reads the true lengths and ignores
    the padding.

    Clip boundaries are the one wrinkle. A turn whose onset sits at frame
    95 of a 96-frame clip has most of its pixel evidence outside the clip,
    so the label is there and the evidence is not. That is the standard
    CTC-on-fixed-windows compromise and it is left in deliberately: the
    alternative (dropping edge labels) creates the mirror-image problem of
    evidence with no label, which trains the model to emit blank on a real
    turn — strictly worse, since blank is already 76% of frames.

    Measured on the prepared sessions a 96-frame clip holds 5.1 onsets on
    average (p90 = 10, max = 22), comfortably inside CTC's requirement
    that the input be at least as long as the label sequence.
    """

    def __getitem__(self, i):
        si, start = self.index[i]
        s = self.streams[si]
        end = start + self.clip_len

        block = s.clip_block(start, self.clip_len)
        sel = (s.onset_idx >= start) & (s.onset_idx < end)
        order = np.argsort(s.onset_idx[sel])
        labels = s.onset_class[sel][order].astype(np.int64)

        if self.augment:
            block, flipped = augment_block_color(block, self.rng,
                                                 self.aug_strength)
            if flipped:
                # A mirrored R turn IS an L' turn: relabel in place. Order
                # is untouched — a horizontal flip is not a time reversal.
                labels = WCA_FLIP_PERM[labels]

        return (to_tensor_color(block), torch.from_numpy(labels.copy()),
                len(labels))


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker_init_fn — MANDATORY whenever num_workers > 0 here.

    The clip datasets carry their own `random.Random(seed)` for
    augmentation. Windows spawns workers by pickling the dataset, so every
    worker would otherwise start from the SAME rng state and draw the same
    flips, brightness jitters and shifts as all the others — silently
    cutting augmentation diversity by a factor of num_workers while looking
    like it is working fine.

    torch varies `info.seed` per worker AND per epoch, so this also
    preserves the property the single-process path gets for free: fresh
    augmentation every epoch rather than the same sequence replayed.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = random.Random(info.seed)


def ctc_collate(batch):
    """Pad label sequences to the batch max; return (x, labels, lengths)."""
    xs, labels, lens = zip(*batch)
    maxlen = max(1, max(lens))
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, l in enumerate(labels):
        if len(l):
            padded[i, :len(l)] = l
    return (torch.stack(xs), padded,
            torch.tensor(lens, dtype=torch.long))


def load_joint_streams(session_dirs: list[Path], sigma: float = SIGMA,
                       count_radius: int = COUNT_RADIUS
                       ) -> list[JointSessionStream]:
    """Colour-stream counterpart of load_streams — reads
    detector_stream_color.npz (prepare_data.py --color), not the deployed
    grayscale detector_stream.npz."""
    streams, missing = [], []
    for d in sorted(session_dirs):
        p = d / STREAM_FILE_COLOR
        if p.exists():
            streams.append(JointSessionStream(p, sigma=sigma,
                                              count_radius=count_radius))
        else:
            missing.append(d.name)
    if missing:
        print(f"  {len(missing)} session(s) have no {STREAM_FILE_COLOR} and "
              f"were skipped:")
        for name in missing[:6]:
            print(f"    {name}")
        if len(missing) > 6:
            print(f"    ... and {len(missing) - 6} more")
        print(f"  Run `prepare_data.py --color` on them first.")
    return streams
