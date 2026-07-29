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


class JointArrayStream(ArrayStream):
    """
    Stage A's colour counterpart of ArrayStream: (N, H, W, 3) BGR frames,
    the same onset target as the parent (reused verbatim — see
    build_dense_targets), plus a (N, 13) class_target.
    """

    def __init__(self, frames: np.ndarray, name: str = "live",
                 fps: float = 30.0, onset_idx: np.ndarray | None = None,
                 onset_class: np.ndarray | None = None, sigma: float = SIGMA):
        self.onset_class = np.asarray(onset_class, dtype=int) \
            if onset_class is not None else np.array([], dtype=int)
        super().__init__(frames=frames, name=name, fps=fps,
                         onset_idx=onset_idx, sigma=sigma)
        self.class_target = build_dense_targets(
            self.onset_idx, self.onset_class, len(self.frames), self.sigma,
            self.target)


class JointSessionStream(JointArrayStream):
    """One prepared session loaded from its detector_stream_color.npz."""

    def __init__(self, path: Path, sigma: float = SIGMA):
        data = np.load(path, allow_pickle=True)
        super().__init__(frames=data["frames"], name=str(data["name"]),
                         fps=float(data["fps"]),
                         onset_idx=data["onset_idx"].astype(int),
                         onset_class=data["onset_class"].astype(int),
                         sigma=sigma)
        self.crop_mode = (str(data["crop_mode"]) if "crop_mode" in data
                          else "unknown")


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


def augment_block_color(block: np.ndarray, rng: random.Random
                        ) -> tuple[np.ndarray, bool]:
    """Colour counterpart of augment_block; also returns whether it
    flipped, since (unlike the onset-only detector) the CLASS target is
    not flip-invariant and the caller must permute it through
    WCA_FLIP_PERM to match."""
    out = block
    flipped = rng.random() < 0.5
    if flipped:
        out = out[:, :, ::-1, :]

    alpha = rng.uniform(0.85, 1.15)
    beta = rng.uniform(-18, 18)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255)

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
                augment: bool = True, seed: int = 0):
        self.streams  = streams
        self.clip_len = clip_len
        self.augment  = augment
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

        if self.augment:
            block, flipped = augment_block_color(block, self.rng)
            if flipped:
                y_class = np.concatenate(
                    [y_class[:, WCA_FLIP_PERM], y_class[:, 12:13]], axis=1)

        return (to_tensor_color(block), torch.from_numpy(y_onset),
                torch.from_numpy(y_class))


def load_joint_streams(session_dirs: list[Path], sigma: float = SIGMA
                       ) -> list[JointSessionStream]:
    """Colour-stream counterpart of load_streams — reads
    detector_stream_color.npz (prepare_data.py --color), not the deployed
    grayscale detector_stream.npz."""
    streams, missing = [], []
    for d in sorted(session_dirs):
        p = d / STREAM_FILE_COLOR
        if p.exists():
            streams.append(JointSessionStream(p, sigma=sigma))
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
