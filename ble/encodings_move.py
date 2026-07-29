"""
encodings_move.py

How a move window becomes a tensor.

The move detector hands the classifier an ordered window of frames
([before, mid_00, mid_01, mid_02, after]). Turning that window into
network input is a *representation choice*, and it is the one lever on
the classifier that has not been pulled: the crop fix (2026-07-24) and
the anchor-jitter retrain (2026-07-26) both changed what the window
contains, never how it is encoded.

Every encoder here obeys the same two rules, which is what makes them
comparable:

  1. Static pixels do not survive. In the STRONG form (`cancels_static`)
     a motionless window maps to a constant image — the encoder's
     `neutral` — so nothing at all is left of the cube's appearance. In
     the WEAK form, static pixels survive but only achromatically: they
     are grey, and every coloured pixel in the output is one that moved.
     Static content carries no label information (the cube looks the same
     before and after the turn), so whatever survives is nuisance the
     network has to learn to ignore.
  2. The temporal ORDER of the window survives the encoding. CW and CCW
     of the same face are time-reverses of each other; an encoding that
     is order-free cannot separate them at all, no matter how good the
     network is.

The strong/weak split is not a technicality — it is the one thing this
comparison is actually testing. `rgbtime` (weak) keeps the cube visible,
so the network can see WHERE the moving layer sits on the cube, at the
cost of carrying a large static signal it must suppress. `rgbtime0` and
the chroma pair (strong) hand it motion and nothing else, which is
cleaner but discards the spatial reference frame. Which trade wins is an
empirical question, so both forms are here.

Encoders
--------
diffstack  (legacy, 12ch) — 4 consecutive signed BGR diffs concatenated
    as channels. Order lives in the channel axis. Rule 1 holds exactly
    (no motion → every channel 128). Costs 12 input channels, so
    ResNet's pretrained conv1 must be inflated and its first layer is
    no longer really pretrained.

rgbtime    (3ch, weak) — the five frames are collapsed onto the R, G and
    B channels through overlapping triangular temporal kernels (early →
    R, middle → G, late → B), in luma. A pixel that never changes gets
    the same value in all three channels and renders as grey; a pixel
    that moves picks up a colour fringe whose hue says WHEN it moved,
    and whose spatial arrangement says which way the layer swept. This
    is the classic RGB-time composite. Native 3 channels, so conv1 stays
    genuinely pretrained.

rgbtime0   (3ch, strong) — rgbtime with the temporal MEAN removed and
    the result re-centred on 128. Identical construction, except that
    what survives is each channel's DEVIATION from the window average, so
    a static pixel lands on exactly 128 and vanishes. This is the
    strict-static-cancellation version of the same idea, and the pair
    (rgbtime vs rgbtime0) isolates what the cube's visible appearance is
    worth on its own.

chroma     (3ch) — motion energy, hue-coded by time. Per pixel: total
    motion across the window sets brightness, and the temporal centre of
    mass of that motion picks the hue from a sweep (red = start of the
    window → blue = end). Static pixels are black, not grey — the
    background is not just uninformative, it is gone. A pixel that moved
    at one instant is saturated; a pixel that moved throughout washes
    out to white, which is itself a meaningful distinction (edge of the
    swept region vs. its interior).

chroma8    (3ch) — chroma, but the sign of each diff is kept: a
    brightening pixel and a darkening one at the same instant get
    complementary hues instead of being merged by the absolute value.
    Eight phases instead of four. Costs nothing extra at the tensor, and
    tests whether the polarity chroma throws away was load-bearing.

The encoding a checkpoint was trained under is recorded IN the
checkpoint and re-applied at inference, for the same reason `crop_regime`
is (see train_move_classifier.py): a model scored on a representation it
was not trained on does not degrade gracefully, it reads as a broken
model, and that ambiguity has already cost this project a day once.
"""

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

try:
    import torch
except ImportError:                                     # pragma: no cover
    torch = None

IMG_SIZE = 224

# Chroma: energy below this (in 8-bit luma units, summed over the window)
# is treated as "no motion happened here" rather than being normalised up
# to full brightness. Without a floor, per-sample normalisation turns a
# window containing only sensor noise — which is exactly what a phantom
# onset detection hands us — into a vivid, confident-looking input.
CHROMA_ENERGY_FLOOR = 24.0
CHROMA_PCT          = 99.5   # percentile used as the per-sample scale
CHROMA_GAMMA        = 0.7    # <1 lifts faint motion into visible range

# rgbtime0: deviations from the window mean are a few tens of luma levels
# at most. Without a gain the encoding lives in ~10% of the uint8 range and
# most of its precision is thrown away by the round-trip to bytes.
RGBTIME0_GAIN = 3.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resized(frames_bgr: list[np.ndarray], size: int) -> list[np.ndarray]:
    return [cv2.resize(f, (size, size)) for f in frames_bgr]


def _luma(frames_bgr: list[np.ndarray], size: int) -> list[np.ndarray]:
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for f in _resized(frames_bgr, size)]


def hue_sweep(k: int, span: float = 270.0) -> np.ndarray:
    """
    k evenly spaced, maximally distinguishable RGB colours (0-1 floats)
    covering `span` degrees of hue. Used as the time axis: index 0 is the
    start of the move window, index k-1 the end.

    Stops at 270° rather than wrapping the full circle on purpose — a full
    wrap would give the first and last instant of the window nearly the
    same hue, which is the one confusion this encoding exists to avoid
    (it is exactly the CW/CCW distinction).
    """
    hsv = np.zeros((1, k, 3), dtype=np.uint8)
    hsv[0, :, 0] = (np.linspace(0, span, k) / 2).astype(np.uint8)  # OpenCV: 0-179
    hsv[0, :, 1] = 255
    hsv[0, :, 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0].astype(np.float32) / 255.0
    return rgb                                          # (k, 3) in R,G,B


def triangular_basis(n: int, out: int = 3) -> np.ndarray:
    """
    (out, n) row-normalised temporal weights: `out` overlapping triangular
    kernels whose centres span the window evenly.

    Overlapping rather than disjoint bins because a hard bin boundary makes
    the encoding jump discontinuously when the window anchor shifts by a
    frame — and a frame of anchor slop is the norm, not the exception
    (measured: 52% exact on live takes). Overlap makes the representation
    move smoothly with the anchor instead.
    """
    if n < out:
        raise ValueError(f"need at least {out} frames, got {n}")
    t = np.arange(n, dtype=np.float32)
    centres = np.linspace(0, n - 1, out, dtype=np.float32)
    spread = (n - 1) / (out - 1)
    w = np.maximum(0.0, 1.0 - np.abs(t[None, :] - centres[:, None]) / spread)
    return w / w.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Encoders — window of ordered BGR frames -> HWC uint8
# ---------------------------------------------------------------------------

def encode_diffstack(frames_bgr: list[np.ndarray],
                     size: int = IMG_SIZE) -> np.ndarray:
    """
    N ordered BGR frames -> HWC uint8 stack of N-1 consecutive signed
    diffs (RGB each), 128 = no change. Channel order = temporal order.

    (This is the historical `build_diff_stack`, moved here unchanged so
    every encoding lives in one place.)
    """
    resized = [f.astype(np.int16) for f in _resized(frames_bgr, size)]
    diffs = []
    for a, b in zip(resized, resized[1:]):
        d = np.clip(b - a + 128, 0, 255).astype(np.uint8)
        diffs.append(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))
    return np.concatenate(diffs, axis=2)


def encode_rgbtime(frames_bgr: list[np.ndarray],
                   size: int = IMG_SIZE) -> np.ndarray:
    """
    N ordered frames -> one 3-channel RGB image, early/middle/late luma in
    R/G/B via overlapping triangular temporal kernels.

    Unchanged pixel -> R = G = B -> grey, whatever colour the sticker was.
    That is deliberate beyond the static-cancellation rule: the label is a
    camera-relative layer (U/R/F...), so sticker colour is a shortcut
    feature that only holds for the recorder's habitual grip. Luma keeps
    the geometry and drops the shortcut.
    """
    return _rgbtime(frames_bgr, size, centre=False)


def encode_rgbtime0(frames_bgr: list[np.ndarray],
                    size: int = IMG_SIZE) -> np.ndarray:
    """
    rgbtime with the temporal mean removed: each channel becomes its
    deviation from the window average, re-centred on 128, so an unchanged
    pixel lands on exactly 128 and the cube's static appearance is gone.

    The deviations are small (a turn moves luma by tens of levels, not
    hundreds), so they are scaled up by RGBTIME0_GAIN before clipping —
    otherwise the whole image sits within a few levels of neutral and the
    encoding is quantised down to noise by the uint8 round-trip.
    """
    return _rgbtime(frames_bgr, size, centre=True)


def _rgbtime(frames_bgr: list[np.ndarray], size: int, centre: bool
             ) -> np.ndarray:
    g = _luma(frames_bgr, size)
    w = triangular_basis(len(g), 3)
    stack = np.stack(g, axis=0)                          # (n, H, W)
    out = np.tensordot(w, stack, axes=(1, 0))            # (3, H, W)
    if centre:
        out = (out - out.mean(axis=0, keepdims=True)) * RGBTIME0_GAIN + 128.0
    return np.clip(out, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def _chroma(frames_bgr: list[np.ndarray], size: int, signed: bool
            ) -> np.ndarray:
    g = _luma(frames_bgr, size)
    diffs = [b - a for a, b in zip(g, g[1:])]

    if signed:
        # Positive and negative excursions become separate phases, so a
        # brightening and a darkening pixel at the same instant no longer
        # land on the same hue.
        energy = []
        for d in diffs:
            energy.append(np.maximum(d, 0))
            energy.append(np.maximum(-d, 0))
    else:
        energy = [np.abs(d) for d in diffs]

    # 3x3 blur first: at this scale single-pixel sensor noise is the main
    # thing competing with real motion for the per-sample normaliser.
    e = np.stack([cv2.GaussianBlur(x, (3, 3), 0) for x in energy], axis=0)

    total = e.sum(axis=0)                                # (H, W)
    prof  = e / np.maximum(total, 1e-6)                  # temporal profile
    colours = hue_sweep(e.shape[0])                      # (k, 3)
    hue = np.tensordot(prof, colours, axes=(0, 0))       # (H, W, 3)

    scale = max(float(np.percentile(total, CHROMA_PCT)), CHROMA_ENERGY_FLOOR)
    inten = np.clip(total / scale, 0.0, 1.0) ** CHROMA_GAMMA

    out = hue * inten[..., None]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def encode_chroma(frames_bgr: list[np.ndarray],
                  size: int = IMG_SIZE) -> np.ndarray:
    """Motion energy as brightness, time-of-motion as hue (4 phases)."""
    return _chroma(frames_bgr, size, signed=False)


def encode_chroma8(frames_bgr: list[np.ndarray],
                   size: int = IMG_SIZE) -> np.ndarray:
    """chroma, with the sign of each diff kept as its own phase (8)."""
    return _chroma(frames_bgr, size, signed=True)


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------
#
# Why flow, and why the RESIDUAL specifically.
#
# The diff- and chroma-style encodings all answer "which pixels changed".
# On this data that question has a boring answer: during a solve the whole
# cube and both hands are moving, so essentially every pixel on the cube
# changed, and the renders in viz_out/ show exactly that — they light up
# everywhere. The 2026-07-27 sweep measured what that costs: four such
# encodings all landed inside the diffstack seed envelope.
#
# The quantity that actually names the move is different. A turn rotates
# ONE layer about a face axis; the other two layers are rigid with the cube
# body. So in the cube's own frame the motion field is zero except on the
# turning layer. What the camera sees is that field PLUS whatever the hands
# are doing to the cube as a whole (translation, rotation, scale as it
# tilts toward or away).
#
# Fit that global component and subtract it, and what is left is the
# layer's motion relative to the cube: WHERE the residual lives says which
# layer turned, and its SIGN says which way. That is the hypothesis, and
# `flow` (uncompensated) vs `flowres` (compensated) is the ablation that
# tests it — they differ in exactly one step.
#
# Farneback rather than RAFT: it runs in the dataloader worker on CPU
# (~14ms per 224x224 pair), which keeps --anchor-jitter working. Jitter
# rebuilds a different window every epoch, so a precomputed flow cache —
# the only sane way to use a GPU flow net here — would be invalid for the
# sessions that have raw frames, i.e. most of them. If flowres shows
# promise and flow quality looks like the ceiling, RAFT + dropping jitter
# is the upgrade path, in that order.

# Farneback parameters, imported in spirit from flow_direction.py, which
# tuned them for this capture: levels=4 because 'before' to 'mid_00' is
# ~180ms at 30fps and the displacement over that gap is large.
FB_PARAMS = dict(pyr_scale=0.5, levels=4, winsize=21,
                 iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

# Below this (pixels, 99th-percentile flow magnitude over the window) a
# window is treated as motionless instead of having its noise normalised up
# to full scale. Same guard, same reason, as CHROMA_ENERGY_FLOOR.
# Measured 2026-07-27 on 40 real cropped move windows: p99 |flow| is
# 7.3-19.7 px for a real turn, and 0.007 px for an identical repeated
# frame. A 1 px floor therefore never clips a real move and never lets a
# near-still window be normalised up into a vivid, confident-looking input.
FLOW_FLOOR = 1.0
FLOW_PCT   = 99.0

# Robust global-motion fit
GLOBAL_ITERS  = 3      # iteratively reweighted least squares passes
GLOBAL_TUKEY  = 2.0    # Tukey biweight cutoff, in units of residual MAD
GLOBAL_STRIDE = 4      # subsample the flow field this much before fitting
GLOBAL_SIGMA  = 0.35   # centre-weight width, as a fraction of image size


def dense_flow(frames_bgr: list[np.ndarray], size: int) -> np.ndarray:
    """(K, H, W, 2) Farneback flow for each consecutive frame pair."""
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in _resized(frames_bgr, size)]
    return np.stack([cv2.calcOpticalFlowFarneback(a, b, None, **FB_PARAMS)
                     for a, b in zip(g, g[1:])], axis=0)


def _centre_weights(h: int, w: int) -> np.ndarray:
    """
    Gaussian weight favouring the middle of the crop.

    Not a heuristic bolted on: crop_utils builds the crop square and
    centred on the detected cube with a 12% margin, so the cube reliably
    occupies the middle and the hands reach in from the edges. Without
    this, a two-handed grip can put more moving pixels on skin than on
    cube and the "global" fit locks onto the hands instead.
    """
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    s2 = 2.0 * (GLOBAL_SIGMA * max(h, w)) ** 2
    return np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / s2))


def fit_global_flow(flow: np.ndarray) -> np.ndarray:
    """
    Fit the affine motion field that best explains `flow` (H, W, 2), and
    return it evaluated on the full grid, same shape.

    Affine rather than pure translation because the cube does not merely
    slide in the hand — it tilts and rotates between frames, which reads
    as rotation plus scale plus shear in the image. A translation-only
    model would leave that in the residual and drown the layer signal.

    The fit is iteratively reweighted (Tukey biweight on residual
    magnitude, centre-weighted to start) because the turning layer is up
    to a third of the cube's pixels and is precisely the thing that must
    NOT influence the fit. Plain least squares would absorb part of the
    layer's motion into the "global" term and subtract the signal away.
    """
    h, w = flow.shape[:2]
    s = GLOBAL_STRIDE
    sub = flow[::s, ::s]
    hh, ww = sub.shape[:2]

    y, x = np.mgrid[0:hh, 0:ww].astype(np.float32)
    x *= s
    y *= s
    A = np.stack([x.ravel(), y.ravel(), np.ones(hh * ww, np.float32)], axis=1)
    u = sub[:, :, 0].ravel()
    v = sub[:, :, 1].ravel()

    wt = _centre_weights(h, w)[::s, ::s].ravel().astype(np.float32)

    coef = np.zeros((3, 2), np.float32)
    for _ in range(GLOBAL_ITERS):
        sw = np.sqrt(wt)[:, None]
        try:
            coef, *_ = np.linalg.lstsq(A * sw,
                                       np.stack([u, v], axis=1) * sw,
                                       rcond=None)
        except np.linalg.LinAlgError:                    # pragma: no cover
            break
        pred = A @ coef
        res = np.hypot(u - pred[:, 0], v - pred[:, 1])
        mad = float(np.median(res)) + 1e-6
        r = res / (GLOBAL_TUKEY * mad)
        wt = np.where(r < 1.0, (1.0 - r ** 2) ** 2, 0.0).astype(np.float32)
        wt *= _centre_weights(h, w)[::s, ::s].ravel()
        if wt.sum() < 16:            # degenerate — everything called outlier
            break

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    full = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w, np.float32)],
                    axis=1) @ coef
    return full.reshape(h, w, 2)


def _flow_stack(frames_bgr: list[np.ndarray], size: int, compensate: bool
                ) -> np.ndarray:
    flow = dense_flow(frames_bgr, size)
    if compensate:
        flow = np.stack([f - fit_global_flow(f) for f in flow], axis=0)
    scale = max(float(np.percentile(np.abs(flow), FLOW_PCT)), FLOW_FLOOR)
    q = np.clip(flow / scale, -1.0, 1.0) * 127.0 + 128.0
    # (K, H, W, 2) -> HWC with channels in temporal order u0,v0,u1,v1,...
    k, h, w, _ = q.shape
    return q.astype(np.uint8).transpose(1, 2, 0, 3).reshape(h, w, k * 2)


def encode_flow(frames_bgr: list[np.ndarray],
                size: int = IMG_SIZE) -> np.ndarray:
    """Raw dense flow per frame pair, 2 channels each. The ablation."""
    return _flow_stack(frames_bgr, size, compensate=False)


def encode_flowres(frames_bgr: list[np.ndarray],
                   size: int = IMG_SIZE) -> np.ndarray:
    """Dense flow with the global (whole-cube) motion fitted out."""
    return _flow_stack(frames_bgr, size, compensate=True)


def flow_to_wheel(uv: np.ndarray, scale: float | None = None) -> np.ndarray:
    """
    (H, W, 2) flow -> RGB uint8 colour wheel: hue = direction, value =
    magnitude. Display convention, and the tensor for `flowwheel`.
    """
    mag, ang = cv2.cartToPolar(uv[:, :, 0].astype(np.float32),
                               uv[:, :, 1].astype(np.float32))
    if scale is None:
        scale = max(float(np.percentile(mag, FLOW_PCT)), FLOW_FLOOR)
    hsv = np.zeros(uv.shape[:2] + (3,), np.uint8)
    hsv[:, :, 0] = (ang * 90 / np.pi).astype(np.uint8)   # OpenCV hue 0-179
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag / scale * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def encode_flowwheel(frames_bgr: list[np.ndarray],
                     size: int = IMG_SIZE) -> np.ndarray:
    """
    Residual flow summed over the window (magnitude-weighted) and rendered
    as one colour wheel: 3 channels, conv1 pretrained as-is.

    Included because the 07-27 sweep found 3 channels tying 12, so the
    cheap version deserves a fair shot. It is NOT strictly better-behaved
    input than `flowres`: hue wraps at 360 degrees, so two nearly-identical
    directions can sit at opposite ends of the channel range, and a
    convolution has to learn around that discontinuity. Kept honest by
    running both.
    """
    flow = dense_flow(frames_bgr, size)
    flow = np.stack([f - fit_global_flow(f) for f in flow], axis=0)
    # Mean of the vector field, not of the angles: averaging vectors keeps
    # temporal order (a turn and its time-reverse average to opposite
    # vectors) and weights by how much actually moved at each instant.
    #
    # Mean rather than SUM specifically so the result stays on the same
    # per-pair scale that FLOW_FLOOR is calibrated in. Summing put the
    # field 4x above that floor, which meant a near-still window — a
    # phantom onset — had its noise normalised up into a vivid input
    # instead of being floored to black. The self-test caught it.
    return flow_to_wheel(flow.mean(axis=0))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def flip_fixup_uv(img: np.ndarray) -> np.ndarray:
    """
    After a horizontal mirror, negate the horizontal flow component.

    Mirroring maps a velocity (u, v) to (-u, v). Without this the flipped
    sample shows mirrored geometry carrying unmirrored motion — a physically
    impossible field, and worse, the flip is what generates the L/R class
    balance, so the error would land squarely on the pair of classes the
    augmentation exists to serve.
    """
    out = img.copy()
    out[:, :, 0::2] = 255 - out[:, :, 0::2]
    return out


def flip_fixup_wheel(img: np.ndarray) -> np.ndarray:
    """
    Same correction for the colour-wheel encoding, where direction lives in
    hue: theta -> pi - theta, i.e. hue -> (90 - hue) mod 180 in OpenCV units.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (90 - hsv[:, :, 0].astype(np.int16)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


@dataclass(frozen=True)
class Encoding:
    name: str
    fn: Callable[[list[np.ndarray], int], np.ndarray]
    channels: int
    frames: int      # window frames the encoder consumes
    neutral: int     # pixel value of "nothing moved" — border fill, fallbacks
    photometric: bool     # True if brightness/contrast jitter is meaningful
    cancels_static: bool  # True = motionless window -> constant `neutral`
    blurb: str
    # How far from `neutral` a motionless window may land, in 0-255 units.
    # Nonzero only for the flow encodings: Farneback is an iterative
    # estimator and returns a few hundredths of a pixel of boundary noise
    # on identical frames rather than exact zero. This is a tolerance on
    # ESTIMATOR noise, not a licence for the encoding to carry signal.
    static_tol: int = 1
    # Applied after a horizontal mirror. None for encodings whose pixels are
    # intensities (mirroring them is already correct); required for the flow
    # encodings, whose pixels are signed vectors.
    flip_fixup: Callable[[np.ndarray], np.ndarray] | None = None


ENCODINGS: dict[str, Encoding] = {
    "diffstack": Encoding(
        "diffstack", encode_diffstack, 12, 5, 128, True, True,
        "4 signed BGR diffs as 12 channels (legacy)"),
    "diffstack1": Encoding(
        "diffstack1", encode_diffstack, 3, 2, 128, True, True,
        "single signed BGR diff (ablation)"),
    "rgbtime": Encoding(
        "rgbtime", encode_rgbtime, 3, 5, 128, True, False,
        "early/mid/late luma into R/G/B; statics grey but visible"),
    "rgbtime0": Encoding(
        "rgbtime0", encode_rgbtime0, 3, 5, 128, True, True,
        "rgbtime, temporal mean removed; statics vanish"),
    "chroma": Encoding(
        "chroma", encode_chroma, 3, 5, 0, False, True,
        "motion energy x time-coded hue (4 phases)"),
    "chroma8": Encoding(
        "chroma8", encode_chroma8, 3, 5, 0, False, True,
        "motion energy x time-coded hue, signed (8 phases)"),
    "flow": Encoding(
        "flow", encode_flow, 8, 5, 128, False, True,
        "dense Farneback flow, 2ch per pair (uncompensated)",
        flip_fixup=flip_fixup_uv, static_tol=4),
    "flowres": Encoding(
        "flowres", encode_flowres, 8, 5, 128, False, True,
        "dense flow minus fitted global cube motion",
        flip_fixup=flip_fixup_uv, static_tol=4),
    "flowwheel": Encoding(
        "flowwheel", encode_flowwheel, 3, 5, 0, False, True,
        "residual flow summed over window, as a colour wheel",
        flip_fixup=flip_fixup_wheel, static_tol=6),
}

DEFAULT_ENCODING = "diffstack"


def get(name: str | None) -> Encoding:
    """Look up an encoding; None means a checkpoint from before this existed."""
    if name is None:
        return ENCODINGS[DEFAULT_ENCODING]
    if name not in ENCODINGS:
        raise KeyError(f"unknown encoding {name!r}; "
                       f"have {sorted(ENCODINGS)}")
    return ENCODINGS[name]


def encode(name: str, frames_bgr: list[np.ndarray],
           size: int = IMG_SIZE) -> np.ndarray:
    return get(name).fn(frames_bgr, size)


def to_tensor(img: np.ndarray) -> "torch.Tensor":
    """
    HWC uint8 -> CHW float32 normalised with the ImageNet stats tiled over
    the channel axis.

    Channel counts that are not a multiple of 3 (the flow encodings are
    2 per frame pair) tile and truncate rather than being rejected. The
    stats are only there to put activations in the range the pretrained
    weights expect; which of the three a given flow channel lands on is
    arbitrary either way.
    """
    n = img.shape[2]
    reps = -(-n // 3)                                    # ceil
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
    t /= 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).repeat(reps)[:n].view(-1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).repeat(reps)[:n].view(-1, 1, 1)
    return (t - mean) / std


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def preview(name: str, frames_bgr: list[np.ndarray],
            size: int = IMG_SIZE) -> np.ndarray:
    """
    A single BGR image showing what the encoder produces, for humans.

    3-channel encodings are shown as-is. diffstack has no single-image
    form, so its channel groups are tiled in temporal order (reading
    order, left to right then down) into a square of the same size — the
    point of the comparison is what one glance at each encoding shows,
    and a 4x-wide strip would not be a fair one.
    """
    enc = get(name)
    img = enc.fn(frames_bgr, size)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if name in ("flow", "flowres"):
        # (u, v) pairs mean nothing shown as grey planes — render each
        # pair the way flow is always read, as a direction wheel.
        uv = img.astype(np.float32) - 128.0
        tiles = [cv2.cvtColor(flow_to_wheel(uv[:, :, 2*i:2*i+2], scale=127.0),
                              cv2.COLOR_RGB2BGR)
                 for i in range(img.shape[2] // 2)]
        img = np.concatenate([cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
                              for t in tiles], axis=2)

    k = img.shape[2] // 3
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    cell = size // cols
    out = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i in range(k):
        tile = cv2.resize(cv2.cvtColor(img[:, :, 3*i:3*i+3],
                                       cv2.COLOR_RGB2BGR), (cell, cell))
        r, c = divmod(i, cols)
        out[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = tile
    return cv2.resize(out, (size, size))


def time_legend(k: int = 4, width: int = 256, height: int = 28,
                span: float = 270.0) -> np.ndarray:
    """A BGR strip of the hue sweep — the colour-to-time key for chroma."""
    colours = hue_sweep(k, span)
    xs = np.linspace(0, k - 1, width)
    lo = np.floor(xs).astype(int)
    hi = np.minimum(lo + 1, k - 1)
    f  = (xs - lo)[:, None]
    row = colours[lo] * (1 - f) + colours[hi] * f        # (width, 3) RGB
    strip = np.repeat(row[None, :, :], height, axis=0)
    return cv2.cvtColor((strip * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Self-test — the two invariants every encoder is supposed to hold
# ---------------------------------------------------------------------------

def _selftest() -> int:
    rng = np.random.default_rng(0)
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
        if not cond:
            fails += 1

    # Flat colour blocks with hard edges, not noise: the still-scene test
    # is partly a test of a dense-flow ESTIMATOR, and noise is a
    # pathological input for one (aperture problem everywhere), so a noise
    # image measures OpenCV's failure mode rather than this module's.
    # Measured 2026-07-27, spurious flow on five identical frames:
    # white noise 0.024 px, blurred noise 0.047 px, blocks 0.008 px — and
    # real repeated cube frames 0.007 px. The blocks image is the one that
    # reproduces reality, so the tolerances below stay tight enough to
    # mean something.
    _s = np.zeros((96, 96, 3), np.uint8)
    for _i in range(3):
        for _j in range(3):
            _s[_i*32:(_i+1)*32, _j*32:(_j+1)*32] = rng.integers(40, 240, 3)
    _s = cv2.GaussianBlur(_s, (3, 3), 0)
    still = [_s.copy() for _ in range(5)]

    # A synthetic "turn": a bright bar sweeping left to right, and its
    # time-reverse. Direction has to survive the encoding or the encoder
    # cannot express CW vs CCW at all.
    def sweep(reverse=False):
        base = rng.integers(60, 90, (96, 96, 3), dtype=np.uint8)
        out = []
        for i in range(5):
            f = base.copy()
            x = 10 + i * 15
            f[30:60, x:x + 12] = 240
            out.append(f)
        return out[::-1] if reverse else out

    fwd, rev = sweep(), sweep(reverse=True)

    for name, enc in ENCODINGS.items():
        img = enc.fn(still[:enc.frames], 96)
        check(img.shape == (96, 96, enc.channels),
              f"{name}: shape {img.shape} == (96, 96, {enc.channels})")
        # Rule 1, strong form: a motionless window is constant at neutral.
        # Weak form: it survives, but only as grey (R = G = B everywhere).
        if enc.cancels_static:
            flat = np.abs(img.astype(int) - enc.neutral).max()
            check(flat <= enc.static_tol,
                  f"{name}: static window -> constant neutral "
                  f"({enc.neutral}), max deviation {flat} "
                  f"(tol {enc.static_tol})")
        else:
            chroma_left = img.astype(int).max(axis=2) - \
                          img.astype(int).min(axis=2)
            check(chroma_left.max() <= 1,
                  f"{name}: static window -> achromatic, max channel "
                  f"spread {chroma_left.max()}")
        # Rule 2: a motion and its time-reverse do not encode alike.
        a = enc.fn(fwd[:enc.frames], 96).astype(float)
        b = enc.fn(rev[:enc.frames], 96).astype(float)
        d = np.abs(a - b).mean()
        check(d > 1.0, f"{name}: forward vs reversed sweep differ "
                       f"(mean abs diff = {d:.1f})")

    # Flip fixup: mirroring the INPUT frames and encoding must agree with
    # encoding then mirroring + fixup. If it does not, the flip augmentation
    # is manufacturing impossible motion fields (and it is the L/R
    # augmentation specifically, so the damage is targeted).
    for name in ("flow", "flowres", "flowwheel"):
        enc = ENCODINGS[name]
        mirrored_in = enc.fn([f[:, ::-1] for f in fwd[:enc.frames]], 96)
        fixed = enc.flip_fixup(np.fliplr(enc.fn(fwd[:enc.frames], 96)).copy())
        d = np.abs(mirrored_in.astype(float) - fixed.astype(float)).mean()
        check(d < 12.0, f"{name}: flip fixup matches mirroring the input "
                        f"(mean abs diff = {d:.1f})")

    # Global-motion compensation: a pure translation of the whole scene is
    # entirely global, so the residual must be ~zero. This is the property
    # flowres exists for; if it fails, flowres is just a noisier flow.
    base = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    pan = [np.roll(base, 4 * i, axis=1) for i in range(5)]
    raw = ENCODINGS["flow"].fn(pan, 96).astype(float) - 128.0
    res = ENCODINGS["flowres"].fn(pan, 96).astype(float) - 128.0
    inner = slice(16, 80)   # ignore the wrap seam np.roll leaves at the edge
    rawm = np.abs(raw[inner, inner]).mean()
    resm = np.abs(res[inner, inner]).mean()
    check(resm < 0.5 * rawm,
          f"flowres: global pan is removed (residual {resm:.1f} vs raw "
          f"{rawm:.1f})")

    w = triangular_basis(5, 3)
    check(np.allclose(w.sum(axis=1), 1.0), "triangular_basis rows sum to 1")
    check(w[0, 0] > w[0, -1] and w[2, -1] > w[2, 0],
          "triangular_basis is ordered early -> late")

    print(f"\n  {'ALL PASS' if not fails else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
