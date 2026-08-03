"""
flow_direction.py

Zero-training CW/CCW direction head based on dense optical flow.

Motivation: the ResNet move classifier sees a single signed diff image,
which carries almost no temporal-order information — CW and CCW of the
same face are time-reversals of each other, so an order-free input cannot
separate them reliably. Direction, however, is literally the sign of the
motion field: a face turn produces rotational flow (signed curl) and a
layer turn produces a signed translation of the moving strip. Both are
directly measurable with classical optical flow, no training needed.

What this module does
---------------------
`compute_features(frames)` takes the ordered move-window frames
(before, mid_00, mid_01, mid_02, after — any >= 2 of them, in order),
finds the motion region, runs Farneback dense flow over consecutive
pairs, and returns a 21-dim signed motion descriptor:

    [ curl,  mean_fx,  mean_fy,  9 x (cell_fx, cell_fy) on a 3x3 grid ]

  * curl     > 0  means visually-clockwise rotation about the motion
               centroid (image coords, y down).
  * mean_fx/fy    magnitude-weighted mean translation.
  * grid cells    localize the moving strip (e.g. an R turn is vertical
               flow concentrated in the right column of cells).

All components are signed, so CW and CCW of the same face produce
(approximately) negated descriptors — exactly the invariant the diff
image lacks.

How direction is decided
------------------------
Which *pattern* (rotation vs. strip translation, and its polarity) maps
to "CW" depends on which face is turning and where it sits relative to
the camera. Rather than hand-code cube pose geometry, `DirectionModel`
learns one signed axis per face from labeled sessions: nearest-centroid
in z-scored feature space (w_face = mean(CW) - mean(CCW), ~6 numbers of
"training" — effectively calibration, not learning). BLE-labeled
sessions give this for free.

Usage
-----
  # Evaluate direction separability (leave-one-session-out):
  python flow_direction.py --sessions training_data/solve_*/

  # Fit on all sessions and save a model for live use:
  python flow_direction.py --sessions training_data/solve_*/ --fit-out direction_model.json

  # In another script:
  from flow_direction import compute_features, DirectionModel
  model = DirectionModel.load("direction_model.json")
  is_cw = model.predict(face_id, compute_features(frames))
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_ORDER = ["before", "mid_00", "mid_01", "mid_02", "after"]

FACE_NAMES = ["blue", "green", "white", "yellow", "red", "orange"]

FLOW_SIZE   = 256     # motion ROI is resized to this before flow
GRID        = 3       # grid cells per side for localized flow features
N_FEATURES  = 3 + 2 * GRID * GRID

# Farneback parameters — pyramid levels sized for the large inter-frame
# displacement (~180ms between 'before' and 'mid_00' at ~30fps)
FB_PARAMS = dict(pyr_scale=0.5, levels=4, winsize=21,
                 iterations=3, poly_n=5, poly_sigma=1.1, flags=0)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _motion_roi(grays: list[np.ndarray], margin: float = 0.12) -> tuple:
    """
    Bounding box (x0, y0, x1, y1) of the region that changed across the
    window, from accumulated absolute frame diffs. Falls back to the full
    frame if motion is too weak/diffuse to localize.
    """
    h, w = grays[0].shape
    acc = np.zeros((h, w), np.float32)
    for a, b in zip(grays, grays[1:]):
        acc += cv2.absdiff(a, b).astype(np.float32)

    acc = cv2.GaussianBlur(acc, (15, 15), 0)
    peak = float(acc.max())
    if peak < 10.0:
        return (0, 0, w, h)

    mask = acc > max(15.0, 0.25 * peak)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return (0, 0, w, h)

    mx = int((xs.max() - xs.min()) * margin) + 4
    my = int((ys.max() - ys.min()) * margin) + 4
    return (max(0, xs.min() - mx), max(0, ys.min() - my),
            min(w, xs.max() + mx), min(h, ys.max() + my))


def _flow_pairs_farneback(crops: list[np.ndarray]):
    """Farneback flow (H, W, 2) for each consecutive pair of GRAYSCALE crops."""
    for a, b in zip(crops, crops[1:]):
        yield cv2.calcOpticalFlowFarneback(a, b, None, **FB_PARAMS)


# Lazy-loaded: flow_direction.py's default path (Farneback) has zero torch
# dependency, and G1 (MODEL_REWORK_PLAN.md) only needs RAFT loaded when
# --flow-backend raft is actually requested.
_RAFT_STATE: dict = {}


def _load_raft():
    if not _RAFT_STATE:
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.DEFAULT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _RAFT_STATE["model"] = raft_small(weights=weights).to(device).eval()
        _RAFT_STATE["device"] = device
        _RAFT_STATE["transforms"] = weights.transforms()
        _RAFT_STATE["torch"] = torch
    return _RAFT_STATE["model"], _RAFT_STATE["device"], \
        _RAFT_STATE["transforms"], _RAFT_STATE["torch"]


def _flow_pairs_raft(crops: list[np.ndarray]):
    """
    RAFT flow (H, W, 2) for each consecutive pair of BGR crops (RAFT wants
    colour, unlike Farneback's grayscale input — see the module docstring's
    G1 note on why this is worth testing at all: cube turns are exactly the
    fast-blurred-small-object regime classical pyramidal flow is weak at).
    """
    model, device, transforms, torch = _load_raft()

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        for a, b in zip(crops, crops[1:]):
            t1, t2 = transforms(to_tensor(a), to_tensor(b))
            flow = model(t1.to(device), t2.to(device))[-1]
            yield flow[0].permute(1, 2, 0).cpu().numpy()


FLOW_BACKENDS = {"farneback": _flow_pairs_farneback, "raft": _flow_pairs_raft}


def compute_features(frames_bgr: list[np.ndarray],
                     flow_backend: str = "farneback") -> np.ndarray | None:
    """
    21-dim signed motion descriptor for an ordered move window.

    frames_bgr : >= 2 frames in temporal order (missing window slots
                 simply omitted). Returns None if flow is degenerate
                 (fewer than 2 usable frames, or no measurable motion).
    flow_backend : "farneback" (default, zero extra deps, byte-identical
                 to the original implementation) or "raft" (G1,
                 MODEL_REWORK_PLAN.md — torchvision's pretrained RAFT-small,
                 loaded lazily on first use).
    """
    frames_bgr = [f for f in frames_bgr if f is not None]
    if len(frames_bgr) < 2:
        return None

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    x0, y0, x1, y1 = _motion_roi(grays)
    if flow_backend == "raft":
        crops = [cv2.resize(f[y0:y1, x0:x1], (FLOW_SIZE, FLOW_SIZE))
                 for f in frames_bgr]
    else:
        crops = [cv2.resize(g[y0:y1, x0:x1], (FLOW_SIZE, FLOW_SIZE))
                 for g in grays]

    # Accumulate magnitude-weighted flow statistics across all pairs
    sum_w     = 0.0
    sum_fx    = 0.0
    sum_fy    = 0.0
    sum_curl  = 0.0
    cell_fx   = np.zeros((GRID, GRID), np.float64)
    cell_fy   = np.zeros((GRID, GRID), np.float64)
    cell_w    = np.zeros((GRID, GRID), np.float64)

    yy, xx = np.mgrid[0:FLOW_SIZE, 0:FLOW_SIZE].astype(np.float32)
    cell   = FLOW_SIZE // GRID

    for flow in FLOW_BACKENDS[flow_backend](crops):
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)

        # Suppress static background / noise: weight by magnitude, but
        # only where it exceeds a fraction of this pair's peak motion.
        thr = 0.15 * float(mag.max())
        w = np.where(mag > max(thr, 0.3), mag, 0.0).astype(np.float64)
        tw = float(w.sum())
        if tw < 1.0:
            continue

        # Motion centroid for this pair
        cx = float((w * xx).sum() / tw)
        cy = float((w * yy).sum() / tw)

        # curl: tangential speed about the centroid.
        # cross_z = rx*fy - ry*fx; > 0 is visually clockwise (y down).
        rx, ry = xx - cx, yy - cy
        r = np.sqrt(rx * rx + ry * ry) + 1e-6
        curl = (rx * fy - ry * fx) / r

        sum_w    += tw
        sum_fx   += float((w * fx).sum())
        sum_fy   += float((w * fy).sum())
        sum_curl += float((w * curl).sum())

        for gy in range(GRID):
            for gx in range(GRID):
                sl = np.s_[gy*cell:(gy+1)*cell, gx*cell:(gx+1)*cell]
                cw_ = w[sl].sum()
                cell_w[gy, gx]  += cw_
                cell_fx[gy, gx] += (w[sl] * fx[sl]).sum()
                cell_fy[gy, gx] += (w[sl] * fy[sl]).sum()

    if sum_w < 1.0:
        return None

    feats = [sum_curl / sum_w, sum_fx / sum_w, sum_fy / sum_w]
    for gy in range(GRID):
        for gx in range(GRID):
            cw_ = max(cell_w[gy, gx], 1.0)
            feats.append(cell_fx[gy, gx] / cw_)
            feats.append(cell_fy[gy, gx] / cw_)
    return np.array(feats, np.float64)


# ---------------------------------------------------------------------------
# Per-face direction model (nearest-centroid — calibration, not learning)
# ---------------------------------------------------------------------------

class DirectionModel:
    """
    One signed axis per face: w = mean(CW feats) - mean(CCW feats) in
    z-scored feature space; predict CW when the sample projects past the
    midpoint between the two class means.
    """

    def __init__(self):
        self.mu    = np.zeros(N_FEATURES)   # z-score stats (fit data)
        self.sigma = np.ones(N_FEATURES)
        self.axes  = {}                      # face_id -> (w, bias)

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    def fit(self, samples: list[tuple[int, int, np.ndarray]]):
        """samples: [(face_id, is_ccw, features), ...]"""
        X = np.stack([f for _, _, f in samples])
        self.mu    = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9

        self.axes = {}
        for face in range(6):
            cw  = [self._z(f) for fc, d, f in samples if fc == face and d == 0]
            ccw = [self._z(f) for fc, d, f in samples if fc == face and d == 1]
            if not cw or not ccw:
                continue
            m_cw, m_ccw = np.mean(cw, axis=0), np.mean(ccw, axis=0)
            w    = m_cw - m_ccw
            bias = float(w @ (m_cw + m_ccw) / 2.0)
            self.axes[face] = (w, bias)
        return self

    def predict(self, face_id: int, features: np.ndarray) -> bool | None:
        """True = CW, False = CCW, None = no axis for this face."""
        if face_id not in self.axes or features is None:
            return None
        w, bias = self.axes[face_id]
        return bool(w @ self._z(features) > bias)

    def score(self, face_id: int, features: np.ndarray) -> float | None:
        """Signed margin (>0 CW); magnitude ~ confidence."""
        if face_id not in self.axes or features is None:
            return None
        w, bias = self.axes[face_id]
        return float(w @ self._z(features) - bias)

    def save(self, path: str):
        data = {
            "mu":    self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "axes":  {str(k): {"w": w.tolist(), "bias": b}
                      for k, (w, b) in self.axes.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "DirectionModel":
        data = json.loads(Path(path).read_text())
        m = cls()
        m.mu    = np.array(data["mu"])
        m.sigma = np.array(data["sigma"])
        m.axes  = {int(k): (np.array(v["w"]), float(v["bias"]))
                   for k, v in data["axes"].items()}
        return m


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

# Move window relative to the BLE event timestamp, matching
# postprocess_session.py's before/after offsets.
WINDOW_PRE  = 0.18
WINDOW_POST = 0.28

# Search span for self-centering on the motion peak: BLE event latency vs.
# the physical turn is uncertain, so look wider, find where motion actually
# peaks, and take a tight window around that.
SEARCH_PRE  = 0.35
SEARCH_POST = 0.40
PEAK_HALF   = 0.15


def _center_on_motion_peak(frames: list[np.ndarray],
                           ts: list[float]) -> list[np.ndarray]:
    """
    Trim a wide candidate window to ±PEAK_HALF s around the frame pair
    with the highest motion energy (computed on tiny grayscale thumbs).
    """
    if len(frames) < 3:
        return frames
    thumbs = [cv2.cvtColor(cv2.resize(f, (64, 64)), cv2.COLOR_BGR2GRAY)
              for f in frames]
    energy = [float(np.mean(cv2.absdiff(a, b)))
              for a, b in zip(thumbs, thumbs[1:])]
    peak_t = ts[int(np.argmax(energy))]
    return [f for f, t in zip(frames, ts)
            if peak_t - PEAK_HALF <= t <= peak_t + PEAK_HALF]


def _load_stream_index(session_dir: Path) -> tuple[list[float], list[str]]:
    """(timestamps, filenames) from frames.jsonl, if the raw stream exists."""
    idx_file = session_dir / "frames.jsonl"
    if not idx_file.exists() or not (session_dir / "frames").is_dir():
        return [], []
    ts, files = [], []
    with open(idx_file) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                ts.append(r["ts"])
                files.append(r["file"])
    return ts, files


def load_session_samples(session_dir: Path, verbose: bool = True,
                         source: str = "labeled",
                         flow_backend: str = "farneback"
                         ) -> list[tuple[int, int, np.ndarray]]:
    """
    [(face_id, is_ccw, features), ...] for every usable labeled move.

    source:
      "labeled"     the 5 postprocessed snapshots per move (default).
                    Empirically the best flow source despite the 60-180ms
                    gaps — measured 65.1% cross-session direction acc.
      "stream"      consecutive ~30fps frames from frames.jsonl in the
                    BLE-anchored window. Cleaner per-pair flow but scored
                    worse (59.2%) — fine hand motion seems to wash out
                    the net turn signal.
      "stream-peak" like "stream" but self-centered on the motion-energy
                    peak. Worst (53.7%) — the peak often belongs to the
                    regrip, not the turn. Kept for reproducibility.

    Any stream source falls back to "labeled" per-move when frames are
    missing.
    """
    labeled = session_dir / "moves_labeled.jsonl"
    if not labeled.exists():
        print(f"  WARNING: no moves_labeled.jsonl in {session_dir.name} "
              f"— run postprocess_session.py first")
        return []

    stream_ts, stream_files = ([], [])
    if source in ("stream", "stream-peak"):
        stream_ts, stream_files = _load_stream_index(session_dir)

    samples, skipped = [], 0
    with open(labeled) as f:
        for line in f:
            m = json.loads(line.strip())
            label = m.get("ble_raw")
            if label is None or not (0 <= label <= 11):
                skipped += 1
                continue

            frames = []
            if stream_ts:
                t = m["timestamp"]
                pre  = SEARCH_PRE  if source == "stream-peak" else WINDOW_PRE
                post = SEARCH_POST if source == "stream-peak" else WINDOW_POST
                lo = np.searchsorted(stream_ts, t - pre)
                hi = np.searchsorted(stream_ts, t + post)
                kept_ts = []
                for i in range(lo, hi):
                    img = cv2.imread(str(session_dir / "frames" / stream_files[i]))
                    if img is not None:
                        frames.append(img)
                        kept_ts.append(stream_ts[i])
                if source == "stream-peak":
                    frames = _center_on_motion_peak(frames, kept_ts)

            if len(frames) < 2:   # labeled source, stream missing, or gap
                frames = []
                for win in FRAME_ORDER:
                    rel = m.get("frames", {}).get(win)
                    if rel:
                        img = cv2.imread(str(session_dir / rel))
                        if img is not None:
                            frames.append(img)

            feats = compute_features(frames, flow_backend=flow_backend)
            if feats is None:
                skipped += 1
                continue
            samples.append((label // 2, label % 2, feats))

    if verbose:
        print(f"  {session_dir.name}: {len(samples)} moves featurized"
              + (f" ({skipped} skipped)" if skipped else ""))
    return samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval_fold(train, test) -> list[tuple[int, int, bool]]:
    """Returns [(face_id, is_ccw_true, correct), ...] on the test fold."""
    model = DirectionModel().fit(train)
    out = []
    for face, d, feats in test:
        pred_cw = model.predict(face, feats)
        if pred_cw is None:
            continue
        out.append((face, d, pred_cw == (d == 0)))
    return out


def _report(results: list[tuple[int, int, bool]], title: str):
    print(f"\n  {title}")
    print(f"  {'Face':<8} {'n':>5} {'Direction acc':>14}")
    print(f"  {'-'*30}")
    total_n = total_c = 0
    for face in range(6):
        rs = [c for fc, _, c in results if fc == face]
        if not rs:
            continue
        n, c = len(rs), sum(rs)
        total_n += n
        total_c += c
        bar = "▓" * int(c / n * 20)
        print(f"  {FACE_NAMES[face]:<8} {n:>5} {c/n*100:>12.1f}%  {bar}")
    if total_n:
        print(f"  {'-'*30}")
        print(f"  {'ALL':<8} {total_n:>5} {total_c/total_n*100:>12.1f}%")
        print(f"  (chance = 50.0%)")


def evaluate(session_dirs: list[Path], source: str = "labeled",
            flow_backend: str = "farneback"):
    per_session = {d.name: load_session_samples(d, source=source,
                                                 flow_backend=flow_backend)
                   for d in session_dirs}
    per_session = {k: v for k, v in per_session.items() if v}

    if not per_session:
        sys.exit("No usable samples found in any session.")

    all_samples = [s for v in per_session.values() for s in v]

    if len(per_session) >= 2:
        # Leave-one-session-out: the honest number — calibration from one
        # session, tested on a different recording (lighting, grip, pose).
        results = []
        for held_out, test in per_session.items():
            train = [s for k, v in per_session.items()
                     if k != held_out for s in v]
            results.extend(_eval_fold(train, test))
        _report(results, "Leave-one-session-out (cross-session)")

    # Within-pool 80/20 — upper bound with matched conditions
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(all_samples))
    split = int(len(all_samples) * 0.8)
    train = [all_samples[i] for i in idx[:split]]
    test  = [all_samples[i] for i in idx[split:]]
    _report(_eval_fold(train, test), "Pooled 80/20 split (matched conditions)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optical-flow CW/CCW direction head — evaluate or fit")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Session folder(s) — supports globs: training_data/solve_*/")
    parser.add_argument("--fit-out", type=str, default=None,
                        help="Fit on ALL sessions and save DirectionModel JSON here")
    parser.add_argument("--source", choices=["labeled", "stream", "stream-peak"],
                        default="labeled",
                        help="Which frames to run flow on (see load_session_samples)")
    parser.add_argument("--flow-backend", choices=["farneback", "raft"],
                        default="farneback",
                        help="G1 (MODEL_REWORK_PLAN.md): substitute "
                             "torchvision's pretrained RAFT-small for the "
                             "default Farneback flow")
    args = parser.parse_args()

    session_dirs = [Path(p) for pattern in args.sessions
                    for p in (Path(".").glob(pattern)
                              if "*" in pattern else [Path(pattern)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]
    if not session_dirs:
        sys.exit("No session directories found. Check --sessions argument.")

    print(f"\n{'='*55}")
    print(f"  Optical-flow direction head")
    print(f"{'='*55}")

    evaluate(session_dirs, source=args.source, flow_backend=args.flow_backend)

    if args.fit_out:
        samples = [s for d in session_dirs
                   for s in load_session_samples(d, verbose=False,
                                                 source=args.source,
                                                 flow_backend=args.flow_backend)]
        DirectionModel().fit(samples).save(args.fit_out)
        print(f"\n  Model fit on all {len(samples)} samples → {args.fit_out}")
