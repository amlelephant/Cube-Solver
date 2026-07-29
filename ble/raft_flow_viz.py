"""
raft_flow_viz.py

Gate G1 of MODEL_REWORK_PLAN.md: visual half of the RAFT-vs-Farneback
comparison. Renders the SAME sample move windows viz_encodings.py already
uses (same MoveDiffDataset + pick_samples, same seed) as three columns:

    raw frame | Farneback flowwheel | RAFT flowwheel

Both flowwheel columns use the identical construction (dense flow per
consecutive frame pair, global-motion compensated via
encodings_move.fit_global_flow, magnitude-weighted mean over the window,
flow_to_wheel) -- only the underlying flow estimator differs. If RAFT
does not visibly isolate the turning layer any better than Farneback here,
that is real evidence against reviving flow, independent of the
quantitative CW/CCW probe in flow_direction.py.

    python raft_flow_viz.py --sessions training_data/solve_20260724_*/
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

import encodings_move as enc_mod
from encodings_move import fit_global_flow, flow_to_wheel
from flow_direction import _load_raft
from train_move_classifier import MoveDiffDataset, WCA_CLASS_NAMES
from viz_encodings import pick_samples, load_window, label, BG, FG, MUTED

IMG_SIZE = enc_mod.IMG_SIZE


def raft_flowwheel(frames_bgr: list, size: int = IMG_SIZE) -> np.ndarray:
    """RAFT counterpart of encodings_move.encode_flowwheel, same recipe."""
    import torch
    model, device, transforms, torch_mod = _load_raft()

    resized = [cv2.resize(f, (size, size)) for f in frames_bgr]

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch_mod.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)

    flows = []
    with torch.no_grad():
        for a, b in zip(resized, resized[1:]):
            t1, t2 = transforms(to_tensor(a), to_tensor(b))
            flow = model(t1.to(device), t2.to(device))[-1]
            flows.append(flow[0].permute(1, 2, 0).cpu().numpy())
    flow = np.stack([f - fit_global_flow(f) for f in flows], axis=0)
    return flow_to_wheel(flow.mean(axis=0))


def main():
    ap = argparse.ArgumentParser(
        description="Render RAFT vs Farneback flowwheel side by side on "
                    "real move windows (G1, MODEL_REWORK_PLAN.md)")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=10,
                    help="How many sampled moves to render")
    ap.add_argument("--out", default="viz_flow/raft_vs_farneback.png")
    args = ap.parse_args()

    session_dirs = [Path(p) for pat in args.sessions
                    for p in (Path(".").glob(pat) if "*" in pat else [Path(pat)])]
    session_dirs = [d for d in session_dirs if d.is_dir()]
    if not session_dirs:
        sys.exit("No session directories found.")

    ds = MoveDiffDataset(session_dirs, augment=False, label_mode="wca",
                         encoding="diffstack")
    if not len(ds):
        sys.exit("No samples found. Run postprocess_session.py first.")

    samples = pick_samples(ds, args.per_class, args.seed)
    if not samples:
        sys.exit("No CROPPED samples found - run cache_crops.py first.")
    rng = random.Random(args.seed)
    samples = rng.sample(samples, min(args.n, len(samples)))

    size, pad, head, lgut = args.size, 10, 46, 74
    cols = ["raw", "farneback", "raft"]
    w = lgut + len(cols) * (size + pad) + pad
    h = head + len(samples) * (size + pad) + pad
    canvas = np.full((h, w, 3), BG, dtype=np.uint8)
    for c, name in enumerate(cols):
        label(canvas, name, (lgut + c * (size + pad), head - 16), 0.52, FG)

    print(f"  rendering {len(samples)} sample move(s)...")
    for r, (idx, lab) in enumerate(samples):
        frames = load_window(ds.samples[idx])
        if frames is None:
            continue
        y = head + r * (size + pad)
        label(canvas, WCA_CLASS_NAMES[lab], (pad, y + size // 2), 0.85, FG, 2)

        tiles = [cv2.resize(frames[0], (size, size))]
        fb = enc_mod.encode_flowwheel(frames, IMG_SIZE)
        tiles.append(cv2.cvtColor(cv2.resize(fb, (size, size)),
                                  cv2.COLOR_RGB2BGR))
        rf = raft_flowwheel(frames, IMG_SIZE)
        tiles.append(cv2.cvtColor(cv2.resize(rf, (size, size)),
                                  cv2.COLOR_RGB2BGR))

        for c, tile in enumerate(tiles):
            x = lgut + c * (size + pad)
            canvas[y:y + size, x:x + size] = tile
        print(f"    {r+1}/{len(samples)}  {WCA_CLASS_NAMES[lab]}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(out), canvas)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
