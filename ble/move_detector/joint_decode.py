"""
joint_decode.py

Stage A (MODEL_REWORK_PLAN.md): turns a joint model's per-frame
posteriorgram into the SAME interchange format live_detect.analyse()
produces — a list of {"frame", "move", "conf", "probs", "score"} dicts —
so every consumer downstream of that format (reconstruct.costs_from_moves,
verify_solve.py, oracle_attribution.py, falsifiability_batch.py) works
completely unchanged against the new model. Nothing in reconstruct.py or
verify_solve.py needs to change for the joint model to be decodable.

This generalises D1's soft-onset lattice (PATH_TO_VERIFICATION.md Sec5):
D1 fed a separate detector's sub-threshold peaks to the decoder as
near-free candidates; here every candidate already carries its own full
posterior straight from one model, with no window/anchor step in between
at all — the anchor-quantisation and window-collapse defects that section
inventories cannot occur because there is no window.

    from joint_decode import posteriorgram_to_moves, load_joint_replay
    moves = load_joint_replay(session_dir, model, device)["moves"]
    pred_names, cost_rows, del_costs = RC.costs_from_moves(moves, threshold)
"""

import json
from pathlib import Path

import numpy as np

from decode import peak_pick, MIN_SEP
from reconstruct import WCA12
from model import score_stream_joint
from dataset import JointSessionStream


def posteriorgram_to_moves(onset_prob: np.ndarray, class_prob: np.ndarray,
                          threshold: float = 0.5,
                          min_sep: int = MIN_SEP) -> list[dict]:
    """
    (T,) onset_prob, (T, 13) class_prob -> a moves list.

    Peaks are picked on (1 - background posterior) — the CLASS head's own
    belief a move is happening — not the separate onset head's sigmoid:
    the two are trained to agree (background column's target is exactly
    1 - the onset target, see dataset.build_dense_targets) but the 12-way
    distribution attached to each candidate is drawn from the class head,
    so the peak and the distribution it carries come from the same source
    rather than mixing two different heads' beliefs.
    """
    fg = 1.0 - class_prob[:, 12]
    onsets = peak_pick(fg, threshold=threshold, min_sep=min_sep)

    moves = []
    for o in onsets:
        row = class_prob[int(o), :12].astype(np.float64)
        total = row.sum()
        probs = (row / total) if total > 1e-9 else np.full(12, 1 / 12)
        cls = int(np.argmax(probs))
        moves.append({"frame": int(o), "move": WCA12[cls],
                      "conf": float(probs[cls]),
                      "probs": [float(p) for p in probs],
                      "score": float(fg[int(o)])})
    return moves


def load_joint_replay(session_dir: Path, model, device, sigma: float = 1.0,
                      threshold: float | None = None,
                      min_sep: int | None = None,
                      cache: bool = True, refresh_cache: bool = False
                      ) -> dict | None:
    """
    Score one session with a joint model and return {"moves": [...],
    "class_names": WCA12, "onset_prob": ..., "class_prob": ...} — the
    joint-model counterpart of reconstruct._load_replay.

    Cached to <session>/joint_replay_<tag>.json keyed by the checkpoint's
    own identity (epoch + seed, since Stage A checkpoints have no stable
    filename convention yet) so a retrained checkpoint under the same
    path can never silently serve a stale lattice — the exact footgun
    reconstruct._load_replay's own docstring warns about for D1.
    """
    stream_path = session_dir / "detector_stream_color.npz"
    if not stream_path.exists():
        print(f"  {session_dir.name}: no detector_stream_color.npz — "
              f"skipping (run prepare_data.py --color)")
        return None
    stream = JointSessionStream(stream_path, sigma=sigma)

    ckpt_tag = getattr(model, "_ckpt_tag", "unknown")
    cache_path = session_dir / f"joint_replay_{ckpt_tag}.json"
    if cache and cache_path.exists() and not refresh_cache:
        data = json.loads(cache_path.read_text())
        return {**data, "class_names": WCA12}

    onset_prob, class_prob = score_stream_joint(model, stream, device)
    threshold = 0.5 if threshold is None else threshold
    min_sep = MIN_SEP if min_sep is None else min_sep
    moves = posteriorgram_to_moves(onset_prob, class_prob, threshold, min_sep)

    out = {"moves": moves, "onset_idx": stream.onset_idx.tolist(),
          "onset_class": stream.onset_class.tolist(), "fps": stream.fps,
          "n_frames": len(stream)}
    if cache:
        cache_path.write_text(json.dumps(out))
    return {**out, "class_names": WCA12}
