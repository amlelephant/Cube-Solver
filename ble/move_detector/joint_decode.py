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
import time
from pathlib import Path

import numpy as np

from decode import peak_pick, MIN_SEP
from reconstruct import WCA12
from model import score_stream_joint
from dataset import JointArrayStream, JointSessionStream


def posteriorgram_to_moves(onset_prob: np.ndarray, class_prob: np.ndarray,
                          threshold: float = 0.5,
                          min_sep: int = MIN_SEP,
                          fps: float = 30.0,
                          count_prob: np.ndarray | None = None,
                          count_radius: int = 2) -> list[dict]:
    """
    (T,) onset_prob, (T, 13) class_prob -> a moves list.

    Peaks are picked on (1 - background posterior) — the CLASS head's own
    belief a move is happening — not the separate onset head's sigmoid:
    the two are trained to agree (background column's target is exactly
    1 - the onset target, see dataset.build_dense_targets) but the 12-way
    distribution attached to each candidate is drawn from the class head,
    so the peak and the distribution it carries come from the same source
    rather than mixing two different heads' beliefs.

    `fps` only feeds the "time" field (seconds from stream start) that
    live_detect.print_sequence reads on low-confidence moves — cosmetic,
    not consumed by the decoder.

    `count_prob` (optional, (T, 3) from a count-head checkpoint) enables
    peak SPLITTING — see split_peak. Without it this function behaves
    exactly as it did before the count head existed.
    """
    fg = 1.0 - class_prob[:, 12]
    onsets = peak_pick(fg, threshold=threshold, min_sep=min_sep)

    moves = []
    for o in onsets:
        o = int(o)
        emit = split_peak(o, class_prob, count_prob, count_radius) \
            if count_prob is not None else [(o, None)]
        for frame, probs in emit:
            if probs is None:
                row = class_prob[frame, :12].astype(np.float64)
                total = row.sum()
                probs = (row / total) if total > 1e-9 else np.full(12, 1 / 12)
            cls = int(np.argmax(probs))
            moves.append({"frame": frame, "time": float(frame) / fps,
                          "move": WCA12[cls], "conf": float(probs[cls]),
                          "probs": [float(p) for p in probs],
                          "score": float(fg[frame])})
    moves.sort(key=lambda m: m["frame"])
    return moves


def split_peak(o: int, class_prob: np.ndarray, count_prob: np.ndarray,
               radius: int) -> list[tuple[int, np.ndarray | None]]:
    """
    One picked peak -> one or two (frame, probs) candidates.

    peak_pick requires a strict local maximum, so two onsets within
    ~MIN_SEP frames of each other can never both be reported: they arrive
    as a single merged hump, and decode.py measures 34% of sub-150ms pairs
    doing exactly that even at sigma=1. Those are unrecoverable misses no
    threshold can reach. The count head answers the question the curve's
    shape cannot — "are there TWO onsets within +/-radius of here" — as a
    label rather than a shape, and this is where that label is spent.

    When the count head's argmax over the peak's neighbourhood is 2, the
    peak is emitted TWICE, taking the top two classes of the (renormalised
    12-way) posterior at the peak. Two justifications for reading both
    moves off one frame's posterior rather than hunting two frames:

      * build_dense_targets already trains that frame's row to be the
        proportionally-rescaled MIXTURE of both classes when two onsets
        overlap, so the second-ranked class there is a supervised
        quantity, not a leftover.
      * ordering mostly does not matter. Of the 32 same-frame onset pairs
        across the prepared sessions, 26 are opposite-face (R with L', U
        with D') — one two-handed motion — and opposite-face turns
        COMMUTE, so either order yields the same cube state and the
        decoder is indifferent. The two are emitted one frame apart in
        posterior-rank order, which is arbitrary for those 26 and a coin
        flip for the remaining 6.

    A split is only taken when the runner-up class actually holds mass
    (>= SPLIT_MIN_PROB); a count of 2 attached to a posterior that is
    ~all one class is a count-head false positive, and emitting a
    near-zero-probability second move there just buys a phantom.
    """
    row = class_prob[o, :12].astype(np.float64)
    total = row.sum()
    probs = (row / total) if total > 1e-9 else np.full(12, 1 / 12)

    lo, hi = max(0, o - radius), min(len(count_prob), o + radius + 1)
    says_two = int(count_prob[lo:hi].argmax(axis=1).max()) >= 2
    order = np.argsort(probs)[::-1]
    if not says_two or probs[order[1]] < SPLIT_MIN_PROB:
        return [(o, probs)]

    first = np.zeros(12); first[order[0]] = 1.0
    second = np.zeros(12); second[order[1]] = 1.0
    # Keep the real posterior on the primary; the secondary is a
    # one-hot-ish claim carrying the runner-up's own mass as confidence so
    # reconstruct.costs_from_moves can still price a substitution on it.
    second = second * probs[order[1]] + (1 - probs[order[1]]) / 12
    second /= second.sum()
    return [(o, probs), (min(o + 1, len(class_prob) - 1), second)]


SPLIT_MIN_PROB = 0.15   # see split_peak


def load_joint_replay(session_dir: Path, model, device, sigma: float = 1.0,
                      threshold: float | None = None,
                      min_sep: int | None = None,
                      cache: bool = True, refresh_cache: bool = False,
                      use_count: bool = True, count_radius: int = 2
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
    # `split` is part of the cache identity, not just the checkpoint: the
    # SAME count-head checkpoint decodes to two different move lists with
    # peak splitting on and off, and that A/B is the whole measurement.
    split = use_count and getattr(model, "n_counts", None)
    cache_path = (session_dir /
                  f"joint_replay_{ckpt_tag}{'_split' if split else ''}.json")
    if cache and cache_path.exists() and not refresh_cache:
        data = json.loads(cache_path.read_text())
        return {**data, "class_names": WCA12}

    onset_prob, class_prob, count_prob = score_stream_joint(model, stream, device)
    threshold = 0.5 if threshold is None else threshold
    min_sep = MIN_SEP if min_sep is None else min_sep
    moves = posteriorgram_to_moves(onset_prob, class_prob, threshold, min_sep,
                                   fps=stream.fps,
                                   count_prob=count_prob if split else None,
                                   count_radius=count_radius)

    out = {"moves": moves, "onset_idx": stream.onset_idx.tolist(),
          "onset_class": stream.onset_class.tolist(), "fps": stream.fps,
          "n_frames": len(stream)}
    if cache:
        cache_path.write_text(json.dumps(out))
    return {**out, "class_names": WCA12}


def live_posteriorgram(load_color, n_frames: int, fps: float, detector,
                       model, device, verbose: bool = True) -> dict:
    """
    Live capture frames -> the model's per-frame posteriorgram.

    Everything up to and including scoring is identical for both Stage A
    arms — the peak-picked joint model and the CTC model share a trunk and
    an input pipeline, and differ ONLY in how the posteriorgram is turned
    into a move list. Splitting the shared part out here is what makes that
    literally true rather than approximately true: a crop or stream
    difference between the two would be indistinguishable from a decoder
    difference in a live A/B, which is the comparison this exists to serve.
    """
    from prepare_data import per_frame_boxes, build_color_stream, center_square

    t0 = time.time()
    if detector is not None:
        boxes, n_det = per_frame_boxes(detector, load_color, n_frames)
        cropped = n_det > 0
        crop_note = (f"{n_det} cube detections -> per-frame boxes" if cropped
                     else "cube NEVER detected -> centered square crop "
                          "(the joint model trained on cube crops!)")
    else:
        probe = load_color(0)
        boxes = np.tile(center_square(probe.shape),
                        (n_frames, 1)).astype(np.int32)
        cropped = False
        crop_note = "no detector -> centered square crop"
    if verbose:
        print(f"  crop:   {crop_note}  ({time.time()-t0:.1f}s)")

    frames = build_color_stream(load_color, boxes, n_frames)
    stream = JointArrayStream(frames, name="live", fps=fps, sigma=1.0)

    onset_prob, class_prob, count_prob = score_stream_joint(model, stream,
                                                            device)
    return {"onset_prob": onset_prob, "class_prob": class_prob,
            "count_prob": count_prob, "boxes": boxes, "cropped": cropped,
            "t0": t0}


def analyse_joint_live(load_color, n_frames: int, fps: float, detector,
                       model, device, threshold: float, min_sep: int,
                       verbose: bool = True, count_radius: int = 2) -> dict:
    """
    Live counterpart of live_detect.analyse() for the joint model: same
    capture -> crop -> score -> peak-pick shape verify_solve.py already
    drives, but ONE model instead of a detector+classifier pair, and
    colour input instead of grayscale. Returns the same
    {"moves": [...], "class_names": WCA12} shape live_detect.analyse()
    does, so verify_solve.py's downstream code (costs_from_moves,
    verify_claim, falsifiability_sweep, print_sequence) is unchanged.
    """
    pg = live_posteriorgram(load_color, n_frames, fps, detector, model,
                            device, verbose)
    moves = posteriorgram_to_moves(pg["onset_prob"], pg["class_prob"],
                                   threshold, min_sep, fps=fps,
                                   count_prob=pg["count_prob"],
                                   count_radius=count_radius)
    if verbose:
        print(f"  joint model: {len(moves)} moves "
              f"({time.time()-pg['t0']:.1f}s total)")

    return {"scores": pg["onset_prob"], "moves": moves, "boxes": pg["boxes"],
           "class_names": WCA12, "cropped": pg["cropped"], "fps": fps,
           "n_frames": n_frames}


def analyse_ctc_live(load_color, n_frames: int, fps: float, detector,
                     model, device, beam: int = 16, lm=None,
                     alpha: float = 0.0, beta: float = 0.0,
                     verbose: bool = True) -> dict:
    """
    Live counterpart for the CTC arm (train_ctc.py): the same capture,
    crop and scoring as analyse_joint_live, then ctc_decode.
    prefix_beam_decode over FRAMES instead of peak_pick over candidates,
    and the same moves-list interchange format out the other end.

    No `threshold` and no `min_sep` appear here, and their absence is the
    point rather than an omission: with CTC there is no candidate list to
    threshold, so "is there a move at this frame" is settled inside the
    beam search. Both knobs are still accepted by the caller for the other
    arms and are simply not consulted on this path — passing one and
    expecting it to bite would be a silent no-op, so it cannot be passed.
    """
    from ctc_decode import prefix_beam_decode, ctc_to_moves

    pg = live_posteriorgram(load_color, n_frames, fps, detector, model,
                            device, verbose)
    class_prob = pg["class_prob"]
    log_probs = np.log(np.maximum(class_prob, 1e-12))
    labels, frames = prefix_beam_decode(log_probs, beam=beam, lm=lm,
                                        alpha=alpha, beta=beta)
    moves = ctc_to_moves(class_prob, labels, frames, fps=fps)
    if verbose:
        print(f"  CTC model: {len(moves)} moves, prefix beam {beam}"
              + (f" + LM (alpha {alpha}, beta {beta})" if lm is not None
                 else "")
              + f"  ({time.time()-pg['t0']:.1f}s total)")

    return {"scores": pg["onset_prob"], "moves": moves, "boxes": pg["boxes"],
           "class_names": WCA12, "cropped": pg["cropped"], "fps": fps,
           "n_frames": n_frames}
