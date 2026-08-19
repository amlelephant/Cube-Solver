"""
m5_diagnostics.py — the measurements that explain the headline number
rather than restate it.

  A. Holdout proximity ladder. The same weights scored on training
     sessions, on the same-day validation sessions that picked the epoch,
     on held-out days BRACKETED by training days, and on held-out days
     strictly AFTER the last training day. This is the difference between
     a number that describes memorisation and one that describes a user.

  B. Truth frame. The smart cube reports turns relative to its own core.
     A middle slice rotates that core, so after an `M` the BLE label names
     a face the camera never saw turn. Scoring the same predictions against
     cube-frame vs camera-frame truth measures what that costs.

  C. Per-class confusion and substitution taxonomy — is a naming error a
     wrong direction, an adjacent face, or the opposite face?

  D. Where in the solve the errors fall (normalised position), and how
     accuracy tracks solve speed, onset crowding and hour of day.

    python m5_diagnostics.py
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import torch

import common as C

CKPTS = ["move_ctc_spd_s0.pt", "move_ctc_spd_s1.pt"]
POST = C.DATA / "post"

# §1c of GAMEPLAN.md quotes "the 8 never-trained *_solve sessions". Two of
# them are the checkpoints' own validation sessions. Kept verbatim so the
# comparison is against what was actually reported, not a cleaned-up version.
OLD8 = ["solve_20260731_211018_solve", "solve_20260803_095533_solve",
        "solve_20260731_213559_solve", "solve_20260729_221809_solve",
        "solve_20260730_113054_solve", "solve_20260730_111941_solve",
        "solve_20260724_100120_solve", "solve_20260723_105530_solve"]


def day(name: str) -> dt.date:
    s = name.split("_")[1]
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def score_session(model, ck, d, dev, truth_key="camera"):
    from dataset import JointSessionStream
    from model import score_stream_joint
    from ctc_decode import prefix_beam_decode, ctc_to_moves
    stream = JointSessionStream(d / "detector_stream_color.npz")
    tag = None
    for t in CKPTS:
        if t.startswith(ck.get("_tag", "")):
            tag = t
    cache = POST / f"{ck['_tag']}__{d.name}.npz"
    if cache.exists():
        class_prob = np.load(cache)["class_prob"]
    else:
        _, class_prob, _ = score_stream_joint(model, stream, dev)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, class_prob=class_prob.astype(np.float32))
    lab, fr = prefix_beam_decode(np.log(np.maximum(class_prob, 1e-12)),
                                 beam=16)
    moves = ctc_to_moves(class_prob, lab, fr, fps=stream.fps)
    return [m["move"] for m in moves]


def main():
    from model import build_joint_from_ckpt
    from decode import align_sequences
    from reconstruct import WCA12

    dev = C.device()
    paths = [C.MD / "checkpoints" / c for c in CKPTS]
    seen = C.ckpt_seen(paths)
    ck0 = torch.load(paths[0], map_location="cpu", weights_only=False)
    train_names = set(ck0["train_session_names"])
    val_names = set(ck0["val_session_names"])
    last_train_day = max(day(n) for n in train_names)
    print(f"\n  last training day: {last_train_day}")

    # Every prepared *_solve session, tagged by proximity tier.
    tiers = {}
    for npz in sorted(C.SESSIONS.glob("solve_*_solve/detector_stream_color.npz")):
        n = npz.parent.name
        if n in train_names:
            t = "train"
        elif n in val_names:
            t = "val (same-day)"
        elif day(n) <= last_train_day:
            t = "held out, bracketed"
        else:
            t = "held out, after"
        tiers[n] = t
    # The 07-21/22 sessions have no _solve suffix (recorded before the
    # scramble/solve split existed) — include them as training-tier solves.
    for npz in sorted(C.SESSIONS.glob("solve_*/detector_stream_color.npz")):
        n = npz.parent.name
        if n in tiers or n.endswith("_scramble"):
            continue
        tiers[n] = ("train" if n in train_names else
                    "val (same-day)" if n in val_names else
                    "held out, bracketed" if day(n) <= last_train_day else
                    "held out, after")

    rows = []
    for cp in paths:
        ck = torch.load(cp, map_location=dev, weights_only=False)
        ck["_tag"] = cp.stem
        model = build_joint_from_ckpt(ck, dev)
        model.eval()
        for n, tier in tiers.items():
            d = C.SESSIONS / n
            gt_cam = C.truth_word(d)
            gt_cube = C.cube_word(d)
            if gt_cam is None or gt_cube is None:
                continue
            pred = score_session(model, ck, d, dev)
            cam = C.channel_split(gt_cam, pred)
            cube = C.channel_split(gt_cube, pred)
            meta = C.session_meta(d)
            subs = [(w, p) for o, w, p in align_sequences(gt_cam, pred)
                    if o == "sub"]
            # normalised position of every non-ok op
            ops = align_sequences(gt_cam, pred)
            npos = {k: [] for k in ("sub", "miss", "phantom")}
            i = 0
            for op, want, _ in ops:
                if want is not None:
                    if op in npos:
                        npos[op].append(i / max(len(gt_cam) - 1, 1))
                    i += 1
                elif op == "phantom":
                    npos["phantom"].append(i / max(len(gt_cam) - 1, 1))
            rows.append({
                "model": cp.stem, "session": n, "tier": tier,
                "day": str(day(n)), "hour": meta["hour"],
                "evening": meta["evening"], "tps": meta["tps"],
                "crowded_frac": meta["crowded_frac"],
                "ctc_floor": meta["ctc_floor"],
                "acc_camera": cam["acc"], "acc_cube": cube["acc"],
                "n_gt": cam["n_gt"], "miss": cam["miss"], "sub": cam["sub"],
                "phantom": cam["phantom"],
                "slice_session": gt_cam != gt_cube,
                "sub_pairs": subs, "err_pos": npos,
                "gt": gt_cam, "pred": pred,
            })
        print(f"  scored {cp.stem} on {len(tiers)} sessions")

    C.dump("m5_sessions.json", rows)

    # ---- A. proximity ladder -------------------------------------------
    print(f"\n{'='*84}\n  A. HOLDOUT PROXIMITY LADDER (mean per-session "
          f"accuracy, camera-frame truth)\n{'='*84}")
    print(f"  {'tier':<26}{'n':>4}{'seed 0':>11}{'seed 1':>11}{'mean':>10}")
    ladder = []
    for tier in ("train", "val (same-day)", "held out, bracketed",
                 "held out, after"):
        per = []
        for cp in paths:
            g = [r for r in rows if r["tier"] == tier and r["model"] == cp.stem]
            per.append(np.mean([r["acc_camera"] for r in g]) if g else np.nan)
        n = len([r for r in rows if r["tier"] == tier
                 and r["model"] == paths[0].stem])
        ladder.append({"tier": tier, "n": n, "seed0": float(per[0]),
                       "seed1": float(per[1]),
                       "mean": float(np.mean(per))})
        print(f"  {tier:<26}{n:>4}{per[0]*100:>10.1f}%{per[1]*100:>10.1f}%"
              f"{np.mean(per)*100:>9.1f}%")
    C.dump("m5_ladder.json", ladder)

    # ---- old vs new holdout ---------------------------------------------
    print(f"\n  Same weights, two holdout definitions:")
    for label, sel in (
            ("GAMEPLAN's 8 (incl. 2 val)", lambda r: r["session"] in OLD8),
            ("never-seen 14 (this paper)", lambda r: r["session"] not in seen
             and r["session"].endswith("_solve"))):
        per = [np.mean([r["acc_camera"] for r in rows
                        if sel(r) and r["model"] == cp.stem]) for cp in paths]
        n = len([r for r in rows if sel(r) and r["model"] == paths[0].stem])
        print(f"    {label:<32} n={n:<3} "
              f"s0 {per[0]*100:5.1f}%  s1 {per[1]*100:5.1f}%")

    # ---- B. truth frame --------------------------------------------------
    print(f"\n{'='*84}\n  B. TRUTH FRAME — cube-frame (BLE core) vs "
          f"camera-frame labels\n{'='*84}")
    sl = [r for r in rows if r["slice_session"] and r["tier"].startswith("held")]
    frame = {"n_slice_holdout_sessions": len(sl) // 2}
    if sl:
        print(f"  {'session':<34}{'seed':>6}{'cube-frame':>13}"
              f"{'camera-frame':>15}")
        for r in sorted(sl, key=lambda r: (r["session"], r["model"])):
            print(f"  {r['session']:<34}{r['model'][-2:]:>6}"
                  f"{r['acc_cube']*100:>12.1f}%{r['acc_camera']*100:>14.1f}%")
        frame["cube_mean"] = float(np.mean([r["acc_cube"] for r in sl]))
        frame["camera_mean"] = float(np.mean([r["acc_camera"] for r in sl]))
        print(f"\n  mean over slice-bearing held-out sessions: "
              f"{frame['cube_mean']*100:.1f}% -> {frame['camera_mean']*100:.1f}%"
              f"  ({(frame['camera_mean']-frame['cube_mean'])*100:+.1f} pts)")
    allh = [r for r in rows if r["tier"].startswith("held")]
    frame["corpus_cube_mean"] = float(np.mean([r["acc_cube"] for r in allh]))
    frame["corpus_camera_mean"] = float(np.mean([r["acc_camera"] for r in allh]))
    print(f"  over the whole holdout: "
          f"{frame['corpus_cube_mean']*100:.1f}% -> "
          f"{frame['corpus_camera_mean']*100:.1f}%")
    C.dump("m5_frame.json", frame)

    # ---- C. confusion ----------------------------------------------------
    held = [r for r in rows if r["tier"].startswith("held")]
    M = np.zeros((12, 12), dtype=int)
    kinds = {"inverse": 0, "adjacent": 0, "opposite": 0}
    for r in held:
        for w, p in r["sub_pairs"]:
            M[WCA12.index(w), WCA12.index(p)] += 1
            kinds[C.sub_kind(w, p)] += 1
    tot = sum(kinds.values())
    print(f"\n{'='*84}\n  C. SUBSTITUTION TAXONOMY ({tot} substitutions, "
          f"held-out, both seeds)\n{'='*84}")
    for k, v in kinds.items():
        print(f"  {k:<12}{v:>5}  {v/max(tot,1)*100:5.1f}%")
    C.dump("m5_confusion.json",
           {"matrix": M.tolist(), "labels": WCA12, "kinds": kinds,
            "n_sub": tot})

    # ---- D. error position + correlates ----------------------------------
    pos = {k: [p for r in held for p in r["err_pos"][k]]
           for k in ("sub", "miss", "phantom")}
    print(f"\n{'='*84}\n  D. ERROR POSITION AND CORRELATES\n{'='*84}")
    for k, v in pos.items():
        if v:
            print(f"  {k:<9} n={len(v):<5} mean normalised position "
                  f"{np.mean(v):.3f}  (0 = start of solve, 1 = end)")
    corr = {}
    for xk in ("tps", "crowded_frac", "hour", "ctc_floor"):
        xs = np.array([r[xk] for r in held], dtype=float)
        ys = np.array([r["acc_camera"] for r in held], dtype=float)
        ok = ~np.isnan(xs)
        r_ = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        corr[xk] = r_
        print(f"  accuracy vs {xk:<14} r = {r_:+.2f}  (n={ok.sum()} "
              f"session-seed pairs)")
    C.dump("m5_positions.json", {"positions": pos, "correlations": corr})
    print("\n  A correlation over ~28 points with several candidate "
          "variables is a hypothesis, not a finding — see GAMEPLAN §1d.")


if __name__ == "__main__":
    main()
