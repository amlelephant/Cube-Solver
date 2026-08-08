"""
trajectory_anomaly.py — one-class anomaly model over cube-trajectory
features, as a learned complement to continuity_guard.py's hand-tuned
motion thresholds.

The idea
--------
The guard's motion rules are ~8 interacting hand-tuned constants, and their
history shows the approach at its limit: GAP_FLAG_S 0.30->0.18, SIG_DIST
0.35->0.45, SIG_DIST_INTERIOR 0.55->0.65, EDGE_MARGIN 0.08->0.12, each
retune fixing one case and risking others. A single frame cannot express
"this trajectory is weird"; a trailing window can.

Trained ONE-CLASS on legit sessions only — never on recorded attacks.
Attacks are scarce (you can record a few dozen), so training on them would
fit one person's swap style and miss techniques nobody thought to perform.
Characterising NORMAL from the ~69 recorded legit sessions generalises to
unseen attacks, and keeps recorded attacks as an honest held-out test. Same
discipline as verify_solve.py's falsifiability decoys: they validate, they
never train.

Inputs are pure detection geometry from trajectory.npz — box position,
size, confidence, count — never pixels. ~16 floats per window, so scoring
costs nothing next to YOLO and needs no GPU.

Scale/position normalisation matters for cross-room generalisation:
positions and sizes are fractions of frame dimensions, and speeds are in
box-diagonals per second (the same scale-invariant convention as the
guard's jump_frac). A model keyed to one room's pixel scale would not
transfer.

  fit    fit the model on legit sessions, with a SESSION-LEVEL held-out
         split so the reported false-positive rate is honest (measuring it
         on training sessions would be meaningless), and pick the operating
         threshold from the false-DQ budget rather than by hand.
  score  score sessions and report per-session anomaly, with per-feature
         attribution so a flag is explainable ("speed_p95 at the 99.9th
         percentile"), not an opaque number.

Run from inside cv/detection (see CLAUDE.md):
    python dump_trajectories.py --root ../../ble/training_data
    python trajectory_anomaly.py fit   --root ../../ble/training_data
    python trajectory_anomaly.py score --root ../../cv/labeling/attack_sessions
"""

import argparse
import json
import os

import numpy as np

from continuity_guard import (EDGE_MARGIN, MULTI_CONF, TRACK_CONTINUITY_RADIUS,
                              dedup_boxes, pick_continuity)

MODEL_FILE   = "trajectory_anomaly.npz"
WINDOW_S     = 2.0    # trailing window length. Matches move_detector/model.py's
                      #   ~2.0s TCN receptive field, and comfortably contains a
                      #   swap round trip (GAP_DQ_S=0.5s is the guard's ceiling).
STRIDE_S     = 0.25   # emit a window this often
MIN_FRAMES   = 8      # windows thinner than this are not scoreable
TRIM_FRAC    = 0.02   # robust covariance: iteratively drop this fraction of
                      #   worst-scoring training windows and refit. The legit
                      #   corpus contains real detector glitches (2026-08-03:
                      #   phantom background boxes, split boxes) and a plain
                      #   Gaussian fit would model those as normal.
TRIM_ROUNDS  = 3
FP_BUDGET    = 0.01   # false-DQ budget (LAUNCH_ROADMAP §5 go/no-go: <1%).
                      #   The threshold is chosen to meet this on held-out
                      #   legit sessions, not tuned by hand.

FEATURE_NAMES = [
    # kinematics (continuous, well-conditioned — see ZERO-INFLATION below)
    "speed_mean", "speed_p95", "speed_max", "accel_p95", "straightness",
    # size / shape
    "area_cv", "area_rate_p95", "aspect_std",
    # detector quality
    "conf_mean", "conf_min",
    # border geometry
    "edge_prox_min", "edge_frac",
    # presence / uniqueness (overlaps the rule-based checks)
    "presence_frac", "max_gap_s", "gap_rate", "multi_frac",
]

# ZERO-INFLATION (measured 2026-08-04, 68 sessions). Several features are
# exactly 0.0 (or exactly 1.0) in the overwhelming majority of legit
# windows: edge_frac, max_gap_s, gap_rate, multi_frac, presence_frac. Their
# MAD is then ~0, so MAD-standardisation turns any tiny deviation into a
# huge z-score — a 62ms detection blink scored 1302 while the median legit
# session scored 22. That is a modelling artifact, not an anomaly.
#
# Tail ratio (held-out max / median — lower means "normal" is characterised
# more tightly, leaving more headroom to separate an attack):
#     all 16               272 params   1781x
#     motion 12            156 params    670x
#     kinematics 7          56 params    799x
#     kinematics 5 (below)  30 params     21x
#
# So the zero-inflated features belong in continuity_guard.py's RULES, where
# a hard bound is exactly right, not in a Gaussian anomaly model. This also
# matches the intended division of labour: rules own presence and
# uniqueness (simple, provable), the learned model owns motion.
#
# CAUTION: a tighter fit to legit data is NOT automatically a better
# detector. edge_prox_min/edge_frac are dropped here because they are
# ill-conditioned, but the border genuinely IS the swap route (the guard's
# whole EDGE_MARGIN premise). Dropping them may cost detection power. Which
# set actually separates attacks is UNANSWERABLE without attack data — do
# not treat "kinematics5" as settled until record_attack.py sessions have
# been scored. A rank/quantile transform is the untried middle path that
# would keep the border signal without the conditioning blowup.
FEATURE_SETS = {
    "kinematics5": ["speed_mean", "speed_p95", "speed_max", "accel_p95",
                    "straightness"],
    "kinematics7": ["speed_mean", "speed_p95", "speed_max", "accel_p95",
                    "straightness", "edge_prox_min", "edge_frac"],
    "motion12": FEATURE_NAMES[:12],
    "all16": FEATURE_NAMES,
}
DEFAULT_SET = "kinematics5"


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------

def load_trajectory(session_dir):
    path = os.path.join(session_dir, "trajectory.npz")
    if not os.path.isfile(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {"frame_idx": d["frame_idx"], "t": d["t"], "boxes": d["boxes"],
            "n_frames": int(d["n_frames"]), "fw": int(d["fw"]),
            "fh": int(d["fh"]), "name": str(d["name"])}


def per_frame_track(traj):
    """Collapse the ragged box stream into one tracked box per frame.

    Uses continuity_guard.pick_continuity — the same rule the guard itself
    uses — so the model sees the cube the guard is tracking, not whichever
    box YOLO scored highest that frame (2026-08-03: a background object
    briefly outscoring the real cube is exactly what produced a false
    teleport DQ).
    """
    fw, fh = traj["fw"], traj["fh"]
    by_frame = {}
    for fi, t, b in zip(traj["frame_idx"], traj["t"], traj["boxes"]):
        by_frame.setdefault(int(fi), {"t": float(t), "boxes": []})["boxes"].append(tuple(b))

    out = []          # (t, cx, cy, w, h, conf, n_multi) normalized; None box -> absent
    ref = None
    for fi in range(traj["n_frames"]):
        rec = by_frame.get(fi)
        if rec is None:
            out.append(None)
            continue
        boxes = dedup_boxes(rec["boxes"])
        if not boxes:
            out.append(None)
            continue
        best = pick_continuity(boxes, ref)
        ref = best
        x1, y1, x2, y2, conf = best
        n_multi = sum(1 for b in boxes if b[4] >= MULTI_CONF)
        out.append((rec["t"], ((x1 + x2) / 2) / fw, ((y1 + y2) / 2) / fh,
                    (x2 - x1) / fw, (y2 - y1) / fh, conf, n_multi))
    # timestamps for absent frames, so gaps have duration
    times = _fill_times(out, by_frame, traj["n_frames"])
    return out, times


def _fill_times(track, by_frame, n_frames):
    times = np.full(n_frames, np.nan, dtype=np.float64)
    for fi, rec in by_frame.items():
        times[fi] = rec["t"]
    # linear interpolation across absent frames (they carry no detection but
    # do carry elapsed time, which is what gap duration needs)
    idx = np.arange(n_frames)
    known = ~np.isnan(times)
    if known.sum() >= 2:
        times = np.interp(idx, idx[known], times[known])
    elif known.sum() == 1:
        times = np.full(n_frames, times[known][0])
    else:
        times = idx / 30.0
    return times


def window_features(track, times, i0, i1):
    """Features for track[i0:i1]. Returns None if too thin to be meaningful."""
    seg = track[i0:i1]
    tseg = times[i0:i1]
    if len(seg) < MIN_FRAMES:
        return None
    dur = float(tseg[-1] - tseg[0])
    if dur <= 1e-3:
        return None

    present = [(t, s) for t, s in zip(tseg, seg) if s is not None]
    presence_frac = len(present) / len(seg)

    # gap structure over the whole window (absences carry the swap window)
    gaps, run_start = [], None
    for t, s in zip(tseg, seg):
        if s is None and run_start is None:
            run_start = t
        elif s is not None and run_start is not None:
            gaps.append(t - run_start)
            run_start = None
    if run_start is not None:
        gaps.append(tseg[-1] - run_start)
    max_gap_s = max(gaps) if gaps else 0.0
    gap_rate = len(gaps) / dur

    multi_frac = (sum(1 for _, s in present if s[6] >= 2) / len(present)
                  if present else 0.0)

    if len(present) < 3:
        # not enough detections to speak about motion; emit a maximally
        # "absent" row rather than dropping the window, since sustained
        # absence is itself the signal.
        return dict(zip(FEATURE_NAMES, [
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
            0.5, 0.0,
            presence_frac, max_gap_s, gap_rate, multi_frac,
        ]))

    ts = np.array([t for t, _ in present], dtype=np.float64)
    cx = np.array([s[1] for _, s in present])
    cy = np.array([s[2] for _, s in present])
    w = np.array([s[3] for _, s in present])
    h = np.array([s[4] for _, s in present])
    conf = np.array([s[5] for _, s in present])

    diag = np.sqrt(w ** 2 + h ** 2)
    diag = np.maximum(diag, 1e-6)
    dt = np.diff(ts)
    dt = np.maximum(dt, 1e-4)
    # speed in box-diagonals per second: scale-invariant, matching the
    # guard's jump_frac convention so a value is comparable across rooms.
    step = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2) / diag[:-1]
    speed = step / dt
    accel = np.abs(np.diff(speed) / dt[:-1]) if len(speed) >= 2 else np.array([0.0])

    path_len = float(np.sum(step))
    net = float(np.sqrt((cx[-1] - cx[0]) ** 2 + (cy[-1] - cy[0]) ** 2) / diag[0])
    straightness = net / (path_len + 1e-6)

    area = w * h
    area_cv = float(np.std(area) / (np.mean(area) + 1e-9))
    area_rate = np.abs(np.diff(area) / dt) / (area[:-1] + 1e-9)
    aspect = w / np.maximum(h, 1e-6)

    edge_dist = np.minimum.reduce([cx, cy, 1.0 - cx, 1.0 - cy])
    edge_frac = float(np.mean(edge_dist < EDGE_MARGIN))

    return dict(zip(FEATURE_NAMES, [
        float(np.mean(speed)), float(np.percentile(speed, 95)), float(np.max(speed)),
        float(np.percentile(accel, 95)), straightness,
        area_cv, float(np.percentile(area_rate, 95)), float(np.std(aspect)),
        float(np.mean(conf)), float(np.min(conf)),
        float(np.min(edge_dist)), edge_frac,
        presence_frac, max_gap_s, gap_rate, multi_frac,
    ]))


def session_windows(session_dir):
    """[(t_start, t_end, feature_vector)] for one session."""
    traj = load_trajectory(session_dir)
    if traj is None or traj["n_frames"] < MIN_FRAMES:
        return []
    track, times = per_frame_track(traj)
    out = []
    t_end_total = times[-1]
    t = times[0]
    while t + WINDOW_S <= t_end_total + 1e-9:
        i0 = int(np.searchsorted(times, t, side="left"))
        i1 = int(np.searchsorted(times, t + WINDOW_S, side="right"))
        feats = window_features(track, times, i0, i1)
        if feats is not None:
            out.append((float(t), float(t + WINDOW_S),
                        np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float64)))
        t += STRIDE_S
    return out


# ---------------------------------------------------------------------------
# one-class model: robust Gaussian -> squared Mahalanobis distance
# ---------------------------------------------------------------------------

def _fit_robust(X):
    """Iteratively-trimmed Gaussian. Few parameters by construction: a mean
    vector and a covariance matrix over 16 features."""
    keep = np.ones(len(X), dtype=bool)
    mu = cov_inv = None
    for _ in range(TRIM_ROUNDS + 1):
        Xk = X[keep]
        mu = Xk.mean(axis=0)
        cov = np.cov(Xk, rowvar=False)
        # ridge for numerical stability: some features are near-constant on
        # legit data (that is informative, not a bug, but it makes the raw
        # covariance singular)
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.pinv(cov)
        d = _mahal(X, mu, cov_inv)
        cut = np.quantile(d, 1.0 - TRIM_FRAC)
        keep = d <= cut
    return mu, cov_inv


def _mahal(X, mu, cov_inv):
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, cov_inv, d)


def _standardize_fit(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-9, 1.0, mad)
    return med, mad


def discover_sessions(root):
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "trajectory.npz"))
    )


def cmd_fit(args):
    sessions = discover_sessions(args.root)
    if not sessions:
        raise SystemExit(f"no sessions with trajectory.npz under {args.root} "
                         "— run dump_trajectories.py first")

    rng = np.random.default_rng(0)
    order = list(sessions)
    rng.shuffle(order)
    n_val = max(1, int(len(order) * args.val_frac))
    val_sessions, fit_sessions = order[:n_val], order[n_val:]
    print(f"{len(sessions)} legit sessions: {len(fit_sessions)} fit / "
          f"{len(val_sessions)} held out\n")

    def collect(names):
        per = {}
        for n in names:
            wins = session_windows(os.path.join(args.root, n))
            if wins:
                per[n] = np.stack([w[2] for w in wins])
        return per

    fit_per = collect(fit_sessions)
    val_per = collect(val_sessions)
    if not fit_per:
        raise SystemExit("no scoreable windows in the fit split")

    Xfit = np.concatenate(list(fit_per.values()))
    feats = FEATURE_SETS[args.features]
    keep_idx = [FEATURE_NAMES.index(n) for n in feats]
    Xfit = Xfit[:, keep_idx]

    med, mad = _standardize_fit(Xfit)
    Z = (Xfit - med) / mad
    mu, cov_inv = _fit_robust(Z)

    # session score = max window distance (a swap is localised; averaging
    # over a 60s solve would bury it)
    def session_score(X):
        Z = ((X[:, keep_idx] - med) / mad)
        return float(np.max(_mahal(Z, mu, cov_inv)))

    fit_scores = {n: session_score(X) for n, X in fit_per.items()}
    val_scores = {n: session_score(X) for n, X in val_per.items()}

    # Threshold from the false-DQ budget on HELD-OUT legit sessions.
    #
    # HONESTY NOTE: with ~17 held-out sessions, a 1% quantile cannot be
    # estimated — 1% of 17 is far below one session. Setting the threshold
    # at the observed held-out max would report "0 false positives" purely
    # BY CONSTRUCTION and mean nothing. So the operating threshold is the
    # held-out max with a margin, and the false-positive rate it implies is
    # reported as a bound (<1/n_val), never as a measurement. A real <1%
    # figure needs on the order of several hundred legit sessions, or an
    # attack set that separates cleanly enough that the exact operating
    # point stops mattering.
    vals = np.array(sorted(val_scores.values()))
    thr = float(vals.max()) * 1.5 if len(vals) else float("inf")
    n_fp = sum(1 for s in val_scores.values() if s > thr)
    fp_bound = 1.0 / max(len(vals), 1)

    np.savez(MODEL_FILE, med=med, mad=mad, mu=mu, cov_inv=cov_inv,
             keep_idx=np.array(keep_idx, dtype=np.int32),
             threshold=np.float64(thr),
             feature_names=np.array([FEATURE_NAMES[i] for i in keep_idx]),
             window_s=np.float64(WINDOW_S), stride_s=np.float64(STRIDE_S),
             fit_sessions=np.array(list(fit_per.keys())),
             val_sessions=np.array(list(val_per.keys())))

    print(f"features: {len(keep_idx)}  windows: {len(Xfit)} fit")
    print(f"parameters: {len(keep_idx)} mean + {len(keep_idx)**2} covariance "
          f"= {len(keep_idx) + len(keep_idx)**2}\n")
    print("held-out legit session scores (these are the false-positive risk):")
    for n, s in sorted(val_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {s:9.1f}  {n}")
    tail = vals.max() / max(np.median(vals), 1e-9)
    print(f"\nheld-out median {np.median(vals):.1f}  max {vals.max():.1f}  "
          f"tail {tail:.0f}x  (lower tail = normal is characterised more "
          f"tightly = more headroom to separate an attack)")
    print(f"threshold {thr:.1f} (held-out max x1.5) -> {n_fp}/{len(val_scores)} "
          f"held-out legit flagged")
    print(f"false-positive rate: BOUNDED at <{100 * fp_bound:.0f}% by the "
          f"{len(vals)}-session held-out split — NOT measured at <1%. "
          f"The threshold sits above the observed legit max by construction, "
          f"so the 0 above is not evidence.")
    print(f"fit-split max {max(fit_scores.values()):.1f} "
          f"(training data — not a false-positive estimate)")
    print(f"\nmodel -> {MODEL_FILE}")
    print("This is calibrated on legit data ONLY. It says nothing yet about "
          "whether real swaps exceed the threshold — a model can fit normal "
          "tightly and still fail to separate attacks. Record attacks with "
          "cv/labeling/record_attack.py, then `score` to measure detection.")


def cmd_score(args):
    if not os.path.isfile(MODEL_FILE):
        raise SystemExit(f"{MODEL_FILE} not found — run `fit` first")
    m = np.load(MODEL_FILE, allow_pickle=True)
    med, mad, mu, cov_inv = m["med"], m["mad"], m["mu"], m["cov_inv"]
    keep_idx = m["keep_idx"]
    thr = float(m["threshold"])
    names = [str(x) for x in m["feature_names"]]

    sessions = discover_sessions(args.root)
    if not sessions:
        raise SystemExit(f"no sessions with trajectory.npz under {args.root}")

    n_flag = n_att = n_att_flag = n_legit = n_legit_flag = 0
    for name in sessions:
        sdir = os.path.join(args.root, name)
        wins = session_windows(sdir)
        if not wins:
            print(f"{name}: no scoreable windows")
            continue
        X = np.stack([w[2] for w in wins])[:, keep_idx]
        Z = (X - med) / mad
        d = _mahal(Z, mu, cov_inv)
        k = int(np.argmax(d))
        score = float(d[k])
        flagged = score > thr

        meta = {}
        mpath = os.path.join(sdir, "attack.json")
        if os.path.isfile(mpath):
            with open(mpath) as f:
                meta = json.load(f)

        # per-feature attribution: which features drove the worst window
        z = Z[k] - mu
        contrib = z * (cov_inv @ z)
        top = np.argsort(-contrib)[:3]
        why = ", ".join(f"{names[i]}={X[k][i]:.3g}" for i in top)

        tag = ""
        if meta:
            is_att = meta.get("is_attack")
            tag = f"  [{meta.get('attack_type')}]"
            if is_att:
                n_att += 1
                n_att_flag += flagged
                marks = meta.get("marks") or []
                if flagged and marks:
                    w0, w1 = wins[k][0], wins[k][1]
                    hit = any(w0 <= mk["t"] <= w1 for mk in marks)
                    tag += "  AT-SWAP" if hit else "  (not at marked swap)"
            else:
                n_legit += 1
                n_legit_flag += flagged
        n_flag += flagged
        print(f"{'FLAG' if flagged else 'ok  '}  {score:9.1f}  {name}{tag}\n"
              f"        worst window {wins[k][0]:.1f}-{wins[k][1]:.1f}s: {why}")

    print(f"\nthreshold {thr:.1f}   flagged {n_flag}/{len(sessions)}")
    if n_att:
        print(f"attacks detected : {n_att_flag}/{n_att}")
    if n_legit:
        print(f"legit false-flags: {n_legit_flag}/{n_legit}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--root", required=True)
    f.add_argument("--val-frac", type=float, default=0.25)
    f.add_argument("--features", choices=sorted(FEATURE_SETS), default=DEFAULT_SET,
                   help=f"feature set (default {DEFAULT_SET}). See the "
                        "ZERO-INFLATION note in this module before changing: "
                        "the wider sets are ill-conditioned, but the choice "
                        "is not settled until attack data exists.")
    s = sub.add_parser("score")
    s.add_argument("--root", required=True)
    args = ap.parse_args()
    (cmd_fit if args.cmd == "fit" else cmd_score)(args)


if __name__ == "__main__":
    main()
