"""
Deterministic tests for the continuity-guard state machine (ROADMAP §2.1).
No model, no camera: feeds synthetic (t, boxes) streams at 10 Hz.

Severity model (two-tier, 2026-07-17): verdict is pass or dq — NO review
tier. Every swap-indicative signal is a hard DQ: border-zone gaps/teleports,
appearance change across any gap (checked everywhere), far reappearance
after any gap (2026-07-18 — checked everywhere too: a mid-frame table edge
is an interior exit route), and pipeline stalls. Only genuinely benign
interior events (table slams under continuous detection, brief in-place
interior occlusion, isolated background ghosts) are forensic notes that
pass.

Run from inside cv/:  python test_continuity_guard.py
"""

from continuity_guard import ContinuityGuard, dedup_boxes, _sig_dist

W, H = 1280, 720
CENTER = (500, 260, 780, 540)      # a centered ~280px cube box
SMALL = (560, 320, 720, 480)       # a smaller centered box (~160px)
FAR = (60, 60, 300, 300)           # a second cube, interior (clears margins)
EDGE = (0, 300, 180, 480)          # box in the border zone (center x=90 <
                                   # the 153px margin)
LFACE = (500, 260, 645, 540)       # left visible face of the CENTER cube...
RFACE = (650, 260, 780, 540)       # ...and its right face (abutting boxes)
TWIN_A = (400, 260, 680, 540)      # two real cubes held side-by-side,
TWIN_B = (685, 260, 965, 540)      # touching (union ~2:1 — must NOT merge)
DIAG_A = (400, 200, 600, 400)      # two real cubes held diagonally: square
DIAG_B = (610, 410, 810, 610)      # union (aspect-sane!) but the boxes only
                                   # cover ~48% of it — must NOT merge
HZ = 10                            # simulated analysis rate

# synthetic appearance signatures: orthogonal histograms, distance 1.0
SIG_A = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
SIG_B = [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]
assert _sig_dist(SIG_A, SIG_A) < 1e-9
assert abs(_sig_dist(SIG_A, SIG_B) - 1.0) < 1e-9
# ...and a moderately-different pair, distance ~0.57 — the motion-blur band
# (between SIG_DIST 0.45 and SIG_DIST_INTERIOR 0.65)
SIG_P = [1.0, 0.0]
SIG_Q = [0.45, 0.55]
assert abs(_sig_dist(SIG_P, SIG_Q) - 0.574) < 0.01


def run(stream):
    """stream: list of boxes-lists, one per tick at HZ."""
    g = ContinuityGuard(W, H)
    for i, boxes in enumerate(stream):
        g.update(i / HZ, boxes)
    return g.report()


def run_sig(stream):
    """stream: list of (boxes, sig) pairs, one per tick at HZ."""
    g = ContinuityGuard(W, H)
    for i, (boxes, sig) in enumerate(stream):
        g.update(i / HZ, boxes, sig)
    return g.report()


def box(b, conf=0.9):
    return (*b, conf)


def has(rep, typ):
    return any(e["type"] == typ for e in rep["events"])


PASS_COUNT = [0, 0]  # passed, failed


def check(name, cond, rep):
    ok = "PASS" if cond else "FAIL"
    PASS_COUNT[0 if cond else 1] += 1
    print(f"[{ok}] {name}  -> verdict={rep['verdict']} dq={rep['dq_reasons']} "
          f"notes={rep['notes']}")


# 1. Legit solve: steady detection with occlusion flicker (0.4s gaps) → pass
stream = []
for i in range(100):
    # two 0.4s flicker gaps — normal hand occlusion, under GAP_FLAG_S
    if 30 <= i < 33 or 70 <= i < 73:
        stream.append([])
    else:
        stream.append([box(CENTER)])
rep = run(stream)
check("legit solve w/ 0.4s flicker gaps passes", rep["verdict"] == "pass", rep)

# 2. Cube absent 1.5s → DQ (presence_gap)
stream = [[box(CENTER)]] * 30 + [[]] * 15 + [[box(CENTER)]] * 30
rep = run(stream)
check("1.5s absence DQs", rep["verdict"] == "dq"
      and "presence_gap" in rep["dq_reasons"], rep)

# 3. Two cubes sustained (5 multi frames = 0.5s) → DQ (two_cubes)
stream = [[box(CENTER)]] * 20 + [[box(CENTER), box(FAR)]] * 5 + [[box(CENTER)]] * 20
rep = run(stream)
check("sustained two cubes DQs", rep["verdict"] == "dq"
      and "two_cubes" in rep["dq_reasons"], rep)

# 4. Single-frame double detection (ghost) → note only: pass, NOT review/dq
stream = [[box(CENTER)]] * 20 + [[box(CENTER), box(FAR)]] + [[box(CENTER)]] * 20
rep = run(stream)
check("isolated multi frame is a note, passes", rep["verdict"] == "pass"
      and not rep["dq_reasons"] and has(rep, "note_multi_frame"), rep)

# 5. Two multi frames inside the window (below the 3-hit DQ bar) → forensic
#    note only, passes (no review tier; a rare 2-frame background ghost must
#    not DQ a good player). A real swap sustains past MULTI_DQ_HITS.
stream = [[box(CENTER)]] * 20 + [[box(CENTER), box(FAR)]] * 2 + [[box(CENTER)]] * 20
rep = run(stream)
check("two multi frames in window are a note, pass", rep["verdict"] == "pass"
      and not rep["dq_reasons"] and has(rep, "note_multi_frame"), rep)

# 6. Two multi frames far apart in time (outside 1s window) → notes, pass
stream = ([[box(CENTER)]] * 10 + [[box(CENTER), box(FAR)]]
          + [[box(CENTER)]] * 20 + [[box(CENTER), box(FAR)]]
          + [[box(CENTER)]] * 10)
rep = run(stream)
check("multi frames outside window pass", rep["verdict"] == "pass"
      and not rep["dq_reasons"], rep)

# 7. Rare static background FP (light-up keyboard, ~1 ghost frame / 100) → pass
stream = [[box(CENTER), box(FAR)] if i % 100 == 99 else [box(CENTER)]
          for i in range(500)]
rep = run(stream)
check("rare background ghost (1/100 frames) passes", rep["verdict"] == "pass"
      and not rep["dq_reasons"], rep)

# 8. YOLO duplicate box on one cube (high IoU) → one cube, pass
dup = (510, 270, 790, 550)  # overlaps CENTER heavily
stream = [[box(CENTER), box(dup, 0.7)]] * 40
rep = run(stream)
check("overlapping duplicate boxes are one cube", rep["verdict"] == "pass", rep)
assert len(dedup_boxes([box(CENTER), box(dup, 0.7)])) == 1

# 9. Face split: resting cube corner-on, one box per visible face, sustained
#    → merged into one cube, pass
stream = [[box(LFACE), box(RFACE, 0.8)]] * 40
rep = run(stream)
check("abutting face-split boxes merge to one cube", rep["verdict"] == "pass"
      and not rep["dq_reasons"], rep)
assert len(dedup_boxes([box(LFACE), box(RFACE, 0.8)])) == 1

# 10. Two real cubes held side-by-side, touching → union too wide to merge,
#     still two cubes → DQ when sustained
stream = [[box(CENTER)]] * 20 + [[box(TWIN_A), box(TWIN_B)]] * 5 + [[box(TWIN_A)]] * 20
rep = run(stream)
check("side-by-side touching cubes still DQ", rep["verdict"] == "dq"
      and "two_cubes" in rep["dq_reasons"], rep)
assert len(dedup_boxes([box(TWIN_A), box(TWIN_B)])) == 2

# 11. Low-confidence second box (below MULTI_CONF) → not a second cube
stream = [[box(CENTER), box(FAR, 0.4)]] * 40
rep = run(stream)
check("low-conf second box ignored for uniqueness",
      rep["verdict"] == "pass", rep)

# 12. INTERIOR teleport (the table slam): violent jump, both ends away from
#     the border → benign note, verdict pass
stream = [[box(SMALL)]] * 20 + [[box(FAR)]] * 20
rep = run(stream)
check("interior teleport (slam) is a note, passes", rep["verdict"] == "pass"
      and not rep["dq_reasons"] and has(rep, "note_teleport"), rep)

# 13. BORDER teleport: same jump but into the border zone → hard DQ
stream = [[box(SMALL)]] * 20 + [[box(EDGE)]] * 20
rep = run(stream)
check("border teleport DQs", rep["verdict"] == "dq"
      and "teleport" in rep["dq_reasons"], rep)

# 14. Edge exit with a flicker gap into the border zone → hard DQ (edge_gap).
#     (Was a review-only flag before 2026-07-17; the tightened GAP_FLAG_S
#     0.30 now catches this short a side-swap gap.)
stream = [[box(CENTER)]] * 20 + [[box(EDGE)]] * 3 + [[]] * 3 + [[box(EDGE)]] * 20
rep = run(stream)
check("short border edge-exit gap DQs", rep["verdict"] == "dq"
      and "edge_gap" in rep["dq_reasons"], rep)

# 15. The swap route: 0.5s gap that starts/ends in the border zone → hard DQ
#     (was review before 2026-07-16 — an easy manual swap in live testing)
stream = [[box(CENTER)]] * 20 + [[box(EDGE)]] * 2 + [[]] * 4 + [[box(EDGE)]] * 20
rep = run(stream)
check("border-zone 0.5s gap DQs (edge_gap)", rep["verdict"] == "dq"
      and "edge_gap" in rep["dq_reasons"], rep)

# 16. Same-length gap fully interior, comes back IN PLACE → benign note,
#     pass (0.4s — in-place occlusion is the only interior leniency left)
stream = [[box(CENTER)]] * 20 + [[]] * 3 + [[box(CENTER)]] * 20
rep = run(stream)
check("interior in-place 0.4s gap is a note, passes", rep["verdict"] == "pass"
      and not rep["dq_reasons"] and has(rep, "note_gap"), rep)

# 17. 0.8s gap anywhere → DQ (presence_gap)
stream = [[box(CENTER)]] * 20 + [[]] * 7 + [[box(CENTER)]] * 20
rep = run(stream)
check("0.8s gap DQs", rep["verdict"] == "dq"
      and "presence_gap" in rep["dq_reasons"], rep)

# 18. Interior far reappearance after a 0.3s gap → hard DQ (reappear_jump),
#     even with no border involvement and no signatures. This is the
#     under-the-table swap (2026-07-18): a mid-frame table edge is an
#     interior exit route — dip below it, swap, raise the other cube
#     somewhere else. Was a benign note before; a legit occluded cube
#     reappears where it was.
stream = [[box(CENTER)]] * 20 + [[]] * 2 + [[box(FAR)]] * 20
rep = run(stream)
check("interior far reappearance after gap DQs (table swap)",
      rep["verdict"] == "dq" and "reappear_jump" in rep["dq_reasons"], rep)

# 19. Comes back in place but LOOKING different (scrambled → solved face)
#     → hard DQ (appearance_change); appearance is checked everywhere because
#     a palmed swap never touches the border. (Was review before 2026-07-17.)
stream = ([([box(CENTER)], SIG_A)] * 20 + [([], None)] * 4
          + [([box(CENTER)], SIG_B)] * 20)
rep = run_sig(stream)
check("appearance change across gap DQs",
      rep["verdict"] == "dq" and "appearance_change" in rep["dq_reasons"], rep)

# 20. Interior jump + appearance change: even with no border involvement the
#     appearance change alone hard-DQs now (two-tier: an in-place palmed swap
#     must not need a human to catch). This is the deliberate strictness
#     tradeoff — a legit solver must keep the cube visible across occlusions.
stream = ([([box(CENTER)], SIG_A)] * 20 + [([], None)] * 4
          + [([box(FAR)], SIG_B)] * 20)
rep = run_sig(stream)
check("interior appearance change now DQs",
      rep["verdict"] == "dq" and "appearance_change" in rep["dq_reasons"], rep)

# 21. The full border swap: gap via border zone + far reappearance + looks
#     different → DQ with the swap fingerprint detail
stream = ([([box(CENTER)], SIG_A)] * 20 + [([], None)] * 4
          + [([box(EDGE)], SIG_B)] * 20)
rep = run_sig(stream)
check("border swap DQs with swap_reappearance", rep["verdict"] == "dq"
      and "edge_gap" in rep["dq_reasons"]
      and "swap_reappearance" in rep["dq_reasons"], rep)

# 22. No cube ever appears → DQ (never_seen)
rep = run([[]] * 30)
check("cube never seen DQs", rep["verdict"] == "dq"
      and "never_seen" in rep["dq_reasons"], rep)

# 23. CADENCE — slow analysis (0.35s between frames, cube present the whole
#     time, interior, in place). Below the presence/gap checks' horizon, so
#     the old guard passed it silently; a fast swap fits in 0.35s → hard DQ
#     (single cutoff, no review tier).
g = ContinuityGuard(W, H)
for i in range(20):
    g.update(i * 0.35, [box(CENTER)])
rep = g.report()
check("slow analysis cadence DQs", rep["verdict"] == "dq"
      and "analysis_stall" in rep["dq_reasons"], rep)

# 24. CADENCE — egregious stall (0.7s between two frames) → hard DQ; the
#     pipeline was too slow to rule out a swap between samples.
g = ContinuityGuard(W, H)
g.update(0.0, [box(CENTER)])
g.update(0.1, [box(CENTER)])
g.update(0.8, [box(CENTER)])   # 0.7s stall
rep = g.report()
check("egregious analysis stall DQs", rep["verdict"] == "dq"
      and "analysis_stall" in rep["dq_reasons"], rep)

# 25. CADENCE — pre-cube frames are exempt: a slow interval BEFORE any cube
#     is tracked must not raise a stall (real monitoring hasn't started; the
#     live wiring also warms the model up before timing). Cube then arrives
#     at a healthy cadence → clean pass. (Note: an interval this slow with a
#     cube ALREADY tracked would flag — that's tests 23/24.)
g = ContinuityGuard(W, H)
g.update(0.0, [])          # nothing yet
g.update(0.4, [])          # 0.4s slow pre-cube frame (under never_seen 0.5s)
for i in range(20):
    g.update(0.4 + i * 0.05, [box(CENTER)])
rep = g.report()
check("pre-cube slow frame does not stall", rep["verdict"] == "pass"
      and not rep["dq_reasons"] and not has(rep, "flag_analysis_stall"), rep)

# 26. Two real cubes held DIAGONALLY: their union is square-ish (passes the
#     aspect gate — this merged and dodged the DQ before 2026-07-18) but the
#     boxes cover only ~48% of it → NOT merged, sustained → DQ (two_cubes)
stream = [[box(CENTER)]] * 20 + [[box(DIAG_A), box(DIAG_B)]] * 5 + [[box(DIAG_A)]] * 20
rep = run(stream)
check("diagonal touching cubes don't merge, DQ", rep["verdict"] == "dq"
      and "two_cubes" in rep["dq_reasons"], rep)
assert len(dedup_boxes([box(DIAG_A), box(DIAG_B)])) == 2

# 27. INTERIOR in-place reappearance with a moderate appearance shift
#     (~0.57 — motion blur from a fast center-frame move, not a different
#     cube) → note, pass (this band was false-DQing real solves)
stream = ([([box(CENTER)], SIG_P)] * 20 + [([], None)] * 2
          + [([box(CENTER)], SIG_Q)] * 20)
rep = run_sig(stream)
check("interior moderate appearance shift is a note, passes",
      rep["verdict"] == "pass" and not rep["dq_reasons"]
      and has(rep, "note_appearance_change"), rep)

# 28. The same moderate shift across a BORDER gap keeps the tight bar → DQ
stream = ([([box(CENTER)], SIG_P)] * 20 + [([], None)] * 2
          + [([box(EDGE)], SIG_Q)] * 20)
rep = run_sig(stream)
check("border moderate appearance shift still DQs",
      rep["verdict"] == "dq" and "appearance_change" in rep["dq_reasons"], rep)

# 29. A REAL palmed swap in the interior (orthogonal sticker layout, distance
#     1.0, back in place so reappear_jump can't be the catch) still clears
#     the laxer interior bar → DQ (guards test 20's intent)
stream = ([([box(CENTER)], SIG_A)] * 20 + [([], None)] * 2
          + [([box(CENTER)], SIG_B)] * 20)
rep = run_sig(stream)
check("interior full appearance change still DQs",
      rep["verdict"] == "dq" and "appearance_change" in rep["dq_reasons"], rep)

print(f"\n{PASS_COUNT[0]} passed, {PASS_COUNT[1]} failed")
raise SystemExit(1 if PASS_COUNT[1] else 0)
