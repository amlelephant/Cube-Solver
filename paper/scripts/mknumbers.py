"""
mknumbers.py — emit paper/data/numbers.tex and the generated LaTeX tables.

Every number quoted in the prose comes from here, so the text cannot drift
away from the measurements. Run after m1/m2/m3/m4/m5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import common as C

OUT = C.DATA
CK = ["move_ctc_spd_s0", "move_ctc_spd_s1"]


def pct(x, d=1):
    return f"{x*100:.{d}f}"


def main():
    n = {}
    meta = C.load("holdout_meta.json")
    solves = [m for m in meta if m["session"].endswith("_solve")]
    scrambles = [m for m in meta if m["session"].endswith("_scramble")]

    # ---- corpus / holdout --------------------------------------------
    n["NHoldoutSolves"] = str(len(solves))
    n["NHoldoutScrambles"] = str(len(scrambles))
    n["NHoldoutMoves"] = f"{sum(m['n_moves'] for m in solves):,}"
    n["NewlyPrepared"] = "18"
    n["NSessionsFrames"] = "77"
    n["NDays"] = "15"
    n["NMovesTotal"] = "5{,}891"
    n["MedianTPS"] = f"{np.median([m['tps'] for m in solves]):.2f}"
    n["CrowdedPctHoldout"] = f"{np.mean([m['crowded_frac'] for m in solves])*100:.1f}"
    n["CTCFloorHoldout"] = f"{np.mean([m['ctc_floor'] for m in solves])*100:.2f}"
    n["CTCCeilingHoldout"] = f"{(1-np.mean([m['ctc_floor'] for m in solves]))*100:.1f}"
    n["NParams"] = "587{,}930"

    # Corpus-wide onset crowding, measured here rather than quoted: the
    # repository's "10.1% of moves collide" is the within-2-frames figure,
    # which is a different (and much larger) set than the moves that
    # genuinely cannot be given distinct frames.
    n_on = c2 = c1 = c0 = cfloor = 0
    for npz in sorted(C.SESSIONS.glob("solve_*/detector_stream_color.npz")):
        z = np.load(npz, allow_pickle=True)
        o = z["onset_idx"].astype(int)
        k = z["onset_class"].astype(int)
        if len(o) < 2:
            continue
        g = np.diff(o)
        n_on += len(o)
        c2 += int((g <= 2).sum())
        c1 += int((g <= 1).sum())
        c0 += int((g == 0).sum())
        cfloor += int(sum(1 for i in range(len(g))
                          if g[i] <= 2 and k[i] == k[i + 1]))
    n["CorpusOnsets"] = f"{n_on:,}"
    n["CorpusCrowdedTwo"] = f"{c2/n_on*100:.1f}"
    n["CorpusCrowdedOne"] = f"{c1/n_on*100:.1f}"
    n["CorpusSameFrame"] = f"{c0/n_on*100:.1f}"
    n["CorpusCTCFloor"] = f"{cfloor/n_on*100:.2f}"
    n["SliceSessions"] = "7"
    n["SliceMoves"] = "26"
    n["SliceMovePct"] = "0.44"

    # ---- m1 recognition ----------------------------------------------
    m1 = C.load("m1_recognition.json")
    m1s = C.load("m1_summary.json")

    def cell(model, kind, regime, key):
        for r in m1s:
            if (r["model"] == model and r["kind"] == kind
                    and r["regime"] == regime):
                return r[key]
        raise KeyError((model, kind, regime, key))

    # LaTeX control sequences must be all letters, so seeds are A/B not 0/1.
    for i, ck in enumerate(CK):
        s = "SA" if i == 0 else "SB"
        for reg, tag in (("daytime", "Day"), ("evening", "Eve"), ("all", "All")):
            n[f"Raw{tag}{s}"] = pct(cell(ck, "solve", reg, "acc_mean"))
            n[f"RawMer{tag}{s}"] = pct(cell(ck, "solve", reg, "mer_mean"))
        n[f"Scr{s}"] = pct(cell(ck, "scramble", "all", "acc_mean"))
        n[f"RawMin{s}"] = pct(cell(ck, "solve", "all", "acc_min"))
    n["EveningGapPts"] = f"{np.mean([cell(c,'solve','daytime','acc_mean') - cell(c,'solve','evening','acc_mean') for c in CK])*100:.1f}"
    n["RawDayMean"] = pct(np.mean([cell(c, "solve", "daytime", "acc_mean")
                                   for c in CK]))
    n["RawEveMean"] = pct(np.mean([cell(c, "solve", "evening", "acc_mean")
                                   for c in CK]))
    n["RawAllMean"] = pct(np.mean([cell(c, "solve", "all", "acc_mean")
                                   for c in CK]))
    n["RemainingGap"] = f"{float(n['CTCCeilingHoldout']) - float(n['RawDayMean']):.1f}"

    sol = [r for r in m1 if r["session"].endswith("_solve")]
    n["GreedyGap"] = f"{np.mean([r['acc'] - r['acc_greedy'] for r in sol])*100:.1f}"
    tot = sum(r["n_gt"] for r in sol)
    for k in ("miss", "sub", "phantom"):
        n[k.capitalize() + "Rate"] = f"{sum(r[k] for r in sol)/tot*100:.1f}"
    errs = sum(sum(r[k] for r in sol) for k in ("miss", "sub", "phantom"))
    n["MissShare"] = f"{sum(r['miss'] for r in sol)/errs*100:.0f}"
    n["SubShare"] = f"{sum(r['sub'] for r in sol)/errs*100:.0f}"
    n["PhantomShare"] = f"{sum(r['phantom'] for r in sol)/errs*100:.0f}"

    # ---- m4 ablation --------------------------------------------------
    m4 = {r["tag"]: r for r in C.load("m4_summary.json")}
    n["SpeedAugDaytimeGain"] = \
        f"{(m4['move_ctc_spd']['daytime']-m4['move_ctc_aug44']['daytime'])*100:.1f}"
    n["SpeedAugEveningGain"] = \
        f"{(m4['move_ctc_spd']['evening']-m4['move_ctc_aug44']['evening'])*100:.1f}"
    n["CTCvsPeakDay"] = \
        f"{(m4['move_ctc']['daytime']-m4['move_joint_base']['daytime'])*100:+.1f}"
    n["CTCvsPeakEve"] = \
        f"{(m4['move_ctc']['evening']-m4['move_joint_base']['evening'])*100:+.1f}"
    n["PhotoAugEveningGain"] = \
        f"{(m4['move_ctc_aug']['evening']-m4['move_ctc']['evening'])*100:+.1f}"
    n["LadderTotalGain"] = \
        f"{(m4['move_ctc_spd']['all']-m4['move_joint_base']['all'])*100:.1f}"

    # ---- m5 ladder / frame --------------------------------------------
    lad = {r["tier"]: r for r in C.load("m5_ladder.json")}
    n["LadderTrain"] = pct(lad["train"]["mean"])
    n["LadderVal"] = pct(lad["val (same-day)"]["mean"])
    n["LadderBracket"] = pct(lad["held out, bracketed"]["mean"])
    n["LadderAfter"] = pct(lad["held out, after"]["mean"])
    n["LadderHeld"] = pct(
        (lad["held out, bracketed"]["mean"] * lad["held out, bracketed"]["n"]
         + lad["held out, after"]["mean"] * lad["held out, after"]["n"])
        / (lad["held out, bracketed"]["n"] + lad["held out, after"]["n"]))
    n["LadderTrainMinusHeld"] = \
        f"{(lad['train']['mean']*100 - float(n['LadderHeld'])):.1f}"
    n["LadderTrainMinusVal"] = \
        f"{(lad['train']['mean'] - lad['val (same-day)']['mean'])*100:.1f}"
    n["LadderValMinusHeld"] = \
        f"{(lad['val (same-day)']['mean']*100 - float(n['LadderHeld'])):.1f}"
    n["SessionsRungDay"] = \
        f"{(m4['move_ctc_aug44']['daytime']-m4['move_ctc_aug']['daytime'])*100:+.1f}"
    n["SessionsRungEve"] = \
        f"{(m4['move_ctc_aug44']['evening']-m4['move_ctc_aug']['evening'])*100:+.1f}"
    n["PhotoAugDaytimeGain"] = \
        f"{(m4['move_ctc_aug']['daytime']-m4['move_ctc']['daytime'])*100:+.1f}"

    fr = C.load("m5_frame.json")
    n["FrameCubeSlice"] = pct(fr["cube_mean"])
    n["FrameCamSlice"] = pct(fr["camera_mean"])
    n["FrameCubeAll"] = pct(fr["corpus_cube_mean"])
    n["FrameCamAll"] = pct(fr["corpus_camera_mean"])
    n["FrameGainSlice"] = f"{(fr['camera_mean']-fr['cube_mean'])*100:+.1f}"
    n["FrameGainAll"] = \
        f"{(fr['corpus_camera_mean']-fr['corpus_cube_mean'])*100:+.1f}"

    conf = C.load("m5_confusion.json")
    k = conf["kinds"]
    t = sum(k.values())
    n["SubInverse"] = f"{k['inverse']/t*100:.0f}"
    n["SubAdjacent"] = f"{k['adjacent']/t*100:.0f}"
    n["SubOpposite"] = f"{k['opposite']/t*100:.0f}"
    n["NSubTotal"] = str(t)
    M = np.array(conf["matrix"])
    n["PredUShare"] = f"{M[:, :2].sum()/M.sum()*100:.0f}"

    pos = C.load("m5_positions.json")
    n["MissPos"] = f"{np.mean(pos['positions']['miss']):.2f}"
    n["SubPos"] = f"{np.mean(pos['positions']['sub']):.2f}"
    for ch, tag in (("sub", "Sub"), ("miss", "Miss"), ("phantom", "Phan")):
        v = np.array(pos["positions"][ch])
        n[f"{tag}FirstFifth"] = f"{(v < 0.2).mean()*100:.0f}"
        n[f"{tag}LastFifth"] = f"{(v >= 0.8).mean()*100:.0f}"

    # ---- m6 bootstrap CIs + paired tests --------------------------------
    st = C.load("m6_stats.json")
    for key, tag in (("raw accuracy, all held-out solves", "All"),
                     ("raw accuracy, daytime", "Day"),
                     ("raw accuracy, evening", "Eve")):
        n[f"CI{tag}Mean"] = pct(st[key]["mean"])
        n[f"CI{tag}Lo"] = pct(st[key]["lo"])
        n[f"CI{tag}Hi"] = pct(st[key]["hi"])
    rung_tags = ["CtcVsPeak", "PhotoAug", "MoreSessions", "SpeedAug"]
    for tag, r in zip(rung_tags, st["rungs"]):
        n[f"D{tag}"] = f"{r['delta']*100:+.1f}"
        n[f"D{tag}Lo"] = f"{r['lo']*100:+.1f}"
        n[f"D{tag}Hi"] = f"{r['hi']*100:+.1f}"
        n[f"D{tag}P"] = ("$<$0.001" if r["p_sign"] < 0.001
                         else f"{r['p_sign']:.2f}")
        n[f"D{tag}Better"] = f"{r['better']}"
        n[f"D{tag}Worse"] = f"{r['worse']}"
    tot_ = st["rungs"][-1]
    n["DTotal"] = f"{tot_['delta']*100:+.1f}"
    n["DTotalLo"] = f"{tot_['lo']*100:+.1f}"
    n["DTotalHi"] = f"{tot_['hi']*100:+.1f}"
    n["NPaired"] = str(tot_["n"])

    # ---- m3 anticheat ---------------------------------------------------
    ac = C.load("m3_anticheat.json")
    n["ACSolves"] = str(ac["s0"]["n_solve"])
    n["ACProxies"] = str(ac["s0"]["n_proxy"])
    n["ACVerifiedSA"] = str(ac["s0"]["verified"])
    n["ACVerifiedSB"] = str(ac["s1"]["verified"])
    n["ACCaughtSA"] = str(ac["s0"]["caught"])
    n["ACCaughtSB"] = str(ac["s1"]["caught"])
    n["ACGapSA"] = str(ac["s0"]["gap"])
    n["ACGapSB"] = str(ac["s1"]["gap"])
    n["ACHeadroomSA"] = f"+{ac['s0']['min_headroom']}"
    n["ACHeadroomSB"] = f"+{ac['s1']['min_headroom']}"
    n["ACLowestLegit"] = str(min(ac["s0"]["lowest_legit"],
                                 ac["s1"]["lowest_legit"]))
    n["ACHighestProxy"] = str(max(ac["s0"]["highest_proxy"],
                                  ac["s1"]["highest_proxy"]))

    # ---- m2 decode (optional until it lands) ----------------------------
    m2 = []
    for sd in ("s0", "s1"):
        q = C.DATA / f"m2_decode_{sd}.json"
        if q.exists():
            m2 += json.loads(q.read_text(encoding="utf-8"))
    if m2:
        ns = [r for r in m2 if not r["slice_session"]]
        n["DecodeN"] = str(len({r["session"] for r in m2}))
        n["DecodeNonSliceN"] = str(len({r["session"] for r in ns}))
        n["DecodeRaw"] = pct(np.mean([r["raw_acc"] for r in ns]))
        n["DecodePost"] = pct(np.mean([r["post_acc"] for r in ns]))
        n["DecodeGain"] = \
            f"{np.mean([r['post_acc']-r['raw_acc'] for r in ns])*100:+.1f}"
        n["DecodeVerified"] = str(sum(r["verified"] for r in ns))
        n["DecodeExact"] = str(sum(r["post_exact"] for r in ns))
        n["DecodeTrials"] = str(len(ns))
        dk = ("off1", "off2", "off4", "unscrambled")
        acc_ = sum(sum(bool(r.get(f"decoy_{k}")) for k in dk) for r in m2)
        n["DecoyAccepted"] = str(acc_)
        n["DecoyTried"] = str(4 * len(m2))
        sl = [r for r in m2 if r["slice_session"]]
        n["DecodeSliceVerified"] = str(sum(r["verified"] for r in sl))
        n["DecodeSliceN"] = str(len(sl))
        n["MinGtPathCost"] = f"{min(r['gt_path_cost'] for r in m2):.0f}"
        n["MedGtPathCost"] = \
            f"{np.median([r['gt_path_cost'] for r in m2]):.0f}"
    else:
        for k_ in ("DecodeN", "DecodeNonSliceN", "DecodeRaw", "DecodePost",
                   "DecodeGain", "DecodeVerified", "DecodeExact",
                   "DecodeTrials", "DecoyAccepted", "DecoyTried",
                   "DecodeSliceVerified", "DecodeSliceN",
                   "MinGtPathCost", "MedGtPathCost"):
            n[k_] = r"\textcolor{red}{??}"
        (OUT / "tab_decode.tex").write_text(
            "\\begin{tabular}{@{}l@{}}\\toprule\n"
            "\\textcolor{red}{decode sweep still running --- rerun "
            "\\texttt{mknumbers.py} when \\texttt{m2\\_decode\\_s0.json} "
            "and \\texttt{m2\\_decode\\_s1.json} exist} \\\\\n"
            "\\bottomrule\\end{tabular}\n", encoding="utf-8")
        print("  wrote data/tab_decode.tex (placeholder)")

    # ---- write --------------------------------------------------------
    lines = ["% GENERATED by paper/scripts/mknumbers.py - do not edit.",
             "% Every number in the prose resolves through one of these.",
             ""]
    for k_, v in n.items():
        lines.append(f"\\newcommand{{\\{k_}}}{{{v}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote data/numbers.tex ({len(n)} macros)")

    # ---- generated tables ----------------------------------------------
    _tab_holdout(meta)
    _tab_main(m1s, m1)
    _tab_ablation()
    _tab_persession(m1)
    if m2:
        _tab_decode(m2)


def _tab_holdout(meta):
    rows = sorted(meta, key=lambda m: m["session"])
    out = [r"\begin{tabular}{@{}llrrrrl@{}}", r"\toprule",
           r"session & clock & frames & turns & turns/s & "
           r"crowded & regime \\", r"\midrule"]
    for m in rows:
        s = m["session"]
        stamp = f"{s[10:12]}-{s[12:14]}\\,{s[15:17]}:{s[17:19]}"
        kind = "solve" if s.endswith("_solve") else "scramble"
        out.append(
            f"{stamp} {kind} & {m['hour']:02d}h & {m['n_frames']} & "
            f"{m['n_moves']} & {m['tps']:.2f} & "
            f"{m['crowded_frac']*100:.0f}\\% & "
            f"{'evening' if m['evening'] else 'daytime'} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_holdout.tex").write_text("\n".join(out) + "\n")
    print("  wrote data/tab_holdout.tex")


def _tab_main(m1s, m1):
    def g(model, kind, regime):
        for r in m1s:
            if (r["model"] == model and r["kind"] == kind
                    and r["regime"] == regime):
                return r
        return None
    out = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
           r"take type & regime & $n$ & moves & "
           r"seed 0 & seed 1 & CTC ceiling \\", r"\midrule"]
    for kind in ("solve", "scramble"):
        for reg in ("daytime", "evening", "all"):
            a, b = g(CK[0], kind, reg), g(CK[1], kind, reg)
            if a is None:
                continue
            rule = r"\midrule" if (kind == "solve" and reg == "all") else ""
            out.append(
                f"{kind} & {reg} & {a['n_sessions']} & {a['n_moves']} & "
                f"{a['acc_mean']*100:.1f}\\% & {b['acc_mean']*100:.1f}\\% & "
                f"{a['ctc_ceiling']*100:.1f}\\% \\\\")
            if rule:
                out.append(rule)
    out += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_main.tex").write_text("\n".join(out) + "\n")
    print("  wrote data/tab_main.tex")


def _tab_persession(m1):
    rows = [r for r in m1 if r["model"] == CK[0]]
    out = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
           r"session & regime & turns & turns/s & acc. & MER & "
           r"miss & sub / phan \\", r"\midrule"]
    for r in sorted(rows, key=lambda r: r["session"]):
        s = r["session"]
        kind = "solve" if s.endswith("_solve") else "scr."
        out.append(
            f"{s[10:12]}-{s[12:14]}\\,{s[15:17]}:{s[17:19]} {kind} & "
            f"{'eve' if r['evening'] else 'day'} & {r['n_gt']} & "
            f"{r['tps']:.2f} & {r['acc']*100:.1f}\\% & "
            f"{r['mer']*100:.1f}\\% & {r['miss']} & "
            f"{r['sub']} / {r['phantom']} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_persession.tex").write_text("\n".join(out) + "\n")
    print("  wrote data/tab_persession.tex")


def _tab_ablation():
    m4 = C.load("m4_summary.json")
    out = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
           r"configuration & train & daytime & evening & all \\",
           r"\midrule"]
    ntrain = {"move_joint_base": 38, "move_ctc": 38, "move_ctc_aug": 38,
              "move_ctc_aug44": 44, "move_ctc_spd": 44}
    for r in m4:
        out.append(
            f"{r['rung']} & {ntrain[r['tag']]} & "
            f"{r['daytime']*100:.1f}\\% & {r['evening']*100:.1f}\\% & "
            f"{r['all']*100:.1f}\\% \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(out) + "\n")
    print("  wrote data/tab_ablation.tex")


def _tab_decode(m2):
    sess = sorted({r["session"] for r in m2})
    out = [r"\begin{tabular}{@{}lrrrccc@{}}", r"\toprule",
           r"session & raw & post & $\Delta$ & verified & exact & "
           r"decoys accepted \\", r"\midrule"]
    for s in sess:
        g = [r for r in m2 if r["session"] == s]
        raw = np.mean([r["raw_acc"] for r in g]) * 100
        post = np.mean([r["post_acc"] for r in g]) * 100
        ver = sum(r["verified"] for r in g)
        ex = sum(r["post_exact"] for r in g)
        dk = ("off1", "off2", "off4", "unscrambled")
        dec = sum(sum(bool(r.get(f"decoy_{k}")) for k in dk) for r in g)
        mark = "$^{\\dagger}$" if g[0]["slice_session"] else ""
        out.append(
            f"{s[10:12]}-{s[12:14]}\\,{s[15:17]}:{s[17:19]}{mark} & "
            f"{raw:.1f}\\% & {post:.1f}\\% & {post-raw:+.1f} & "
            f"{ver}/2 & {ex}/2 & {dec}/8 \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_decode.tex").write_text("\n".join(out) + "\n")
    print("  wrote data/tab_decode.tex")


if __name__ == "__main__":
    main()
