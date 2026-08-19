# `paper/` — the study report

A ground-up technical report on the move-recognition pipeline, written to be
read rather than skimmed, with every number regenerated from scratch on a
strictly-defined never-seen holdout.

## Getting it into Overleaf

Zip this whole folder and upload it as a new project (Overleaf: *New Project
→ Upload Project*). Set the main document to `main.tex`, compiler
**pdfLaTeX**. Everything it needs is here; no internet packages beyond a
standard TeX Live install.

    paper/
      main.tex              preamble, palette, section includes
      sections/*.tex        the prose
      figures/*.pdf         generated figures (+ .png previews)
      figures/tikz_pipeline.tex   the architecture diagram, drawn in TikZ
      data/numbers.tex      every measured number, as \newcommand macros
      data/tab_*.tex        generated tables
      refs.bib
      scripts/*.py          the measurement and figure code

## The one rule

No number in the report may come from a session any evaluated checkpoint has
seen. `scripts/common.py::holdout()` derives the evaluable set from each
checkpoint's own recorded `train_session_names` and `val_session_names`,
intersected across every checkpoint being compared. No script accepts a
hand-written session list — that is exactly how this project previously
reported memorisation as generalisation.

Second rule: ground truth is the **camera-frame** move name
(`camera_notation`), not the cube-frame name the smart cube reports. See
§3.3 of the report.

## Regenerating everything

Run from inside `ble/move_detector/` (repo convention — scripts resolve
models and data by bare relative name):

```bash
cd ble/move_detector
PY=../../.venv/Scripts/python.exe
S=../../paper/scripts

$PY $S/m1_recognition.py     # raw CTC, both seeds; caches posteriorgrams
$PY $S/m4_ablation.py        # the five-rung checkpoint ladder
$PY $S/m5_diagnostics.py     # proximity ladder, truth frame, confusion
$PY anticheat_gate.py score --ctc checkpoints/move_ctc_spd_s0.pt
$PY anticheat_gate.py score --ctc checkpoints/move_ctc_spd_s1.pt

# slow (~2 h/seed) — run the two seeds in parallel
$PY $S/m2_decode.py --only move_ctc_spd_s0 --out m2_decode_s0.json &
$PY $S/m2_decode.py --only move_ctc_spd_s1 --out m2_decode_s1.json &

$PY $S/mknumbers.py          # -> data/numbers.tex and data/tab_*.tex
$PY $S/f1_data_and_model.py
$PY $S/f2_results.py
$PY $S/f3_decode.py
```

Then, from `paper/`:

```bash
python scripts/check_tex.py  # undefined macros, unbalanced envs, missing figs
python scripts/lint_tex.py   # unescaped _, stray %, non-ASCII
```

Both are pre-flights for the fact that there is no local TeX install; they
catch what actually goes wrong.

## What each measurement script produces

| script | writes | answers |
|---|---|---|
| `m1_recognition.py` | `m1_recognition.json`, `m1_summary.json`, `post/*.npz` | raw per-move accuracy, MER, error channels, greedy vs beam |
| `m2_decode.py` | `m2_decode_s{0,1}.json` | post-decode accuracy, verification, and the decoy sweep |
| `m4_ablation.py` | `m4_ablation.json`, `m4_summary.json` | what each training change was worth, on one holdout |
| `m5_diagnostics.py` | `m5_ladder/frame/confusion/positions.json` | memorisation vs generalisation, label frame, error taxonomy |
| `anticheat_gate.py score` | `anticheat_gate_*.json` | the legitimacy verdict on legit solves and proxy attacks |

## Note on the data

`data/post/` holds cached posteriorgrams so the figure and decode scripts do
not re-run the network. Delete it to force a recompute (`m1_recognition.py
--refresh`). It is regenerable and need not be uploaded to Overleaf.
