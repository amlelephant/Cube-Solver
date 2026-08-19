"""
check_tex.py — a cheap pre-flight for the LaTeX, since there is no local
TeX install. Catches the two mistakes that actually happen: a number macro
used in the prose but never generated, and unbalanced environments.
"""

import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

KNOWN = set("""
section subsection subsubsection paragraph label ref cref Cref cite
begin end item textbf textit emph texttt small footnotesize file wca
includegraphics caption centering input newcommand toprule midrule bottomrule
bibliographystyle bibliography appendix tableofcontents clearpage today
linewidth hfill vspace vfill noindent par color textcolor sim times le ge
approx cdot arg max min sum prod frac tfrac mathbb mathcal mathrm dagger
varnothing pi ell alpha beta rho lambda delta sigma leftmargin itemsep style
nextline hspace usebox savebox lrbox colorbox minipage dimexpr relax
rightarrow leftarrow leq geq neq quad qquad text binom in notin cup cap
newtheorem theoremstyle definecolor captionsetup titleformat normalfont
Large large bfseries pagestyle fancyhf lhead rhead renewcommand headrulewidth
documentclass usepackage usetikzlibrary newenvironment Huge verb url textsc
mid setminus circ ldots dots keybox description enumerate itemize tabular
table figure equation align verbatim center titlepage minipage document
Needleman Wunsch Levenshtein Bluetooth Kociemba Roux Gaussian Bonferroni
CFOP OLL PLL ROC MER QTM HTM BLE CNN TCN CTC RGB CUDA GPU YOLOv OpenCV
PyTorch Python Windows NVIDIA RTX JPEG Wunsch McNemar UI Left Right Front
Back Down Up
""".split())


def main():
    numbers = ROOT / "data" / "numbers.tex"
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                             numbers.read_text(encoding="utf-8")))
    files = sorted((ROOT / "sections").glob("*.tex")) + [ROOT / "main.tex"]

    missing, envs_bad = {}, []
    for p in files:
        txt = p.read_text(encoding="utf-8")
        for m in re.findall(r"\\([A-Za-z]+)", txt):
            if m in defined or m in KNOWN:
                continue
            # a generated-number macro is CamelCase with an inner capital
            if m[0].isupper() and any(c.isupper() for c in m[1:]):
                missing.setdefault(m, set()).add(p.name)
        opens = re.findall(r"\\begin\{([a-zA-Z*]+)\}", txt)
        closes = re.findall(r"\\end\{([a-zA-Z*]+)\}", txt)
        for e in set(opens) | set(closes):
            if opens.count(e) != closes.count(e):
                envs_bad.append(f"{p.name}: {e} "
                                f"({opens.count(e)} begin, "
                                f"{closes.count(e)} end)")

    print(f"  {len(defined)} macros defined in data/numbers.tex")
    if missing:
        print("  UNDEFINED number-like macros:")
        for m, w in sorted(missing.items()):
            print(f"    \\{m:<26} used in {sorted(w)}")
    else:
        print("  no undefined number-like macros")
    if envs_bad:
        print("  UNBALANCED environments:")
        for e in envs_bad:
            print("    " + e)
    else:
        print("  all environments balanced")

    # figures referenced vs present
    figs = set()
    for p in files:
        figs |= set(re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}",
                               p.read_text(encoding="utf-8")))
        figs |= {f"{m}.tex" for m in
                 re.findall(r"\\input\{(figures/[^}]+)\}",
                            p.read_text(encoding="utf-8"))}
    absent = [f for f in sorted(figs) if not (ROOT / f).exists()]
    print(f"  {len(figs)} figures referenced; "
          + (f"MISSING: {absent}" if absent else "all present"))

    # cross-references
    labels, refs = set(), {}
    for p in files:
        t = p.read_text(encoding="utf-8")
        labels |= set(re.findall(r"\\label\{([^}]+)\}", t))
        for r in re.findall(r"\\[cC]ref\{([^}]+)\}", t):
            for one in r.split(","):
                refs.setdefault(one.strip(), set()).add(p.name)
    dangling = {k: v for k, v in refs.items() if k not in labels}
    if dangling:
        print("  DANGLING cross-references:")
        for k, v in sorted(dangling.items()):
            print(f"    {k:<28} referenced in {sorted(v)}")
    else:
        print(f"  {len(labels)} labels, all cross-references resolve")

    tabs = set()
    for p in files:
        tabs |= set(re.findall(r"\\input\{(data/[^}]+)\}",
                               p.read_text(encoding="utf-8")))
    absent = [t for t in sorted(tabs)
              if not (ROOT / (t + ".tex")).exists()
              and not (ROOT / t).exists()]
    print(f"  {len(tabs)} generated inputs; "
          + (f"MISSING: {absent}" if absent else "all present"))


if __name__ == "__main__":
    main()
