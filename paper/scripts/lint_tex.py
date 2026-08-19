"""lint_tex.py — catch the LaTeX mistakes a missing local TeX install hides."""

import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERB = re.compile(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", re.S)
# display math is legitimate _ territory; blank it out before linting
DISPLAY = re.compile(
    r"\\begin\{(equation|align|equation\*|align\*|gather)\}.*?"
    r"\\end\{(equation|align|equation\*|align\*|gather)\}", re.S)
MATH = re.compile(r"\$[^$]*\$")
# LaTeX reads these arguments as filenames, where _ is legal
FILEARG = re.compile(r"\\(input|includegraphics|bibliography)"
                     r"(\[[^\]]*\])?\{[^}]*\}")
OK_UNICODE = set("—–’‘“”×≈§±→↔·")


def main():
    files = sorted((ROOT / "sections").glob("*.tex")) + [ROOT / "main.tex"]
    issues = 0
    for p in files:
        txt = p.read_text(encoding="utf-8")
        body = DISPLAY.sub(lambda m: "\n" * m.group(0).count("\n"),
                           VERB.sub("", txt))
        for i, line in enumerate(body.split("\n"), 1):
            s = FILEARG.sub("", MATH.sub("", line))
            for m in re.finditer(r"(?<!\\)_", s):
                print(f"  {p.name}:{i} unescaped _ :: {line.strip()[:88]}")
                issues += 1
            # a bare % that is not a comment marker at line start and not
            # escaped is almost always a percent sign that should be \%
            for m in re.finditer(r"(?<![\\%])%", s):
                if m.start() == 0 or s[:m.start()].strip() == "":
                    continue
                print(f"  {p.name}:{i} bare % :: {line.strip()[:88]}")
                issues += 1
        for i, line in enumerate(txt.split("\n"), 1):
            for ch in set(line):
                if ord(ch) > 127 and ch not in OK_UNICODE:
                    print(f"  {p.name}:{i} non-ascii {ch!r} :: "
                          f"{line.strip()[:70]}")
                    issues += 1
    print(f"  {issues} issue(s)")


if __name__ == "__main__":
    main()
