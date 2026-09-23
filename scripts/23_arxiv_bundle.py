"""Package the manuscript as an arXiv submission.

arXiv compiles the LaTeX source itself, so the upload has to be self-contained apart from what its
TeX Live already provides. IEEEtran, booktabs, hyperref and the rest are installed there; the bundle
therefore carries only this paper's own files: `main.tex`, the two generated table files, the figures
it actually includes, and `main.bbl`.

`main.bbl` matters. arXiv runs LaTeX but not BibTeX, so without the compiled bibliography every
citation comes out as a question mark. `refs.bib` is deliberately left out: it is not read, and
including it invites the impression that it was.

Figures are taken from the `\\includegraphics` calls rather than from the directory listing, because
`paper/figures/` also holds superseded plots (`fig_gap`, `fig_specificity`) that the manuscript no
longer draws and that would silently inflate the upload.

    python scripts/23_arxiv_bundle.py          # build dist/arxiv-<date>.tar.gz
    python scripts/23_arxiv_bundle.py --check  # report what would go in, build nothing
"""

import argparse
import re
import sys
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PAPER = PROJECT_ROOT / "paper"
# Everything else main.tex needs is a package arXiv already has.
SOURCES = ["main.tex", "main.bbl", "cpu_table.tex", "exploratory_tables.tex"]


def included_figures(tex: str) -> list[str]:
    """The figure files `main.tex` draws, in the order it draws them."""
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    # `\graphicspath{{figures/}}` and the omitted extension are resolved by LaTeX, not by the name.
    return [name if Path(name).suffix else f"{name}.png" for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "dist")
    parser.add_argument("--check", action="store_true", help="list the contents and exit")
    args = parser.parse_args()

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    # A red draftnote in a public preprint is worse than a missing one; the build stops rather than
    # packaging it.
    drafts = len(re.findall(r"\\draftnote\{", tex)) - tex.count(r"\newcommand{\draftnote}")
    if drafts > 0:
        raise SystemExit(f"main.tex still has {drafts} \\draftnote use(s); resolve them first.")

    members: list[tuple[Path, str]] = []
    for name in SOURCES:
        path = PAPER / name
        if not path.exists():
            raise SystemExit(f"missing {path}. Build the PDF first so that main.bbl exists.")
        members.append((path, name))
    for name in included_figures(tex):
        path = PAPER / "figures" / name
        if not path.exists():
            raise SystemExit(f"main.tex draws {name}, which is not in paper/figures.")
        members.append((path, f"figures/{name}"))

    total = sum(path.stat().st_size for path, _ in members)
    for path, arcname in members:
        print(f"  {arcname:34s} {path.stat().st_size / 1e3:7.1f} kB")
    print(f"  {'total':34s} {total / 1e6:7.2f} MB")
    if args.check:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"arxiv-{time.strftime('%Y%m%d')}.tar.gz"
    with tarfile.open(target, "w:gz") as bundle:
        for path, arcname in members:
            bundle.add(path, arcname)
    print(f"\nwrote {target} ({target.stat().st_size / 1e6:.2f} MB)")
    print("Upload it at https://arxiv.org/submit, primary category cs.NI, cross-list cs.LG.")


if __name__ == "__main__":
    main()
