"""Assemble the data deposit: the artefacts too large for the source repository.

The paper releases its code on GitHub, but the per-flow unknown-scores and the model checkpoints are
several gigabytes and belong in a data archive with a DOI. This script collects them, records exactly
what is in the deposit, and writes the Zenodo metadata, so that uploading is the only manual step.

Nothing here redistributes CESNET-TLS-Year22. The scores are derived quantities --- one float per flow
per model --- and the checkpoints are trained weights; neither contains the flows themselves.

    python scripts/21_build_deposit.py                 # manifest and metadata only
    python scripts/21_build_deposit.py --archive       # also build the zip files to upload

Hashing several gigabytes takes a few minutes and competes for disk with anything else running, so run
it when the machine is otherwise idle.
"""

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPOSITORY = "https://github.com/Mahmoud-Abbasi-svg/kd-encrypted-traffic-inheritance"
REGISTRATION = "https://osf.io/rts6n"
# Reserved on Zenodo before publication, so the paper could cite it while the upload was still going.
DOI = "10.5281/zenodo.22916038"

# (archive name, what it holds, which run directories). Grouped so that a reader who wants only the
# confirmatory scores does not have to download the exploratory arms as well.
GROUPS = [
    ("scores-confirmatory", "Per-flow unknown-scores for the three pre-registered runs",
     ["results/track_a/*/", "results/pilot/*/"], "*.npz"),
    ("scores-exploratory", "Per-flow unknown-scores for the conditions added at review",
     ["results/track_a_exploratory/*/"], "*.npz"),
    ("scores-featurespace", "Mahalanobis and feature k-NN scores for every scored checkpoint",
     ["results/feature_scores/*/"], "*.npz"),
    ("checkpoints", "Trained teacher and student weights",
     ["results/track_a/*/", "results/track_a_exploratory/*/", "results/pilot/*/",
      "results/shortcut/*/"], "*.pt"),
]


def usable(directory: Path) -> bool:
    """Skip smoke runs, interrupted runs and anything without the table that describes it."""
    return (directory.is_dir() and "smoke" not in directory.name
            and "stopped" not in directory.name
            and any(directory.glob("config.json")))


def collect(patterns: list[str], suffix: str) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        for directory in sorted(PROJECT_ROOT.glob(pattern)):
            if usable(directory):
                found.extend(sorted(directory.rglob(suffix)))
    return found


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def human(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


def readme(groups: list[tuple[str, str, list[Path], int]], total: int) -> str:
    lines = [
        "# Data deposit: *What Does the Student Inherit?*",
        "",
        "Per-flow unknown-scores and trained model weights for the pre-registered study of what a",
        "distilled encrypted-traffic classifier inherits from its teacher.",
        "",
        f"- Cite this deposit as https://doi.org/{DOI}",
        f"- Code and analysis: {REPOSITORY}",
        f"- Pre-registration: {REGISTRATION}. The OSF project also archives the frozen protocol and",
        "  the service split the study was registered with, and links to this deposit.",
        "- Dataset: CESNET-TLS-Year22 (Luxemburk, Plesnik and Hynek, *Scientific Data* 11, 2024),",
        "  obtained through `cesnet-datazoo`. **This deposit does not redistribute it.** The scores here",
        "  are derived quantities, one floating-point value per flow per model, and the checkpoints are",
        "  trained weights; neither contains the captured flows.",
        "",
        "## Contents",
        "",
        "| Archive | Contents | Files | Size |",
        "|---|---|---|---|",
    ]
    for name, description, files, size in groups:
        lines.append(f"| `{name}.zip` | {description} | {len(files)} | {human(size)} |")
    lines += [
        f"| | **Total** | **{sum(len(f) for _, _, f, _ in groups)}** | **{human(total)}** |",
        "",
        "`MANIFEST.csv` lists every file with its size and SHA-256 digest.",
        "",
        "## Reading a score file",
        "",
        "Each `scores_<split>.npz` holds one array per model, in the flow order of that evaluation",
        "window, plus the arrays needed to evaluate them:",
        "",
        "```python",
        "import numpy as np",
        "scores = np.load('scores_test_w16-19.npz')",
        "scores['y']                      # true service index, -1 for an unknown service",
        "scores['day']                    # day of the capture, for the cluster bootstrap",
        "scores['teacherA__energy']       # one score per flow, higher means 'more likely known'",
        "scores['student_kdA4_s0__msp']   # student condition, seed 0",
        "```",
        "",
        "Model names follow the paper's conditions (Table I). Score suffixes are `energy`, `msp`, and,",
        "in the feature-space archive, `maha` and `knnfeat`.",
        "",
        "## Licence",
        "",
        "CC BY 4.0. The code that produced these files is released separately under the MIT licence.",
        "Use of CESNET-TLS-Year22 itself remains subject to that dataset's own terms.",
    ]
    return "\n".join(lines) + "\n"


def metadata(total: int, file_count: int) -> dict:
    return {
        "metadata": {
            "title": ("Per-flow unknown-scores and model checkpoints for "
                      "\"What Does the Student Inherit? Unknown-Traffic Detection, Calibration and "
                      "Shortcuts in Distilled Traffic Classifiers over Time\""),
            "upload_type": "dataset",
            "description": (
                "<p>Derived data for a pre-registered study of knowledge distillation in "
                "encrypted-traffic classification. It contains the per-flow unknown-traffic scores of "
                "every teacher and student, for every evaluation window, and the trained model "
                f"weights: {file_count} files, {human(total)} in total.</p>"
                "<p>The models are trained on CESNET-TLS-Year22 at three start dates and evaluated on "
                "18 test windows spanning 35 weeks. Scores are given under the two pre-registered "
                "logit-based rules (energy, maximum softmax probability) and, for the checkpoints "
                "scored in the revision, under two feature-space rules (Mahalanobis distance and "
                "feature-space k-nearest-neighbour distance).</p>"
                "<p><strong>This deposit does not redistribute CESNET-TLS-Year22.</strong> The scores "
                "are derived quantities and the checkpoints are trained weights; neither contains the "
                "captured flows. Use of the dataset itself remains subject to its own terms.</p>"
                f"<p>Code and analysis: <a href=\"{REPOSITORY}\">{REPOSITORY}</a>. "
                f"Pre-registration: <a href=\"{REGISTRATION}\">{REGISTRATION}</a>.</p>"),
            "creators": [{"name": "Abbasi, Mahmoud", "affiliation": "University of Salamanca",
                          "orcid": "0000-0002-1886-8284"}],
            "license": "cc-by-4.0",
            "access_right": "open",
            "keywords": ["encrypted traffic classification", "knowledge distillation",
                         "open-set recognition", "concept drift", "calibration",
                         "shortcut learning", "pre-registration", "CESNET-TLS-Year22"],
            "related_identifiers": [
                {"identifier": REPOSITORY, "relation": "isSupplementTo", "scheme": "url"},
                {"identifier": REGISTRATION, "relation": "isDocumentedBy", "scheme": "url"},
            ],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "deposit")
    parser.add_argument("--archive", action="store_true", help="also build the zip files")
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 (faster, weaker manifest)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    groups, rows, total = [], [], 0
    for name, description, patterns, suffix in GROUPS:
        files = collect(patterns, suffix)
        size = sum(f.stat().st_size for f in files)
        groups.append((name, description, files, size))
        total += size
        for path in files:
            rows.append({"archive": name,
                         "path": path.relative_to(PROJECT_ROOT).as_posix(),
                         "bytes": path.stat().st_size,
                         "sha256": "" if args.no_hash else digest(path)})
        print(f"  {name:22s} {len(files):4d} files  {human(size):>8s}")

    with (args.out / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["archive", "path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "README.md").write_text(readme(groups, total), encoding="utf-8")
    (args.out / "zenodo.json").write_text(json.dumps(metadata(total, len(rows)), indent=2),
                                          encoding="utf-8")

    if args.archive:
        for name, _, files, _ in groups:
            target = args.out / f"{name}.zip"
            # stored, not deflated: these are float arrays and compression buys a few percent for
            # minutes of CPU
            with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED, allowZip64=True) as bundle:
                for path in files:
                    bundle.write(path, path.relative_to(PROJECT_ROOT).as_posix())
            print(f"  wrote {target.name} ({human(target.stat().st_size)})")

    print(f"\n{len(rows)} files, {human(total)} total -> {args.out}")
    print("Upload the archives to Zenodo with deposit/zenodo.json as the metadata, then put the DOI "
          "in the paper's Reproducibility section.")


if __name__ == "__main__":
    main()
