# Where things stand — 24 September 2026

## Done and verified

- **The paper builds clean.** 14 pages, zero undefined references, no LaTeX warnings, no overfull
  boxes. `paper/main.pdf` is current.
- **Zenodo deposit is published.** DOI `10.5281/zenodo.22916038` (version) and
  `10.5281/zenodo.22916037` (concept) both resolve. Six files, 8.51 GB, and every MD5 was checked
  against the local copies in `deposit/` — all six match.
- **GitHub is current**, through commit `576741b`. The `prereg-frozen` tag is pushed.
- **Section V-G** (feature distillation, `kdF`) is written from `results/exploratory/REVISION_FULL3/`,
  which reproduces all 78 detection-advantage rows of `REVISION_FULL2` exactly.
- **Concern 8** of `paper/response-to-reviewers.md` is rewritten; it no longer says the arm was declined.
- **A content audit found and fixed twelve problems** — four wrong table cells from double rounding,
  and five claims the data did not support. See commit `09c6025` for the full list.
- **The abstract fits arXiv.** It was 2252 characters against a 1920 limit; now 1915.
  `paper/arxiv-abstract.txt` is the plain-text version for the submission form.
- **arXiv bundle built:** `dist/arxiv-20260924.tar.gz`, 0.46 MB. It compiles standalone under pdflatex
  alone, without BibTeX or `refs.bib`, which is how arXiv builds it.

## The one thing left

**Submit to arXiv.** https://arxiv.org/submit

- Upload `dist/arxiv-20260924.tar.gz` (rebuild with `python scripts/23_arxiv_bundle.py` if the paper changes).
- Abstract: paste `paper/arxiv-abstract.txt` verbatim.
- Primary category `cs.NI`, cross-list `cs.LG`.
- **Licence: take arXiv's default non-exclusive licence, not CC BY.** CC BY is irrevocable and some
  publishers treat it as a barrier; the default keeps a journal submission open.
- Register with the `usal.es` address so endorsement is automatic.

## Unfinished: the deep audit

A nine-lens adversarial audit was running when the session ended (workflow `wf_24d7214c-a8a`). Nine
lenses raised **86 findings**; roughly 60% of the skeptic votes were in, running a ~45% refute rate.

`docs/audit-findings-raw.md` holds all 86, **unverified**. Do not act on any of them without
re-checking — the whole point of the verification stage is that a lens working alone over-reports,
and on the earlier solo pass most raised items turned out not to be real.

The one item flagged **critical** is in the bibliography lens. Check that first: an unresolved
citation key or a fabricated entry would be exactly that severity, and it is quickly settled either
way. Note the paper currently builds with zero undefined citations, which makes a genuine critical
bibliography failure less likely than the label suggests.

To resume the workflow and let the verification finish:

```
Workflow({scriptPath: "<path in the run's tool result>", resumeFromRunId: "wf_24d7214c-a8a"})
```

Completed agents return from cache, so only the unfinished verifiers re-run.

## Standing constraints

- `kdtraffic/analysis.py` and `scripts/08_analyze.py` are frozen with the pre-registration. `analysis.py`
  is byte-identical to the freeze; `08_analyze.py` carries one commit confined to `figures()`, which
  §11 documents.
- `SCORES` in the frozen code must never be extended — it sets the Holm family size.
- Exploratory runs must never be merged into the confirmatory `--runs` list.
- `ZENODO_TOKEN` is read from the environment only, never a file or an argument.
- `scripts/11_cpu_table.py` pins `REPORTED` deliberately. A second `cpu_cost` run exists that re-times
  the re-tuned XGBoost; defaulting to the newest run once silently replaced all seven rows of Table VII.
