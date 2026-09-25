# arXiv submission package

Everything the submission form asks for. Rebuild the archive with
`python scripts/23_arxiv_bundle.py`; it regenerates `dist/arxiv-<date>.tar.gz` from the current
`paper/` sources and refuses to build if an unresolved `\draftnote` survives.

## 1. The upload

`dist/arxiv-20260925.tar.gz` — 446 KB, nine files:

```
main.tex  main.bbl  cpu_table.tex  exploratory_tables.tex
figures/fig_design.png  figures/fig_inheritance.png  figures/fig_gap_split.png
figures/fig_reliance.png  figures/fig_advantage.png
```

`main.bbl` is the compiled bibliography and has to be there: arXiv runs LaTeX but not BibTeX, so
without it every citation renders as a question mark. `refs.bib` is deliberately **not** included —
it would not be read, and shipping it suggests otherwise. Nothing else is needed; IEEEtran,
booktabs, hyperref and the rest are in arXiv's TeX Live.

## 2. Form fields

**Title** — one line, no break:

```
Unknown-Traffic Detection, Calibration and Shortcut Reliance in Distilled Encrypted-Traffic Classifiers over One Year
```

**Authors**

```
Mahmoud Abbasi
```

**Abstract** — paste the whole of `paper/arxiv-abstract.txt`. It is 1912 characters against arXiv's
1920 limit, plain ASCII, no LaTeX, and matches the abstract in the PDF word for word.

**Primary category** — `cs.NI` (Networking and Internet Architecture)

**Cross-list** — `cs.LG` (Machine Learning)

**Comments**

```
15 pages, 5 figures, 11 tables. Pre-registered at OSF (https://osf.io/rts6n) before any test-window result was computed. Code: https://github.com/Mahmoud-Abbasi-svg/kd-encrypted-traffic-inheritance. Per-flow scores and model checkpoints: https://doi.org/10.5281/zenodo.22916038
```

**ACM-class** (optional) — `C.2.3; I.2.6`

**Licence** — keep arXiv's default, *arXiv.org perpetual, non-exclusive license to distribute this
article*. It leaves every journal option open. A CC licence would not, and cannot be revoked later.

## 3. Leave blank

| Field | Why |
|---|---|
| DOI | This field is for the DOI of a **published journal version of this paper**. The Zenodo DOI identifies the dataset, not the article, and is already in Comments. Putting it here would misrepresent the preprint as published. |
| Journal reference | Not published. |
| Report number | None. |

## 4. Endorsement

`cs.NI` needs an endorsement from first-time submitters, and arXiv grants it automatically on the
strength of a recognised **academic institution** e-mail domain.

The account is registered to **mahmoud.abbasi@ieee.org**. IEEE is a professional society rather than
an institution, so that address may not trigger automatic endorsement; **mahmoudabbasi@usal.es**
would. Change it under "Change User Information" on the Start page before going further, and expect
a verification e-mail before the change takes effect. The address printed in the paper is a separate
thing and does not have to match.

## 5. Known and deliberate

- **A 1.05 pt vertical overflow on page 15.** Roughly a third of a millimetre on the final page,
  invisible in the rendered output, and a consequence of where the text breaks rather than of any
  line being too long. Left alone.
- **The running head prints `PREPRINT, SEPTEMBER 2026` on every page.** If arXiv posts the paper in
  a later month this will disagree with arXiv's own datestamp. Change `\markboth` and the second
  `\thanks` in `main.tex` to drop the month if that matters.
- **The Zenodo record and the OSF registration carry the earlier working title**, *What Does the
  Student Inherit?*. A Zenodo record's **metadata** can be edited after publication without minting a
  new version — only its files are frozen — so the record title can be brought into line. An OSF
  registration is immutable by design and will keep the old title. `README.md` explains both.
- **The deposited score files store MSP in `float16`.** Disclosed in Section V-A and in the
  limitations; no confirmatory outcome depends on it, and the energy and feature-space scores are
  unaffected.

## 6. Verification record

The archive was extracted into an empty directory and compiled there with `pdflatex` alone, three
passes, no BibTeX — the way arXiv does it:

- 15 pages, **zero** undefined references or citations, all 51 bibliography entries resolved
- the only log warning is `balance`, which is benign and predates this work
- the text extracted from that PDF is **byte-identical** (84,804 characters) to `paper/main.pdf`;
  the two files differ in 64 bytes, all of them inside the embedded `CreationDate`/`ModDate`
- a scan of the rendered text finds no orphaned control sequences, no `[?]` and no `??`
- every bibliography entry has a year, none is dated after today, none carries a placeholder
- all five figures were checked against the numbers in the text they illustrate
- the Zenodo DOI resolves to a published CC BY 4.0 record whose 129 score files, 8.2 GB and 206
  checkpoints match the Reproducibility section
