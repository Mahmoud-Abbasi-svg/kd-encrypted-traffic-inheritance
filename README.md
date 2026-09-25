# Unknown-Traffic Detection, Calibration and Shortcut Reliance in Distilled Encrypted-Traffic Classifiers

Code and analysis for a pre-registered study of what a distilled encrypted-traffic classifier
inherits from its teacher besides accuracy: its detection of unknown traffic, its calibration, its
reliance on shortcut features, and whether any of it lasts a year.

> The pre-registration (`osf.io/rts6n`), the Zenodo deposit and the frozen documents under `docs/`
> were registered under the earlier working title *What Does the Student Inherit?*. Those records are
> frozen and still carry it; the manuscript title changed after they were filed.

Dataset: [CESNET-TLS-Year22](https://doi.org/10.1038/s41597-024-03927-4) (size S, 25M flows), via
`cesnet-datazoo`. Inputs are the first 30 packets' sizes, directions and inter-arrival times, plus 44
flow statistics. No SNI, IP, port, ASN or JA3 is used, so the setting stands in for a network where
Encrypted Client Hello has removed the plaintext identifiers.

## The design in one paragraph

A 101k-parameter student is distilled from two teachers of equal accuracy that differ in construction
(a five-member ensemble and a single wider model). Because soft targets are also a regulariser, a
student resembling its teacher proves nothing on its own; the teacher-swap control asks whether each
student's per-flow scores follow **its own** teacher more than the other one does, measured against a
directly trained student. Ten hypotheses were tested on 18 evaluation windows spanning 35 weeks after
training, from three start dates, with a day x service cluster bootstrap and Holm correction.

## Pre-registration

The protocol, the ten hypotheses and the analysis code were frozen before any test-window result was
computed, and registered publicly at [osf.io/rts6n](https://osf.io/rts6n).

- `docs/preregistration.md` is the frozen document, including the deviation log (section 11).
- The tag **`prereg-frozen`** marks the freeze commit, so the registered protocol and the analysis
  code implementing it can be checked against each other.
- `kdtraffic/analysis.py` and `scripts/08_analyze.py` implement the confirmatory analysis and are
  frozen with the protocol. Analyses added later live in `kdtraffic/exploratory.py` and
  `scripts/12_exploratory.py`, which import the frozen primitives rather than modifying them, so that
  `git diff prereg-frozen -- kdtraffic/analysis.py` stays empty.

Analyses added after the confirmatory results were known are labelled exploratory in the paper, carry
their own intervals, and share no multiple-comparison family with the registered ten.

## Layout

| Path | What is in it |
|---|---|
| `kdtraffic/` | library: data, models, distillation objectives, metrics, analysis |
| `scripts/` | numbered pipeline, run in order; each writes a timestamped directory under `results/` |
| `configs/splits.json` | the frozen known/unknown service split |
| `docs/preregistration.md` | the frozen protocol and deviation log |
| `paper/` | manuscript sources, and the response to the reviewer |
| `results/` | per-window metric tables, analysis output, figures |
| `deposit/` | manifest and metadata for the Zenodo data deposit |
| `tests/` | `python -m pytest tests` |

## Reproducing

Scripts share `--size`, week and cache arguments, and the training and analysis scripts take
`--smoke` for a fast pass on a tiny subset. A single 4 GB GPU is enough; the whole study was run on a
laptop.

```bash
python scripts/03_make_splits.py --size S          # frozen service split
python scripts/05_pilot.py --size S --start 11     # teachers, and the pilot gate
python scripts/06_track_a.py --size S --start 11 --with-test   # the condition grid
python scripts/08_analyze.py --windows test --runs <run dirs>  # the ten hypotheses
```

`scripts/08_analyze.py` refuses to run against test windows until the pre-registration is marked
frozen.

## Data availability

The per-window metric tables the analysis consumes are in `results/`. The per-flow unknown-scores and
the trained checkpoints behind them are 335 files and 8.5 GB, too large for a source repository, and
are deposited at [doi.org/10.5281/zenodo.22916038](https://doi.org/10.5281/zenodo.22916038) under
CC BY 4.0. `scripts/21_build_deposit.py` assembles that deposit and writes its manifest, so what is on
Zenodo can be regenerated and checked file by file against a SHA-256 digest.

The deposit holds derived quantities only, one floating-point score per flow per model, plus trained
weights. It does not redistribute CESNET-TLS-Year22.

## Licence

Code is under the [MIT Licence](LICENSE). Result tables, figures, the service split, the
pre-registration and the manuscript sources are under
[CC BY 4.0](LICENSE-DATA). Neither covers CESNET-TLS-Year22 itself, which is distributed by its own
authors under its own terms; no raw traffic is redistributed here.

## Citation

The manuscript is a preprint and has not been peer reviewed. Cite it together with the OSF
registration (<https://osf.io/rts6n>), which is what fixes the protocol and the date, and with the
Zenodo deposit if you use the scores or the checkpoints.
