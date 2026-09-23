# Data deposit: *What Does the Student Inherit?*

Per-flow unknown-scores and trained model weights for the pre-registered study of what a
distilled encrypted-traffic classifier inherits from its teacher.

- Cite this deposit as https://doi.org/10.5281/zenodo.22916038
- Code and analysis: https://github.com/Mahmoud-Abbasi-svg/kd-encrypted-traffic-inheritance
- Pre-registration: https://osf.io/rts6n. The OSF project also archives the frozen protocol and
  the service split the study was registered with, and links to this deposit.
- Dataset: CESNET-TLS-Year22 (Luxemburk, Plesnik and Hynek, *Scientific Data* 11, 2024),
  obtained through `cesnet-datazoo`. **This deposit does not redistribute it.** The scores here
  are derived quantities, one floating-point value per flow per model, and the checkpoints are
  trained weights; neither contains the captured flows.

## Contents

| Archive | Contents | Files | Size |
|---|---|---|---|
| `scores-confirmatory.zip` | Per-flow unknown-scores for the three pre-registered runs | 27 | 2.6 GB |
| `scores-exploratory.zip` | Per-flow unknown-scores for the conditions added at review | 61 | 3.5 GB |
| `scores-featurespace.zip` | Mahalanobis and feature k-NN scores for every scored checkpoint | 41 | 2.1 GB |
| `checkpoints.zip` | Trained teacher and student weights | 206 | 356 MB |
| | **Total** | **335** | **8.5 GB** |

`MANIFEST.csv` lists every file with its size and SHA-256 digest.

## Reading a score file

Each `scores_<split>.npz` holds one array per model, in the flow order of that evaluation
window, plus the arrays needed to evaluate them:

```python
import numpy as np
scores = np.load('scores_test_w16-19.npz')
scores['y']                      # true service index, -1 for an unknown service
scores['day']                    # day of the capture, for the cluster bootstrap
scores['teacherA__energy']       # one score per flow, higher means 'more likely known'
scores['student_kdA4_s0__msp']   # student condition, seed 0
```

Model names follow the paper's conditions (Table I). Score suffixes are `energy`, `msp`, and,
in the feature-space archive, `maha` and `knnfeat`.

## Licence

CC BY 4.0. The code that produced these files is released separately under the MIT licence.
Use of CESNET-TLS-Year22 itself remains subject to that dataset's own terms.
