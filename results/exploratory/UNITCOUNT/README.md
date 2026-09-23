# UNITCOUNT — a verification run, not a result

This directory exists to prove one thing: that correcting the unit count in
`detection_advantage.csv` changed no estimate. It was produced by `scripts/12_exploratory.py`
with `--n-boot 1`, because a point estimate needs no resampling.

**Read `estimate` and `units` here; ignore every interval.** With a single bootstrap draw the
`ci_low`, `ci_high` and `p_one_sided` columns in all of these files are meaningless. They are
written because the run used the ordinary code path, not because they mean anything.

All 78 estimates in `detection_advantage.csv` match `../REVISION_FULL3/` exactly. The `units`
column is the one that changed: the conditions trained on start date 11 alone — the four
width-sweep rows, `kdC` and `kdF` — report the nine windows they actually use instead of the
eighteen the unit list contains.

The run the paper reports is `../REVISION_FULL3/`. See section 11 of `docs/preregistration.md`.
