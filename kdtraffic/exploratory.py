"""Exploratory analyses added during the revision, at the reviewer's request.

**None of this is pre-registered.** The ten confirmatory hypotheses live in `kdtraffic/analysis.py`
and `scripts/08_analyze.py`, which the pre-registration freezes together with the protocol
(`docs/preregistration.md` §6). Those two files are therefore never imported-and-modified here, only
imported: this module reuses their primitives so that the exploratory numbers are computed exactly
the way the confirmatory ones were, while leaving the frozen code untouched.

What it adds:
    drift_by_member          does the teacher ensemble lose unknown detection faster than its own
                             members? (tests the mechanism the discussion currently speculates about)
    slopes_per_start         the H3 time trend fitted separately per start date, because the three
                             start dates' windows overlap in calendar time
    calendar_overlap         which calendar weeks each unit covers, and how many are shared
    teacher_preference_null  how much an *undistilled* student's preference between two equally good
                             same-family teachers varies by chance - the scale H1's effect is read against
    swap_components          teacher-swap shifts for arbitrary (student, own teacher, other teacher)
                             triples, so single-vs-single and identity-only swaps can be tested

Conventions follow the frozen code: unknown-scores are oriented so higher means "more likely known",
a shift is a difference in differences against the directly trained student, and uncertainty comes
from the same day x service cluster bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kdtraffic.analysis import RankedScore, fixed_effects_slope, ranks, student, weighted_pearson

SCORES = ("energy", "msp")


# Loading ---------------------------------------------------------------------------------------

@dataclass
class ExUnit:
    """One start date x one evaluation split, holding only what the exploratory statistics need."""
    start: int
    split: str
    weeks_since: float
    cluster: np.ndarray = field(repr=False, default=None)
    known: np.ndarray = field(repr=False, default=None)
    seeds: list[int] = field(default_factory=list)
    detect: dict[str, RankedScore] = field(repr=False, default_factory=dict)  # "<score>:<model>"
    rank: dict[str, np.ndarray] = field(repr=False, default_factory=dict)  # model -> ranked energy
    models: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.known)


def seeds_of(keys, condition: str) -> list[int]:
    """Seeds present for `condition`, read from the score-file keys."""
    prefix, suffix = f"student_{condition}_s", "__energy"
    out = []
    for key in keys:
        if key.startswith(prefix) and key.endswith(suffix):
            tail = key[len(prefix):-len(suffix)]
            if tail.isdigit():
                out.append(int(tail))
    return sorted(out)


def merge_scores(paths, require=("y", "day")) -> dict[str, np.ndarray]:
    """Per-flow arrays for one split, merging several files that describe the same flows.

    Unlike the frozen loader this refuses to overwrite: two files may not define the same array,
    because that silently discards one of them.
    """
    data: dict[str, np.ndarray] = {}
    shared = {"y", "app", "day", "duplicate", "ppi_len"}
    for path in paths:
        with np.load(path, allow_pickle=False) as npz:
            part = {k: npz[k] for k in npz.files}
        if data:
            if not np.array_equal(data["y"], part["y"]):
                raise SystemExit(f"{path} describes different flows than the other files of this split")
            clash = sorted((set(part) - shared) & set(data))
            if clash:
                raise SystemExit(f"{path} redefines {len(clash)} arrays already loaded "
                                 f"(first: {clash[:4]}); give the runs distinct condition names")
        data.update({k: v for k, v in part.items() if k not in data})
    missing = [k for k in require if k not in data]
    if missing:
        raise SystemExit(f"{paths[0]} and its companions lack {missing}")
    return data


def build_unit(data: dict[str, np.ndarray], start: int, split: str, weeks_since: float,
               models: list[str], rank_models: list[str], mask: np.ndarray | None = None) -> ExUnit:
    """`models` get AUROC-ready score objects; `rank_models` get ranked energies for correlations."""
    mask = np.ones(len(data["y"]), dtype=bool) if mask is None else mask
    known = data["y"][mask] >= 0
    unit = ExUnit(start, split, weeks_since, known=known, models=list(models))
    unit.seeds = seeds_of(data.keys(), "direct")
    for model in models:
        for score in SCORES:
            key = f"{model}__{score}"
            if key in data:
                unit.detect[f"{score}:{model}"] = RankedScore(data[key][mask].astype(np.float64), known)
    for model in rank_models:
        key = f"{model}__energy"
        if key in data:
            unit.rank[model] = ranks(data[key][mask].astype(np.float64))
    return unit


# Statistics ------------------------------------------------------------------------------------

def auroc(unit: ExUnit, model: str, score: str, weights: np.ndarray) -> float:
    ranked = unit.detect.get(f"{score}:{model}")
    return float("nan") if ranked is None else ranked.auroc(weights)


def shift(unit: ExUnit, condition: str, own: str, other: str, weights: np.ndarray,
          baseline: str = "direct") -> float:
    """Difference in differences: how much more `condition` follows `own` than `other` does,
    relative to the directly trained student, averaged over seeds.

    This is the H1 statistic, generalised to arbitrary teacher pairs.
    """
    values = []
    for seed in unit.seeds:
        s, b = student(condition, seed), student(baseline, seed)
        if s not in unit.rank or b not in unit.rank or own not in unit.rank or other not in unit.rank:
            continue
        distilled = weighted_pearson(unit.rank[s], unit.rank[own], weights) - \
            weighted_pearson(unit.rank[s], unit.rank[other], weights)
        plain = weighted_pearson(unit.rank[b], unit.rank[own], weights) - \
            weighted_pearson(unit.rank[b], unit.rank[other], weights)
        values.append(distilled - plain)
    return float(np.mean(values)) if values else float("nan")


def teacher_preference_null(unit: ExUnit, members: list[str], weights: np.ndarray,
                            baseline: str = "direct") -> list[float]:
    """|ρ(direct, member_i) − ρ(direct, member_j)| over all member pairs.

    The scale on which a teacher-swap shift should be read: this is how much an undistilled student's
    preference between two equally good, same-family teachers varies for no reason at all.
    """
    out = []
    for seed in unit.seeds:
        b = student(baseline, seed)
        if b not in unit.rank:
            continue
        rho = {m: weighted_pearson(unit.rank[b], unit.rank[m], weights) for m in members if m in unit.rank}
        names = sorted(rho)
        out.extend(abs(rho[a] - rho[c]) for i, a in enumerate(names) for c in names[i + 1:])
    return out


def drift_by_member(units: list[ExUnit], members: list[str], ensemble: str = "teacherA") -> pd.DataFrame:
    """AUROC of the ensemble and of each member per unit, with the H3-style slope of each.

    If the ensemble's slope is more negative than its members', its faster ageing is a property of
    the aggregation rather than of model size.
    """
    ones = {id(u): np.ones(len(u)) for u in units}
    rows = []
    for unit in units:
        for score in SCORES:
            row = {"start": unit.start, "split": unit.split, "weeks_since": unit.weeks_since, "score": score,
                   "ensemble": auroc(unit, ensemble, score, ones[id(unit)])}
            member_values = [auroc(unit, m, score, ones[id(unit)]) for m in members]
            row["member_mean"] = float(np.nanmean(member_values))
            row["member_spread"] = float(np.nanmax(member_values) - np.nanmin(member_values))
            for m, v in zip(members, member_values):
                row[m] = v
            rows.append(row)
    table = pd.DataFrame(rows)
    slopes = []
    for score, part in table.groupby("score"):
        for column in ["ensemble", "member_mean"] + members:
            slopes.append({"score": score, "model": column,
                           "slope_per_week": fixed_effects_slope(part.weeks_since.to_numpy(),
                                                                 part[column].to_numpy(),
                                                                 part.start.to_numpy())})
    return table, pd.DataFrame(slopes)


def slopes_per_start(table: pd.DataFrame, value: str) -> pd.DataFrame:
    """The time trend fitted separately for each start date (the three series overlap in calendar time)."""
    rows = []
    for start, part in table.groupby("start"):
        x, y = part.weeks_since.to_numpy(), part[value].to_numpy()
        keep = np.isfinite(x) & np.isfinite(y)
        slope = float("nan")
        if keep.sum() >= 2 and np.ptp(x[keep]) > 0:
            slope = float(np.polyfit(x[keep], y[keep], 1)[0])
        rows.append({"start": int(start), "n_windows": int(keep.sum()), "slope_per_week": slope,
                     "first": float(y[keep][0]) if keep.any() else float("nan"),
                     "last": float(y[keep][-1]) if keep.any() else float("nan")})
    return pd.DataFrame(rows)


def calendar_overlap(units: list[ExUnit]) -> pd.DataFrame:
    """Which calendar weeks each unit covers, and how many other start dates also cover them."""
    def weeks_of(split: str) -> list[int]:
        body = split.split("_w")[-1]
        ends = [int(p) for p in body.split("-")]
        return list(range(ends[0], ends[-1] + 1))

    coverage: dict[int, set[int]] = {}
    for unit in units:
        for week in weeks_of(unit.split):
            coverage.setdefault(week, set()).add(unit.start)
    rows = []
    for unit in units:
        weeks = weeks_of(unit.split)
        shared = [w for w in weeks if len(coverage[w]) > 1]
        rows.append({"start": unit.start, "split": unit.split, "weeks": f"{weeks[0]}-{weeks[-1]}",
                     "n_weeks": len(weeks), "weeks_shared_with_another_start": len(shared)})
    return pd.DataFrame(rows)


def cluster_bootstrap_table(units: list[ExUnit], statistic, n_clusters: int, n_boot: int = 300,
                            seed: int = 2022, alpha: float = 0.05, progress=None) -> dict[str, float]:
    """Pooled mean of `statistic(unit, weights)` over units, with a day x service cluster bootstrap.

    Seeds are averaged inside `statistic`; this resamples clusters only, which is the cheaper half of
    the frozen procedure and adequate for exploratory intervals.
    """
    point = float(np.nanmean([statistic(u, np.ones(len(u))) for u in units]))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        draws[b] = np.nanmean([statistic(u, counts[u.cluster]) for u in units])
        if progress and (b + 1) % max(1, n_boot // 5) == 0:
            progress(f"    bootstrap {b + 1}/{n_boot}")
    finite = draws[np.isfinite(draws)]
    low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2]) if len(finite) else (np.nan, np.nan)
    return {"estimate": point, "ci_low": float(low), "ci_high": float(high),
            "p_one_sided": float((1 + np.sum(finite <= 0)) / (len(finite) + 1)) if len(finite) else float("nan"),
            "n_boot": int(len(finite))}
