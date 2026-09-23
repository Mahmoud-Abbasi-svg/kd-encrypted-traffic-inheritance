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
from pathlib import Path

import numpy as np
import pandas as pd

from kdtraffic.analysis import RankedScore, fixed_effects_slope, ranks, student, weighted_pearson

# The two pre-registered scores, plus the two feature-space scores added in revision. This constant is
# deliberately NOT the frozen `analysis.SCORES`: that one sets the size of the Holm family, so adding
# to it would change every corrected p-value in the confirmatory table. Here nothing is corrected
# across scores, and a score simply absent from a run's files is skipped.
SCORES = ("energy", "msp", "maha", "knnfeat")


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


def merge_scores(paths, require=("y", "day"), tolerance: float = 1e-3) -> dict[str, np.ndarray]:
    """Per-flow arrays for one split, merging several files that describe the same flows.

    Unlike the frozen loader this never silently overwrites. Two files may define the same array only
    if they agree: every run of `06_track_a.py` re-scores the same teachers on the same windows, so
    the teacher arrays legitimately appear in several files and must match. An array that is present
    twice with *different* values means the files describe different models under the same name, which
    would corrupt every statistic computed from them, and stops the analysis.

    Scores are stored as float16, so agreement is checked to `tolerance` rather than exactly.
    """
    data: dict[str, np.ndarray] = {}
    source: dict[str, str] = {}
    shared = {"y", "app", "day", "duplicate", "ppi_len"}
    for path in paths:
        with np.load(path, allow_pickle=False) as npz:
            part = {k: npz[k] for k in npz.files}
        if data:
            if not np.array_equal(data["y"], part["y"]):
                raise SystemExit(f"{path} describes different flows than the other files of this split")
            for key in sorted((set(part) - shared) & set(data)):
                old, new = np.asarray(data[key], dtype=np.float64), np.asarray(part[key], dtype=np.float64)
                if old.shape != new.shape or not np.allclose(old, new, rtol=0.0, atol=tolerance,
                                                             equal_nan=True):
                    worst = float(np.nanmax(np.abs(old - new))) if old.shape == new.shape else float("nan")
                    raise SystemExit(
                        f"'{key}' is defined by both {Path(source[key]).name} and {Path(path).name} "
                        f"with different values (largest difference {worst:.4g}). The same name "
                        f"refers to two different models; give the runs distinct condition names.")
        for key, value in part.items():
            if key not in data:
                data[key], source[key] = value, str(path)
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
          seed_counts: np.ndarray | None = None, baseline: str = "direct") -> float:
    """Difference in differences: how much more `condition` follows `own` than `other` does,
    relative to the directly trained student, averaged over seeds.

    This is the H1 statistic, generalised to arbitrary teacher pairs. `seed_counts` gives each seed's
    multiplicity, as `unit_statistics` does in the frozen code, so that a bootstrap resample can vary
    the seeds as well as the flows; `None` weights every seed equally.
    """
    counts = np.ones(len(unit.seeds)) if seed_counts is None else np.asarray(seed_counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    value, used = 0.0, 0.0
    for seed, count in zip(unit.seeds, counts):
        s, b = student(condition, seed), student(baseline, seed)
        if count <= 0 or any(k not in unit.rank for k in (s, b, own, other)):
            continue
        distilled = weighted_pearson(unit.rank[s], unit.rank[own], weights) - \
            weighted_pearson(unit.rank[s], unit.rank[other], weights)
        plain = weighted_pearson(unit.rank[b], unit.rank[own], weights) - \
            weighted_pearson(unit.rank[b], unit.rank[other], weights)
        value += count * (distilled - plain)
        used += count
    return value / used if used > 0 else float("nan")


def _seed_weights(unit: ExUnit, seed_counts: np.ndarray | None) -> np.ndarray:
    return np.ones(len(unit.seeds)) if seed_counts is None else np.asarray(seed_counts, dtype=float)


def _auroc(unit: ExUnit, key: str, weights: np.ndarray, cache: dict | None) -> float:
    """AUROC of one ranked score, memoised within a bootstrap draw.

    Several comparisons share the same models - every student condition is compared against
    `direct`, and every score against the same teacher - so without this each draw would recompute
    the same AUROCs a dozen times over. The cache is keyed by unit and score and is cleared by
    `cluster_bootstrap_many` whenever the weights change.
    """
    ranked = unit.detect.get(key)
    if ranked is None:
        return float("nan")
    if cache is None:
        return ranked.auroc(weights)
    entry = (id(unit), key)
    value = cache.get(entry)
    if value is None:
        value = cache[entry] = ranked.auroc(weights)
    return value


def detection_shift(unit: ExUnit, condition: str, baseline: str, score: str, weights: np.ndarray,
                    seed_counts: np.ndarray | None = None, cache: dict | None = None) -> float:
    """Seed-averaged AUROC(`condition`) - AUROC(`baseline`) under one scoring rule.

    Used to ask, under the feature-space scores, the question H2 and H5 ask under the logit scores:
    does a distilled student detect unknown traffic better than the same student trained otherwise?
    Both models are the same architecture, so this comparison is free of the feature-dimensionality
    difference that makes a teacher-versus-student comparison awkward.
    """
    value, used = 0.0, 0.0
    for seed, count in zip(unit.seeds, _seed_weights(unit, seed_counts)):
        if count <= 0:
            continue
        a = _auroc(unit, f"{score}:{student(condition, seed)}", weights, cache)
        b = _auroc(unit, f"{score}:{student(baseline, seed)}", weights, cache)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        value += count * (a - b)
        used += count
    return value / used if used > 0 else float("nan")


def teacher_advantage(unit: ExUnit, teacher: str, baseline: str, score: str, weights: np.ndarray,
                      seed_counts: np.ndarray | None = None, cache: dict | None = None) -> float:
    """Seed-averaged AUROC(`teacher`) - AUROC(the `baseline` student) under one scoring rule.

    This is the quantity a reviewer means by "is there a teacher advantage to inherit at all?".
    Across models of different feature dimensionality it should be read with care.
    """
    teacher_auroc = _auroc(unit, f"{score}:{teacher}", weights, cache)
    if not np.isfinite(teacher_auroc):
        return float("nan")
    value, used = 0.0, 0.0
    for seed, count in zip(unit.seeds, _seed_weights(unit, seed_counts)):
        b = _auroc(unit, f"{score}:{student(baseline, seed)}", weights, cache)
        if count <= 0 or not np.isfinite(b):
            continue
        value += count * (teacher_auroc - b)
        used += count
    return value / used if used > 0 else float("nan")


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
    """Pooled mean of `statistic(unit, weights, seed_counts)` over units, resampling day x service
    clusters and training seeds exactly as `pooled_bootstrap` does in the frozen code.

    The draws are generated in the frozen order - clusters first, then one seed resample per start
    date - so that an exploratory interval is built the same way as the confirmatory ones and the two
    can be compared. With the same units, seed and number of resamples, the H1 replication row
    reproduces the confirmatory interval as well as the point estimate.
    """
    return cluster_bootstrap_many(units, {"value": statistic}, n_clusters, n_boot, seed,
                                  alpha, progress)["value"]


def cluster_bootstrap_many(units: list[ExUnit], statistics: dict, n_clusters: int, n_boot: int = 300,
                           seed: int = 2022, alpha: float = 0.05, progress=None) -> dict[str, dict]:
    """Bootstrap many statistics together, resampling once per draw instead of once per statistic.

    Every statistic sees the same resampled clusters and seeds, which is both far cheaper and the
    way `pooled_bootstrap` works in the frozen code: one table per draw, all components derived
    from it. Statistics are called as `fn(unit, weights, seed_counts, cache)`, where `cache` is
    cleared between draws and lets them share repeated AUROC computations; a statistic that does not
    want it may accept and ignore the argument.
    """
    starts = sorted({u.start for u in units})
    seeds_by_start = {s: next(u.seeds for u in units if u.start == s) for s in starts}
    ones = {s: np.ones(len(v)) for s, v in seeds_by_start.items()}

    def pooled(weights_of, seed_counts_of, cache) -> dict[str, float]:
        return {name: float(np.nanmean([fn(u, weights_of(u), seed_counts_of(u), cache) for u in units]))
                for name, fn in statistics.items()}

    # The point estimate is a nanmean, so a statistic whose condition was never trained at a given
    # start date simply skips those units. That is the intended behaviour, but it makes the number of
    # units a property of the statistic rather than of the list: an arm run on one start date pools
    # over nine windows while the registered arms pool over eighteen. Counting here, where every
    # statistic is evaluated on every unit anyway, costs nothing and keeps the caller from guessing.
    cache: dict = {}
    per_unit = {name: np.array([fn(u, np.ones(len(u)), ones[u.start], cache) for u in units],
                               dtype=float)
                for name, fn in statistics.items()}
    point = {name: float(np.nanmean(values)) for name, values in per_unit.items()}
    used = {name: int(np.isfinite(values).sum()) for name, values in per_unit.items()}
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_boot) for name in statistics}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        seed_counts = {s: np.bincount(rng.integers(0, len(v), size=len(v)), minlength=len(v))
                       for s, v in seeds_by_start.items()}
        values = pooled(lambda u: counts[u.cluster], lambda u: seed_counts[u.start], {})
        for name, value in values.items():
            draws[name][b] = value
        if progress and (b + 1) % max(1, n_boot // 5) == 0:
            progress(f"    bootstrap {b + 1}/{n_boot}")

    out = {}
    for name, series in draws.items():
        finite = series[np.isfinite(series)]
        low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2]) if len(finite) else (np.nan, np.nan)
        out[name] = {"estimate": point[name], "ci_low": float(low), "ci_high": float(high),
                     "p_one_sided": (float((1 + np.sum(finite <= 0)) / (len(finite) + 1))
                                     if len(finite) else float("nan")),
                     "n_boot": int(len(finite)), "units_used": used[name]}
    return out
