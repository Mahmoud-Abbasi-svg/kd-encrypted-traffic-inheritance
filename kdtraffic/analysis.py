"""Confirmatory analysis (docs/preregistration.md, section 6).

The unit of analysis is one start date x one evaluation split (a test window). Uncertainty comes from
a bootstrap that, in every resample,
    * redraws day x service clusters with replacement, *globally*: a cluster that appears in the
      windows of several start dates gets the same multiplicity in all of them, and
    * redraws the student seeds with replacement, separately for each start date.
Resampled clusters enter the metrics as integer flow weights, so each metric is a weighted version
that equals the ordinary metric when all weights are 1. Scores are sorted once per unit, which keeps
a weighted AUROC at O(n) per resample.

A hypothesis may have several components (e.g. H2 compares kdA with two controls on energy AUROC and
on post-hoc NLL, the calibration outcome chosen in decision D7; ECE is reported, not tested);
it is supported only if every component is, so its p-value is the largest component p-value
(intersection-union test). One-sided bootstrap p-values are (1 + #{resampled estimate <= 0}) / (B + 1).
Holm's correction is applied across the hypotheses of the primary analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata

N_BINS = 15
CONDITIONS = ("direct", "directTS", "ls", "kdA", "kdB", "enddA")
SCORES = ("energy", "msp")  # co-primary unknown-scores (decision D6)


# Weighted metrics ------------------------------------------------------------------------------

class RankedScore:
    """An unknown-score sorted once, so AUROC under many flow weightings costs O(n) each."""

    def __init__(self, score: np.ndarray, known: np.ndarray):
        self.order = np.argsort(score, kind="stable")
        s = score[self.order]
        self.group = np.r_[0, np.cumsum(s[1:] != s[:-1])] if len(s) else np.zeros(0, dtype=int)
        self.n_groups = int(self.group[-1]) + 1 if len(s) else 0
        self.known = known[self.order]

    def auroc(self, weights: np.ndarray) -> float:
        """P(score of a known flow > score of an unknown flow), ties counted as 1/2."""
        w = weights[self.order].astype(np.float64)
        pos = np.bincount(self.group, weights=w * self.known, minlength=self.n_groups)
        neg = np.bincount(self.group, weights=w * ~self.known, minlength=self.n_groups)
        total = pos.sum() * neg.sum()
        if total == 0:
            return float("nan")
        below = np.cumsum(neg) - neg
        return float(np.sum(pos * (below + 0.5 * neg)) / total)


def weighted_macro_f1(y: np.ndarray, pred: np.ndarray, weights: np.ndarray, num_classes: int) -> float:
    """Macro-F1 over the classes present in `y` (as metrics.closed_set_metrics)."""
    w = weights.astype(np.float64)
    hit = pred == y
    tp = np.bincount(y, weights=w * hit, minlength=num_classes)
    support = np.bincount(y, weights=w, minlength=num_classes)
    predicted = np.bincount(pred, weights=w, minlength=num_classes)
    present = support > 0
    if not present.any():
        return float("nan")
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(f1[present].mean())


def confidence_bins(confidence: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)


def weighted_ece(confidence: np.ndarray, correct: np.ndarray, bins: np.ndarray, weights: np.ndarray,
                 n_bins: int = N_BINS) -> float:
    """Expected calibration error with the same bins as metrics.expected_calibration_error."""
    w = weights.astype(np.float64)
    total = w.sum()
    if total == 0:
        return float("nan")
    acc = np.bincount(bins, weights=w * correct, minlength=n_bins)
    conf = np.bincount(bins, weights=w * confidence, minlength=n_bins)
    return float(np.abs(acc - conf).sum() / total)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights.astype(np.float64)
    total = w.sum()
    return float((w * values).sum() / total) if total > 0 else float("nan")


def weighted_pearson(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    w = weights.astype(np.float64)
    total = w.sum()
    if total == 0:
        return float("nan")
    ma, mb = (w * a).sum() / total, (w * b).sum() / total
    da, db = a - ma, b - mb
    denominator = np.sqrt((w * da * da).sum() * (w * db * db).sum())
    return float((w * da * db).sum() / denominator) if denominator > 0 else float("nan")


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks; the weighted Pearson correlation of ranks is the Spearman correlation at unit weights."""
    return rankdata(x.astype(np.float64)).astype(np.float64)


# Units -----------------------------------------------------------------------------------------

@dataclass
class Unit:
    """One start date x one evaluation split, with everything the hypotheses need precomputed."""
    start: int
    split: str
    weeks_since: float
    cluster: np.ndarray  # global day x service cluster of each flow
    known: np.ndarray
    seeds: list[int]
    num_classes: int
    y_known: np.ndarray = field(repr=False, default=None)
    detect: dict[str, RankedScore] = field(repr=False, default_factory=dict)
    preds: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    calib: dict[str, tuple] = field(repr=False, default_factory=dict)
    nll: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    kd_conditions: list[str] = field(default_factory=list)  # kdA/kdB (tuned) and kdA4/kdB4 (T = 4) if trained
    rank: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def __len__(self) -> int:
        return len(self.known)


def student(condition: str, seed: int) -> str:
    return f"student_{condition}_s{seed}"


def _seeds(files: list[str]) -> list[int]:
    prefix = "student_kdA_s"
    return sorted(int(k[len(prefix):].split("__")[0]) for k in files if k.startswith(prefix) and k.endswith("__energy"))


def build_unit(data, start: int, split: str, weeks_since: float, num_classes: int, mask: np.ndarray) -> Unit:
    """`data` is a loaded scores_<split>.npz (or a dict with the same keys); `mask` selects the flows used."""
    known = data["y"][mask] >= 0
    seeds = _seeds(list(data.keys()))
    unit = Unit(start, split, weeks_since, np.empty(0, dtype=np.int64), known, seeds, num_classes)
    unit.y_known = data["y"][mask][known].astype(np.int64)

    def get(model: str, what: str) -> np.ndarray:
        return data[f"{model}__{what}"][mask].astype(np.float64 if what != "pred" else np.int64)

    detection_models = ["teacherA"] + [student(c, s) for c in ("kdA", "ls", "directTS", "enddA") for s in seeds]
    for score in SCORES:  # co-primary unknown-scores (decision D6)
        for model in detection_models:
            unit.detect[f"{score}:{model}"] = RankedScore(get(model, score), known)
    for c in ("kdA", "ls", "directTS"):
        for s in seeds:
            model = student(c, s)
            pred = get(model, "pred")[known]
            unit.preds[model] = pred
            # post-hoc calibration: temperature-scaled confidence (directTS is already scaled)
            what = "msp" if c == "directTS" else "msp_ts"
            confidence = get(model, what)[known]
            unit.calib[model] = (confidence, (pred == unit.y_known).astype(np.float64), confidence_bins(confidence))
            unit.nll[model] = get(model, "nll_ts")[known]  # post-hoc NLL (decision D7)
    kd_conditions = [c for c in ("kdA", "kdB", "kdA4", "kdB4") if f"{student(c, seeds[0])}__energy" in data]
    for model in ["teacherA", "teacherB"] + [student(c, s) for c in kd_conditions + ["direct"] for s in seeds]:
        unit.rank[model] = ranks(get(model, "energy"))
    unit.kd_conditions = kd_conditions
    return unit


def assign_clusters(units: list[Unit], days: list[np.ndarray], apps: list[np.ndarray]) -> int:
    """Give every flow a global day x service cluster id; returns the number of clusters."""
    keys = pd.Series(np.concatenate([d.astype(str) for d in days])) + "|" + pd.Series(np.concatenate(apps).astype(str))
    ids = pd.factorize(keys)[0]
    offset = 0
    for unit in units:
        unit.cluster = ids[offset:offset + len(unit)]
        offset += len(unit)
    return int(ids.max()) + 1 if len(ids) else 0


# Statistics ------------------------------------------------------------------------------------

def unit_statistics(unit: Unit, weights: np.ndarray, seed_counts: np.ndarray) -> dict[str, float]:
    """Seed-averaged metrics of one unit under flow weights and seed multiplicities."""
    wk = weights[unit.known]
    share = seed_counts / seed_counts.sum()

    def seed_mean(fn) -> float:
        return float(sum(p * fn(s) for p, s in zip(share, unit.seeds) if p > 0))

    out = {}
    for score in SCORES:
        out[f"auroc_{score}_teacherA"] = unit.detect[f"{score}:teacherA"].auroc(weights)
        for c in ("kdA", "ls", "directTS", "enddA"):
            out[f"auroc_{score}_{c}"] = seed_mean(
                lambda s, c=c, score=score: unit.detect[f"{score}:{student(c, s)}"].auroc(weights))
    for c in ("kdA", "ls", "directTS"):
        out[f"f1_{c}"] = seed_mean(lambda s, c=c: weighted_macro_f1(unit.y_known, unit.preds[student(c, s)], wk,
                                                                    unit.num_classes))
        out[f"ece_{c}"] = seed_mean(lambda s, c=c: weighted_ece(*unit.calib[student(c, s)], wk))
        out[f"nll_{c}"] = seed_mean(lambda s, c=c: weighted_mean(unit.nll[student(c, s)], wk))
    for c in unit.kd_conditions + ["direct"]:
        for teacher in ("A", "B"):
            out[f"rho_{c}_{teacher}"] = seed_mean(
                lambda s, c=c, t=teacher: weighted_pearson(unit.rank[student(c, s)], unit.rank[f"teacher{t}"], weights))
    return out


def fixed_effects_slope(x: np.ndarray, y: np.ndarray, group: np.ndarray) -> float:
    """OLS slope of y on x with one intercept per group."""
    frame = pd.DataFrame({"x": x, "y": y, "g": group})
    centred = frame[["x", "y"]] - frame.groupby("g")[["x", "y"]].transform("mean")
    sxx = float((centred.x ** 2).sum())
    return float((centred.x * centred.y).sum() / sxx) if sxx > 0 else float("nan")


def hypothesis_components(table: pd.DataFrame) -> dict[str, float]:
    """Pooled components (all oriented so that > 0 supports the hypothesis) from per-unit statistics."""
    m = table.mean(numeric_only=True)
    direct_a_minus_b = m.rho_direct_A - m.rho_direct_B  # how much more any student follows A than B
    # H1 uses the conventional-KD arm (T = 4) when it was trained; the tuned arm is then reported only
    tested, reported = ("kdA4", "kdB4"), ("kdA", "kdB")
    if f"rho_{tested[0]}_A" not in m:
        tested, reported = reported, None
    out = {
        # shift toward the own teacher, relative to the directly trained student (difference in differences)
        f"H1:{tested[0]}_shift_to_A": (m[f"rho_{tested[0]}_A"] - m[f"rho_{tested[0]}_B"]) - direct_a_minus_b,
        f"H1:{tested[1]}_shift_to_B": (m[f"rho_{tested[1]}_B"] - m[f"rho_{tested[1]}_A"]) + direct_a_minus_b,
    }
    if reported is not None:
        out[f"info:{reported[0]}_shift_to_A"] = (m[f"rho_{reported[0]}_A"] - m[f"rho_{reported[0]}_B"]) - direct_a_minus_b
        out[f"info:{reported[1]}_shift_to_B"] = (m[f"rho_{reported[1]}_B"] - m[f"rho_{reported[1]}_A"]) + direct_a_minus_b
    for score in SCORES:
        a = lambda c: m[f"auroc_{score}_{c}"]  # noqa: E731
        gap = table[f"auroc_{score}_teacherA"] - table[f"auroc_{score}_kdA"]
        out.update({
            f"H2[{score}]:auroc_kdA_minus_ls": a("kdA") - a("ls"),
            f"H2[{score}]:auroc_kdA_minus_directTS": a("kdA") - a("directTS"),
            f"H2[{score}]:nll_ls_minus_kdA": m.nll_ls - m.nll_kdA,
            f"H2[{score}]:nll_directTS_minus_kdA": m.nll_directTS - m.nll_kdA,
            f"H3[{score}]:gap_slope_per_week": fixed_effects_slope(
                table.weeks_since.to_numpy(), gap.to_numpy(), table.start.to_numpy()),
            f"H5[{score}]:auroc_enddA_minus_kdA": a("enddA") - a("kdA"),
            f"info:mean_gap_teacherA_minus_kdA_{score}": float(gap.mean()),
        })
    out.update({
        # reported, not tested
        "info:raw_kdA_own_minus_other": m.rho_kdA_A - m.rho_kdA_B,
        "info:raw_kdB_own_minus_other": m.rho_kdB_B - m.rho_kdB_A,
        "info:f1_kdA_minus_ls": m.f1_kdA - m.f1_ls,  # the accuracy match behind H2 (decision D3)
        "info:f1_kdA_minus_directTS": m.f1_kdA - m.f1_directTS,
        "info:ece_ls_minus_kdA": m.ece_ls - m.ece_kdA,
        "info:ece_directTS_minus_kdA": m.ece_directTS - m.ece_kdA,
    })
    return out


def _table(units: list[Unit], weights_of, seed_counts_of) -> pd.DataFrame:
    rows = []
    for unit in units:
        stats = unit_statistics(unit, weights_of(unit), seed_counts_of(unit))
        rows.append({"start": unit.start, "split": unit.split, "weeks_since": unit.weeks_since, **stats})
    return pd.DataFrame(rows)


def pooled_bootstrap(units: list[Unit], n_clusters: int, n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimates, percentile intervals and one-sided p-values of every component.

    Returns (components, per-unit statistics at the point estimate).
    """
    starts = sorted({u.start for u in units})
    seeds_by_start = {s: next(u.seeds for u in units if u.start == s) for s in starts}
    ones = {s: np.ones(len(seeds_by_start[s])) for s in starts}
    point_table = _table(units, lambda u: np.ones(len(u)), lambda u: ones[u.start])
    point = hypothesis_components(point_table)

    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_boot) for k in point}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        seed_counts = {s: np.bincount(rng.integers(0, len(v), size=len(v)), minlength=len(v))
                       for s, v in seeds_by_start.items()}
        table = _table(units, lambda u: counts[u.cluster], lambda u: seed_counts[u.start])
        for k, v in hypothesis_components(table).items():
            draws[k][b] = v
        if progress and (b + 1) % max(1, n_boot // 10) == 0:
            progress(f"  bootstrap {b + 1}/{n_boot}")

    rows = []
    for k, estimate in point.items():
        d = draws[k][np.isfinite(draws[k])]
        low, high = np.quantile(d, [alpha / 2, 1 - alpha / 2]) if len(d) else (np.nan, np.nan)
        p = (1 + np.sum(d <= 0)) / (len(d) + 1) if len(d) and not k.startswith("info:") else np.nan
        rows.append({"hypothesis": k.split(":")[0], "component": k.split(":")[1], "estimate": estimate,
                     "ci_low": low, "ci_high": high, "p_one_sided": p, "n_boot": len(d)})
    return pd.DataFrame(rows), point_table


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (NaN entries are left out of the family)."""
    valid = sorted(((p, k) for k, p in p_values.items() if np.isfinite(p)))
    m, adjusted, running = len(valid), {}, 0.0
    for i, (p, k) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return {k: adjusted.get(k, float("nan")) for k in p_values}


def summarise_hypotheses(components: pd.DataFrame, extra_p: dict[str, float] | None = None,
                         alpha: float = 0.05) -> pd.DataFrame:
    """One row per hypothesis: intersection-union p-value, Holm-adjusted p-value and decision."""
    tested = components[components.hypothesis != "info"]
    p = tested.groupby("hypothesis")["p_one_sided"].max().to_dict()
    p.update(extra_p or {})
    adjusted = holm(p)
    rows = [{"hypothesis": h, "p_iut": p[h], "p_holm": adjusted[h],
             "supported": bool(np.isfinite(adjusted[h]) and adjusted[h] < alpha)} for h in sorted(p)]
    return pd.DataFrame(rows)


# Shortcut experiment (H4a / H4b; decision D4) ---------------------------------------------------

def shortcut_components(reliance: pd.DataFrame, shortcut: pd.DataFrame, rhos=(0.9, 1.0)) -> pd.DataFrame:
    """Paired one-sided t-tests over (rho, seed) pairs; returns component rows like pooled_bootstrap."""
    from scipy.stats import ttest_1samp

    def test(hypothesis: str, component: str, diffs: np.ndarray) -> dict:
        diffs = diffs[np.isfinite(diffs)]
        if len(diffs) >= 2 and np.ptp(diffs) > 0:
            result = ttest_1samp(diffs, 0.0, alternative="greater")
            ci = result.confidence_interval(0.95)
            p, low = float(result.pvalue), float(ci.low)
        else:
            p = low = float("nan")
        return {"hypothesis": hypothesis, "component": component, "estimate": float(np.mean(diffs)) if len(diffs) else np.nan,
                "ci_low": low, "ci_high": np.nan, "p_one_sided": p, "n_boot": len(diffs)}

    rel = reliance[reliance.rho.astype(float).isin(rhos)]
    wide = rel.pivot_table(index=["rho", "seed"], columns="model", values="reliance_f1_drop")
    rows = [test("H4a", "reliance_kd_minus_direct",
                 (wide.get("student_both_kd") - wide.get("student_both_direct")).to_numpy(dtype=float))]

    view = shortcut[(shortcut.setting == "teacher_only") & (shortcut.model == "student_teacheronly_kd")
                    & (shortcut.flows == "all") & (shortcut.unknown == "all")].copy()
    view["rho"] = view.rho.astype(float)
    base = view[view.rho == 0.0].set_index("seed")
    shifted = view[view.rho.isin(rhos)].join(base[["ece", "auroc_energy"]], on="seed", rsuffix="_rho0")
    rows.append(test("H4b", "ece_minus_rho0", (shifted.ece - shifted.ece_rho0).to_numpy(dtype=float)))
    rows.append(test("H4b", "rho0_minus_auroc", (shifted.auroc_energy_rho0 - shifted.auroc_energy).to_numpy(dtype=float)))
    return pd.DataFrame(rows)
