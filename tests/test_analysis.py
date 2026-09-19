import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from kdtraffic import metrics
from kdtraffic.analysis import (RankedScore, assign_clusters, build_unit, confidence_bins, fixed_effects_slope, holm,
                                pooled_bootstrap, ranks, shortcut_components, summarise_hypotheses, weighted_ece,
                                weighted_macro_f1, weighted_pearson)
from kdtraffic.data import sequence_keys


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_weighted_auroc_matches_repeated_flows(rng):
    score = np.round(rng.normal(size=400), 1)  # many ties
    known = rng.random(400) < 0.6
    score[known] += 0.5
    weights = rng.integers(0, 4, size=400)
    ranked = RankedScore(score, known)
    assert ranked.auroc(np.ones(400)) == pytest.approx(roc_auc_score(known, score))
    rep = np.repeat(np.arange(400), weights)
    assert ranked.auroc(weights) == pytest.approx(roc_auc_score(known[rep], score[rep]))


def test_weighted_macro_f1_and_ece_match_repeated_flows(rng):
    y = rng.integers(0, 6, size=500)
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, 8, size=500))
    probs = rng.dirichlet(np.ones(8), size=500)
    probs[np.arange(500), pred] += 2
    probs /= probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)
    weights = rng.integers(0, 3, size=500)
    rep = np.repeat(np.arange(500), weights)
    assert weighted_macro_f1(y, pred, np.ones(500), 8) == pytest.approx(metrics.closed_set_metrics(y, pred)["macro_f1"])
    assert weighted_macro_f1(y, pred, weights, 8) == pytest.approx(metrics.closed_set_metrics(y[rep], pred[rep])["macro_f1"])
    correct = (pred == y).astype(float)
    bins = confidence_bins(confidence)
    assert weighted_ece(confidence, correct, bins, weights) == pytest.approx(
        metrics.expected_calibration_error(probs[rep], y[rep]))


def test_weighted_rank_correlation_is_spearman_at_unit_weights(rng):
    a = rng.normal(size=300)
    b = a + rng.normal(size=300)
    assert weighted_pearson(ranks(a), ranks(b), np.ones(300)) == pytest.approx(spearmanr(a, b).statistic)


def test_fixed_effects_slope_ignores_group_offsets():
    x = np.array([1, 2, 3, 1, 2, 3], dtype=float)
    y = 0.5 * x + np.array([0, 0, 0, 10, 10, 10])
    assert fixed_effects_slope(x, y, np.array([0, 0, 0, 1, 1, 1])) == pytest.approx(0.5)


def test_holm():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
    assert np.isnan(adjusted["d"])


def test_sequence_keys_find_exact_duplicates(rng):
    ppi = rng.random((50, 3, 30)).astype(np.float32)
    other = ppi.copy()
    other[::2, 2, 5] += 1.0
    assert np.array_equal(np.isin(sequence_keys(other), sequence_keys(ppi)), np.arange(50) % 2 == 1)


def synthetic_scores(rng, n=3000, classes=5, seeds=(0, 1, 2), drift=0.0):
    """Teacher A detects unknowns well; kdA follows A, kdB follows B; ls and directTS are noisier."""
    y = np.where(rng.random(n) < 0.7, rng.integers(0, classes, size=n), -1)
    known = y >= 0
    base_a, base_b = rng.normal(size=n), rng.normal(size=n)
    data = {"y": y, "app": np.where(known, y, 100 + rng.integers(0, 3, size=n)).astype(str),
            "day": rng.integers(0, 10, size=n) + 20220101, "ppi_len": rng.integers(1, 30, size=n),
            "duplicate": rng.random(n) < 0.1}

    def add(model, signal, own, noise, accuracy):
        energy = signal * known + own + noise * rng.normal(size=n)
        pred = np.where(rng.random(n) < accuracy, np.maximum(y, 0), rng.integers(0, classes, size=n))
        confidence = np.clip(accuracy + 0.05 * rng.normal(size=n), 0.01, 1.0)
        data[f"{model}__energy"] = energy
        data[f"{model}__pred"] = pred
        data[f"{model}__msp"] = confidence
        data[f"{model}__msp_ts"] = confidence
        p_true = np.where(pred == y, confidence, (1 - confidence) / (classes - 1))
        data[f"{model}__nll_ts"] = np.where(known, -np.log(np.clip(p_true, 1e-12, 1.0)), 0.0)

    add("teacherA", 3.0, base_a, 0.2, 0.9)
    add("teacherB", 3.0, base_b, 0.2, 0.9)
    for s in seeds:
        add(f"student_direct_s{s}", 1.0, 0.0, 1.0, 0.85)
        add(f"student_kdA_s{s}", 2.5 - drift, base_a, 0.5, 0.85)
        add(f"student_kdB_s{s}", 2.5, base_b, 0.5, 0.85)
        add(f"student_enddA_s{s}", 2.6 - drift, base_a, 0.5, 0.85)
        add(f"student_ls_s{s}", 1.0, 0.0, 1.0, 0.85)
        add(f"student_directTS_s{s}", 1.0, 0.0, 1.0, 0.85)
    return data


def test_pooled_bootstrap_detects_planted_effects(rng):
    units, days, apps = [], [], []
    for start in (11, 24):
        for w, weeks_since in enumerate((3.0, 7.0, 11.0)):
            data = synthetic_scores(rng, drift=0.15 * w)
            mask = np.ones(len(data["y"]), dtype=bool)
            units.append(build_unit(data, start, f"test_{w}", weeks_since, 5, mask))
            days.append(data["day"])
            apps.append(data["app"])
    n_clusters = assign_clusters(units, days, apps)
    components, table = pooled_bootstrap(units, n_clusters, n_boot=200, seed=1)  # p >= 1/(B+1)
    assert len(table) == 6
    comp = components.set_index(["hypothesis", "component"])
    assert comp.loc[("H1", "kdA_shift_to_A"), "estimate"] > 0.3
    assert comp.loc[("H1", "kdB_shift_to_B"), "p_one_sided"] < 0.05
    assert comp.loc[("H2[energy]", "auroc_kdA_minus_ls"), "p_one_sided"] < 0.05
    assert comp.loc[("H3[energy]", "gap_slope_per_week"), "estimate"] > 0
    # the planted MSP scores carry no unknown signal, so the MSP versions are null
    assert abs(comp.loc[("H2[msp]", "auroc_kdA_minus_ls"), "estimate"]) < 0.05
    hypotheses = summarise_hypotheses(components).set_index("hypothesis")
    assert set(hypotheses.index) == {"H1", "H2[energy]", "H2[msp]", "H3[energy]", "H3[msp]", "H5[energy]", "H5[msp]"}
    assert hypotheses.loc["H1", "supported"]
    assert not hypotheses.loc["H2[energy]", "supported"]  # the planted NLL components are null
    h2 = set(components[components.hypothesis == "H2[energy]"].component)
    assert h2 == {"auroc_kdA_minus_ls", "auroc_kdA_minus_directTS", "nll_ls_minus_kdA", "nll_directTS_minus_kdA"}


def test_h1_is_relative_to_the_direct_student(rng):
    """Every student follows teacher A more (A is 'central'); only the shift beyond the direct student counts."""
    data = synthetic_scores(rng)
    known = data["y"] >= 0
    central = data["teacherA__energy"]
    for s in (0, 1, 2):
        for c in ("direct", "kdB"):
            data[f"student_{c}_s{s}__energy"] = data[f"student_{c}_s{s}__energy"] + 2.0 * central * (c == "direct")
    unit = build_unit(data, 11, "test_0", 3.0, 5, np.ones(len(known), dtype=bool))
    assign_clusters([unit], [data["day"]], [data["app"]])
    components, _ = pooled_bootstrap([unit], int(unit.cluster.max()) + 1, n_boot=20, seed=0)
    comp = components.set_index(["hypothesis", "component"])
    assert comp.loc[("info", "raw_kdB_own_minus_other"), "estimate"] > 0
    assert comp.loc[("H1", "kdB_shift_to_B"), "estimate"] > comp.loc[("info", "raw_kdB_own_minus_other"), "estimate"]


def test_shortcut_components():
    reliance = pd.DataFrame([
        {"model": m, "rho": r, "seed": s, "reliance_f1_drop": d + 0.01 * s}
        for r in (0.0, 0.9, 1.0) for s in range(3)
        for m, d in (("student_both_kd", 0.3 * r), ("student_both_direct", 0.1 * r), ("teacher", 0.3 * r))
    ])
    shortcut = pd.DataFrame([
        {"model": "student_teacheronly_kd", "setting": "teacher_only", "flows": "all", "unknown": "all",
         "rho": r, "seed": s, "ece": 0.05 + 0.1 * r + 0.001 * s, "auroc_energy": 0.9 - 0.05 * r - 0.002 * s}
        for r in (0.0, 0.9, 1.0) for s in range(3)
    ])
    comp = shortcut_components(reliance, shortcut).set_index("component")
    assert comp.loc["reliance_kd_minus_direct", "estimate"] > 0
    assert comp.loc["reliance_kd_minus_direct", "p_one_sided"] < 0.01
    assert comp.loc["ece_minus_rho0", "p_one_sided"] < 0.01
    assert set(comp.hypothesis) == {"H4a", "H4b1", "H4b2"}  # decision D4: H4b is split in two
