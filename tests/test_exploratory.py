import numpy as np
import pytest

from kdtraffic.analysis import assign_clusters
from kdtraffic.exploratory import (build_unit, calendar_overlap, cluster_bootstrap_many,
                                   cluster_bootstrap_table, detection_shift, merge_scores, seeds_of,
                                   shift, teacher_advantage, teacher_preference_null)


def scores_for(rng, n=600, models=("teacherA", "teacherB"), students=(), seeds=(0, 1, 2), signal=None):
    y = np.where(rng.random(n) < 0.7, rng.integers(0, 5, size=n), -1)
    data = {"y": y, "app": np.where(y >= 0, "known", "unknown"),
            "day": (20220101 + rng.integers(0, 7, size=n)).astype(np.int32),
            "duplicate": np.zeros(n, bool), "ppi_len": rng.integers(1, 30, size=n).astype(np.int8)}
    base = {m: rng.normal(size=n) for m in models}
    for m, value in base.items():
        data[f"{m}__energy"] = value
        data[f"{m}__msp"] = rng.random(n)
    for condition in students:
        for seed in seeds:
            follow = (signal or {}).get(condition)
            energy = rng.normal(size=n) * 0.5
            if follow:
                energy = energy + 2.0 * base[follow]
            data[f"student_{condition}_s{seed}__energy"] = energy
            data[f"student_{condition}_s{seed}__msp"] = rng.random(n)
    return data


def make_unit(data, start=11, split="test_w16-19", weeks_since=3.5):
    models = sorted({k.split("__")[0] for k in data if k.endswith("__energy")})
    unit = build_unit(data, start, split, weeks_since, models, models)
    assign_clusters([unit], [data["day"]], [data["app"]])
    return unit


def test_seeds_of_reads_only_matching_conditions():
    keys = ["student_direct_s0__energy", "student_direct_s2__energy", "student_kdA_s1__energy",
            "student_direct_sX__energy", "teacherA__energy"]
    assert seeds_of(keys, "direct") == [0, 2]
    assert seeds_of(keys, "kdA") == [1]
    assert seeds_of(keys, "missing") == []


def test_shift_detects_which_teacher_a_student_follows():
    rng = np.random.default_rng(0)
    data = scores_for(rng, students=("direct", "kdA4", "kdB4"),
                      signal={"kdA4": "teacherA", "kdB4": "teacherB"})
    unit = make_unit(data)
    weights = np.ones(len(unit))
    assert shift(unit, "kdA4", "teacherA", "teacherB", weights) > 0.2
    assert shift(unit, "kdB4", "teacherB", "teacherA", weights) > 0.2
    # the undistilled student follows neither, so its shift against itself is zero by construction
    assert shift(unit, "direct", "teacherA", "teacherB", weights) == pytest.approx(0.0, abs=1e-12)


def test_shift_is_nan_when_the_condition_is_absent():
    unit = make_unit(scores_for(np.random.default_rng(1), students=("direct",)))
    assert np.isnan(shift(unit, "kdC", "teacherA", "teacherB", np.ones(len(unit))))


def test_teacher_preference_null_is_small_for_unrelated_teachers():
    rng = np.random.default_rng(2)
    members = [f"teacherA_{i}" for i in range(3)]
    data = scores_for(rng, models=tuple(["teacherA", "teacherB"] + members), students=("direct",))
    unit = make_unit(data)
    null = teacher_preference_null(unit, members, np.ones(len(unit)))
    assert len(null) == 3 * 3  # three pairs per seed
    assert max(null) < 0.3


def test_merge_scores_accepts_the_same_teacher_scored_twice(tmp_path):
    """Every run re-scores the same teachers, so identical arrays in two files are legitimate."""
    rng = np.random.default_rng(3)
    first = scores_for(rng, students=("direct",))
    np.savez(tmp_path / "a.npz", **first)
    np.savez(tmp_path / "b.npz", **first)
    merged = merge_scores([tmp_path / "a.npz", tmp_path / "b.npz"])
    assert np.array_equal(merged["teacherA__energy"], first["teacherA__energy"])


def test_merge_scores_tolerates_float16_rounding_but_not_a_real_difference(tmp_path):
    rng = np.random.default_rng(3)
    first = scores_for(rng, students=("direct",))
    rounded = dict(first)
    rounded["teacherA__energy"] = first["teacherA__energy"].astype(np.float16).astype(np.float64)
    np.savez(tmp_path / "a.npz", **first)
    np.savez(tmp_path / "b.npz", **rounded)
    merge_scores([tmp_path / "a.npz", tmp_path / "b.npz"])  # rounding alone must not stop the analysis

    different = dict(first)
    different["teacherA__energy"] = first["teacherA__energy"] + 0.5  # a genuinely different model
    np.savez(tmp_path / "c.npz", **different)
    with pytest.raises(SystemExit, match="different values"):
        merge_scores([tmp_path / "a.npz", tmp_path / "c.npz"])


def test_merge_scores_combines_distinct_conditions(tmp_path):
    rng = np.random.default_rng(4)
    first = scores_for(rng, students=("direct",))
    second = {k: v for k, v in scores_for(rng, students=("kdM0",)).items() if k.startswith("student_kdM0")}
    second.update({k: first[k] for k in ("y", "app", "day", "duplicate", "ppi_len")})
    np.savez(tmp_path / "a.npz", **first)
    np.savez(tmp_path / "b.npz", **second)
    merged = merge_scores([tmp_path / "a.npz", tmp_path / "b.npz"])
    assert seeds_of(merged.keys(), "direct") == [0, 1, 2]
    assert seeds_of(merged.keys(), "kdM0") == [0, 1, 2]


def test_merge_scores_rejects_different_flows(tmp_path):
    rng = np.random.default_rng(5)
    np.savez(tmp_path / "a.npz", **scores_for(rng, n=100))
    np.savez(tmp_path / "b.npz", **scores_for(rng, n=120))
    with pytest.raises(SystemExit, match="different flows"):
        merge_scores([tmp_path / "a.npz", tmp_path / "b.npz"])


def test_calendar_overlap_counts_weeks_shared_between_start_dates():
    rng = np.random.default_rng(6)
    units = [make_unit(scores_for(rng), start=11, split="test_w16-19"),
             make_unit(scores_for(rng), start=11, split="test_w20-23"),
             make_unit(scores_for(rng), start=24, split="test_w20-23")]
    table = calendar_overlap(units).set_index(["start", "split"])
    assert table.loc[(11, "test_w16-19"), "weeks_shared_with_another_start"] == 0
    assert table.loc[(11, "test_w20-23"), "weeks_shared_with_another_start"] == 4
    assert table.loc[(24, "test_w20-23"), "n_weeks"] == 4


def test_detection_shift_matches_a_direct_auroc_difference():
    """The shift must equal the plain AUROC difference computed from the same ranked scores."""
    rng = np.random.default_rng(3)
    data = scores_for(rng, students=("direct", "kdA4"), seeds=(0,))
    unit = make_unit(data)
    weights = np.ones(len(unit))
    got = detection_shift(unit, "kdA4", "direct", "energy", weights)
    expected = (unit.detect["energy:student_kdA4_s0"].auroc(weights)
                - unit.detect["energy:student_direct_s0"].auroc(weights))
    assert got == pytest.approx(expected)
    # a score no model carries yields nan rather than a silent zero
    assert np.isnan(detection_shift(unit, "kdA4", "direct", "maha", weights))


def test_teacher_advantage_is_teacher_minus_student():
    rng = np.random.default_rng(4)
    data = scores_for(rng, students=("direct",), seeds=(0, 1))
    unit = make_unit(data)
    weights = np.ones(len(unit))
    teacher = unit.detect["energy:teacherA"].auroc(weights)
    students = [unit.detect[f"energy:student_direct_s{s}"].auroc(weights) for s in (0, 1)]
    assert teacher_advantage(unit, "teacherA", "direct", "energy", weights) == pytest.approx(
        teacher - np.mean(students))
    # dropping the second seed in a resample must use the first alone
    assert teacher_advantage(unit, "teacherA", "direct", "energy", weights,
                             seed_counts=np.array([1.0, 0.0])) == pytest.approx(teacher - students[0])
    assert np.isnan(teacher_advantage(unit, "teacherC", "direct", "energy", weights))


def test_cluster_bootstrap_table_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    data = scores_for(rng, students=("direct", "kdA4"), signal={"kdA4": "teacherA"})
    unit = make_unit(data)
    n_clusters = int(unit.cluster.max()) + 1
    result = cluster_bootstrap_table(
        [unit], lambda u, w, sc, cache: shift(u, "kdA4", "teacherA", "teacherB", w, sc),
        n_clusters, n_boot=40, seed=1)
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
    assert result["p_one_sided"] < 0.1
    assert result["n_boot"] == 40


def test_bootstrapping_many_statistics_equals_bootstrapping_them_one_by_one():
    """Sharing one resampling loop must not change any statistic's interval."""
    rng = np.random.default_rng(9)
    data = scores_for(rng, students=("direct", "kdA4", "kdB4"),
                      signal={"kdA4": "teacherA", "kdB4": "teacherB"})
    unit = make_unit(data)
    n_clusters = int(unit.cluster.max()) + 1
    statistics = {
        "a": lambda u, w, sc, cache: shift(u, "kdA4", "teacherA", "teacherB", w, sc),
        "b": lambda u, w, sc, cache: shift(u, "kdB4", "teacherB", "teacherA", w, sc),
    }
    together = cluster_bootstrap_many([unit], statistics, n_clusters, n_boot=25, seed=5)
    for name, fn in statistics.items():
        alone = cluster_bootstrap_table([unit], fn, n_clusters, n_boot=25, seed=5)
        assert together[name] == pytest.approx(alone)


def test_the_draw_cache_does_not_change_a_statistic():
    rng = np.random.default_rng(13)
    data = scores_for(rng, students=("direct", "kdA4"), signal={"kdA4": "teacherA"})
    unit = make_unit(data)
    weights = np.ones(len(unit))
    cached = detection_shift(unit, "kdA4", "direct", "energy", weights, cache={})
    plain = detection_shift(unit, "kdA4", "direct", "energy", weights, cache=None)
    assert cached == pytest.approx(plain)


def test_seed_counts_weight_the_seeds_and_a_dropped_seed_is_ignored():
    """A bootstrap resample that drops a seed must equal the statistic over the seeds it kept."""
    rng = np.random.default_rng(11)
    data = scores_for(rng, students=("direct", "kdA4"), signal={"kdA4": "teacherA"}, seeds=(0, 1))
    unit = make_unit(data)
    weights = np.ones(len(unit))
    only_first = shift(unit, "kdA4", "teacherA", "teacherB", weights, seed_counts=np.array([1.0, 0.0]))
    both = shift(unit, "kdA4", "teacherA", "teacherB", weights, seed_counts=np.array([1.0, 1.0]))
    doubled = shift(unit, "kdA4", "teacherA", "teacherB", weights, seed_counts=np.array([2.0, 2.0]))
    assert doubled == pytest.approx(both)
    assert only_first != pytest.approx(both)
    assert shift(unit, "kdA4", "teacherA", "teacherB", weights) == pytest.approx(both)
