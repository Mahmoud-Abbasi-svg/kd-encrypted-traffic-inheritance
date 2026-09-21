import numpy as np
import pytest

from kdtraffic.analysis import assign_clusters
from kdtraffic.exploratory import (build_unit, calendar_overlap, cluster_bootstrap_table, merge_scores,
                                   seeds_of, shift, teacher_preference_null)


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


def test_merge_scores_refuses_to_overwrite(tmp_path):
    rng = np.random.default_rng(3)
    first = scores_for(rng, students=("direct",))
    np.savez(tmp_path / "a.npz", **first)
    np.savez(tmp_path / "b.npz", **first)  # same keys: a silent overwrite would hide one run
    with pytest.raises(SystemExit, match="redefines"):
        merge_scores([tmp_path / "a.npz", tmp_path / "b.npz"])


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


def test_cluster_bootstrap_table_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    data = scores_for(rng, students=("direct", "kdA4"), signal={"kdA4": "teacherA"})
    unit = make_unit(data)
    n_clusters = int(unit.cluster.max()) + 1
    result = cluster_bootstrap_table([unit], lambda u, w: shift(u, "kdA4", "teacherA", "teacherB", w),
                                     n_clusters, n_boot=40, seed=1)
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
    assert result["p_one_sided"] < 0.1
    assert result["n_boot"] == 40
