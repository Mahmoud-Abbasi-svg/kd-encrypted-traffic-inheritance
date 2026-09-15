from pathlib import Path

import pytest

from kdtraffic.splits import Splits, load_splits

SPLITS_PATH = Path(__file__).resolve().parents[1] / "configs" / "splits.json"
CATEGORIES = {"a": "x", "b": "y", "c": "x", "d": "z", "e": "y", "f": "w", "g": "x"}


def make(**overrides) -> Splits:
    fields = dict(known=["a", "b"], val_unknown_near=["c"], val_unknown_far=["d"], test_unknown_near=["e"],
                  test_unknown_far=["f"], disabled=["g"], categories=dict(CATEGORIES))
    fields.update(overrides)
    return Splits(**fields)


def test_valid_split():
    make().validate(available=list("abcdefg"))


def test_overlap_rejected():
    with pytest.raises(ValueError, match="both"):
        make(disabled=["a"]).validate()


def test_far_unknown_sharing_known_category_rejected():
    with pytest.raises(ValueError, match="Far unknown"):
        make(categories={**CATEGORIES, "d": "x"}).validate()


def test_near_unknown_without_known_category_rejected():
    with pytest.raises(ValueError, match="Near unknown"):
        make(categories={**CATEGORIES, "c": "q"}).validate()


def test_unassigned_dataset_service_rejected():
    with pytest.raises(ValueError, match="not assigned"):
        make().validate(available=list("abcdefgh"))


@pytest.mark.skipif(not SPLITS_PATH.exists(), reason="configs/splits.json not generated yet")
def test_frozen_splits_are_consistent():
    splits = load_splits(SPLITS_PATH)
    assert set(splits.val_unknown).isdisjoint(splits.test_unknown)
    assert sum(len(v) for v in splits.groups().values()) == 180
    assert len(splits.known) >= 100
    assert splits.val_unknown_near and splits.val_unknown_far and splits.test_unknown_near and splits.test_unknown_far
