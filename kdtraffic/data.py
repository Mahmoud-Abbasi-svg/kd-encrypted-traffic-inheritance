"""Loading CESNET-TLS-Year22 through cesnet-datazoo into in-memory arrays, with an on-disk cache.

Flows are read once through DataZoo's dataloaders (packet sequences and flow statistics scaled by
transforms fitted on the training window), filtered to the configured known / unknown services,
and saved as .npz files. Training then works on plain arrays, which is much faster than
re-reading the HDF5 database every epoch and avoids DataZoo's memory-hungry dataframes.

DataZoo serves unknown-service flows only through its *test* dataloader (its validation loader
returns known flows only). The data is therefore loaded in stages, each with its own DataZoo
configuration and FIXED service selection:
    1. training window  -> train set; fits the scalers
    2. validation week  -> served as the test period of a second configuration (known + validation
                           unknown services), reusing the fitted scalers
    3. test windows     -> the same, with the test unknown services (see load_window)
Evaluation sets also record the day of each flow, for day x service cluster bootstraps.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, Union

import numpy as np
import pandas as pd

from kdtraffic.splits import Splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("KD_DATA_ROOT", "C:/datasets"))
DIR_CHANNEL = 1  # PPI channels: inter-packet time, direction, size
CACHE_VERSION = 3  # bump when the content of cached arrays changes
LOW_COVERAGE_WEEKS = (50, 52)  # less than half the median weekly volume (results/week1/weekly_totals.csv)
Size = Union[int, str]  # an integer or "all"


@dataclass
class DataSpec:
    size: str = "S"
    train_weeks: tuple[int, int] = (11, 14)
    val_weeks: tuple[int, int] = (15, 15)
    test_weeks: tuple[int, int] = (16, 19)
    with_test: bool = False
    train_size: Size = "all"
    val_known_size: Size = 200_000
    val_unknown_size: Size = "all"
    test_known_size: Size = 400_000
    test_unknown_size: Size = "all"
    min_train_samples: int = 100
    seed: int = 420

    def cache_key(self, splits_digest: str) -> str:
        payload = json.dumps({**asdict(self), "splits": splits_digest, "cache_version": CACHE_VERSION}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


@dataclass
class Arrays:
    ppi: np.ndarray  # (N, 3, 30) float32, scaled
    flowstats: np.ndarray  # (N, F) float32, scaled
    y: np.ndarray  # (N,) int64 class index; -1 for unknown services
    app: np.ndarray  # (N,) service tag
    day: np.ndarray | None = None  # (N,) int yyyymmdd; evaluation sets only

    def __len__(self) -> int:
        return len(self.y)

    @property
    def ppi_len(self) -> np.ndarray:
        """Number of packets in the sequence (padding has direction 0)."""
        return (self.ppi[:, DIR_CHANNEL, :] != 0).sum(axis=1)

    def subset(self, index: np.ndarray) -> "Arrays":
        day = None if self.day is None else self.day[index]
        return Arrays(self.ppi[index], self.flowstats[index], self.y[index], self.app[index], day)

    def with_flowstats(self, flowstats: np.ndarray) -> "Arrays":
        return Arrays(self.ppi, flowstats, self.y, self.app, self.day)

    def save(self, path: Path) -> None:
        extra = {} if self.day is None else {"day": self.day}
        np.savez(path, ppi=self.ppi, flowstats=self.flowstats, y=self.y, app=self.app, **extra)

    @classmethod
    def load(cls, path: Path) -> "Arrays":
        with np.load(path) as data:
            day = data["day"] if "day" in data.files else None
            return cls(data["ppi"], data["flowstats"], data["y"], data["app"], day)


@dataclass
class DataBundle:
    spec: DataSpec
    known: list[str]
    train: Arrays
    val: Arrays
    test: Arrays | None
    cache_dir: Path
    meta: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.known)

    @property
    def flowstats_dim(self) -> int:
        return int(self.train.flowstats.shape[1])


def week_range(weeks: tuple[int, int]) -> list[int]:
    return list(range(weeks[0], weeks[1] + 1))


def week_dates(dataset, weeks: Iterable[int]) -> list[str]:
    dates: list[str] = []
    for week in weeks:
        dates.extend(dataset.time_periods[f"W-2022-{week}"])
    return dates


def period_name(weeks: Sequence[int]) -> str:
    """Unique name for a set of weeks (DataZoo caches indices by period name)."""
    weeks = sorted(weeks)
    name = f"W-2022-{weeks[0]}-{weeks[-1]}"
    skipped = sorted(set(range(weeks[0], weeks[-1] + 1)) - set(weeks))
    return name + ("-skip" + "-".join(map(str, skipped)) if skipped else "")


def evaluation_windows(val_weeks: tuple[int, int], length: int = 4, last_week: int = 52,
                       excluded: Sequence[int] = LOW_COVERAGE_WEEKS) -> list[list[int]]:
    """Consecutive `length`-week test windows after the validation week, without low-coverage weeks."""
    windows, start = [], val_weeks[1] + 1
    while start <= last_week:
        block = [w for w in range(start, min(start + length, last_week + 1)) if w not in excluded]
        if block:
            windows.append(block)
        start += length
    return windows


def _to_numpy(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _collect(loader, keep: set[str], label_index: dict[str, int], name: str, log: Callable[[str], None],
             with_day: bool = False) -> Arrays:
    keep_array = np.array(sorted(keep))
    ppis, stats, apps, days = [], [], [], []
    seen, started = 0, time.time()
    for other, x_ppi, x_flowstats, labels in loader:
        labels = np.asarray(labels).astype(str)
        seen += len(labels)
        mask = np.isin(labels, keep_array)
        if mask.any():
            ppis.append(_to_numpy(x_ppi)[mask].astype(np.float32, copy=False))
            stats.append(_to_numpy(x_flowstats)[mask].astype(np.float32, copy=False))
            apps.append(labels[mask])
            if with_day:
                day = pd.to_datetime(other["TIME_FIRST"]).dt.strftime("%Y%m%d").astype(np.int32).to_numpy()
                days.append(day[mask])
    if not apps:
        raise RuntimeError(f"No flows collected for the {name} set")
    app = np.concatenate(apps)
    y = pd.Series(app).map(label_index).fillna(-1).astype(np.int64).to_numpy()
    log(f"  {name}: kept {len(app):,} of {seen:,} flows ({int((y < 0).sum()):,} unknown) in {time.time() - started:.0f}s")
    return Arrays(np.concatenate(ppis), np.concatenate(stats), y, app, np.concatenate(days) if with_day else None)


def _counts(arrays: Arrays) -> dict[str, int]:
    return {
        "flows": len(arrays),
        "unknown_flows": int((arrays.y < 0).sum()),
        "services": int(len(np.unique(arrays.app))),
    }


def _common_options(spec: DataSpec, workers: int) -> dict:
    return dict(
        use_packet_histograms=True,
        use_tcp_features=False,
        disable_label_encoding=True,
        return_tensors=False,
        batch_size=4096,
        test_batch_size=4096,
        train_workers=workers,
        val_workers=workers,
        test_workers=workers,
        random_state=spec.seed,
    )


def _dataset_root(data_root: Path) -> str:
    return str(Path(data_root) / "CESNET-TLS-Year22")


def _evaluation_arrays(data_root: Path, spec: DataSpec, splits: Splits, weeks: Sequence[int], unknown: list[str],
                       known_size: Size, unknown_size: Size, transforms: tuple, workers: int,
                       name: str, log: Callable[[str], None]) -> Arrays:
    """Known + `unknown` services of `weeks`, served through DataZoo's test loader with pre-fitted scalers."""
    from cesnet_datazoo.config import AppSelection, DatasetConfig
    from cesnet_datazoo.datasets import CESNET_TLS_Year22

    ppi_transform, flowstats_transform, phist_transform = transforms
    stage = CESNET_TLS_Year22(_dataset_root(data_root), size=spec.size, silent=True)
    stage.set_dataset_config_and_initialize(DatasetConfig(
        dataset=stage,
        need_train_set=False,
        need_val_set=False,
        test_period_name=period_name(weeks),
        test_dates=week_dates(stage, weeks),
        apps_selection=AppSelection.FIXED,
        apps_selection_fixed_known=list(splits.known),
        apps_selection_fixed_unknown=list(unknown),
        test_known_size=known_size,
        test_unknown_size=unknown_size,
        return_other_fields=True,  # TIME_FIRST gives the day of each flow
        ppi_transform=ppi_transform,
        flowstats_transform=flowstats_transform,
        flowstats_phist_transform=phist_transform,
        **_common_options(spec, workers),
    ))
    label_index = {app: i for i, app in enumerate(splits.known)}
    arrays = _collect(stage.get_test_dataloader(), set(splits.known) | set(unknown), label_index, name, log,
                      with_day=True)
    if unknown and not (arrays.y < 0).any():
        raise RuntimeError(f"The {name} set has no flows of its unknown services {unknown}")
    return arrays


def rebuild_transforms(meta: dict) -> tuple:
    """Scalers fitted on the training window, reconstructed from the cache metadata."""
    from cesnet_models.transforms import ClipAndScaleFlowstats, ClipAndScalePPI, NormalizeHistograms

    return ClipAndScalePPI(**meta["ppi_transform"]), ClipAndScaleFlowstats(**meta["flowstats_transform"]), NormalizeHistograms()


def build_bundle(
    spec: DataSpec,
    splits: Splits,
    data_root: Path = DATA_ROOT,
    cache_root: Path | None = None,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> DataBundle:
    """Return train / val (/ test) arrays for `spec`, reading them from the cache when available."""
    cache_root = Path(cache_root) if cache_root else Path(data_root) / "cache"
    cache_dir = cache_root / f"tls-year22_{spec.size}_{spec.cache_key(splits.digest)}"
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        log(f"Loading cached arrays from {cache_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        test = Arrays.load(cache_dir / "test.npz") if spec.with_test else None
        return DataBundle(spec, meta["known"], Arrays.load(cache_dir / "train.npz"),
                          Arrays.load(cache_dir / "val.npz"), test, cache_dir, meta)

    from cesnet_datazoo.config import AppSelection, DataLoaderOrder, DatasetConfig
    from cesnet_datazoo.datasets import CESNET_TLS_Year22
    from cesnet_models.transforms import ClipAndScaleFlowstats, ClipAndScalePPI, NormalizeHistograms

    started = time.time()
    log(f"Building arrays with cesnet-datazoo (size {spec.size}); cache: {cache_dir}")
    label_index = {app: i for i, app in enumerate(splits.known)}

    # Stage 1: training window (known services only); fits the scalers.
    # FIXED selection uses only the listed services and ignores DataZoo's min-train-samples check,
    # which is therefore done below.
    dataset = CESNET_TLS_Year22(_dataset_root(data_root), size=spec.size, silent=True)
    splits.validate(available=list(dataset.available_classes))
    dataset.set_dataset_config_and_initialize(DatasetConfig(
        dataset=dataset,
        train_period_name=period_name(week_range(spec.train_weeks)),
        train_dates=week_dates(dataset, week_range(spec.train_weeks)),
        need_val_set=False,
        need_test_set=False,
        apps_selection=AppSelection.FIXED,
        apps_selection_fixed_known=list(splits.known),
        train_size=spec.train_size,
        train_dataloader_order=DataLoaderOrder.SEQUENTIAL,
        ppi_transform=ClipAndScalePPI(),
        flowstats_transform=ClipAndScaleFlowstats(),
        flowstats_phist_transform=NormalizeHistograms(),
        **_common_options(spec, workers),
    ))
    fitted = dataset.dataset_config
    transforms = (fitted.ppi_transform, fitted.flowstats_transform, fitted.flowstats_phist_transform)
    log(f"  training configuration initialised in {time.time() - started:.0f}s")
    train = _collect(dataset.get_train_dataloader(), set(splits.known), label_index, "train", log)
    train_counts = pd.Series(train.app).value_counts()
    too_few = sorted(s for s in splits.known if train_counts.get(s, 0) < spec.min_train_samples)
    if too_few:
        raise RuntimeError(f"Known services with fewer than {spec.min_train_samples} training flows: {too_few}")

    # Stage 2: validation week with the validation unknown services.
    val = _evaluation_arrays(data_root, spec, splits, week_range(spec.val_weeks), splits.val_unknown,
                             spec.val_known_size, spec.val_unknown_size, transforms, workers, "val", log)
    # Stage 3: one test window with the test unknown services, only once the pre-registration is frozen.
    test = None
    if spec.with_test:
        test = _evaluation_arrays(data_root, spec, splits, week_range(spec.test_weeks), splits.test_unknown,
                                  spec.test_known_size, spec.test_unknown_size, transforms, workers, "test", log)

    meta = {
        "spec": asdict(spec),
        "cache_version": CACHE_VERSION,
        "splits_digest": splits.digest,
        "known": list(splits.known),
        "flowstats_dim": int(train.flowstats.shape[1]),
        "counts": {name: _counts(arr) for name, arr in (("train", train), ("val", val), ("test", test)) if arr is not None},
        "ppi_transform": fitted.ppi_transform.to_dict(),
        "flowstats_transform": fitted.flowstats_transform.to_dict(),
        "build_seconds": round(time.time() - started),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    train.save(cache_dir / "train.npz")
    val.save(cache_dir / "val.npz")
    if test is not None:
        test.save(cache_dir / "test.npz")
    meta_path.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")  # written last: marks a complete cache
    log(f"  cached arrays in {time.time() - started:.0f}s total")
    return DataBundle(spec, list(splits.known), train, val, test, cache_dir, meta)


def load_window(bundle: DataBundle, splits: Splits, weeks: Sequence[int], unknown: list[str], kind: str,
                known_size: Size, unknown_size: Size, data_root: Path = DATA_ROOT, workers: int = 4,
                log: Callable[[str], None] = print) -> Arrays:
    """Known + `unknown` services of `weeks`, scaled with the bundle's training scalers; cached in the bundle.

    Callers are responsible for the pre-registration rule: test unknown services only after it is frozen.
    """
    key = hashlib.sha1(json.dumps(sorted(unknown)).encode()).hexdigest()[:8]
    path = bundle.cache_dir / "windows" / f"{kind}_{period_name(weeks)}_{known_size}_{unknown_size}_{key}.npz"
    if path.exists():
        log(f"  {kind} {period_name(weeks)}: loading cached window")
        return Arrays.load(path)
    arrays = _evaluation_arrays(data_root, bundle.spec, splits, weeks, unknown, known_size, unknown_size,
                                rebuild_transforms(bundle.meta), workers, f"{kind} {period_name(weeks)}", log)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays.save(path)
    return arrays
