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
    3. test window      -> the same, with the test unknown services (only when with_test is set)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd

from kdtraffic.splits import Splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("KD_DATA_ROOT", "C:/datasets"))
DIR_CHANNEL = 1  # PPI channels: inter-packet time, direction, size
CACHE_VERSION = 2  # bump when the content of cached arrays changes
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

    def __len__(self) -> int:
        return len(self.y)

    @property
    def ppi_len(self) -> np.ndarray:
        """Number of packets in the sequence (padding has direction 0)."""
        return (self.ppi[:, DIR_CHANNEL, :] != 0).sum(axis=1)

    def subset(self, index: np.ndarray) -> "Arrays":
        return Arrays(self.ppi[index], self.flowstats[index], self.y[index], self.app[index])

    def save(self, path: Path) -> None:
        np.savez(path, ppi=self.ppi, flowstats=self.flowstats, y=self.y, app=self.app)

    @classmethod
    def load(cls, path: Path) -> "Arrays":
        with np.load(path) as data:
            return cls(data["ppi"], data["flowstats"], data["y"], data["app"])


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


def week_dates(dataset, weeks: tuple[int, int]) -> list[str]:
    dates: list[str] = []
    for week in range(weeks[0], weeks[1] + 1):
        dates.extend(dataset.time_periods[f"W-2022-{week}"])
    return dates


def period_name(weeks: tuple[int, int]) -> str:
    return f"W-2022-{weeks[0]}-{weeks[1]}"


def _to_numpy(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _collect(loader, keep: set[str], label_index: dict[str, int], name: str, log: Callable[[str], None]) -> Arrays:
    keep_array = np.array(sorted(keep))
    ppis, stats, apps = [], [], []
    seen, started = 0, time.time()
    for _, x_ppi, x_flowstats, labels in loader:
        labels = np.asarray(labels).astype(str)
        seen += len(labels)
        mask = np.isin(labels, keep_array)
        if mask.any():
            ppis.append(_to_numpy(x_ppi)[mask].astype(np.float32, copy=False))
            stats.append(_to_numpy(x_flowstats)[mask].astype(np.float32, copy=False))
            apps.append(labels[mask])
    if not apps:
        raise RuntimeError(f"No flows collected for the {name} set")
    app = np.concatenate(apps)
    y = pd.Series(app).map(label_index).fillna(-1).astype(np.int64).to_numpy()
    log(f"  {name}: kept {len(app):,} of {seen:,} flows ({int((y < 0).sum()):,} unknown) in {time.time() - started:.0f}s")
    return Arrays(np.concatenate(ppis), np.concatenate(stats), y, app)


def _counts(arrays: Arrays) -> dict[str, int]:
    return {
        "flows": len(arrays),
        "unknown_flows": int((arrays.y < 0).sum()),
        "services": int(len(np.unique(arrays.app))),
    }


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
    dataset_root = str(Path(data_root) / "CESNET-TLS-Year22")
    label_index = {app: i for i, app in enumerate(splits.known)}
    known = set(splits.known)
    common = dict(
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

    # Stage 1: training window (known services only); fits the scalers.
    # FIXED selection uses only the listed services and ignores DataZoo's min-train-samples check,
    # which is therefore done below.
    dataset = CESNET_TLS_Year22(dataset_root, size=spec.size, silent=True)
    splits.validate(available=list(dataset.available_classes))
    dataset.set_dataset_config_and_initialize(DatasetConfig(
        dataset=dataset,
        train_period_name=period_name(spec.train_weeks),
        train_dates=week_dates(dataset, spec.train_weeks),
        need_val_set=False,
        need_test_set=False,
        apps_selection=AppSelection.FIXED,
        apps_selection_fixed_known=list(splits.known),
        train_size=spec.train_size,
        train_dataloader_order=DataLoaderOrder.SEQUENTIAL,
        ppi_transform=ClipAndScalePPI(),
        flowstats_transform=ClipAndScaleFlowstats(),
        flowstats_phist_transform=NormalizeHistograms(),
        **common,
    ))
    fitted = dataset.dataset_config
    log(f"  training configuration initialised in {time.time() - started:.0f}s")
    train = _collect(dataset.get_train_dataloader(), known, label_index, "train", log)
    train_counts = pd.Series(train.app).value_counts()
    too_few = sorted(s for s in splits.known if train_counts.get(s, 0) < spec.min_train_samples)
    if too_few:
        raise RuntimeError(f"Known services with fewer than {spec.min_train_samples} training flows: {too_few}")

    def evaluation_set(name: str, weeks: tuple[int, int], unknown: list[str], known_size: Size, unknown_size: Size) -> Arrays:
        stage = CESNET_TLS_Year22(dataset_root, size=spec.size, silent=True)
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
            ppi_transform=fitted.ppi_transform,  # fitted on the training window
            flowstats_transform=fitted.flowstats_transform,
            flowstats_phist_transform=fitted.flowstats_phist_transform,
            **common,
        ))
        arrays = _collect(stage.get_test_dataloader(), known | set(unknown), label_index, name, log)
        if unknown and not (arrays.y < 0).any():
            raise RuntimeError(f"The {name} set has no flows of its unknown services {unknown}")
        return arrays

    # Stage 2: validation week with the validation unknown services.
    val = evaluation_set("val", spec.val_weeks, splits.val_unknown, spec.val_known_size, spec.val_unknown_size)
    # Stage 3: test window with the test unknown services, only once the pre-registration is frozen.
    test = None
    if spec.with_test:
        test = evaluation_set("test", spec.test_weeks, splits.test_unknown, spec.test_known_size, spec.test_unknown_size)

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
