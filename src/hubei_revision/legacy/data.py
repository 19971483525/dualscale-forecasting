from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_CALENDAR_FEATURES = [
    "weekday_sin",
    "weekday_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


INTRADAY_DAILY_FEATURES = [
    "daily_load",
    "load_peak",
    "load_valley",
    "load_mean_96",
    "load_std_96",
    "temp_avg",
    "temp_max",
    "temp_min",
    "weekday_sin",
    "weekday_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


GLOBAL_DAILY_FEATURES = [
    "daily_load",
    "load_peak",
    "load_valley",
    "load_mean_96",
    "load_std_96",
    "sales_total",
    "all_industry_total",
    "sales_gap",
    "sales_ratio",
    "temp_avg",
    "temp_max",
    "temp_min",
    "weekday_sin",
    "weekday_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


TARGET_COLUMNS = ["daily_load", "load_peak", "load_valley", "load_mean_96"]


@dataclass
class DailyForecastConfig:
    intraday_history: int = 14
    industry_history: int = 28
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def transform(self, value: np.ndarray) -> np.ndarray:
        return (value - self.mean) / self.std

    def inverse(self, value: np.ndarray) -> np.ndarray:
        return value * self.std + self.mean


class TargetTransform:
    def __init__(self, delta_mean: float, delta_std: float, aux_mean: np.ndarray, aux_std: np.ndarray):
        self.delta_mean = float(delta_mean)
        self.delta_std = float(max(delta_std, 1e-6))
        self.aux_mean = aux_mean.astype(np.float32)
        self.aux_std = np.where(aux_std < 1e-6, 1.0, aux_std).astype(np.float32)

    def transform(self, target: np.ndarray, base_load: np.ndarray) -> np.ndarray:
        out = np.zeros_like(target, dtype=np.float32)
        out[..., 0] = (target[..., 0] - base_load[..., 0] - self.delta_mean) / self.delta_std
        out[..., 1:] = (target[..., 1:] - self.aux_mean) / self.aux_std
        return out.astype(np.float32)

    def inverse(self, pred: np.ndarray, base_load: np.ndarray) -> np.ndarray:
        out = np.zeros_like(pred, dtype=np.float32)
        out[..., 0] = base_load[..., 0] + pred[..., 0] * self.delta_std + self.delta_mean
        out[..., 1:] = pred[..., 1:] * self.aux_std + self.aux_mean
        return out.astype(np.float32)


class DailyForecastDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = indices.astype(int)
        self.dates = arrays["dates"][self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        item = {}
        for key, value in self.arrays.items():
            if key == "dates":
                continue
            tensor = torch.tensor(value[real_idx], dtype=torch.float32)
            item[key] = tensor
        return item


def build_sample_arrays(
    df: pd.DataFrame,
    config: DailyForecastConfig,
    industry_features: List[str],
) -> Dict[str, np.ndarray]:
    df = df.sort_values("date").reset_index(drop=True)
    intraday_curves = df[[f"q_{i:02d}" for i in range(96)]].to_numpy(dtype=np.float32)
    intraday_daily = df[INTRADAY_DAILY_FEATURES].to_numpy(dtype=np.float32)
    industry_daily = df[industry_features].to_numpy(dtype=np.float32)
    global_daily = df[GLOBAL_DAILY_FEATURES].to_numpy(dtype=np.float32)
    targets = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(df["date"]).to_numpy()

    start = max(config.intraday_history, config.industry_history)
    end = len(df) - config.horizon + 1
    sample_dates = []
    intra_curve_list = []
    intra_feat_list = []
    industry_list = []
    global_list = []
    flat_list = []
    target_calendar = []
    target_list = []
    industry_target_list = []
    base_load_list = []

    for target_idx in range(start, end):
        horizon_idx = target_idx + config.horizon - 1
        intra_slice = slice(target_idx - config.intraday_history, target_idx)
        inter_slice = slice(target_idx - config.industry_history, target_idx)

        intra_curve = intraday_curves[intra_slice]
        intra_feat = intraday_daily[intra_slice]
        industry_hist = industry_daily[inter_slice]
        global_hist = global_daily[inter_slice]
        daily_flat = np.concatenate(
            [intraday_curves[inter_slice], intraday_daily[inter_slice], industry_daily[inter_slice], global_daily[inter_slice]],
            axis=1,
        )
        intra_curve_list.append(intra_curve)
        intra_feat_list.append(intra_feat)
        industry_list.append(industry_hist)
        global_list.append(global_hist)
        flat_list.append(daily_flat)
        target_calendar.append(df.loc[horizon_idx, TARGET_CALENDAR_FEATURES].to_numpy(dtype=np.float32))
        target_list.append(targets[horizon_idx])
        industry_target_list.append(industry_daily[horizon_idx])
        base_load_list.append([targets[horizon_idx - 1, 0]])
        sample_dates.append(dates[horizon_idx])

    arrays = {
        "intraday_curve_hist": np.stack(intra_curve_list),
        "intraday_daily_hist": np.stack(intra_feat_list),
        "industry_hist": np.stack(industry_list),
        "global_hist": np.stack(global_list),
        "flat_hist": np.stack(flat_list),
        "target_calendar": np.stack(target_calendar),
        "target": np.stack(target_list),
        "industry_target": np.stack(industry_target_list),
        "base_load": np.stack(base_load_list),
        "dates": np.array(sample_dates),
    }
    return arrays


def split_indices(num_samples: int, train_ratio: float, val_ratio: float) -> Dict[str, np.ndarray]:
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = num_samples - n_train - n_val
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_standardizers(arrays: Dict[str, np.ndarray], train_idx: np.ndarray) -> Dict[str, Standardizer]:
    stats: Dict[str, Standardizer] = {}
    for key, value in arrays.items():
        if key in {"dates", "target", "base_load"}:
            continue
        train_value = value[train_idx]
        flat = train_value.reshape(-1, train_value.shape[-1])
        stats[key] = Standardizer(flat.mean(axis=0), flat.std(axis=0))
    return stats


def build_target_transform(arrays: Dict[str, np.ndarray], train_idx: np.ndarray) -> TargetTransform:
    train_target = arrays["target"][train_idx]
    train_base = arrays["base_load"][train_idx]
    delta = train_target[:, 0] - train_base[:, 0]
    aux = train_target[:, 1:]
    return TargetTransform(delta.mean(), delta.std(), aux.mean(axis=0), aux.std(axis=0))


def apply_standardizers(arrays: Dict[str, np.ndarray], stats: Dict[str, Standardizer]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {"dates": arrays["dates"]}
    for key, value in arrays.items():
        if key == "dates":
            continue
        if key in {"target", "base_load"}:
            out[key] = value.astype(np.float32)
        else:
            out[key] = stats[key].transform(value)
    return out


def build_datasets(
    merged_csv: Path,
    industry_features: List[str],
    output_dir: Path,
    config: DailyForecastConfig | None = None,
) -> Tuple[Dict[str, DailyForecastDataset], Dict[str, object], Dict[str, object]]:
    config = config or DailyForecastConfig()
    df = pd.read_csv(merged_csv, parse_dates=["date"])
    arrays = build_sample_arrays(df, config, industry_features)
    indices = split_indices(len(arrays["dates"]), config.train_ratio, config.val_ratio)
    stats = build_standardizers(arrays, indices["train"])
    target_transform = build_target_transform(arrays, indices["train"])
    scaled = apply_standardizers(arrays, stats)
    scaled["target"] = target_transform.transform(arrays["target"], arrays["base_load"])
    datasets = {split: DailyForecastDataset(scaled, idx) for split, idx in indices.items()}
    split_manifest = {
        "config": asdict(config),
        "num_samples": int(len(arrays["dates"])),
        "train_dates": [str(pd.Timestamp(arrays["dates"][indices["train"][0]]).date()), str(pd.Timestamp(arrays["dates"][indices["train"][-1]]).date())],
        "val_dates": [str(pd.Timestamp(arrays["dates"][indices["val"][0]]).date()), str(pd.Timestamp(arrays["dates"][indices["val"][-1]]).date())] if len(indices["val"]) else [],
        "test_dates": [str(pd.Timestamp(arrays["dates"][indices["test"][0]]).date()), str(pd.Timestamp(arrays["dates"][indices["test"][-1]]).date())] if len(indices["test"]) else [],
        "industry_features": industry_features,
        "target_mode": "delta_from_last_day_for_daily_load",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)
    bundle = {"feature_stats": stats, "target_transform": target_transform}
    return datasets, bundle, split_manifest
