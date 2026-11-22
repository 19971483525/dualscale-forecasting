from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    min_epochs: int = 10
    aux_weight: float = 0.2
    industry_aux_weight: float = 0.15
    industry_focus_weight: float = 0.10
    underpredict_penalty: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def asymmetric_huber(pred: torch.Tensor, target: torch.Tensor, penalty: float = 1.0) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.tensor(1.0, device=pred.device))
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 + linear
    under_mask = (pred < target).float()
    loss = loss * (1.0 + under_mask * max(0.0, penalty - 1.0))
    return loss.mean()


def build_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    target_inverse_fn,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    bases = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model_out = model({k: batch[k] for k in ["intraday_curve_hist", "intraday_daily_hist", "industry_hist", "global_hist", "flat_hist", "target_calendar"]})
            output = model_out["prediction"]
            loss = nn.functional.smooth_l1_loss(output, batch["target"])
            if "industry_prediction" in model_out and "industry_target" in batch:
                industry_loss = nn.functional.smooth_l1_loss(model_out["industry_prediction"], batch["industry_target"])
                primary_industry_loss = nn.functional.smooth_l1_loss(model_out["industry_prediction"][:, :5], batch["industry_target"][:, :5])
                loss = loss + 0.1 * industry_loss
            losses.append(float(loss.item()))
            preds.append(output.detach().cpu().numpy())
            trues.append(batch["target"].detach().cpu().numpy())
            bases.append(batch["base_load"].detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    base = np.concatenate(bases, axis=0)
    pred_real = target_inverse_fn(pred, base)
    true_real = target_inverse_fn(true, base)
    score = mae(true_real[:, 0], pred_real[:, 0])
    dates = getattr(loader.dataset, "dates", np.arange(len(true_real)))
    return float(np.mean(losses)), score, true_real, pred_real, dates


def metric_payload(true_real: np.ndarray, pred_real: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mae(true_real[:, 0], pred_real[:, 0]),
        "rmse": rmse(true_real[:, 0], pred_real[:, 0]),
        "mape": mape(true_real[:, 0], pred_real[:, 0]),
    }


def save_split_predictions(
    split_name: str,
    model: nn.Module,
    loader: DataLoader,
    target_inverse_fn,
    output_dir: Path,
    device: str,
) -> Dict[str, float]:
    loss, score, true_real, pred_real, dates = evaluate_model(model, loader, target_inverse_fn, device)
    import pandas as pd

    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "true_daily_load": true_real[:, 0],
            "pred_daily_load": pred_real[:, 0],
            "true_peak": true_real[:, 1],
            "pred_peak": pred_real[:, 1],
            "true_valley": true_real[:, 2],
            "pred_valley": pred_real[:, 2],
            "true_mean_96": true_real[:, 3],
            "pred_mean_96": pred_real[:, 3],
        }
    )
    pred_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
    return {
        "loss": loss,
        "mae": score,
        "rmse": rmse(true_real[:, 0], pred_real[:, 0]),
        "mape": mape(true_real[:, 0], pred_real[:, 0]),
    }


def train_model(
    model: nn.Module,
    datasets: Dict[str, object],
    target_inverse_fn,
    output_dir: Path,
    config: TrainConfig | None = None,
) -> Dict[str, object]:
    config = config or TrainConfig()
    set_seed(config.seed)
    device = config.device
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    loaders = {
        "train": build_loader(datasets["train"], config.batch_size, True),
        "val": build_loader(datasets["val"], config.batch_size, False),
        "test": build_loader(datasets["test"], config.batch_size, False),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in loaders["train"]:
            batch = {k: v.to(device) for k, v in batch.items()}
            model_out = model({k: batch[k] for k in ["intraday_curve_hist", "intraday_daily_hist", "industry_hist", "global_hist", "flat_hist", "target_calendar"]})
            output = model_out["prediction"]
            target = batch["target"]
            loss_main = asymmetric_huber(output[:, :1], target[:, :1], config.underpredict_penalty)
            loss_aux = nn.functional.smooth_l1_loss(output[:, 1:], target[:, 1:])
            loss = loss_main + config.aux_weight * loss_aux
            if "industry_prediction" in model_out and "industry_target" in batch:
                loss_industry = nn.functional.smooth_l1_loss(model_out["industry_prediction"], batch["industry_target"])
                focus_cols = min(5, model_out["industry_prediction"].shape[1])
                loss_industry_focus = nn.functional.smooth_l1_loss(
                    model_out["industry_prediction"][:, :focus_cols],
                    batch["industry_target"][:, :focus_cols],
                )
                loss = (
                    loss
                    + config.industry_aux_weight * loss_industry
                    + config.industry_focus_weight * loss_industry_focus
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()
        train_eval_loss, train_mae, train_true, train_pred, _ = evaluate_model(model, loaders["train"], target_inverse_fn, device)
        val_loss, val_mae, val_true, val_pred, _ = evaluate_model(model, loaders["val"], target_inverse_fn, device)
        train_metrics_epoch = metric_payload(train_true, train_pred)
        val_metrics_epoch = metric_payload(val_true, val_pred)
        train_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
                "train_mae": train_mae,
                "train_rmse": train_metrics_epoch["rmse"],
                "train_mape": train_metrics_epoch["mape"],
                "val_mae": val_mae,
                "val_rmse": val_metrics_epoch["rmse"],
                "val_mape": val_metrics_epoch["mape"],
            }
        )
        import pandas as pd
        pd.DataFrame(history).to_csv(output_dir / "training_history_live_model_space.csv", index=False, encoding="utf-8-sig")
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_model_live.pt")
        else:
            bad_epochs += 1
        if epoch >= config.min_epochs and bad_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / "best_model.pt")

    train_metrics = save_split_predictions("train", model, loaders["train"], target_inverse_fn, output_dir, device)
    val_metrics = save_split_predictions("val", model, loaders["val"], target_inverse_fn, output_dir, device)
    test_loss, test_mae, test_true, test_pred, test_dates = evaluate_model(model, loaders["test"], target_inverse_fn, device)
    metrics = {
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "val_best_mae": best_val,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": rmse(test_true[:, 0], test_pred[:, 0]),
        "test_mape": mape(test_true[:, 0], test_pred[:, 0]),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_config": asdict(config),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    import pandas as pd

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "true_daily_load": test_true[:, 0],
            "pred_daily_load": test_pred[:, 0],
            "true_peak": test_true[:, 1],
            "pred_peak": test_pred[:, 1],
            "true_valley": test_true[:, 2],
            "pred_valley": test_pred[:, 2],
            "true_mean_96": test_true[:, 3],
            "pred_mean_96": test_pred[:, 3],
        }
    )
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    return metrics
