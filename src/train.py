# -*- coding: utf-8 -*-
"""Train the teacher model with relative project paths.

The complete training data is not included in the test-case release. Use
``--dry-run`` to print the expected training files and workflow without loading
missing data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from tqdm.auto import tqdm

from data import load_indexed_dataloaders_from_tables

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_config.json"
REQUIRED_PATH_KEYS = ["enzyme_csv", "molecule_csv", "reaction_csv", "unimol_pkl", "enzyme_pt", "hypergraph_dir"]


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    config["project_root"] = str(config_path.parents[1])
    return config


def project_path(config: dict, key: str) -> Path:
    path = Path(config[key])
    return path if path.is_absolute() else Path(config["project_root"]) / path


def checkpoint_path(config: dict, seed: int, output_dir: Path) -> Path:
    pattern = config.get("checkpoint_pattern")
    if pattern:
        path = Path(pattern.format(seed=seed))
        return path if path.is_absolute() else Path(config["project_root"]) / path
    return output_dir / f"train_seed_{seed}.pt"


def required_input_paths(config: dict) -> dict[str, Path]:
    return {key: project_path(config, key) for key in REQUIRED_PATH_KEYS}


def print_training_plan(config: dict, seed: int, output_dir: Path) -> None:
    print("expected training inputs:")
    for key, path in required_input_paths(config).items():
        status = "present" if path.exists() else "missing"
        print(f"  {key}: {path} [{status}]")
    print("expected outputs:")
    print(f"  checkpoint: {checkpoint_path(config, seed, output_dir)}")
    print(f"  metrics: {output_dir / f'train_metrics_seed_{seed}.csv'}")
    print("workflow:")
    print("  1. Put train tables and embeddings under data/train/.")
    print("  2. Build train protein graphs under data/train/train_protein_graphs/.")
    print("  3. Run python src/train.py --config configs/train_config.json --epochs <N>.")
    print("  4. Run python src/predict.py for the provided reproducible test case.")


def assert_required_paths(config: dict) -> None:
    missing = [f"{key}: {path}" for key, path in required_input_paths(config).items() if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError("Missing training inputs. Run with --dry-run to view the workflow.\n" + joined)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = spearmanr(y_true.reshape(-1), y_pred.reshape(-1)).statistic
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_nonzero: np.ndarray, threshold: float = 0.5) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    y_true_bin = (y_true.reshape(-1) > 0).astype(int)
    y_pred_bin = (p_nonzero.reshape(-1) >= threshold).astype(int)

    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": safe_spearman(y_true, y_pred),
        "acc": float((y_true_bin == y_pred_bin).mean()),
        "roc_auc": None,
        "pr_auc": None,
    }
    if len(np.unique(y_true_bin)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_bin, p_nonzero.reshape(-1)))
        metrics["pr_auc"] = float(average_precision_score(y_true_bin, p_nonzero.reshape(-1)))
    return metrics


def make_model(config: dict, molecule_smiles: list[str], device: torch.device):
    from model import CombinationModel

    return CombinationModel(
        unique_mol_smiles=molecule_smiles,
        pkl_by_smiles=str(project_path(config, "unimol_pkl")),
        enzyme_pt_path=str(project_path(config, "enzyme_pt")),
        unimol_mode=config.get("unimol_mode", "concat"),
        hidden_dim=int(config.get("hidden_dim", 1024)),
        dropout_rate=float(config.get("dropout", 0.2)),
        use_hypergraph=True,
        hypergraph_dir=str(project_path(config, "hypergraph_dir")),
        hg_hidden_dim=1024,
        hg_out_dim=1536,
        hg_dropout=0.2,
        hg_use_attention=False,
        hg_heads=4,
        cross_attn_heads=4,
        cross_attn_dropout=0.2,
        reaction_hidden_dim=512,
        reaction_dropout=0.2,
    ).to(device)


def move_batch_to_device(batch: dict, device: torch.device, index_to_enzyme_id: list[str]) -> dict:
    enzyme_idx = batch["enzyme_idx"].long().to(device)
    return {
        "enzyme_idx": enzyme_idx,
        "react_idx": batch["react_idx"].long().to(device),
        "prod_idx": batch["prod_idx"].long().to(device),
        "labels": batch["labels"].float().to(device),
        "enzyme_ids": [index_to_enzyme_id[int(idx)] for idx in enzyme_idx.detach().cpu().tolist()],
    }


def forward_loss(model, batch: dict, bce_loss: nn.Module, reg_loss: nn.Module, zero_pos_weight: float) -> tuple[torch.Tensor, dict]:
    logits_zero, reg_pos, labels = model(batch, return_emb=False)
    labels = labels.float()
    y_binary = (labels > 0).float()
    p_nonzero = torch.sigmoid(logits_zero)
    pred_z = p_nonzero * reg_pos

    cls_loss = bce_loss(logits_zero, y_binary)
    reg_weights = torch.where(labels > 0, torch.full_like(labels, zero_pos_weight), torch.ones_like(labels))
    reg_loss_value = (reg_loss(pred_z, labels) * reg_weights).mean()
    loss = cls_loss + reg_loss_value
    return loss, {
        "labels": labels.detach(),
        "pred_z": pred_z.detach(),
        "p_nonzero": p_nonzero.detach(),
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "reg_loss": float(reg_loss_value.detach().cpu().item()),
    }


def train_one_epoch(model, loader, optimizer, device, index_to_enzyme_id, zero_pos_weight: float):
    model.train()
    bce_loss = nn.BCEWithLogitsLoss()
    reg_loss = nn.HuberLoss(delta=1.0, reduction="none")
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device, index_to_enzyme_id)
        loss, _ = forward_loss(model, batch, bce_loss, reg_loss, zero_pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size

    return {"loss": total_loss / max(total_count, 1)}


@torch.no_grad()
def evaluate(model, loader, device, index_to_enzyme_id, zero_pos_weight: float, threshold: float):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()
    reg_loss = nn.HuberLoss(delta=1.0, reduction="none")
    labels_all, pred_all, prob_all = [], [], []
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device, index_to_enzyme_id)
        loss, out = forward_loss(model, batch, bce_loss, reg_loss, zero_pos_weight)
        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size
        labels_all.append(out["labels"].cpu().numpy())
        pred_all.append(out["pred_z"].cpu().numpy())
        prob_all.append(out["p_nonzero"].cpu().numpy())

    y_true = np.concatenate(labels_all, axis=0)
    y_pred = np.concatenate(pred_all, axis=0)
    p_nonzero = np.concatenate(prob_all, axis=0)
    metrics = compute_metrics(y_true, y_pred, p_nonzero, threshold=threshold)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def metric_is_better(current: float, best: float | None, metric_name: str) -> bool:
    if best is None:
        return True
    if metric_name in {"mse", "rmse", "mae", "loss"}:
        return current < best
    return current > best


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "split", "loss", "mse", "rmse", "mae", "r2", "spearman", "acc", "roc_auc", "pr_auc"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the teacher model with relative project paths.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config JSON. Default: configs/train_config.json.")
    parser.add_argument("--output-dir", default=None, help="Directory for training metrics. Default: config output_dir.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--zero-pos-weight", type=float, default=3.0)
    parser.add_argument("--best-metric", default="mse", choices=["loss", "mse", "rmse", "mae", "r2", "spearman"])
    parser.add_argument("--dry-run", action="store_true", help="Print the expected training workflow without loading training data.")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    output_dir = Path(args.output_dir or config.get("output_dir", "training_outputs"))
    if not output_dir.is_absolute():
        output_dir = Path(config["project_root"]) / output_dir

    if args.dry_run:
        print_training_plan(config, args.seed, output_dir)
        return

    assert_required_paths(config)
    batch_size = int(args.batch_size or config.get("batch_size", 32))
    loaded = load_indexed_dataloaders_from_tables(
        str(project_path(config, "enzyme_csv")),
        str(project_path(config, "molecule_csv")),
        str(project_path(config, "reaction_csv")),
        batch_size=batch_size,
        random_seed=args.seed,
    )
    mappings = loaded["mappings"]
    molecule_smiles = loaded["molecules"][config.get("molecule_smiles_col", "Molecule")].astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, molecule_smiles, device)

    split = loaded["split_indices"]
    print(f"device = {device}")
    print(f"rows: train={len(split['train'])}, valid={len(split['valid'])}, holdout={len(split['holdout'])}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_path(config, args.seed, output_dir)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"train_metrics_seed_{args.seed}.csv"

    best_value = None
    best_epoch = None
    rows = []
    threshold = float(config.get("cls_threshold", 0.5))

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaded["train_loader"],
            optimizer,
            device,
            mappings.index_to_enzyme_id,
            args.zero_pos_weight,
        )
        valid_metrics = evaluate(
            model,
            loaded["valid_loader"],
            device,
            mappings.index_to_enzyme_id,
            args.zero_pos_weight,
            threshold,
        )
        rows.append({"epoch": epoch, "split": "train", **train_metrics})
        rows.append({"epoch": epoch, "split": "valid", **valid_metrics})
        current = valid_metrics[args.best_metric]
        if current is not None and metric_is_better(float(current), best_value, args.best_metric):
            best_value = float(current)
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"valid_mse={valid_metrics['mse']:.6f} valid_rmse={valid_metrics['rmse']:.6f} "
            f"valid_mae={valid_metrics['mae']:.6f} valid_r2={valid_metrics['r2']:.6f} "
            f"valid_spearman={valid_metrics['spearman']:.6f}"
        )
        write_metrics_csv(metrics_path, rows)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    holdout_metrics = evaluate(
        model,
        loaded["holdout_loader"],
        device,
        mappings.index_to_enzyme_id,
        args.zero_pos_weight,
        threshold,
    )
    rows.append({"epoch": best_epoch, "split": "holdout", **holdout_metrics})
    write_metrics_csv(metrics_path, rows)
    print(f"best_epoch = {best_epoch}")
    print(f"best_checkpoint = {best_path}")
    print(f"metrics_csv = {metrics_path}")
    print("holdout_metrics = " + json.dumps(holdout_metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

