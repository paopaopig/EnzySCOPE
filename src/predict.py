# -*- coding: utf-8 -*-
"""Run test-case prediction for one or more random seeds."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from tqdm.auto import tqdm



class HyperData(Data):
    """Compatibility class for graph_pt files saved from the original scripts."""

    pass


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "test_case.json"


def load_config(path: str | os.PathLike) -> dict:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    root = config_path.parents[1]
    config["project_root"] = str(root)
    return config


def project_path(config: dict, key: str) -> str:
    path = Path(config[key])
    if path.is_absolute():
        return str(path)
    return str(Path(config["project_root"]) / path)


def parse_seed_values(seed_args: list[str]) -> list[int]:
    seeds: list[int] = []
    for item in seed_args:
        item = str(item).strip()
        if "-" in item:
            start, end = item.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(item))
    return sorted(dict.fromkeys(seeds))


def make_index2enzyme_id_from_csv(enzyme_csv: str, id_col: str) -> list[str]:
    df = pd.read_csv(enzyme_csv)
    if id_col not in df.columns:
        raise KeyError(f"{id_col!r} column not found in {enzyme_csv}; got {list(df.columns)}")
    return df[id_col].astype(str).tolist()


def make_molecule_id_to_index_from_csv(molecule_csv: str, id_col: str) -> dict[str, int]:
    df = pd.read_csv(molecule_csv)
    if id_col not in df.columns:
        raise KeyError(f"{id_col!r} column not found in {molecule_csv}; got {list(df.columns)}")
    return {mol_id: idx for idx, mol_id in enumerate(df[id_col].astype(str).tolist())}


def load_unique_mol_smiles_from_csv(molecule_csv: str, smiles_col: str) -> list[str]:
    df = pd.read_csv(molecule_csv)
    if smiles_col not in df.columns:
        raise KeyError(f"{smiles_col!r} column not found in {molecule_csv}; got {list(df.columns)}")
    return df[smiles_col].astype(str).tolist()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = spearmanr(y_true.reshape(-1), y_pred.reshape(-1)).statistic
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def compute_metrics_from_df(df: pd.DataFrame, label_col: str, cls_threshold: float):
    if label_col not in df.columns:
        return None, None
    eval_df = df[df[label_col].notna()].copy()
    if len(eval_df) == 0:
        return None, None

    y_true_raw = eval_df[label_col].astype(float).to_numpy()
    y_true = np.log1p(np.clip(y_true_raw, 0.0, None))
    y_pred = eval_df["pred_z"].astype(float).to_numpy()

    mse = float(mean_squared_error(y_true, y_pred))
    p_nonzero = eval_df["p_nonzero"].astype(float).to_numpy()
    y_true_bin = (y_true_raw > 0).astype(int)
    y_pred_bin = (p_nonzero >= cls_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    metrics = {
        "n_samples": int(len(eval_df)),
        "label_col": label_col,
        "cls_threshold": float(cls_threshold),
        "metric_space": "z=log1p(kcat)",
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": safe_spearman(y_true, y_pred),
        "acc": float(accuracy_score(y_true_bin, y_pred_bin)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    cm_df = pd.DataFrame([[int(tn), int(fp)], [int(fn), int(tp)]], index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    return metrics, cm_df


def build_model(config: dict, checkpoint_path: str, unique_mol_smiles: list[str], device: torch.device):
    from model import CombinationModel

    model = CombinationModel(
        unique_mol_smiles=unique_mol_smiles,
        pkl_by_smiles=project_path(config, "unimol_pkl"),
        enzyme_pt_path=project_path(config, "enzyme_pt"),
        unimol_mode=config.get("unimol_mode", "concat"),
        hidden_dim=int(config.get("hidden_dim", 1024)),
        dropout_rate=float(config.get("dropout", 0.2)),
        use_hypergraph=True,
        hypergraph_dir=project_path(config, "hypergraph_dir"),
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_full_table(model, reaction_df, enzyme_id_to_idx, mol_id_to_idx, config: dict, device: torch.device):
    reaction_df = reaction_df.copy()
    for col in ["Enzyme_ID", "Reactant_ID", "Product_ID"]:
        reaction_df[col] = reaction_df[col].astype(str)

    missing_enzymes = sorted(set(reaction_df["Enzyme_ID"]) - set(enzyme_id_to_idx))
    missing_reactants = sorted(set(reaction_df["Reactant_ID"]) - set(mol_id_to_idx))
    missing_products = sorted(set(reaction_df["Product_ID"]) - set(mol_id_to_idx))
    if missing_enzymes or missing_reactants or missing_products:
        raise KeyError({"missing_enzymes": missing_enzymes[:20], "missing_reactants": missing_reactants[:20], "missing_products": missing_products[:20]})

    pred_scores, pred_probs, pred_regpos = [], [], []
    enzyme_indices, react_indices, prod_indices = [], [], []
    batch_size = int(config.get("batch_size", 32))

    for start in tqdm(range(0, len(reaction_df), batch_size), desc="predict", leave=False):
        sub = reaction_df.iloc[start:start + batch_size]
        enzyme_ids = sub["Enzyme_ID"].astype(str).tolist()
        react_ids = sub["Reactant_ID"].astype(str).tolist()
        prod_ids = sub["Product_ID"].astype(str).tolist()
        enz_idx = [enzyme_id_to_idx[eid] for eid in enzyme_ids]
        react_idx = [mol_id_to_idx[mid] for mid in react_ids]
        prod_idx = [mol_id_to_idx[mid] for mid in prod_ids]

        batch = {
            "enzyme_idx": torch.tensor(enz_idx, dtype=torch.long, device=device),
            "react_idx": torch.tensor(react_idx, dtype=torch.long, device=device),
            "prod_idx": torch.tensor(prod_idx, dtype=torch.long, device=device),
            "labels": torch.zeros((len(sub), 1), dtype=torch.float32, device=device),
            "enzyme_ids": enzyme_ids,
        }
        logits_zero, reg_pos, _ = model(batch, return_emb=False)
        p_nonzero = torch.sigmoid(logits_zero).squeeze(1)
        reg_pos = reg_pos.squeeze(1)
        pred_z = p_nonzero * reg_pos

        pred_scores.extend(pred_z.detach().cpu().tolist())
        pred_probs.extend(p_nonzero.detach().cpu().tolist())
        pred_regpos.extend(reg_pos.detach().cpu().tolist())
        enzyme_indices.extend(enz_idx)
        react_indices.extend(react_idx)
        prod_indices.extend(prod_idx)

    out_df = reaction_df.copy()
    out_df["enzyme_idx"] = enzyme_indices
    out_df["react_idx"] = react_indices
    out_df["prod_idx"] = prod_indices
    out_df["p_nonzero"] = pred_probs
    out_df["reg_pos"] = pred_regpos
    out_df["pred_z"] = pred_scores
    out_df["pred_score"] = pred_scores

    label_col = config.get("label_col", "kcat")
    if label_col in out_df.columns:
        y_true_raw = pd.to_numeric(out_df[label_col], errors="coerce").to_numpy(dtype=np.float64)
        out_df["true_z"] = np.log1p(np.clip(y_true_raw, 0.0, None))
    return out_df


def run_seed(seed: int, config: dict, device: torch.device):
    enzyme_csv = project_path(config, "enzyme_csv")
    molecule_csv = project_path(config, "molecule_csv")
    reaction_csv = project_path(config, "reaction_csv")
    checkpoint = Path(config["checkpoint_pattern"].format(seed=seed))
    if not checkpoint.is_absolute():
        checkpoint = Path(config["project_root"]) / checkpoint

    index2enzyme_id = make_index2enzyme_id_from_csv(enzyme_csv, config.get("enzyme_id_col", "Enzyme_ID"))
    enzyme_id_to_idx = {eid: idx for idx, eid in enumerate(index2enzyme_id)}
    mol_id_to_idx = make_molecule_id_to_index_from_csv(molecule_csv, config.get("molecule_id_col", "Molecule_ID"))
    unique_mol_smiles = load_unique_mol_smiles_from_csv(molecule_csv, config.get("molecule_smiles_col", "Molecule"))
    reaction_df = pd.read_csv(reaction_csv)

    model = build_model(config, str(checkpoint), unique_mol_smiles, device)
    out_df = predict_full_table(model, reaction_df, enzyme_id_to_idx, mol_id_to_idx, config, device)

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(config["project_root"]) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"prediction_results_full_table_seed_{seed}.csv"
    metric_json = output_dir / f"prediction_metrics_full_table_seed_{seed}.json"
    confusion_csv = output_dir / f"prediction_confusion_matrix_seed_{seed}.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metrics, cm_df = compute_metrics_from_df(out_df, config.get("label_col", "kcat"), float(config.get("cls_threshold", 0.5)))
    if metrics is not None:
        with open(metric_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        cm_df.to_csv(confusion_csv, encoding="utf-8-sig")
    print(f"seed={seed} saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run the provided test case. Defaults: configs/test_case.json and seeds 1231-1240.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config file. Default: configs/test_case.json.")
    parser.add_argument("--seeds", nargs="+", default=["1231-1240"], help="Seeds to run. Default: 1231-1240.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    for seed in parse_seed_values(args.seeds):
        run_seed(seed, config, device)


if __name__ == "__main__":
    main()


