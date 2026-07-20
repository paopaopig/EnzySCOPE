# -*- coding: utf-8 -*-
"""Data loading utilities for enzyme-reaction tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TableMappings:
    enzyme_id_to_index: Dict[str, int]
    molecule_id_to_index: Dict[str, int]
    index_to_enzyme_id: List[str]
    index_to_molecule_id: List[str]


class ReactionIndexDataset(Dataset):
    """Dataset that stores integer enzyme, reactant, product indices and log1p(kcat)."""

    def __init__(self, enzyme_idx, react_idx, prod_idx, labels):
        self.enzyme_idx = torch.as_tensor(np.asarray(enzyme_idx, dtype=np.int64), dtype=torch.long)
        self.react_idx = torch.as_tensor(np.asarray(react_idx, dtype=np.int64), dtype=torch.long)
        self.prod_idx = torch.as_tensor(np.asarray(prod_idx, dtype=np.int64), dtype=torch.long)
        self.labels = torch.as_tensor(np.asarray(labels, dtype=np.float32), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return {
            "enzyme_idx": self.enzyme_idx[index],
            "react_idx": self.react_idx[index],
            "prod_idx": self.prod_idx[index],
            "labels": self.labels[index],
        }


def _require_columns(df: pd.DataFrame, required: set[str], table_name: str):
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _split_per_enzyme(indexed: pd.DataFrame, seed: int, valid_fraction: float, test_fraction: float):
    rng = np.random.default_rng(seed)
    train_ids, valid_ids, test_ids = [], [], []
    indexed = indexed.copy()
    indexed["row_id"] = np.arange(len(indexed))

    for _, group in indexed.groupby("enzyme_idx", sort=False):
        rows = group["row_id"].to_numpy()
        rng.shuffle(rows)
        n = len(rows)
        if n == 1:
            train_ids.extend(rows.tolist())
            continue
        test_n = max(1, int(round(n * test_fraction))) if test_fraction > 0 else 0
        valid_n = max(1, int(round(n * valid_fraction))) if valid_fraction > 0 and n - test_n > 1 else 0
        test_ids.extend(rows[:test_n].tolist())
        valid_ids.extend(rows[test_n:test_n + valid_n].tolist())
        train_ids.extend(rows[test_n + valid_n:].tolist())

    return sorted(train_ids), sorted(valid_ids), sorted(test_ids)


def load_tables(enzyme_csv: str, molecule_csv: str, reaction_csv: str):
    enzymes = pd.read_csv(enzyme_csv)
    molecules = pd.read_csv(molecule_csv)
    reactions = pd.read_csv(reaction_csv)

    _require_columns(enzymes, {"Enzyme_ID", "Enzyme_Sequence"}, "enzyme table")
    _require_columns(molecules, {"Molecule_ID", "Molecule"}, "molecule table")
    _require_columns(reactions, {"Enzyme_ID", "Reactant_ID", "Product_ID", "kcat"}, "reaction table")

    enzymes["Enzyme_ID"] = enzymes["Enzyme_ID"].astype(str)
    molecules["Molecule_ID"] = molecules["Molecule_ID"].astype(str)
    reactions["Enzyme_ID"] = reactions["Enzyme_ID"].astype(str)
    reactions["Reactant_ID"] = reactions["Reactant_ID"].astype(str)
    reactions["Product_ID"] = reactions["Product_ID"].astype(str)

    index_to_enzyme_id = enzymes["Enzyme_ID"].tolist()
    index_to_molecule_id = molecules["Molecule_ID"].tolist()
    enzyme_id_to_index = {value: index for index, value in enumerate(index_to_enzyme_id)}
    molecule_id_to_index = {value: index for index, value in enumerate(index_to_molecule_id)}

    indexed = pd.DataFrame({
        "enzyme_idx": reactions["Enzyme_ID"].map(enzyme_id_to_index),
        "react_idx": reactions["Reactant_ID"].map(molecule_id_to_index),
        "prod_idx": reactions["Product_ID"].map(molecule_id_to_index),
        "label": pd.to_numeric(reactions["kcat"], errors="coerce"),
    })

    if indexed.isna().any().any():
        bad = indexed.columns[indexed.isna().any()].tolist()
        raise ValueError(f"reaction table contains IDs or labels that could not be mapped: {bad}")

    indexed = indexed.astype({"enzyme_idx": int, "react_idx": int, "prod_idx": int})
    indexed["label"] = np.log1p(np.clip(indexed["label"].astype(float).to_numpy(), 0.0, None))

    mappings = TableMappings(
        enzyme_id_to_index=enzyme_id_to_index,
        molecule_id_to_index=molecule_id_to_index,
        index_to_enzyme_id=index_to_enzyme_id,
        index_to_molecule_id=index_to_molecule_id,
    )
    return enzymes, molecules, reactions, indexed, mappings


def load_indexed_dataloaders_from_tables(
    enzyme_csv: str,
    molecule_csv: str,
    reaction_csv: str,
    *,
    batch_size: int = 32,
    random_seed: int = 1234,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.2,
    shuffle_train: bool = True,
):
    enzymes, molecules, reactions, indexed, mappings = load_tables(enzyme_csv, molecule_csv, reaction_csv)
    train_ids, valid_ids, test_ids = _split_per_enzyme(indexed, random_seed, valid_fraction, test_fraction)

    def make_loader(row_ids, shuffle=False):
        part = indexed.iloc[row_ids]
        dataset = ReactionIndexDataset(part["enzyme_idx"], part["react_idx"], part["prod_idx"], part["label"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return {
        "train_loader": make_loader(train_ids, shuffle=shuffle_train),
        "valid_loader": make_loader(valid_ids),
        "test_loader": make_loader(test_ids),
        "holdout_loader": make_loader(test_ids),
        "mappings": mappings,
        "enzymes": enzymes,
        "molecules": molecules,
        "reactions": reactions,
        "indexed_reactions": indexed,
        "split_indices": {"train": train_ids, "valid": valid_ids, "holdout": test_ids},
    }




