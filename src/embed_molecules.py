# -*- coding: utf-8 -*-
"""Example UniMol2 molecule embedding script."""

from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate per-atom UniMol2 embeddings from a molecule table.")
    parser.add_argument("--molecule-csv", required=True, help="CSV with Molecule_ID and Molecule columns.")
    parser.add_argument("--output-pkl", required=True, help="Output pickle keyed by SMILES.")
    parser.add_argument("--id-col", default="Molecule_ID")
    parser.add_argument("--smiles-col", default="Molecule")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading UniMol.")
    args = parser.parse_args()

    table = pd.read_csv(args.molecule_csv)
    for column in [args.id_col, args.smiles_col]:
        if column not in table.columns:
            raise ValueError(f"Missing column in molecule table: {column}")
    table = table.iloc[args.start_index:].copy()
    print(f"Molecules to embed: {len(table)}")
    print(f"Output: {args.output_pkl}")
    if args.dry_run:
        return

    from unimol_tools import UniMolRepr

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch = None
        device = "cpu"

    model = UniMolRepr(data_type="molecule", model_name="unimolv2", model_size="570m", remove_hs=False, device=device)
    embeddings = {}
    failures = []
    for _, row in tqdm(table.iterrows(), total=len(table), desc="embedding molecules"):
        smiles = str(row[args.smiles_col]).strip()
        try:
            output = model.get_repr([smiles], return_atomic_reprs=True)
            atoms = np.asarray(output["atomic_reprs"][0], dtype=np.float32)
            embeddings[smiles] = atoms
        except Exception as exc:
            failures.append({args.id_col: row[args.id_col], args.smiles_col: smiles, "error": repr(exc)})
        finally:
            gc.collect()
            if torch is not None and device == "cuda":
                torch.cuda.empty_cache()

    output_path = Path(args.output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if failures:
        pd.DataFrame(failures).to_csv(output_path.with_suffix(".failures.csv"), index=False)
    print(f"Saved {len(embeddings)} molecule embeddings.")


if __name__ == "__main__":
    main()
