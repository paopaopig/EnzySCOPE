# -*- coding: utf-8 -*-
"""Example protein hypergraph construction script.

This entry point documents the expected interface for rebuilding protein graphs
from structure files. The cleaned repository already includes the graph files
needed for prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build protein hypergraph files from PDB or CIF structures.")
    parser.add_argument("--enzyme-csv", required=True, help="CSV with Enzyme_ID values.")
    parser.add_argument("--structure-dir", required=True, help="Directory containing <Enzyme_ID>.pdb or <Enzyme_ID>.cif files.")
    parser.add_argument("--output-dir", required=True, help="Directory for graph .pt files.")
    parser.add_argument("--radius", type=float, default=15.0, help="Radius used for residue neighborhood hyperedges.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned files only.")
    args = parser.parse_args()

    table = pd.read_csv(args.enzyme_csv)
    if "Enzyme_ID" not in table.columns:
        raise ValueError("enzyme table must contain Enzyme_ID")
    structure_dir = Path(args.structure_dir)
    missing = []
    planned = []
    for enzyme_id in table["Enzyme_ID"].astype(str):
        pdb = structure_dir / f"{enzyme_id}.pdb"
        cif = structure_dir / f"{enzyme_id}.cif"
        if pdb.exists():
            planned.append((enzyme_id, pdb))
        elif cif.exists():
            planned.append((enzyme_id, cif))
        else:
            missing.append(enzyme_id)

    print(f"Structures found: {len(planned)}")
    print(f"Structures missing: {len(missing)}")
    print(f"Output directory: {args.output_dir}")
    if missing:
        print("Missing examples: " + ", ".join(missing[:10]))
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError("Cannot build all graphs because some structure files are missing.")

    raise NotImplementedError(
        "This repository ships the cleaned graph files used for prediction. "
        "Use the original structure-processing environment to rebuild graphs from raw PDB/CIF files."
    )


if __name__ == "__main__":
    main()
