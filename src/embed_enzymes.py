# -*- coding: utf-8 -*-
"""Example ESM3 enzyme embedding script."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate residue-level ESM3 embeddings from an enzyme table.")
    parser.add_argument("--enzyme-csv", required=True, help="CSV with Enzyme_ID and Enzyme_Sequence columns.")
    parser.add_argument("--output-pt", required=True, help="Output torch file keyed by Enzyme_ID.")
    parser.add_argument("--id-col", default="Enzyme_ID")
    parser.add_argument("--sequence-col", default="Enzyme_Sequence")
    parser.add_argument("--max-len", type=int, default=1700)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading ESM3.")
    args = parser.parse_args()

    table = pd.read_csv(args.enzyme_csv)
    for column in [args.id_col, args.sequence_col]:
        if column not in table.columns:
            raise ValueError(f"Missing column in enzyme table: {column}")
    print(f"Enzymes to embed: {len(table)}")
    print(f"Output: {args.output_pt}")
    if args.dry_run:
        return

    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    from torch import amp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM3.from_pretrained("esm3_sm_open_v1").to(device).eval()
    embeddings = {}
    with torch.no_grad():
        for _, row in tqdm(table.iterrows(), total=len(table), desc="embedding enzymes"):
            enzyme_id = str(row[args.id_col])
            sequence = str(row[args.sequence_col]).strip()[:args.max_len]
            protein = ESMProtein(sequence=sequence)
            tokens = model.encode(protein).sequence.to(device).unsqueeze(0)
            with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(sequence_tokens=tokens).embeddings
            embeddings[enzyme_id] = output.float().squeeze(0)[1:-1, :].cpu()

    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved {len(embeddings)} enzyme embeddings.")


if __name__ == "__main__":
    main()
