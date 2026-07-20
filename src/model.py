# -*- coding: utf-8 -*-

import os
import pickle
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
from torch_geometric.nn import HypergraphConv


class SequenceMeanPool(nn.Module):
    """
    Protein residue sequence pooling: mean pooling without softmax attention.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if x.ndim != 2:
            raise ValueError(f"SequenceMeanPool expects [N, D], got {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("SequenceMeanPool got empty input")

        pooled = x.mean(dim=0)

        if return_attn:
            # Return ones as a compatibility placeholder instead of learned attention.
            token_strength = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
            return pooled, token_strength
        return pooled


class EntmaxPool(nn.Module):
    """
    Molecular atom pooling with entmax15.
    """
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if x.ndim != 2:
            raise ValueError(f"EntmaxPool expects [N, D], got {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("EntmaxPool got empty input")

        h = torch.tanh(self.fc1(x))
        scores = self.fc2(h).squeeze(-1)
        attn = entmax15(scores.float(), dim=0).to(x.dtype)
        pooled = (attn.unsqueeze(-1) * x).sum(dim=0)

        if return_attn:
            return pooled, attn
        return pooled


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if x.ndim != 2:
            raise ValueError(f"MeanPool expects [N, D], got {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("MeanPool got empty input")

        pooled = x.mean(dim=0)
        if return_attn:
            n = x.shape[0]
            attn = torch.full((n,), 1.0 / n, dtype=x.dtype, device=x.device)
            return pooled, attn
        return pooled


class HyperGraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 1024,
        out_dim: int = 1536,
        dropout: float = 0.2,
        use_attention: bool = False,
        heads: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = HypergraphConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            use_attention=use_attention,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.conv2 = HypergraphConv(
            in_channels=hidden_dim,
            out_channels=out_dim,
            use_attention=use_attention,
            heads=heads,
            concat=False,
            dropout=dropout,
        )

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        if not hasattr(data, "x"):
            raise ValueError("Hypergraph data missing x")
        if not hasattr(data, "hyperedge_index"):
            raise ValueError("Hypergraph data missing hyperedge_index")

        x = data.x
        hyperedge_index = data.hyperedge_index

        if x.ndim != 2:
            raise ValueError(f"data.x must be [num_nodes, feat_dim], got {tuple(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"node feature dim mismatch: got {x.shape[1]}, expected {self.in_dim}"
            )
        if hyperedge_index.ndim != 2 or hyperedge_index.shape[0] != 2:
            raise ValueError(
                f"hyperedge_index must be [2, E], got {tuple(hyperedge_index.shape)}"
            )
        if hyperedge_index.numel() == 0:
            raise ValueError("hyperedge_index is empty")

        x = self.conv1(x, hyperedge_index)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x, hyperedge_index)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.use_residual = use_residual

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query_nodes: torch.Tensor,
        react_atoms: torch.Tensor,
        return_attn: bool = False,
    ):
        if query_nodes.ndim != 2:
            raise ValueError(f"query_nodes must be [N, D], got {tuple(query_nodes.shape)}")
        if react_atoms.ndim != 2:
            raise ValueError(f"react_atoms must be [N, D], got {tuple(react_atoms.shape)}")
        if query_nodes.shape[0] == 0:
            raise ValueError("query_nodes is empty")
        if react_atoms.shape[0] == 0:
            raise ValueError("react_atoms is empty")
        if query_nodes.shape[1] != react_atoms.shape[1]:
            raise ValueError(
                f"feature dim mismatch: query={query_nodes.shape[1]}, react={react_atoms.shape[1]}"
            )

        context, attn = self.mha(
            query=query_nodes.unsqueeze(0),
            key=react_atoms.unsqueeze(0),
            value=react_atoms.unsqueeze(0),
            need_weights=return_attn,
            average_attn_weights=False,
        )
        context = context.squeeze(0)
        context = self.out_proj(context)
        context = self.drop(context)

        if self.use_residual:
            out = self.norm(query_nodes + context)
        else:
            out = self.norm(context)

        if return_attn:
            attn = attn.mean(dim=1).squeeze(0)
            return out, attn
        return out


class ProductModulatedReaction(nn.Module):
    def __init__(self, dim: int = 1536, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, R_global: torch.Tensor, P_global: torch.Tensor, return_gate: bool = False):
        if R_global.ndim != 2 or P_global.ndim != 2:
            raise ValueError("R_global and P_global must both be [B, D]")
        if R_global.shape != P_global.shape:
            raise ValueError(
                f"shape mismatch: R_global={tuple(R_global.shape)}, P_global={tuple(P_global.shape)}"
            )

        gate = self.gate_net(P_global)
        reaction_emb = self.norm(R_global + gate * R_global)

        if return_gate:
            return reaction_emb, gate
        return reaction_emb


class CombinationModel(nn.Module):
    def __init__(
        self,
        unique_mol_smiles,
        pkl_by_smiles,
        enzyme_pt_path,
        unimol_mode: str = "concat",
        hidden_dim: int = 1024,
        dropout_rate: float = 0.2,
        use_hypergraph: bool = True,
        hypergraph_dir: Optional[str] = None,
        hg_hidden_dim: int = 1024,
        hg_out_dim: int = 1536,
        hg_dropout: float = 0.2,
        hg_use_attention: bool = False,
        hg_heads: int = 4,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
        reaction_hidden_dim: int = 512,
        reaction_dropout: float = 0.1,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm3_emb_dim = 1536
        self.unimol_emb_dim = 1536
        self.unimol_mode = unimol_mode
        self.use_hypergraph = bool(use_hypergraph)
        self.hypergraph_dir = hypergraph_dir
        self._hypergraph_cache: Dict[str, Any] = {}

        self._enzyme_table = torch.load(enzyme_pt_path, map_location="cpu")
        if not isinstance(self._enzyme_table, dict):
            raise ValueError("enzyme_pt_path must load a dict: enzyme_id -> [L,1536] tensor/array")

        # Protein sequence branch: mean pooling without softmax attention.
        self.residue_pool = SequenceMeanPool().to(self.device)

        # Molecular atom branch: entmax pooling.
        self.atom_pool = EntmaxPool(dim=self.unimol_emb_dim, hidden=256).to(self.device)

        self._unique_mol_smiles = list(unique_mol_smiles)
        self._atom_tables = self._load_atomic_tables(self._unique_mol_smiles, pkl_by_smiles)

        self.cross_attn = CrossAttentionBlock(
            dim=self.esm3_emb_dim,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
            use_residual=False,
        ).to(self.device)

        if not self.use_hypergraph:
            raise ValueError("This model is teacher-only; please set use_hypergraph=True.")
        if hypergraph_dir is None or (not os.path.isdir(hypergraph_dir)):
            raise ValueError("A valid hypergraph_dir is required when use_hypergraph=True.")

        self.hg_encoder = HyperGraphEncoder(
            in_dim=self.esm3_emb_dim,
            hidden_dim=hg_hidden_dim,
            out_dim=hg_out_dim,
            dropout=hg_dropout,
            use_attention=hg_use_attention,
            heads=hg_heads,
        ).to(self.device)

        self.final_mean_pool = MeanPool().to(self.device)
        self.protein_proj = nn.Linear(hg_out_dim, self.esm3_emb_dim).to(self.device)
        nn.init.xavier_uniform_(self.protein_proj.weight)
        nn.init.zeros_(self.protein_proj.bias)

        self.hg_gate = nn.Parameter(torch.tensor(0.0))
        self.cross_gate = nn.Parameter(torch.tensor(0.0))

        self.reaction_modulator = ProductModulatedReaction(
            dim=self.unimol_emb_dim,
            hidden_dim=reaction_hidden_dim,
            dropout=reaction_dropout,
        ).to(self.device)

        self.layer_norm_seq = nn.LayerNorm(self.esm3_emb_dim)
        self.layer_norm_uni = nn.LayerNorm(self.unimol_emb_dim)

        final_in_dim = self.esm3_emb_dim + self.unimol_emb_dim
        self.shared = nn.Sequential(
            nn.Linear(final_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.zero_head = nn.Linear(hidden_dim // 2, 1)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def _load_atomic_tables(self, expect_smiles, pkl_path):
        with open(pkl_path, "rb") as f:
            smi_to_vec = pickle.load(f)

        tables = []
        for smi in expect_smiles:
            if smi not in smi_to_vec:
                raise KeyError(f"SMILES not found in pkl_by_smiles: {smi}")
            arr = np.asarray(smi_to_vec[smi], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.unimol_emb_dim:
                raise ValueError(
                    f"SMILES {smi} shape error: got {arr.shape}, expected [num_atoms, {self.unimol_emb_dim}]"
                )
            tables.append(torch.from_numpy(arr))
        return tables

    def _load_hyperdata(self, enzyme_id: str):
        enzyme_id = str(enzyme_id)
        if enzyme_id not in self._hypergraph_cache:
            pt_path = os.path.join(self.hypergraph_dir, f"{enzyme_id}.pt")
            if not os.path.isfile(pt_path):
                raise FileNotFoundError(f"Missing hypergraph file: {pt_path}")

            data = torch.load(pt_path, map_location="cpu")
            if not hasattr(data, "x"):
                raise ValueError(f"{enzyme_id} missing x")
            if not hasattr(data, "hyperedge_index"):
                raise ValueError(f"{enzyme_id} missing hyperedge_index")
            if data.x.ndim != 2 or data.x.shape[1] != self.esm3_emb_dim:
                raise ValueError(
                    f"{enzyme_id} x shape error: {tuple(data.x.shape)}, expected [num_nodes, {self.esm3_emb_dim}]"
                )
            if data.hyperedge_index.ndim != 2 or data.hyperedge_index.shape[0] != 2:
                raise ValueError(
                    f"{enzyme_id} hyperedge_index shape error: {tuple(data.hyperedge_index.shape)}"
                )
            if data.hyperedge_index.numel() == 0:
                raise ValueError(f"{enzyme_id} hyperedge_index is empty")

            self._hypergraph_cache[enzyme_id] = data
        return self._hypergraph_cache[enzyme_id]

    def _get_atom_matrix(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self._atom_tables):
            raise IndexError(f"molecule index out of range: {idx}")
        return self._atom_tables[idx].to(self.device)

    def _get_seq_vecs(self, enzyme_ids: List[str], return_token_strength: bool = False):
        seq_vec_list = []
        seq_strength_list = [] if return_token_strength else None
        seq_token_list = [] if return_token_strength else None

        for eid in enzyme_ids:
            x = self._enzyme_table[str(eid)]
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device)

            if x.ndim != 2 or x.shape[1] != self.esm3_emb_dim:
                raise ValueError(
                    f"enzyme {eid} token shape error: got {tuple(x.shape)}, expected [L, {self.esm3_emb_dim}]"
                )

            seq_vec, token_strength = self.residue_pool(x, return_attn=True)
            seq_vec_list.append(seq_vec)
            if return_token_strength:
                seq_strength_list.append(token_strength)
                seq_token_list.append(x)

        seq_vec = torch.stack(seq_vec_list, dim=0)
        return seq_vec, seq_strength_list, seq_token_list


    def _process_molecules(self, batch, return_react_attn: bool = False):
        react_idx = batch["react_idx"].long().to(self.device)
        prod_idx = batch["prod_idx"].long().to(self.device)

        if react_idx.ndim != 1 or prod_idx.ndim != 1:
            raise ValueError("react_idx and prod_idx must be 1D")
        if react_idx.shape[0] != prod_idx.shape[0]:
            raise ValueError("react_idx and prod_idx batch size mismatch")

        react_atom_list = []
        prod_atom_list = []
        R_global_list = []
        P_global_list = []
        react_pool_attn_list = [] if return_react_attn else None

        for i in react_idx.tolist():
            atom_x = self._get_atom_matrix(i)
            react_atom_list.append(atom_x)
            if return_react_attn:
                pooled, attn = self.atom_pool(atom_x, return_attn=True)
                R_global_list.append(pooled)
                react_pool_attn_list.append(attn)
            else:
                R_global_list.append(self.atom_pool(atom_x))

        for j in prod_idx.tolist():
            atom_x = self._get_atom_matrix(j)
            prod_atom_list.append(atom_x)
            P_global_list.append(self.atom_pool(atom_x))

        R_global = torch.stack(R_global_list, dim=0)
        P_global = torch.stack(P_global_list, dim=0)
        return react_atom_list, R_global, prod_atom_list, P_global, react_pool_attn_list

    def _compute_final_residue_strength(
        self,
        seq_tokens: torch.Tensor,
        H_final: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Static residue importance without sequence softmax attention:
        importance_i = || x_i + alpha * proj(H_i) ||
        """
        if seq_tokens.ndim != 2:
            raise ValueError(f"seq_tokens must be [N, D], got {tuple(seq_tokens.shape)}")
        if H_final.ndim != 2:
            raise ValueError(f"H_final must be [N, D], got {tuple(H_final.shape)}")
        if seq_tokens.shape[0] != H_final.shape[0]:
            raise ValueError(
                f"seq_tokens and H_final node mismatch: {seq_tokens.shape[0]} vs {H_final.shape[0]}"
            )

        graph_tokens = self.protein_proj(H_final)
        seq_contrib = seq_tokens
        graph_contrib = alpha * graph_tokens
        final_contrib = seq_contrib + graph_contrib
        return torch.norm(final_contrib, dim=-1)


    def _get_teacher_protein_embedding(
        self,
        enzyme_ids: List[str],
        react_atom_list: List[torch.Tensor],
        return_residue_strength: bool = False,
    ):
        seq_vec, seq_attn_list, seq_token_list = self._get_seq_vecs(
            enzyme_ids,
            return_token_strength=return_residue_strength,
        )

        final_struct_list = []
        final_node_strength_list = [] if return_residue_strength else None

        beta = torch.sigmoid(self.cross_gate)
        alpha = torch.sigmoid(self.hg_gate)

        for i, eid in enumerate(enzyme_ids):
            data = self._load_hyperdata(str(eid)).to(self.device)
            H = self.hg_encoder(data)
            react_atoms = react_atom_list[i]
            H_attn = self.cross_attn(query_nodes=H, react_atoms=react_atoms, return_attn=False)
            H_final = H + beta * H_attn
            final_struct = self.final_mean_pool(H_final)

            final_struct_list.append(final_struct)
            if return_residue_strength:
                if seq_token_list[i].shape[0] != H_final.shape[0]:
                    raise ValueError(
                        f"enzyme {eid} residue length mismatch between seq tokens ({seq_token_list[i].shape[0]}) and hypergraph nodes ({H_final.shape[0]})"
                    )
                residue_strength = self._compute_final_residue_strength(
                    seq_tokens=seq_token_list[i],
                    H_final=H_final,
                    alpha=alpha,
                )
                final_node_strength_list.append(residue_strength)

        hg_vec = torch.stack(final_struct_list, dim=0)
        protein_emb = seq_vec + alpha * self.protein_proj(hg_vec)

        out = {
            "protein_emb_raw": protein_emb,
            "seq_vec": seq_vec,
            "hg_vec": hg_vec,
            "alpha": alpha,
            "beta": beta,
        }
        if return_residue_strength:
            out["final_residue_strength"] = final_node_strength_list
        return out

    def forward(
        self,
        batch,
        return_emb: bool = False,
        return_residue_strength: bool = False,
        return_reactant_atom_strength: bool = False,
        return_gate: bool = False,
    ):
        required_keys = ["enzyme_ids", "react_idx", "prod_idx", "labels"]
        for k in required_keys:
            if k not in batch:
                raise KeyError(f"batch missing key: {k}")

        enzyme_ids = batch["enzyme_ids"]
        react_atom_list, R_global, _, P_global, react_pool_attn_list = self._process_molecules(
            batch,
            return_react_attn=return_reactant_atom_strength,
        )

        prot_out = self._get_teacher_protein_embedding(
            enzyme_ids=enzyme_ids,
            react_atom_list=react_atom_list,
            return_residue_strength=return_residue_strength,
        )
        protein_emb = prot_out["protein_emb_raw"]

        if return_gate:
            reaction_emb, prod_gate = self.reaction_modulator(
                R_global=R_global,
                P_global=P_global,
                return_gate=True,
            )
        else:
            reaction_emb = self.reaction_modulator(
                R_global=R_global,
                P_global=P_global,
                return_gate=False,
            )

        protein_emb = self.layer_norm_seq(protein_emb)
        reaction_emb = self.layer_norm_uni(reaction_emb)

        x = torch.cat([protein_emb, reaction_emb], dim=-1)
        h = self.shared(x)
        logits_zero = self.zero_head(h)
        reg_pos = F.softplus(self.reg_head(h))

        labels = batch["labels"].to(self.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        elif labels.ndim != 2 or labels.shape[1] != 1:
            raise ValueError(f"labels shape must be [B] or [B,1], got {tuple(labels.shape)}")

        if not return_emb:
            return logits_zero, reg_pos, labels

        extra = {
            "protein_emb": protein_emb,
            "reaction_emb": reaction_emb,
            "concat_emb": x,
            "fused_emb": h,
            "alpha": prot_out["alpha"],
            "beta": prot_out["beta"],
            "hg_scale": prot_out["alpha"],
        }
        if return_residue_strength:
            extra["final_residue_strength"] = prot_out["final_residue_strength"]
        if return_reactant_atom_strength:
            extra["reactant_atom_strength"] = react_pool_attn_list
        if return_gate:
            extra["prod_gate"] = prod_gate

        return logits_zero, reg_pos, labels, extra

    # =========================
    # Single-sample explanation: final residue weights plus reactant atom weights.
    # =========================
    def build_single_sample_graph(
        self,
        enzyme_id: str,
        react_idx: int,
        prod_idx: int,
        return_all_attn: bool = False,
    ):
        enzyme_id = str(enzyme_id)

        seq_x = self._enzyme_table[enzyme_id]
        if not torch.is_tensor(seq_x):
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_x = seq_x.to(self.device)
        seq_vec, seq_token_strength = self.residue_pool(seq_x, return_attn=True)

        react_atoms = self._get_atom_matrix(int(react_idx)).detach().clone().requires_grad_(False)
        prod_atoms = self._get_atom_matrix(int(prod_idx)).detach().clone().requires_grad_(False)

        R_global, react_pool_attn = self.atom_pool(react_atoms, return_attn=True)
        P_global, prod_pool_attn = self.atom_pool(prod_atoms, return_attn=True)

        data = self._load_hyperdata(enzyme_id).to(self.device)
        H = self.hg_encoder(data)

        alpha = torch.sigmoid(self.hg_gate)
        beta = torch.sigmoid(self.cross_gate)

        if return_all_attn:
            H_attn, cross_attn_map = self.cross_attn(
                query_nodes=H,
                react_atoms=react_atoms,
                return_attn=True,
            )
        else:
            H_attn = self.cross_attn(
                query_nodes=H,
                react_atoms=react_atoms,
                return_attn=False,
            )
            cross_attn_map = None

        H_final = H + beta * H_attn
        hg_vec = self.final_mean_pool(H_final)
        final_residue_strength = self._compute_final_residue_strength(
            seq_tokens=seq_x,
            H_final=H_final,
            alpha=alpha,
        )

        hg_vec_proj = self.protein_proj(hg_vec.unsqueeze(0)).squeeze(0)
        protein_emb = seq_vec + alpha * hg_vec_proj
        protein_emb = self.layer_norm_seq(protein_emb.unsqueeze(0)).squeeze(0)

        reaction_emb, prod_gate = self.reaction_modulator(
            R_global=R_global.unsqueeze(0),
            P_global=P_global.unsqueeze(0),
            return_gate=True,
        )
        reaction_emb = self.layer_norm_uni(reaction_emb).squeeze(0)

        fused = torch.cat([protein_emb, reaction_emb], dim=-1).unsqueeze(0)
        h = self.shared(fused)
        logits_zero = self.zero_head(h)
        reg_pos = F.softplus(self.reg_head(h))
        pred_score = (torch.sigmoid(logits_zero) * reg_pos).squeeze()

        return {
            "pred_score": pred_score,
            "seq_token_strength": seq_token_strength.detach().cpu(),
            "final_residue_strength": final_residue_strength.detach().cpu(),
            "react_pool_attn": react_pool_attn.detach().cpu(),
            "prod_pool_attn": prod_pool_attn.detach().cpu(),
            "cross_attn_map": None if cross_attn_map is None else cross_attn_map.detach().cpu(),
            "alpha": alpha.detach().cpu(),
            "beta": beta.detach().cpu(),
            "prod_gate": prod_gate.squeeze(0).detach().cpu(),
        }

    def explain_one_sample(self, enzyme_id: str, react_idx: int, prod_idx: int):
        return self.build_single_sample_graph(
            enzyme_id=enzyme_id,
            react_idx=react_idx,
            prod_idx=prod_idx,
            return_all_attn=False,
        )


if __name__ == "__main__":
    print("This file defines CombinationModel with no-softmax sequence pooling and reactant atom ranking export support.")
