from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGTLinkPredictor(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for link prediction on the battery KG.

    Architecture
    ────────────
    • Per-type input: HeteroData `.x` (Linear F→D) or ID embedding if `.x` is empty
    • Input projection  (embedding_dim → embedding_dim) per node type
    • Layer Normalisation before each HGTConv (training stability)
    • N x HGTConv layers with multi-head, type-specific attention
    • DistMult decoder:  score(h, r, t) = Σ_d  emb_h[d] · r_emb[d] · emb_t[d]

    Parameters
    ──────────
    node_types      : list of str          e.g. ['material', 'element', 'crystal', ...]
    edge_types      : list of (str,str,str) e.g. [('material','hasElement','element'), ...]
    embedding_dim   : int                  hidden dimension (default 64)
    num_heads       : int                  HGT attention heads (default 4)
    num_layers      : int                  GNN depth (default 2)
    dropout         : float                dropout probability (default 0.2)
    """

    def __init__(
        self,
        node_types:    list[str],
        edge_types:    list[tuple[str, str, str]],
        embedding_dim: int   = 64,
        num_heads:     int   = 4,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
        feat_dims:     dict[str, int] | None = None,
        num_nodes:     dict[str, int] | None = None,
    ):
        super().__init__()

        self.node_types    = node_types
        self.edge_types    = edge_types
        self.embedding_dim = embedding_dim
        self.num_layers    = num_layers
        self.dropout_p     = dropout
        self.feat_dims     = feat_dims or {}

        metadata = (node_types, edge_types)

        # ── Inputs: .x projection and/or ID tables ────────────────────────────
        self.feat_proj: nn.ModuleDict = nn.ModuleDict()
        self.node_embeddings: nn.ModuleDict = nn.ModuleDict()
        if feat_dims is not None and num_nodes is not None:
            self._build_inputs(feat_dims, num_nodes)

        # ── Input projection: embedding_dim → embedding_dim per node type ────
        self.input_proj = nn.ModuleDict({
            ntype: Linear(embedding_dim, embedding_dim)
            for ntype in node_types
        })

        # ── Layer normalisation before each HGTConv ───────────────────────────
        # Applied per-node-type to stabilise training on heterogeneous graphs.
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                ntype: nn.LayerNorm(embedding_dim)
                for ntype in node_types
            })
            for _ in range(num_layers)
        ])

        # ── HGT encoder layers ────────────────────────────────────────────────
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels  = embedding_dim,
                out_channels = embedding_dim,
                metadata     = metadata,
                heads        = num_heads,
            )
            for _ in range(num_layers)
        ])

        # ── Output layer norm (applied to final node representations) ─────────
        self.output_norm = nn.ModuleDict({
            ntype: nn.LayerNorm(embedding_dim)
            for ntype in node_types
        })

        # ── Relation embeddings for DistMult decoder ──────────────────────────
        relation_types = sorted({rel for _, rel, _ in edge_types})
        self.relation2idx  = {r: i for i, r in enumerate(relation_types)}
        self.rel_embeddings = nn.Embedding(len(relation_types), embedding_dim)

        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self):
        """Xavier uniform initialisation for all learnable weight tensors."""
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        for proj in self.input_proj.values():
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _build_inputs(self, feat_dims: dict[str, int], num_nodes: dict[str, int]):
        """Linear(F→D) for types with useful .x; ID table otherwise."""
        for ntype in self.node_types:
            fdim = int(feat_dims.get(ntype, 0) or 0)
            if fdim > 0:
                lin = Linear(fdim, self.embedding_dim)
                nn.init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
                self.feat_proj[ntype] = lin
            else:
                n = int(num_nodes[ntype])
                emb = nn.Embedding(n, self.embedding_dim)
                nn.init.xavier_uniform_(emb.weight)
                self.node_embeddings[ntype] = emb

    def init_node_embeddings(self, num_nodes_per_type: dict[str, int]):
        """
        CSV / ID-only path: one embedding table per type.
        Not needed when feat_dims + num_nodes were passed to __init__.
        """
        for ntype, n in num_nodes_per_type.items():
            if ntype in self.feat_proj or ntype in self.node_embeddings:
                continue
            emb = nn.Embedding(n, self.embedding_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.node_embeddings[ntype] = emb

    # ──────────────────────────────────────────────────────────────────────────

    def encode(
        self,
        x_dict:          dict[str, torch.Tensor],    # {ntype: arange(N_type)}
        edge_index_dict: dict[tuple, torch.Tensor],  # {(src,rel,dst): [2, E]}
    ) -> dict[str, torch.Tensor]:
        """
        Run the HGT encoder and return per-node-type embeddings.

        Parameters
        ──────────
        x_dict          : {node_type: FloatTensor [N, F] features, or LongTensor IDs}
        edge_index_dict : {(src_type, rel, dst_type): LongTensor [2, E]}

        Returns
        ───────
        h_dict : {node_type: FloatTensor [N_type, embedding_dim]}
        """
        # Initial embeddings + input projection
        h: dict[str, torch.Tensor] = {}
        for ntype, x in x_dict.items():
            if ntype in self.feat_proj:
                raw = self.feat_proj[ntype](x)               # [N, D] from .x
            else:
                raw = self.node_embeddings[ntype](x)         # [N, D] from IDs
            proj = self.input_proj[ntype](raw)
            h[ntype] = self.dropout(proj)

        # HGT layers with pre-norm + residual connection
        for i, conv in enumerate(self.convs):
            # Pre-norm
            h_normed = {
                ntype: self.layer_norms[i][ntype](feat)
                for ntype, feat in h.items()
            }
            # HGT message passing
            h_new = conv(h_normed, edge_index_dict)
            # ELU + dropout + residual — keep types that received no messages
            h_updated = {}
            for ntype, feat in h.items():
                if ntype in h_new:
                    h_updated[ntype] = self.dropout(F.elu(h_new[ntype])) + feat
                else:
                    h_updated[ntype] = feat
            h = h_updated

        # Final layer norm
        h = {
            ntype: self.output_norm[ntype](feat)
            for ntype, feat in h.items()
        }

        return h

    # ──────────────────────────────────────────────────────────────────────────

    def decode(
        self,
        h_dict:   dict[str, torch.Tensor],
        src_type: str,
        rel:      str,
        dst_type: str,
        src_idx:  torch.Tensor,   # [B]  local indices into ntype2nodes[src_type]
        dst_idx:  torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        DistMult score for a batch of (src_type, rel, dst_type) triples.

        score(h, r, t) = Σ_d  h_src[d] · r_emb[d] · h_dst[d]

        Returns
        ───────
        scores : FloatTensor [B]
        """
        h_src = h_dict[src_type][src_idx]                               # [B, D]
        h_dst = h_dict[dst_type][dst_idx]                               # [B, D]

        r_idx  = torch.tensor(
            self.relation2idx[rel], dtype=torch.long, device=h_src.device
        )
        r_emb  = self.rel_embeddings(r_idx.expand(src_idx.size(0)))     # [B, D]

        return (h_src * r_emb * h_dst).sum(dim=-1)                      # [B]

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x_dict:          dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        src_type: str,
        rel:      str,
        dst_type: str,
        src_idx:  torch.Tensor,
        dst_idx:  torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        h = self.encode(x_dict, edge_index_dict)
        return self.decode(h, src_type, rel, dst_type, src_idx, dst_idx)
