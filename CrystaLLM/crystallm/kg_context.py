"""
kg_context.py — Build HGT KG context tensors for CrystaLLM cross-attention
==========================================================================

Loads export_package/node_embeddings.pt + id_maps.json and, given a
composition string (e.g. "LiFePO4"), returns stacked HGT vectors:

    kg_raw : FloatTensor [1, N_kg, kg_dim]
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import torch

# Node-type ids for KGContextBuilder (optional metadata; not used by kg_proj).
KG_TYPE_TO_ID: dict[str, int] = {
    "material": 0,
    "crystal": 1,
    "element": 2,
    "species": 3,
    "role": 4,
    "property": 5,
    "other": 6,
}


def parse_elements(composition: str) -> list[str]:
    """Parse chemical formula into element symbols. 'LiFePO4' → ['Li','Fe','P','O']."""
    formula = composition.strip()
    # Strip data_ prefix if present
    if formula.startswith("data_"):
        formula = formula[5:]
    # Expand simple parentheses groups: (PO4)3 → PO4PO4PO4
    while True:
        m = re.search(r"\(([A-Za-z0-9]+)\)(\d*)", formula)
        if not m:
            break
        body, n = m.group(1), int(m.group(2) or "1")
        formula = formula[: m.start()] + body * n + formula[m.end() :]
    elems = re.findall(r"[A-Z][a-z]?", formula)
    # Deduplicate preserving order
    seen = set()
    out = []
    for e in elems:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


class KGContextBuilder:
    """Lookup HGT vectors for a composition from an HGT export_package."""

    def __init__(self, export_dir: str, device: str | torch.device = "cpu"):
        self.export_dir = export_dir
        self.device = torch.device(device)

        emb_path = os.path.join(export_dir, "node_embeddings.pt")
        id_path = os.path.join(export_dir, "id_maps.json")
        meta_path = os.path.join(export_dir, "meta.json")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Missing {emb_path}. Run export_embeddings.py first.")
        if not os.path.exists(id_path):
            raise FileNotFoundError(f"Missing {id_path}. Run export_embeddings.py first.")

        self.node_embeddings: dict[str, torch.Tensor] = torch.load(
            emb_path, map_location="cpu", weights_only=False
        )
        with open(id_path, encoding="utf-8") as f:
            self.id_maps: dict[str, Any] = json.load(f)
        self.meta: dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)

        self.kg_dim = int(self.meta.get("embedding_dim") or next(iter(self.node_embeddings.values())).shape[-1])
        self.name_to_nodes: dict[str, list[dict]] = self.id_maps.get("name_to_nodes", {})

    def _lookup(self, key: str) -> list[tuple[str, int, str]]:
        """Return list of (node_type, local_idx, display_name) for key."""
        hits = self.name_to_nodes.get(key) or self.name_to_nodes.get(key.lower()) or []
        out = []
        for h in hits:
            out.append((h["node_type"], int(h["local_idx"]), h.get("name", key)))
        return out

    def build(
        self,
        composition: str,
        top_k: int = 8,
        include_material: bool = True,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """
        Build kg_raw [1, N_kg, kg_dim] for a composition.

        Returns
        -------
        kg_raw : FloatTensor [1, N, kg_dim]
        matched_names : list of node names used
        type_ids : LongTensor [1, N]
        """
        elements = parse_elements(composition)
        matched: list[tuple[str, int, str]] = []
        seen_ids = set()       # (node_type, local_idx)
        seen_names = set()     # avoid 50× "material:LiFePO4" crowding out elements

        def add_hit(hit, *, allow_dup_name: bool = False):
            ntype, local_idx, name = hit
            sig = (ntype, local_idx)
            name_key = f"{ntype}:{name}"
            if sig in seen_ids:
                return
            if not allow_dup_name and name_key in seen_names:
                return
            seen_ids.add(sig)
            seen_names.add(name_key)
            matched.append(hit)

        # Prefer one material node, then one crystal, for the formula
        if include_material:
            material_hit = None
            crystal_hit = None
            for key in (composition, composition.strip(), f"Material {composition}"):
                for hit in self._lookup(key):
                    if hit[0] == "material" and material_hit is None:
                        material_hit = hit
                    elif hit[0] == "crystal" and crystal_hit is None:
                        crystal_hit = hit
                if material_hit:
                    break
            if material_hit:
                add_hit(material_hit)
            elif crystal_hit:
                add_hit(crystal_hit)

        # Then one vector per element symbol (prefer type "element")
        for el in elements:
            hits = self._lookup(el)
            elem_hits = [h for h in hits if h[0] == "element"] or hits
            for hit in elem_hits[:1]:
                add_hit(hit)

        # Optional extras (species, roles) if room under top_k
        if include_material:
            for key in (composition, composition.strip()):
                for hit in self._lookup(key):
                    if hit[0] not in ("material", "crystal"):
                        add_hit(hit)
                if len(matched) >= top_k:
                    break

        if not matched:
            # Fallback: zero vector so generation still runs (caller may skip KG)
            kg = torch.zeros(1, 1, self.kg_dim, device=self.device)
            type_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            return kg, ["<null>"], type_ids

        matched = matched[:top_k]
        rows = []
        names = []
        type_ids = []
        for ntype, local_idx, name in matched:
            emb = self.node_embeddings[ntype][local_idx]
            rows.append(emb)
            names.append(f"{ntype}:{name}")
            type_ids.append(KG_TYPE_TO_ID.get(ntype, KG_TYPE_TO_ID["other"]))

        kg = torch.stack(rows, dim=0).unsqueeze(0).to(self.device)  # [1, N, D]
        types = torch.tensor(type_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return kg, names, types


def build_kg_context(
    export_dir: str,
    composition: str,
    device: str | torch.device = "cpu",
    top_k: int = 8,
) -> tuple[torch.Tensor, list[str], int, torch.Tensor]:
    """Convenience wrapper. Returns (kg_raw, matched_names, kg_dim, type_ids)."""
    builder = KGContextBuilder(export_dir, device=device)
    kg, names, type_ids = builder.build(composition, top_k=top_k)
    return kg, names, builder.kg_dim, type_ids
