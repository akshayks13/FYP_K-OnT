"""
hetero_graph.py — Load output/hetero_data.pt for HGT training

Keeps the same CrystaLLM-oriented subgraph as convert_hetero_to_hgt.py
(drop site/bond edges, cap properties, add Material→contains_element→Element)
but feeds HeteroData `.x` features instead of throwing them away.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any

import torch


KEEP_EDGES = [
    ("Material", "hasStructure", "Crystal"),
    ("Crystal", "hasUnitCell", "UnitCell"),
    ("Crystal", "hasSpaceGroup", "SpaceGroup"),
    ("Crystal", "hasCrystalSystem", "CrystalSystem"),
    ("Material", "belongsToElectrode", "BatteryRole"),
    ("BatteryRole", "usesMaterial", "Material"),
    ("Material", "hasProperty", "Property"),
    ("AtomicSpecies", "hasElement", "Element"),
]

SITE_EDGES = [
    ("UnitCell", "containsSite", "AtomicSite"),
    ("AtomicSite", "containsSpecies", "AtomicSpecies"),
    ("AtomicSite", "hasBondTo", "AtomicSite"),
]


def sanitize(s: str) -> str:
    return s.lower().replace("/", "_").replace(" ", "_").replace("-", "_")


def parse_elements(formula: str) -> list[str]:
    f = formula.strip()
    while True:
        m = re.search(r"\(([A-Za-z0-9]+)\)(\d*)", f)
        if not m:
            break
        body, n = m.group(1), int(m.group(2) or "1")
        f = f[: m.start()] + body * n + f[m.end() :]
    elems = re.findall(r"[A-Z][a-z]?", f)
    seen, out = set(), []
    for e in elems:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def node_id(ntype: str, idx: int) -> str:
    return f"{sanitize(ntype)}/{idx}"


def node_name(meta: dict, ntype: str, idx: int) -> str:
    entry = meta[ntype][idx]
    if ntype == "Material":
        return entry.get("formula") or entry.get("material_id") or str(idx)
    if ntype == "Element":
        return entry.get("symbol") or str(idx)
    if ntype == "AtomicSpecies":
        return entry.get("symbol") or entry.get("species") or str(idx)
    if ntype == "SpaceGroup":
        return entry.get("symbol") or str(entry.get("spacegroup_number", idx))
    if ntype == "CrystalSystem":
        return entry.get("crystal_system") or str(idx)
    if ntype == "BatteryRole":
        return entry.get("role_name") or str(idx)
    if ntype == "Crystal":
        return entry.get("formula") or entry.get("material_id") or str(idx)
    if ntype == "UnitCell":
        return entry.get("material_id") or str(idx)
    if ntype == "Property":
        return entry.get("uri", "").rsplit("/", 1)[-1] or str(idx)
    return str(idx)


def default_hetero_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "output"))


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    """Per-column z-score. All-zero columns stay zero."""
    x = x.float().clone()
    if x.numel() == 0:
        return x
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (x - mean) / std


def features_are_useful(x: torch.Tensor) -> bool:
    return x.numel() > 0 and bool((x.abs().sum() > 0).item())


@dataclass
class HeteroBundle:
    train_edges: list[dict]
    val_edges: list[dict]
    test_edges: list[dict]
    node2localid: dict[str, dict[str, int]]
    ntype2nodes: dict[str, list[str]]
    node_names: dict[str, str]
    rel2id: dict[str, int]
    meta_rels: list[tuple[str, str, str]]
    feat_dims: dict[str, int]
    feat_x: dict[str, torch.Tensor]
    num_nodes: dict[str, int]
    feature_types: list[str] = field(default_factory=list)
    id_types: list[str] = field(default_factory=list)


def load_hetero_bundle(
    hetero_dir: str,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    max_property_edges: int = 20000,
    include_sites: bool = False,
    max_train_edges: int | None = None,
) -> HeteroBundle:
    pt_path = os.path.join(hetero_dir, "hetero_data.pt")
    nm_path = os.path.join(hetero_dir, "node_mapping.json")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing {pt_path}")
    if not os.path.exists(nm_path):
        raise FileNotFoundError(f"Missing {nm_path}")

    print(f"  Loading HeteroData: {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    print(f"  Loading node_mapping: {nm_path}")
    node_meta: dict[str, Any] = json.load(open(nm_path, encoding="utf-8"))

    keep = list(KEEP_EDGES)
    if include_sites:
        keep.extend(SITE_EDGES)

    used_pascal = set()
    for src, _rel, dst in keep:
        used_pascal.add(src)
        used_pascal.add(dst)
    used_pascal.add("Element")
    used_pascal.add("Material")

    ntype2nodes: dict[str, list[str]] = {}
    node2localid: dict[str, dict[str, int]] = {}
    node_names: dict[str, str] = {}
    num_nodes: dict[str, int] = {}

    for pascal in sorted(used_pascal):
        nt = sanitize(pascal)
        n = len(node_meta[pascal])
        num_nodes[nt] = n
        nodes = [node_id(pascal, i) for i in range(n)]
        ntype2nodes[nt] = nodes
        node2localid[nt] = {nid: i for i, nid in enumerate(nodes)}
        for i in range(n):
            node_names[nodes[i]] = str(node_name(node_meta, pascal, i))

    typed_edges: list[dict] = []
    rng = random.Random(seed)

    for src, rel, dst in keep:
        if (src, rel, dst) not in data.edge_types:
            print(f"  skip missing edge type: {src}-{rel}-{dst}")
            continue
        ei = data[src, rel, dst].edge_index
        n_e = int(ei.size(1))
        take = list(range(n_e))
        if (src, rel, dst) == ("Material", "hasProperty", "Property") and n_e > max_property_edges:
            take = rng.sample(take, max_property_edges)
            print(f"  capped {src}-{rel}-{dst}: {n_e} → {len(take)}")
        seen_pairs: set[tuple[int, int]] = set()
        added = 0
        for j in take:
            s_idx = int(ei[0, j])
            d_idx = int(ei[1, j])
            if (src, rel, dst) == ("AtomicSpecies", "hasElement", "Element"):
                key = (s_idx, d_idx)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
            typed_edges.append({
                "head_id": node_id(src, s_idx),
                "head_type": sanitize(src),
                "relation": sanitize(rel),
                "tail_id": node_id(dst, d_idx),
                "tail_type": sanitize(dst),
            })
            added += 1
        print(f"  {src}-{rel}-{dst}: +{added} (total {len(typed_edges)})")

    # Material → Element from formulas (needed for CrystaLLM composition lookup)
    sym_to_idx: dict[str, int] = {}
    for e in node_meta["Element"]:
        if e.get("symbol"):
            sym_to_idx[e["symbol"]] = int(e["tensor_index"])

    me_count = 0
    for m in node_meta["Material"]:
        formula = m.get("formula") or ""
        m_idx = int(m["tensor_index"])
        for el in parse_elements(formula):
            if el not in sym_to_idx:
                continue
            typed_edges.append({
                "head_id": node_id("Material", m_idx),
                "head_type": "material",
                "relation": "contains_element",
                "tail_id": node_id("Element", sym_to_idx[el]),
                "tail_type": "element",
            })
            me_count += 1
    print(f"  Material-contains_element-Element: {me_count}")

    seen: set[tuple[str, str, str]] = set()
    uniq: list[dict] = []
    for e in typed_edges:
        key = (e["head_id"], e["relation"], e["tail_id"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    typed_edges = uniq
    print(f"  Unique edges: {len(typed_edges):,}")

    rng.shuffle(typed_edges)
    n = len(typed_edges)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_edges = typed_edges[:n_train]
    val_edges = typed_edges[n_train:n_train + n_val]
    test_edges = typed_edges[n_train + n_val:]
    if max_train_edges is not None:
        train_edges = train_edges[:max_train_edges]
        print(f"  Capped train edges: {len(train_edges):,}")

    rel2id = {r: i for i, r in enumerate(sorted({e["relation"] for e in train_edges}))}
    meta_rels = sorted({
        (e["head_type"], e["relation"], e["tail_type"]) for e in train_edges
    })

    feat_x: dict[str, torch.Tensor] = {}
    feat_dims: dict[str, int] = {}
    feature_types: list[str] = []
    id_types: list[str] = []

    print("  Node features (.x):")
    for pascal in sorted(used_pascal):
        nt = sanitize(pascal)
        x = data[pascal].x.float()
        useful = features_are_useful(x)
        if useful:
            x_n = normalize_features(x)
            feat_x[nt] = x_n
            feat_dims[nt] = int(x_n.shape[1])
            feature_types.append(nt)
            fill = float((x.abs() > 1e-9).float().mean())
            print(f"    {nt:<16} {tuple(x.shape)}  z-scored  fill={fill:.2f}")
        else:
            feat_dims[nt] = 0
            id_types.append(nt)
            print(f"    {nt:<16} {tuple(x.shape)}  all-zero → ID embedding")

    return HeteroBundle(
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        node2localid=node2localid,
        ntype2nodes=ntype2nodes,
        node_names=node_names,
        rel2id=rel2id,
        meta_rels=meta_rels,
        feat_dims=feat_dims,
        feat_x=feat_x,
        num_nodes=num_nodes,
        feature_types=feature_types,
        id_types=id_types,
    )


def make_x_dict_from_bundle(bundle: HeteroBundle, device: torch.device) -> dict[str, torch.Tensor]:
    x_dict: dict[str, torch.Tensor] = {}
    for nt, n in bundle.num_nodes.items():
        if nt in bundle.feat_x:
            x_dict[nt] = bundle.feat_x[nt].to(device)
        else:
            x_dict[nt] = torch.arange(n, dtype=torch.long, device=device)
    return x_dict


def make_x_dict_from_maps(maps: dict, device: torch.device) -> dict[str, torch.Tensor]:
    if maps.get("input_mode") != "hetero_features":
        ntype2nodes = maps["ntype2nodes"]
        return {
            nt: torch.arange(len(nodes), dtype=torch.long, device=device)
            for nt, nodes in ntype2nodes.items()
        }
    x_dict: dict[str, torch.Tensor] = {}
    feat_x = maps["feat_x"]
    num_nodes = maps["num_nodes"]
    for nt in maps["node_types"]:
        if nt in feat_x:
            x_dict[nt] = feat_x[nt].to(device)
        else:
            x_dict[nt] = torch.arange(int(num_nodes[nt]), dtype=torch.long, device=device)
    return x_dict


def edge_index_from_maps(maps: dict, device: torch.device) -> dict[tuple, torch.Tensor]:
    raw = maps.get("edge_index_cpu") or {}
    out = {}
    for k, v in raw.items():
        key = tuple(k) if not isinstance(k, tuple) else k
        out[key] = v.to(device)
    return out
