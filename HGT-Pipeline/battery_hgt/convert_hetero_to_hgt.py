#!/usr/bin/env python3
"""
convert_hetero_to_hgt.py — Convert output/hetero_data.pt → HGT edge CSVs
=======================================================================

Uses the Stage-3 HeteroData graph (preferred over re-parsing 500MB TTL).
Builds a CrystaLLM-oriented subgraph: materials, elements, structure,
battery roles — skips AtomicSite bond edges (too large for CPU HGT demo).

Also adds Material → contains_element → Element from formulas in node_mapping.

Usage
-----
  python convert_hetero_to_hgt.py \\
    --output-dir ../../output \\
    --out-dir data/BatteryKG_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

import torch


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
    """Human-readable name for id_maps / composition lookup."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=os.path.join("..", "..", "output"),
                   help="Folder with hetero_data.pt + node_mapping.json")
    p.add_argument("--out-dir", default=os.path.join("data", "BatteryKG_v2"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--max-property-edges", type=int, default=20000,
                   help="Cap Material→Property edges for CPU tractability.")
    p.add_argument("--include-sites", action="store_true",
                   help="Include AtomicSite edges (very large; slow).")
    args = p.parse_args()

    out_src = os.path.abspath(args.output_dir)
    pt_path = os.path.join(out_src, "hetero_data.pt")
    nm_path = os.path.join(out_src, "node_mapping.json")
    if not os.path.exists(pt_path):
        print(f"Missing {pt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading HeteroData: {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    print(f"Loading node_mapping: {nm_path}")
    node_meta = json.load(open(nm_path, encoding="utf-8"))

    # Edge types to keep for CrystaLLM-oriented HGT
    keep_edges = [
        ("Material", "hasStructure", "Crystal"),
        ("Crystal", "hasUnitCell", "UnitCell"),
        ("Crystal", "hasSpaceGroup", "SpaceGroup"),
        ("Crystal", "hasCrystalSystem", "CrystalSystem"),
        ("Material", "belongsToElectrode", "BatteryRole"),
        ("BatteryRole", "usesMaterial", "Material"),
        ("Material", "hasProperty", "Property"),
        ("AtomicSpecies", "hasElement", "Element"),
    ]
    if args.include_sites:
        keep_edges.extend([
            ("UnitCell", "containsSite", "AtomicSite"),
            ("AtomicSite", "containsSpecies", "AtomicSpecies"),
            ("AtomicSite", "hasBondTo", "AtomicSite"),
        ])

    typed_edges: list[dict] = []
    node_type_map: dict[str, str] = {}
    node_label_map: dict[str, str] = {}

    def ensure(ntype: str, idx: int):
        nid = node_id(ntype, idx)
        if nid not in node_type_map:
            node_type_map[nid] = sanitize(ntype)
            node_label_map[nid] = str(node_name(node_meta, ntype, idx))
        return nid

    # Register all nodes of kept types from mapping
    used_types = set()
    for src, rel, dst in keep_edges:
        used_types.add(src)
        used_types.add(dst)
    used_types.add("Element")
    used_types.add("Material")

    for ntype in used_types:
        for i in range(len(node_meta[ntype])):
            ensure(ntype, i)

    # Hetero edges
    for src, rel, dst in keep_edges:
        if (src, rel, dst) not in data.edge_types:
            print(f"  skip missing edge type: {src}-{rel}-{dst}")
            continue
        ei = data[src, rel, dst].edge_index
        n_e = ei.size(1)
        take = list(range(n_e))
        if (src, rel, dst) == ("Material", "hasProperty", "Property") and n_e > args.max_property_edges:
            random.seed(args.seed)
            take = random.sample(take, args.max_property_edges)
            print(f"  capped {src}-{rel}-{dst}: {n_e} → {len(take)}")
        # Deduplicate AtomicSpecies→Element
        seen_pairs = set()
        for j in take:
            s_idx = int(ei[0, j])
            d_idx = int(ei[1, j])
            if (src, rel, dst) == ("AtomicSpecies", "hasElement", "Element"):
                key = (s_idx, d_idx)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
            h_id = ensure(src, s_idx)
            t_id = ensure(dst, d_idx)
            typed_edges.append({
                "head_id": h_id,
                "head_type": sanitize(src),
                "relation": sanitize(rel),
                "tail_id": t_id,
                "tail_type": sanitize(dst),
            })
        print(f"  {src}-{rel}-{dst}: +edges (total now {len(typed_edges)})")

    # Material → Element from formulas (critical for CrystaLLM composition lookup)
    # Build symbol → element tensor index
    sym_to_idx = {}
    for e in node_meta["Element"]:
        sym_to_idx[e["symbol"]] = e["tensor_index"]

    me_count = 0
    for m in node_meta["Material"]:
        formula = m.get("formula") or ""
        m_idx = m["tensor_index"]
        h_id = ensure("Material", m_idx)
        for el in parse_elements(formula):
            if el not in sym_to_idx:
                continue
            t_id = ensure("Element", sym_to_idx[el])
            typed_edges.append({
                "head_id": h_id,
                "head_type": "material",
                "relation": "contains_element",
                "tail_id": t_id,
                "tail_type": "element",
            })
            me_count += 1
    print(f"  Material-contains_element-Element: {me_count}")

    # Deduplicate
    seen = set()
    uniq = []
    for e in typed_edges:
        key = (e["head_id"], e["relation"], e["tail_id"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    typed_edges = uniq
    print(f"Total unique edges: {len(typed_edges):,}")
    print(f"Total nodes: {len(node_type_map):,}")

    # Schema
    node_types = sorted(set(node_type_map.values()))
    relation_types = sorted(set(e["relation"] for e in typed_edges))
    meta_counts: dict[tuple, int] = defaultdict(int)
    for e in typed_edges:
        meta_counts[(e["head_type"], e["relation"], e["tail_type"])] += 1
    schema = {
        "node_types": node_types,
        "relation_types": relation_types,
        "meta_relations": [
            {"head_type": ht, "relation": rel, "tail_type": tt, "count": cnt}
            for (ht, rel, tt), cnt in sorted(meta_counts.items(), key=lambda x: -x[1])
        ],
        "source": os.path.abspath(out_src),
        "source_version": "output hetero_data.pt (Stage 2.3.2)",
    }

    random.seed(args.seed)
    random.shuffle(typed_edges)
    n = len(typed_edges)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    train_e = typed_edges[:n_train]
    val_e = typed_edges[n_train:n_train + n_val]
    test_e = typed_edges[n_train + n_val:]

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    def write_csv(rows, path, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved {len(rows):>8,} → {path}")

    nodes = [
        {"node_id": nid, "node_type": nt, "node_name": node_label_map[nid]}
        for nid, nt in sorted(node_type_map.items())
    ]
    fields_e = ["head_id", "head_type", "relation", "tail_id", "tail_type"]
    write_csv(nodes, os.path.join(out, "nodes.csv"), ["node_id", "node_type", "node_name"])
    write_csv(train_e, os.path.join(out, "edges_train.csv"), fields_e)
    write_csv(val_e, os.path.join(out, "edges_val.csv"), fields_e)
    write_csv(test_e, os.path.join(out, "edges_test.csv"), fields_e)
    write_csv(typed_edges, os.path.join(out, "edges_all.csv"), fields_e)
    with open(os.path.join(out, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema → {os.path.join(out, 'schema.json')}")
    print("\nTrain with:")
    print(f"  python train.py --data-dir {out} --model-dir models/battery_v2 --output-dir outputs/battery_v2")


if __name__ == "__main__":
    main()
