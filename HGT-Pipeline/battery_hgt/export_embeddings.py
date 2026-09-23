#!/usr/bin/env python3
"""
export_embeddings.py — Export trained HGT node vectors for CrystaLLM
====================================================================

Loads hgt_checkpoint.pt + hgt_mappings.pt,
runs the HGT encoder on the training graph, and writes:

  export_package/
    node_embeddings.pt   {node_type: FloatTensor [N, d]}
    id_maps.json         name/id lookups for composition matching
    meta.json            embedding_dim, model_dir, etc.

Usage
-----
  python export_embeddings.py \\
    --model-dir models/battery \\
    --data-dir data/BatteryKG \\
    --out-dir models/battery/export_package
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch

from models.hgt import HGTLinkPredictor
from hetero_graph import edge_index_from_maps, make_x_dict_from_maps


def parse_args():
    p = argparse.ArgumentParser(description="Export HGT node embeddings.")
    p.add_argument("--model-dir", default=os.path.join("models", "battery_v2"))
    p.add_argument("--data-dir", default=os.path.join("data", "BatteryKG_v2"))
    p.add_argument("--out-dir", default=None,
                   help="Default: <model-dir>/export_package")
    p.add_argument("--max-train-edges", type=int, default=None)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_mappings(model_dir: str):
    for name in ("hgt_mappings.pt", "battery_mappings.pt"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False), path
    print(f"Error: no hgt_mappings.pt in {model_dir}", file=sys.stderr)
    sys.exit(1)


def build_edge_index_dict(data_dir, node2localid, meta_rels, device, max_rows=None):
    path = os.path.join(data_dir, "edges_train.csv")
    buckets: dict[tuple, tuple[list, list]] = {mr: ([], []) for mr in meta_rels}
    count = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if max_rows is not None and count >= max_rows:
                break
            key = (row["head_type"], row["relation"], row["tail_type"])
            if key not in buckets:
                continue
            h = node2localid.get(row["head_type"], {}).get(row["head_id"])
            t = node2localid.get(row["tail_type"], {}).get(row["tail_id"])
            if h is None or t is None:
                continue
            buckets[key][0].append(h)
            buckets[key][1].append(t)
            count += 1
    result = {}
    for mr, (src, dst) in buckets.items():
        if src:
            result[mr] = torch.tensor([src, dst], dtype=torch.long, device=device)
    return result


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = args.out_dir or os.path.join(args.model_dir, "export_package")
    os.makedirs(out_dir, exist_ok=True)

    maps, maps_path = load_mappings(args.model_dir)
    ckpt_path = os.path.join(args.model_dir, "hgt_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Train first.", file=sys.stderr)
        sys.exit(1)

    node2localid = maps["node2localid"]
    ntype2nodes = maps["ntype2nodes"]
    node_names = maps.get("node_names", {})
    meta_rels = maps["meta_rels"]
    node_types = maps["node_types"]
    edge_types = maps["edge_types"]
    embedding_dim = maps["embedding_dim"]
    num_heads = maps["num_heads"]
    num_layers = maps["num_layers"]
    dropout = maps.get("dropout", 0.2)

    input_mode = maps.get("input_mode", "csv_ids")
    model = HGTLinkPredictor(
        node_types=node_types,
        edge_types=edge_types,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        feat_dims=maps.get("feat_dims") if input_mode == "hetero_features" else None,
        num_nodes=maps.get("num_nodes") if input_mode == "hetero_features" else None,
    )
    if input_mode != "hetero_features":
        model.init_node_embeddings({nt: len(nodes) for nt, nodes in ntype2nodes.items()})
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    model.to(device)

    if input_mode == "hetero_features" and maps.get("edge_index_cpu"):
        edge_index_dict = edge_index_from_maps(maps, device)
        x_dict = make_x_dict_from_maps(maps, device)
    else:
        edge_index_dict = build_edge_index_dict(
            args.data_dir, node2localid, meta_rels, device, args.max_train_edges
        )
        x_dict = {
            nt: torch.arange(len(nodes), device=device)
            for nt, nodes in ntype2nodes.items()
        }

    with torch.no_grad():
        h_dict = model.encode(x_dict, edge_index_dict)

    node_embeddings = {nt: h.detach().cpu() for nt, h in h_dict.items()}
    emb_path = os.path.join(out_dir, "node_embeddings.pt")
    torch.save(node_embeddings, emb_path)

    # Build reverse lookups for composition matching
    name_to_nodes: dict[str, list[dict]] = {}
    for ntype, nodes in ntype2nodes.items():
        for local_idx, nid in enumerate(nodes):
            name = node_names.get(nid, nid)
            entry = {
                "node_id": nid,
                "node_type": ntype,
                "local_idx": local_idx,
                "name": name,
            }
            # Index by full name and by short tokens
            for key in {name, name.lower(), nid, nid.split("/")[-1]}:
                name_to_nodes.setdefault(key, []).append(entry)
            # Element type: also index by symbol
            if ntype == "element":
                sym = name if len(name) <= 2 else nid.split("/")[-1]
                name_to_nodes.setdefault(sym, []).append(entry)
                name_to_nodes.setdefault(sym.lower(), []).append(entry)

    id_maps = {
        "node_types": node_types,
        "ntype2nodes": {k: list(v) for k, v in ntype2nodes.items()},
        "node_names": {str(k): v for k, v in node_names.items()},
        "name_to_nodes": name_to_nodes,
    }
    id_path = os.path.join(out_dir, "id_maps.json")
    with open(id_path, "w", encoding="utf-8") as f:
        json.dump(id_maps, f, indent=2)

    meta = {
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "model_dir": os.path.abspath(args.model_dir),
        "data_dir": os.path.abspath(args.data_dir),
        "mappings_path": os.path.abspath(maps_path),
        "node_counts": {nt: int(t.shape[0]) for nt, t in node_embeddings.items()},
        "input_mode": maps.get("input_mode", "csv_ids"),
        "feature_types": maps.get("feature_types", []),
        "id_types": maps.get("id_types", []),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported embeddings → {emb_path}")
    for nt, t in node_embeddings.items():
        print(f"  {nt:<20} {tuple(t.shape)}")
    print(f"id_maps → {id_path}")
    print(f"meta    → {meta_path}")


if __name__ == "__main__":
    main()
