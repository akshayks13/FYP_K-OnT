#!/usr/bin/env python3
"""
predict.py — HGT Link Prediction Inference for Battery KG
==========================================================

Loads a trained HGT checkpoint and answers typed link prediction queries:

    Given (head_node, relation) → Top-k most likely tail nodes

Usage
-----
  # Interactive mode (shows available types & relations, lets you query)
  python predict.py --model-dir models/battery_v2 --data-dir data/BatteryKG_v2

  # Direct query by node ID / name
  python predict.py --head LiFePO4 --head-type material --relation hasElement --topk 10
  python predict.py --head-name "LiFePO4" --head-type material --relation hasElement

  # Show schema / all available node types and relations
  python predict.py --schema --model-dir models/battery_v2 --data-dir data/BatteryKG_v2

  # List nodes of a given type
  python predict.py --list-nodes material --limit 20 --model-dir models/battery_v2
"""

from __future__ import annotations

import os
import sys
import csv
import json
import random
import argparse
import torch
from collections import defaultdict
from models.hgt import HGTLinkPredictor
from hetero_graph import edge_index_from_maps, make_x_dict_from_maps


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="HGT link-prediction inference on battery knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Query
    p.add_argument("--head",       default=None, help="Head node ID  (e.g. HP:0001250).")
    p.add_argument("--head-type",  default=None, help="Head node type (e.g. phenotype).")
    p.add_argument("--head-name",  default=None, help="Head node name (fuzzy match against node labels).")
    p.add_argument("--relation",   default=None, help="Relation type  (e.g. is_a, part_of).")
    p.add_argument("--tail-type",  default=None, help="Force a specific tail type (auto-detected otherwise).")
    p.add_argument("--topk",       type=int, default=10, help="Number of top results to show.")

    # Utility modes
    p.add_argument("--schema",      action="store_true", help="Print schema and exit.")
    p.add_argument("--list-nodes",  metavar="TYPE",      help="List nodes of a given type.")
    p.add_argument("--list-rels",   action="store_true", help="List all relation types.")
    p.add_argument("--limit",       type=int, default=30, help="Limit for --list-* outputs.")

    # Paths
    p.add_argument("--data-dir",    default=os.path.join("data", "BatteryKG_v2"))
    p.add_argument("--model-dir",   default=os.path.join("models", "battery_v2"),
                   help="Directory containing checkpoint and mappings.")
    p.add_argument(
        "--max-train-edges", type=int, default=None,
        help="Must match value used in training if a subset was loaded.",
    )
    return p.parse_args()


# =============================================================================
# Load helpers
# =============================================================================

def resolve_mappings_path(model_dir: str) -> str:
    for name in ("hgt_mappings.pt", "battery_mappings.pt"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(model_dir, "hgt_mappings.pt")


def load_model_and_vocab(device: torch.device, model_dir: str = "models"):
    """Load mappings + model from saved files. Returns (model, maps)."""
    mapping_path = resolve_mappings_path(model_dir)
    ckpt_path    = os.path.join(model_dir, "hgt_checkpoint.pt")

    for path in [mapping_path, ckpt_path]:
        if not os.path.exists(path):
            print(f"Error: {path} not found.", file=sys.stderr)
            print("Train first:  python train.py --data-dir data/BatteryKG_v2 --model-dir models/battery_v2", file=sys.stderr)
            sys.exit(1)

    maps = torch.load(mapping_path, map_location="cpu", weights_only=False)

    node2localid  = maps["node2localid"]
    ntype2nodes   = maps["ntype2nodes"]
    node_names    = maps.get("node_names", {})
    meta_rels     = maps["meta_rels"]
    node_types    = maps["node_types"]
    edge_types    = maps["edge_types"]
    embedding_dim = maps["embedding_dim"]
    num_heads     = maps["num_heads"]
    num_layers    = maps["num_layers"]
    dropout       = maps.get("dropout", 0.2)

    input_mode = maps.get("input_mode", "csv_ids")
    model = HGTLinkPredictor(
        node_types    = node_types,
        edge_types    = edge_types,
        embedding_dim = embedding_dim,
        num_heads     = num_heads,
        num_layers    = num_layers,
        dropout       = dropout,
        feat_dims     = maps.get("feat_dims") if input_mode == "hetero_features" else None,
        num_nodes     = maps.get("num_nodes") if input_mode == "hetero_features" else None,
    )
    if input_mode != "hetero_features":
        model.init_node_embeddings({
            ntype: len(nodes) for ntype, nodes in ntype2nodes.items()
        })
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    model.to(device)

    return model, maps


def load_train_graph(
    data_dir:     str,
    node2localid: dict,
    meta_rels:    list,
    device:       torch.device,
    max_rows:     int | None = None,
) -> dict[tuple, torch.Tensor]:
    """Reconstruct training graph tensors for HGT message passing."""
    path = os.path.join(data_dir, "edges_train.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    buckets: dict[tuple, tuple[list, list]] = {mr: ([], []) for mr in meta_rels}
    count = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["head_type"], row["relation"], row["tail_type"])
            if key not in buckets:
                continue
            ht, tt = row["head_type"], row["tail_type"]
            h_local = node2localid.get(ht, {}).get(row["head_id"])
            t_local = node2localid.get(tt, {}).get(row["tail_id"])
            if h_local is not None and t_local is not None:
                buckets[key][0].append(h_local)
                buckets[key][1].append(t_local)
            count += 1
            if max_rows and count >= max_rows:
                break

    return {
        mr: torch.tensor([src, dst], dtype=torch.long, device=device)
        for mr, (src, dst) in buckets.items()
        if src
    }


# =============================================================================
# Utility display functions
# =============================================================================

def print_schema(maps: dict, data_dir: str | None = None):
    candidates = []
    if data_dir:
        candidates.append(os.path.join(data_dir, "schema.json"))
    candidates.append(os.path.join("data", "BatteryKG_v2", "schema.json"))
    candidates.append(os.path.join("data", "BatteryKG", "schema.json"))
    schema_path = next((p for p in candidates if os.path.exists(p)), None)
    print()
    print("=" * 70)
    print("  Graph Schema")
    print("=" * 70)

    if schema_path is not None:
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        print(f"\n  Node types ({len(schema['node_types'])}):")
        for nt in schema["node_types"]:
            n = len(maps["ntype2nodes"].get(nt, []))
            print(f"    {nt:<35}  {n:>8,} nodes")

        print(f"\n  Relation types ({len(schema['relation_types'])}):")
        for rt in schema["relation_types"]:
            print(f"    {rt}")

        print(f"\n  Top Meta-Relations (head → relation → tail):")
        print(f"  {'Head':<28} {'Relation':<28} {'Tail':<28} {'Count':>8}")
        print(f"  {'-'*28} {'-'*28} {'-'*28} {'-'*8}")
        for mr in schema["meta_relations"][:20]:
            print(
                f"  {mr['head_type']:<28} {mr['relation']:<28} "
                f"{mr['tail_type']:<28} {mr['count']:>8,}"
            )
        if len(schema["meta_relations"]) > 20:
            print(f"  … ({len(schema['meta_relations']) - 20} more)")
    else:
        # Reconstruct from mappings
        print(f"\n  Node types  : {maps['node_types']}")
        print(f"  Meta-rels   : {maps['meta_rels'][:10]} ...")
    print()


def list_nodes(node_type: str, maps: dict, limit: int):
    ntype2nodes = maps["ntype2nodes"]
    node_names  = maps.get("node_names", {})

    if node_type not in ntype2nodes:
        print(f"Error: node type '{node_type}' not found.")
        print(f"  Available types: {list(ntype2nodes.keys())}")
        return

    nodes = ntype2nodes[node_type]
    print(f"\n  Nodes of type '{node_type}' (showing {min(limit, len(nodes))} of {len(nodes):,}):")
    for nid in nodes[:limit]:
        name = node_names.get(nid, "")
        print(f"    {nid:<30}  {name}")
    if len(nodes) > limit:
        print(f"  … ({len(nodes) - limit} more)")


def list_relations(maps: dict, limit: int):
    meta_rels = maps["meta_rels"]
    print(f"\n  Available relations (from {len(meta_rels)} meta-relations):")
    print(f"  {'Head Type':<28} {'Relation':<25} {'Tail Type':<28}")
    print(f"  {'-'*28} {'-'*25} {'-'*28}")
    for ht, rel, tt in meta_rels[:limit]:
        print(f"  {ht:<28} {rel:<25} {tt:<28}")
    if len(meta_rels) > limit:
        print(f"  … ({len(meta_rels) - limit} more)")


def fuzzy_find_node(name_query: str, node_type: str, maps: dict) -> tuple[str | None, str]:
    """Find the best matching node ID for a name query."""
    ntype2nodes = maps["ntype2nodes"]
    node_names  = maps.get("node_names", {})

    if node_type not in ntype2nodes:
        return None, f"Type '{node_type}' not found."

    query = name_query.lower()
    nodes = ntype2nodes[node_type]

    # Exact match first
    for nid in nodes:
        if node_names.get(nid, "").lower() == query:
            return nid, f"Exact match: '{node_names[nid]}'"

    # Prefix match
    matches = [
        (nid, node_names.get(nid, ""))
        for nid in nodes
        if query in node_names.get(nid, "").lower()
    ]
    if matches:
        best_nid, best_name = matches[0]
        if len(matches) > 1:
            print(f"  Multiple matches for '{name_query}' in {node_type}:")
            for nid, name in matches[:5]:
                print(f"    {nid:<30}  {name}")
            print(f"  Using first: {best_nid} ({best_name})\n")
        return best_nid, f"Name match: '{best_name}'"

    return None, f"No node found matching '{name_query}' in type '{node_type}'."


# =============================================================================
# Default query picker
# =============================================================================

def pick_default_query(maps: dict) -> tuple[str, str, str] | None:
    """Pick an interesting default query from the schema."""
    meta_rels = maps["meta_rels"]
    ntype2nodes = maps["ntype2nodes"]

    # Prefer cross-type relations (more interesting)
    cross_type = [(ht, rel, tt) for ht, rel, tt in meta_rels if ht != tt]
    pool = cross_type or meta_rels

    for ht, rel, tt in pool:
        if ht in ntype2nodes and ntype2nodes[ht]:
            head_node = ntype2nodes[ht][0]
            return head_node, ht, rel

    return None


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = torch.device("cpu")

    # ── Utility modes (don't need model) ─────────────────────────────────────
    if args.schema or args.list_nodes or args.list_rels:
        # Load mappings only (no model needed)
        mapping_path = resolve_mappings_path(args.model_dir)
        if not os.path.exists(mapping_path):
            print(f"Error: {mapping_path} not found. Train first.")
            sys.exit(1)
        maps = torch.load(mapping_path, map_location="cpu", weights_only=False)

        if args.schema:
            print_schema(maps, data_dir=args.data_dir)
        if args.list_nodes:
            list_nodes(args.list_nodes, maps, limit=args.limit)
        if args.list_rels:
            list_relations(maps, limit=args.limit)
        return

    # ── Load model & vocab ────────────────────────────────────────────────────
    print("\nLoading model…")
    model, maps = load_model_and_vocab(device, model_dir=args.model_dir)

    node2localid = maps["node2localid"]
    ntype2nodes  = maps["ntype2nodes"]
    node_names   = maps.get("node_names", {})
    meta_rels    = maps["meta_rels"]

    # ── Reconstruct training graph for GNN message passing ───────────────────
    print("Reconstructing training graph for HGT message passing…")
    if maps.get("input_mode") == "hetero_features" and maps.get("edge_index_cpu"):
        edge_index_dict = edge_index_from_maps(maps, device)
        x_dict = make_x_dict_from_maps(maps, device)
    else:
        edge_index_dict = load_train_graph(
            args.data_dir, node2localid, meta_rels, device,
            max_rows=args.max_train_edges,
        )
        x_dict = {
            nt: torch.arange(len(nodes), dtype=torch.long, device=device)
            for nt, nodes in ntype2nodes.items()
        }

    # ── Encode all nodes ──────────────────────────────────────────────────────
    print("Running HGT encoder…")
    with torch.no_grad():
        h_dict = model.encode(x_dict, edge_index_dict)
    print("Ready.\n")

    # ── Resolve head node ─────────────────────────────────────────────────────
    head      = args.head
    head_type = args.head_type
    relation  = args.relation

    # Name-based search
    if args.head_name and head_type:
        found_id, msg = fuzzy_find_node(args.head_name, head_type, maps)
        if found_id is None:
            print(f"Error: {msg}")
            print(f"  Try: python predict.py --list-nodes {head_type}")
            return
        head = found_id
        print(f"  Found: {head}  ({msg})")

    # ── Default query if nothing given ────────────────────────────────────────
    if head is None or head_type is None or relation is None:
        print("No query specified — selecting a default example...\n")
        result = pick_default_query(maps)
        if result is None:
            print("Error: could not find a default query. Specify --head, --head-type, --relation.")
            return
        head, head_type, relation = result
        head_name = node_names.get(head, head)
        print(f"  Default query: ({head} [{head_type}], {relation}, ?)")
        print(f"  Head name    : {head_name}")
        print()
        print("  To run your own query:")
        print(f"    python predict.py --head <node_id> --head-type <type> --relation <relation>")
        print(f"    python predict.py --schema          # see all node types & relations")
        print(f"    python predict.py --list-nodes {head_type}")
        print()

    # ── Validate query ────────────────────────────────────────────────────────
    if head_type not in node2localid:
        print(f"Error: head type '{head_type}' not in vocabulary.")
        print(f"  Valid types: {list(node2localid.keys())}")
        return

    if head not in node2localid[head_type]:
        print(f"Error: head node '{head}' not in vocabulary for type '{head_type}'.")
        sample = list(node2localid[head_type].keys())[:5]
        print(f"  Sample nodes: {sample}")
        print(f"  Try: python predict.py --head-name '<name>' --head-type {head_type} --relation {relation}")
        return

    if relation not in model.relation2idx:
        print(f"Error: relation '{relation}' not in vocabulary.")
        print(f"  Valid relations: {sorted(model.relation2idx.keys())}")
        return

    # ── Find valid tail types ─────────────────────────────────────────────────
    candidate_tail_types = [
        tt for (ht, rel, tt) in meta_rels
        if ht == head_type and rel == relation
    ]
    if args.tail_type:
        if args.tail_type not in candidate_tail_types:
            print(f"Error: ({head_type}, {relation}, {args.tail_type}) is not a valid meta-relation.")
            print(f"  Valid tail types for this query: {candidate_tail_types}")
            return
        candidate_tail_types = [args.tail_type]

    if not candidate_tail_types:
        print(f"Error: No meta-relation found for ({head_type}, {relation}, ?).")
        print(f"  Try: python predict.py --list-rels")
        return

    # ── Score candidates and show results ─────────────────────────────────────
    head_name = node_names.get(head, head)
    hi = node2localid[head_type][head]

    print("=" * 70)
    print(f"  Query: ({head} [{head_type}], {relation}, ?)")
    print(f"  Head : {head_name}")
    print("=" * 70)

    with torch.no_grad():
        r_idx = torch.tensor(model.relation2idx[relation], device=device)
        r_emb = model.rel_embeddings(r_idx)     # [D]
        h_src = h_dict[head_type][hi]           # [D]

        for tail_type in candidate_tail_types:
            h_all  = h_dict[tail_type]          # [N_tt, D]
            scores = (h_all * h_src.unsqueeze(0) * r_emb.unsqueeze(0)).sum(-1)

            k = min(args.topk, scores.size(0))
            top_scores, top_indices = torch.topk(scores, k=k)

            print(f"\n  Tail type: '{tail_type}'  ({k} of {scores.size(0):,} candidates)")
            print(f"  {'Rank':<6}  {'Score':>8}  {'Node ID':<30}  {'Name'}")
            print(f"  {'-'*6}  {'-'*8}  {'-'*30}  {'-'*35}")

            for rank, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
                node_id   = ntype2nodes[tail_type][idx.item()]
                node_name = node_names.get(node_id, "—")
                print(
                    f"  {rank:<6}  {score.item():>8.4f}  {node_id:<30}  {node_name}"
                )

    print()


if __name__ == "__main__":
    main()
