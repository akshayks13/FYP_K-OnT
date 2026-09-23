#!/usr/bin/env python3
"""
test.py — Evaluate trained HGT on held-out test edges and write results to files.

Usage
-----
  python test.py
  python test.py --output-dir outputs --sample 200
  python test.py --predictions 20   # write example (h,r,?) predictions to file
"""

from __future__ import annotations

import os
import sys
import json
import csv
import random
import argparse
import torch

from models.hgt import HGTLinkPredictor
from train import (
    load_edges,
    load_node_names,
    build_vocab,
    build_edge_index_dict,
    make_x_dict,
    evaluate,
)
from hetero_graph import edge_index_from_maps, make_x_dict_from_maps


def parse_args():
    p = argparse.ArgumentParser(description="Test HGT on edges_test.csv and save metrics.")
    p.add_argument("--data-dir", default=os.path.join("data", "BatteryKG_v2"))
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--model-dir", default=os.path.join("models", "battery_v2"),
                   help="Directory containing hgt_checkpoint.pt and hgt_mappings.pt.")
    p.add_argument("--max-train-edges", type=int, default=None,
                   help="Must match training if a subset was used.")
    p.add_argument("--sample", type=int, default=300,
                   help="Max test triples used for MRR / Hits@K.")
    p.add_argument("--predictions", type=int, default=10,
                   help="Number of test edges to run as (h,r,?) prediction examples.")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def filter_test_edges(test_edges, node2localid, rel2id):
    return [
        e for e in test_edges
        if (e["head_type"] in node2localid
            and e["head_id"] in node2localid[e["head_type"]]
            and e["tail_type"] in node2localid
            and e["tail_id"] in node2localid[e["tail_type"]]
            and e["relation"] in rel2id)
    ]


def write_prediction_examples(
    model,
    examples: list[dict],
    node2localid,
    ntype2nodes,
    node_names: dict,
    edge_index_dict,
    device,
    topk: int,
    path: str,
    x_dict=None,
):
    model.eval()
    if x_dict is None:
        x_dict = make_x_dict(ntype2nodes, device)
    with torch.no_grad():
        h_dict = model.encode(x_dict, edge_index_dict)

    lines = []
    lines.append("Sample link predictions on TEST edges: (head, relation, ?)")
    lines.append("=" * 72)

    for i, e in enumerate(examples, 1):
        ht, tt, rel = e["head_type"], e["tail_type"], e["relation"]
        hi = node2localid[ht][e["head_id"]]
        ti = node2localid[tt][e["tail_id"]]
        h_name = node_names.get(e["head_id"], e["head_id"])
        t_name = node_names.get(e["tail_id"], e["tail_id"])

        r_idx = torch.tensor(model.relation2idx[rel], device=device)
        r_emb = model.rel_embeddings(r_idx)
        h_src = h_dict[ht][hi]
        h_all = h_dict[tt]
        scores = (h_all * h_src.unsqueeze(0) * r_emb.unsqueeze(0)).sum(-1)
        k = min(topk, scores.size(0))
        top_scores, top_idx = torch.topk(scores, k=k)
        true_rank = (scores > scores[ti]).sum().item() + 1

        lines.append("")
        lines.append(f"Example {i}")
        lines.append(f"  Query   : ({e['head_id']}, {rel}, ?)  [{ht} -> {tt}]")
        lines.append(f"  Head    : {h_name}")
        lines.append(f"  True tail: {e['tail_id']}  ({t_name})  |  rank={true_rank}")
        lines.append(f"  Top-{k} predictions:")
        for rank, (sc, idx) in enumerate(zip(top_scores, top_idx), 1):
            nid = ntype2nodes[tt][idx.item()]
            mark = " <-- TRUE" if idx.item() == ti else ""
            lines.append(
                f"    {rank:>2}. score={sc.item():.4f}  {nid}  "
                f"{node_names.get(nid, '')}{mark}"
            )

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    mapping_path = os.path.join(args.model_dir, "hgt_mappings.pt")
    ckpt_path = os.path.join(args.model_dir, "hgt_checkpoint.pt")
    for p in [mapping_path, ckpt_path]:
        if not os.path.exists(p):
            print(f"Error: missing {p}. Run train.py first.", file=sys.stderr)
            sys.exit(1)

    maps = torch.load(mapping_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_mode = maps.get("input_mode", "csv_ids")

    if input_mode == "hetero_features":
        node2localid = maps["node2localid"]
        ntype2nodes = maps["ntype2nodes"]
        node_names = maps.get("node_names", {})
        rel2id = maps["rel2id"]
        test_edges = maps.get("test_edges") or []
        test_filtered = filter_test_edges(test_edges, node2localid, rel2id)
        x_dict = make_x_dict_from_maps(maps, device)
        edge_index_dict = edge_index_from_maps(maps, device)
        model = HGTLinkPredictor(
            node_types=maps["node_types"],
            edge_types=maps["edge_types"],
            embedding_dim=maps["embedding_dim"],
            num_heads=maps["num_heads"],
            num_layers=maps["num_layers"],
            dropout=maps.get("dropout", 0.2),
            feat_dims=maps.get("feat_dims"),
            num_nodes=maps.get("num_nodes"),
        )
    else:
        train_path = os.path.join(args.data_dir, "edges_train.csv")
        test_path = os.path.join(args.data_dir, "edges_test.csv")
        nodes_path = os.path.join(args.data_dir, "nodes.csv")
        for p in [train_path, test_path]:
            if not os.path.exists(p):
                print(f"Error: missing {p}.", file=sys.stderr)
                sys.exit(1)
        train_edges = load_edges(train_path, max_rows=args.max_train_edges)
        test_edges = load_edges(test_path)
        node_names = load_node_names(nodes_path)
        node2localid, ntype2nodes, rel2id, meta_rels, _ = build_vocab(train_edges, [])
        test_filtered = filter_test_edges(test_edges, node2localid, rel2id)
        x_dict = make_x_dict(ntype2nodes, device)
        edge_index_dict = build_edge_index_dict(train_edges, meta_rels, node2localid, device)
        model = HGTLinkPredictor(
            node_types=maps["node_types"],
            edge_types=maps["edge_types"],
            embedding_dim=maps["embedding_dim"],
            num_heads=maps["num_heads"],
            num_layers=maps["num_layers"],
            dropout=maps.get("dropout", 0.2),
        )
        model.init_node_embeddings({nt: len(nodes) for nt, nodes in ntype2nodes.items()})

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)

    print(f"Test edges (in vocab): {len(test_filtered):,}  (sample={args.sample})")
    metrics = evaluate(
        model, test_filtered, node2localid, ntype2nodes,
        edge_index_dict, device, sample_size=args.sample,
        x_dict=x_dict,
    )

    summary = {
        "split": "test",
        "test_edges_total": len(test_edges),
        "test_edges_in_vocab": len(test_filtered),
        "eval_sample": min(args.sample, len(test_filtered)),
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "checkpoint": ckpt_path,
    }

    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(args.output_dir, "test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HGT Ontology Link Prediction — TEST SET\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test edges (file)     : {len(test_edges):,}\n")
        f.write(f"Test edges (in vocab) : {len(test_filtered):,}\n")
        f.write(f"Eval sample size      : {summary['eval_sample']}\n\n")
        f.write("Metrics (head + tail corruption):\n")
        f.write(f"  MRR       : {metrics['mrr']:.4f}\n")
        f.write(f"  Hits@1    : {metrics['h1']:.4f}\n")
        f.write(f"  Hits@3    : {metrics['h3']:.4f}\n")
        f.write(f"  Hits@10   : {metrics['h10']:.4f}\n")
        f.write(f"  BinAcc    : {metrics['bin_acc']:.4f}\n")
        f.write(f"  PairAcc   : {metrics['pair_acc']:.4f}\n")

    print(f"\nWrote {metrics_path}")
    print(f"Wrote {report_path}")

    if args.predictions > 0 and test_filtered:
        n = min(args.predictions, len(test_filtered))
        examples = random.sample(test_filtered, n)
        pred_path = os.path.join(args.output_dir, "test_predictions.txt")
        write_prediction_examples(
            model, examples, node2localid, ntype2nodes, node_names,
            edge_index_dict, device, args.topk, pred_path,
            x_dict=x_dict,
        )
        print(f"Wrote {pred_path}")

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
