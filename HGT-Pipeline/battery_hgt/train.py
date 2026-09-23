#!/usr/bin/env python3
"""
train.py — HGT Training on Battery Knowledge Graph
====================================================

Trains a Heterogeneous Graph Transformer (HGT) for link prediction
from `output/hetero_data.pt` (typed `.x` + `edge_index`). CSV is optional.

Workflow
--------
  python train.py --model-dir models/battery_v2 --output-dir outputs/battery_v2
  python train.py --from-csv --data-dir data/BatteryKG_v2   # old path

Outputs
-------
  models/.../hgt_checkpoint.pt  — trained model weights
  models/.../hgt_mappings.pt    — vocabulary + (for HeteroData) feat_x / edge_index
"""

from __future__ import annotations

import os
import sys
import json
import csv
import time
import random
import argparse
import torch
import torch.nn.functional as F
from collections import defaultdict
from models.hgt import HGTLinkPredictor
from hetero_graph import (
    default_hetero_dir,
    load_hetero_bundle,
    make_x_dict_from_bundle,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train HGT on battery knowledge graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",       default=os.path.join("data", "BatteryKG_v2"))
    p.add_argument(
        "--hetero-dir", default=default_hetero_dir(),
        help="Folder with hetero_data.pt + node_mapping.json (default: ../../output).",
    )
    p.add_argument(
        "--from-csv", action="store_true",
        help="Use BatteryKG CSVs instead of HeteroData (legacy).",
    )
    p.add_argument(
        "--include-sites", action="store_true",
        help="Include AtomicSite / bond edges (large; slow).",
    )
    p.add_argument(
        "--max-property-edges", type=int, default=20000,
        help="Cap Material→Property edges when loading HeteroData.",
    )
    p.add_argument("--embedding-dim",  type=int,   default=64)
    p.add_argument("--num-heads",      type=int,   default=4)
    p.add_argument("--num-layers",     type=int,   default=2)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--lr",             type=float, default=0.001)
    p.add_argument("--batch-size",     type=int,   default=2048)
    p.add_argument("--dropout",        type=float, default=0.2)
    p.add_argument(
        "--device", default="",
        help="cpu | mps | cuda. Default: cuda if available, else mps, else cpu.",
    )
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument(
        "--max-train-edges", type=int, default=None,
        help="Cap training edges (useful for fast CPU testing).",
    )
    p.add_argument(
        "--eval-every", type=int, default=10,
        help="Run evaluation every N epochs (default: 10). Use 1 for every epoch.",
    )
    p.add_argument(
        "--eval-sample", type=int, default=500,
        help="Number of val triples sampled per evaluation.",
    )
    p.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation entirely (max training speed).",
    )
    p.add_argument(
        "--neg-samples", type=int, default=1,
        help="Negative samples per positive edge (default: 1).",
    )
    p.add_argument(
        "--output-dir", default="outputs",
        help="Directory for training_log.csv and training_summary.json.",
    )
    p.add_argument(
        "--model-dir", default="models",
        help="Directory for hgt_checkpoint.pt and hgt_mappings.pt.",
    )
    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def load_edges(path: str, max_rows: int | None = None) -> list[dict]:
    """Load a typed-edge TSV (BatteryKG schema)."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def load_node_names(path: str) -> dict[str, str]:
    """Load node_id → node_name mapping from nodes.csv."""
    names: dict[str, str] = {}
    if not os.path.exists(path):
        return names
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            names[row["node_id"]] = row.get("node_name", row["node_id"])
    return names


def build_vocab(
    train_edges: list[dict],
    val_edges:   list[dict],
) -> tuple[dict, dict, dict, list, list]:
    """
    Build closed-world vocabulary from training edges.

    Returns
    -------
    node2localid  : {node_type: {node_id: local_int}}
    ntype2nodes   : {node_type: [node_id, ...]}
    rel2id        : {relation_str: int}
    meta_rels     : sorted list of (head_type, relation, tail_type)
    val_filtered  : val edges inside the training vocabulary
    """
    node_sets:  dict[str, set]   = defaultdict(set)
    rel_set:    set[str]         = set()
    meta_set:   set[tuple]       = set()

    for e in train_edges:
        node_sets[e["head_type"]].add(e["head_id"])
        node_sets[e["tail_type"]].add(e["tail_id"])
        rel_set.add(e["relation"])
        meta_set.add((e["head_type"], e["relation"], e["tail_type"]))

    ntype2nodes  = {nt: sorted(ns) for nt, ns in node_sets.items()}
    node2localid = {
        nt: {nid: i for i, nid in enumerate(nodes)}
        for nt, nodes in ntype2nodes.items()
    }
    rel2id    = {r: i for i, r in enumerate(sorted(rel_set))}
    meta_rels = sorted(meta_set)

    val_filtered = [
        e for e in val_edges
        if (e["head_type"] in node2localid
            and e["head_id"]  in node2localid[e["head_type"]]
            and e["tail_type"] in node2localid
            and e["tail_id"]  in node2localid[e["tail_type"]]
            and e["relation"] in rel2id)
    ]

    print("\n  Vocabulary built from training edges:")
    for nt in sorted(ntype2nodes, key=lambda x: -len(ntype2nodes[x])):
        print(f"    {nt:<35}  {len(ntype2nodes[nt]):>8,} nodes")
    print(f"    {'Relations':<35}  {len(rel2id):>8,}")
    print(f"    {'Meta-relations':<35}  {len(meta_rels):>8,}")
    print(f"    {'Val edges (in-vocab)':<35}  {len(val_filtered):>8,}")

    return node2localid, ntype2nodes, rel2id, meta_rels, val_filtered


# =============================================================================
# Graph tensor builders
# =============================================================================

def build_edge_index_dict(
    edges:        list[dict],
    meta_rels:    list[tuple],
    node2localid: dict[str, dict[str, int]],
    device:       torch.device,
) -> dict[tuple, torch.Tensor]:
    """Build {(src_type, rel, dst_type): LongTensor[2, E]} for HGTConv."""
    buckets: dict[tuple, tuple[list, list]] = {mr: ([], []) for mr in meta_rels}

    for e in edges:
        key = (e["head_type"], e["relation"], e["tail_type"])
        if key not in buckets:
            continue
        ht, tt = e["head_type"], e["tail_type"]
        h_local = node2localid.get(ht, {}).get(e["head_id"])
        t_local = node2localid.get(tt, {}).get(e["tail_id"])
        if h_local is not None and t_local is not None:
            buckets[key][0].append(h_local)
            buckets[key][1].append(t_local)

    result = {}
    for mr, (src, dst) in buckets.items():
        if src:
            result[mr] = torch.tensor([src, dst], dtype=torch.long, device=device)
    return result


def make_x_dict(
    ntype2nodes: dict[str, list],
    device:      torch.device,
) -> dict[str, torch.Tensor]:
    """Build {node_type: arange(N)} index tensors for embedding lookup."""
    return {
        nt: torch.arange(len(nodes), dtype=torch.long, device=device)
        for nt, nodes in ntype2nodes.items()
    }


# =============================================================================
# Evaluation — bidirectional MRR & Hits@K
# =============================================================================

@torch.no_grad()
def evaluate(
    model:           HGTLinkPredictor,
    val_edges:       list[dict],
    node2localid:    dict[str, dict[str, int]],
    ntype2nodes:     dict[str, list],
    edge_index_dict: dict[tuple, torch.Tensor],
    device:          torch.device,
    sample_size:     int = 500,
    x_dict:          dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """
    Evaluate link prediction with BOTH head and tail corruption
    (same protocol as most KGE benchmarks — more rigorous than tail-only).

    Returns dict with mrr, hits@1, hits@3, hits@10, bin_acc, pair_acc.
    """
    model.eval()
    if not val_edges:
        return {"mrr": 0.0, "h1": 0.0, "h3": 0.0, "h10": 0.0, "bin_acc": 0.0, "pair_acc": 0.0}

    sample  = random.sample(val_edges, min(sample_size, len(val_edges)))
    if x_dict is None:
        x_dict = make_x_dict(ntype2nodes, device)
    h_dict  = model.encode(x_dict, edge_index_dict)

    mrr_sum = hits1 = hits3 = hits10 = 0.0
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    n_corruptions = 0

    for e in sample:
        ht  = e["head_type"]
        tt  = e["tail_type"]
        rel = e["relation"]
        hi  = node2localid[ht][e["head_id"]]
        ti  = node2localid[tt][e["tail_id"]]

        r_idx = torch.tensor(model.relation2idx[rel], device=device)
        r_emb = model.rel_embeddings(r_idx)   # [D]

        # ── Tail corruption (h, r, ?)  ────────────────────────────────────────
        h_src  = h_dict[ht][hi]               # [D]
        h_all  = h_dict[tt]                   # [N_tt, D]
        scores_tail = (h_all * h_src.unsqueeze(0) * r_emb.unsqueeze(0)).sum(-1)

        true_t  = scores_tail[ti]
        rank_t  = (scores_tail > true_t).sum().item() + 1
        mrr_sum += 1.0 / rank_t
        hits1   += 1.0 if rank_t <= 1  else 0.0
        hits3   += 1.0 if rank_t <= 3  else 0.0
        hits10  += 1.0 if rank_t <= 10 else 0.0
        n_corruptions += 1

        # ── Head corruption (?, r, t)  ────────────────────────────────────────
        h_dst   = h_dict[tt][ti]              # [D]
        h_all_h = h_dict[ht]                  # [N_ht, D]
        scores_head = (h_all_h * h_dst.unsqueeze(0) * r_emb.unsqueeze(0)).sum(-1)

        true_h  = scores_head[hi]
        rank_h  = (scores_head > true_h).sum().item() + 1
        mrr_sum += 1.0 / rank_h
        hits1   += 1.0 if rank_h <= 1  else 0.0
        hits3   += 1.0 if rank_h <= 3  else 0.0
        hits10  += 1.0 if rank_h <= 10 else 0.0
        n_corruptions += 1

        # ── Pairwise accuracy scores ──────────────────────────────────────────
        pos_scores.append(true_t.item())
        neg_j = random.randrange(len(ntype2nodes[tt]))
        neg_scores.append(scores_tail[neg_j].item())

    n = n_corruptions
    pos_t = torch.tensor(pos_scores)
    neg_t = torch.tensor(neg_scores)
    bin_acc  = ((torch.sigmoid(pos_t) >= 0.5).sum() +
                (torch.sigmoid(neg_t)  < 0.5).sum()).item() / (2 * len(pos_scores))
    pair_acc = (pos_t > neg_t).float().mean().item()

    return {
        "mrr":      mrr_sum / n,
        "h1":       hits1   / n,
        "h3":       hits3   / n,
        "h10":      hits10  / n,
        "bin_acc":  bin_acc,
        "pair_acc": pair_acc,
    }


# =============================================================================
# Training loop
# =============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── 1. Load graph ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  HGT Training on Battery Knowledge Graph")
    print("=" * 65)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")
    bundle = None
    input_mode = "csv_ids"

    if not args.from_csv:
        print("\n  Source: HeteroData (.x + edge_index)")
        try:
            bundle = load_hetero_bundle(
                args.hetero_dir,
                seed=args.seed,
                max_property_edges=args.max_property_edges,
                include_sites=args.include_sites,
                max_train_edges=args.max_train_edges,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print("Pass --from-csv --data-dir data/BatteryKG_v2 to use CSVs.", file=sys.stderr)
            sys.exit(1)
        input_mode = "hetero_features"
        train_edges = bundle.train_edges
        val_edges = bundle.val_edges
        node_names = bundle.node_names
        node2localid = bundle.node2localid
        ntype2nodes = bundle.ntype2nodes
        rel2id = bundle.rel2id
        meta_rels = bundle.meta_rels
        val_filtered = [
            e for e in val_edges
            if (e["head_type"] in node2localid
                and e["head_id"] in node2localid[e["head_type"]]
                and e["tail_type"] in node2localid
                and e["tail_id"] in node2localid[e["tail_type"]]
                and e["relation"] in rel2id)
        ]
        print(f"  Train edges : {len(train_edges):,}")
        print(f"  Val edges   : {len(val_filtered):,}")
    else:
        schema_path = os.path.join(args.data_dir, "schema.json")
        train_path  = os.path.join(args.data_dir, "edges_train.csv")
        val_path    = os.path.join(args.data_dir, "edges_val.csv")
        nodes_path  = os.path.join(args.data_dir, "nodes.csv")

        for path in [schema_path, train_path]:
            if not os.path.exists(path):
                print(f"Error: {path} not found.", file=sys.stderr)
                print("Use default HeteroData path, or generate CSVs with convert_hetero_to_hgt.py", file=sys.stderr)
                sys.exit(1)

        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        print(f"  Source: CSV  |  {len(schema['node_types'])} node types  |  "
              f"{len(schema['relation_types'])} relations")
        print(f"\n  Loading training edges (limit={args.max_train_edges or 'all'})...")
        train_edges = load_edges(train_path, max_rows=args.max_train_edges)
        val_edges   = load_edges(val_path)
        node_names  = load_node_names(nodes_path)
        print(f"  Train edges : {len(train_edges):,}")
        print(f"  Val edges   : {len(val_edges):,}")
        node2localid, ntype2nodes, rel2id, meta_rels, val_filtered = build_vocab(
            train_edges, val_edges
        )

    # ── 2. Build graph tensors ────────────────────────────────────────────────
    print("\n  Building heterogeneous graph tensors...")
    edge_index_dict = build_edge_index_dict(train_edges, meta_rels, node2localid, device)
    if bundle is not None:
        x_dict = make_x_dict_from_bundle(bundle, device)
    else:
        x_dict = make_x_dict(ntype2nodes, device)

    active_meta_rels = [mr for mr in meta_rels if mr in edge_index_dict]
    total_train_edges = sum(
        edge_index_dict[mr].shape[1] for mr in active_meta_rels
    )
    print(f"  Active meta-relations : {len(active_meta_rels)}")
    print(f"  Graph edges (tensor)  : {total_train_edges:,}")

    # ── 3. Initialise model ───────────────────────────────────────────────────
    node_types_list = list(ntype2nodes.keys())
    edge_types_list = active_meta_rels

    print(f"\n  HGT config:")
    print(f"    input_mode    = {input_mode}")
    print(f"    embedding_dim = {args.embedding_dim}")
    print(f"    num_heads     = {args.num_heads}")
    print(f"    num_layers    = {args.num_layers}")
    print(f"    dropout       = {args.dropout}")
    print(f"    node_types    = {node_types_list}")

    feat_dims = bundle.feat_dims if bundle is not None else None
    num_nodes = bundle.num_nodes if bundle is not None else {
        nt: len(nodes) for nt, nodes in ntype2nodes.items()
    }

    model = HGTLinkPredictor(
        node_types    = node_types_list,
        edge_types    = edge_types_list,
        embedding_dim = args.embedding_dim,
        num_heads     = args.num_heads,
        num_layers    = args.num_layers,
        dropout       = args.dropout,
        feat_dims     = feat_dims,
        num_nodes     = num_nodes if bundle is not None else None,
    ).to(device)

    if bundle is None:
        model.init_node_embeddings({nt: len(nodes) for nt, nodes in ntype2nodes.items()})

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # ── 6. Save vocabulary ────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    mapping_path = os.path.join(args.model_dir, "hgt_mappings.pt")
    maps_payload = {
        "node2localid":  node2localid,
        "ntype2nodes":   ntype2nodes,
        "node_names":    node_names,
        "rel2id":        rel2id,
        "meta_rels":     active_meta_rels,
        "node_types":    node_types_list,
        "edge_types":    edge_types_list,
        "embedding_dim": args.embedding_dim,
        "num_heads":     args.num_heads,
        "num_layers":    args.num_layers,
        "dropout":       args.dropout,
        "data_dir":      args.data_dir,
        "hetero_dir":    args.hetero_dir,
        "max_train_edges": args.max_train_edges,
        "input_mode":    input_mode,
        "feat_dims":     feat_dims or {},
        "num_nodes":     num_nodes,
    }
    if bundle is not None:
        maps_payload["feat_x"] = {nt: t.cpu() for nt, t in bundle.feat_x.items()}
        maps_payload["edge_index_cpu"] = {
            mr: ei.detach().cpu() for mr, ei in edge_index_dict.items()
        }
        maps_payload["test_edges"] = bundle.test_edges
        maps_payload["val_edges"] = bundle.val_edges
        maps_payload["feature_types"] = bundle.feature_types
        maps_payload["id_types"] = bundle.id_types
    torch.save(maps_payload, mapping_path)
    print(f"  Vocabulary saved → {mapping_path}")

    # ── 7. Group training edges by meta-relation ──────────────────────────────
    meta_buckets: dict[tuple, list[dict]] = defaultdict(list)
    for e in train_edges:
        key = (e["head_type"], e["relation"], e["tail_type"])
        if key in edge_index_dict:
            meta_buckets[key].append(e)
    active_buckets = [(k, v) for k, v in meta_buckets.items() if v]

    batches_per_epoch = sum(
        (len(edges) + args.batch_size - 1) // args.batch_size
        for _, edges in active_buckets
    )
    print(f"  Batches / epoch       : ~{batches_per_epoch}")
    print(f"  Encode calls / epoch  : 1  (encode-once strategy)")

    # ── 8. Training loop ──────────────────────────────────────────────────────
    print(f"\n  Training for {args.epochs} epochs…")
    print("  " + "─" * 90)
    header = (
        f"  {'Epoch':>6}  {'Time':>6}  {'Loss':>8}  "
        f"{'MRR':>7}  {'H@1':>6}  {'H@3':>6}  {'H@10':>6}  "
        f"{'BinAcc':>7}  {'PairAcc':>8}  {'LR':>8}"
    )
    print(header)
    print("  " + "─" * 90)

    best_mrr   = 0.0
    best_epoch = 0
    ckpt_path  = os.path.join(args.model_dir, "hgt_checkpoint.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    log_csv_path = os.path.join(args.output_dir, "training_log.csv")
    epoch_rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        # ── Training: encode ONCE per epoch, accumulate losses, single backward
        # This is ~100× faster on CPU than encoding inside every batch because
        # HGTConv is expensive; the decoder (DistMult dot-product) is cheap.
        optimizer.zero_grad()
        h_dict = model.encode(x_dict, edge_index_dict)   # <-- ONE encode per epoch

        all_losses: list[torch.Tensor] = []

        for (ht, rel, tt), edge_group in active_buckets:
            n_tail = len(ntype2nodes[tt])
            n_head = len(ntype2nodes[ht])
            random.shuffle(edge_group)

            for i in range(0, len(edge_group), args.batch_size):
                batch = edge_group[i : i + args.batch_size]

                b_src = torch.tensor(
                    [node2localid[ht][e["head_id"]] for e in batch],
                    dtype=torch.long, device=device,
                )
                b_dst = torch.tensor(
                    [node2localid[tt][e["tail_id"]] for e in batch],
                    dtype=torch.long, device=device,
                )

                neg_tail  = torch.randint(0, n_tail, (len(batch) * args.neg_samples,), device=device)
                neg_head  = torch.randint(0, n_head, (len(batch) * args.neg_samples,), device=device)
                b_src_rep = b_src.repeat(args.neg_samples)
                b_dst_rep = b_dst.repeat(args.neg_samples)

                pos_score      = model.decode(h_dict, ht, rel, tt, b_src,     b_dst)
                neg_score_tail = model.decode(h_dict, ht, rel, tt, b_src_rep, neg_tail)
                neg_score_head = model.decode(h_dict, ht, rel, tt, neg_head,  b_dst_rep)

                loss = (
                    - F.logsigmoid(pos_score).mean()
                    - F.logsigmoid(-neg_score_tail).mean()
                    - F.logsigmoid(-neg_score_head).mean()
                ) / 3.0
                all_losses.append(loss)

        # Single backward pass over all accumulated batch losses
        if all_losses:
            total_loss_tensor = torch.stack(all_losses).mean()
            total_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            avg_loss = total_loss_tensor.item()
        else:
            avg_loss = 0.0

        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "time_sec": round(elapsed, 2),
            "loss": round(avg_loss, 6),
            "lr": current_lr,
        }

        # ── Evaluation ────────────────────────────────────────────────────────
        if (not args.no_eval) and (epoch % args.eval_every == 0 or epoch == 1):
            metrics = evaluate(
                model, val_filtered, node2localid, ntype2nodes,
                edge_index_dict, device, sample_size=args.eval_sample,
                x_dict=x_dict,
            )
            mrr = metrics["mrr"]
            row.update({k: round(v, 6) for k, v in metrics.items()})

            print(
                f"  {epoch:>6}  {elapsed:>5.1f}s  {avg_loss:>8.4f}  "
                f"{mrr:>7.4f}  {metrics['h1']:>6.4f}  {metrics['h3']:>6.4f}  "
                f"{metrics['h10']:>6.4f}  {metrics['bin_acc']:>7.4f}  "
                f"{metrics['pair_acc']:>8.4f}  {current_lr:.2e}"
            )

            # Save best checkpoint
            if mrr > best_mrr:
                best_mrr = mrr
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
        else:
            print(
                f"  {epoch:>6}  {elapsed:>5.1f}s  {avg_loss:>8.4f}  "
                f"{'—':>7}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>7}  {'—':>8}  {current_lr:.2e}"
            )
            # Still save periodically
            if epoch % 20 == 0:
                torch.save(model.state_dict(), ckpt_path)

        epoch_rows.append(row)

    # ── Final checkpoint (if no eval was done) ────────────────────────────────
    if args.no_eval or best_mrr == 0.0:
        torch.save(model.state_dict(), ckpt_path)

    if epoch_rows:
        fields = list(epoch_rows[0].keys())
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(epoch_rows)

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "epochs": args.epochs,
            "input_mode": input_mode,
            "train_edges": len(train_edges),
            "val_edges_in_vocab": len(val_filtered),
            "best_val_mrr": round(best_mrr, 6),
            "best_epoch": best_epoch,
            "checkpoint": ckpt_path,
            "mappings": mapping_path,
            "training_log_csv": log_csv_path,
        }, f, indent=2)

    print("  " + "─" * 90)
    print(f"\n  ✓ Training complete!")
    print(f"    Best MRR        : {best_mrr:.4f}  (epoch {best_epoch})")
    print(f"    Checkpoint      : {ckpt_path}")
    print(f"    Vocabulary      : {mapping_path}")
    print(f"    Training log    : {log_csv_path}")
    print(f"    Summary         : {summary_path}")
    print()
    print("  Next step — test on held-out edges:")
    print("    python test.py --max-train-edges", args.max_train_edges or "all", "--output-dir outputs")
    print("  Or query interactively:")
    print("    python predict.py --head-name \"Holt-Oram\" --head-type disease --relation is_a")


if __name__ == "__main__":
    main()
