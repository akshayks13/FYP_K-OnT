"""Full-scale ABox training driver: extract -> verbalize -> merge (real TBox)
-> train -> Phase 5b evals (type check, hasStructure ranking, crowding).

Generalizes smoke_twenty_materials.py: material count and epochs are CLI
args instead of hardcoded at 20/1, and the real TBox (battgpt_tbox_ont/, built
by extract_tbox.py + ont/data/prepare.py) is used instead of a hand-written
class list. Run extract_tbox.py once before this if battgpt_tbox_ont/ is
missing.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from geoopt.manifolds import PoincareBall
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

ONT_ROOT = Path(__file__).resolve().parents[2]
if str(ONT_ROOT) not in sys.path:
    sys.path.insert(0, str(ONT_ROOT))

from ont.data.extract_abox import extract
from ont.data.load import load_local_dataset
from ont.data.merge_abox import ELECTRODE_ROLE_CLASSES, ROLE, ROLE_ELECTRODE, TRUE_TYPE, load_tbox, merge, write_ont_data
from ont.data.verbalize_abox import run as verbalize_run
from ont.hit import HierarchyTransformer
from ont.losses.hit_loss import HierarchyTransformerLoss
from ont.losses.logical_loss import LogicalConstraintLoss
from ont.model import OntologyTransformer
from ont.trainer import HierarchyTransformerTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_abox")
logging.getLogger("httpx").setLevel(logging.WARNING)

TTL = ONT_ROOT.parent / "output" / "battery_kg.ttl"
BASE_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_ball(model: OntologyTransformer, sentences: list[str], batch_size: int = 32):
    raw = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)
    x = raw.detach().float().cpu() if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32)
    c = model.manifold.c
    c = float(c.detach().cpu()) if torch.is_tensor(c) else float(c)
    radius = 1.0 / float(np.sqrt(c))
    norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return x * torch.clamp((radius - 1e-5) / norms, max=1.0), PoincareBall(c=c)


def type_report(model, verbalizations: dict[str, str], ids: list[str], concept_names: list[str]) -> dict:
    class_x, man = to_ball(model, concept_names)
    class_idx = {name: i for i, name in enumerate(concept_names)}
    crystal_name = "crystal" if "crystal" in class_idx else None
    rows = []
    n_ok = n_vs_crystal = n_parent = 0
    for mid in ids:
        v, _ = to_ball(model, [verbalizations[mid]])
        dist = {name: float(man.dist(v, class_x[i : i + 1]).item()) for name, i in class_idx.items()}
        nearest = min(dist, key=dist.get)
        dist0_v = float(man.dist0(v).item())
        dist0_s = float(man.dist0(class_x[class_idx[TRUE_TYPE] : class_idx[TRUE_TYPE] + 1]).item())
        ok = nearest == TRUE_TYPE
        vs_c = (dist[TRUE_TYPE] < dist[crystal_name]) if crystal_name else None
        parent = dist0_s < dist0_v
        n_ok += int(ok)
        n_vs_crystal += int(bool(vs_c))
        n_parent += int(parent)
        rows.append(
            {
                "mp_id": mid,
                "nearest_type": nearest,
                "dist_true_type": dist[TRUE_TYPE],
                "dist_crystal": dist.get(crystal_name),
                "sits_under_true_type": ok,
                "closer_than_crystal": vs_c,
                "parent_nearer_origin": parent,
            }
        )
    n = max(len(ids), 1)
    return {
        "n": len(ids),
        "type_accuracy": n_ok / n,
        "closer_than_crystal": n_vs_crystal / n,
        "parent_nearer_origin": n_parent / n,
        "rows": rows,
    }


def crowding(model, verbalizations: dict[str, str], ids: list[str]) -> dict:
    if len(ids) < 2:
        return {"n": len(ids), "mean_pairwise_dist": None}
    sents = [verbalizations[m] for m in ids]
    x, man = to_ball(model, sents)
    dists = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dists.append(float(man.dist(x[i : i + 1], x[j : j + 1]).item()))
    arr = np.array(dists)
    return {
        "n": len(ids),
        "n_pairs": len(dists),
        "mean_pairwise_dist": float(arr.mean()),
        "min_pairwise_dist": float(arr.min()),
        "p10_pairwise_dist": float(np.percentile(arr, 10)),
        "p50_pairwise_dist": float(np.percentile(arr, 50)),
        "max_pairwise_dist": float(arr.max()),
    }


def electrode_role_report(
    model, verbalizations: dict[str, str], battery_role: dict[str, str], ids: list[str]
) -> dict:
    """Accuracy of the belongsToElectrode exist relation: is a material's
    true role (Positive/Negative/Electrolyte/Separator) the nearest of the
    4 possible fillers, by "belongs to electrode some <role>" phrasing?"""
    labeled = [m for m in ids if battery_role.get(m) in ELECTRODE_ROLE_CLASSES]
    if not labeled:
        return {"n": 0, "accuracy": None}
    fillers = [f"{ROLE_ELECTRODE} some {r}" for r in ELECTRODE_ROLE_CLASSES]
    filler_x, man = to_ball(model, fillers)
    rows = []
    n_ok = 0
    for mid in labeled:
        v, _ = to_ball(model, [verbalizations[mid]])
        dists = {r: float(man.dist(v, filler_x[i : i + 1]).item()) for i, r in enumerate(ELECTRODE_ROLE_CLASSES)}
        nearest = min(dists, key=dists.get)
        true_role = battery_role[mid]
        ok = nearest == true_role
        n_ok += int(ok)
        rows.append({"mp_id": mid, "true_role": true_role, "nearest_role": nearest, "correct": ok})
    n = len(labeled)
    return {"n": n, "accuracy": n_ok / n, "random_baseline": 1.0 / len(ELECTRODE_ROLE_CLASSES), "rows": rows}


def _random_mrr(pool_size: int) -> float:
    """E[1/rank] for a uniformly random rank in [1, pool_size]."""
    return sum(1.0 / r for r in range(1, pool_size + 1)) / pool_size


def hasstructure_ranking(
    model,
    verbalizations: dict[str, str],
    crystals: dict[str, str],
    ids: list[str],
    decoy_pool_ids: list[str],
    seed: int = 8888,
    pool_size: int = 50,
) -> dict:
    """Phase 5b item 2: rank each material's true crystal against decoy
    crystals by Poincare distance. decoy_pool_ids should span train+held-out
    crystals so held-out materials aren't ranked against a shrunken pool."""
    rng = random.Random(seed)
    reciprocal_ranks, random_mrrs = [], []
    hits1 = hits3 = 0
    rows = []
    for mid in ids:
        v_m, _ = to_ball(model, [verbalizations[mid]])
        true_c = crystals[mid]
        decoys = [crystals[o] for o in decoy_pool_ids if o != mid and crystals[o] != true_c]
        rng.shuffle(decoys)
        pool = [true_c] + decoys[: pool_size - 1]
        c_x, man = to_ball(model, pool)
        dists = [float(man.dist(v_m, c_x[i : i + 1]).item()) for i in range(len(pool))]
        rank = sorted(range(len(pool)), key=lambda i: dists[i]).index(0) + 1
        reciprocal_ranks.append(1.0 / rank)
        random_mrrs.append(_random_mrr(len(pool)))
        hits1 += int(rank == 1)
        hits3 += int(rank <= 3)
        rows.append({"mp_id": mid, "pool_size": len(pool), "rank": rank})
    n = max(len(ids), 1)
    return {
        "n": len(ids),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "hits@1": hits1 / n,
        "hits@3": hits3 / n,
        "random_mrr_baseline": float(np.mean(random_mrrs)) if random_mrrs else None,
        "random_hits@1_baseline": float(np.mean([1.0 / r["pool_size"] for r in rows])) if rows else None,
        "rows": rows,
    }


def train(
    device: torch.device,
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    clustering_loss_weight: float = 1.0,
) -> OntologyTransformer:
    dataset = load_local_dataset(str(data_dir))
    logger.info("Train rows: %d  exist: %d", len(dataset["train"]), len(dataset["train_exist"]))
    hit_model = HierarchyTransformer.from_pretrained(model_name_or_path=BASE_MODEL, device=str(device))
    model = OntologyTransformer(hit_model, role_emd_mode="sentenceEmbedding", role_model_mode="rotation")
    hit_loss = HierarchyTransformerLoss(
        model=model.hit_model,
        clustering_loss_weight=clustering_loss_weight,
        clustering_loss_margin=3.0,
        centripetal_loss_weight=1.0,
        centripetal_loss_margin=0.5,
    )
    logical_loss = LogicalConstraintLoss(
        model=model,
        hit_loss=hit_loss,
        batch_size=batch_size,
        data_exist=dataset["train_exist"],
        data_conj=dataset["train_conj"],
        conj_weight=1.0,
        exist_weight=1.0,
        existence_loss_kind="hit",
        all_concepts=[item["name"] for item in dataset["concept_names"]],
    )
    exp = out_dir / "experiment"
    exp.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(exp),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=0,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        disable_tqdm=False,
        report_to="none",
        use_cpu=device.type == "cpu",
        dataloader_pin_memory=False,
        seed=seed,
    )
    trainer = HierarchyTransformerTrainer(
        model=model.hit_model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        loss=logical_loss,
        evaluator=None,
    )
    trainer.train()
    model.best_lambda = 0.0
    return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-materials", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--n-heldout", type=int, default=200)
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--ranking-pool-size", type=int, default=50)
    p.add_argument("--tbox-oversample-ratio", type=float, default=1.0, help="TBox:ABox-type row ratio; >1 gives the small real TBox axiom set more relative weight")
    p.add_argument("--clustering-loss-weight", type=float, default=1.0)
    p.add_argument("--abox-dir", type=Path, default=ONT_ROOT / "data" / "battgpt_abox_full")
    p.add_argument("--tbox-ont-dir", type=Path, default=ONT_ROOT / "data" / "battgpt_tbox_ont")
    p.add_argument("--force-extract", action="store_true", help="Re-extract even if entities.json already has enough materials")
    args = p.parse_args()

    abox_dir = args.abox_dir
    abox_dir.mkdir(parents=True, exist_ok=True)
    entities_path = abox_dir / "entities.json"

    t0 = time.time()
    bundle = None
    if entities_path.exists() and not args.force_extract:
        with entities_path.open(encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached.get("materials") or []) >= args.max_materials or cached.get("max_materials", 0) >= args.max_materials:
            bundle = cached
            logger.info("Reusing cached %s (%d materials)", entities_path, len(bundle["materials"]))
    if bundle is None:
        logger.info("Extracting up to %d materials from %s", args.max_materials, TTL)
        bundle = extract(TTL, max_materials=args.max_materials)
        with entities_path.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
    logger.info("Extraction: %d materials in %.1fs", len(bundle["materials"]), time.time() - t0)

    verbalize_run(entities_path, abox_dir)
    with (abox_dir / "verbalizations.json").open(encoding="utf-8") as f:
        verbal = json.load(f)
    with (abox_dir / "crystal_verbalizations.json").open(encoding="utf-8") as f:
        crystals = json.load(f)

    tbox_rows, concept_names, role_names = load_tbox(args.tbox_ont_dir)
    concept_list = list(concept_names.values())

    chemsys = {row["mp_id"]: row["chemsys"] for row in bundle["materials"]}
    battery_role = {row["mp_id"]: row.get("battery_role") for row in bundle["materials"]}
    ids = [row["mp_id"] for row in bundle["materials"]]
    rng = random.Random(args.seed)
    held = set(rng.sample(ids, k=min(args.n_heldout, max(0, len(ids) - 1))))
    train_ids = [i for i in ids if i not in held]

    t0 = time.time()
    train_rows, exist, counts = merge(
        verbal, crystals, train_ids, chemsys, tbox_rows, concept_names,
        battery_role=battery_role, seed=args.seed,
        tbox_oversample_ratio=args.tbox_oversample_ratio,
    )
    counts["heldout_ids_sample"] = sorted(held)[:20]
    counts["n_heldout"] = len(held)
    write_ont_data(abox_dir / "ont_data", train_rows, exist, counts, concept_names, role_names)
    logger.info("Merge: %s (%.1fs)", {k: v for k, v in counts.items() if k not in ("heldout_ids_sample", "concept_names")}, time.time() - t0)

    device = pick_device()
    logger.info("Device %s", device)

    base_model_for_eval = OntologyTransformer(HierarchyTransformer.from_pretrained(BASE_MODEL, device=str(device)))
    held_sorted = sorted(held)
    before_hold = type_report(base_model_for_eval, verbal, held_sorted, concept_list)
    before_crowd = crowding(base_model_for_eval, verbal, train_ids)
    del base_model_for_eval

    t0 = time.time()
    torch.manual_seed(args.seed)
    model = train(
        device, abox_dir / "ont_data", abox_dir / "train_out", args.epochs, args.batch_size, args.lr, args.seed,
        clustering_loss_weight=args.clustering_loss_weight,
    )
    train_time = time.time() - t0
    logger.info("Training took %.1fs", train_time)

    after_hold = type_report(model, verbal, held_sorted, concept_list)
    after_train = type_report(model, verbal, train_ids, concept_list)
    after_crowd = crowding(model, verbal, train_ids)

    ranking_held = hasstructure_ranking(model, verbal, crystals, held_sorted, ids, seed=args.seed, pool_size=args.ranking_pool_size)
    ranking_train = hasstructure_ranking(model, verbal, crystals, train_ids, ids, seed=args.seed, pool_size=args.ranking_pool_size)

    role_held = electrode_role_report(model, verbal, battery_role, held_sorted)
    role_train = electrode_role_report(model, verbal, battery_role, train_ids)

    passed = after_hold["type_accuracy"] >= 0.5 and after_hold["closer_than_crystal"] >= 0.5
    report = {
        "n_materials": len(ids),
        "n_train_materials": len(train_ids),
        "n_heldout": len(held),
        "counts": {k: v for k, v in counts.items() if k != "heldout_ids_sample"},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "train_time_seconds": train_time,
        "before_heldout_type": before_hold,
        "after_heldout_type": after_hold,
        "after_trainset_type": after_train,
        "crowding_before": before_crowd,
        "crowding_after": after_crowd,
        "hasstructure_ranking_heldout": ranking_held,
        "hasstructure_ranking_trainset": ranking_train,
        "electrode_role_heldout": role_held,
        "electrode_role_trainset": role_train,
        "pass": passed,
    }
    out = abox_dir / "phase5b_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    model.save(str(abox_dir / "train_out" / "final"))

    print()
    print(f"{'PASS' if passed else 'FAIL'}: held-out type accuracy {after_hold['type_accuracy']:.0%}")
    print(f"  materials: {len(ids)} (train {len(train_ids)}, held-out {len(held)})")
    print(f"  held-out type acc:      {after_hold['type_accuracy']:.0%}  (before {before_hold['type_accuracy']:.0%})")
    print(f"  held-out vs crystal:    {after_hold['closer_than_crystal']:.0%}")
    print(f"  train-set type acc:     {after_train['type_accuracy']:.0%}")
    print(f"  crowding mean dist:     before {before_crowd['mean_pairwise_dist']:.3f} -> after {after_crowd['mean_pairwise_dist']:.3f}")
    print(f"  crowding min dist:      before {before_crowd['min_pairwise_dist']:.3f} -> after {after_crowd['min_pairwise_dist']:.3f}")
    print(f"  hasStructure MRR (held): {ranking_held['mrr']:.3f}  (random baseline {ranking_held['random_mrr_baseline']:.3f})")
    print(f"  hasStructure Hits@1 (held): {ranking_held['hits@1']:.0%}  (random baseline {ranking_held['random_hits@1_baseline']:.1%})")
    if role_held["n"]:
        print(f"  electrode role acc (held): {role_held['accuracy']:.0%}  (n={role_held['n']}, random baseline {role_held['random_baseline']:.0%})")
    else:
        print("  electrode role acc (held): no held-out materials have a battery role")
    print(f"  wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
