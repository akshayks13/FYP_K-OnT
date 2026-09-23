"""20-material OnT slice: extract → verbalize → merge → 1 epoch → Phase 5b type check."""
from __future__ import annotations

import json
import logging
import os
import sys
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
from ont.data.merge_abox import CLASSES, TRUE_TYPE, TYPE_CLASSES, merge, write_ont_data
from ont.data.verbalize_abox import FEATURE_NAMES, run as verbalize_run
from ont.hit import HierarchyTransformer
from ont.losses.hit_loss import HierarchyTransformerLoss
from ont.losses.logical_loss import LogicalConstraintLoss
from ont.model import OntologyTransformer
from ont.trainer import HierarchyTransformerTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("smoke_twenty")
logging.getLogger("httpx").setLevel(logging.WARNING)

ABOX_DIR = ONT_ROOT / "data" / "battgpt_abox"
TTL = ONT_ROOT.parent / "output" / "battery_kg.ttl"
BASE_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
BATCH_SIZE = 16
EPOCHS = 1
LR = 1e-5
SEED = 8888
N_HELDOUT = 4
MAX_MATERIALS = 20


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_ball(model: OntologyTransformer, sentences: list[str]) -> torch.Tensor:
    raw = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False, batch_size=16)
    x = raw.detach().float().cpu() if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32)
    c = model.manifold.c
    c = float(c.detach().cpu()) if torch.is_tensor(c) else float(c)
    radius = 1.0 / float(np.sqrt(c))
    norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return x * torch.clamp((radius - 1e-5) / norms, max=1.0), PoincareBall(c=c)


def type_report(model: OntologyTransformer, verbalizations: dict[str, str], ids: list[str]) -> dict:
    class_sents = list(CLASSES)
    class_x, man = to_ball(model, class_sents)
    class_idx = {name: i for i, name in enumerate(class_sents)}
    rows = []
    n_ok = 0
    n_vs_crystal = 0
    n_parent = 0
    for mid in ids:
        v, _ = to_ball(model, [verbalizations[mid]])
        dist = {name: float(man.dist(v, class_x[i : i + 1]).item()) for name, i in class_idx.items()}
        typed = {k: dist[k] for k in TYPE_CLASSES}
        nearest = min(typed, key=typed.get)
        dist0_v = float(man.dist0(v).item())
        dist0_s = float(man.dist0(class_x[class_idx[TRUE_TYPE] : class_idx[TRUE_TYPE] + 1]).item())
        ok = nearest == TRUE_TYPE
        vs_c = dist[TRUE_TYPE] < dist["crystal"]
        parent = dist0_s < dist0_v
        n_ok += int(ok)
        n_vs_crystal += int(vs_c)
        n_parent += int(parent)
        rows.append(
            {
                "mp_id": mid,
                "nearest_type": nearest,
                "dist_substance": dist[TRUE_TYPE],
                "dist_crystal": dist["crystal"],
                "dist_element": dist["element"],
                "sits_under_substance": ok,
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


def crowding(model: OntologyTransformer, verbalizations: dict[str, str], ids: list[str]) -> dict:
    if len(ids) < 2:
        return {"n": len(ids), "mean_pairwise_dist": None}
    sents = [verbalizations[m] for m in ids]
    x, man = to_ball(model, sents)
    dists = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dists.append(float(man.dist(x[i : i + 1], x[j : j + 1]).item()))
    return {
        "n": len(ids),
        "mean_pairwise_dist": float(np.mean(dists)),
        "min_pairwise_dist": float(np.min(dists)),
        "max_pairwise_dist": float(np.max(dists)),
    }


def train(device: torch.device, data_dir: Path, out_dir: Path) -> OntologyTransformer:
    dataset = load_local_dataset(str(data_dir))
    logger.info("Train rows: %d  exist: %d", len(dataset["train"]), len(dataset["train_exist"]))
    hit_model = HierarchyTransformer.from_pretrained(model_name_or_path=BASE_MODEL, device=str(device))
    model = OntologyTransformer(hit_model, role_emd_mode="sentenceEmbedding", role_model_mode="rotation")
    hit_loss = HierarchyTransformerLoss(
        model=model.hit_model,
        clustering_loss_weight=1.0,
        clustering_loss_margin=3.0,
        centripetal_loss_weight=1.0,
        centripetal_loss_margin=0.5,
    )
    logical_loss = LogicalConstraintLoss(
        model=model,
        hit_loss=hit_loss,
        batch_size=BATCH_SIZE,
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
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=0,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=5,
        disable_tqdm=False,
        report_to="none",
        use_cpu=device.type == "cpu",
        dataloader_pin_memory=False,
        seed=SEED,
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
    import random

    ABOX_DIR.mkdir(parents=True, exist_ok=True)
    entities_path = ABOX_DIR / "entities.json"
    if entities_path.exists():
        with entities_path.open(encoding="utf-8") as f:
            bundle = json.load(f)
        if len(bundle.get("materials") or []) >= MAX_MATERIALS:
            logger.info("Reusing %s", entities_path)
        else:
            bundle = None
    else:
        bundle = None
    if bundle is None:
        logger.info("Extracting %d materials from %s", MAX_MATERIALS, TTL)
        bundle = extract(TTL, max_materials=MAX_MATERIALS)
        with entities_path.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
    verbalize_run(entities_path, ABOX_DIR)
    with (ABOX_DIR / "verbalizations.json").open(encoding="utf-8") as f:
        verbal = json.load(f)
    with (ABOX_DIR / "crystal_verbalizations.json").open(encoding="utf-8") as f:
        crystals = json.load(f)
    chemsys = {row["mp_id"]: row["chemsys"] for row in bundle["materials"]}
    ids = [row["mp_id"] for row in bundle["materials"]]
    rng = random.Random(SEED)
    held = set(rng.sample(ids, k=min(N_HELDOUT, max(0, len(ids) - 1))))
    train_ids = [i for i in ids if i not in held]
    train_rows, exist, counts = merge(verbal, crystals, train_ids, chemsys, seed=SEED)
    counts["heldout_ids"] = sorted(held)
    counts["train_ids"] = train_ids
    write_ont_data(ABOX_DIR / "ont_data", train_rows, exist, counts)
    logger.info("Merged %s", counts)

    device = pick_device()
    logger.info("Device %s", device)
    before_hold = type_report(
        OntologyTransformer(HierarchyTransformer.from_pretrained(BASE_MODEL, device=str(device))),
        verbal,
        sorted(held),
    )
    torch.manual_seed(SEED)
    model = train(device, ABOX_DIR / "ont_data", ABOX_DIR / "train_out")
    after_hold = type_report(model, verbal, sorted(held))
    after_train = type_report(model, verbal, train_ids)
    crowd = crowding(model, verbal, train_ids)
    passed = after_hold["type_accuracy"] >= 0.5 and after_hold["closer_than_crystal"] >= 0.5
    report = {
        "n_materials": len(ids),
        "train_ids": train_ids,
        "heldout_ids": sorted(held),
        "counts": counts,
        "epochs": EPOCHS,
        "device": str(device),
        "before_heldout": before_hold,
        "after_heldout": after_hold,
        "after_trainset": after_train,
        "crowding_train_v": crowd,
        "pass": passed,
        "verdict": (
            f"PASS: held-out type accuracy {after_hold['type_accuracy']:.0%} "
            f"({after_hold['n']} materials)."
            if passed
            else f"FAIL: held-out type accuracy {after_hold['type_accuracy']:.0%}."
        ),
    }
    out = ABOX_DIR / "phase5b_type_check.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    model.save(str(ABOX_DIR / "train_out" / "final"))
    print()
    print(report["verdict"])
    print(f"  held-out type acc:     {after_hold['type_accuracy']:.0%}  (before {before_hold['type_accuracy']:.0%})")
    print(f"  held-out vs crystal:   {after_hold['closer_than_crystal']:.0%}")
    print(f"  train-set type acc:    {after_train['type_accuracy']:.0%}")
    print(f"  train pairwise dist:   {crowd['mean_pairwise_dist']}")
    print(f"  wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
