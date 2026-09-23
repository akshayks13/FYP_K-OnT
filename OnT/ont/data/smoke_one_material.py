"""1-material OnT smoke: train type + hasStructure, check LiMgP sits under substance.

Follows ONT_ABOX_EXTENSION.md Phases 4-5 for a single material (mp-10178 / LiMgP).
Does not change ont/model.py or losses. Concat is export-only (after encode).
"""
from __future__ import annotations

import json
import logging
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

ONT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ONT_ROOT not in sys.path:
    sys.path.insert(0, ONT_ROOT)

from ont.data.load import load_local_dataset
from ont.hit import HierarchyTransformer
from ont.losses.hit_loss import HierarchyTransformerLoss
from ont.losses.logical_loss import LogicalConstraintLoss
from ont.model import OntologyTransformer
from ont.trainer import HierarchyTransformerTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("smoke_one_material")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

SMOKE_DIR = os.path.join(ONT_ROOT, "data", "battgpt_abox_smoke")
DATA_DIR = os.path.join(SMOKE_DIR, "ont_data")
OUT_DIR = os.path.join(SMOKE_DIR, "train_out")
PROBE_VERBAL = os.path.join(ONT_ROOT, "data", "numeric_probe", "verbalizations.json")
PROBE_NUM = os.path.join(ONT_ROOT, "data", "numeric_probe", "numeric_features.json")

CLASSES = [
    "entity",
    "substance",
    "crystal",
    "element",
    "battery role",
    "space group",
    "unit cell",
]
TYPE_CLASSES = [c for c in CLASSES if c != "entity"]
TRUE_TYPE = "substance"
ROLE = "has structure"
BASE_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
MAX_STEPS = 20
BATCH_SIZE = 4
LR = 1e-5
SEED = 8888


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _rows_per_negative(child: str, parent: str, negatives: list[str]) -> list[dict]:
    """OnT load.py keeps only negative[:1], so emit one JSONL row per negative."""
    return [{"child": child, "parent": parent, "negative": [neg]} for neg in negatives]


def build_rows(v_material: str, v_crystal: str) -> tuple[list[dict], list[dict]]:
    """TBox class rows + material type rows + hasStructure exist rows."""
    siblings = list(TYPE_CLASSES)
    tbox: list[dict] = []
    for child in TYPE_CLASSES:
        neg = [c for c in siblings if c != child] or ["entity"]
        tbox.extend(_rows_per_negative(child, "entity", neg[:1]))
    tbox = tbox * 3

    type_negs = [c for c in TYPE_CLASSES if c != TRUE_TYPE]
    abox_type = _rows_per_negative(v_material, TRUE_TYPE, type_negs)
    # ~1:1 TBox vs ABox type rows (plan default), after expanding negatives.
    repeat = max(1, len(tbox) // max(len(abox_type), 1))
    train = tbox + abox_type * repeat

    exist_parent = f"{ROLE} some {v_crystal}"
    train.extend(
        _rows_per_negative(
            v_material,
            exist_parent,
            [f"{ROLE} some {c}" for c in ("crystal", "element", "unit cell")],
        )
    )
    exist = [{"Concept": exist_parent, "role": ROLE, "con": v_crystal}]
    return train, exist


def write_ont_data(v_material: str, v_crystal: str) -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    train, exist = build_rows(v_material, v_crystal)
    n_abox_type = sum(1 for r in train if r["parent"] == TRUE_TYPE)
    n_abox_exist = sum(1 for r in train if str(r["parent"]).startswith(ROLE))
    n_tbox = len(train) - n_abox_type - n_abox_exist
    write_jsonl(os.path.join(DATA_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(DATA_DIR, "train_exist.jsonl"), exist)
    write_jsonl(os.path.join(DATA_DIR, "train_conj.jsonl"), [])
    with open(os.path.join(DATA_DIR, "concept_names.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): name for i, name in enumerate(CLASSES)}, f, indent=2)
    with open(os.path.join(DATA_DIR, "role_names.json"), "w", encoding="utf-8") as f:
        json.dump({"0": ROLE}, f, indent=2)
    with open(os.path.join(DATA_DIR, "role_inverse.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    empty_eval = {
        "query_sentences": {"nf1": [], "nf2": [], "nf3": [], "nf4": []},
        "answer_ids": {"nf1": [], "nf2": [], "nf3": [], "nf4": []},
    }
    with open(os.path.join(DATA_DIR, "val.json"), "w", encoding="utf-8") as f:
        json.dump(empty_eval, f, indent=2)
    counts = {
        "n_tbox": n_tbox,
        "n_abox_type": n_abox_type,
        "n_abox_exist": n_abox_exist,
        "n_train_jsonl": len(train),
        "concept_names": CLASSES,
        "note": "concept_names is classes only; V(a) is not listed there",
    }
    with open(os.path.join(SMOKE_DIR, "merge_counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    logger.info("Wrote OnT rows: %s", counts)
    return counts


def poincare_report(model: OntologyTransformer, v_material: str) -> dict:
    """Nearest class of V(a) by manifold.dist; parent should also be closer to origin."""
    from geoopt.manifolds import PoincareBall

    labels = [v_material] + CLASSES
    raw = model.encode(labels, convert_to_tensor=True, show_progress_bar=False)
    if isinstance(raw, torch.Tensor):
        x = raw.detach().float().cpu()
    else:
        x = torch.tensor(raw, dtype=torch.float32)
    # Trainer may leave manifold.k on MPS; distances must run on CPU.
    c = model.manifold.c
    c = float(c.detach().cpu()) if torch.is_tensor(c) else float(c)
    man = PoincareBall(c=c)
    radius = 1.0 / float(np.sqrt(c))
    norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    x = x * torch.clamp((radius - 1e-5) / norms, max=1.0)
    v = x[0:1]
    dist_to_class = {}
    for i, name in enumerate(CLASSES, start=1):
        dist_to_class[name] = float(man.dist(v, x[i : i + 1]).item())
    dist0 = {"V(LiMgP)": float(man.dist0(v).item())}
    for i, name in enumerate(CLASSES, start=1):
        dist0[name] = float(man.dist0(x[i : i + 1]).item())
    typed = {k: dist_to_class[k] for k in TYPE_CLASSES}
    nearest = min(typed, key=typed.get)
    return {
        "dist_to_class": dist_to_class,
        "dist0": dist0,
        "nearest_type": nearest,
        "sits_under_substance": nearest == TRUE_TYPE,
        "closer_to_substance_than_crystal": dist_to_class[TRUE_TYPE] < dist_to_class["crystal"],
        "parent_nearer_origin": dist0[TRUE_TYPE] < dist0["V(LiMgP)"],
    }


def train_one_material(device: torch.device) -> OntologyTransformer:
    dataset = load_local_dataset(DATA_DIR)
    logger.info("Train rows after load.py (first negative only): %d", len(dataset["train"]))
    logger.info("Loading %s on %s", BASE_MODEL, device)
    hit_model = HierarchyTransformer.from_pretrained(model_name_or_path=BASE_MODEL, device=str(device))
    model = OntologyTransformer(hit_model, role_emd_mode="sentenceEmbedding", role_model_mode="rotation")
    hit_loss = HierarchyTransformerLoss(
        model=model.hit_model,
        clustering_loss_weight=1.0,
        clustering_loss_margin=3.0,
        centripetal_loss_weight=1.0,
        centripetal_loss_margin=0.5,
    )
    all_concept_names = [item["name"] for item in dataset["concept_names"]]
    logical_loss = LogicalConstraintLoss(
        model=model,
        hit_loss=hit_loss,
        batch_size=BATCH_SIZE,
        data_exist=dataset["train_exist"],
        data_conj=dataset["train_conj"],
        conj_weight=1.0,
        exist_weight=1.0,
        existence_loss_kind="hit",
        all_concepts=all_concept_names,
    )
    os.makedirs(os.path.join(OUT_DIR, "experiment"), exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(OUT_DIR, "experiment"),
        num_train_epochs=3,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=0,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
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


def concat_preview(v: np.ndarray, x_raw: list[float]) -> dict:
    """Export-time concat only. Not Poincaré. 1 material cannot fit a real scaler."""
    x = np.asarray(x_raw, dtype=np.float64)
    z = np.concatenate([v, x])
    return {
        "v_dim": int(v.shape[-1]),
        "x_dim": int(x.shape[-1]),
        "z_dim": int(z.shape[-1]),
        "note": "z is not in the Poincaré ball; 1-material scaler skipped (std would be 0)",
    }


def main() -> int:
    with open(PROBE_VERBAL, encoding="utf-8") as f:
        verbal = json.load(f)
    with open(PROBE_NUM, encoding="utf-8") as f:
        nums = json.load(f)
    v_material = verbal["real-mp-10178-LiMgP"]
    v_crystal = "Crystal structure of LiMgP"
    x_raw = [
        float(nums["real-mp-10178-LiMgP"][k])
        for k in (
            "band_gap_eV",
            "formation_energy_eV_atom",
            "energy_above_hull_eV_atom",
            "bulk_modulus_GPa",
            "shear_modulus_GPa",
            "is_metal_flag",
            "is_stable_flag",
        )
    ]
    os.makedirs(SMOKE_DIR, exist_ok=True)
    counts = write_ont_data(v_material, v_crystal)
    device = pick_device()
    logger.info("Device: %s", device)

    hit_before = HierarchyTransformer.from_pretrained(model_name_or_path=BASE_MODEL, device=str(device))
    model_before = OntologyTransformer(hit_before)
    before = poincare_report(model_before, v_material)
    logger.info("BEFORE train nearest type=%s dist(substance)=%.4f dist(crystal)=%.4f",
                before["nearest_type"], before["dist_to_class"][TRUE_TYPE], before["dist_to_class"]["crystal"])

    torch.manual_seed(SEED)
    model = train_one_material(device)
    after = poincare_report(model, v_material)
    logger.info("AFTER train nearest type=%s dist(substance)=%.4f dist(crystal)=%.4f",
                after["nearest_type"], after["dist_to_class"][TRUE_TYPE], after["dist_to_class"]["crystal"])

    v_np = model.encode(v_material, show_progress_bar=False)
    concat = concat_preview(np.asarray(v_np).reshape(-1), x_raw)
    passed = bool(after["sits_under_substance"] and after["closer_to_substance_than_crystal"])
    report = {
        "material": "LiMgP (mp-10178)",
        "true_type": TRUE_TYPE,
        "base_model": BASE_MODEL,
        "max_steps": MAX_STEPS,
        "device": str(device),
        "counts": counts,
        "before": before,
        "after": after,
        "concat": concat,
        "pass": passed,
        "verdict": (
            "PASS: after OnT training, V(LiMgP) is nearest to substance "
            "and closer to substance than to crystal on the Poincaré ball."
            if passed
            else "FAIL: V(LiMgP) did not sit under substance after this tiny train."
        ),
    }
    out_path = os.path.join(SMOKE_DIR, "sit_below_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print()
    print(report["verdict"])
    print(f"  nearest type after train: {after['nearest_type']}")
    print(f"  dist to substance: {after['dist_to_class'][TRUE_TYPE]:.4f}")
    print(f"  dist to crystal:   {after['dist_to_class']['crystal']:.4f}")
    print(f"  parent nearer origin: {after['parent_nearer_origin']}")
    print(f"  concat z dim (export only): {concat['z_dim']}")
    print(f"  wrote {out_path}")
    final_dir = os.path.join(OUT_DIR, "final")
    model.save(final_dir)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
