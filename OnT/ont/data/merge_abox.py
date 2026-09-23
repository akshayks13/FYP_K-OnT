"""Merge the real TBox (from extract_tbox.py + ont/data/prepare.py) with ABox
type + hasStructure rows for OnT.

TBox rows come straight from battgpt_tbox_ont/train.jsonl (real subClassOf
axioms verbalised by DeepOnto from battgpt_tbox.owl) -- nothing here invents
class names or hierarchy edges.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ONT_ROOT = Path(__file__).resolve().parents[2]
ABOX_DIR = ONT_ROOT / "data" / "battgpt_abox"
TBOX_ONT_DIR = ONT_ROOT / "data" / "battgpt_tbox_ont"

TRUE_TYPE = "substance"
ROLE = "has structure"
ROLE_ELECTRODE = "belongs to electrode"
# The only 4 fillers battgpt:belongsToElectrode ever points at (see
# extract_abox.py's ROLE_INDIVIDUAL_TO_CLASS_LABEL) -- a fixed, small set,
# unlike hasStructure's per-material crystal pool.
ELECTRODE_ROLE_CLASSES = [
    "Positive Electrode Role",
    "Negative Electrode Role",
    "Electrolyte Role",
    "Separator Role",
]


def load_tbox(tbox_ont_dir: Path) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    with (tbox_ont_dir / "train.jsonl").open(encoding="utf-8") as f:
        tbox_rows = [json.loads(line) for line in f if line.strip()]
    with (tbox_ont_dir / "concept_names.json").open(encoding="utf-8") as f:
        concept_names = json.load(f)
    with (tbox_ont_dir / "role_names.json").open(encoding="utf-8") as f:
        role_names = json.load(f)
    if TRUE_TYPE not in concept_names.values():
        raise ValueError(f"{TRUE_TYPE!r} missing from {tbox_ont_dir}/concept_names.json")
    if ROLE not in role_names.values():
        raise ValueError(f"{ROLE!r} missing from {tbox_ont_dir}/role_names.json")
    if ROLE_ELECTRODE not in role_names.values():
        raise ValueError(f"{ROLE_ELECTRODE!r} missing from {tbox_ont_dir}/role_names.json")
    missing_role_classes = [c for c in ELECTRODE_ROLE_CLASSES if c not in concept_names.values()]
    if missing_role_classes:
        raise ValueError(f"{missing_role_classes} missing from {tbox_ont_dir}/concept_names.json")
    return tbox_rows, concept_names, role_names


def _rows_per_negative(child: str, parent: str, negatives: list[str]) -> list[dict]:
    return [{"child": child, "parent": parent, "negative": [neg]} for neg in negatives]


def merge(
    verbalizations: dict[str, str],
    crystals: dict[str, str],
    train_ids: list[str],
    chemsys: dict[str, str],
    tbox_rows: list[dict],
    concept_names: dict[str, str],
    battery_role: dict[str, str] | None = None,
    seed: int = 8888,
    n_type_negatives: int = 10,
) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    concept_list = list(concept_names.values())
    type_pool = [c for c in concept_list if c != TRUE_TYPE]

    abox_type: list[dict] = []
    for mid in train_ids:
        negs = rng.sample(type_pool, k=min(n_type_negatives, len(type_pool)))
        abox_type.extend(_rows_per_negative(verbalizations[mid], TRUE_TYPE, negs))

    # ~1:1 TBox vs ABox type (plan default). Repeat the (small) real TBox
    # axiom set so it isn't drowned by thousands of ABox type rows.
    if abox_type and tbox_rows:
        tbox_rep = tbox_rows * max(1, (len(abox_type) + len(tbox_rows) - 1) // len(tbox_rows))
        tbox_rep = tbox_rep[: len(abox_type)]
    else:
        tbox_rep = list(tbox_rows)

    exist_rows: list[dict] = []
    train_exist: list[dict] = []
    crystal_pool = [crystals[m] for m in train_ids]
    for mid in train_ids:
        v_m = verbalizations[mid]
        v_c = crystals[mid]
        exist_parent = f"{ROLE} some {v_c}"
        negs = []
        same = [crystals[o] for o in train_ids if o != mid and chemsys.get(o) == chemsys.get(mid)]
        others = [crystals[o] for o in train_ids if o != mid]
        rng.shuffle(same)
        rng.shuffle(others)
        for c in same[:1] + others[:2]:
            if c != v_c and c not in negs:
                negs.append(c)
        if not negs:
            negs = [c for c in crystal_pool if c != v_c][:2] or ["crystal"]
        exist_rows.extend(_rows_per_negative(v_m, exist_parent, [f"{ROLE} some {n}" for n in negs]))
        train_exist.append({"Concept": exist_parent, "role": ROLE, "con": v_c})
    n_structure_rows = len(exist_rows)

    # Second exist channel: battgpt:belongsToElectrode covers ~73% of the
    # real KG's materials and wires them into the real battery-role
    # hierarchy (BatteryRole -> {Electrolyte,Positive,Negative,Separator}
    # Role) rather than just the flat "material sub substance" typing.
    # Only 4 possible fillers total, so no hard-negative mining needed --
    # the other 3 role classes are the negatives.
    n_electrode_rows = 0
    if battery_role:
        for mid in train_ids:
            role = battery_role.get(mid)
            if not role or role not in ELECTRODE_ROLE_CLASSES:
                continue
            v_m = verbalizations[mid]
            exist_parent = f"{ROLE_ELECTRODE} some {role}"
            negs = [f"{ROLE_ELECTRODE} some {r}" for r in ELECTRODE_ROLE_CLASSES if r != role]
            exist_rows.extend(_rows_per_negative(v_m, exist_parent, negs))
            train_exist.append({"Concept": exist_parent, "role": ROLE_ELECTRODE, "con": role})
            n_electrode_rows += 1

    train = tbox_rep + abox_type + exist_rows
    rng.shuffle(train)
    counts = {
        "n_tbox": len(tbox_rep),
        "n_tbox_unique_axioms": len(tbox_rows),
        "n_abox_type": len(abox_type),
        "n_abox_exist_structure": n_structure_rows,
        "n_abox_exist_electrode": len(exist_rows) - n_structure_rows,
        "n_train_jsonl": len(train),
        "n_train_exist": len(train_exist),
        "n_train_materials": len(train_ids),
        "n_concepts": len(concept_names),
        "concept_names": concept_list,
    }
    return train, train_exist, counts


def write_ont_data(
    data_dir: Path,
    train: list[dict],
    exist: list[dict],
    counts: dict,
    concept_names: dict[str, str],
    role_names: dict[str, str],
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    with (data_dir / "train_exist.jsonl").open("w", encoding="utf-8") as f:
        for row in exist:
            f.write(json.dumps(row) + "\n")
    (data_dir / "train_conj.jsonl").write_text("", encoding="utf-8")
    with (data_dir / "concept_names.json").open("w", encoding="utf-8") as f:
        json.dump(concept_names, f, indent=2)
    with (data_dir / "role_names.json").open("w", encoding="utf-8") as f:
        json.dump(role_names, f, indent=2)
    with (data_dir / "role_inverse.json").open("w", encoding="utf-8") as f:
        json.dump({}, f)
    empty_eval = {
        "query_sentences": {"nf1": [], "nf2": [], "nf3": [], "nf4": []},
        "answer_ids": {"nf1": [], "nf2": [], "nf3": [], "nf4": []},
    }
    with (data_dir / "val.json").open("w", encoding="utf-8") as f:
        json.dump(empty_eval, f, indent=2)
    with (data_dir.parent / "merge_counts.json").open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--abox-dir", type=Path, default=ABOX_DIR)
    p.add_argument("--tbox-ont-dir", type=Path, default=TBOX_ONT_DIR)
    p.add_argument("--n-heldout", type=int, default=4)
    p.add_argument("--seed", type=int, default=8888)
    args = p.parse_args()
    with (args.abox_dir / "verbalizations.json").open(encoding="utf-8") as f:
        verbal = json.load(f)
    with (args.abox_dir / "crystal_verbalizations.json").open(encoding="utf-8") as f:
        crystals = json.load(f)
    with (args.abox_dir / "entities.json").open(encoding="utf-8") as f:
        ents = json.load(f)
    tbox_rows, concept_names, role_names = load_tbox(args.tbox_ont_dir)
    chemsys = {row["mp_id"]: row["chemsys"] for row in ents["materials"]}
    battery_role = {row["mp_id"]: row.get("battery_role") for row in ents["materials"]}
    ids = [row["mp_id"] for row in ents["materials"]]
    rng = random.Random(args.seed)
    held = set(rng.sample(ids, k=min(args.n_heldout, max(0, len(ids) - 1))))
    train_ids = [i for i in ids if i not in held]
    train, exist, counts = merge(
        verbal, crystals, train_ids, chemsys, tbox_rows, concept_names,
        battery_role=battery_role, seed=args.seed,
    )
    counts["heldout_ids"] = sorted(held)
    counts["train_ids"] = train_ids
    write_ont_data(args.abox_dir / "ont_data", train, exist, counts, concept_names, role_names)
    print(json.dumps({k: v for k, v in counts.items() if k not in ("heldout_ids", "train_ids")}, indent=2))


if __name__ == "__main__":
    main()
