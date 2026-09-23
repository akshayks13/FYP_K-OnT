"""Stream battery_kg.ttl and keep the first N materials plus their crystals / quantities.

Does not load the TTL as one OWL/rdflib graph. Reuses the HGT turtle block splitter.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ONT_ROOT = Path(__file__).resolve().parents[2]
PROJ_ROOT = ONT_ROOT.parent
DEFAULT_TTL = PROJ_ROOT / "output" / "battery_kg.ttl"
DEFAULT_OUT = ONT_ROOT / "data" / "battgpt_abox" / "entities.json"

PROP_MAP = {
    "band_gap": "band_gap_eV",
    "formation_energy_per_atom": "formation_energy_eV_atom",
    "energy_above_hull": "energy_above_hull_eV_atom",
    "bulk_modulus": "bulk_modulus_GPa",
    "shear_modulus": "shear_modulus_GPa",
}

# battgpt:belongsToElectrode links a material to one of exactly 4 fixed
# NamedIndividuals (never the 6 concrete leaf role classes) -- see
# ONT_ABOX_EXTENSION.md's "belongsToElectrode" finding. Names must match the
# real rdfs:label DeepOnto assigns these classes in battgpt_tbox_ont/
# concept_names.json (extract_tbox.py), so ABox exist rows and TBox concept
# names refer to the same string.
ROLE_INDIVIDUAL_TO_CLASS_LABEL = {
    "https://w3id.org/battgpt/kg#PositiveElectrodeRoleIndividual": "Positive Electrode Role",
    "https://w3id.org/battgpt/kg#NegativeElectrodeRoleIndividual": "Negative Electrode Role",
    "https://w3id.org/battgpt/kg#ElectrolyteRoleIndividual": "Electrolyte Role",
    "https://w3id.org/battgpt/kg#SeparatorRoleIndividual": "Separator Role",
}


def _load_ttl_helpers():
    path = PROJ_ROOT / "HGT-Pipeline" / "battery_hgt" / "rdf_to_hetero.py"
    spec = importlib.util.spec_from_file_location("battgpt_rdf_to_hetero", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _formula_elements(formula: str) -> list[str]:
    f = formula.strip()
    while True:
        m = re.search(r"\(([A-Za-z0-9]+)\)(\d*)", f)
        if not m:
            break
        body, n = m.group(1), int(m.group(2) or "1")
        f = f[: m.start()] + body * n + f[m.end() :]
    seen: list[str] = []
    for e in re.findall(r"[A-Z][a-z]?", f):
        if e not in seen:
            seen.append(e)
    return seen


def extract(ttl_path: Path, max_materials: int = 20) -> dict:
    h = _load_ttl_helpers()
    materials: dict[str, dict] = {}
    crystals: dict[str, dict] = {}
    props: dict[str, dict] = defaultdict(dict)
    n_sites: dict[str, int] = defaultdict(int)
    selected: list[str] = []
    selected_set: set[str] = set()

    n_blocks = 0
    for prefix_lines, subj_tok, block in h.iter_turtle_blocks(ttl_path):
        n_blocks += 1
        looks_material = "/material/" in subj_tok
        mid_hint = None
        m = re.search(r"mp-\d+", subj_tok)
        if m:
            mid_hint = m.group(0)
        if not looks_material:
            if not selected_set or mid_hint is None or mid_hint not in selected_set:
                continue
        parsed = h.fast_parse_block(prefix_lines, block)
        if not parsed:
            continue
        subj, type_uris, obj_e, lits = parsed
        mid = h.material_id_from_uri(subj)
        if looks_material and mid:
            if mid not in selected_set:
                if len(selected) >= max_materials:
                    continue
                selected.append(mid)
                selected_set.add(mid)
            rec = materials.setdefault(mid, {"mp_id": mid, "uri": subj, "type": "substance"})
            if "label" in lits and isinstance(lits["label"], str):
                rec["label"] = lits["label"]
            if "hasFormula" in lits:
                rec["formula"] = str(lits["hasFormula"])
            if "hasChemsys" in lits:
                rec["chemsys"] = str(lits["hasChemsys"])
            if "isMetal" in lits:
                rec["is_metal_flag"] = float(lits["isMetal"])
            if "isStable" in lits:
                rec["is_stable_flag"] = float(lits["isStable"])
            for pred, obj in obj_e:
                if pred in {"hasStructure", "hasstructure"} or pred.endswith("hasStructure"):
                    rec["structure_uri"] = obj
                elif pred in {"belongsToElectrode", "belongstoelectrode"} or pred.endswith("belongsToElectrode"):
                    role_label = ROLE_INDIVIDUAL_TO_CLASS_LABEL.get(obj)
                    if role_label:
                        rec["battery_role"] = role_label
        elif "/crystal/" in subj and mid and mid in selected_set:
            crec = crystals.setdefault(mid, {"mp_id": mid, "uri": subj})
            if "label" in lits and isinstance(lits["label"], str):
                crec["label"] = lits["label"]
        elif "/property/" in subj and mid and mid in selected_set:
            name = subj.rstrip("/").rsplit("/", 1)[-1]
            if name in PROP_MAP and "_value" in lits:
                try:
                    props[mid][PROP_MAP[name]] = float(lits["_value"])
                except (TypeError, ValueError):
                    pass
        elif "/site/" in subj and mid and mid in selected_set:
            n_sites[mid] += 1

    entities = []
    for mid in selected:
        mat = materials.get(mid, {"mp_id": mid, "type": "substance"})
        formula = str(mat.get("formula") or mid)
        elems = _formula_elements(formula)
        n = int(n_sites.get(mid) or 0)
        crystal = crystals.get(mid, {})
        structure = crystal.get("label") or f"Crystal structure of {formula}"
        row = {
            "mp_id": mid,
            "uri": mat.get("uri", f"https://w3id.org/battgpt/kg/material/{mid}"),
            "label": mat.get("label") or f"Material {formula} ({mid})",
            "formula": formula,
            "chemsys": str(mat.get("chemsys") or "-".join(elems)),
            "type": "substance",
            "is_metal_flag": float(mat.get("is_metal_flag", 0.0)),
            "is_stable_flag": float(mat.get("is_stable_flag", 0.0)),
            "structure": structure,
            "structure_uri": mat.get("structure_uri", f"https://w3id.org/battgpt/kg/crystal/{mid}"),
            "battery_role": mat.get("battery_role"),
            "sites": f"{n} ({', '.join(elems)})" if elems else str(n),
            "n_sites": n,
            "elements": elems,
            **props.get(mid, {}),
        }
        entities.append(row)

    return {
        "ttl": str(ttl_path),
        "max_materials": max_materials,
        "n_blocks": n_blocks,
        "materials": entities,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ttl", type=Path, default=DEFAULT_TTL)
    p.add_argument("--max-materials", type=int, default=20)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    print(f"Streaming {args.ttl} (first {args.max_materials} materials)…", flush=True)
    bundle = extract(args.ttl, max_materials=args.max_materials)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"Wrote {len(bundle['materials'])} materials → {args.out} ({bundle['n_blocks']:,} blocks)")
    for row in bundle["materials"][:5]:
        print(f"  {row['mp_id']} {row['formula']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
