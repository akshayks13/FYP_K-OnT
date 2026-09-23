#!/usr/bin/env python3
"""
hetero_to_cifs.py — Write one CIF per battery-KG material from HeteroData
=======================================================================

Do NOT use this for bridge fine-tune. Fine-tune CIFs come from Materials Project
(`CrystaLLM/bin/fetch_mp_cifs.py`). This script only dumps graph tensors as CIF-like
text for debugging HeteroData.

Usage
-----
  python hetero_to_cifs.py \\
      --hetero-dir ../../output \\
      --out-dir ../../CrystaLLM/kg_finetune_cifs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

CRYSTALLM = Path(__file__).resolve().parents[2] / "CrystaLLM"
if str(CRYSTALLM) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM))

from crystallm import get_atomic_props_block_for_formula  # noqa: E402


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]


def _one_to_one(ei) -> dict[int, int]:
    src, dst = ei[0].tolist(), ei[1].tolist()
    return {int(s): int(d) for s, d in zip(src, dst)}


def _one_to_many(ei) -> dict[int, list[int]]:
    src, dst = ei[0].tolist(), ei[1].tolist()
    out: dict[int, list[int]] = defaultdict(list)
    for s, d in zip(src, dst):
        out[int(s)].append(int(d))
    return out


def build_cif(
    formula: str,
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
    sg_symbol: str,
    sg_num: int,
    sites: list[tuple[str, float, float, float]],
) -> str:
    try:
        atom_block = get_atomic_props_block_for_formula(formula).rstrip()
    except Exception:
        atom_block = ""
    sg = sg_symbol or "P1"
    lines = [
        f"data_{formula}",
    ]
    if atom_block:
        lines.append(atom_block)
    lines.extend([
        f"_symmetry_space_group_name_H-M {sg}",
        f"_cell_length_a {a:.4f}",
        f"_cell_length_b {b:.4f}",
        f"_cell_length_c {c:.4f}",
        f"_cell_angle_alpha {alpha:.4f}",
        f"_cell_angle_beta {beta:.4f}",
        f"_cell_angle_gamma {gamma:.4f}",
        f"_symmetry_Int_Tables_number {int(sg_num) if sg_num else 1}",
        f"_chemical_formula_structural {formula}",
        f"_chemical_formula_sum '{formula}'",
        "loop_",
        "_symmetry_equiv_pos_site_id",
        "_symmetry_equiv_pos_as_xyz",
        "1 'x, y, z'",
        "loop_",
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_symmetry_multiplicity",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ])
    counts: dict[str, int] = defaultdict(int)
    for el, x, y, z in sites:
        counts[el] += 1
        lab = f"{el}{counts[el]}"
        lines.append(f"{el} {lab} 1 {x:.4f} {y:.4f} {z:.4f} 1")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="HeteroData → one CIF per KG material")
    p.add_argument("--hetero-dir", type=Path, default=Path("../../output"))
    p.add_argument("--out-dir", type=Path, default=Path("../../CrystaLLM/kg_finetune_cifs"))
    p.add_argument("--max-materials", type=int, default=None)
    args = p.parse_args()

    hetero_dir = args.hetero_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(hetero_dir / "hetero_data.pt", map_location="cpu", weights_only=False)
    meta = json.loads((hetero_dir / "node_mapping.json").read_text(encoding="utf-8"))

    mat_to_cryst = _one_to_one(data["Material", "hasStructure", "Crystal"].edge_index)
    cryst_to_uc = _one_to_one(data["Crystal", "hasUnitCell", "UnitCell"].edge_index)
    cryst_to_sg = _one_to_one(data["Crystal", "hasSpaceGroup", "SpaceGroup"].edge_index)
    uc_to_sites = _one_to_many(data["UnitCell", "containsSite", "AtomicSite"].edge_index)
    site_to_sp = _one_to_one(data["AtomicSite", "containsSpecies", "AtomicSpecies"].edge_index)

    uc_x = data["UnitCell"].x
    site_x = data["AtomicSite"].x
    sp_symbol = {
        int(e["tensor_index"]): e.get("species_symbol") or "X"
        for e in meta["AtomicSpecies"]
    }
    sg_info = {
        int(e["tensor_index"]): (
            str(e.get("symbol") or "P1"),
            int(e.get("spacegroup_number") or 1),
        )
        for e in meta["SpaceGroup"]
    }

    n_ok = n_skip = 0
    materials = meta["Material"]
    if args.max_materials:
        materials = materials[: args.max_materials]

    for m in materials:
        mid = int(m["tensor_index"])
        formula = (m.get("formula") or "").replace(" ", "")
        if not formula:
            n_skip += 1
            continue
        cid = mat_to_cryst.get(mid)
        if cid is None:
            n_skip += 1
            continue
        uid = cryst_to_uc.get(cid)
        if uid is None:
            n_skip += 1
            continue
        site_ids = uc_to_sites.get(uid) or []
        if not site_ids:
            n_skip += 1
            continue
        sites = []
        for sid in site_ids:
            sp = site_to_sp.get(sid)
            el = sp_symbol.get(sp, "X") if sp is not None else "X"
            if el == "X":
                continue
            xyz = site_x[sid]
            sites.append((el, float(xyz[0]), float(xyz[1]), float(xyz[2])))
        if not sites:
            n_skip += 1
            continue
        lat = uc_x[uid]
        sg_i = cryst_to_sg.get(cid)
        sg_sym, sg_num = sg_info.get(sg_i, ("P1", 1)) if sg_i is not None else ("P1", 1)
        cif = build_cif(
            formula,
            float(lat[0]), float(lat[1]), float(lat[2]),
            float(lat[3]), float(lat[4]), float(lat[5]),
            sg_sym, sg_num, sites,
        )
        mp = m.get("material_id") or str(mid)
        fname = f"{_safe(str(mp))}_{_safe(formula)}.cif"
        (out_dir / fname).write_text(cif, encoding="utf-8")
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  wrote {n_ok} CIFs…")

    print(f"Wrote {n_ok} CIFs → {out_dir}  (skipped {n_skip})")


if __name__ == "__main__":
    main()
