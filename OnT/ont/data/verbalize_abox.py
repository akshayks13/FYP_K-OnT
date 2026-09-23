"""Structured V(a) + numeric side channel from extract_abox.py entities.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ONT_ROOT = Path(__file__).resolve().parents[2]
ABOX_DIR = ONT_ROOT / "data" / "battgpt_abox"

FEATURE_NAMES = [
    "band_gap_eV",
    "formation_energy_eV_atom",
    "energy_above_hull_eV_atom",
    "bulk_modulus_GPa",
    "shear_modulus_GPa",
    "is_metal_flag",
    "is_stable_flag",
]


def _fmt(row: dict, key: str, spec: str, suffix: str) -> str | None:
    if key not in row or row[key] is None:
        return None
    try:
        return f"{spec.format(float(row[key]))}{suffix}"
    except (TypeError, ValueError):
        return None


def verbalize_material(row: dict) -> str:
    metal = "true" if float(row.get("is_metal_flag") or 0) else "false"
    stable = "true" if float(row.get("is_stable_flag") or 0) else "false"
    lines = [
        f"Material: {row['formula']} ({row['mp_id']})",
        f"Type: {row.get('type', 'substance')}",
        f"Formula: {row['formula']}",
        f"Chemical system: {row['chemsys']}",
        f"Metallic: {metal}",
        f"Stable: {stable}",
    ]
    for key, spec, suffix in (
        ("band_gap_eV", "Band gap: {:.2f}", " eV"),
        ("formation_energy_eV_atom", "Formation energy: {:.2f}", " eV/atom"),
        ("energy_above_hull_eV_atom", "Energy above hull: {:.2f}", " eV/atom"),
        ("bulk_modulus_GPa", "Bulk modulus: {:.1f}", " GPa"),
        ("shear_modulus_GPa", "Shear modulus: {:.1f}", " GPa"),
    ):
        line = _fmt(row, key, spec, suffix)
        if line:
            lines.append(line)
    lines.append(f"Structure: {row['structure']}")
    if row.get("battery_role"):
        lines.append(f"Battery role: {row['battery_role']}")
    if row.get("sites"):
        lines.append(f"Sites: {row['sites']}")
    return "\n".join(lines)


def verbalize_crystal(row: dict) -> str:
    return str(row.get("structure") or f"Crystal structure of {row['formula']}")


def numeric_vector(row: dict) -> list[float]:
    out = []
    for name in FEATURE_NAMES:
        val = row.get(name)
        out.append(0.0 if val is None else float(val))
    return out


def run(entities_path: Path, out_dir: Path) -> dict:
    with entities_path.open(encoding="utf-8") as f:
        bundle = json.load(f)
    verbalizations = {}
    crystals = {}
    numeric = {}
    for row in bundle["materials"]:
        mid = row["mp_id"]
        verbalizations[mid] = verbalize_material(row)
        crystals[mid] = verbalize_crystal(row)
        numeric[mid] = dict(zip(FEATURE_NAMES, numeric_vector(row)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "verbalizations.json").open("w", encoding="utf-8") as f:
        json.dump(verbalizations, f, indent=2)
    with (out_dir / "crystal_verbalizations.json").open("w", encoding="utf-8") as f:
        json.dump(crystals, f, indent=2)
    with (out_dir / "numeric_features.json").open("w", encoding="utf-8") as f:
        json.dump({"feature_names": FEATURE_NAMES, "materials": numeric}, f, indent=2)
    return {"n": len(verbalizations), "out_dir": str(out_dir)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entities", type=Path, default=ABOX_DIR / "entities.json")
    p.add_argument("--out-dir", type=Path, default=ABOX_DIR)
    args = p.parse_args()
    info = run(args.entities, args.out_dir)
    print(f"Wrote {info['n']} verbalizations → {info['out_dir']}")


if __name__ == "__main__":
    main()
