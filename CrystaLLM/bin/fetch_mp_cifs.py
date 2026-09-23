#!/usr/bin/env python3
"""
fetch_mp_cifs.py — Download official Materials Project CIFs for battery-KG mp-ids
==============================================================================

Reads MP_API_KEY from the project .env (gitignored). Does not reconstruct
structures from HeteroData.

Usage
-----
  python bin/fetch_mp_cifs.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRYSTALLM = Path(__file__).resolve().parents[1]
if str(CRYSTALLM) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM))

from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter

from crystallm import add_atomic_props_block


def load_api_key(env_path: Path) -> str:
    if not env_path.exists():
        raise SystemExit(f"Missing {env_path}. Put MP_API_KEY=... there.")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("MP_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("MP_API_KEY not found in .env")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hetero-dir", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=CRYSTALLM / "kg_finetune_cifs")
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--max-materials", type=int, default=None)
    args = p.parse_args()

    key = load_api_key(args.env)
    meta = json.loads((args.hetero_dir / "node_mapping.json").read_text(encoding="utf-8"))
    materials = [m for m in meta["Material"] if str(m.get("material_id", "")).startswith("mp-")]
    if args.max_materials:
        materials = materials[: args.max_materials]
    id_to_formula = {m["material_id"]: (m.get("formula") or "").replace(" ", "") for m in materials}
    mp_ids = list(id_to_formula)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(mp_ids)} MP CIFs → {args.out_dir}", flush=True)

    n_ok = n_fail = 0
    with MPRester(key) as mpr:
        for i in range(0, len(mp_ids), args.chunk):
            chunk = mp_ids[i : i + args.chunk]
            try:
                docs = mpr.get_summary(
                    {"material_ids": ",".join(chunk)},
                    fields=["material_id", "structure"],
                )
            except Exception as exc:
                print(f"  chunk {i} failed ({exc}); retrying one-by-one")
                docs = []
                for mid in chunk:
                    try:
                        docs.append(mpr.get_summary_by_material_id(mid, fields=["material_id", "structure"]))
                    except Exception as e2:
                        print(f"    skip {mid}: {e2}")
                        n_fail += 1
            got = {}
            if docs and len(docs) == len(chunk):
                # MP now returns new-style ids (mp-aaaabcdl) for old queries (mp-19017).
                # Batch order matches the requested ids.
                for mid, d in zip(chunk, docs):
                    if d.get("structure") is not None:
                        got[mid] = d["structure"]
            else:
                for d in docs or []:
                    mid_key = str(d.get("material_id") or "")
                    if mid_key in id_to_formula and d.get("structure") is not None:
                        got[mid_key] = d["structure"]
                missing = [mid for mid in chunk if mid not in got]
                for mid in missing:
                    try:
                        st = mpr.get_structure_by_material_id(mid)
                        got[mid] = st
                    except Exception as e2:
                        print(f"    skip {mid}: {e2}", flush=True)
            if i == 0:
                print(f"  debug: got {len(got)}/{len(chunk)} in first chunk", flush=True)
            for mid in chunk:
                st = got.get(mid)
                if st is None:
                    n_fail += 1
                    continue
                formula = id_to_formula.get(mid) or getattr(st.composition, "reduced_formula", mid)
                try:
                    cif = str(CifWriter(st, symprec=0.1))
                    cif = add_atomic_props_block(cif)
                    # Keep KG formula in the data_ block so kg_context lookup hits.
                    cif = re.sub(r"^data_\S+", f"data_{formula}", cif, count=1, flags=re.M)
                    if "_chemical_formula_structural" in cif:
                        cif = re.sub(
                            r"_chemical_formula_structural\s+\S+",
                            f"_chemical_formula_structural {formula}",
                            cif,
                            count=1,
                        )
                except Exception as exc:
                    print(f"    CIF write failed {mid}: {exc}")
                    n_fail += 1
                    continue
                fname = f"{safe_name(mid)}_{safe_name(formula)}.cif"
                (args.out_dir / fname).write_text(cif, encoding="utf-8")
                n_ok += 1
            print(f"  {min(i + args.chunk, len(mp_ids))}/{len(mp_ids)}  wrote={n_ok} fail={n_fail}", flush=True)
            time.sleep(0.2)

    print(f"Done. CIFs={n_ok} failed={n_fail} → {args.out_dir}")


if __name__ == "__main__":
    main()
