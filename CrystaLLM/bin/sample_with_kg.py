#!/usr/bin/env python3
"""
sample_with_kg.py — Generate CIFs with HGT KG cross-attention context
=====================================================================

Loads a CrystaLLM checkpoint, enables KG cross-attention layers (random
if not yet fine-tuned), builds kg_context from composition + HGT export,
and generates CIF samples.

Usage
-----
  cd CrystaLLM
  python bin/sample_with_kg.py \\
    out_dir=out_kg_finetune \\
    composition=LiFePO4 \\
    hgt_export=../HGT-Pipeline/battery_hgt/models/battery_v2/export_package \\
    num_samples=10 \\
    max_new_tokens=2000 \\
    target=file
"""

import os
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass, fields

import torch
from omegaconf import OmegaConf

# Ensure crystallm package is importable when run as script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crystallm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
    get_atomic_props_block_for_formula,
    parse_config,
    parse_elements,
)
from crystallm.kg_context import build_kg_context


@dataclass
class SampleWithKGDefaults:
    out_dir: str = "out_kg_finetune"
    composition: str = "LiFePO4"
    hgt_export: str = "../HGT-Pipeline/battery_hgt/models/battery_v2/export_package"
    start: str = ""  # if empty: data_<composition>\\n  (official CrystaLLM default; no space group)
    spacegroup: str = ""  # optional; only added if you pass it
    num_samples: int = 2
    max_new_tokens: int = 2000
    temperature: float = 0.6
    top_k: int = 10
    seed: int = 1337
    device: str = "cpu"
    dtype: str = "float32"
    compile: bool = False
    target: str = "console"  # or "file"
    samples_out: str = ""  # if set with target=file, write CIFs here instead of cwd
    top_kg: int = 8
    require_elements: bool = True  # drop samples that omit formula elements (e.g. missing Fe)
    trim_sites: bool = False  # if true, drop trailing malformed site rows


def build_composition_prompt(composition: str, spacegroup: str = "") -> str:
    """Official CrystaLLM default is data_<formula>\\n — no space group.

    Space group is only included if you pass spacegroup=... (same as
    make_prompt_file.py --spacegroup).
    """
    if not spacegroup:
        return f"data_{composition}\n"
    try:
        block = get_atomic_props_block_for_formula(composition)
    except Exception:
        block = ""
    lines = [f"data_{composition}"]
    if block:
        lines.append(block.rstrip())
    lines.append(f"_symmetry_space_group_name_H-M {spacegroup}")
    return "\n".join(lines) + "\n"


SITE_ROW = re.compile(
    r"^([A-Z][a-z]?)\s+\S+\s+\d+\s+"
    r"[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*\s+"
    r"[-+]?\d+\.?\d*\s*$"
)


def trim_to_valid_sites(cif_text: str) -> str:
    """Keep CIF header + consecutive well-formed atom-site rows; drop trailing junk."""
    lines = cif_text.splitlines()
    occ_i = None
    for i, line in enumerate(lines):
        if line.strip().startswith("_atom_site_occupancy"):
            occ_i = i
            break
    if occ_i is None:
        return cif_text.rstrip() + "\n"
    kept = lines[: occ_i + 1]
    for line in lines[occ_i + 1 :]:
        s = line.strip()
        if not s:
            break
        if not SITE_ROW.match(s):
            break
        kept.append(line.rstrip())
    return "\n".join(kept) + "\n"


def missing_formula_elements(cif_text: str, composition: str) -> list[str]:
    """Elements required by the formula that never appear as atom-site symbols."""
    needed = parse_elements(composition)
    if "_atom_site_type_symbol" not in cif_text and "_atom_site_label" not in cif_text:
        return needed  # incomplete CIF — no site loop yet
    site_els: set[str] = set()
    in_sites = False
    for line in cif_text.splitlines():
        s = line.strip()
        if re.match(r"_atom_site_(type_symbol|label|occupancy)", s):
            in_sites = True
            continue
        if not in_sites:
            continue
        if not s or s.startswith("loop_") or (s.startswith("_") and not s.startswith("_atom")):
            break
        tok = s.split()[0]
        m = re.match(r"[A-Z][a-z]?", tok)
        if m:
            site_els.add(m.group(0))
    return [el for el in needed if el not in site_els]


def load_model_with_kg(out_dir: str, kg_dim: int, device: str) -> GPT:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    allowed = {f.name for f in fields(GPTConfig)}
    model_args = {k: v for k, v in dict(checkpoint["model_args"]).items() if k in allowed}
    sd = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(unwanted_prefix):
            sd[k[len(unwanted_prefix):]] = sd.pop(k)

    saved = dict(checkpoint.get("model_args") or {})
    prefix = bool(saved.get("use_kg_prefix")) or (
        "kg_proj.weight" in sd and not any("cross_attn" in k for k in sd)
    )
    ca = bool(saved.get("use_kg_cross_attn")) or any("cross_attn" in k for k in sd)
    if prefix and not ca:
        model_args["use_kg_prefix"] = True
        model_args["use_kg_cross_attn"] = False
    else:
        model_args["use_kg_cross_attn"] = True
        model_args["use_kg_prefix"] = False
    model_args["kg_dim"] = kg_dim

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded KG model from {ckpt_path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    if any("kg_proj" in k or "cross_attn" in k for k in missing):
        print("  WARNING: KG params missing — run bin/finetune_with_kg.py first.")
    if model.use_kg_prefix:
        print("  mode: soft prefix (HGT added to first N token embeddings)")
    else:
        n_ca = sum(1 for b in model.transformer.h if b.use_kg_cross_attn)
        print(f"  mode: ungated cross-attention in {n_ca}/{model.config.n_layer} blocks")
    return model


def main():
    C = parse_config(SampleWithKGDefaults)
    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    torch.manual_seed(C.seed)
    device_type = "cuda" if "cuda" in C.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    hgt_export = C.hgt_export
    if not os.path.isabs(hgt_export):
        hgt_export = os.path.abspath(os.path.join(os.getcwd(), hgt_export))

    kg_raw, matched, kg_dim, type_ids = build_kg_context(
        hgt_export, C.composition, device=C.device, top_k=C.top_kg
    )
    print(f"Composition: {C.composition}")
    print(f"KG matches ({len(matched)}): {matched}")
    print(f"kg_raw shape: {tuple(kg_raw.shape)}  kg_dim={kg_dim}")

    model = load_model_with_kg(C.out_dir, kg_dim=kg_dim, device=C.device)
    model.eval()
    model.to(C.device)
    if C.compile:
        model = torch.compile(model)

    tokenizer = CIFTokenizer()
    if C.start.strip():
        prompt = C.start.strip()
    else:
        prompt = build_composition_prompt(C.composition, C.spacegroup.strip())
    if prompt.startswith("FILE:"):
        with open(prompt[5:], encoding="utf-8") as f:
            prompt = f.read()
    print("Prompt:")
    print(prompt)
    print("---")

    start_ids = tokenizer.encode(tokenizer.tokenize_cif(prompt))
    x = torch.tensor(start_ids, dtype=torch.long, device=C.device)[None, ...]

    written = 0
    attempts = 0
    max_attempts = C.num_samples * 8 if C.require_elements else C.num_samples
    with torch.no_grad():
        with ctx:
            while written < C.num_samples and attempts < max_attempts:
                attempts += 1
                y = model.generate(
                    x,
                    C.max_new_tokens,
                    temperature=C.temperature,
                    top_k=C.top_k,
                    kg_context=kg_raw,
                    kg_types=type_ids,
                )
                generated = tokenizer.decode(y[0].tolist())
                if C.trim_sites:
                    generated = trim_to_valid_sites(generated)
                missing = missing_formula_elements(generated, C.composition) if C.require_elements else []
                if missing:
                    print(f"  attempt {attempts}: dropped {missing} — resampling")
                    continue
                if C.target == "console":
                    print(generated)
                    print("---------------")
                else:
                    out_folder = C.samples_out.strip() or "."
                    os.makedirs(out_folder, exist_ok=True)
                    fname = os.path.join(out_folder, f"sample_kg_{written + 1}.cif")
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(generated)
                    print(f"wrote {fname}")
                written += 1
    if written < C.num_samples:
        print(f"Only {written}/{C.num_samples} samples kept the full formula. "
              "Try more attempts, a stronger prompt (start=FILE:...), or require_elements=false.")


if __name__ == "__main__":
    main()
