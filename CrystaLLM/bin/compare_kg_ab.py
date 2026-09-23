#!/usr/bin/env python3
"""A/B generate: vanilla CrystaLLM vs ungated KG cross-attention. Same prompt/seed."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import fields

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crystallm import CIFTokenizer, GPT, GPTConfig, parse_elements
from crystallm.kg_context import build_kg_context

SITE_ROW = re.compile(
    r"^([A-Z][a-z]?)\s+\S+\s+\d+\s+"
    r"[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*\s+"
    r"[-+]?\d+\.?\d*\s*$"
)


def count_site_rows(text: str) -> int:
    n = 0
    in_sites = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("_atom_site_occupancy"):
            in_sites = True
            continue
        if not in_sites:
            continue
        if not s:
            break
        if SITE_ROW.match(s):
            n += 1
        else:
            break
    return n


def site_elements(text: str) -> set[str]:
    els = set()
    in_sites = False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"_atom_site_(type_symbol|label|occupancy)", s):
            in_sites = True
            continue
        if not in_sites:
            continue
        if not s or s.startswith("loop_") or (s.startswith("_") and not s.startswith("_atom")):
            break
        m = re.match(r"[A-Z][a-z]?", s.split()[0])
        if m:
            els.add(m.group(0))
    return els


def pymatgen_ok(text: str) -> bool:
    try:
        from pymatgen.io.cif import CifParser
        CifParser.from_str(text).parse_structures(primitive=False)
        return True
    except Exception:
        return False


def strip_prefix(sd):
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    return sd


def load_vanilla(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {f.name for f in fields(GPTConfig)}
    args = {k: v for k, v in dict(ckpt["model_args"]).items() if k in allowed}
    args["use_kg_cross_attn"] = False
    model = GPT(GPTConfig(**args))
    model.load_state_dict(strip_prefix(ckpt["model"]), strict=False)
    model.eval().to(device)
    return model


def load_kg(path, kg_dim, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {f.name for f in fields(GPTConfig)}
    args = {k: v for k, v in dict(ckpt["model_args"]).items() if k in allowed}
    sd = strip_prefix(ckpt["model"])
    saved = dict(ckpt.get("model_args") or {})
    prefix = bool(saved.get("use_kg_prefix")) or (
        "kg_proj.weight" in sd and not any("cross_attn" in k for k in sd)
    )
    ca = bool(saved.get("use_kg_cross_attn")) or any("cross_attn" in k for k in sd)
    if prefix and not ca:
        args["use_kg_prefix"] = True
        args["use_kg_cross_attn"] = False
        print(f"  {path}: soft prefix")
    else:
        args["use_kg_cross_attn"] = True
        args["use_kg_prefix"] = False
        print(f"  {path}: ungated cross-attention, all layers")
    args["kg_dim"] = kg_dim
    if "kg_proj.weight" not in sd:
        print(f"  {path}: no kg_proj (untrained KG layers)")
    model = GPT(GPTConfig(**args))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:6]:
        print("    missing sample:", missing[:6])
    model.eval().to(device)
    return model


def generate(model, x, n_tokens, temperature, top_k, kg_raw=None, kg_types=None):
    return model.generate(
        x, n_tokens, temperature=temperature, top_k=top_k,
        kg_context=kg_raw, kg_types=kg_types,
    )


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    composition = "Li3Mn2NiO6"
    prompt_path = os.path.join(ROOT, "prompt_lino2.txt")
    hgt = os.path.abspath(os.path.join(ROOT, "../HGT-Pipeline/battery_hgt/models/battery_v2/export_package"))
    out_dir = os.path.join(ROOT, "compare_lino2")
    os.makedirs(out_dir, exist_ok=True)

    n_samples = 3
    max_new = 400
    temperature = 0.6
    top_k = 10
    seed = 1337

    with open(prompt_path, encoding="utf-8") as f:
        prompt = f.read()
    tok = CIFTokenizer()
    ids = tok.encode(tok.tokenize_cif(prompt))
    needed = parse_elements(composition)
    print(f"device={device} prompt_tokens={len(ids)} need={needed}")

    kg_raw, names, kg_dim, type_ids = build_kg_context(hgt, composition, device=device)
    print(f"KG {names} shape={tuple(kg_raw.shape)}")

    models = [
        ("vanilla", os.path.join(ROOT, "crystallm_v1_small/ckpt.pt"), False),
        ("august_ca", os.path.join(ROOT, "out_kg_finetune/ckpt.pt"), True),
        ("ca_adapt", os.path.join(ROOT, "out_kg_ca_adapt/ckpt.pt"), True),
    ]

    loaded = []
    for tag, path, use_kg in models:
        if not os.path.exists(path):
            print(f"SKIP {tag}: missing {path}")
            continue
        print(f"Loading {tag}...")
        if use_kg:
            m = load_kg(path, kg_dim, device)
        else:
            m = load_vanilla(path, device)
        loaded.append((tag, m, use_kg))

    rows = []
    for tag, model, use_kg in loaded:
        torch.manual_seed(seed)
        if device == "mps":
            torch.mps.manual_seed(seed)
        parsed = sites = full_el = 0
        for i in range(n_samples):
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            with torch.no_grad():
                y = generate(
                    model, x, max_new, temperature, top_k,
                    kg_raw=kg_raw if use_kg else None,
                    kg_types=type_ids if use_kg else None,
                )
            text = tok.decode(y[0].tolist())
            out_p = os.path.join(out_dir, f"{tag}_{i+1}.cif")
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(text)
            n_site = count_site_rows(text)
            els = site_elements(text)
            missing = [e for e in needed if e not in els]
            ok = pymatgen_ok(text)
            parsed += int(ok)
            sites += n_site
            full_el += int(not missing)
            print(f"  {tag} [{i+1}] parse={ok} sites={n_site} els={sorted(els)} missing={missing} -> {out_p}")
        rows.append((tag, parsed, n_samples, sites / n_samples, full_el, n_samples))

    print("\n===== A/B summary (prompt_lino2.txt, Li3Mn2NiO6, seed=1337, 3 samples) =====")
    print(f"{'model':<12} {'parse':>8} {'full_formula':>14} {'avg_site_rows':>14}")
    for tag, parsed, n, avg_sites, full_el, n2 in rows:
        print(f"{tag:<12} {parsed}/{n:<6} {full_el}/{n2:<6} {avg_sites:14.1f}")

    # Official CrystaLLM prompt — where KG can beat vanilla on formula.
    print("\n===== official short prompt data_LiFePO4 =====")
    short_ids = tok.encode(tok.tokenize_cif("data_LiFePO4\n"))
    kg2, names2, _, type2 = build_kg_context(hgt, "LiFePO4", device=device)
    print(f"KG {names2}")
    need2 = parse_elements("LiFePO4")
    short_dir = os.path.join(ROOT, "compare_short")
    os.makedirs(short_dir, exist_ok=True)
    print(f"{'model':<12} {'parse':>8} {'full_formula':>14} {'avg_site_rows':>14}")
    for tag, model, use_kg in loaded:
        torch.manual_seed(seed)
        if device == "mps":
            torch.mps.manual_seed(seed)
        parsed = sites = full_el = 0
        for i in range(n_samples):
            x = torch.tensor(short_ids, dtype=torch.long, device=device)[None, ...]
            with torch.no_grad():
                y = generate(
                    model, x, 800, temperature, top_k,
                    kg_raw=kg2 if use_kg else None,
                    kg_types=type2 if use_kg else None,
                )
            text = tok.decode(y[0].tolist())
            out_p = os.path.join(short_dir, f"{tag}_{i+1}.cif")
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(text)
            n_site = count_site_rows(text)
            els = site_elements(text)
            missing = [e for e in need2 if e not in els]
            ok = pymatgen_ok(text)
            parsed += int(ok)
            sites += n_site
            full_el += int(not missing)
            print(f"  {tag} [{i+1}] parse={ok} sites={n_site} els={sorted(els)} missing={missing}")
        print(f"{tag:<12} {parsed}/{n_samples:<6} {full_el}/{n_samples:<6} {sites / n_samples:14.1f}")
    # Rich LiFePO4 prompt — same one used in earlier A/B.
    print("\n===== rich prompt_lifepo4.txt =====")
    rich_path = os.path.join(ROOT, "prompt_lifepo4.txt")
    if os.path.exists(rich_path):
        with open(rich_path, encoding="utf-8") as f:
            rich_prompt = f.read()
        rich_ids = tok.encode(tok.tokenize_cif(rich_prompt))
        rich_dir = os.path.join(ROOT, "compare_ab")
        os.makedirs(rich_dir, exist_ok=True)
        print(f"prompt_tokens={len(rich_ids)} KG {names2}")
        print(f"{'model':<12} {'parse':>8} {'full_formula':>14} {'avg_site_rows':>14}")
        for tag, model, use_kg in loaded:
            torch.manual_seed(seed)
            if device == "mps":
                torch.mps.manual_seed(seed)
            parsed = sites = full_el = 0
            for i in range(n_samples):
                x = torch.tensor(rich_ids, dtype=torch.long, device=device)[None, ...]
                with torch.no_grad():
                    y = generate(
                        model, x, 400, temperature, top_k,
                        kg_raw=kg2 if use_kg else None,
                        kg_types=type2 if use_kg else None,
                    )
                text = tok.decode(y[0].tolist())
                out_p = os.path.join(rich_dir, f"{tag}_{i+1}.cif")
                with open(out_p, "w", encoding="utf-8") as f:
                    f.write(text)
                n_site = count_site_rows(text)
                els = site_elements(text)
                missing = [e for e in need2 if e not in els]
                ok = pymatgen_ok(text)
                parsed += int(ok)
                sites += n_site
                full_el += int(not missing)
                print(f"  {tag} [{i+1}] parse={ok} sites={n_site} els={sorted(els)} missing={missing}")
            print(f"{tag:<12} {parsed}/{n_samples:<6} {full_el}/{n_samples:<6} {sites / n_samples:14.1f}")
        print(f"CIFs written under {out_dir}, {short_dir}, and {rich_dir}")
    else:
        print(f"CIFs written under {out_dir} and {short_dir}")


if __name__ == "__main__":
    main()
