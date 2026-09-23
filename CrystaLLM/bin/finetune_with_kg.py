#!/usr/bin/env python3
"""
finetune_with_kg.py — Train ungated KG cross-attention from vanilla CrystaLLM
=============================================================================

Correct recipe: start from the **base** checkpoint (crystallm_v1_small),
add new ungated CA in all layers, train CA together with the last 2 GPT
blocks. August CA (`out_kg_finetune`) is kept as a frozen comparison; this
run writes a new folder.

Usage
-----
  python bin/finetune_with_kg.py \\
    out_dir=crystallm_v1_small \\
    save_dir=out_kg_ca_adapt \\
    cif_dir=kg_finetune_cifs \\
    hgt_export=../HGT-Pipeline/battery_hgt/models/battery_v2/export_package \\
    kg_mode=ca \\
    fresh_kg=true \\
    freeze_kg=false \\
    unfreeze_last_n=2 \\
    max_cifs=1000 \\
    max_iters=2000 \\
    kg_lr=1e-4 \\
    learning_rate=1e-5 \\
    device=mps
"""

import glob
import os
import random
import re
import sys
from dataclasses import dataclass, fields

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crystallm import CIFTokenizer, GPT, GPTConfig, parse_config, parse_elements, round_numbers
from crystallm.kg_context import KGContextBuilder


@dataclass
class FinetuneDefaults:
    out_dir: str = "crystallm_v1_small"
    hgt_export: str = "../HGT-Pipeline/battery_hgt/models/battery_v2/export_package"
    cif_dir: str = ""
    save_dir: str = "out_kg_ca_adapt"
    max_iters: int = 2000
    learning_rate: float = 1e-5  # last GPT blocks
    kg_lr: float = 1e-4  # new CA / kg_proj
    weight_decay: float = 0.01
    batch_size: int = 1
    block_size: int = 512
    seed: int = 1337
    device: str = "cpu"
    top_kg: int = 8
    max_cifs: int = 1000
    fresh_kg: bool = True
    kg_dropout: float = 0.1
    kg_drop_p: float = 0.1
    kg_mode: str = "ca"  # "ca" (main) or "prefix" (ablation)
    freeze_kg: bool = False  # train the new CA layers
    unfreeze_last_n: int = 2  # last GPT blocks (self-attn + MLP; CA is separate)
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 20
    val_frac: float = 0.05


def extract_composition_from_cif(text: str, path: str = "") -> str | None:
    m = re.search(r"_chemical_formula_structural\s+(\S+)", text)
    if m:
        return m.group(1).strip("'\"")
    m = re.search(
        r"_chemical_formula_sum\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))",
        text,
    )
    if m:
        return (m.group(1) or m.group(2) or m.group(3)).replace(" ", "")
    m = re.search(r"data_([A-Za-z0-9()]+)", text)
    if m:
        name = m.group(1)
        if not re.match(r"(?i)image\d*$", name):
            return name
    if path:
        base = os.path.basename(path)
        m = re.search(r"mp-\d+_(.+)\.cif$", base, re.I)
        if m:
            return m.group(1).replace("_", "")
    return None


def list_cif_paths(cif_dir: str, max_cifs: int, seed: int) -> list[str]:
    paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    prefer = []
    rest = []
    for p in paths:
        name = os.path.basename(p).lower()
        if any(s in name for s in ("lifepo4", "licoo2", "limn2o4", "linio2", "nafepo4")):
            prefer.append(p)
        else:
            rest.append(p)
    rng = random.Random(seed)
    rng.shuffle(rest)
    chosen = prefer + rest
    if max_cifs and len(chosen) > max_cifs:
        chosen = chosen[:max_cifs]
    return chosen


def _normalize_cif_format(text: str) -> str:
    """Match vanilla CrystaLLM CIF number style (4 decimals; occupancy 1.0 → 1)."""
    text = round_numbers(text, decimal_places=4)
    text = re.sub(r'(?<!\d)(1)\.0+(?!\d)', r'\1', text)
    return text


def encode_cif(path: str, tokenizer: CIFTokenizer):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if len(text) > 80_000:
        return None
    composition = extract_composition_from_cif(text, path)
    if not composition:
        return None
    try:
        text = _normalize_cif_format(text)
        ids = tokenizer.encode(tokenizer.tokenize_cif(text))
    except Exception:
        return None
    if len(ids) < 8:
        return None
    return {"ids": ids, "composition": composition, "path": path}


def crop_xy(ids: list[int], block_size: int, rng: random.Random | None = None, from_start: bool = False):
    if len(ids) < 2:
        return None, None
    if len(ids) > block_size + 1:
        if from_start or rng is None:
            ids = ids[: block_size + 1]
        else:
            start = rng.randint(0, len(ids) - block_size - 1)
            ids = ids[start : start + block_size + 1]
    x = ids[:-1]
    y = ids[1:]
    if len(x) > block_size:
        x = x[:block_size]
        y = y[:block_size]
    return x, y


def is_kg_param(name: str) -> bool:
    return (
        "cross_attn" in name
        or ".ln_x." in name
        or name.startswith("kg_proj")
        or name == "kg_null"
        or name == "kg_prefix_scale"
    )


def is_trainable(name: str, n_layer: int, unfreeze_last_n: int, train_kg: bool) -> bool:
    """August CA stays frozen unless train_kg. Last N blocks' self-attn/MLP can adapt."""
    if train_kg and is_kg_param(name):
        return True
    if unfreeze_last_n <= 0:
        return False
    if is_kg_param(name):
        return False
    last_start = n_layer - unfreeze_last_n
    if name.startswith("transformer.ln_f"):
        return True
    m = re.match(r"transformer\.h\.(\d+)\.", name)
    if m and int(m.group(1)) >= last_start:
        return True
    return False


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
        m = re.match(r"[A-Z][a-z]?", s.split()[0] if s.split() else "")
        if m:
            els.add(m.group(0))
    return els


@torch.no_grad()
def gen_probe(model, tokenizer, builder, device, top_kg, prompt: str, composition: str,
              n_samples: int, max_new: int, seed: int) -> dict:
    model.eval()
    ids = tokenizer.encode(tokenizer.tokenize_cif(prompt))
    kg_raw, _, type_ids = builder.build(composition, top_k=top_kg)
    needed = parse_elements(composition)
    sites = []
    full = 0
    for i in range(n_samples):
        torch.manual_seed(seed + i)
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
        y = model.generate(
            x, max_new, temperature=0.6, top_k=10,
            kg_context=kg_raw, kg_types=type_ids,
        )
        text = tokenizer.decode(y[0].tolist())
        n_site = count_site_rows(text)
        sites.append(n_site)
        els = site_elements(text)
        if all(el in els for el in needed):
            full += 1
    model.train()
    return {
        "avg_sites": sum(sites) / max(len(sites), 1),
        "full_formula": full,
        "n": n_samples,
        "sites": sites,
    }


@torch.no_grad()
def estimate_loss(model, samples, builder, block_size, device, top_kg, n_iters, seed):
    model.eval()
    rng = random.Random(seed)
    losses = []
    for _ in range(n_iters):
        sample = rng.choice(samples)
        x_ids, y_ids = crop_xy(sample["ids"], block_size, rng=None, from_start=True)
        if x_ids is None:
            continue
        x = torch.tensor(x_ids, dtype=torch.long, device=device)[None, ...]
        y = torch.tensor(y_ids, dtype=torch.long, device=device)[None, ...]
        kg_raw, _, type_ids = builder.build(sample["composition"], top_k=top_kg)
        _, loss = model(x, targets=y, kg_context=kg_raw, kg_types=type_ids)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_ckpt(path: str, model: GPT, model_args: dict, it: int, cfg_dump):
    torch.save({
        "model": model.state_dict(),
        "model_args": model_args,
        "iter_num": int(it),
        "best_val_loss": None,
        "config": cfg_dump,
    }, path)


def main():
    C = parse_config(FinetuneDefaults)
    print(OmegaConf.to_yaml(C))
    torch.manual_seed(C.seed)
    rng = random.Random(C.seed)

    hgt_export = C.hgt_export
    if not os.path.isabs(hgt_export):
        hgt_export = os.path.abspath(os.path.join(os.getcwd(), hgt_export))

    builder = KGContextBuilder(hgt_export, device=C.device)
    kg_dim = builder.kg_dim
    kg_mode = str(C.kg_mode).strip().lower()
    if kg_mode not in ("prefix", "ca"):
        print("kg_mode must be prefix or ca", file=sys.stderr)
        sys.exit(1)
    print(f"kg_mode={kg_mode}")

    ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
    vanilla = os.path.join("crystallm_v1_small", "ckpt.pt")
    august = os.path.join("out_kg_finetune", "ckpt.pt")
    if C.fresh_kg and os.path.exists(vanilla):
        ckpt_path = vanilla
        print(f"fresh_kg=true → start from vanilla CrystaLLM: {ckpt_path}")
    elif os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
    elif os.path.exists(august):
        ckpt_path = august
        print(f"Using August CA: {ckpt_path}")
    elif os.path.exists(vanilla):
        ckpt_path = vanilla
        print(f"Using base checkpoint: {ckpt_path}")
    else:
        print(f"Missing checkpoint: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    allowed = {f.name for f in fields(GPTConfig)}
    model_args = {k: v for k, v in dict(checkpoint["model_args"]).items() if k in allowed}
    model_args["use_kg_cross_attn"] = kg_mode == "ca"
    model_args["use_kg_prefix"] = kg_mode == "prefix"
    model_args["kg_dim"] = kg_dim
    model_args["kg_dropout"] = float(C.kg_dropout)
    model = GPT(GPTConfig(**model_args))

    state_dict = checkpoint["model"]
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    kg_new = [k for k in missing if is_kg_param(k) or k.startswith("kg_")]
    print(f"loaded weights from {ckpt_path}; missing KG params: {len(kg_new)}  unexpected: {len(unexpected)}")
    if kg_mode == "ca" and kg_new:
        print("WARNING: CA weights missing — this is not the August checkpoint.", file=sys.stderr)
    del checkpoint, state_dict
    model.to(C.device)

    n_layer = int(model.config.n_layer)
    train_kg = not bool(C.freeze_kg)
    unfreeze_last_n = int(C.unfreeze_last_n)
    for name, p in model.named_parameters():
        p.requires_grad = is_trainable(name, n_layer, unfreeze_last_n, train_kg)
    trained_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"freeze_kg={not train_kg}  unfreeze_last_n={unfreeze_last_n}  "
        f"trainable params: {sum(p.numel() for p in trainable):,}"
    )
    if not trainable:
        print("Nothing to train. Set unfreeze_last_n>0 and/or freeze_kg=false.", file=sys.stderr)
        sys.exit(1)
    layers = sorted({int(m.group(1)) for n in trained_names if (m := re.match(r"transformer\.h\.(\d+)\.", n))})
    print(f"  unfrozen GPT layers: {layers}")
    kg_params = [p for n, p in model.named_parameters() if p.requires_grad and is_kg_param(n)]
    gpt_params = [p for n, p in model.named_parameters() if p.requires_grad and not is_kg_param(n)]
    groups = []
    if kg_params:
        groups.append({"params": kg_params, "lr": float(C.kg_lr)})
        print(f"  CA/kg_proj params: {sum(p.numel() for p in kg_params):,} @ {C.kg_lr:.2e}")
    if gpt_params:
        groups.append({"params": gpt_params, "lr": float(C.learning_rate)})
        print(f"  GPT params: {sum(p.numel() for p in gpt_params):,} @ {C.learning_rate:.2e}")
    optimizer = torch.optim.AdamW(groups, weight_decay=C.weight_decay)

    tokenizer = CIFTokenizer()
    cif_dir = C.cif_dir
    if not cif_dir:
        print("Set cif_dir=... to a folder of KG-aligned .cif files.", file=sys.stderr)
        sys.exit(1)
    paths = list_cif_paths(cif_dir, C.max_cifs, C.seed)
    if not paths:
        print(f"No CIFs in {cif_dir}", file=sys.stderr)
        sys.exit(1)

    samples = []
    skipped = 0
    for i, path in enumerate(paths, 1):
        sample = encode_cif(path, tokenizer)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
        if i % 500 == 0:
            print(f"  tokenized {i}/{len(paths)}")
    if len(samples) < 8:
        print(f"Too few usable CIFs ({len(samples)}). Check cif_dir.", file=sys.stderr)
        sys.exit(1)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * C.val_frac))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:] or samples
    print(f"Using {len(train_samples)} train / {len(val_samples)} val CIFs "
          f"(skipped {skipped})")

    os.makedirs(C.save_dir, exist_ok=True)
    try:
        cfg_dump = OmegaConf.to_container(C, resolve=True)
    except Exception:
        cfg_dump = {"max_iters": int(C.max_iters), "learning_rate": float(C.learning_rate)}

    protected = {
        os.path.abspath("out_kg_finetune"),
        os.path.abspath("crystallm_v1_small"),
    }
    if os.path.abspath(C.save_dir) in protected:
        print(f"Refusing to overwrite {C.save_dir}. Use save_dir=out_kg_ca_adapt", file=sys.stderr)
        sys.exit(1)

    model.train()
    for it in range(1, C.max_iters + 1):
        sample = rng.choice(train_samples)
        x_ids, y_ids = crop_xy(
            sample["ids"], C.block_size, rng=rng, from_start=(kg_mode == "prefix"),
        )
        if x_ids is None:
            continue
        x = torch.tensor(x_ids, dtype=torch.long, device=C.device)[None, ...]
        y = torch.tensor(y_ids, dtype=torch.long, device=C.device)[None, ...]

        kg_raw, names, type_ids = builder.build(sample["composition"], top_k=C.top_kg)
        if rng.random() < C.kg_drop_p:
            # Train "no KG" as kg_null through kg_proj (skipping would waste the step
            # because the frozen LM loss has no gradient).
            n_drop = kg_raw.size(1) if kg_raw is not None else int(C.top_kg)
            kg_raw = model.kg_null.expand(1, n_drop, -1)
            type_ids = None
            names = ["<drop>"]
        _, loss = model(x, targets=y, kg_context=kg_raw, kg_types=type_ids)

        if not loss.requires_grad:
            if it % 100 == 0 or it == 1:
                print(
                    f"iter {it:5d}  loss {float(loss):.4f}  lr {C.learning_rate:.2e}  "
                    f"comp={sample['composition']}  kg={names}"
                )
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if C.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, C.grad_clip)
        optimizer.step()

        if it % 100 == 0 or it == 1:
            print(
                f"iter {it:5d}  loss {loss.item():.4f}  kg_lr {C.kg_lr:.2e} gpt_lr {C.learning_rate:.2e}  "
                f"comp={sample['composition']}  kg={names}"
                + (
                    f"  ||kg_proj||={model.kg_proj.weight.detach().norm().cpu():.4f}"
                    if model.kg_proj is not None else ""
                )
            )

        if it % C.eval_interval == 0 or it == C.max_iters:
            val_loss = estimate_loss(
                model, val_samples, builder, C.block_size, C.device,
                C.top_kg, C.eval_iters, C.seed + it,
            )
            short = gen_probe(
                model, tokenizer, builder, C.device, C.top_kg,
                "data_LiFePO4\n", "LiFePO4", n_samples=1, max_new=300, seed=C.seed + it,
            )
            lifepo4_text = ""
            lifepo4_path = os.path.join(os.getcwd(), "prompt_lifepo4.txt")
            if os.path.exists(lifepo4_path):
                lifepo4_text = open(lifepo4_path, encoding="utf-8").read()
            rich = {"avg_sites": 0.0, "full_formula": 0, "n": 0, "sites": []}
            if lifepo4_text:
                rich = gen_probe(
                    model, tokenizer, builder, C.device, C.top_kg,
                    lifepo4_text, "LiFePO4", n_samples=1, max_new=200, seed=C.seed + it,
                )
            print(
                f"  val_loss {val_loss:.4f}  "
                f"short_sites={short['avg_sites']:.1f} full={short['full_formula']}/{short['n']}  "
                f"rich_sites={rich['avg_sites']:.1f} full={rich['full_formula']}/{rich['n']}  "
                f"{rich.get('sites', [])}"
            )
            save_ckpt(os.path.join(C.save_dir, "ckpt_last.pt"), model, model_args, it, cfg_dump)
            keep = (short["avg_sites"] > 0) or (rich["avg_sites"] > 0)
            if keep:
                save_ckpt(os.path.join(C.save_dir, "ckpt.pt"), model, model_args, it, cfg_dump)
                print(f"  saved → {os.path.join(C.save_dir, 'ckpt.pt')}")
            else:
                print("  skip ckpt.pt (0 site rows on both probes)")
            continue

    print(f"Done. ckpt={os.path.join(C.save_dir, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
