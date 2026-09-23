"""Probe: does MiniLM actually separate numbers written into V(a)?

Small data in OnT/data/numeric_probe/. Encodes the §7 template with the same
MiniLM OnT uses (mean pooling, no extra Normalize layer). Compares:

  v  = encode(V(a))          # text channel (OnT)
  x  = number list           # the concat we would append at export
  z  = concat(v_hat, x_hat)  # preview of §8B, equal-weight unit vectors

Does not train. Hierarchy/exist losses never supervise floats, so this is the
honest test of "are numbers learned in the extension" before we append x.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PROBE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "numeric_probe"
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

# Cosine below this => the encoder moved. Nearby floats almost never do.
NEAR_PASS = 0.995
FAR_PASS = 0.97
FORMULA_PASS = 0.95


def verbalize(row: dict) -> str:
    """Structured V(a) from ONT_ABOX_EXTENSION.md §7. Missing clauses omitted."""
    metal = "true" if row["is_metal_flag"] else "false"
    stable = "true" if row["is_stable_flag"] else "false"
    return "\n".join(
        [
            f"Material: {row['formula']} ({row['mp_id']})",
            f"Type: {row['type']}",
            f"Formula: {row['formula']}",
            f"Chemical system: {row['chemsys']}",
            f"Metallic: {metal}",
            f"Stable: {stable}",
            f"Band gap: {row['band_gap_eV']:.2f} eV",
            f"Formation energy: {row['formation_energy_eV_atom']:.2f} eV/atom",
            f"Energy above hull: {row['energy_above_hull_eV_atom']:.2f} eV/atom",
            f"Bulk modulus: {row['bulk_modulus_GPa']:.1f} GPa",
            f"Shear modulus: {row['shear_modulus_GPa']:.1f} GPa",
            f"Structure: {row['structure']}",
            f"Sites: {row['sites']}",
        ]
    )


def feature_vector(row: dict, names: List[str]) -> List[float]:
    return [float(row[n]) for n in names]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_sentences(sentences: List[str], device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
        pooled = mean_pool(out.last_hidden_state, encoded["attention_mask"])
    return pooled.cpu().numpy()


def cosine_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.clip(norms, 1e-12, None)
    return unit @ unit.T


def unit_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def tokenize_numbers(tokenizer) -> Dict[str, List[str]]:
    samples = ["0.00", "0.01", "0.35", "1.34", "3.50", "8.00", "Band gap: 0.00 eV", "Band gap: 8.00 eV"]
    out = {}
    for s in samples:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        out[s] = tokenizer.convert_ids_to_tokens(ids)
    return out


def pair_cosine(sim: np.ndarray, ids: List[str], a: str, b: str) -> float:
    i, j = ids.index(a), ids.index(b)
    return float(sim[i, j])


def main() -> None:
    probe_dir = os.path.abspath(PROBE_DIR)
    with open(os.path.join(probe_dir, "materials.json")) as f:
        payload = json.load(f)

    names: List[str] = payload["feature_names"]
    rows: List[dict] = payload["materials"]
    ids = [r["id"] for r in rows]
    sentences = [verbalize(r) for r in rows]
    X = np.array([feature_vector(r, names) for r in rows], dtype=np.float64)

    verbalizations = {r["id"]: verbalize(r) for r in rows}
    numeric_features = {
        r["id"]: {n: float(r[n]) for n in names} for r in rows
    }
    with open(os.path.join(probe_dir, "verbalizations.json"), "w") as f:
        json.dump(verbalizations, f, indent=2)
    with open(os.path.join(probe_dir, "numeric_features.json"), "w") as f:
        json.dump(numeric_features, f, indent=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"encoder: {MODEL_NAME}  device: {device}")
    print(f"n_materials: {len(rows)}  text_dim: 384  x_dim: {len(names)}")
    print()
    print("--- V(a) for synth-LiMgP-bg0.00 ---")
    print(verbalizations["synth-LiMgP-bg0.00"])
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenize_numbers(tokenizer)
    print("--- MiniLM tokens for numbers ---")
    for s, toks in tokens.items():
        print(f"  {s!r:28s} → {toks}")
    print()

    V = encode_sentences(sentences, device)
    sim_v = cosine_matrix(V)
    sim_x = cosine_matrix(X)
    # Naive concat: unit text + unit raw x. Bulk modulus ~55 drowns band gap.
    Z_naive = np.concatenate([unit_rows(V), unit_rows(X)], axis=1)
    sim_z_naive = cosine_matrix(Z_naive)

    # §8B append: per-feature z-score, then concat. Constant features (std=0) → 0.
    std = X.std(axis=0)
    mean = X.mean(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    Xz = (X - mean) / std_safe
    Xz[:, std < 1e-12] = 0.0
    sim_x_z = cosine_matrix(Xz)
    # Equal-weight concat so 7 numeric dims are not drowned by 384 text dims.
    Z = np.concatenate([unit_rows(V), unit_rows(Xz)], axis=1)
    sim_z = cosine_matrix(Z)

    pairs = [
        ("synth-LiMgP-bg0.00", "synth-LiMgP-bg0.01", "near  +0.01 eV"),
        ("synth-LiMgP-bg0.00", "synth-LiMgP-bg0.35", "near  +0.35 eV"),
        ("synth-LiMgP-bg0.00", "synth-LiMgP-bg1.34", "mid   +1.34 eV"),
        ("synth-LiMgP-bg0.00", "synth-LiMgP-bg3.50", "far   +3.50 eV"),
        ("synth-LiMgP-bg0.00", "synth-LiMgP-bg8.00", "far   +8.00 eV"),
        ("synth-LiMgP-bg0.00", "ctrl-LiZnP-same-numbers", "formula only (same x)"),
        ("real-mp-10178-LiMgP", "real-mp-10182-LiZnP", "real LiMgP vs LiZnP"),
    ]

    print(
        f"{'pair':<42s} {'Δbg':>6s}  {'cos(v)':>8s}  {'cos(x)':>8s}  "
        f"{'cos(x_z)':>9s}  {'cos(z)':>8s}"
    )
    table = []
    for a, b, note in pairs:
        ia, ib = ids.index(a), ids.index(b)
        dbg = abs(rows[ia]["band_gap_eV"] - rows[ib]["band_gap_eV"])
        row = {
            "a": a,
            "b": b,
            "note": note,
            "delta_band_gap": dbg,
            "cosine_v": float(sim_v[ia, ib]),
            "cosine_x_raw": float(sim_x[ia, ib]),
            "cosine_x_zscore": float(sim_x_z[ia, ib]),
            "cosine_z_naive": float(sim_z_naive[ia, ib]),
            "cosine_z": float(sim_z[ia, ib]),
        }
        table.append(row)
        print(
            f"{note:<42s} {dbg:6.2f}  {row['cosine_v']:8.4f}  "
            f"{row['cosine_x_raw']:8.4f}  {row['cosine_x_zscore']:9.4f}  "
            f"{row['cosine_z']:8.4f}"
        )
    print()

    # Rank test: among synthetic band-gap clones, is nearest neighbour the closest gap?
    synth = [r for r in rows if r["kind"] == "synthetic_band_gap"]
    synth_ids = [r["id"] for r in synth]
    synth_idx = [ids.index(i) for i in synth_ids]
    gaps = np.array([r["band_gap_eV"] for r in synth])
    rank_ok = 0
    rank_z_ok = 0
    rank_rows = []
    for k, i in enumerate(synth_idx):
        others = [j for j in range(len(synth_idx)) if j != k]
        nn_v = others[int(np.argmax([sim_v[i, synth_idx[j]] for j in others]))]
        nn_x = others[int(np.argmax([sim_x_z[i, synth_idx[j]] for j in others]))]
        nn_z = others[int(np.argmax([sim_z[i, synth_idx[j]] for j in others]))]
        true_nn = others[int(np.argmin(np.abs(gaps[others] - gaps[k])))]
        ok = nn_v == true_nn
        ok_z = nn_z == true_nn
        rank_ok += int(ok)
        rank_z_ok += int(ok_z)
        rank_rows.append(
            {
                "id": synth_ids[k],
                "true_nearest_gap": synth_ids[true_nn],
                "nearest_by_v": synth_ids[nn_v],
                "nearest_by_x_zscore": synth_ids[nn_x],
                "nearest_by_z": synth_ids[nn_z],
                "v_matches_true": ok,
                "z_matches_true": nn_z == true_nn,
            }
        )
        print(
            f"NN {synth_ids[k]:22s}  true={synth_ids[true_nn]:22s}  "
            f"v={synth_ids[nn_v]:22s}  x_z={synth_ids[nn_x]:22s}  "
            f"z={synth_ids[nn_z]:22s}  "
            f"v={'OK' if ok else 'MISS'} z={'OK' if nn_z == true_nn else 'MISS'}"
        )
    print()

    cos_near = pair_cosine(sim_v, ids, "synth-LiMgP-bg0.00", "synth-LiMgP-bg0.01")
    cos_far = pair_cosine(sim_v, ids, "synth-LiMgP-bg0.00", "synth-LiMgP-bg8.00")
    cos_formula = pair_cosine(sim_v, ids, "synth-LiMgP-bg0.00", "ctrl-LiZnP-same-numbers")

    verdict = {
        "text_separates_near_floats": cos_near < NEAR_PASS,
        "text_separates_far_floats": cos_far < FAR_PASS,
        "text_separates_formula": cos_formula < FORMULA_PASS,
        "band_gap_nn_hits_v": f"{rank_ok}/{len(synth)}",
        "band_gap_nn_hits_z": f"{rank_z_ok}/{len(synth)}",
        "need_numeric_concat": not (cos_near < NEAR_PASS and cos_far < FAR_PASS),
        "thresholds": {
            "near_cosine_must_be_below": NEAR_PASS,
            "far_cosine_must_be_below": FAR_PASS,
            "formula_cosine_must_be_below": FORMULA_PASS,
        },
        "observed": {
            "cosine_v_near_0.00_vs_0.01": cos_near,
            "cosine_v_far_0.00_vs_8.00": cos_far,
            "cosine_v_formula_control": cos_formula,
        },
    }
    print("--- verdict ---")
    print(f"  formula control (LiMgP vs LiZnP, same numbers): cos(v)={cos_formula:.4f}  "
          f"{'PASS' if verdict['text_separates_formula'] else 'FAIL'}  (encoder sees words)")
    print(f"  near numbers (0.00 vs 0.01 eV):                  cos(v)={cos_near:.4f}  "
          f"{'PASS' if verdict['text_separates_near_floats'] else 'FAIL'}  (MiniLM should fail)")
    print(f"  far numbers  (0.00 vs 8.00 eV):                  cos(v)={cos_far:.4f}  "
          f"{'PASS' if verdict['text_separates_far_floats'] else 'FAIL'}")
    print(f"  nearest-neighbour by band gap from v:            {rank_ok}/{len(synth)}")
    print(f"  nearest-neighbour by band gap from z=concat(v,x): {rank_z_ok}/{len(synth)}")
    print(f"  append number list (concat z-scored x)?          "
          f"{'YES' if verdict['need_numeric_concat'] else 'optional'}")

    results = {
        "model": MODEL_NAME,
        "device": str(device),
        "feature_names": names,
        "ids": ids,
        "cosine_v": sim_v.tolist(),
        "cosine_x_raw": sim_x.tolist(),
        "cosine_x_zscore": sim_x_z.tolist(),
        "cosine_z_naive": sim_z_naive.tolist(),
        "cosine_z": sim_z.tolist(),
        "pairs": table,
        "band_gap_nn": rank_rows,
        "tokenization": tokens,
        "verdict": verdict,
    }
    out_path = os.path.join(probe_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
