# HGT × CrystaLLM — Experiment Study (what was tried, what we got, why it failed)

**Project:** Final Year Project — battery KG + HGT + CrystaLLM CIF generation  
**Updated:** 1 September 2026  
**Status:** Cross-attention integration implemented and measured. Chemistry bias works on one checkpoint; CIF syntax is worse than vanilla. Later adapter variants collapsed generation.

This document is the **study log**: all models, all injection variants, A/B results, and the diagnosis. The pipeline wiring is described in `HGT_CRYSTALLM_INTEGRATION.md`.

---

## 1. Goal

Generate CIF crystal structures for a composition (e.g. LiFePO4), **conditioned on a battery knowledge graph**.

| Piece | Role |
|-------|------|
| Battery KG | Materials, elements, unit cells, properties (~5000 materials) |
| HGT | Link prediction → frozen 32-d node vectors |
| CrystaLLM | Autoregressive CIF language model (closed 371-token vocab) |
| Bridge | Inject HGT vectors into CrystaLLM **without** putting KG text in the prompt |

**Hard constraint:** CrystaLLM’s tokenizer maps unknown text to `<unk>`. KG facts cannot be written into the CIF prompt. Knowledge must enter as **vectors**.

---

## 2. What is *not* comparable: the vanilla base vs our KG runs

The published CrystaLLM checkpoint is **not** “1000 CIFs × 2000 steps.”

| | Vanilla `crystallm_v1_small` | Our KG fine-tunes |
|--|------------------------------|-------------------|
| Who | CrystaLLM authors (2023) | This project |
| Data | Large MP / OQMD / NOMAD CIF corpus | `kg_finetune_cifs` (~1000 files) |
| Training | **100,000** iters, **full GPT** trained | **~2,000** iters, GPT **frozen**, only KG layers |
| Hardware | CUDA (original) | Apple MPS laptop |
| KG | None | HGT `node_embeddings.pt` (dim 32) |

Every KG experiment starts from this frozen vanilla LM and only trains a small bridge. That is why syntax can collapse: we are adding a new residual path into a model that already knows CIF, without retraining the CIF writer.

---

## 3. Models currently on disk

| # | Path | What it is | Saved at | Use |
|---|------|------------|----------|-----|
| 1 | `CrystaLLM/crystallm_v1_small/ckpt.pt` | Vanilla CrystaLLM | iter **100000** | Syntax baseline |
| 2 | `CrystaLLM/out_kg_finetune/ckpt.pt` | August: ungated CA, **all 8 layers** | iter **2000** | **Main KG result** (chemistry) |
| 3 | `CrystaLLM/out_kg_finetune2/ckpt.pt` | Same architecture + occupancy fix + 30% KG-drop + gen probe | iter **1000** (iter 2000 skipped because 0 site rows) | Not fully A/B’d vs vanilla |
| 3b | `CrystaLLM/out_kg_finetune2/ckpt_last.pt` | Same run, last iterate | iter **2000** | Generation collapsed |
| 4 | `CrystaLLM/out_kg_gated/ckpt.pt` | Gated CA, last 2 layers, zero-init `c_proj` | iter **0** | Aborted; identity / untrained |
| — | `HGT-Pipeline/battery_hgt/models/battery_v2/export_package/` | Frozen HGT lookup | Aug 31 12:48 | Shared by all KG checkpoints |

**Not built:** soft / prefix prompting. Discussed only.

**HGT (not a CIF model):** `models/battery_v2/hgt_checkpoint.pt` — 30 epochs, embedding dim 32, 1 layer, 2 heads.

---

## 4. Pipeline that *is* correct

### 4.1 Knowledge graph → HeteroData

`output/hetero_data.pt` (31 Aug 12:47) has **real `.x` features**, not all-zero ID placeholders:

| Node type | `.x` shape | Non-zero fraction |
|-----------|------------|-------------------|
| Element | (83, 7) | 100% |
| AtomicSpecies | (83, 5) | 80% |
| Material | (5000, 11) | 63% |
| UnitCell | (5000, 6) | 100% |
| Crystal | (5000, 1) | 100% |
| Property | (35565, 1) | 94% |

Export `meta.json` lists **all** types as `feature_types` and `id_types: []`. HGT used `.x`. Do **not** rebuild hetero_data for “missing features.”

### 4.2 HGT training quality (weak, not broken)

From `HGT-Pipeline/battery_hgt/outputs/battery_v2/training_summary.json` and `training_log.csv`:

| Epoch | Val MRR | Hits@1 | Hits@10 |
|-------|---------|--------|---------|
| 1 | 0.048 | 0.018 | 0.11 |
| 10 | 0.054 | 0.020 | 0.116 |
| 30 (best) | **0.066** | **0.031** | **0.13** |

Train loss 0.82 → 0.64. The model learned something. Link-ranking is still near-random. Vectors are a **blurry neighbourhood encoding**, not coordinates or occupancy.

### 4.3 Lookup at generation time

For `LiFePO4`, `kg_context.py` stacks (example):

`material:LiFePO4`, `element:Li`, `element:Fe`, `element:P`, `element:O` → tensor `[1, N, 32]`.

This lookup **works**. Embeddings are not zeros; std ≈ 1 after HGT output LN.

### 4.4 August cross-attention (current code)

In every GPT block when `use_kg_cross_attn=True`:

```
x = x + SelfAttn(LN(x))          # frozen
x = x + CrossAttn(LN(x), kg)     # trained; ungated; all 8 layers
x = x + MLP(LN(x))               # frozen
```

- Q = CIF hidden states, K/V = `kg_proj(HGT vectors)` (32 → 512)
- Trainable: `kg_proj`, `cross_attn`, `ln_x`, `kg_null` (~**8.4M** params)
- Frozen: embeddings, self-attn, MLP, `lm_head`

`out_kg_finetune` loads this architecture with **no missing / unexpected keys**.

---

## 5. Cases tried (chronological)

### Case A — Text in the prompt (rejected, not implemented)

**Idea:** prepend KG facts as CIF comments.

**Why not:** closed tokenizer → `# similar_materials: NMC811` becomes `<unk>`. Ruled out before coding.

---

### Case B — Ungated cross-attention, all 8 layers (`out_kg_finetune`) — **August**

**Setup**

- Start: `crystallm_v1_small`
- Train: freeze GPT; train CA + `kg_proj` + `ln_x`
- Data: `kg_finetune_cifs`, `max_cifs=1000`
- Steps: **2000**, lr **1e-4**, batch 1, block 512, MPS
- Export: `battery_v2/export_package`
- Architecture: CA in **every** layer, **no** tanh gate, **no** zero-init `c_proj`

**Result — `prompt_lifepo4.txt`, seed 1337, 3 samples** (`compare_ab/`)

| Model | Parse (pymatgen) | Li+Fe+P+O in sites | What it wrote |
|-------|------------------|--------------------|---------------|
| Vanilla | 0/3 | 0/3 | **Ba** / lone O / malformed first site |
| **August KG** | **1/3** | **3/3** | Li, Fe, P, O rows; messy numbers (`1.0`, glued tokens) |
| Gated (same prompt, later) | 0/3 | 0/3 | tags then Cu / Mn |

**Result — `prompt_lino2.txt` (full CIF header minus sites), Li3Mn2NiO6** (`compare_lino2/`)

| Model | Parse | Full formula | Avg site rows |
|-------|-------|--------------|---------------|
| Vanilla | 2/3 | 3/3 | **8.3** (clean `Li … 1` rows) |
| August KG | 1/3 | 3/3 | **1.7** (real Li/Mn/Ni/O but messy) |
| Gated | 0/3 | 0/3 | **0.0** |

**Interpretation**

- KG **did** help: stayed on formula elements while vanilla drifted (Ba).
- KG **hurt** CIF grammar: occupancy `1.0`, truncated / glued site lines.
- This is the **only** in-model KG checkpoint that still writes real site rows for LiFePO4.

---

### Case C — Gated CA + last 2 layers (`out_kg_gated`)

**Setup**

- `tanh(gate)` init 0 so residual starts at 0 (Flamingo / LLaMA-Adapter idea)
- CA only on **last 2** of 8 layers (`kg_skip_last=2`)
- Zero-init CA `c_proj`
- Same 1000 CIFs / 2000 steps planned, lr 3e-5
- Abort if Li3Mn2NiO6 site rows = 0

**Result**

- Teacher-forcing val loss **dropped** (e.g. ~5.3 → ~1.1 in earlier gated family; this run aborted).
- Autoregressive **site rows → 0**.
- Saved `ckpt.pt` is **iter 0** (identity: `c_proj` still zeros / gates unused).
- On A/B: garbage after `_atom_site_occupancy`.

**Why it failed**

- Either the gate never opened (model ≈ vanilla but with extra unused modules) **or** a small gate still perturbed the **last** blocks where site-row logits are produced.
- Last-2 + `ln_ca` (related variant) forced unit-variance residuals; even gate ≈ 0.21 is a large hidden-state change at the syntax bottleneck.
- **Val loss ≠ generation quality** for this family.

---

### Case D — Last-4 layers + gate + `kg_ln` (code experiment, not a valid reload of August)

**Setup (user edit, later reverted)**

- `kg_ca_layers=4` (CA only on layers 4–7)
- Per-block `tanh(gate)` init 0
- LayerNorm on KG keys/values

**What happened when loading `out_kg_finetune`**

- Missing: `gate`, `kg_ln` on layers 4–7
- Unexpected / dropped: **all CA weights on layers 0–3**
- After load: `tanh(gate) = 0` → **KG path is a no-op**

**Why it failed**

- Architecture no longer matched the August checkpoint. Not a fair retest of “last 4 layers.” KG was turned off, not improved.

---

### Case E — Same August architecture, different **training recipe** (`out_kg_finetune2`)

**What changed (not the block)**

- Occupancy `1.0` → `1` in training CIFs (match vanilla token style)
- **30% of steps** `kg_context=None` (skip CA; no gradient that step)
- Grad clip 1.0
- Every 500 steps: generate `data_LiFePO4\n`; **do not** overwrite `ckpt.pt` if site rows = 0
- Save dir **new** so August is not overwritten

**Train log (2000 steps, MPS, ~5 min)**

| Iter | Train loss (spot) | Val loss | Short-prompt site rows | `ckpt.pt` |
|------|-------------------|----------|------------------------|-----------|
| 1 | 17.61 | — | — | — |
| 500 | 0.79 | 0.95 | **0** | skipped |
| 1000 | 0.50 | 0.69 | **8** (formula incomplete) | **saved** |
| 2000 | 0.59 | **0.52** | **0** | skipped |

**Interpretation**

- Occupancy + KG-drop **did not** remove the collapse.
- Best generation during this run was **mid-training** (iter 1000), not lowest val loss (iter 2000).
- Same signature as gated: **teacher-forcing improves, sampling dies**.

---

### Case F — Soft prefix (not implemented)

**Idea:** project HGT vectors to a **short prefix** in front of CIF embeddings; self-attention over prefix + CIF; scale init 0 → step 0 = vanilla. Train only `kg_proj` + scale (~tiny vs 8.4M).

**Status:** recommended next in-model path. **No checkpoint.**

---

### Case G — Vanilla generate + KG formula filter (not a trained model)

**Idea:** keep vanilla syntax; drop samples missing Li/Fe/P/O.

**Status:** `sample_with_kg.py` has `require_elements`; not the main trained story.

---

## 6. Why it is not coming out “properly”

Four separate issues. Mixing them caused confusion.

### 6.1 Cross-attention training vs CIF syntax (main reason CIFs look broken)

Frozen GPT already writes occupancy and site rows. Ungated CA adds a **new residual in every layer** (or in the last layers, which still own the logits).

- Fine-tune is **teacher forcing** (next-token loss on gold CIFs). Loss can fall while the **autoregressive** path after `_atom_site_occupancy` is destroyed.
- Occupancy token shift (`1` vs `1.0`) and glued numbers are typical of that disruption.
- More steps / lower val loss **made this worse** (`finetune2` iter 2000, gated runs).

This is **not** “the attention formula is wrong.” Q=CIF, K/V=KG, residual add is standard. The failure is **injecting an undertrained residual into a frozen LM**.

### 6.2 HGT signal is weak (why CA cannot “fix chemistry enough” to justify the syntax cost)

Val MRR ≈ **0.07**. Dim 32, 1 HGT layer. The graph encodes materials / properties / cells, **not** `fract_x/y/z`. CA cannot invent valid coordinates from that.

The export **is** wired correctly. The vectors are **usable for formula bias** (August vs Ba). They are **not** a strong structure prior.

### 6.3 Prompt difficulty

| Prompt | Who looks good |
|--------|----------------|
| Full header `prompt_lino2.txt` / `prompt_lifepo4.txt` | Vanilla: easy continuation; already has cell + atom types |
| Official short `data_LiFePO4\n` | Harder; this is where KG *could* beat vanilla on formula |

Early “KG better than vanilla” was mostly on **element identity**, with a **rich prompt**. Vanilla still wrote cleaner numbers.

### 6.4 Later “fixes” either collapsed generation or turned KG off

| Fix | Outcome |
|-----|---------|
| Tanh gate init 0 | Identity or 0 site rows |
| Last 2 / last 4 layers | Same collapse, or August weights not loadable |
| Zero-init `c_proj` | Dead or aborted identity ckpt |
| KG-drop 30% | Still collapsed by iter 2000 |
| `kg_ln` + last-4 on August ckpt | Gates 0; dropped early-layer CA |

---

## 7. Which is performing best (as of 1 Sep 2026)

Depends on the metric:

| Metric | Winner |
|--------|--------|
| CIF syntax / well-formed site rows / parse | **Vanilla** |
| Stay on formula elements (Li+Fe+P+O, not Ba) | **`out_kg_finetune` (August)** |
| Overall “publishable CIF” | **Vanilla** |
| Worst | **`out_kg_gated`** (0 site rows) |

Nothing in this study **beats vanilla on both** syntax and chemistry.

**Thesis-honest split:** August CA = proof that HGT lookup + CA can bias elements; vanilla = CIF writer; later CA variants = failed attempts to reduce syntax damage.

---

## 8. Evidence files

| Path | Contents |
|------|----------|
| `CrystaLLM/compare_ab/` | Vanilla vs August vs gated on `prompt_lifepo4.txt` |
| `CrystaLLM/compare_lino2/` | Same three on `prompt_lino2.txt` (Li3Mn2NiO6) |
| `HGT-Pipeline/battery_hgt/outputs/battery_v2/training_log.csv` | HGT MRR / Hits |
| `HGT-Pipeline/battery_hgt/models/battery_v2/export_package/` | Frozen vectors + `id_maps.json` |
| `CrystaLLM/bin/finetune_with_kg.py` | Current trainer (August CA + occupancy + kg_drop; default save `out_kg_finetune2`) |
| `CrystaLLM/crystallm/_model.py` | Current ungated all-layer CA |
| `CrystaLLM/crystallm/kg_context.py` | Composition → HGT stack |

---

## 9. What not to do next

- Rebuild `hetero_data.pt` for “missing `.x`” (already filled).
- Another ungated 8-layer CA run expecting it to beat vanilla syntax.
- Gates / skip-last / `ln_ca` / zero `c_proj` as the main path.
- Overwrite `out_kg_finetune`.
- Full CrystaLLM retrain from scratch on a laptop.

---

## 10. What to do next (if continuing)

1. Keep **`out_kg_finetune`** as the chemistry ablation.
2. Keep **vanilla** as the syntax baseline.
3. If another in-model method is required: **soft prefix** from the same export (scale init 0, new folder `out_kg_prefix`), evaluate on `prompt_lifepo4.txt` **and** `data_LiFePO4\n` with the same seed.
4. Optional: vanilla + drop samples missing formula elements (cannot worsen occupancy syntax).

---

## 11. One-paragraph summary

The HGT export and composition lookup are correct; Element/Material `.x` were filled and used. HGT link prediction is weak (MRR ~0.07) but sufficient to **bias site elements** when injected via ungated cross-attention in all layers (`out_kg_finetune`, 1000 CIFs / 2000 steps). That same injection **damages CIF occupancy and site-row syntax** relative to the published CrystaLLM checkpoint (100k-iter full LM). Gated CA, last-N layers, and a second 2000-step run with occupancy normalisation and KG dropout all either collapsed autoregressive site rows while val loss fell, or failed to load the August weights. The problem is **how strongly and where** KG is injected into a frozen CIF LM, not a broken embedding file.
