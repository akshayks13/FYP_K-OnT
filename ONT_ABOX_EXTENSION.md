# Extending OnT to a populated knowledge graph (ABox)

**Project:** BattGPT battery materials KG → instance embeddings → CrystaLLM  
**Paper we start from:** Yang, Chen, He, Gao, Horrocks. *Language Models as Ontology Encoders* (OnT), arXiv:2507.14334 / ISWC 2025. Code: `OnT/`  
**Updated:** 17 September 2026  

**Earlier shorter plan:** `ONT_ABOX_EXTENSION_original.md` (same idea; this file is the full spec).  
**Numeric concat:** this file §8 (probe + step-by-step).

---

## 1. The job

```text
Ontology (TBox)  +  populated KG (ABox in output/battery_kg.ttl)
        ↓
OnT: same PLM, same Poincaré space, extra JSONL rows for individuals
        ↓
material / crystal / element vectors  (+ numeric side channel)
        ↓
CrystaLLM lookup by formula (LiFePO4), not by mp-id
```

OnT is TBox-only today (DeepOnto verbalises **classes**). We add the ABox by **template-verbalising individuals** and **appending** the same `child` / `parent` / `negative` rows. No new `model.py`.

**Claim:**

> OnT is designed for TBox-level ontology encoding. We extend it to a populated knowledge graph by adding ABox individuals and instance-level relations as extra training rows, so the same model embeds both ontology concepts and KG instances. Literals that MiniLM cannot resolve as text are concatenated as floats at export.

---

## 2. What this KG actually is

Source of truth: `output/battery_kg.ttl` (~9.6M lines, ~8.4M triples, ontology `https://w3id.org/battgpt/kg`).

| Layer | What exists | Scale |
|---|---|---|
| **TBox** | OWL classes, `subClassOf`, object/datatype properties | ~12 classes, ~21 object properties, ~31 datatype properties |
| **ABox** | Named individuals from Materials Project | ~5,000 materials, ~5,000 crystals, elements, roles, unit cells, **plus** ~10⁵ sites, ~6×10⁵ bonds, tens of thousands of quantity nodes |

Type mapping (`output/ontology_mapping.json`):

| Informal name | OWL class |
|---|---|
| Material | `chsub:Substance` |
| Crystal | `cryst:Crystal` |
| UnitCell | `cryst:UnitCell` |
| Element | `chsub:Element` |
| Property (quantity) | EMMO quantity class |
| BatteryRole | `battgpt:BatteryRole` |

A material is **not** a class. Example (`mp-10178`):

```turtle
<https://w3id.org/battgpt/kg/material/mp-10178> a chsub:Substance ;
    rdfs:label "Material LiMgP (mp-10178)"@en ;
    battgpt:hasFormula "LiMgP"^^xsd:string ;
    battgpt:hasChemsys "Li-Mg-P"^^xsd:string ;
    battgpt:isMetal true ;
    battgpt:isStable false ;
    battgpt:hasStructure <https://w3id.org/battgpt/kg/crystal/mp-10178> ;
    battgpt:hasBandGap <https://w3id.org/battgpt/kg/property/mp-10178/band_gap> .
```

Band gap is an EMMO quantity individual (number + unit), not a float on the material. User-facing CrystaLLM input is the **formula** (`LiFePO4`), not `mp-10178`.

There is **one** populated graph with unique `mp-` URIs. This is not ontology alignment and not entity resolution.

### 2.1 Do we need the full TTL?

**Yes.** Instance facts exist only in `output/battery_kg.ttl`.

**No** — do **not** pass it to OnT as an OWL file. `OntologyTransformer.fit(owl_path="battery_kg.ttl")` will fail: DeepOnto / OWL API cannot load 9.6M lines, and OnT’s loader only walks `owl_classes`.

```text
battery_kg.ttl          ← keep; stream/extract
        │
        ▼
entities.json + sentences + JSONL
        │
        ▼
OnT trainer             ← never opens the TTL
```

**Correction (implemented, 23 Sep 2026):** TBox classes do **not** sit only in the first part of the file. `owl:Class` / `owl:ObjectProperty` / `owl:DatatypeProperty` declarations are scattered across all 9.6M lines (e.g. `battgpt:ElectrolyteRole`, `battgpt:PositiveElectrodeRole`, `battgpt:BatteryRole` first appear at lines ~31,037 and ~6,293,196 and ~6,329,914, not in the head). `extract_tbox.py` does one full streaming pass (same block reader as `extract_abox.py`) collecting every such declaration by type, not just the first ~360 lines. It also drops the one dangling `rdfs:subClassOf` edge that points at an unimported external EMMO class, and stub-declares the EMMO classes ABox individuals are typed under (`chsub:Substance`, `cryst:Crystal`, …) since those are never declared locally. Copy those declarations into a tiny schema OWL — confirmed correct, just not confined to the head. Do not `owl:imports` full EMMO.

---

## 3. What OnT does today (TBox) — paper and code

Code under `OnT/ont/`. Paper + `prepare.py`:

```text
tiny TBox OWL
    → DeepOnto OntologyVerbaliser
    → "substance", "has structure some crystal", …
    → train.jsonl / train_exist.jsonl / concept_names.json
```

**Recipe:**

1. Verbalize a **class** as English (labels; `∃r.C` as “has structure some …”).  
2. Encode with a pretrained LM (`HierarchyTransformer`).  
3. Map into the **Poincaré ball**.  
4. Train hierarchy (`ont/losses/hit_loss.py`: clustering + centripetal) and existential role rotation (`ont/losses/logical_loss.py`).

**Data pipeline** (`ont/data/prepare.py`):

- Loads OWL via DeepOnto.  
- `initial_atomic_names` iterates **`ont.owl_classes`** and **`ont.owl_object_properties` only**.  
- Named individuals are ignored.  
- Writes `train.jsonl`, `train_exist.jsonl`, `train_conj.jsonl`, `concept_names.json`, `role_names.json`, `val.json`.

`train.jsonl` row:

```json
{"child": "<sentence>", "parent": "<sentence>", "negative": ["<sentence>", "..."]}
```

The trainer does not care whether `child` is a class name or a material block. **We** never put instances in `concept_names.json`.

OnT can encode an instance **if** `V(instance)` is in `child` and we later `encode()` that same string.

---

## 4. What we are not doing

| Approach | Why not |
|---|---|
| PARIS, SILK, LIMES, LogMap, OAEI, DeepMatcher, Ditto, OpenEA | Align two graphs / `sameAs`. We have one mint of URIs. |
| Nominal reduction (`A(a)` → `{a} ⊑ A` for every node) | Fake classes at KG scale. |
| A second encoder fused with OnT | One `OntologyTransformer` only. |
| One vector per site, bond, or property URI | Pack those facts into \(V(\text{material})\) / \(V(\text{crystal})\). |
| Raw CIF / per-site xyz in \(V(a)\) | Short site summary only. |
| Labels not in the TTL (`insulator`, …) | Use `isMetal` / `isStable` and the actual numbers. |

---

## 5. How TBox and ABox live in hyperbolic space

**One Poincaré ball** for **training** and **Phase 5b** (`v = encode(…)`, `manifold.dist`). Class names and instance blocks are both points. After export, `z = concat(v, x)` is **not** in that ball (see §8, §10).

```text
origin (more general)
    ●  encode("substance")          TBox
    ●  encode("crystal")            TBox
       │
       ●  V(LiFePO4)                 ABox leaf under substance
       ●  V(its crystal)             ABox leaf under crystal
```

- **Clustering:** child nearer true parent than a negative class.  
- **Centripetal:** child is a leaf (farther from origin than parent).  
- **Exist:** child under `∃ has structure . V(crystal)` (role rotation + same hierarchy geometry).

TBox **places** the ABox vector (under the right class, tied to its crystal). The **sentence + numbers** **fill** it (chemistry). Type rows with ~12 classes are easy for MiniLM; **`hasStructure` is the load-bearing ABox relation.** Other object properties may appear in \(V(a)\) as text; they do not all get exist rows unless we add them later.

---

## 6. How ABox is included (DeepOnto does not do this)

**Overview** (same split, short):

```text
                    YOUR KG (battery_kg.ttl)
                           │
                 ┌─────────┴─────────┐
                 │                   │
               TBox                ABox
                 │                   │
                 │ copy schema       │ stream extract
                 ▼                   ▼
          tiny .owl            entities.json
                 │                   │
                 │ prepare.py        │ verbalize
                 ▼                   ▼
          class JSONL          V(a) sentences
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    merge + TBox oversample
                           ▼
              train.jsonl / train_exist.jsonl
                           ▼
                 Existing OnT trainer
                 (same PLM, same losses)
                           ▼
                    CrystaLLM lookup table
```

**Full pipeline** (scripts, losses, numbers, formula lookup):

```text
                         output/battery_kg.ttl
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              │  TBox                                 │  ABox
              │  copy classes / properties            │  stream (do not load as one OWL)
              │  no full EMMO import                  │
              ▼                                       ▼
     OnT/data/battgpt_tbox.owl              extract_abox.py
              │                                       │
              │  DeepOnto + prepare.py                ├─ entities.json
              │  OntologyVerbaliser                   │     types, formula, hasStructure,
              │                                       │     quantity values, flags
              ▼                                       │
     concept_names.json  (classes ONLY)               ├─ verbalize_abox.py
     role_names.json                                  │     structured V(a)  (§7)
     class rows in train.jsonl                        │     same string at train and encode()
              │                                       │
              │                                       ├─ numeric_features.json
              │                                       │     band gap, E_form, …  (§8)
              │                                       ▼
              │                              V(mp-10178), V(crystal), V(Li), …
              │                                       │
              └──────────────────┬────────────────────┘
                                 │
                        merge_abox.py
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              type rows     exist rows     TBox rows
              V(a) under    V(a) under     repeated so
              "substance"   ∃ has structure  classes are
              (hierarchy    . V(crystal)    not drowned
               reuse =      + hard crystal
               hypothesis)    negatives
                    │            │            │
                    └────────────┼────────────┘
                                 ▼
                  train.jsonl + train_exist.jsonl
                                 │
                                 ▼
                  existing OnT trainer
                  MiniLM → Poincaré ball
                  L_hierarchy + L_exist
                  (do not change model.py)
                                 │
                                 ▼
                  export_instance_embeddings.py
                                 │
                         encode(V(a))  →  v
                         concat(v, x)  →  z
                                 │
                                 ▼
                  export_package/
                    node_embeddings.pt
                    id_maps.json
                      LiFePO4  →  material z
                      Li, Fe, P, O  →  element vectors
                    meta.json  (kg_dim, scaler)
                                 │
                                 ▼
                  CrystaLLM  query "LiFePO4"
                    fetch material + elements
                    (not the whole TBox)
```

### 6.1 Three channels (same files, not a new net)

Default implementation **reuses** OnT losses so we do not change `model.py`. That reuse is a **hypothesis**, not a theorem: subsumption \(C ⊑ D\) is not the same logical job as typing \(a : C\) or as \(r(a,b)\). Validate it in Phase 5 **before** CrystaLLM.

| Channel | KG fact | OnT view | Default loss | Status |
|---|---|---|---|---|
| TBox | class ⊑ class | class names | hierarchy | OnT as designed |
| ABox type | `a : Substance` | \(V(a)\) ⊑ `"substance"` | **reuse** hierarchy | **test vs** a simple class margin / softmax on \(V(a)\) → class |
| ABox relation | `hasStructure(m, c)` | \(V(a)\) ⊑ ∃hasStructure.\(V(c)\) | **reuse** exist | load-bearing; still **measure** (Phase 5b). This is **narrower** than OnT’s paper: one role, two point clouds (materials ↔ crystals). `f_r` is closer to an alignment than a logical operator over many concept pairs. Do not treat Hits@k as a proof of Proposition 1. |
| Literals | formula, band gap, … | fields in \(V(a)\) | none in OnT | concat floats at export (§8) |
| ABox relation 2 | `belongsToElectrode(m, role)` | \(V(a)\) ⊑ ∃belongsToElectrode.(role class) | **reuse** exist | **added 23 Sep 2026.** Not in the original plan text below — found while implementing: covers **3,626 / 5,000 materials (73%)**, pointing at one of exactly 4 fixed role individuals (`Positive/Negative ElectrodeRoleIndividual`, `ElectrolyteRoleIndividual`, `SeparatorRoleIndividual`), each typed under the real `BatteryRole` subtree built in §3/Phase 3. This is what actually ties most ABox materials into the *deep* hierarchy (BatteryRole → …), not just the flat `material ⊑ substance` line. Only 4 possible fillers, so negatives are just "the other 3 roles" — no hard-negative mining needed, unlike `hasStructure`. |

```text
L = L_hierarchy(TBox + type rows) + λ_exist · L_exist(hasStructure, …)
```

Oversample TBox rows so class axioms are not drowned in the **optimizer**. That does **not** fix **geometric crowding**: ~5k material points as leaves under ~12 parents. Check pairwise distances (Phase 5).

**Negatives:** type rows → other **classes**. Exist rows → mix of (1) random other crystals and (2) **hard** tails: crystals of materials with similar formula / same chemsys / same space group. Random-only negatives are too easy.

**Class vs instance:** OnT has no flag. **We** keep classes in `concept_names.json`; individuals only in `verbalizations.json` + export.

### 6.2 What “TBox in the ABox embedding” means

We do **not** concatenate `"substance"` into \(V(a)\) at fetch by default.

Training already pulls \(V(a)\) under `"substance"` and toward its crystal. At CrystaLLM fetch we send **material + elements**. Do **not** stack every TBox class vector on every query.

---

## 7. Verbalization (structured template)

**TBox:** DeepOnto short names (`substance`, `crystal`, …).

**ABox:** our template. Labelled fields (not one run-on paragraph). Only facts **present** in the TTL; omit missing clauses. Do not invent labels.

```text
Material: LiMgP (mp-10178)
Type: substance
Formula: LiMgP
Chemical system: Li-Mg-P
Metallic: true
Stable: false
Band gap: 0.00 eV
Formation energy: … eV/atom
Bulk modulus: 55.7 GPa
Structure: Crystal structure of LiMgP
Space group: …
Lattice: a, b, c / α, β, γ
Sites: 3 (Li, Mg, P)
```

- Round floats; keep units.  
- **Sites:** count + element list. **Do not** dump per-site xyz.  
- **Bonds:** optional short list of a few distances, or omit if long.  
- Verbalize tails (`Crystal structure of LiMgP`), not raw URIs.  
- Formula / type / quantities **first** (MiniLM truncation).  
- Same URI → **same string** at train and at `encode()`.

`Metallic` / `Stable` from `battgpt:isMetal` / `battgpt:isStable` only.

**Elements** (looked up on every CrystaLLM query — do not leave as `"Element: Li"`):

```text
Element: Li
Atomic number: 3
Group: 1  Period: 2
Electronegativity: 0.98
Valence electrons: 1
```

Static chemistry (periodic table / Element columns already in the KG). No extra TTL walk.

---

## 8. Numbers (text + concat) — full detail

We want **both**:

- verbalization \(V(a)\) for **words** (LiMgP ≠ LiZnP)
- concat for **numbers** (0 eV ≠ 8 eV)

in **one** vector that CrystaLLM sees:

```text
v = encode(the sentence)          # words   (OnT trains on this)
x = the real floats, scaled       # numbers (TTL, not MiniLM)
z = concat(v, x)                  # glue lists; not arithmetic add
```

`+` / concat means stick the lists end to end. It is not `0.81 + 8.0`.

Probe code: `OnT/ont/data/probe_numeric.py`, `OnT/ont/data/plot_poincare.py`.  
Probe data: `OnT/data/numeric_probe/` (including `poincare_print.pdf`, `embeddings_both_cases.pdf`).

### 8.1 Why text is not enough (MiniLM probe)

MiniLM **does** see `Band gap: 0.00 eV` in the sentence. It does **not** treat that as a number on a number line. It cuts strings into pieces:

```text
"0.00"  →  0   .   00
"8.00"  →  8   .   00
```

`0.00` and `8.00` share the piece `00`, so they can look **more** alike than `0.00` and `3.50`. The model matches **text**, not magnitude.

We still write the number **in** \(V(a)\) (honest TTL, and the words “band gap” / “eV” help). We **also** keep `0.0` as a real float on the side.

**Experiment (pretrained MiniLM-L12, no OnT training).** Same LiMgP sentence; only the band-gap **value** changed (`0.00`, `0.01`, `0.35`, `1.34`, `3.50`, `8.00` eV). Control: LiZnP with the **same** numbers as LiMgP-0.00 (only words differ). Cosine: `1` = same vector, `0` = different.

| What we compared | Verbalization only `v` | `v` + scaled concat `z` |
|---|---|---|
| LiMgP vs LiZnP (different **words**, same numbers) | already good (`0.81`) | still good (`0.90`) |
| Same LiMgP, `0` vs `8` eV (different **numbers**, same words) | almost the same (`0.9999`) | clearly different (`0.35`) |
| Nearest neighbour by true band gap (6 clones) | 2 / 6 | 5 / 6 |

Words work. Digits in the sentence do not. Concat is what makes `0 eV` ≠ `8 eV`.

**Similar numbers are not a problem.** They *should* stay close.

| Situation | `v` | `z` | Problem? |
|---|---|---|---|
| Same formula, `0.00` vs `0.01` eV | together | together | No — same chemistry |
| Different formula, **same** numbers (LiMgP vs LiZnP) | split (`0.81`) | still split | No — words still work |
| Same formula, `0` vs `8` eV | piled (`0.9999`) | split (`0.35`) | **This** is what concat fixes |

**Full probe cosines** (six LiMgP clones; only band gap changes):

| Pair | Only sentence `v` | Raw `x` (no scale) | Z-scored `x` | Sentence + scaled concat `z` |
|---|---|---|---|---|
| 0.00 vs 0.01 eV | 0.9998 | 1.00 | 1.00 | 0.9999 |
| 0.00 vs 1.34 eV | 0.9992 | 1.00 | 0.87 | **0.94** |
| 0.00 vs 3.50 eV | 0.9993 | 1.00 | 0.21 | **0.61** |
| 0.00 vs 8.00 eV | **0.9999** | 0.99 | **−0.31** | **0.35** |
| LiMgP vs LiZnP, same numbers | **0.81** | 1.00 | 1.00 | 0.90 |

Raw concat **without** z-score still fails (`0.99` for 0 vs 8 eV): bulk modulus ~55 drowns band gap 0–8. **Must** scale per column on train materials.

OnT-style Poincaré disks (training metric is `manifold.dist` on `v`, not cosine): `OnT/data/numeric_probe/poincare_print.pdf`. Left disk = `v` (all LiMgP gaps on one point). Right disk = `z` drawn the same way for illustration — **not** OnT’s trained ball; class names are **not** concat’d. Embedding heatmaps: `embeddings_both_cases.pdf`.

This probe was **before** OnT fine-tune. After training, `v` still will not split 0 vs 8 eV (the loss never asks for that). Concat still happens **after** train.

### 8.2 How concat is built (step by step)

Take real `mp-10178` LiMgP.

**Step 1 — Sentence \(V(a)\)** (same template as §7):

```text
Material: LiMgP (mp-10178)
Type: substance
Formula: LiMgP
Chemical system: Li-Mg-P
Metallic: true
Stable: false
Band gap: 0.00 eV
Formation energy: -0.40 eV/atom
Energy above hull: 0.35 eV/atom
Bulk modulus: 55.7 GPa
Shear modulus: 22.3 GPa
Structure: Crystal structure of LiMgP
Sites: 3 (Li, Mg, P)
```

OnT `encode(V(a))` → **384** numbers = `v` (word part). LiMgP vs LiZnP: different `v`. LiMgP at 0 eV vs 8 eV: almost the **same** `v`.

**Step 2 — Same facts as a float list** (no words):

```text
x_raw = [
  0.0,      # band gap (eV)
  -0.40,    # formation energy (eV/atom)
  0.35,     # energy above hull (eV/atom)
  55.7,     # bulk modulus (GPa)   — drop if >50% missing in the extract
  22.3,     # shear modulus (GPa)
  1.0,      # is metal?  yes=1 no=0
  0.0,      # is stable? yes=1 no=0
]
```

Missing in the TTL: do **not** invent chemistry. Count missing **per column**. If a column is **>50% missing**, **drop it from `x`** rather than “0 + mask.” A linear `kg_proj` may ignore a mask bit and treat missing-as-zero as a real low value. If a column is mostly present, 0 + missing-bit is allowed.

**Step 3 — Scale each column (z-score) on train materials only:**

```text
scaled = (value - mean_of_that_column) / std_of_that_column
```

Example shape (illustrative means):

```text
band gap  0.0  →  (0.0  - 1.2) / 2.0  =  -0.60
band gap  8.0  →  (8.0  - 1.2) / 2.0  =   3.40
bulk      55.7 →  (55.7 - 60) / 20    =  -0.22
```

Store `mean` / `std` (and feature names) in `meta.json`. Call the scaled list `x`.

**Step 4 — Glue:**

```text
v  = [ v1, v2, … , v384 ]
x  = [ x1, … , x7 ]          # after drops, |x| may be < 7

z  = [ v1, … , v384,  x1, … , x7 ]
      |------ words ------|  |-- numbers --|
```

`kg_dim` = `384 + |x|`. CrystaLLM `kg_proj` **must** match.

- Same formula, different band gap → `v` ≈ equal, `x` different → `z` different.  
- Different formula, same numbers → `v` different, `x` equal → `z` different.

**Step 5 — When (full train cycle):**

```text
START     pretrained MiniLM
TRAIN     encode(V(a)) → v in Poincaré ball
          loss: child nearer parent than negative; child farther from origin;
                V(material) under ∃ has structure . V(crystal)
          backprop updates MiniLM  →  v for the same sentence MOVES
          (hierarchy fills the ball; 0 eV vs 8 eV still overlap)
FREEZE    stop updating weights
EXPORT    v = encode(V(a))     # trained encoder, still 384-d
          x = scaled TTL floats
          z = concat(v, x)     # does not change v; does not retrain
          write z → node_embeddings.pt
CRYSTALLM reads z. OnT never trains on x.
```

Concat is **not** inside `model.py`. Training **rewrites** `v`. Concat **appends** `x` after that.

### 8.3 Hyperbolic contract

Poincaré geometry (`d_κ`, `manifold.dist`) applies to **`v` only** — training and Phase 5b. `z` at export is a **plain concatenated feature vector** with **no** metric guarantees. Do not compute hyperbolic distance on exported `z`.

Do **not** concat onto class names (`substance`, `crystal`). Dummy zeros on classes made them collapse in a 2D plot of `z`; that was a drawing error, not the method.

Elements/crystals: `x` may be empty or a smaller set; pad consistently or skip concat for those types.

### 8.4 What we always do

1. Write numbers **in** \(V(a)\) (value + unit).  
2. Keep the same numbers as a **float list**.  
3. Scale on **train** materials; save scaler.  
4. After OnT `encode`, concat → CrystaLLM.  
5. Do not change OnT’s model or losses for numbers.

---

## 9. Full TTL: all facts in, not all as vectors

**Read** the entire `battery_kg.ttl` (stream). Do not skip materials or crystals.

**Own \(V(\cdot)\) and own embedding:**

- materials, crystals, elements, battery roles  
- space groups / crystal systems / unit cells if we verbalise them (short labels)

**No own vector** — facts written into the parent sentence (and into `x` for quantities):

- quantity nodes → number in \(V(a)\) **and** in `x`  
- atomic sites, bonds → **summary** only  

**Not in \(V(a)\):** raw CIF loop.

**Main exist relation:** `hasStructure`. **Second exist relation (added):** `belongsToElectrode`, 73% coverage — see §6.1 table. Other object properties: pack into \(V(a)\) unless we later add more exist rows.

Target: **thousands of training rows**, not one row per triple.

---

## 10. CrystaLLM: lookup, fetch, unseen formulas

`CrystaLLM/crystallm/kg_context.py` looks up **composition strings** (`LiFePO4`, `Li`, `Fe`, …) via `id_maps.json` → `name_to_nodes`, not `mp-10178`.

### 10.1 Export (`export_package/`)

Must match what `CrystaLLM/crystallm/kg_context.py` already reads (`node_embeddings.pt`, `id_maps.json`):

- `node_embeddings.pt` — tensors per type (`material`, `crystal`, `element`, …). Each material row is **`z = concat(v, x)`** (or `v` only if `x` empty; then `kg_dim` is OnT dim).  
- `id_maps.json` — `name_to_nodes` keys **must** include:

  - formula (`LiMgP`, `LiFePO4`)  
  - `Material LiFePO4` (builder also tries this)  
  - `mp-10178` / `material/mp-10178` (debug)  
  - element symbols (`Li`, `Fe`, `P`, `O`)

Internal index can still be `material/mp-…`.

**Polymorphs (limitation, not solved):** several `mp-` ids can share one formula and **different structures**. Lookup today picks **one** vector per name (prefer `isStable true`, else lowest energy-above-hull, else first; record the `mp-` in `meta.json`). That **throws away** polymorph variation CrystaLLM might need for CIF. Flag it; revisit if Phase 7 looks format-collapsed (one structure per formula).

`meta.json`: `embedding_dim` = `len(z)`, scaler, concat feature names, OnT checkpoint path.

### 10.2 What to send at fetch

For query `LiFePO4`:

1. One **material** vector (formula lookup).  
2. One vector per **element** in the formula.  
3. Optional: that crystal / battery role if exported and it fits `top_k`.

Do **not** send all TBox class vectors on every query.

### 10.3 Unseen formula (not in the KG)

OnT can still `encode` a minimal block:

```text
Material: LiFePO4
Type: substance
Formula: LiFePO4
```

`x` = missing/zeros unless numbers are known. If lookup misses, `encode` this block on the fly; otherwise `kg_context` would get a zero vector.

---

## 11. Next steps

Do not change `ont/model.py`, `ont/losses/*`, `ont/hit.py`. Adapter is TTL → JSONL → encode → concat → export. Touch `kg_proj` **input size** if `kg_dim` grows.

### Phase 0 — Scope

20-material slice: label, formula, quantities, `hasStructure`; sites as **summary**; no extra site entities.

### Phase 1 — Extract

`OnT/ont/data/extract_abox.py`

- Stream `output/battery_kg.ttl` (block-wise; do not `rdflib.Graph.parse` the full file).  
- `--max-materials 20`.  
- Write `OnT/data/battgpt_abox/entities.json` (uris, types, literals, quantity values, `hasStructure` tails, formulas).

Role strings must match OnT `role_names.json` (e.g. `"has structure"`).

### Phase 2 — Verbalizer

`OnT/ont/data/verbalize_abox.py` → `verbalizations.json`. Structured template (§7). Read five blocks by eye.

Also write `numeric_features.json` (per material id): raw floats for concat.

### Phase 3 — TBox only

Copy schema into `OnT/data/battgpt_tbox.owl` (no EMMO import).

```text
prepare_ontology_data("OnT/data/battgpt_tbox.owl", "OnT/data/battgpt_ont/")
```

**`concept_names.json` stays classes only.**

### Phase 4 — Merge

`OnT/ont/data/merge_abox.py`

Membership → `train.jsonl`:

```json
{
  "child": "<V(mp-10178)>",
  "parent": "substance",
  "negative": ["crystal", "element", "battery role", "space group"]
}
```

Role → `train_exist.jsonl` (`has structure some <V(c)>`). Negatives = random crystals **plus** hard crystals (similar chemsys / space group). Oversample TBox.

**Compute the counts** before train (do not pick 50–100× in isolation):

```text
n_tbox          = class axioms after prepare.py
n_abox_type     = materials (+ crystals if typed)
n_abox_exist    = hasStructure rows
repeat TBox until n_tbox_oversampled ≈ n_abox_type   # ~1:1 is the default target
log these four numbers in merge output / meta
```

~5k ABox type rows vs a few dozen TBox axioms: 100× TBox ≈ 5k vs 5k. Print the actual post-merge sizes.

### Phase 5 — Train + OnT-only checks (before CrystaLLM)

`pipeline.fit` skips OWL prepare. Data: `OnT/data/battgpt_ont/`.

- Base: `Hui97/OnT-MiniLM-L12-galen` if available, else `all-MiniLM-L12-v2`.  
- Batch 16–32, 1–3 epochs, `existence_loss_kind=hit`.  
- Device: MPS/CPU if no CUDA.

**Ablation (type loss):** default = hierarchy on \(V(a)\) ⊑ class. One extra run = simple margin or softmax \(V(a)\) → class names (TBox rows still use hierarchy). Keep whichever wins on Phase 5b type accuracy. Do not skip this and treat reuse as proven.

**Crowding diagnostic:** sample materials of the same class; pairwise Poincaré (or cosine) distances **before vs after** fine-tune. Confirm they have **not** collapsed to one blob near the ball boundary. Oversampling TBox does not replace this check.

Sanity: `encode("substance")` ≠ `encode("crystal")`; \(V(\text{mp-10178})\) under `substance`; \(V(\text{LiMgP})\) ≠ \(V(\text{LiZnP})\); loss moves.

### Phase 5b — Standalone ABox eval (required, before Phase 7)

Held-out materials (not used as the only CrystaLLM score):

1. **Type:** is the nearest class of \(V(a)\) the true type (Hits / accuracy vs `"substance"` vs others)?  
2. **hasStructure:** rank the true crystal vs other crystals (MRR / Hits@k). If this is chance, do not blame CrystaLLM yet. “Good” here means **better than random / hard-negative baseline**, not OnT-paper exist MRR on GO/SNOMED. One role, two instance clouds.

OnT’s built-in evaluator stays class subsumption on `concept_names`; it is **not** this ABox eval.

### Phase 6 — Export

`OnT/ont/export_instance_embeddings.py`

```python
v = model.encode(V(a))
z = concat(v, normalize(x[a]))
```

Write `export_package/` for `kg_context.py`: `node_embeddings.pt`, `id_maps.json` (formula + element keys), `meta.json`. Fit scaler on train materials only.

`z` is **not** a Poincaré point. Phase 5b distances stay on `v`.

### Phase 7 — Experiments

| Condition | Vectors |
|---|---|
| CrystaLLM, no KG | existing baseline |
| CrystaLLM + OnT TBox only | class vectors only |
| CrystaLLM + OnT+ABox (`v` only) | `encode(V(a))` |
| CrystaLLM + OnT+ABox + numeric `z` | **main** |

OnT’s class-ranking MRR is TBox-only. Phase 7 is **after** Phase 5b. If CrystaLLM is flat, use 5b to see whether instance embeddings failed first.

---

## 12. Files

```text
NEW   OnT/ont/data/extract_abox.py
NEW   OnT/ont/data/verbalize_abox.py
NEW   OnT/ont/data/merge_abox.py
NEW   OnT/ont/export_instance_embeddings.py
NEW   OnT/data/battgpt_tbox.owl
NEW   OnT/data/battgpt_abox/entities.json
NEW   OnT/data/battgpt_abox/verbalizations.json
NEW   OnT/data/battgpt_abox/numeric_features.json

PROBE OnT/ont/data/probe_numeric.py          # MiniLM: words vs digits vs concat
PROBE OnT/ont/data/plot_poincare.py
PROBE OnT/data/numeric_probe/                # results, Poincaré PDFs, heatmaps

TOUCH OnT/ont/pipeline.py                    # device
TOUCH OnT/ont/data/load.py                   # TBox/ABox mix if needed
TOUCH CrystaLLM kg_proj / meta kg_dim        # if concat grows dim

DO NOT CHANGE
      ont/model.py
      ont/losses/hit_loss.py
      ont/losses/logical_loss.py
      ont/hit.py
```

Order: extractor (20 materials) → read verbalizations → 10 JSONL rows + one train step → tiny TBox OWL → merge + full extract → train 1 epoch → encode + concat → export with formula keys.

---

## 12.5 Implementation status (added 23 Sep 2026)

Everything below is **built and unit-verified at small scale (25 materials)**. Full-scale (~5,000 materials) training has **not** been run — the dev machine (Mac, MPS backend) thrashes system swap once past ~200 materials in a single process (verified: a 200-material/1-epoch run made 20/127 steps in 17 minutes before being killed, swap at 94%). Full-scale training is deferred to different hardware (a CUDA machine); `train_abox.py`'s `pick_device()` already auto-detects CUDA, no code change needed there.

**Done and matching this plan:**

- Phase 1 (`extract_abox.py`): streams the whole file once, verified correct at full 5,000-material scale (71s, no OWL/rdflib load of the full graph). Now also captures `belongsToElectrode` → role label.
- Phase 2 (`verbalize_abox.py`): structured template per §7, now includes a `Battery role: …` line when present.
- Phase 3 (`extract_tbox.py` + `ont/data/prepare.py`): real TBox — 12 locally-declared classes + 9 stub classes for EMMO types, 21 real object properties, 31 real datatype properties, 10 real `subClassOf` axioms (the battery-role hierarchy: `BatteryRole → {Electrolyte,Positive,Negative,Separator}Role → 6 concrete role classes`). No fabricated class list.
- Phase 4 (`merge_abox.py`): merges real TBox rows + ABox type rows + two ABox exist channels (`hasStructure`, `belongsToElectrode`).
- Phase 5 / 5b (`train_abox.py`): parametrized driver (material count, epochs, batch size all CLI args) plus four checks: type accuracy, `hasStructure` ranking (MRR/Hits@k vs. random baseline), `belongsToElectrode` role accuracy (vs. 25% random baseline), and crowding (pairwise Poincaré distance, before vs. after).

**Bug found and fixed:** `ont/data/prepare.py`'s negative sampling (`sample(concept_list, k=10)`, present in the original unmodified OnT code) never excluded the true child/parent from the sampled negatives. Harmless on the large ontologies OnT was validated on (hundreds/thousands of classes); on this project's small TBox (21 concepts, 10 negatives ≈ 48% of the pool) it meant the true parent landed in its own negative list on 9/10 axioms, corrupting training itself, not just eval. Fixed by excluding child/parent/filler from the negative pool per axiom type (`_safe_sample`); regenerated `battgpt_tbox_ont/` afterward (verified 0/10 contaminated).

**Real result at 25-material scale (1–8 epochs), after the fix:**

| Check | Result |
|---|---|
| ABox type accuracy (held-out) | 0% (pretrained) → 100% (trained), vs. real 21-class pool |
| `hasStructure` ranking (held-out) | MRR 1.0, Hits@1 100%, vs. random baseline MRR 0.15 |
| Crowding (mean pairwise dist, train set) | 4.25 → 7.08 (1 epoch) → 11.8 (8 epochs) — spreads, does not collapse |
| TBox centripetal (child farther from origin than parent) | 10/10 |
| TBox clustering (child nearer true parent than sibling) | 3/10 (1 epoch), 4/10 (8 epochs) — **weak**, matches Pitfall #4 below; more epochs alone doesn't fix it, and this is independent of ABox material count since the TBox axiom set is fixed at 10 unique axioms regardless of scale |

Honest read: the ABox-critical paths (material typing, `hasStructure` relation — the ones that matter for feeding CrystaLLM) are learning correctly. The TBox's own internal battery-role hierarchy clusters only partially at this epoch count — a real, now-measured instance of Pitfall #4 (loss-reuse hypothesis), not a blocking bug.

**Not yet done:** Phase 6 export (`export_instance_embeddings.py` / numeric concat per §8) does not exist yet. Phase 7 (CrystaLLM wiring) explicitly deferred by request.

## 13. Pitfalls

1. **Imports closure.** TBox OWL must not import full EMMO.  
2. **`concept_names` pollution.** No instance strings in that file.  
3. **Hyperbolic crowding.** ~5k leaves under ~12 classes. Oversample ≠ spread. Phase 5 pairwise distances.  
4. **Loss reuse is a hypothesis.** Hierarchy for typing may lose to a simple class margin — Phase 5 ablation.  
5. **Easy exist negatives.** Include hard crystals, not only random ones.  
6. **Numbers.** Text in \(V(a)\) **and** concat `x`; units; scaler in `meta.json`. `z` has **no** \(d_κ\). Count missing per feature; drop columns that are mostly empty.  
7. **Truncation.** Short sites; important fields first.  
8. **Evaluator mismatch.** CrystaLLM score ≠ ABox embedding quality — Phase 5b first.  
9. **Formula lookup.** `name_to_nodes` for formulas and elements, not only `mp-` ids.  
10. **Polymorphs.** One material per formula name — **information loss**, not a solved lookup.  
11. **TBox drown (gradients).** Oversample class rows; **log** post-merge TBox vs ABox counts (aim ~1:1 type rows).  
12. **Unique URIs.** Do not invent `sameAs` evaluation.  
13. **Thin elements.** Put atomic number / group / period / electronegativity in \(V(\text{Li})\), not only the symbol.  
14. **Exist reuse.** `hasStructure` ranking ≠ paper-level role semantics.

---

## 14. One-line summary

**TBox = DeepOnto class names. ABox = structured \(V(a)\) appended (type + hasStructure). Default = reuse OnT losses (validate). Export `encode(V(a))` + TTL floats; CrystaLLM looks up the formula (and elements).**
