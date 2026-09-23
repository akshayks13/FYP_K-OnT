# Extending OnT to a populated knowledge graph (ABox)

**Earlier / shorter plan.** The current spec is `ONT_ABOX_EXTENSION.md`.  
**Project:** BattGPT battery materials KG → instance embeddings → CrystaLLM  
**Paper:** Yang, Chen, He, Gao, Horrocks. *Language Models as Ontology Encoders* (OnT), ISWC 2025. Code: `OnT/`

---

## 1. The job

```text
Target ontology (EMMO + battgpt schema)
        +
Populated KG  (output/battery_kg.ttl)
        ↓
Embeddings of the *instances* (materials, crystals, elements, …)
        ↓
CrystaLLM cross-attention
```

OnT uses a language model to encode ontology text, then hyperbolic geometry trains `C ⊑ D` and `∃r.C`. That is TBox-only. This graph is almost entirely instance data. Keep OnT’s encoder and losses; feed it individuals.

**Claim:**

> OnT is designed for TBox-level ontology encoding. We extend it to a populated knowledge graph by adding ABox individuals and instance-level relations as extra training rows, so the same model embeds both ontology concepts and KG instances.

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
    battgpt:hasStructure <https://w3id.org/battgpt/kg/crystal/mp-10178> ;
    battgpt:hasBandGap <https://w3id.org/battgpt/kg/property/mp-10178/band_gap> .
```

Band gap is an EMMO quantity individual (number + unit), not a float on the material.

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

TBox classes sit in the **first part** of the same TTL. Copy those declarations into a tiny schema OWL. Do not `owl:imports` full EMMO.

---

## 3. What OnT does today (and where it stops)

Code under `OnT/ont/`.

**Recipe:**

1. Verbalize a **class** as English.  
2. Encode with a pretrained LM (`HierarchyTransformer`).  
3. Map into the Poincaré ball.  
4. Train hierarchy (`ont/losses/hit_loss.py`) and existential role rotation (`ont/losses/logical_loss.py`).

**Data pipeline** (`ont/data/prepare.py`):

- Loads OWL via DeepOnto.  
- `initial_atomic_names` iterates **`ont.owl_classes`** and **`ont.owl_object_properties` only**.  
- Named individuals are ignored.  
- Writes `train.jsonl`, `train_exist.jsonl`, `train_conj.jsonl`, `concept_names.json`, `role_names.json`, `val.json`.

`train.jsonl` row:

```json
{"child": "<sentence>", "parent": "<sentence>", "negative": ["<sentence>", "..."]}
```

`ont/data/load.py` turns each negative into `(child, parent, negative)`. The trainer does not care whether `child` is a class name or a material paragraph.

OnT can encode an instance **if** `V(instance)` is in `child`. The gap is data preparation, not a missing neural module. Inference is the same: `encode(sentence)`.

---

## 4. What we are not doing

| Approach | Why not |
|---|---|
| PARIS, SILK, LIMES, LogMap, OAEI, DeepMatcher, Ditto, OpenEA | Align two graphs / `sameAs`. We have one mint of URIs. |
| Nominal reduction (`A(a)` → `{a} ⊑ A` for every node) | Fake classes at KG scale; no singleton semantics. |
| A second ABox encoder fused with `COMBINE` | Two towers. We use **one** `OntologyTransformer`. |
| Euclidean TransE on OnT vectors | OnT is Poincaré; a Euclidean triple score fights that geometry. |
| One OnT sentence / one vector per site, bond, or property URI | Drowns MiniLM; that is nominals at KG scale. Pack those **facts into** \(V(\text{material})\) / \(V(\text{crystal})\) instead. |
| Pasting raw CIF into \(V(a)\) | Too long. Keep the **values** (sites, coords, bonds) as a short summary. |

---

## 5. How OnT becomes ABox-capable

Not a new network. Extra training rows in the files OnT already reads.

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
              e(Substance), e(Crystal), …
              e(mp-10178), e(crystal/mp-10178), …
                           ▼
                    CrystaLLM lookup table
```

No `COMBINE` block. Class and instance vectors are both `model.encode(sentence)` in the same Poincaré space.

### 5.1 Map ABox facts onto OnT losses

| KG fact | OnT view | File | Loss already in code |
|---|---|---|---|
| `OlivineCathode ⊑ PositiveElectrodeRole` | `C ⊑ D` | `train.jsonl` | hierarchy (`hit_loss`) |
| `mp-10178 a Substance` | `V(a) ⊑ substance` | `train.jsonl` | **same** hierarchy loss |
| `mp-10178 hasStructure crystal/mp-10178` | `V(a) ⊑ ∃hasStructure.V(c)` | `train.jsonl` + `train_exist.jsonl` | **same** exist / rotation loss |
| `hasFormula`, band gap 0.0 eV | inside `V(a)` | sentence text | no extra head |

```text
L_OnT       = class ⊑ class                 (unchanged)
L_type      = V(a) ⊑ class                  (hierarchy loss, ABox rows)
L_relation  = V(a) ⊑ ∃r.V(b)                (exist loss, ABox rows)
literals    = text inside V(a)              (template)
```

\(\lambda\) is how often TBox vs ABox rows appear. Oversample TBox when merging (repeat each class axiom, cap ~50–100) so the original OnT signal is not drowned.

### 5.2 What “compatible embeddings” means

- **Type:** hyperbolic clustering (instance nearer its class than a negative class) + centripetal (instance is a leaf).  
- **Relation:** `encode_existence(role, V(tail))` with `V(head)` under the exist loss.

Do **not** score `(h, rdf:type, C)` as a generic triple. `rdf:type` **is** the hierarchy loss.

### 5.3 OnT does not know “class” vs “data”

There is no type flag. OnT only sees `child` / `parent` / `negative` strings and \(L_≺\).

**We** keep two piles:

- **Classes** → short names in `concept_names.json` (`substance`, `crystal`, …).  
- **Individuals** → long \(V(a)\) sentences, **not** in `concept_names.json`. After training we `encode(V(a))` and store that vector as `material/mp-10178`.

That is how we avoid the paper’s “treat instances as classes.” Same trainer; we never add `mp-10178` to the class list.

`"negative"` is the paper’s \(D_{\mathrm{neg}}\) (Section 4.3): a wrong parent so child is closer to the true parent than to a random other class.

---

## 6. Verbalization

OnT’s \(V(C)\) for a class is a short English label. For individuals, \(V(a)\) is a **template that must carry all TTL facts attached to that entity** (and a short summary of nested sites/bonds/quantities). Not Wikipedia. Not one vector per site.

**Classes:** `substance`, `crystal`, `olivine phosphate cathode`, …

**Individuals (pack the TTL, do not drop facts):**

```text
{label}. Formula {formula}. Chemical system {chemsys}.
{Metallic|Non-metallic}. {Stable|Unstable}.
Band gap {bg:.2f} {unit}. Formation energy {fe:.3f} eV/atom.
Bulk modulus {k:.1f} {unit}.
Space group {sg}. Lattice a,b,c / α,β,γ.
N sites: {el x y z; …}. Bonds: {A–B d Å; …}.
```

`Metallic` / `Stable` come from `battgpt:isMetal` / `battgpt:isStable` in the TTL. If a property is missing, omit that clause. Do not invent labels that are not in the KG.

Example (`mp-10178`):

```text
Material LiMgP (mp-10178). Formula LiMgP. Chemical system Li-Mg-P.
Metallic. Unstable. Band gap 0.00 eV. Bulk modulus 55.7 GPa.
Crystal structure of LiMgP. 3 sites (Li …; Mg …; P …).
```

- Round floats. Never dump `0.35048874333333346`.  
- Keep units.  
- Do **not** paste raw CIF; **do** keep site/bond/lattice **values** as a compact list.  
- Verbalize tails (`Crystal structure of LiMgP`), not raw URIs.  
- Prefer keeping facts over a hard 80-token cap. Truncate boilerplate, not numbers. MiniLM may truncate very long strings — put formula, type, quantities, then sites/bonds.  
- Same URI → same string at train and at `encode()`.

Numbers go in as text (`Band gap 0.00 eV`). MiniLM may treat nearby floats as similar; still write the value and unit.

OnT will **run** on \(V(a)\). It will not “understand” `mp-10178` like a GO class name until we **fine-tune** on these rows.

---

## 7. Full TTL: all facts in, not all as vectors

**Read** the entire `battery_kg.ttl` (stream). Do not skip materials or crystals.

**Own \(V(\cdot)\) and own embedding** (every instance of that kind in the file):

- materials, crystals, unit cells (if verbalized), space groups, crystal systems, elements, battery roles

**No own \(V(\cdot)\) / no own vector** — facts still **written into** the parent sentence:

- quantity nodes (band gap URI → `0.00 eV` inside \(V(\text{material})\))  
- atomic sites, bonds (coords / distances inside \(V(\text{crystal})\) or \(V(\text{material})\))

**Not in the sentence as a blob:** raw CIF loop. Values from the CIF **are** in the summary.

Target: **thousands of training rows** (one / a few per verbalized individual), not one row per triple.

---

## 8. Next steps (OnT + ABox only)

Do not change `ont/model.py`, `ont/losses/*`, `ont/hit.py`. Adapter is TTL → JSONL.

### Phase 0 — Scope

Confirm on a 20-material slice: label, formula, quantities, `hasStructure`, and that site/bond facts from the TTL appear **inside** \(V(a)\), not as extra entities.

### Phase 1 — Extract ABox JSON

`OnT/ont/data/extract_abox.py`

- Stream `output/battery_kg.ttl`. Do not `rdflib.Graph.parse` the full file.  
- `--max-materials 20` for smoke tests.  
- Write `OnT/data/battgpt_abox/entities.json`.

Role strings must match OnT `role_names.json` (spaced labels, e.g. `"has structure"`).

### Phase 2 — Verbalizer

`OnT/ont/data/verbalize_abox.py` → `OnT/data/battgpt_abox/verbalizations.json`.  
Read five sentences by eye before training.

### Phase 3 — TBox only (existing prepare)

Copy prefix + class/property/`subClassOf` into `OnT/data/battgpt_tbox.owl` (no EMMO import closure).

```text
prepare_ontology_data("OnT/data/battgpt_tbox.owl", "OnT/data/battgpt_ont/")
```

**`concept_names.json` stays classes only.**

### Phase 4 — Merge ABox rows

`OnT/ont/data/merge_abox.py`

Membership → `train.jsonl`:

```json
{
  "child": "<V(mp-10178)>",
  "parent": "substance",
  "negative": ["crystal", "element", "battery role", "space group"]
}
```

Negatives = other **classes**.

Role → `train.jsonl` + `train_exist.jsonl` (`has structure some <V(c)>`). NF3 negatives = other same-type tails. Oversample TBox.

### Phase 5 — Train (existing)

Once JSONL exists, `pipeline.fit` skips OWL prepare. Point data at `OnT/data/battgpt_ont/`.

- Base: `Hui97/OnT-MiniLM-L12-galen` if available, else `all-MiniLM-L12-v2`.  
- Batch 16–32, 1–3 epochs, `existence_loss_kind=hit`.  
- Device: `pipeline.py` prefers CUDA; use MPS/CPU here.

Sanity: `encode("substance")` ≠ `encode("crystal")`; `V(mp-10178)` under `substance`; `V(LiMgP)` ≠ `V(LiZnP)`; loss moves.

### Phase 6 — Export for CrystaLLM

`OnT/ont/export_instance_embeddings.py`

```python
vecs = model.encode([V(a) for a in entities], convert_to_tensor=True)
```

Write `export_package/` as `CrystaLLM/crystallm/kg_context.py` expects: `node_embeddings.pt`, `id_maps.json` (short ids such as `material/mp-10178`), `meta.json`. `kg_dim` is 384 (MiniLM-L12) or 768 (MPNet).

### Phase 7 — Experiments

| Condition | Vectors |
|---|---|
| CrystaLLM, no KG | existing baseline |
| CrystaLLM + OnT TBox only | class vectors only (materials collide; weak ablation) |
| CrystaLLM + OnT+ABox | Phase 6 table |

ABox check: membership of `V(a)` vs true type vs a wrong type. OnT’s evaluator stays class subsumption on `concept_names`.

---

## 9. Files

```text
NEW   OnT/ont/data/extract_abox.py
NEW   OnT/ont/data/verbalize_abox.py
NEW   OnT/ont/data/merge_abox.py
NEW   OnT/ont/export_instance_embeddings.py
NEW   OnT/data/battgpt_tbox.owl
NEW   OnT/data/battgpt_abox/entities.json
NEW   OnT/data/battgpt_abox/verbalizations.json

TOUCH OnT/ont/pipeline.py          # device
TOUCH OnT/ont/data/load.py         # only if 50/50 batch mix is needed

DO NOT CHANGE
      ont/model.py
      ont/losses/hit_loss.py
      ont/losses/logical_loss.py
      ont/hit.py
```

Order: extractor (20 materials) → read verbalizations → 10 JSONL rows + one training step → tiny TBox OWL → merge + full extract → train 1 epoch → export.

---

## 10. Pitfalls

1. **Imports closure.** TBox OWL must not import full EMMO.  
2. **`concept_names` pollution.** No instance sentences in that file.  
3. **Centripetal loss.** Instances are leaves. Distinction comes from different text and different `∃r` parents.  
4. **Numeric text.** Units in the sentence are mandatory. Nearby floats may look similar to MiniLM.  
5. **Evaluator mismatch.** OnT MRR is class ranking, not CrystaLLM quality.  
6. **Unique names.** Materials have unique `mp-` ids; do not invent `sameAs` evaluation.

---

## 11. One-line summary

**Keep OnT. Stream the full TTL. Pack every fact into \(V(\text{material/crystal/…})\). Extra `train.jsonl` rows = membership + relations. Classes stay in `concept_names`; individuals do not. One PLM, one space, embeddings only for verbalized entities, all TTL data inside those sentences.**
