# RDF/TTL → GNN graph (R-GCN & HGT)

The RDF/TTL file is **not** the GNN input.  
You parse triples, assign IDs, then build tensors.

Same triples. **Two separate exports:**

| Model | Build from triples | Output |
|-------|--------------------|--------|
| **HGT** | typed nodes + typed edges | `HeteroData` → `hetero_data.pt` |
| **R-GCN** | flat graph **directly** (Way 1) | `x`, `edge_index`, `edge_type` |

Do **not** require HeteroData to train R-GCN. Flattening HeteroData is optional and worse as a primary path.

---

# 1. Start from RDF triples

RDF is only triples:

```
(subject, predicate, object)
```

Example (toy):

```ttl
:Tomato rdf:type :Plant .
:EarlyBlight rdf:type :Disease .
:LeafSpot rdf:type :Symptom .
:Fungicide rdf:type :Treatment .

:Tomato :hasDisease :EarlyBlight .
:EarlyBlight :hasSymptom :LeafSpot .
:EarlyBlight :treatedBy :Fungicide .
```

Useful training triples:

```
(Tomato, hasDisease, EarlyBlight)
(EarlyBlight, hasSymptom, LeafSpot)
(EarlyBlight, treatedBy, Fungicide)
```

Battery KG equivalent:

```
(material/mp-19017, hasStructure, crystal/mp-19017)
```

Still strings/URIs — not ready for PyTorch Geometric.

---

# 2. Parse the RDF

Use an RDF parser (RDFLib, or a streaming Turtle parser for large files).

| Subject | Predicate | Object |
|---------|-----------|--------|
| Tomato | hasDisease | EarlyBlight |
| EarlyBlight | hasSymptom | LeafSpot |
| EarlyBlight | treatedBy | Fungicide |

Also:

- `rdf:type` → node type (Plant, Disease, … / Material, Crystal, …)  
- numeric literals → node features  
- skip schema noise (`rdfs:label`, `owl:imports`, …)

Optional: OWL/RDFS reasoning to add inferred triples. Skip if you only use explicit instance triples.

**In this repo (HGT path):** `HGT-Pipeline/battery_hgt/rdf_to_hetero.py`

```bash
python rdf_to_hetero.py -i ../../output/battery_kg.ttl --out-dir ../../output/hetero_from_rdf
```

---

# 3. Split triples: nodes / edges / features

### Link triple → edge

```
(Tomato, hasDisease, EarlyBlight)
```

- Tomato, EarlyBlight → **nodes**  
- hasDisease → **relation**  
- whole fact → one **edge**

### Literal triple → feature (not an edge)

```
(LiCoO2, hasBandGap, 0.66)
```

`0.66` is not a node. It goes into that entity’s feature row.

Conceptually after this step:

```
NODES + FEATURES
EDGES (edge list)
```

---

# 4. Assign integer IDs

Networks cannot use strings.

### For R-GCN — one global ID space

| Entity | Global ID |
|--------|----------:|
| Tomato | 0 |
| EarlyBlight | 1 |
| LeafSpot | 2 |
| Fungicide | 3 |

| Relation | ID |
|----------|---:|
| hasDisease | 0 |
| hasSymptom | 1 |
| treatedBy | 2 |

Triples become `(src, rel, dst)` numbers: `(0,0,1)`, `(1,1,2)`, `(1,2,3)`.

### For HGT — local IDs per node type

| Type | Entity | Local ID |
|------|--------|---------:|
| Plant | Tomato | 0 |
| Disease | EarlyBlight | 0 |
| Symptom | LeafSpot | 0 |

Plant `0` and Disease `0` are different nodes.

Always save mappings (`node_mapping.json`, `relation_mapping.json`) so you can decode IDs later.

---

# 5. Edge list vs `edge_index` vs `.x`

These names confuse people — they are simple:

| Name | Meaning |
|------|---------|
| **edge list** | list of connections as ID pairs, e.g. `(0→1)`, `(1→2)` |
| **`edge_index`** | same edge list stored as a tensor `[2, num_edges]` |
| **`.x`** | node feature matrix; one row = one node’s numbers |

Example — same edges two ways:

```
edge list:                 edge_index:
  0 → 1                    [[0, 1, 1],
  1 → 2                     [1, 2, 3]]
  1 → 3
```

Example — features:

```
x[0] = [0.3, 0.1, 0.8]    # node 0
x[1] = [0.6, 0.4, 0.9]    # node 1
```

In HeteroData: `data['Material'].x` = feature table for Material nodes only.

**Rule:** `.x` = what a node is (numbers). `edge_index` = who it connects to.

---

# 6. Pipeline split (correct)

```
TTL
 │
 ▼
Parse triples
 │
 ▼
IDs + types + features + edge list
 │
 ├──────────────────────────────┐
 ▼                              ▼
R-GCN (Way 1 — direct)         HGT
x                              HeteroData
edge_index                       = per-type .x
edge_type                        + per-relation edge_index
                               → hetero_data.pt
```

Same KG. Two exports. **No** “must flatten HeteroData for R-GCN”.

---

# 7. R-GCN — Way 1 (preferred)

Build **directly from triples**. Do not go through HeteroData.

### What R-GCN needs

| Tensor | Shape | Role |
|--------|-------|------|
| `x` | `[num_nodes, feat_dim]` | all node features (global IDs) |
| `edge_index` | `[2, num_edges]` | edge list as a tensor |
| `edge_type` | `[num_edges]` | relation id per edge |

Toy example:

```
edge_index = [[0, 1, 1],
              [1, 2, 3]]
edge_type  = [0, 1, 2]
```

= `0→1` (hasDisease), `1→2` (hasSymptom), `1→3` (treatedBy).

### Recipe

```
1. Parse TTL → triples
2. Assign global node IDs + relation IDs
3. Build edge list of (src, dst, rel_id)
4. Pack:
     edge_index = [[src...], [dst...]]
     edge_type  = [rel_id...]
     x          = feature rows (or learnable embeddings)
5. Save x.pt / edge_index.pt / edge_type.pt (+ mappings)
6. Train: RGCNConv(x, edge_index, edge_type)
```

### Why not flatten HeteroData?

Flattening works but is a detour: build typed graph → undo typing.  
Way 1 is simpler and keeps R-GCN independent of the HGT export.

---

# 8. HGT — HeteroData

**HeteroData** = PyG container for a **heterogeneous** graph:

- many node types → each has its own `.x`  
- many relations → each has its own `edge_index`  

It is storage layout, not an algorithm.

### Structure (toy)

```
HeteroData
├── Plant.x
├── Disease.x
├── Symptom.x
├── Treatment.x
├── (Plant, hasDisease, Disease).edge_index
├── (Disease, hasSymptom, Symptom).edge_index
└── (Disease, treatedBy, Treatment).edge_index
```

One `edge_index` column = one link triple with **local** IDs.

### HeteroData vs `.pt`

```
HeteroData  (in RAM)
    │ torch.save
    ▼
hetero_data.pt  (on disk)
```

Same object. `.pt` is only the file.

### Your battery HeteroData

```
HeteroData
├── Material.x, Crystal.x, UnitCell.x, AtomicSite.x, ...
├── (Material, hasStructure, Crystal).edge_index
├── (Crystal, hasUnitCell, UnitCell).edge_index
├── (AtomicSite, hasBondTo, AtomicSite).edge_index  (+ edge_attr distance)
└── ...
```

Saved as `output/hetero_data.pt` with JSON maps beside it.

### How HGT uses it

```python
data = torch.load("hetero_data.pt", weights_only=False)
x_dict = {t: data[t].x for t in data.node_types}
edge_index_dict = {e: data[e].edge_index for e in data.edge_types}
# out_dict = hgt(x_dict, edge_index_dict)
```

Decode predictions with `node_mapping.json`.

---

# 9. Where features come from

RDF entities are names, not vectors. Options:

1. **Learnable embeddings** — `Embedding(node_id)` (common for pure KGs)  
2. **Attributes from RDF** — band gap, lattice, coords (your battery `Type.x`)  
3. **Pretrained vectors** — optional  

Any of these fill `.x` / global `x`.

---

# 10. Lookup: tensor → triple

```python
nodes = json.load(open("node_mapping.json"))
rels  = json.load(open("relation_mapping.json"))

src = data["Plant", "hasDisease", "Disease"].edge_index[0, k].item()
dst = data["Plant", "hasDisease", "Disease"].edge_index[1, k].item()

s = nodes["Plant"][src]["uri"]
p = rels["Plant -> hasDisease -> Disease"]
o = nodes["Disease"][dst]["uri"]
# recovered (s, p, o)
```

---

# 11. What to keep

```
battery_kg.ttl                 ← source RDF (do not overwrite)

# HGT
hetero_data.pt
node_mapping.json
relation_mapping.json
edge_mapping.json
feature_description.json

# R-GCN (Way 1 — export separately from the same triples)
x.pt
edge_index.pt
edge_type.pt
(+ its own node/relation mappings with global IDs)
```

---

# Summary

| Question | Answer |
|----------|--------|
| What do we parse? | RDF triples `(s,p,o)` + types + numeric literals |
| What is an edge list? | List of ID→ID links |
| What is `edge_index`? | Edge list as a `[2, E]` tensor |
| What is `.x`? | Node feature matrix (one row per node) |
| What is HeteroData? | Typed `.x` tables + per-relation `edge_index` (HGT) |
| What is `.pt`? | Saved HeteroData on disk |
| HGT input? | HeteroData directly |
| R-GCN input? | `x`, `edge_index`, `edge_type` built **directly from triples** (Way 1) |
| Flatten HeteroData for R-GCN? | Possible, not preferred |
| Why maps? | Integers ↔ original RDF URIs |
