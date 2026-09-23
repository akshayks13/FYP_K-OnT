# References — RDF/TTL → HeteroData → GNN

Proof that converting a knowledge graph (RDF triples) into numeric graph tensors (`.x`, `edge_index`, HeteroData) for R-GCN/HGT is standard practice.

---

## What our pipeline does

```
battery_kg.ttl  →  parse triples  →  assign IDs  →  HeteroData (.x + edge_index)  →  HGT / R-GCN
```

- Link triples `(s, p, entity)` → `edge_index`
- Literal triples `(s, p, number)` → node feature rows `.x`
- Names/URIs → JSON mappings (not in tensors)

---

## Core GNN methods (why the tensor format is correct)

### R-GCN — Schlichtkrull et al., 2018
**Title:** Modeling Relational Data with Graph Convolutional Networks  
**Link:** https://arxiv.org/abs/1703.06103

**Proves:**
- KGs are input as **triples** `(head, relation, tail)`
- Graph = **entity IDs + relation IDs + edges**
- Standard tensors: `x`, `edge_index`, `edge_type`
- Task: **link prediction** (recover missing triples)

**Maps to us:** R-GCN path — triples → global IDs → flat graph tensors.

---

### HGT — Hu et al., 2020 (WWW)
**Title:** Heterogeneous Graph Transformer  
**Link:** https://arxiv.org/abs/2003.01332

**Proves:**
- Heterogeneous KGs need **typed nodes** and **typed edges**
- Separate parameters per `(src_type, relation, dst_type)`
- PyG `HeteroData` + `HGTConv` implement this

**Maps to us:** HGT path — per-type `.x` + per-relation `edge_index` in `hetero_data.pt`.

---

## Proof: HeteroData is the correct HGT input

HGT does **not** read TTL or JSON. It reads exactly what `HeteroData` stores:

| HGT needs | HeteroData field | Our file |
|-----------|------------------|----------|
| Node types + counts | `data.node_types`, `data['Material'].x` shape | ✅ 10 types |
| Typed edges | `data['Material', 'hasStructure', 'Crystal'].edge_index` | ✅ 11 relations |
| Meta-relation `(src, rel, dst)` | `data.edge_types` | ✅ e.g. `('Material','hasStructure','Crystal')` |

### HGT paper (Hu et al., 2020)

The model is defined on a **heterogeneous graph** with:
- different **node types** (each with its own representation space)
- different **edge types** / meta-relations `⟨τ(s), φ(e), τ(t)⟩`
- message passing **per relation type**, not one flat adjacency matrix

That is precisely the structure `HeteroData` encodes: separate `x` per type + separate `edge_index` per `(src_type, relation, dst_type)`.

**Citation:** Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous Graph Transformer. *WWW 2020*. https://arxiv.org/abs/2003.01332

### PyTorch Geometric (official implementation)

PyG documents `HeteroData` as the standard container for heterogeneous GNNs and provides `HGTConv`, which expects:

```python
conv(h_dict, edge_index_dict)
# h_dict:          {node_type: features [N, D]}
# edge_index_dict: {(src, rel, dst): [2, E]}
```

Loading from disk:

```python
data = torch.load("hetero_data.pt")
edge_index_dict = {e: data[e].edge_index for e in data.edge_types}
```

**Citation:** PyG HeteroData tutorial — https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html  
**PyG HGTConv:** https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HGTConv.html

### Our codebase (HGT-Pipeline)

`models/hgt.py` — HGT encoder signature:

```python
def encode(self, x_dict, edge_index_dict):
    # x_dict:          {node_type: node indices}
    # edge_index_dict: {(src_type, rel, dst_type): [2, E]}
```

`train.py` — builds `edge_index_dict` as `{(src_type, rel, dst_type): LongTensor[2, E]}` — same keys as `HeteroData.edge_types`.

`rdf_to_hetero.py` — exports RDF → `HeteroData` → `hetero_data.pt`.

**Conclusion:** `hetero_data.pt` (HeteroData) is not an arbitrary format — it is the PyG-standard packaging of the exact typed graph structure HGT requires.

---

## RDF → graph tensors (same workflow as us)

### AutoRDF2GML — 2024
**Title:** Bridging RDF Knowledge Graphs with Graph Neural Networks for Semantically-Rich Recommender Systems  
**Link:** https://arxiv.org/html/2506.08743v1

**Proves:**
- RDF KG → **heterogeneous graph** for GNN
- Object properties → **edges** (`edge_index`)
- Literal/datatype properties → **node features** (`x`)
- Explicit pipeline: `KG → parse → HG(x, edge_index) → GNN`

**Maps to us:** Near-identical conversion logic to our `rdf_to_hetero.py`.

---

### EPA OntoSearcher
**Link:** https://github.com/USEPA/OntoSearcher_Collaborative

**Proves:**
- Reads **RDF/TTL**
- Converts to **PyG HeteroData**
- Trains **R-GCN** on ontology/material KGs
- Has `improved_rdf_hetero_converter.py`

**Maps to us:** Same RDF → HeteroData → GNN pipeline, different domain (nanomaterials).

---

### PyG RDF loader (minimal example)
**Link:** https://gist.github.com/aamedina/20d42bda351175ea088fad0dc328e2a5

**Proves:**
- Parse RDF with RDFLib
- Assign node/relation IDs
- Build `edge_index` + `edge_type` tensors

**Maps to us:** Minimal reference implementation of triple → tensor conversion.

---

## Materials-science KG (our domain)

### MatKG — Venkatram et al., 2024
**Title:** MatKG: An autonomously generated knowledge graph in Material Science  
**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10874416/

**Proves:**
- Materials KG stored as **RDF triples**
- Triples used for **link prediction** / graph learning
- Standard KG representation in materials informatics

**Maps to us:** Same RDF/triple source format; we extend to GNN tensors (HGT/R-GCN).

---

### MatKG + Graph Representation Learning — OpenReview
**Link:** https://openreview.net/forum?id=xyJ_0-WCIZN

**Proves:**
- MatKG triples → **graph representation learning** for link prediction

**Maps to us:** Natural predecessor; our step is RDF → HeteroData for GNN.

---

## Implementation standard

### PyTorch Geometric — HeteroData tutorial
**Link:** https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html

**Proves:**
- `HeteroData` stores per-type `x` and per-relation `edge_index`
- Official PyG format for heterogeneous GNNs (including HGT)

**Maps to us:** `hetero_data.pt` is standard PyG HeteroData on disk.

---

## Why RDF2Vec / OWL2Vec are not required

RDF2Vec and OWL2Vec are **alternative** KG→vector methods (random walks + Word2Vec). They are not prerequisites for GNN.

### At HeteroData stage (what we do now) — no random numbers

Numbers come **directly from the KG**, not typed in by hand:

| Source | Example | Becomes |
|--------|---------|---------|
| Literal triple | `(LiCoO2, hasBandGap, 0.66)` | `Material.x[0][0] = 0.66` |
| Literal triple | `(unitCell, hasLatticea, 4.91)` | `UnitCell.x[0][0] = 4.91` |
| Link triple | `(LiCoO2, hasStructure, crystal)` | `edge_index` column `[0, 0]` |
| Bond triple | `(site0, hasBondTo, site2)` + distance | `edge_index` + `edge_attr = 2.078` |

Names (`LiCoO2`, URIs) → JSON. **Not random.**

Some `.x` rows may be **zeros** if that type has no literals in RDF (e.g. Element) — still not random input; structure comes from `edge_index`.

### Later (GNN training) — not part of HeteroData export

When HGT runs **later**, it may initialize **learnable** node vectors randomly, then **updates** them using `edge_index` during training. That is inside the model, not something you manually enter when building `hetero_data.pt`.

RDF2Vec solves “KG → vectors” in one preprocessing step. We instead: **KG → HeteroData (deterministic)** → **GNN learns vectors later (optional step)**.

---

## Suggested citations (report/thesis)

**Minimum set:**
1. Schlichtkrull et al., 2018 — R-GCN (multi-relational graph tensors)
2. Hu et al., 2020 — HGT (heterogeneous typed graph)
3. AutoRDF2GML, 2024 — RDF → heterogeneous graph for GNN (direct pipeline precedent)

**Domain + implementation:**
4. Venkatram et al., 2024 — MatKG (materials RDF KG)
5. PyTorch Geometric docs — HeteroData format

**Optional (near-identical implementation):**
6. EPA OntoSearcher — RDF/TTL → HeteroData → R-GCN

---

## One-line justification

> We parse the BattGPT battery knowledge graph (RDF/TTL) into PyTorch Geometric HeteroData—typed node feature matrices (`.x`) and typed edge index tensors (`edge_index`)—following the established knowledge-graph-to-GNN pipeline described in R-GCN (Schlichtkrull et al., 2018), HGT (Hu et al., 2020), and recent RDF-to-GNN conversion frameworks (AutoRDF2GML, 2024).
