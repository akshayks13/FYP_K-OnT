# Tensors in `hetero_data.pt` — detailed guide

In `hetero_data.pt`, every tensor is a **spreadsheet of one kind of graph data**.

Below: **what each table holds**, **what each row/column means**, and **which triples fill it**.

---

## Big picture

```
hetero_data.pt
│
├── 10 tables of NODES     (.x tensors)
├── 11 tables of EDGES     (edge_index tensors)
└── 1 table of BOND LENGTHS (edge_attr tensor)
```

**22 numeric tables total**, all inside one file.

---

# Part 1 — Node tables (`.x`)

**What the table is:** all nodes of **one type**, listed row by row.

**One row = one node = one vector of numbers.**

---

## `Material.x` — table of 5000 materials

**Shape:** `[5000, 11]` = 5000 rows, 11 columns

| Row | Node (from JSON) | What the row stores |
|-----|------------------|---------------------|
| 0 | LiCoO2 (mp-22526) | 11 material properties |
| 1 | LiFePO4 (mp-19017) | 11 material properties |
| … | … | … |

**Columns (11 numbers per material):**

| Col | Name | From RDF literal triple like… |
|-----|------|-------------------------------|
| 0 | band_gap_eV | `(material, band_gap, 0.6623)` |
| 1 | formation_energy_eV_atom | `(material, formation_energy, -1.7457)` |
| 2 | energy_above_hull_eV_atom | … |
| 3 | density_g_cm3 | … |
| 4 | volume_ang3 | … |
| 5 | e_fermi_eV | … |
| 6 | bulk_modulus_GPa | … |
| 7 | shear_modulus_GPa | … |
| 8 | total_magnetization_muB | … |
| 9 | is_metal_flag | `(material, isMetal, false)` → 0 or 1 |
| 10 | is_stable_flag | `(material, isStable, true)` → 0 or 1 |

**Real row 0:**

```
[0.6623, -1.7457, 0.0, 5.1214, 31.7337, 3.9562, 0.0, 0.0, 0.0, 0.0, 1.0]
```

**Table of:** materials. **Cell =** one property value for one material.

---

## `Crystal.x` — table of 5000 crystals

**Shape:** `[5000, 1]`

| Row | Node | Column 0 |
|-----|------|----------|
| 0 | crystal for LiCoO2 | `4.0` (num_sites) |
| 1 | crystal for LiFePO4 | … |

**Table of:** crystals. **One number each** (site count).

---

## `UnitCell.x` — table of 5000 unit cells

**Shape:** `[5000, 6]`

| Col | Meaning | From RDF |
|-----|---------|----------|
| 0 | lattice_a | hasLatticea |
| 1 | lattice_b | hasLatticeb |
| 2 | lattice_c | hasLatticec |
| 3 | alpha | hasAlpha |
| 4 | beta | hasBeta |
| 5 | gamma | hasGamma |

**Row 0 (LiCoO2):** `[4.9124, 4.9124, 4.9124, 33.25, 33.25, 33.25]`

**Table of:** unit cells. **Row =** one cell's lattice parameters.

---

## `AtomicSite.x` — table of 156,630 atomic sites

**Shape:** `[156630, 4]`

| Col | Meaning |
|-----|---------|
| 0 | fractional_x |
| 1 | fractional_y |
| 2 | fractional_z |
| 3 | coordination_number |

**Row 0:** Li site of LiCoO2 → `[0.0, 0.0, 0.0, 6.0]`

**Table of:** every atomic site in every material.

---

## `AtomicSpecies.x` — table of 83 species

**Shape:** `[83, 5]` — atomic_number, mass, electronegativity, valence, oxidation_state

**Table of:** species like `Li_0`, `Co_0`, `O_0`.

---

## `Element.x` — table of 83 elements

**Shape:** `[83, 7]` — periodic table features

**Table of:** Li, Co, O, Fe, …

---

## `SpaceGroup.x` — table of 155 space groups

**Shape:** `[155, 1]` — column = spacegroup_number (e.g. `166` for R-3m)

**Table of:** space groups (shared across many crystals).

---

## `CrystalSystem.x` — table of 7 crystal systems

**Shape:** `[7, 1]` — codes 0–6 (Trigonal, Orthorhombic, …)

**Table of:** crystal systems.

---

## `Property.x` — table of 35,565 property-value nodes

**Shape:** `[35565, 1]` — column = scalar_value

| Row | Meaning |
|-----|---------|
| 0 | band_gap of mp-22526 = 0.6623 |
| 1 | formation_energy of mp-22526 = -1.7457 |
| … | one row per (material, property) pair |

**Table of:** individual property measurements as separate nodes.

---

## `BatteryRole.x` — table of 4 roles

**Shape:** `[4, 1]` — codes 0–3

| Row | Role |
|-----|------|
| 0 | PositiveElectrode |
| 1 | NegativeElectrode |
| 2 | Electrolyte |
| 3 | Separator |

**Table of:** battery roles.

---

# Part 2 — Edge tables (`edge_index`)

**What the table is:** all **links of one relation type**.

**Shape always:** `[2, E]` — 2 rows, E columns.

**One column = one edge = one link triple.**

```
row 0 = source node IDs
row 1 = destination node IDs
```

**Most edges have NO extra numbers** — just the pair `[src, dst]`.

---

## Example: `(Material, hasStructure, Crystal).edge_index`

**Shape:** `[2, 5000]` — 5000 hasStructure links

```
        col0   col1   col2   ...
row0:  [  0,    1,    2,   ...]   ← Material IDs
row1:  [  0,    1,    2,   ...]   ← Crystal IDs
```

| Column | Triple it came from |
|--------|---------------------|
| 0 | Material 0 (LiCoO2) → Crystal 0 |
| 1 | Material 1 (LiFePO4) → Crystal 1 |

**Table of:** hasStructure connections. **Not** a table of materials — a table of **links**.

---

## All 11 edge tables (what each is a table of)

| edge_index tensor | Table of… | # links |
|-------------------|-----------|--------:|
| Material → hasStructure → Crystal | material–crystal links | 5000 |
| Crystal → hasUnitCell → UnitCell | crystal–cell links | 5000 |
| UnitCell → containsSite → AtomicSite | cell–site links | 156630 |
| AtomicSite → containsSpecies → AtomicSpecies | site–species links | 156630 |
| AtomicSpecies → hasElement → Element | species–element links | 156630 |
| AtomicSite → hasBondTo → AtomicSite | site–site bond links | 596434 |
| Material → hasProperty → Property | material–property links | 35565 |
| Crystal → hasSpaceGroup → SpaceGroup | crystal–spacegroup links | 5000 |
| Crystal → hasCrystalSystem → CrystalSystem | crystal–system links | 5000 |
| Material → belongsToElectrode → BatteryRole | material–role links | 3626 |
| BatteryRole → usesMaterial → Material | role–material links (inverse) | 3626 |

Each is its **own** table because HGT treats each relation differently.

---

# Part 3 — Bond distance table (`edge_attr`)

**Only for** `hasBondTo` edges.

**Shape:** `[596434, 1]`

| Row | Matches edge column | Value |
|-----|---------------------|-------|
| 0 | site 0 → site 2 | 2.0783 Å |
| 1 | site 0 → site 3 | 2.0783 Å |
| … | … | … |

**Table of:** bond distances — **one number per bond edge**.

This is the **only** edge table with extra numbers. All other edges are just ID pairs.

---

# Part 4 — How a triple lands in a cell

```
TRIPLE                                    LANDS IN

(material, hasBandGap, 0.66)         →  Material.x[row][col0] = 0.66

(material, hasStructure, crystal)    →  hasStructure edge_index[:, col] = [mat_id, cry_id]

(site0, hasBondTo, site2)            →  hasBondTo edge_index[:, col] = [0, 2]
                                         edge_attr[row] = 2.0783

(material, hasFormula, "LiCoO2")     →  NOT in any tensor → node_mapping.json
```

---

# Part 5 — Tensor vs vector

| Term | In your file |
|------|--------------|
| **Tensor** | whole table — e.g. all 5000 materials |
| **Vector** | one row — e.g. LiCoO2's 11 numbers |
| **Vector** | one edge — e.g. `[0, 0]` (2 numbers) |

```
Material.x          ← tensor (big table)
Material.x[0]       ← vector (one row = LiCoO2)
edge_index[:, 0]    ← vector (one column = one edge)
```

**Rule:** vector ⊂ tensor. Every vector is a tensor; not every tensor is a vector.

---

# Part 6 — What is NOT in any table

| Thing | Where |
|-------|--------|
| "LiCoO2" | node_mapping.json |
| URI | node_mapping.json |
| "hasStructure" label | relation_mapping.json |
| column name "band_gap_eV" | feature_description.json |

Tensors = **numbers + structure only**.

---

# Summary

**Each tensor is a table of either:**

1. **Nodes of one type** — one row per entity, columns = properties (`.x`)
2. **Edges of one relation** — one column per link, two rows = source and destination IDs (`edge_index`)
3. **Bond distances** — one row per bond (`edge_attr`)

**One file (`hetero_data.pt`) = one HeteroData object = ~22 tables inside.**

See also: `GNN_REPRESENTATION.md` (pipeline), `REFERENCES_GNN_PIPELINE.md` (papers).
