# Graph Reconstruction Report — Stage 2.3.2

## Executive Summary
- **Reconstruction Status**: **PASSED (100% Topologically Recovered)**
- **Source Input**: `stage3_export/` artifacts ONLY (`hetero_data.pt`, `node_mapping.json`, `edge_mapping.json`, `relation_mapping.json`)
- **Execution Time**: 3.44 seconds
- **Canonical RDF Cross-Check**: VERIFIED

## Recovered Node Topology
| Node Type | Tensor Matrix Shape `x` | Recovered Mapping Count | URI Reversibility | Status |
| :--- | :--- | :--- | :--- | :--- |
| `Material` | `[5000, 11]` | 5,000 | 100% Reversible | **VERIFIED** |
| `Crystal` | `[5000, 1]` | 5,000 | 100% Reversible | **VERIFIED** |
| `UnitCell` | `[5000, 6]` | 5,000 | 100% Reversible | **VERIFIED** |
| `AtomicSite` | `[156630, 4]` | 156,630 | 100% Reversible | **VERIFIED** |
| `AtomicSpecies` | `[83, 5]` | 83 | 100% Reversible | **VERIFIED** |
| `Element` | `[83, 7]` | 83 | 100% Reversible | **VERIFIED** |
| `SpaceGroup` | `[155, 1]` | 155 | 100% Reversible | **VERIFIED** |
| `CrystalSystem` | `[7, 1]` | 7 | 100% Reversible | **VERIFIED** |
| `Property` | `[35565, 1]` | 35,565 | 100% Reversible | **VERIFIED** |
| `BatteryRole` | `[4, 1]` | 4 | 100% Reversible | **VERIFIED** |

## Recovered Edge Topology & Relation Mappings
| PyG Edge Triplet | Edge Tensor Shape `edge_index` | Edge Mapping Count | Reversible Bond URIs | Status |
| :--- | :--- | :--- | :--- | :--- |
| `Material -> hasStructure -> Crystal` | `[2, 5000]` | 5,000 | N/A | **VERIFIED** |
| `Crystal -> hasUnitCell -> UnitCell` | `[2, 5000]` | 5,000 | N/A | **VERIFIED** |
| `UnitCell -> containsSite -> AtomicSite` | `[2, 156630]` | 156,630 | N/A | **VERIFIED** |
| `AtomicSite -> containsSpecies -> AtomicSpecies` | `[2, 156630]` | 156,630 | N/A | **VERIFIED** |
| `AtomicSpecies -> hasElement -> Element` | `[2, 156630]` | 156,630 | N/A | **VERIFIED** |
| `AtomicSite -> hasBondTo -> AtomicSite` | `[2, 596434]` | 596,434 | Yes (reified `battgpt:CrystalBond`) | **VERIFIED** |
| `Material -> hasProperty -> Property` | `[2, 35565]` | 35,565 | N/A | **VERIFIED** |
| `Crystal -> hasSpaceGroup -> SpaceGroup` | `[2, 5000]` | 5,000 | N/A | **VERIFIED** |
| `Crystal -> hasCrystalSystem -> CrystalSystem` | `[2, 5000]` | 5,000 | N/A | **VERIFIED** |
| `Material -> belongsToElectrode -> BatteryRole` | `[2, 3626]` | 3,626 | N/A | **VERIFIED** |
| `BatteryRole -> usesMaterial -> Material` | `[2, 3626]` | 3,626 | N/A | **VERIFIED** |

## Verification Diagnostics
1. **Node URI Recovery**: 100% of tensor nodes map directly to valid RDF URIs (`https://w3id.org/battgpt/kg/...`).
2. **Bond URI Recovery**: Every `AtomicSite -> hasBondTo -> AtomicSite` edge tensor index maps to an explicit reified `battgpt:CrystalBond` URI.
3. **Relation Recovery**: 100% of multi-relational edge types map to canonical EMMO/BattGPT Object Property IRIs.
4. **Feature Integrity**: Zero NaN or Inf values detected in any node feature matrix or edge attribute tensor.
