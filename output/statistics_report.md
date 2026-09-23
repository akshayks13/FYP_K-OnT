# Knowledge Graph Statistics Report — Stage 2.3.2

## Overview
- **Total Authentic Materials**: 5,000
- **Total Atomic Sites**: 156,630
- **Total Crystal Connectivity Bonds**: 596,434
- **Total RDF Triples**: 8,365,553
- **Pipeline Execution Time**: 2668.74 seconds

## File Storage & Artifact Benchmarks
| Artifact | File Format | File Size (MB) | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| **Canonical RDF Turtle** | `.ttl` | 526.84 MB | `4c4981f4cab10f05...` |
| **RDF/XML** | `.rdf` | 911.78 MB | `5a2bc8ddd5835882...` |
| **JSON-LD** | `.jsonld` | 1161.39 MB | `30887008c6ed7e69...` |
| **HeteroData PyG Tensor** | `.pt` | 22.39 MB | `5213f996c7894fab...` |

## Battery Role Distribution
| Battery Role | Count | Percentage |
| :--- | :--- | :--- |
| `PositiveElectrode` | 3,294 | 65.88% |
| `Generic / Unclassified` | 1,374 | 27.48% |
| `NegativeElectrode` | 289 | 5.78% |
| `Electrolyte` | 42 | 0.84% |
| `Separator` | 1 | 0.02% |

## Crystal System Breakdown
| Crystal System | Count | Percentage |
| :--- | :--- | :--- |
| `Monoclinic` | 1,851 | 37.02% |
| `Triclinic` | 1,279 | 25.58% |
| `Orthorhombic` | 967 | 19.34% |
| `Tetragonal` | 344 | 6.88% |
| `Trigonal` | 306 | 6.12% |
| `Cubic` | 129 | 2.58% |
| `Hexagonal` | 124 | 2.48% |

## HeteroData Graph Tensor Benchmarks
| Node Type | Num Nodes | Feature Matrix Shape `x` |
| :--- | :--- | :--- |
| `Material` | 5,000 | `[5000, 11]` |
| `Crystal` | 5,000 | `[5000, 1]` |
| `UnitCell` | 5,000 | `[5000, 6]` |
| `AtomicSite` | 156,630 | `[156630, 4]` |
| `AtomicSpecies` | 83 | `[83, 5]` |
| `Element` | 83 | `[83, 7]` |
| `SpaceGroup` | 155 | `[155, 1]` |
| `CrystalSystem` | 7 | `[7, 1]` |
| `Property` | 35,565 | `[35565, 1]` |
| `BatteryRole` | 4 | `[4, 1]` |

| Edge Type | Edge Count | Edge Index Shape | Edge Attr Shape |
| :--- | :--- | :--- | :--- |
| `Material -> hasStructure -> Crystal` | 5,000 | `[2, 5000]` | `N/A` |
| `Crystal -> hasUnitCell -> UnitCell` | 5,000 | `[2, 5000]` | `N/A` |
| `UnitCell -> containsSite -> AtomicSite` | 156,630 | `[2, 156630]` | `N/A` |
| `AtomicSite -> containsSpecies -> AtomicSpecies` | 156,630 | `[2, 156630]` | `N/A` |
| `AtomicSpecies -> hasElement -> Element` | 156,630 | `[2, 156630]` | `N/A` |
| `AtomicSite -> hasBondTo -> AtomicSite` | 596,434 | `[2, 596434]` | `[596434, 1]` |
| `Material -> hasProperty -> Property` | 35,565 | `[2, 35565]` | `N/A` |
| `Crystal -> hasSpaceGroup -> SpaceGroup` | 5,000 | `[2, 5000]` | `N/A` |
| `Crystal -> hasCrystalSystem -> CrystalSystem` | 5,000 | `[2, 5000]` | `N/A` |
| `Material -> belongsToElectrode -> BatteryRole` | 3,626 | `[2, 3626]` | `N/A` |
| `BatteryRole -> usesMaterial -> Material` | 3,626 | `[2, 3626]` | `N/A` |
