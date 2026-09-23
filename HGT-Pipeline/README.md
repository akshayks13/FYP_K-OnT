# HGT Pipeline for Battery Knowledge Graph

Heterogeneous Graph Transformer (HGT) link prediction on **`output/hetero_data.pt`**, using typed **`.x` features + `edge_index`**, then export node vectors for CrystaLLM.

```bash
cd battery_hgt/
```

| Script | Role |
|--------|------|
| `rdf_to_hetero.py` | TTL/RDF → PyG `HeteroData` |
| `hetero_graph.py` | Load HeteroData, trim edges, normalize `.x` |
| `train.py` | Train HGT from HeteroData (default) |
| `export_embeddings.py` | Write `export_package/node_embeddings.pt` |
| `convert_hetero_to_hgt.py` | Optional: HeteroData → CSVs (legacy) |
| `test.py` / `predict.py` | Evaluate / query trained model |

```bash
# Train directly from hetero_data.pt (uses Material.x, UnitCell.x, …)
python train.py --model-dir models/battery_v2 --output-dir outputs/battery_v2 \
  --embedding-dim 32 --num-heads 2 --num-layers 1 --epochs 30

python export_embeddings.py --model-dir models/battery_v2 \
  --out-dir models/battery_v2/export_package
```

Trim is still applied in memory (not by rewriting `hetero_data.pt`):
- drop AtomicSite / bond edges unless `--include-sites`
- cap Material→Property at `--max-property-edges` (default 20000)
- add `Material → contains_element → Element` from formulas
- Element / AtomicSpecies `.x` are all-zero → ID embeddings

Legacy CSV path:

```bash
python convert_hetero_to_hgt.py --output-dir ../../output --out-dir data/BatteryKG_v2
python train.py --from-csv --data-dir data/BatteryKG_v2 --model-dir models/battery_v2
```

See `/Users/akshayks/Desktop/FinalYearProj/HGT_CRYSTALLM_INTEGRATION.md` for CrystaLLM wiring.
