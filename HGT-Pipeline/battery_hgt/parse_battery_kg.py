#!/usr/bin/env python3
"""
parse_battery_kg.py — Parse BattGPT battery KG (TTL/RDF) into HGT edge CSVs
===========================================================================

Reads battery_kg.ttl / battery_kg.rdf and writes typed-edge CSVs + schema.json
for train.py.

Outputs (default: data/BatteryKG/)
  schema.json, nodes.csv, edges_{train,val,test,all}.csv

Usage
-----
  python parse_battery_kg.py \\
    -i ../../battery_kg.ttl \\
    --out-dir data/BatteryKG
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

from rdflib import RDF, RDFS, BNode, Graph, Literal, URIRef


# URI path segment → node type
_PATH_TYPE = {
    "element": "element",
    "material": "material",
    "crystal": "crystal",
    "spacegroup": "space_group",
    "crystalsystem": "crystal_system",
    "site": "atomic_site",
    "species": "atomic_species",
    "property": "property",
    "unitcell": "unit_cell",
}

# rdf:type local name → node type
_RDF_TYPE_MAP = {
    "Element": "element",
    "Substance": "material",
    "Crystal": "crystal",
    "SpaceGroup": "space_group",
    "CrystalSystem": "crystal_system",
    "AtomicSite": "atomic_site",
    "AtomicSpecies": "atomic_species",
    "UnitCell": "unit_cell",
}

# Predicates to skip (annotation / schema noise)
_SKIP_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#imports",
    "http://www.w3.org/2002/07/owl#versionInfo",
    "http://www.w3.org/2002/07/owl#inverseOf",
    "http://www.w3.org/2002/07/owl#Ontology",
    "http://purl.org/dc/terms/source",
    "http://www.w3.org/ns/prov#wasGeneratedBy",
}

# Keep only these object-property local names (battery-relevant)
_KEEP_RELS = {
    "belongsToElectrode",
    "containsSite",
    "containsSpecies",
    "hasBondTo",
    "hasBandGap",
    "hasEnergyAboveHull",
    "hasFormationEnergy",
    "hasElement",
    "hasStructure",
    "hasSpaceGroup",
    "hasCrystalSystem",
    "usesMaterial",
    "EMMO_e1097637_70d2_4895_973f_2396f04fa204",  # hasProperty
}


def _local(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _compact_id(uri: str) -> str:
    """Stable short id, e.g. material/mp-19017 or element/Li."""
    s = str(uri)
    m = re.search(r"/kg/([^/]+)/(.+)$", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return _local(s)


def _sanitize_type(t: str) -> str:
    return t.lower().replace("/", "_").replace(" ", "_").replace("-", "_")


def _infer_type_from_uri(uri: str) -> str | None:
    s = str(uri)
    m = re.search(r"/kg/([^/]+)/", s)
    if m and m.group(1) in _PATH_TYPE:
        return _PATH_TYPE[m.group(1)]
    # battery electrode / EMMO individuals
    if "battery#" in s or "/battery/" in s:
        return "electrode"
    if "emmo#" in s or "/emmo/" in s:
        return "property"
    return None


def _infer_type_from_rdf_types(g: Graph, node: URIRef) -> str | None:
    for t in g.objects(node, RDF.type):
        local = _local(str(t))
        if local in _RDF_TYPE_MAP:
            return _RDF_TYPE_MAP[local]
    return None


def _parse_formula_elements(formula: str) -> list[str]:
    """'LiFePO4' → ['Li','Fe','P','O']; ignores counts and parentheses groups lightly."""
    # Expand simple groups like (PO4)3 → PO4PO4PO4 then strip digits
    def expand(f: str) -> str:
        while True:
            m = re.search(r"\(([A-Za-z0-9]+)\)(\d*)", f)
            if not m:
                break
            body, n = m.group(1), int(m.group(2) or "1")
            f = f[: m.start()] + body * n + f[m.end() :]
        return f

    f = expand(formula)
    return re.findall(r"[A-Z][a-z]?", f)


def parse_battery_kg(path: str) -> tuple[dict, dict, list[dict]]:
    g = Graph()
    ext = os.path.splitext(path)[1].lower()
    fmt = "turtle" if ext == ".ttl" else "xml"
    print(f"Parsing {path} [{fmt}] ...")
    g.parse(path, format=fmt)
    print(f"  Triples: {len(g):,}")

    node_type_map: dict[str, str] = {}
    node_label_map: dict[str, str] = {}
    formula_map: dict[str, str] = {}  # material_id → formula
    typed_edges: list[dict] = []

    def ensure_node(uri: URIRef) -> str | None:
        nid = _compact_id(str(uri))
        if nid in node_type_map:
            return nid
        ntype = _infer_type_from_rdf_types(g, uri) or _infer_type_from_uri(str(uri))
        if ntype is None:
            return None
        ntype = _sanitize_type(ntype)
        labels = list(g.objects(uri, RDFS.label))
        label = str(labels[0]).strip() if labels else nid
        node_type_map[nid] = ntype
        node_label_map[nid] = label
        return nid

    # Register typed individuals first
    for s, _, t in g.triples((None, RDF.type, None)):
        if isinstance(s, BNode):
            continue
        local = _local(str(t))
        if local in _RDF_TYPE_MAP or _infer_type_from_uri(str(s)):
            ensure_node(s)

    # Collect formulas for materials
    for s, p, o in g:
        if _local(str(p)) == "hasFormula" and isinstance(o, Literal):
            nid = ensure_node(s)
            if nid:
                formula = str(o).strip()
                formula_map[nid] = formula
                # Prefer formula as the searchable name
                node_label_map[nid] = formula

    # Object-property edges
    skip_lit = skip_bn = skip_pred = skip_rel = 0
    for s, p, o in g:
        p_uri = str(p)
        if p_uri in _SKIP_PREDICATES:
            skip_pred += 1
            continue
        if isinstance(s, BNode) or isinstance(o, BNode):
            skip_bn += 1
            continue
        if isinstance(o, Literal):
            skip_lit += 1
            continue
        if not isinstance(o, URIRef):
            continue

        rel = _local(p_uri)
        if rel not in _KEEP_RELS and not rel.startswith("has") and not rel.startswith("contains"):
            # drop schema / unknown noise
            skip_rel += 1
            continue
        if rel in ("type", "label", "comment"):
            skip_rel += 1
            continue

        h_id = ensure_node(s)
        t_id = ensure_node(o)
        if h_id is None or t_id is None:
            continue
        if h_id == t_id:
            continue

        typed_edges.append({
            "head_id": h_id,
            "head_type": node_type_map[h_id],
            "relation": _sanitize_type(rel),
            "tail_id": t_id,
            "tail_type": node_type_map[t_id],
        })

    # Synthetic material → element edges from formulas (helps composition lookup)
    for mid, formula in formula_map.items():
        elems = _parse_formula_elements(formula)
        for el in elems:
            eid = f"element/{el}"
            if eid not in node_type_map:
                # create element node if missing
                node_type_map[eid] = "element"
                node_label_map[eid] = el
            typed_edges.append({
                "head_id": mid,
                "head_type": node_type_map[mid],
                "relation": "contains_element",
                "tail_id": eid,
                "tail_type": "element",
            })

    # Deduplicate edges
    seen = set()
    uniq = []
    for e in typed_edges:
        key = (e["head_id"], e["relation"], e["tail_id"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    typed_edges = uniq

    print(f"  Nodes : {len(node_type_map):,}")
    print(f"  Edges : {len(typed_edges):,}")
    print(f"  Skip lit={skip_lit:,} bnode={skip_bn:,} pred={skip_pred:,} rel={skip_rel:,}")
    print(f"  Materials with formula: {len(formula_map)}")
    return node_type_map, node_label_map, typed_edges


def build_schema(node_type_map: dict, typed_edges: list[dict]) -> dict:
    node_types = sorted(set(node_type_map.values()))
    relation_types = sorted(set(e["relation"] for e in typed_edges))
    meta_counts: dict[tuple, int] = defaultdict(int)
    for e in typed_edges:
        meta_counts[(e["head_type"], e["relation"], e["tail_type"])] += 1
    meta_relations = [
        {"head_type": ht, "relation": rel, "tail_type": tt, "count": cnt}
        for (ht, rel, tt), cnt in sorted(meta_counts.items(), key=lambda x: -x[1])
    ]
    return {
        "node_types": node_types,
        "relation_types": relation_types,
        "meta_relations": meta_relations,
    }


def _write_csv(rows: list[dict], path: str, fields: list[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {len(rows):>8,} → {path}")


def main():
    p = argparse.ArgumentParser(description="Parse BattGPT battery KG for HGT.")
    p.add_argument(
        "-i", "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "battery_kg.ttl"),
        help="Path to battery_kg.ttl or .rdf",
    )
    p.add_argument("--out-dir", default=os.path.join("data", "BatteryKG"))
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    path = os.path.abspath(args.input)
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    node_type_map, node_label_map, edges = parse_battery_kg(path)
    if not edges:
        print("Error: no edges extracted.", file=sys.stderr)
        sys.exit(1)

    schema = build_schema(node_type_map, edges)
    print("\n  Node types:", schema["node_types"])
    print("  Relations :", schema["relation_types"][:20], "...")

    random.seed(args.seed)
    random.shuffle(edges)
    n = len(edges)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    train_e = edges[:n_train]
    val_e = edges[n_train : n_train + n_val]
    test_e = edges[n_train + n_val :]

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    nodes = [
        {"node_id": nid, "node_type": ntype, "node_name": node_label_map.get(nid, nid)}
        for nid, ntype in sorted(node_type_map.items())
    ]
    _write_csv(nodes, os.path.join(out, "nodes.csv"), ["node_id", "node_type", "node_name"])
    _write_csv(train_e, os.path.join(out, "edges_train.csv"),
               ["head_id", "head_type", "relation", "tail_id", "tail_type"])
    _write_csv(val_e, os.path.join(out, "edges_val.csv"),
               ["head_id", "head_type", "relation", "tail_id", "tail_type"])
    _write_csv(test_e, os.path.join(out, "edges_test.csv"),
               ["head_id", "head_type", "relation", "tail_id", "tail_type"])
    _write_csv(edges, os.path.join(out, "edges_all.csv"),
               ["head_id", "head_type", "relation", "tail_id", "tail_type"])

    with open(os.path.join(out, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"  Saved schema → {os.path.join(out, 'schema.json')}")
    print("\nDone. Train with:")
    print(f"  python train.py --data-dir {out} --model-dir models/battery --output-dir outputs/battery")


if __name__ == "__main__":
    main()
