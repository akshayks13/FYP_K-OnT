#!/usr/bin/env python3
"""
rdf_to_hetero.py — Parse BattGPT battery_kg.ttl / .rdf → PyG HeteroData
=====================================================================

Converts RDF/TTL into the Stage-2.3.2 graph layout used by HGT:

  hetero_data.pt
  node_mapping.json, edge_mapping.json, relation_mapping.json
  feature_description.json, ontology_mapping.json, metadata.json

Unlike the original MaterialRecord exporter, this reads the RDF file itself.

Usage
-----
  # smoke test (first N materials)
  python rdf_to_hetero.py -i ../../output/battery_kg.ttl \\
      --out-dir ../../output/hetero_from_rdf --max-materials 20

  # full graph (slow / memory-heavy on 8M+ triples)
  python rdf_to_hetero.py -i ../../output/battery_kg.ttl \\
      --out-dir ../../output/hetero_from_rdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

try:
    from pymatgen.core import Element as PMGElement
except ImportError:  # pragma: no cover
    PMGElement = None


BATTGPT = "https://w3id.org/battgpt/kg#"
KG = "https://w3id.org/battgpt/kg/"
EMMO_HAS_PROPERTY = "https://w3id.org/emmo#EMMO_e1097637_70d2_4895_973f_2396f04fa204"
EMMO_HAS_VALUE = "https://w3id.org/emmo#EMMO_faf79f53_749d_40b2_807c_d34244c192f4"
EMMO_QUANTITY = "https://w3id.org/emmo#EMMO_b7bcff25_ffc3_474e_9ab5_01b1664bd4ba"
CHSUB = "https://w3id.org/emmo/domain/chemical-substance#"
CRYST = "https://w3id.org/emmo/domain/crystallography#"

ROLE_INDIVIDUALS = {
    "PositiveElectrodeRoleIndividual": "PositiveElectrode",
    "NegativeElectrodeRoleIndividual": "NegativeElectrode",
    "ElectrolyteRoleIndividual": "Electrolyte",
    "SeparatorRoleIndividual": "Separator",
}

RELATION_URIS = {
    "Material -> hasStructure -> Crystal": f"{BATTGPT}hasStructure",
    "Crystal -> hasUnitCell -> UnitCell": f"{BATTGPT}hasUnitCell",
    "UnitCell -> containsSite -> AtomicSite": f"{BATTGPT}containsSite",
    "AtomicSite -> containsSpecies -> AtomicSpecies": f"{BATTGPT}containsSpecies",
    "AtomicSpecies -> hasElement -> Element": f"{BATTGPT}hasElement",
    "AtomicSite -> hasBondTo -> AtomicSite": f"{BATTGPT}hasBondTo",
    "Material -> hasProperty -> Property": EMMO_HAS_PROPERTY,
    "Crystal -> hasSpaceGroup -> SpaceGroup": f"{BATTGPT}hasSpaceGroup",
    "Crystal -> hasCrystalSystem -> CrystalSystem": f"{BATTGPT}hasCrystalSystem",
    "Material -> belongsToElectrode -> BatteryRole": f"{BATTGPT}belongsToElectrode",
    "BatteryRole -> usesMaterial -> Material": f"{BATTGPT}usesMaterial",
}

ONTOLOGY_MAPPING = {
    "Material": f"{CHSUB}Substance",
    "Crystal": f"{CRYST}Crystal",
    "UnitCell": f"{CRYST}UnitCell",
    "AtomicSite": f"{CRYST}AtomicSite",
    "AtomicSpecies": f"{CHSUB}AtomicSpecies",
    "Element": f"{CHSUB}Element",
    "SpaceGroup": f"{CRYST}SpaceGroup",
    "CrystalSystem": f"{CRYST}CrystalSystem",
    "Property": EMMO_QUANTITY,
    "BatteryRole": f"{BATTGPT}BatteryRole",
    "CrystalBond": f"{BATTGPT}CrystalBond",
}

MAT_PROP_KEYS = [
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "volume",
    "e_fermi",
    "bulk_modulus",
    "shear_modulus",
    "total_magnetization",
]

_PATH_TYPE = {
    "material": "Material",
    "crystal": "Crystal",
    "unitcell": "UnitCell",
    "site": "AtomicSite",
    "species": "AtomicSpecies",
    "element": "Element",
    "spacegroup": "SpaceGroup",
    "crystalsystem": "CrystalSystem",
    "property": "Property",
    "bond": "CrystalBond",
}

_TYPE_LOCAL = {
    "Substance": "Material",
    "Crystal": "Crystal",
    "UnitCell": "UnitCell",
    "AtomicSite": "AtomicSite",
    "AtomicSpecies": "AtomicSpecies",
    "Element": "Element",
    "SpaceGroup": "SpaceGroup",
    "CrystalSystem": "CrystalSystem",
    "CrystalBond": "CrystalBond",
    "PositiveElectrodeRole": "BatteryRole",
    "NegativeElectrodeRole": "BatteryRole",
    "ElectrolyteRole": "BatteryRole",
    "SeparatorRole": "BatteryRole",
    "BatteryRole": "BatteryRole",
}


def _local(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _path_parts(uri: str) -> tuple[str | None, str | None]:
    m = re.search(r"/kg/([^/]+)/(.+)$", uri)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _infer_type(uri: str, type_uris: list[str]) -> str | None:
    for t in type_uris:
        loc = _local(t)
        if loc in _TYPE_LOCAL:
            return _TYPE_LOCAL[loc]
        if loc in ROLE_INDIVIDUALS or loc.endswith("RoleIndividual"):
            return "BatteryRole"
        if t == EMMO_QUANTITY or loc.startswith("EMMO_b7bcff25"):
            return "Property"
    kind, _ = _path_parts(uri)
    if kind in _PATH_TYPE:
        return _PATH_TYPE[kind]
    if uri.startswith(BATTGPT) and _local(uri) in ROLE_INDIVIDUALS:
        return "BatteryRole"
    return None


def _as_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, Literal):
        try:
            return float(v)
        except Exception:
            s = str(v).strip().lower()
            if s == "true":
                return 1.0
            if s == "false":
                return 0.0
            return None
    try:
        return float(v)
    except Exception:
        return None


def element_symbol_from_uri(uri: str) -> str:
    """Last path segment → element symbol. `.../element/Li` or `.../species/Li_0` → Li."""
    tail = uri.rstrip("/").rsplit("/", 1)[-1] if uri else ""
    tail = tail.split("#")[-1]
    if "_" in tail:
        head = tail.split("_", 1)[0]
        if re.fullmatch(r"[A-Z][a-z]?", head):
            return head
    if re.fullmatch(r"[A-Z][a-z]?", tail):
        return tail
    loc = _local(uri)
    m = re.fullmatch(r"([A-Z][a-z]?)", loc)
    return m.group(1) if m else loc


def _element_feats(symbol: str) -> list[float]:
    """Periodic-table row: Z, mass, group, row, electronegativity, radius, valence."""
    symbol = element_symbol_from_uri(symbol) if "/" in symbol or "#" in symbol else symbol
    if not symbol:
        return [0.0] * 7
    if PMGElement is None:
        raise RuntimeError(
            "pymatgen is required to fill Element.x / AtomicSpecies.x. "
            "Install it in the project venv (pip install pymatgen)."
        )
    try:
        el = PMGElement(symbol)
        valence = 0.0
        if el.common_oxidation_states:
            valence = float(el.common_oxidation_states[0])
        return [
            float(el.Z or 0),
            float(el.atomic_mass or 0),
            float(el.group or 0),
            float(el.row or 0),
            float(el.X) if el.X is not None else 0.0,
            float(el.atomic_radius) if el.atomic_radius is not None else 0.0,
            valence,
        ]
    except Exception as exc:
        print(f"  warn: no pymatgen features for symbol {symbol!r}: {exc}")
        return [0.0] * 7


def iter_turtle_blocks(path: Path):
    """Yield (approx_subject, turtle_block) for each top-level Turtle statement."""
    prefix_lines: list[str] = []
    buf: list[str] = []
    in_quotes = False
    quote_delim = None
    started = False

    def flush():
        nonlocal buf, started
        if not buf:
            return None
        block = "".join(buf)
        buf = []
        started = False
        # subject = first token
        m = re.match(r"\s*(\S+)", block)
        subj = m.group(1) if m else ""
        return subj, block

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not started and (line.startswith("@prefix") or line.startswith("@base")):
                prefix_lines.append(line)
                continue
            if not started and not line.strip():
                continue

            # track triple quotes / quotes naively
            i = 0
            while i < len(line):
                if not in_quotes:
                    if line.startswith('"""', i):
                        in_quotes = True
                        quote_delim = '"""'
                        i += 3
                        continue
                    if line[i] == '"':
                        in_quotes = True
                        quote_delim = '"'
                        i += 1
                        continue
                    i += 1
                else:
                    if quote_delim == '"""' and line.startswith('"""', i):
                        in_quotes = False
                        quote_delim = None
                        i += 3
                        continue
                    if quote_delim == '"' and line[i] == '"' and (i == 0 or line[i - 1] != "\\"):
                        in_quotes = False
                        quote_delim = None
                    i += 1

            if not started:
                started = True
                buf = [line]
            else:
                buf.append(line)

            if not in_quotes and line.rstrip().endswith("."):
                item = flush()
                if item:
                    yield prefix_lines, item[0], item[1]


def _parse_prefixes(prefix_lines: list[str]) -> dict[str, str]:
    prefs = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "battgpt": BATTGPT,
        "chsub": CHSUB,
        "cryst": CRYST,
        "emmo": "https://w3id.org/emmo#",
        "unit": "http://qudt.org/vocab/unit/",
        "prov": "http://www.w3.org/ns/prov#",
        "dcterms": "http://purl.org/dc/terms/",
        "battery": "https://w3id.org/emmo/domain/battery#",
        "qudt": "http://qudt.org/schema/qudt/",
    }
    for line in prefix_lines:
        m = re.match(r"@prefix\s+(\w*):\s*<([^>]+)>", line)
        if m:
            prefs[m.group(1)] = m.group(2)
    return prefs


def _expand_term(tok: str, prefs: dict[str, str]) -> str | None:
    tok = tok.strip().rstrip(",;.")
    if not tok or tok in {"a", ",", ";", "."}:
        return None
    if tok.startswith("<") and tok.endswith(">"):
        return tok[1:-1]
    if tok.startswith('"') or tok.startswith("'"):
        return None
    if ":" in tok and not tok.startswith("http"):
        pref, local = tok.split(":", 1)
        base = prefs.get(pref)
        if base is None:
            return None
        return base + local
    if tok in {"true", "false"} or re.fullmatch(r"-?\d+(\.\d+)?", tok):
        return None
    return None


def _literal_value(raw: str) -> float | str | None:
    raw = raw.strip()
    # "value"^^xsd:type or "value"@en
    m = re.match(r'^"(.*)"(?:\^\^([\w:]+)|@[\w-]+)?$', raw, re.DOTALL)
    if not m:
        if raw in {"true", "false"}:
            return 1.0 if raw == "true" else 0.0
        if re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", raw):
            return float(raw)
        return None
    body = m.group(1)
    typ = m.group(2) or ""
    if body in {"true", "false"} or typ.endswith("boolean"):
        return 1.0 if body == "true" else 0.0
    try:
        return float(body)
    except ValueError:
        return body


def fast_parse_block(prefix_lines: list[str], block: str) -> tuple[str, list[str], list[tuple[str, str]], dict[str, float | str]] | None:
    """
    Fast Turtle subject-block extractor (no rdflib).
    Returns (subject_uri, type_uris, obj_edges[(pred_local_or_uri, obj_uri)], literals).
    """
    prefs = _parse_prefixes(prefix_lines)
    # Drop huge CIF / long strings for speed
    cleaned = re.sub(r'"""[\s\S]*?"""', '""', block)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    m = re.match(r"^(\S+)\s+(.*)$", cleaned)
    if not m:
        return None
    subj = _expand_term(m.group(1), prefs)
    if not subj:
        # bare battgpt:Name already expanded; if still prefixed fail
        tok = m.group(1)
        if tok.startswith("battgpt:"):
            subj = BATTGPT + tok.split(":", 1)[1]
        else:
            return None
    rest = m.group(2)

    type_uris: list[str] = []
    obj_e: list[tuple[str, str]] = []
    lits: dict[str, float | str] = {}

    # Split on ' ; ' predicate boundaries at top level (quotes already simplified)
    # First handle `a Type, Type2 ;`
    if rest.startswith("a "):
        # find end of type list before first " ; pred" or " ."
        am = re.match(r"^a\s+(.+?)(?:\s+;\s+|\s+\.\s*$)", rest)
        if am:
            type_part = am.group(1)
            rest = rest[am.end() - (1 if rest[am.end() - 1:am.end()] == ";" else 0):]
            # if matched with ` ; `, strip leading ;
            rest = re.sub(r"^\s*;\s*", "", rest)
            rest = re.sub(r"^\s*\.\s*$", "", rest)
            for t in re.split(r"\s*,\s*", type_part):
                tu = _expand_term(t, prefs)
                if tu:
                    type_uris.append(tu)

    # Remaining: pred obj, obj ; pred obj .
    rest = rest.rstrip().rstrip(".")
    if not rest.strip():
        return subj, type_uris, obj_e, lits

    chunks = re.split(r"\s+;\s+", rest)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        pm = re.match(r"^(\S+)\s+(.+)$", chunk)
        if not pm:
            continue
        pred_tok, objs = pm.group(1), pm.group(2).strip()
        pred_uri = _expand_term(pred_tok, prefs)
        if pred_uri is None and ":" in pred_tok:
            continue
        p_loc = _local(pred_uri) if pred_uri else pred_tok
        if pred_uri == EMMO_HAS_PROPERTY:
            p_loc = "hasProperty"
        if pred_uri == EMMO_HAS_VALUE:
            p_loc = "_value_pred"

        # object list separated by commas — but URIs/literals
        # Match URI <...> or pref:local or literals
        for om in re.finditer(
            r'<[^>]+>|[\w]+:[\w./\-]+|"[^"]*"(?:\^\^[\w:]+|@[\w-]+)?|true|false|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',
            objs,
        ):
            ot = om.group(0)
            if ot.startswith("<") or (":" in ot and not ot.startswith('"') and ot not in {"true", "false"}):
                ou = _expand_term(ot, prefs)
                if ou:
                    key = "hasProperty" if (pred_uri == EMMO_HAS_PROPERTY or p_loc == "hasProperty") else p_loc
                    if key != "_value_pred":
                        obj_e.append((key, ou))
            else:
                lv = _literal_value(ot)
                if lv is None:
                    continue
                if pred_uri == EMMO_HAS_VALUE or p_loc == "_value_pred":
                    lits["_value"] = lv
                else:
                    lits[p_loc] = lv

    return subj, type_uris, obj_e, lits


def material_id_from_uri(uri: str) -> str | None:
    m = re.search(r"/material/([^/#]+)", uri)
    if m:
        return m.group(1)
    m = re.search(r"/(?:crystal|unitcell|site|property|bond)/([^/#]+)", uri)
    if m:
        return m.group(1)
    return None


def build_hetero_from_rdf(
    input_path: Path,
    max_materials: int | None = None,
) -> tuple[HeteroData, dict]:
    t0 = time.time()
    print(f"Streaming parse: {input_path}")

    # Collected entity data
    nodes: dict[str, dict[str, dict]] = defaultdict(dict)  # type -> uri -> attrs
    # attrs may include literals dict and links
    literals: dict[str, dict[str, float | str | bool]] = defaultdict(dict)
    links: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # key rel local name -> list of (s,o)
    obj_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    bonds: list[dict] = []  # reified bonds

    selected_mids: set[str] | None = set() if max_materials else None
    n_materials_seen = 0
    n_blocks = 0

    for prefix_lines, _subj_tok, block in iter_turtle_blocks(input_path):
        n_blocks += 1
        if n_blocks % 200000 == 0:
            print(f"  … {n_blocks:,} turtle blocks ({time.time() - t0:.1f}s)", flush=True)

        parsed = fast_parse_block(prefix_lines, block)
        if not parsed:
            continue
        s_uri, type_uris, obj_e, lits = parsed
        ntype = _infer_type(s_uri, type_uris)
        if ntype is None:
            loc = _local(s_uri)
            if loc in ROLE_INDIVIDUALS:
                ntype = "BatteryRole"
            else:
                continue

        mid = material_id_from_uri(s_uri)

        if ntype == "Material":
            if selected_mids is not None:
                if mid and mid not in selected_mids:
                    if len(selected_mids) >= max_materials:
                        continue
                    selected_mids.add(mid)
                    n_materials_seen += 1
                elif mid:
                    selected_mids.add(mid)
                    n_materials_seen += 1
        elif selected_mids is not None and mid is not None and mid not in selected_mids:
            if ntype not in ("Element", "AtomicSpecies", "SpaceGroup", "CrystalSystem", "BatteryRole"):
                continue

        nodes[ntype][s_uri] = {"uri": s_uri, "types": type_uris}
        for k, v in lits.items():
            literals[s_uri][k] = v
        for p_loc, o_uri in obj_e:
            if p_loc == "_value_pred":
                continue
            # When subsetting, do not ingest role→all-materials fan-out
            if selected_mids is not None and p_loc == "usesMaterial":
                omid = material_id_from_uri(o_uri)
                if omid is None or omid not in selected_mids:
                    continue
            if selected_mids is not None and p_loc == "belongsToElectrode":
                smid = material_id_from_uri(s_uri)
                if smid is None or smid not in selected_mids:
                    continue
            obj_edges[p_loc].append((s_uri, o_uri))
        if ntype == "CrystalBond":
            bonds.append({"uri": s_uri})

    if selected_mids is not None:
        print(f"  Selected materials: {len(selected_mids)}")
        # Drop any Material nodes that slipped in outside the subset
        nodes["Material"] = {
            u: meta for u, meta in nodes["Material"].items()
            if material_id_from_uri(u) in selected_mids
        }

    # Ensure shared vocab nodes exist from edges even if typed late
    def ensure(ntype: str, uri: str):
        if selected_mids is not None and ntype == "Material":
            mid = material_id_from_uri(uri)
            if mid is None or mid not in selected_mids:
                return
        if uri not in nodes[ntype]:
            nodes[ntype][uri] = {"uri": uri, "types": []}

    for s, o in obj_edges.get("hasElement", []):
        ensure("AtomicSpecies", s)
        ensure("Element", o)
    for s, o in obj_edges.get("hasStructure", []):
        ensure("Material", s)
        ensure("Crystal", o)
    for s, o in obj_edges.get("hasUnitCell", []):
        ensure("Crystal", s)
        ensure("UnitCell", o)
    for s, o in obj_edges.get("containsSite", []):
        ensure("UnitCell", s)
        ensure("AtomicSite", o)
    for s, o in obj_edges.get("containsSpecies", []):
        ensure("AtomicSite", s)
        ensure("AtomicSpecies", o)
    for s, o in obj_edges.get("hasSpaceGroup", []):
        ensure("Crystal", s)
        ensure("SpaceGroup", o)
    for s, o in obj_edges.get("hasCrystalSystem", []):
        ensure("Crystal", s)
        ensure("CrystalSystem", o)
    for s, o in obj_edges.get("belongsToElectrode", []):
        ensure("Material", s)
        ensure("BatteryRole", o)
    for s, o in obj_edges.get("usesMaterial", []):
        ensure("BatteryRole", s)
        ensure("Material", o)
    for s, o in obj_edges.get("hasProperty", []):
        # only material→property (skip site coordination props as material feats later)
        if "/material/" in s and "/property/" in o and "/site_" not in o:
            ensure("Material", s)
            ensure("Property", o)
    for s, o in obj_edges.get("hasBondTo", []):
        ensure("AtomicSite", s)
        ensure("AtomicSite", o)
    for s, o in obj_edges.get("hasSourceSite", []):
        ensure("CrystalBond", s)
        ensure("AtomicSite", o)
    for s, o in obj_edges.get("hasTargetSite", []):
        ensure("CrystalBond", s)
        ensure("AtomicSite", o)

    # Predefine 4 roles if missing
    for ind, role_name in ROLE_INDIVIDUALS.items():
        uri = BATTGPT + ind
        ensure("BatteryRole", uri)
        literals[uri]["role_name"] = role_name

    # Index assignment (stable sort by URI)
    id_maps: dict[str, dict[str, int]] = {}
    for ntype, udict in nodes.items():
        if ntype == "CrystalBond":
            continue
        uris = sorted(udict.keys())
        id_maps[ntype] = {u: i for i, u in enumerate(uris)}

    # Feature builders
    def prop_value(prop_uri: str) -> float:
        lit = literals.get(prop_uri, {})
        if "_value" in lit:
            return float(lit["_value"])
        return 0.0

    def material_feature(uri: str) -> list[float]:
        lit = literals.get(uri, {})
        # resolve linked properties by name in URI
        prop_by_name: dict[str, float] = {}
        for s, o in obj_edges.get("hasProperty", []):
            if s != uri:
                continue
            name = o.rstrip("/").rsplit("/", 1)[-1]
            prop_by_name[name] = prop_value(o)
        # also named object props
        for key in ("hasBandGap", "hasFormationEnergy", "hasEnergyAboveHull",
                    "hasBulkModulus", "hasShearModulus"):
            for s, o in obj_edges.get(key, []):
                if s == uri:
                    name = o.rstrip("/").rsplit("/", 1)[-1]
                    prop_by_name[name] = prop_value(o)

        def g(name: str) -> float:
            return float(prop_by_name.get(name, 0.0))

        return [
            g("band_gap"),
            g("formation_energy_per_atom"),
            g("energy_above_hull"),
            g("density"),
            g("volume"),
            g("e_fermi"),
            g("bulk_modulus"),
            g("shear_modulus"),
            g("total_magnetization"),
            float(lit.get("isMetal", 0.0) or 0.0),
            float(lit.get("isStable", 0.0) or 0.0),
        ]

    # Count sites per unitcell / bonds per site
    site_count: dict[str, int] = defaultdict(int)
    for s, o in obj_edges.get("containsSite", []):
        site_count[s] += 1
    bond_degree: dict[str, int] = defaultdict(int)
    for s, o in obj_edges.get("hasBondTo", []):
        bond_degree[s] += 1

    # Build bond list from CrystalBond reification (preferred) else hasBondTo
    src_of = {s: o for s, o in obj_edges.get("hasSourceSite", [])}
    dst_of = {s: o for s, o in obj_edges.get("hasTargetSite", [])}
    bond_edges: list[tuple[str, str, str, float]] = []  # src,dst,bond_uri,dist
    if src_of and dst_of:
        for b_uri in sorted(set(src_of) & set(dst_of)):
            if selected_mids is not None:
                mid = material_id_from_uri(b_uri)
                if mid is not None and mid not in selected_mids:
                    continue
            dist = float(literals.get(b_uri, {}).get("hasBondDistance", 0.0) or 0.0)
            bond_edges.append((src_of[b_uri], dst_of[b_uri], b_uri, dist))
    else:
        # fall back: hasBondTo without distance metadata per pair
        for s, o in obj_edges.get("hasBondTo", []):
            bond_edges.append((s, o, "", 0.0))

    data = HeteroData()
    node_mapping: dict[str, list] = {}
    edge_mapping: dict[str, list] = {}

    def fill_nodes(ntype: str, feat_fn, info_fn, dim: int):
        amap = id_maps.get(ntype, {})
        feats, infos = [], []
        for u, idx in sorted(amap.items(), key=lambda kv: kv[1]):
            feats.append(feat_fn(u))
            infos.append(info_fn(u, idx))
        data[ntype].x = torch.tensor(feats, dtype=torch.float) if feats else torch.empty((0, dim))
        node_mapping[ntype] = infos

    # --- Material ---
    def material_info(u, idx):
        lit = literals.get(u, {})
        formula = ""
        if "hasFormula" in lit and not isinstance(lit["hasFormula"], float):
            formula = str(lit["hasFormula"])
        mid = lit.get("hasMaterialProjectId")
        if isinstance(mid, float) or mid is None:
            mid = material_id_from_uri(u)
        else:
            mid = str(mid)
        role = "Generic"
        for s, o in obj_edges.get("belongsToElectrode", []):
            if s == u:
                role = ROLE_INDIVIDUALS.get(_local(o), _local(o))
                break
        return {
            "tensor_index": idx,
            "uri": u,
            "material_id": mid,
            "formula": formula,
            "role": role,
            "is_stable": bool(lit.get("isStable", 0.0)),
        }

    fill_nodes("Material", material_feature, material_info, 11)

    # Crystal
    cryst_to_uc = {s: o for s, o in obj_edges.get("hasUnitCell", [])}

    def crystal_feat(u):
        uc = cryst_to_uc.get(u)
        n = float(site_count.get(uc, 0)) if uc else 0.0
        return [n]

    fill_nodes(
        "Crystal",
        crystal_feat,
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "material_id": material_id_from_uri(u),
            "formula": "",
        },
        1,
    )

    fill_nodes(
        "UnitCell",
        lambda u: [
            float(literals.get(u, {}).get("hasLatticea", 0.0) or 0.0),
            float(literals.get(u, {}).get("hasLatticeb", 0.0) or 0.0),
            float(literals.get(u, {}).get("hasLatticec", 0.0) or 0.0),
            float(literals.get(u, {}).get("hasAlpha", 0.0) or 0.0),
            float(literals.get(u, {}).get("hasBeta", 0.0) or 0.0),
            float(literals.get(u, {}).get("hasGamma", 0.0) or 0.0),
        ],
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "material_id": material_id_from_uri(u),
        },
        6,
    )

    fill_nodes(
        "AtomicSite",
        lambda u: [
            float(literals.get(u, {}).get("hasFractionalX", 0.0) or 0.0),
            float(literals.get(u, {}).get("hasFractionalY", 0.0) or 0.0),
            float(literals.get(u, {}).get("hasFractionalZ", 0.0) or 0.0),
            float(bond_degree.get(u, 0)),
        ],
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "material_id": material_id_from_uri(u),
            "site_index": int(u.rstrip("/").rsplit("/", 1)[-1]) if u.rstrip("/").rsplit("/", 1)[-1].isdigit() else -1,
            "element": "",
        },
        4,
    )

    def species_feat(u):
        # URI .../species/Li_0
        tail = u.rstrip("/").rsplit("/", 1)[-1]
        sym = element_symbol_from_uri(u)
        ox_s = tail.partition("_")[2]
        try:
            ox = float(ox_s) if ox_s else 0.0
        except ValueError:
            ox = 0.0
        ef = _element_feats(sym)
        return [ef[0], ef[1], ef[4], ef[6], ox]

    fill_nodes(
        "AtomicSpecies",
        species_feat,
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "species_symbol": u.rstrip("/").rsplit("/", 1)[-1].split("_")[0],
        },
        5,
    )

    fill_nodes(
        "Element",
        lambda u: _element_feats(element_symbol_from_uri(u)),
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "symbol": element_symbol_from_uri(u),
        },
        7,
    )

    fill_nodes(
        "SpaceGroup",
        lambda u: [float(literals.get(u, {}).get("hasSpaceGroupNumber",
                   float(u.rstrip("/").rsplit("/", 1)[-1]) if u.rstrip("/").rsplit("/", 1)[-1].isdigit() else 0.0) or 0.0)],
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "spacegroup_number": int(float(literals.get(u, {}).get("hasSpaceGroupNumber", 0) or 0)),
            "symbol": str(literals.get(u, {}).get("hasSymmetrySymbol", "")),
        },
        1,
    )

    # CrystalSystem — feature = local code index
    cs_uris = sorted(id_maps.get("CrystalSystem", {}))
    for i, u in enumerate(cs_uris):
        id_maps["CrystalSystem"][u] = i  # already from sorted

    fill_nodes(
        "CrystalSystem",
        lambda u: [float(id_maps["CrystalSystem"][u])],
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "crystal_system": u.rstrip("/").rsplit("/", 1)[-1],
        },
        1,
    )

    # Property — material quantitative only
    prop_uris = [u for u in id_maps.get("Property", {}) if "/site_" not in u]
    # rebuild property id map compact
    id_maps["Property"] = {u: i for i, u in enumerate(sorted(prop_uris))}
    fill_nodes(
        "Property",
        lambda u: [prop_value(u)],
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "material_id": material_id_from_uri(u),
            "property_name": u.rstrip("/").rsplit("/", 1)[-1],
        },
        1,
    )

    fill_nodes(
        "BatteryRole",
        lambda u: [float(id_maps["BatteryRole"][u])],
        lambda u, idx: {
            "tensor_index": idx,
            "uri": u,
            "role_name": ROLE_INDIVIDUALS.get(_local(u), literals.get(u, {}).get("role_name", _local(u))),
        },
        1,
    )

    # Enrich AtomicSite element from containsSpecies
    sp_of_site = {s: o for s, o in obj_edges.get("containsSpecies", [])}
    for info in node_mapping["AtomicSite"]:
        sp = sp_of_site.get(info["uri"])
        if sp:
            info["element"] = sp.rstrip("/").rsplit("/", 1)[-1].split("_")[0]

    # Enrich Crystal formulas from materials
    mid_to_formula = {m["material_id"]: m["formula"] for m in node_mapping["Material"]}
    for info in node_mapping["Crystal"]:
        info["formula"] = mid_to_formula.get(info.get("material_id"), "")

    def add_edge(src_t, rel, dst_t, pairs, extra_fn=None):
        src_list, dst_list = [], []
        emap = []
        smap = id_maps.get(src_t, {})
        dmap = id_maps.get(dst_t, {})
        rel_iri = RELATION_URIS.get(f"{src_t} -> {rel} -> {dst_t}", f"{BATTGPT}{rel}")
        for i, (s, o, *rest) in enumerate(pairs):
            if s not in smap or o not in dmap:
                continue
            si, di = smap[s], dmap[o]
            src_list.append(si)
            dst_list.append(di)
            entry = {
                "edge_tensor_index": len(emap),
                "src_node_index": si,
                "dst_node_index": di,
                "src_uri": s,
                "dst_uri": o,
                "relation_uri": rel_iri,
            }
            if extra_fn:
                entry.update(extra_fn(s, o, rest))
            emap.append(entry)
        key = (src_t, rel, dst_t)
        if src_list:
            data[key].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            data[key].edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_mapping[f"{src_t}_{rel}_{dst_t}"] = emap
        return len(src_list)

    add_edge("Material", "hasStructure", "Crystal",
             [(s, o) for s, o in obj_edges.get("hasStructure", [])])
    add_edge("Crystal", "hasUnitCell", "UnitCell",
             [(s, o) for s, o in obj_edges.get("hasUnitCell", [])])
    add_edge("UnitCell", "containsSite", "AtomicSite",
             [(s, o) for s, o in obj_edges.get("containsSite", [])])
    add_edge("AtomicSite", "containsSpecies", "AtomicSpecies",
             [(s, o) for s, o in obj_edges.get("containsSpecies", [])])
    add_edge("AtomicSpecies", "hasElement", "Element",
             [(s, o) for s, o in obj_edges.get("hasElement", [])])
    add_edge("Crystal", "hasSpaceGroup", "SpaceGroup",
             [(s, o) for s, o in obj_edges.get("hasSpaceGroup", [])])
    add_edge("Crystal", "hasCrystalSystem", "CrystalSystem",
             [(s, o) for s, o in obj_edges.get("hasCrystalSystem", [])])
    add_edge("Material", "belongsToElectrode", "BatteryRole",
             [(s, o) for s, o in obj_edges.get("belongsToElectrode", [])])
    add_edge("BatteryRole", "usesMaterial", "Material",
             [(s, o) for s, o in obj_edges.get("usesMaterial", [])])

    # Material → Property (exclude site coordination)
    mat_prop_pairs = [
        (s, o) for s, o in obj_edges.get("hasProperty", [])
        if s in id_maps.get("Material", {}) and o in id_maps.get("Property", {}) and "/site_" not in o
    ]
    add_edge("Material", "hasProperty", "Property", mat_prop_pairs)

    add_edge(
        "AtomicSite", "hasBondTo", "AtomicSite",
        [(a, b, uri, dist) for a, b, uri, dist in bond_edges],
        extra_fn=lambda s, o, rest: {
            "bond_uri": rest[0] if rest else "",
            "bond_distance": float(rest[1]) if len(rest) > 1 else 0.0,
            "coordination_method": "CrystalNN",
        },
    )
    bond_attr = []
    smap = id_maps.get("AtomicSite", {})
    for a, b, uri, dist in bond_edges:
        if a in smap and b in smap:
            bond_attr.append([float(dist)])
    if bond_attr:
        data["AtomicSite", "hasBondTo", "AtomicSite"].edge_attr = torch.tensor(bond_attr, dtype=torch.float)
    else:
        data["AtomicSite", "hasBondTo", "AtomicSite"].edge_attr = torch.empty((0, 1), dtype=torch.float)

    feature_description = {
        "Material": {
            "shape": [len(id_maps.get("Material", {})), 11],
            "columns": [
                "band_gap_eV", "formation_energy_eV_atom", "energy_above_hull_eV_atom",
                "density_g_cm3", "volume_ang3", "e_fermi_eV", "bulk_modulus_GPa",
                "shear_modulus_GPa", "total_magnetization_muB", "is_metal_flag", "is_stable_flag",
            ],
        },
        "Crystal": {"shape": [len(id_maps.get("Crystal", {})), 1], "columns": ["num_sites"]},
        "UnitCell": {
            "shape": [len(id_maps.get("UnitCell", {})), 6],
            "columns": ["lattice_a", "lattice_b", "lattice_c", "alpha", "beta", "gamma"],
        },
        "AtomicSite": {
            "shape": [len(id_maps.get("AtomicSite", {})), 4],
            "columns": ["fractional_x", "fractional_y", "fractional_z", "coordination_number"],
        },
        "AtomicSpecies": {
            "shape": [len(id_maps.get("AtomicSpecies", {})), 5],
            "columns": ["atomic_number", "atomic_mass", "electronegativity", "valence_electrons", "oxidation_state"],
        },
        "Element": {
            "shape": [len(id_maps.get("Element", {})), 7],
            "columns": [
                "atomic_number", "atomic_mass", "group", "period",
                "electronegativity", "covalent_radius", "valence_electrons",
            ],
        },
        "SpaceGroup": {"shape": [len(id_maps.get("SpaceGroup", {})), 1], "columns": ["spacegroup_number"]},
        "CrystalSystem": {"shape": [len(id_maps.get("CrystalSystem", {})), 1], "columns": ["crystal_system_code"]},
        "Property": {"shape": [len(id_maps.get("Property", {})), 1], "columns": ["scalar_value"]},
        "BatteryRole": {"shape": [len(id_maps.get("BatteryRole", {})), 1], "columns": ["role_code"]},
        "EdgeAttributes": {
            "AtomicSite -> hasBondTo -> AtomicSite": {
                "shape": [len(bond_attr), 1],
                "columns": ["distance_angstrom"],
            }
        },
    }

    meta = {
        "source": str(input_path.resolve()),
        "num_materials": len(id_maps.get("Material", {})),
        "num_crystals": len(id_maps.get("Crystal", {})),
        "num_unit_cells": len(id_maps.get("UnitCell", {})),
        "num_atomic_sites": len(id_maps.get("AtomicSite", {})),
        "num_atomic_species": len(id_maps.get("AtomicSpecies", {})),
        "num_elements": len(id_maps.get("Element", {})),
        "num_space_groups": len(id_maps.get("SpaceGroup", {})),
        "num_crystal_systems": len(id_maps.get("CrystalSystem", {})),
        "num_properties": len(id_maps.get("Property", {})),
        "num_battery_roles": len(id_maps.get("BatteryRole", {})),
        "num_bonds": len(bond_attr),
        "node_types": list(data.node_types),
        "edge_types": [f"{s}_{r}_{d}" for s, r, d in data.edge_types],
        "elapsed_sec": round(time.time() - t0, 2),
        "turtle_blocks_seen": n_blocks,
    }

    bundle = {
        "node_mapping": node_mapping,
        "edge_mapping": edge_mapping,
        "relation_mapping": RELATION_URIS,
        "feature_description": feature_description,
        "ontology_mapping": ONTOLOGY_MAPPING,
        "metadata": meta,
    }
    print(f"Done graph build in {meta['elapsed_sec']}s — materials={meta['num_materials']} "
          f"sites={meta['num_atomic_sites']} bonds={meta['num_bonds']}")
    return data, bundle


def save_bundle(data: HeteroData, bundle: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "hetero_data.pt")
    (out_dir / "metadata.json").write_text(json.dumps(bundle["metadata"], indent=2), encoding="utf-8")
    (out_dir / "node_mapping.json").write_text(json.dumps(bundle["node_mapping"], indent=2), encoding="utf-8")
    (out_dir / "edge_mapping.json").write_text(json.dumps(bundle["edge_mapping"], indent=2), encoding="utf-8")
    (out_dir / "relation_mapping.json").write_text(json.dumps(bundle["relation_mapping"], indent=2), encoding="utf-8")
    (out_dir / "feature_description.json").write_text(
        json.dumps(bundle["feature_description"], indent=2), encoding="utf-8"
    )
    (out_dir / "ontology_mapping.json").write_text(
        json.dumps(bundle["ontology_mapping"], indent=2), encoding="utf-8"
    )
    print(f"Wrote bundle → {out_dir}")


def main():
    p = argparse.ArgumentParser(description="RDF/TTL → PyG HeteroData (BattGPT KG)")
    p.add_argument("-i", "--input", type=Path, required=True, help="battery_kg.ttl or .rdf")
    p.add_argument("--out-dir", type=Path, default=Path("hetero_from_rdf"))
    p.add_argument("--max-materials", type=int, default=None,
                   help="Only keep the first N materials (and their structure subgraph).")
    args = p.parse_args()

    path = args.input.expanduser().resolve()
    if not path.exists():
        print(f"Missing input: {path}", file=sys.stderr)
        sys.exit(1)

    # .rdf XML path: fall back to full rdflib parse (no streaming)
    if path.suffix.lower() in {".rdf", ".xml", ".owl"}:
        print("RDF/XML detected — loading full graph with rdflib (may use a lot of RAM)…")
        g = Graph()
        g.parse(path, format="xml")
        # Serialize to temp turtle is heavy; instead write a note
        print("Tip: prefer battery_kg.ttl for streaming conversion.", file=sys.stderr)
        # Dump to a temp ttl for the streaming path? Too heavy.
        # Convert in-memory from Graph:
        data, bundle = build_hetero_from_graph(g, max_materials=args.max_materials)
    else:
        data, bundle = build_hetero_from_rdf(path, max_materials=args.max_materials)

    save_bundle(data, bundle, args.out_dir.resolve())


def build_hetero_from_graph(g: Graph, max_materials: int | None = None):
    """Fallback path for RDF/XML: materialize a temp turtle is skipped; use same logic via export."""
    # Write a lightweight intermediate by iterating triples into the streaming structures
    # Reuse build by dumping to a NamedTemporaryFile turtle — can be huge.
    # For practicality, convert triples into the same collectors inline.
    raise SystemExit(
        "RDF/XML full-graph conversion is not implemented via streaming. "
        "Pass battery_kg.ttl instead."
    )


if __name__ == "__main__":
    main()
