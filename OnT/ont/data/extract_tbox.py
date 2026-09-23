"""Extract the real TBox from battery_kg.ttl into a tiny standalone OWL file.

battery_kg.ttl declares its owl:Class / owl:ObjectProperty / owl:DatatypeProperty
entities scattered across the whole 9.6M-line file (not confined to the head),
alongside ~8M ABox triples. This streams the file once with the same
block-reader extract_abox.py uses, collects only the TBox blocks (by their
declared rdf:type), and writes them verbatim into a small standalone OWL file.

battery_kg.ttl also types every ABox individual under classes that are
*never* declared locally (chsub:Substance, cryst:Crystal, ...) -- they belong
to EMMO, which this project deliberately does not import (see
ONT_ABOX_EXTENSION.md S2.1: DeepOnto/OWL API cannot load a 9.6M-line file,
and importing full EMMO defeats the point of a tiny schema). So those classes
are stub-declared here with just a label, giving DeepOnto's
OntologyVerbaliser a name for every class an ABox individual can be typed
under, without pulling in EMMO itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

ONT_ROOT = Path(__file__).resolve().parents[2]
PROJ_ROOT = ONT_ROOT.parent
DEFAULT_TTL = PROJ_ROOT / "output" / "battery_kg.ttl"
DEFAULT_OUT = ONT_ROOT / "data" / "battgpt_tbox.owl"

OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_OBJ_PROP = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_DATA_PROP = "http://www.w3.org/2002/07/owl#DatatypeProperty"

PREFIXES = """@prefix battgpt: <https://w3id.org/battgpt/kg#> .
@prefix chsub: <https://w3id.org/emmo/domain/chemical-substance#> .
@prefix cryst: <https://w3id.org/emmo/domain/crystallography#> .
@prefix emmo: <https://w3id.org/emmo#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

ONTOLOGY_HEADER = "<https://w3id.org/battgpt/kg> a owl:Ontology .\n"

# Classes ABox individuals are typed with (chsub:/cryst:/emmo:) that are
# never locally declared as owl:Class anywhere in battery_kg.ttl -- they are
# EMMO classes, referenced only via rdf:type / rdfs:domain / rdfs:range.
STUB_CLASSES = {
    "chsub:Substance": "substance",
    "cryst:Crystal": "crystal",
    "cryst:UnitCell": "unit cell",
    "cryst:AtomicSite": "atomic site",
    "chsub:AtomicSpecies": "atomic species",
    "chsub:Element": "element",
    "cryst:SpaceGroup": "space group",
    "cryst:CrystalSystem": "crystal system",
    "emmo:EMMO_b7bcff25_ffc3_474e_9ab5_01b1664bd4ba": "property",
}

# battgpt:BatteryRole rdfs:subClassOf <external EMMO class we don't import>.
# Drop just that clause so BatteryRole becomes a (real, un-fabricated) root
# instead of dangling on an undeclared entity.
DANGLING_SUBCLASS = re.compile(
    r";\s*\n\s*rdfs:subClassOf\s+emmo:EMMO_4207e895_8bfe_4b08_8e2d_dd53af54e10b\s*\."
)


def _load_ttl_helpers():
    path = PROJ_ROOT / "HGT-Pipeline" / "battery_hgt" / "rdf_to_hetero.py"
    spec = importlib.util.spec_from_file_location("battgpt_rdf_to_hetero", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract(ttl_path: Path) -> dict:
    h = _load_ttl_helpers()
    class_blocks: dict[str, str] = {}
    obj_prop_blocks: dict[str, str] = {}
    data_prop_blocks: dict[str, str] = {}

    n_blocks = 0
    for prefix_lines, subj_tok, block in h.iter_turtle_blocks(ttl_path):
        n_blocks += 1
        # TBox entities use CURIE subjects (battgpt:Foo, emmo:EMMO_...);
        # every ABox individual uses a full <https://...> URI as subject.
        # Cheap pre-filter before the (slightly) more expensive block parse.
        if not (subj_tok.startswith("battgpt:") or subj_tok.startswith("emmo:")):
            continue
        parsed = h.fast_parse_block(prefix_lines, block)
        if not parsed:
            continue
        subj, type_uris, _obj_e, _lits = parsed
        if OWL_CLASS in type_uris:
            class_blocks[subj_tok] = block
        elif OWL_OBJ_PROP in type_uris:
            obj_prop_blocks[subj_tok] = block
        elif OWL_DATA_PROP in type_uris:
            data_prop_blocks[subj_tok] = block

    return {
        "n_blocks": n_blocks,
        "classes": class_blocks,
        "object_properties": obj_prop_blocks,
        "datatype_properties": data_prop_blocks,
    }


def build_owl_text(bundle: dict) -> str:
    parts = [PREFIXES, "", ONTOLOGY_HEADER, ""]

    parts.append("# --- Locally declared classes (real battgpt: hierarchy) ---")
    for name in sorted(bundle["classes"]):
        block = bundle["classes"][name].strip()
        block = DANGLING_SUBCLASS.sub(".", block)
        parts.append(block)
        parts.append("")

    parts.append("# --- Stub classes for EMMO types ABox individuals are typed")
    parts.append("# under but that battery_kg.ttl never declares locally ---")
    for curie, label in STUB_CLASSES.items():
        parts.append(f'{curie} a owl:Class ;\n    rdfs:label "{label}"@en .')
        parts.append("")

    parts.append("# --- Object properties ---")
    for name in sorted(bundle["object_properties"]):
        parts.append(bundle["object_properties"][name].strip())
        parts.append("")

    parts.append("# --- Datatype properties ---")
    for name in sorted(bundle["datatype_properties"]):
        parts.append(bundle["datatype_properties"][name].strip())
        parts.append("")

    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ttl", type=Path, default=DEFAULT_TTL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    print(f"Streaming {args.ttl} for TBox declarations…", flush=True)
    bundle = extract(args.ttl)
    text = build_owl_text(bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    n_classes = len(bundle["classes"]) + len(STUB_CLASSES)
    print(
        f"Wrote {args.out} ({bundle['n_blocks']:,} blocks scanned): "
        f"{len(bundle['classes'])} local classes + {len(STUB_CLASSES)} stub classes "
        f"= {n_classes} classes, {len(bundle['object_properties'])} object properties, "
        f"{len(bundle['datatype_properties'])} datatype properties"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
