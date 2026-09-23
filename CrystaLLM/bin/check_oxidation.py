"""
Compare input (target) oxidation states with pymatgen-computed oxidation states
for generated CIF structures using Bond Valence Sum (BVS) analysis.
"""
import os
import sys
import glob
from pymatgen.io.cif import CifParser
from pymatgen.analysis.bond_valence import BVAnalyzer

# Target oxidation states from prompt
TARGET_OXI_STATES = {
    "Li": 1,
    "Mn": 2,
    "Ni": 3,
    "O": -2,
}

def check_oxidation_states(cif_path):
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(cif_path)}")
    print(f"{'='*60}")

    try:
        parser = CifParser(cif_path)
        structure = parser.parse_structures()[0]
    except Exception as e:
        print(f"  ERROR parsing CIF: {e}")
        return

    # Use Bond Valence Sum analyzer to estimate oxidation states
    try:
        bva = BVAnalyzer()
        oxi_structure = bva.get_oxi_state_decorated_structure(structure)
    except Exception as e:
        print(f"  ERROR computing oxidation states (BVS): {e}")
        print(f"  This may mean bond lengths are unreasonable for BVS analysis.")
        return

    # Extract computed oxidation states per element
    computed_oxi = {}
    for site in oxi_structure:
        elem = str(site.specie.element)
        oxi = site.specie.oxi_state
        if elem not in computed_oxi:
            computed_oxi[elem] = []
        computed_oxi[elem].append(oxi)

    # Compare
    print(f"\n  {'Element':<10} {'Target Oxi':<15} {'Computed Oxi':<20} {'Match?'}")
    print(f"  {'-'*55}")

    all_match = True
    for elem in sorted(TARGET_OXI_STATES.keys()):
        target = TARGET_OXI_STATES[elem]
        if elem in computed_oxi:
            # Check if all sites of this element have the same oxidation state
            unique_oxi = list(set(computed_oxi[elem]))
            computed_str = ", ".join([f"{o:+.1f}" for o in unique_oxi])
            match = all(abs(o - target) < 0.1 for o in unique_oxi)
        else:
            computed_str = "NOT FOUND"
            match = False

        status = "YES" if match else "NO"
        if not match:
            all_match = False
        print(f"  {elem:<10} {target:+d}{'':<13} {computed_str:<20} {status}")

    print(f"\n  Overall Match: {'ALL MATCH' if all_match else 'MISMATCH DETECTED'}")
    return all_match


if __name__ == "__main__":
    cif_dir = sys.argv[1] if len(sys.argv) > 1 else "processed_cifs"
    cif_files = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))

    if not cif_files:
        print(f"No .cif files found in '{cif_dir}'")
        sys.exit(1)

    print(f"Checking oxidation states for {len(cif_files)} CIF file(s) in '{cif_dir}'")
    print(f"Target oxidation states: {TARGET_OXI_STATES}")

    results = {}
    for cif_path in cif_files:
        result = check_oxidation_states(cif_path)
        results[os.path.basename(cif_path)] = result

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for fname, match in results.items():
        status = "MATCH" if match else ("MISMATCH" if match is not None else "ERROR")
        print(f"  {fname}: {status}")
