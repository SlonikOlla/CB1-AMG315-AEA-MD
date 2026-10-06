from pathlib import Path
import numpy as np

from openff.toolkit import Molecule
from openff.units import unit

INPUT = Path(
    "structures/reference_endocannabinoid_CB1/"
    "AMG315_ZI5_aligned_to_frozen_CB1.sdf"
)
OUTPUT = Path(
    "parameters/AMG315/"
    "AMG315_AM1BCC_sage221.sdf"
)

molecule = Molecule.from_file(
    str(INPUT),
    allow_undefined_stereo=False,
)

assert molecule.n_atoms == 68
assert sum(
    atom.atomic_number > 1 for atom in molecule.atoms
) == 27
assert molecule.n_conformers == 1

initial_coordinates = (
    molecule.conformers[0].m_as(unit.angstrom).copy()
)
initial_atomic_numbers = [
    atom.atomic_number for atom in molecule.atoms
]
initial_bonds = sorted(
    (
        bond.atom1_index,
        bond.atom2_index,
        float(bond.bond_order),
    )
    for bond in molecule.bonds
)

print("=" * 78)
print("AMG315 AM1-BCC PARAMETERIZATION")
print("=" * 78)
print("Input:", INPUT)
print("Atoms:", molecule.n_atoms)
print("Heavy atoms:", sum(
    atom.atomic_number > 1 for atom in molecule.atoms
))
print("Conformers:", molecule.n_conformers)
print("Initial formal charge:", molecule.total_charge)

print()
print("Assigning AM1-BCC charges using the experimental conformer...")

molecule.assign_partial_charges(
    partial_charge_method="am1bcc",
    use_conformers=molecule.conformers,
)

charges = molecule.partial_charges.m_as(
    unit.elementary_charge
)

assert len(charges) == molecule.n_atoms
assert np.isfinite(charges).all()

charge_sum = float(np.sum(charges))

print("Partial-charge sum:", f"{charge_sum:.12f} e")
print("Minimum atomic charge:", f"{charges.min():.8f} e")
print("Maximum atomic charge:", f"{charges.max():.8f} e")

assert abs(charge_sum) < 1.0e-6

molecule.name = "AMG315_AM1BCC_sage221"
molecule.to_file(str(OUTPUT), file_format="SDF")

assert OUTPUT.is_file()
assert OUTPUT.stat().st_size > 0

check = Molecule.from_file(
    str(OUTPUT),
    allow_undefined_stereo=False,
)

assert check.n_atoms == molecule.n_atoms
assert check.n_bonds == molecule.n_bonds
assert check.n_conformers == 1
assert [
    atom.atomic_number for atom in check.atoms
] == initial_atomic_numbers

check_bonds = sorted(
    (
        bond.atom1_index,
        bond.atom2_index,
        float(bond.bond_order),
    )
    for bond in check.bonds
)

assert check_bonds == initial_bonds
assert check.partial_charges is not None

saved_charges = check.partial_charges.m_as(
    unit.elementary_charge
)

assert np.isfinite(saved_charges).all()
assert np.max(np.abs(saved_charges - charges)) < 1.0e-6
assert abs(float(np.sum(saved_charges))) < 1.0e-6

saved_coordinates = check.conformers[0].m_as(
    unit.angstrom
)

coordinate_differences = np.linalg.norm(
    saved_coordinates - initial_coordinates,
    axis=1,
)

print()
print("OUTPUT VALIDATION")
print("Saved:", OUTPUT)
print("Saved atoms:", check.n_atoms)
print("Saved heavy atoms:", sum(
    atom.atomic_number > 1 for atom in check.atoms
))
print(
    "Saved partial-charge sum:",
    f"{float(np.sum(saved_charges)):.12f} e",
)
print(
    "Coordinate RMSD:",
    f"{np.sqrt(np.mean(coordinate_differences ** 2)):.8f} A",
)
print(
    "Maximum coordinate difference:",
    f"{coordinate_differences.max():.8f} A",
)

assert coordinate_differences.max() < 1.0e-5

print()
print("AMG315 AM1-BCC PARAMETERIZATION: PASS")
