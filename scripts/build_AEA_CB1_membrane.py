from openmm.app import PDBFile, Modeller, ForceField, PME, HBonds
from openmm import unit, XmlSerializer
from openff.toolkit import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from pathlib import Path

protein_file = Path(
    "structures/CB1_final/CB1_WT_9B54_model10_PPM_clean.pdb"
)
ligand_file = Path(
    "parameters/AEA/AEA_AM1BCC_sage221.sdf"
)

outdir = Path("systems/AEA_CB1")
outdir.mkdir(parents=True, exist_ok=True)

print("Loading PPM-oriented CB1...")
pdb = PDBFile(str(protein_file))

print("Loading parameterized AEA...")
mol = Molecule.from_file(str(ligand_file))

print("AEA atoms:", mol.n_atoms)
print("AEA charge:", mol.total_charge)
print("AEA partial-charge sum:", mol.partial_charges.sum())

print("\nLoading force fields...")
ff = ForceField(
    "amber19/protein.ff19SB.xml",
    "amber19/lipid21.xml",
    "amber19/tip3p.xml",
)

smirnoff = SMIRNOFFTemplateGenerator(
    molecules=mol,
    forcefield="openff-2.2.1"
)
ff.registerTemplateGenerator(smirnoff.generator)

print("Adding protein hydrogens...")
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(ff, pH=7.4)

print("Atoms after protein H:", modeller.topology.getNumAtoms())

print("Adding frozen AEA pose...")
lig_top = mol.to_topology().to_openmm()
lig_pos = mol.conformers[0].to_openmm()
modeller.add(lig_top, lig_pos)

print("Atoms before membrane:", modeller.topology.getNumAtoms())

print("\nBuilding POPC membrane...")
print("This can take several minutes.")

modeller.addMembrane(
    ff,
    lipidType="POPC",
    membraneCenterZ=0.0*unit.nanometer,
    minimumPadding=0.89*unit.nanometer,
    positiveIon="Na+",
    negativeIon="Cl-",
    ionicStrength=0.15*unit.molar,
    neutralize=True,
)

natoms = modeller.topology.getNumAtoms()

print("\nMembrane construction complete.")
print("Atoms after membrane:", natoms)

# Count residue types
counts = {}
for res in modeller.topology.residues():
    counts[res.name] = counts.get(res.name, 0) + 1

print("\nResidue inventory:")
for name, count in sorted(counts.items()):
    if name in [
        "POPC", "HOH", "WAT",
        "NA", "CL", "Na+", "Cl-",
        "MOL", "UNK"
    ]:
        print(f"{name:8s} {count}")

print("\nPeriodic box vectors:")
box = modeller.topology.getPeriodicBoxVectors()
for v in box:
    print(v)

print("\nWriting coordinates...")
pdb_out = outdir / "AEA_CB1_POPC_solvated.pdb"

with open(pdb_out, "w") as f:
    PDBFile.writeFile(
        modeller.topology,
        modeller.positions,
        f,
        keepIds=True
    )

print("Saved:", pdb_out)

print("\nCreating PME system...")
system = ff.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0*unit.nanometer,
    constraints=HBonds,
    rigidWater=True,
)

print("System particles:", system.getNumParticles())
print("Topology atoms:", natoms)

if system.getNumParticles() != natoms:
    raise RuntimeError(
        "Particle/topology atom count mismatch!"
    )

xml_out = outdir / "AEA_CB1_POPC_system.xml"

with open(xml_out, "w") as f:
    f.write(XmlSerializer.serialize(system))

print("Saved:", xml_out)

print("\nSUCCESS")
