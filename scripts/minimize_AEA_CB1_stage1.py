from openmm import *
from openmm.app import *
from openmm import unit
import numpy as np

PDB = "systems/AEA_CB1/AEA_CB1_POPC_solvated.pdb"
XML = "systems/AEA_CB1/AEA_CB1_POPC_system.xml"

OUT_PDB = "systems/AEA_CB1/AEA_CB1_min_stage1.pdb"
OUT_XML = "systems/AEA_CB1/AEA_CB1_min_stage1_state.xml"

print("Loading system...")
pdb = PDBFile(PDB)

with open(XML) as f:
    system = XmlSerializer.deserialize(f.read())

atoms = list(pdb.topology.atoms())
positions = pdb.positions

print("Particles:", system.getNumParticles())
print("Topology atoms:", len(atoms))

# ------------------------------------------------------------
# Restrain CB1 + AEA heavy atoms.
# Solvent, lipid, ions, and hydrogens are free.
#
# k = 10 kcal/mol/A^2
# OpenMM equivalent = 4184 kJ/mol/nm^2
# ------------------------------------------------------------
k = 4184.0 * unit.kilojoule_per_mole / unit.nanometer**2

protein_names = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
    "TYR","VAL","HID","HIE","HIP","CYX","ASH","GLH"
}

restraint = CustomExternalForce(
    "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
)

restraint.addGlobalParameter("k", k)
restraint.addPerParticleParameter("x0")
restraint.addPerParticleParameter("y0")
restraint.addPerParticleParameter("z0")

xyz = np.asarray(
    positions.value_in_unit(unit.nanometer),
    dtype=float
)

n_restraints = 0

for atom in atoms:
    if atom.element is None:
        continue

    if atom.element.symbol == "H":
        continue

    if atom.residue.name in protein_names or atom.residue.name == "UNK":
        i = atom.index
        restraint.addParticle(
            i,
            [xyz[i,0], xyz[i,1], xyz[i,2]]
        )
        n_restraints += 1

system.addForce(restraint)

print("Restrained heavy atoms:", n_restraints)

# Integrator only supplies a Context; no MD yet.
integrator = LangevinMiddleIntegrator(
    310.0 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds
)

platform = Platform.getPlatformByName("CPU")

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform
)

simulation.context.setPositions(positions)

# Initial energy
state = simulation.context.getState(getEnergy=True)
E0 = state.getPotentialEnergy()

print("Initial potential energy:", E0)

print("Minimizing...")
simulation.minimizeEnergy(
    tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
    maxIterations=5000
)

state = simulation.context.getState(
    getEnergy=True,
    getPositions=True
)

E1 = state.getPotentialEnergy()

print("Final potential energy:", E1)
print("Energy change:", E1-E0)

with open(OUT_PDB, "w") as f:
    PDBFile.writeFile(
        pdb.topology,
        state.getPositions(),
        f,
        keepIds=True
    )

with open(OUT_XML, "w") as f:
    f.write(XmlSerializer.serialize(state))

print("Saved:", OUT_PDB)
print("Saved:", OUT_XML)
print("STAGE 1 MINIMIZATION COMPLETE")
