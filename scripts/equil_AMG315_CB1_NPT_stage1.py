from openmm import *
from openmm.app import *
from openmm import unit
import numpy as np
import os

PDB = "systems/AMG315_CB1/equil/AMG315_CB1_NVT_stage1_final.pdb"
SYSTEM_XML = "systems/AMG315_CB1/AMG315_CB1_POPC_system.xml"

OUTDIR = "systems/AMG315_CB1/equil"
os.makedirs(OUTDIR, exist_ok=True)

OUT_DCD = f"{OUTDIR}/AMG315_CB1_NPT_stage1.dcd"
OUT_LOG = f"{OUTDIR}/AMG315_CB1_NPT_stage1.log"
OUT_STATE = f"{OUTDIR}/AMG315_CB1_NPT_stage1_state.xml"
OUT_PDB = f"{OUTDIR}/AMG315_CB1_NPT_stage1_final.pdb"

pdb = PDBFile(PDB)

with open(SYSTEM_XML) as f:
    system = XmlSerializer.deserialize(f.read())

atoms = list(pdb.topology.atoms())
xyz = np.asarray(
    pdb.positions.value_in_unit(unit.nanometer),
    dtype=float
)

protein_names = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
    "TYR","VAL","HID","HIE","HIP","CYX","ASH","GLH"
}

# Strong positional restraint: 10 kcal/mol/A^2
k = 4184.0 * unit.kilojoule_per_mole / unit.nanometer**2

restraint = CustomExternalForce(
    "0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2"
)
restraint.addGlobalParameter("k", k)
restraint.addPerParticleParameter("x0")
restraint.addPerParticleParameter("y0")
restraint.addPerParticleParameter("z0")

nrest = 0

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
        nrest += 1

system.addForce(restraint)

# Semi-isotropic membrane barostat:
# OpenMM's membrane barostat lets XY scale together
# while Z responds independently.
barostat = MonteCarloMembraneBarostat(
    1.0 * unit.bar,
    0.0 * unit.bar * unit.nanometer,
    310.0 * unit.kelvin,
    MonteCarloMembraneBarostat.XYIsotropic,
    MonteCarloMembraneBarostat.ZFree,
    25
)

system.addForce(barostat)

integrator = LangevinMiddleIntegrator(
    310.0 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds
)

integrator.setRandomNumberSeed(20260906)

platform = Platform.getPlatformByName("OpenCL")
properties = {"Precision": "mixed"}

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform,
    properties
)

simulation.context.setPositions(pdb.positions)

# Use fresh velocities at target temperature.
simulation.context.setVelocitiesToTemperature(
    310.0 * unit.kelvin,
    20260906
)

# 500 ps = 250,000 steps
nsteps = 250000

simulation.reporters.append(
    StateDataReporter(
        OUT_LOG,
        5000,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        speed=True,
        separator="\t"
    )
)

simulation.reporters.append(
    DCDReporter(
        OUT_DCD,
        5000
    )
)

print("Restrained heavy atoms:", nrest)
print("Running 500 ps NPT with membrane barostat...")

simulation.step(nsteps)

state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True
)

with open(OUT_STATE, "w") as f:
    f.write(XmlSerializer.serialize(state))

# Write PDB with the actual final NPT periodic box
pdb.topology.setPeriodicBoxVectors(
    state.getPeriodicBoxVectors()
)

with open(OUT_PDB, "w") as f:
    PDBFile.writeFile(
        pdb.topology,
        state.getPositions(),
        f,
        keepIds=True
    )

print("Final potential energy:", state.getPotentialEnergy())
print("Final kinetic energy:", state.getKineticEnergy())

box = state.getPeriodicBoxVectors()
print("Final box vectors:")
for v in box:
    print(v)

print("Saved:", OUT_DCD)
print("Saved:", OUT_LOG)
print("Saved:", OUT_STATE)
print("Saved:", OUT_PDB)
print("NPT STAGE 1 COMPLETE")
