from openmm import *
from openmm.app import *
from openmm import unit
import numpy as np
import os

PDB = "systems/AEA_CB1/equil/AEA_CB1_NPT_stage2_final.pdb"
STATE_XML = "systems/AEA_CB1/equil/AEA_CB1_NPT_stage2_state.xml"
SYSTEM_XML = "systems/AEA_CB1/AEA_CB1_POPC_system.xml"

OUTDIR = "systems/AEA_CB1/equil"
os.makedirs(OUTDIR, exist_ok=True)

OUT_DCD = f"{OUTDIR}/AEA_CB1_NPT_stage3.dcd"
OUT_LOG = f"{OUTDIR}/AEA_CB1_NPT_stage3.log"
OUT_STATE = f"{OUTDIR}/AEA_CB1_NPT_stage3_state.xml"
OUT_PDB = f"{OUTDIR}/AEA_CB1_NPT_stage3_final.pdb"

pdb = PDBFile(PDB)

with open(SYSTEM_XML) as f:
    system = XmlSerializer.deserialize(f.read())

with open(STATE_XML) as f:
    oldstate = XmlSerializer.deserialize(f.read())

atoms = list(pdb.topology.atoms())

refpos = np.asarray(
    oldstate.getPositions().value_in_unit(unit.nanometer),
    dtype=float
)

protein_names = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
    "TYR","VAL","HID","HIE","HIP","CYX","ASH","GLH"
}

# 1 kcal/mol/A^2 = 418.4 kJ/mol/nm^2
k = 418.4 * unit.kilojoule_per_mole / unit.nanometer**2

restraint = CustomExternalForce(
    "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
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
            [refpos[i,0], refpos[i,1], refpos[i,2]]
        )
        nrest += 1

system.addForce(restraint)

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

integrator.setRandomNumberSeed(20260908)

platform = Platform.getPlatformByName("CPU")

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform
)

# Continue directly from Stage 2
box = oldstate.getPeriodicBoxVectors()

simulation.context.setPeriodicBoxVectors(*box)
simulation.context.setPositions(oldstate.getPositions())
simulation.context.setVelocities(oldstate.getVelocities())

print("Restrained heavy atoms:", nrest)
print("Restraint strength: 1 kcal/mol/A^2")
print("Starting box:")
for v in box:
    print(v)

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

print("Running 500 ps NPT Stage 3...")
simulation.step(250000)

state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True
)

with open(OUT_STATE, "w") as f:
    f.write(XmlSerializer.serialize(state))

# Ensure PDB contains actual final NPT box
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

print("Final box:")
for v in state.getPeriodicBoxVectors():
    print(v)

print("Saved:", OUT_DCD)
print("Saved:", OUT_LOG)
print("Saved:", OUT_STATE)
print("Saved:", OUT_PDB)
print("NPT STAGE 3 COMPLETE")
