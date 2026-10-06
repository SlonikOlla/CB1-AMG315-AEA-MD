from openmm import *
from openmm.app import *
from openmm import unit
import numpy as np
import os

PDB = "systems/AEA_CB1/AEA_CB1_min_stage1.pdb"
SYSTEM_XML = "systems/AEA_CB1/AEA_CB1_POPC_system.xml"

OUTDIR = "systems/AEA_CB1/equil"
os.makedirs(OUTDIR, exist_ok=True)

OUT_DCD = f"{OUTDIR}/AEA_CB1_NVT_stage1.dcd"
OUT_LOG = f"{OUTDIR}/AEA_CB1_NVT_stage1.log"
OUT_STATE = f"{OUTDIR}/AEA_CB1_NVT_stage1_state.xml"
OUT_PDB = f"{OUTDIR}/AEA_CB1_NVT_stage1_final.pdb"

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

# 10 kcal/mol/A^2
k = 4184.0 * unit.kilojoule_per_mole / unit.nanometer**2

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
            [xyz[i,0], xyz[i,1], xyz[i,2]]
        )
        nrest += 1

system.addForce(restraint)

print("Restrained heavy atoms:", nrest)

integrator = LangevinMiddleIntegrator(
    310*unit.kelvin,
    1.0/unit.picosecond,
    0.002*unit.picoseconds
)

integrator.setRandomNumberSeed(20260906)

platform = Platform.getPlatformByName("CPU")

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform
)

simulation.context.setPositions(pdb.positions)

# Use velocities consistent with target temperature.
simulation.context.setVelocitiesToTemperature(
    310*unit.kelvin,
    20260906
)

# 100 ps = 50,000 steps at 2 fs.
nsteps = 50000

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

print("Running 100 ps NVT...")
simulation.step(nsteps)

state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True
)

with open(OUT_STATE, "w") as f:
    f.write(XmlSerializer.serialize(state))

with open(OUT_PDB, "w") as f:
    PDBFile.writeFile(
        pdb.topology,
        state.getPositions(),
        f,
        keepIds=True
    )

print("Final potential energy:", state.getPotentialEnergy())
print("Final kinetic energy:", state.getKineticEnergy())
print("Saved:", OUT_DCD)
print("Saved:", OUT_LOG)
print("Saved:", OUT_STATE)
print("Saved:", OUT_PDB)
print("NVT STAGE 1 COMPLETE")
