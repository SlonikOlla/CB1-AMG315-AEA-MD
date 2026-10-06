from openmm import *
from openmm.app import *
from openmm import unit
import numpy as np
import os

PDB = "systems/AMG315_CB1/equil/AMG315_CB1_NPT_unrestrained_final.pdb"
STATE_XML = "systems/AMG315_CB1/equil/AMG315_CB1_NPT_unrestrained_state.xml"
SYSTEM_XML = "systems/AMG315_CB1/AMG315_CB1_POPC_system.xml"

OUTDIR = "systems/AMG315_CB1/replicas/R3/stabilization"
os.makedirs(OUTDIR, exist_ok=True)

OUT_DCD = f"{OUTDIR}/AMG315_R3_NVT_100ps.dcd"
OUT_LOG = f"{OUTDIR}/AMG315_R3_NVT_100ps.log"
OUT_STATE = f"{OUTDIR}/AMG315_R3_NVT_100ps_state.xml"
OUT_PDB = f"{OUTDIR}/AMG315_R3_NVT_100ps_final.pdb"

pdb = PDBFile(PDB)

with open(SYSTEM_XML) as f:
    system = XmlSerializer.deserialize(f.read())

with open(STATE_XML) as f:
    oldstate = XmlSerializer.deserialize(f.read())

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

print("Restrained heavy atoms:", nrest)

integrator = LangevinMiddleIntegrator(
    310*unit.kelvin,
    1.0/unit.picosecond,
    0.002*unit.picoseconds
)

integrator.setRandomNumberSeed(830201)

platform = Platform.getPlatformByName("OpenCL")
properties = {"Precision": "mixed"}

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform,
    properties
)

simulation.context.setPeriodicBoxVectors(
    *oldstate.getPeriodicBoxVectors()
)
simulation.context.setPositions(pdb.positions)

# Fresh independent R3 velocities.
simulation.context.setVelocitiesToTemperature(
    310*unit.kelvin,
    830202
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

print("AMG315 R3: running 100 ps restrained NVT...")
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
print("AMG315 R3 NVT STABILIZATION COMPLETE")
