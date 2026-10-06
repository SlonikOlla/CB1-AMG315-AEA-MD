from openmm import *
from openmm.app import *
from openmm import unit
import os

PDB = "systems/AEA_CB1/equil/AEA_CB1_NPT_unrestrained_final.pdb"
STATE_XML = "systems/AEA_CB1/equil/AEA_CB1_NPT_unrestrained_state.xml"
SYSTEM_XML = "systems/AEA_CB1/AEA_CB1_POPC_system.xml"

OUTDIR = "systems/AEA_CB1/equil"
os.makedirs(OUTDIR, exist_ok=True)

OUT_DCD = f"{OUTDIR}/AEA_CB1_NPT_unrestrained_ext1500ps.dcd"
OUT_LOG = f"{OUTDIR}/AEA_CB1_NPT_unrestrained_ext1500ps.log"
OUT_STATE = f"{OUTDIR}/AEA_CB1_NPT_unrestrained_ext1500ps_state.xml"
OUT_PDB = f"{OUTDIR}/AEA_CB1_NPT_unrestrained_ext1500ps_final.pdb"

pdb = PDBFile(PDB)

with open(SYSTEM_XML) as f:
    system = XmlSerializer.deserialize(f.read())

with open(STATE_XML) as f:
    oldstate = XmlSerializer.deserialize(f.read())

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

integrator.setRandomNumberSeed(20260910)

platform = Platform.getPlatformByName("CPU")

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform
)

box = oldstate.getPeriodicBoxVectors()

simulation.context.setPeriodicBoxVectors(*box)
simulation.context.setPositions(oldstate.getPositions())
simulation.context.setVelocities(oldstate.getVelocities())

print("UNRESTRAINED NPT")
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

print("Running 1500 ps unrestrained NPT extension...")
simulation.step(750000)

state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True
)

with open(OUT_STATE, "w") as f:
    f.write(XmlSerializer.serialize(state))

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
print("UNRESTRAINED 1500-PS EXTENSION COMPLETE")
