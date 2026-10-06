# ============================================================
# Independent CB1 production replica
# Ligand: AEA
# Replica: R2
#
# Same equilibrated positions and periodic box as R1.
# New velocities + independent Langevin/barostat seeds.
# Continuous production = 10 ns.
# ============================================================

import os
from openmm import *
from openmm.app import *
from openmm.unit import *

PDB = "systems/AEA_CB1/replicas/R2/stabilization/AEA_R2_unrestrained_500ps_final.pdb"
STATE = "systems/AEA_CB1/replicas/R2/stabilization/AEA_R2_unrestrained_500ps_state.xml"
SYSTEM = "systems/AEA_CB1/AEA_CB1_POPC_system.xml"

OUTDIR = "systems/AEA_CB1/replicas/R2"
PREFIX = "systems/AEA_CB1/replicas/R2/AEA_CB1_prod_10ns_R2"
INTEGRATOR_SEED = 820301
BAROSTAT_SEED = 820302

os.makedirs(os.path.dirname(PREFIX), exist_ok=True)

pdb = PDBFile(PDB)

with open(SYSTEM) as f:
    system = XmlSerializer.deserialize(f.read())

system.addForce(
    MonteCarloMembraneBarostat(
        1*bar,
        0*bar*nanometer,
        310*kelvin,
        MonteCarloMembraneBarostat.XYIsotropic,
        MonteCarloMembraneBarostat.ZFree,
        25
    )
)

integrator = LangevinMiddleIntegrator(
    310*kelvin,
    1/picosecond,
    0.002*picoseconds
)
integrator.setRandomNumberSeed(INTEGRATOR_SEED)

# Independent barostat random seed.
_barostat_found = False
for force in system.getForces():
    if isinstance(force, MonteCarloMembraneBarostat):
        force.setRandomNumberSeed(BAROSTAT_SEED)
        _barostat_found = True

if not _barostat_found:
    raise RuntimeError("MonteCarloMembraneBarostat not found")

platform = Platform.getPlatformByName("OpenCL")

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform,
    {"Precision": "mixed"}
)

with open(STATE) as f:
    state = XmlSerializer.deserialize(f.read())

simulation.context.setPeriodicBoxVectors(
    *state.getPeriodicBoxVectors()
)
simulation.context.setPositions(state.getPositions())
simulation.context.setVelocities(state.getVelocities())

# 10 ps trajectory/reporting
simulation.reporters.append(
    DCDReporter(PREFIX + ".dcd", 5000)
)

simulation.reporters.append(
    StateDataReporter(
        PREFIX + ".log",
        5000,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        speed=True,
        remainingTime=True,
        totalSteps=5000000,
        separator="\t"
    )
)

# Restart checkpoint every 50 ps
simulation.reporters.append(
    CheckpointReporter(PREFIX + ".chk", 25000)
)

print("Platform:", simulation.context.getPlatform().getName())
print("Atoms:", system.getNumParticles())
print("Precision:", platform.getPropertyValue(simulation.context, "Precision"))
print("Starting AEA R2 10 ns production...")

simulation.step(5000000)

final_state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True,
    enforcePeriodicBox=True
)

with open(PREFIX + "_state.xml", "w") as f:
    f.write(XmlSerializer.serialize(final_state))

simulation.topology.setPeriodicBoxVectors(
    final_state.getPeriodicBoxVectors()
)

with open(PREFIX + "_final.pdb", "w") as f:
    PDBFile.writeFile(
        simulation.topology,
        final_state.getPositions(),
        f,
        keepIds=True
    )

print("Production complete.")
