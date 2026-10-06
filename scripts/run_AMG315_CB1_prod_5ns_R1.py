from openmm import *
from openmm.app import *
from openmm.unit import *

PDB = "systems/AMG315_CB1/equil/AMG315_CB1_NPT_unrestrained_final.pdb"
STATE = "systems/AMG315_CB1/equil/AMG315_CB1_NPT_unrestrained_state.xml"
SYSTEM = "systems/AMG315_CB1/AMG315_CB1_POPC_system.xml"

OUTDIR = "systems/AMG315_CB1/replicas/R1"
PREFIX = f"{OUTDIR}/AMG315_CB1_prod_5ns_R1"

pdb = PDBFile(PDB)

with open(SYSTEM) as f:
    system = XmlSerializer.deserialize(f.read())

barostat = MonteCarloMembraneBarostat(
    1*bar,
    0*bar*nanometer,
    310*kelvin,
    MonteCarloMembraneBarostat.XYIsotropic,
    MonteCarloMembraneBarostat.ZFree,
    25
)
barostat.setRandomNumberSeed(20260917)
system.addForce(barostat)

integrator = LangevinMiddleIntegrator(
    310*kelvin,
    1/picosecond,
    0.002*picoseconds
)
integrator.setRandomNumberSeed(20260918)

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

simulation.context.setPositions(state.getPositions())
simulation.context.setVelocities(state.getVelocities())
simulation.context.setPeriodicBoxVectors(
    *state.getPeriodicBoxVectors()
)

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
        totalSteps=2500000,
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
print("Starting AMG315 R1 5 ns production...")

simulation.step(2500000)

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

print("AMG315 R1 5-NS PRODUCTION COMPLETE")
