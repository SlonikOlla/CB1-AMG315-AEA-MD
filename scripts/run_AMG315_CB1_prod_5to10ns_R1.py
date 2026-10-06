from openmm import *
from openmm.app import *
from openmm.unit import *
from pathlib import Path

REPLICA = "R1"

PDB = (
    "systems/AMG315_CB1/replicas/R1/"
    "AMG315_CB1_prod_5ns_R1_final.pdb"
)

CHECKPOINT = (
    "systems/AMG315_CB1/replicas/R1/"
    "AMG315_CB1_prod_5ns_R1.chk"
)

SYSTEM = "systems/AMG315_CB1/AMG315_CB1_POPC_system.xml"

OUTDIR = "systems/AMG315_CB1/replicas/R1"
PREFIX = (
    f"{OUTDIR}/AMG315_CB1_prod_5to10ns_R1"
)

Path(OUTDIR).mkdir(
    parents=True,
    exist_ok=True
)

pdb = PDBFile(PDB)

with open(SYSTEM) as handle:
    system = XmlSerializer.deserialize(
        handle.read()
    )

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

platform = Platform.getPlatformByName(
    "OpenCL"
)

properties = {
    "Precision": "mixed"
}

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform,
    properties
)

print("Loading checkpoint:", CHECKPOINT)
simulation.loadCheckpoint(CHECKPOINT)

starting_state = simulation.context.getState()
starting_step = simulation.currentStep
starting_time = starting_state.getTime().value_in_unit(
    picoseconds
)

if starting_step != 2500000:
    raise RuntimeError(
        f"Unexpected starting step: {starting_step}"
    )

if abs(starting_time - 5000.0) >= 0.01:
    raise RuntimeError(
        f"Unexpected starting time: {starting_time} ps"
    )

print("Replica:", REPLICA)
print("Platform:", simulation.context.getPlatform().getName())
print("Atoms:", system.getNumParticles())
print(
    "Precision:",
    platform.getPropertyValue(
        simulation.context,
        "Precision"
    )
)
print("Starting step:", starting_step)
print("Starting time:", starting_state.getTime())

simulation.reporters.append(
    DCDReporter(
        PREFIX + ".dcd",
        5000
    )
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

simulation.reporters.append(
    CheckpointReporter(
        PREFIX + ".chk",
        25000
    )
)

print(
    "Continuing AMG315 R1 "
    "from 5 to 10 ns..."
)

simulation.step(2500000)

final_state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True,
    enforcePeriodicBox=True
)

final_step = simulation.currentStep
final_time = final_state.getTime().value_in_unit(
    picoseconds
)

if final_step != 5000000:
    raise RuntimeError(
        f"Unexpected final step: {final_step}"
    )

if abs(final_time - 10000.0) >= 0.01:
    raise RuntimeError(
        f"Unexpected final time: {final_time} ps"
    )

with open(PREFIX + "_state.xml", "w") as handle:
    handle.write(
        XmlSerializer.serialize(final_state)
    )

simulation.topology.setPeriodicBoxVectors(
    final_state.getPeriodicBoxVectors()
)

with open(PREFIX + "_final.pdb", "w") as handle:
    PDBFile.writeFile(
        simulation.topology,
        final_state.getPositions(),
        handle,
        keepIds=True
    )

print("Final step:", final_step)
print("Final time:", final_state.getTime())
print(
    "Final potential energy:",
    final_state.getPotentialEnergy()
)
print(
    "Final kinetic energy:",
    final_state.getKineticEnergy()
)
print("Saved:", PREFIX + ".dcd")
print("Saved:", PREFIX + ".log")
print("Saved:", PREFIX + ".chk")
print("Saved:", PREFIX + "_state.xml")
print("Saved:", PREFIX + "_final.pdb")
print(
    "AMG315 R1 5-TO-10-NS "
    "PRODUCTION COMPLETE"
)
