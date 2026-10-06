from openmm import *
from openmm.app import *
from openmm.unit import *
from pathlib import Path

REPLICA = "R2"

PDB = (
    "systems/AMG315_CB1/replicas/R2/"
    "AMG315_CB1_prod_5to10ns_R2_final.pdb"
)
CHECKPOINT = (
    "systems/AMG315_CB1/replicas/R2/"
    "AMG315_CB1_prod_5to10ns_R2.chk"
)
SYSTEM = "systems/AMG315_CB1/AMG315_CB1_POPC_system.xml"

OUTDIR = f"systems/AMG315_CB1/replicas/{REPLICA}"
PREFIX = (
    f"{OUTDIR}/AMG315_CB1_prod_10to20ns_{REPLICA}"
)

INTEGRATOR_SEED = 820301
BAROSTAT_SEED = 820302

N_STEPS = 5_000_000
EXPECTED_START_STEP = 5_000_000
EXPECTED_FINAL_STEP = 10_000_000

Path(OUTDIR).mkdir(parents=True, exist_ok=True)

pdb = PDBFile(PDB)

with open(SYSTEM) as handle:
    system = XmlSerializer.deserialize(handle.read())

barostat = MonteCarloMembraneBarostat(
    1 * bar,
    0 * bar * nanometer,
    310 * kelvin,
    MonteCarloMembraneBarostat.XYIsotropic,
    MonteCarloMembraneBarostat.ZFree,
    25
)
barostat.setRandomNumberSeed(BAROSTAT_SEED)
system.addForce(barostat)

integrator = LangevinMiddleIntegrator(
    310 * kelvin,
    1 / picosecond,
    0.002 * picoseconds
)
integrator.setRandomNumberSeed(INTEGRATOR_SEED)

platform = Platform.getPlatformByName("OpenCL")

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform,
    {"Precision": "mixed"}
)

print("Loading checkpoint:", CHECKPOINT)
simulation.loadCheckpoint(CHECKPOINT)

start_step = simulation.currentStep
start_state = simulation.context.getState()
start_time = start_state.getTime()

print("Platform:", simulation.context.getPlatform().getName())
print("Atoms:", system.getNumParticles())
print(
    "Precision:",
    platform.getPropertyValue(simulation.context, "Precision")
)
print("Starting step:", start_step)
print("Starting time:", start_time)

if start_step != EXPECTED_START_STEP:
    raise RuntimeError(
        f"Expected checkpoint step {EXPECTED_START_STEP}, "
        f"found {start_step}"
    )

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
        totalSteps=EXPECTED_FINAL_STEP,
        separator="\t"
    )
)

simulation.reporters.append(
    CheckpointReporter(PREFIX + ".chk", 25000)
)

print(f"Continuing AMG315 {REPLICA} from 10 to 20 ns...")
simulation.step(N_STEPS)

final_state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True,
    enforcePeriodicBox=True
)

if simulation.currentStep != EXPECTED_FINAL_STEP:
    raise RuntimeError(
        f"Expected final step {EXPECTED_FINAL_STEP}, "
        f"found {simulation.currentStep}"
    )

with open(PREFIX + "_state.xml", "w") as handle:
    handle.write(XmlSerializer.serialize(final_state))

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

print("Final step:", simulation.currentStep)
print("Final time:", final_state.getTime())
print("Final potential energy:", final_state.getPotentialEnergy())
print("Final kinetic energy:", final_state.getKineticEnergy())
print("Saved:", PREFIX + ".dcd")
print("Saved:", PREFIX + ".log")
print("Saved:", PREFIX + ".chk")
print("Saved:", PREFIX + "_state.xml")
print("Saved:", PREFIX + "_final.pdb")
print(f"AMG315 {REPLICA} 10-20 ns continuation complete.")
