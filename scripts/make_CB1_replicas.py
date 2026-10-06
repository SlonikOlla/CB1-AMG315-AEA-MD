from pathlib import Path

jobs = [
    ("THC", "R2",
     "scripts/run_THC_CB1_prod_5ns.py",
     "scripts/run_THC_CB1_prod_10ns_R2.py",
     "systems/THC_CB1/replicas/R2/THC_CB1_prod_10ns_R2",
     620201, 620202, 620203),

    ("THC", "R3",
     "scripts/run_THC_CB1_prod_5ns.py",
     "scripts/run_THC_CB1_prod_10ns_R3.py",
     "systems/THC_CB1/replicas/R3/THC_CB1_prod_10ns_R3",
     630201, 630202, 630203),

    ("CP55940", "R2",
     "scripts/run_CP55940_CB1_prod_5ns.py",
     "scripts/run_CP55940_CB1_prod_10ns_R2.py",
     "systems/CP55940_CB1/replicas/R2/CP55940_CB1_prod_10ns_R2",
     720201, 720202, 720203),

    ("CP55940", "R3",
     "scripts/run_CP55940_CB1_prod_5ns.py",
     "scripts/run_CP55940_CB1_prod_10ns_R3.py",
     "systems/CP55940_CB1/replicas/R3/CP55940_CB1_prod_10ns_R3",
     730201, 730202, 730203),
]


for ligand, rep, template, outfile, prefix, iseed, vseed, bseed in jobs:

    text = Path(template).read_text()

    # ------------------------------------------------------------
    # 1. Make sure os is imported.
    # ------------------------------------------------------------
    if "import os\n" not in text:
        text = "import os\n" + text

    # ------------------------------------------------------------
    # 2. Replace PREFIX robustly, regardless of original formatting.
    # ------------------------------------------------------------
    lines = text.splitlines()

    prefix_hits = [
        i for i, line in enumerate(lines)
        if line.strip().startswith("PREFIX =")
    ]

    if len(prefix_hits) != 1:
        raise RuntimeError(
            f"{template}: expected exactly one PREFIX line, "
            f"found {len(prefix_hits)}"
        )

    i = prefix_hits[0]

    lines[i] = f'PREFIX = "{prefix}"'

    seed_lines = [
        f"INTEGRATOR_SEED = {iseed}",
        f"VELOCITY_SEED = {vseed}",
        f"BAROSTAT_SEED = {bseed}",
        "",
        "os.makedirs(os.path.dirname(PREFIX), exist_ok=True)",
    ]

    lines[i+1:i+1] = seed_lines

    text = "\n".join(lines) + "\n"

    # ------------------------------------------------------------
    # 3. Seed Langevin integrator.
    # Insert immediately after closing parenthesis of its constructor.
    # ------------------------------------------------------------
    lines = text.splitlines()

    start = None
    end = None

    for i, line in enumerate(lines):
        if "integrator = LangevinMiddleIntegrator(" in line:
            start = i
            break

    if start is None:
        raise RuntimeError(
            f"{template}: LangevinMiddleIntegrator not found"
        )

    for i in range(start + 1, min(start + 15, len(lines))):
        if lines[i].strip() == ")":
            end = i
            break

    if end is None:
        raise RuntimeError(
            f"{template}: end of integrator constructor not found"
        )

    lines.insert(
        end + 1,
        "integrator.setRandomNumberSeed(INTEGRATOR_SEED)"
    )

    text = "\n".join(lines) + "\n"

    # ------------------------------------------------------------
    # 4. Explicitly seed the existing membrane barostat.
    # Do this before Simulation is constructed.
    # ------------------------------------------------------------
    marker = 'platform = Platform.getPlatformByName("OpenCL")'

    if marker not in text:
        raise RuntimeError(
            f"{template}: OpenCL platform line not found"
        )

    barostat_code = '''# Independent barostat random seed.
_barostat_found = False
for force in system.getForces():
    if isinstance(force, MonteCarloMembraneBarostat):
        force.setRandomNumberSeed(BAROSTAT_SEED)
        _barostat_found = True

if not _barostat_found:
    raise RuntimeError("MonteCarloMembraneBarostat not found")

'''

    text = text.replace(
        marker,
        barostat_code + marker,
        1
    )

    # ------------------------------------------------------------
    # 5. NEW Maxwell-Boltzmann velocities.
    # Do NOT reuse R1 velocities.
    # ------------------------------------------------------------
    old = "simulation.context.setVelocities(state.getVelocities())"

    new = '''simulation.context.setVelocitiesToTemperature(
    310*kelvin,
    VELOCITY_SEED
)'''

    if old not in text:
        raise RuntimeError(
            f"{template}: original velocity line not found"
        )

    text = text.replace(old, new, 1)

    # ------------------------------------------------------------
    # 6. Change production from 5 ns to 10 ns.
    # 2 fs * 5,000,000 steps = 10 ns.
    # ------------------------------------------------------------
    oldstep = "simulation.step(2500000)"
    newstep = "simulation.step(5000000)"

    if oldstep not in text:
        raise RuntimeError(
            f"{template}: simulation.step(2500000) not found"
        )

    text = text.replace(oldstep, newstep, 1)

    # ------------------------------------------------------------
    # 7. Header.
    # ------------------------------------------------------------
    header = f'''# ============================================================
# Independent CB1 production replica
# Ligand: {ligand}
# Replica: {rep}
#
# Same equilibrated positions and periodic box as R1.
# New velocities + independent Langevin/barostat seeds.
# Continuous production = 10 ns.
# ============================================================

'''

    text = header + text

    Path(outfile).write_text(text)

    print(f"CREATED: {outfile}")
    print(f"  PREFIX          = {prefix}")
    print(f"  integrator seed = {iseed}")
    print(f"  velocity seed   = {vseed}")
    print(f"  barostat seed   = {bseed}")
    print()


print("ALL FOUR REPLICA SCRIPTS CREATED")
