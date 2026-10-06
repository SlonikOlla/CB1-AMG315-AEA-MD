# Recovered software environment

This file records software information explicitly recovered from the archived project record.

- Python: 3.11.4
- Open Force Field Sage: 2.2.1
- MDTraj: 1.11.1

The exact OpenMM package build was not preserved in the recovered environment capture and is therefore not inferred here.

## Simulation model details

- Protein force field: Amber ff19SB
- Lipid force field: Amber Lipid21
- Water model: TIP3P
- Ligand force field: Open Force Field Sage 2.2.1
- Ligand charges: AM1-BCC
- Electrostatics: particle-mesh Ewald
- Real-space nonbonded cutoff: 1.0 nm
- Constraints: bonds involving hydrogen
- Temperature: 310 K
- Integrator: LangevinMiddleIntegrator
- Friction coefficient: 1 ps^-1
- Time step: 2 fs
- Pressure: 1 bar
- Barostat: MonteCarloMembraneBarostat, XY isotropic, Z free
- Barostat interval: 25 steps
