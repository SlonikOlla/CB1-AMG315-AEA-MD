# CB1 AMG315–AEA molecular dynamics workflow

Code and workflow documentation supporting the manuscript:

**Comparative molecular dynamics reveals distinct bound-state dynamics of AMG315 and anandamide at cannabinoid receptor 1**

This repository is intentionally **code-only**. It does not contain numerical results, derived result tables, figures, trajectories, checkpoints, coordinate outputs, or other study results.

## Scope

The repository is intended to archive the scripts used for:

- CB1 system construction and ligand parameterization
- minimization and equilibration
- replica-specific production molecular dynamics
- trajectory assembly and quality-control checks
- receptor/ligand stability measurements
- pocket-retention and residue-contact analysis
- ligand pose-state feature construction and clustering
- CB1 receptor microswitch analysis
- comparative AMG315-versus-AEA analysis

## Project layout

```
scripts/        Simulation and analysis scripts
slurm/          Cluster submission scripts
docs/           Workflow and provenance notes
```

## Reproducibility

The manuscript analysis used OpenMM-based molecular dynamics and MDTraj-based trajectory analysis. Ligands were parameterized with Open Force Field Sage 2.2.1 and AM1-BCC charges. Protein, lipid and water models were Amber ff19SB, Lipid21 and TIP3P, respectively.

Exact project scripts from the original Nibi working directory will be added before the archival release is frozen.

## Data and results

No study results are included in this repository. Raw trajectories and derived outputs are intentionally excluded.

## Citation

A versioned archival DOI will be added after the GitHub release is connected to Zenodo.

## License

MIT License.
