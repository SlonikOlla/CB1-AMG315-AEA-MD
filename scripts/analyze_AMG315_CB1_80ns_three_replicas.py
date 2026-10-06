import mdtraj as md
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path("systems/AMG315_CB1")
OUT = Path("results/AMG315_CB1_production_QC_80ns")
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "R1": {
        "reference": BASE / "equil/AMG315_CB1_NPT_unrestrained_final.pdb",
        "dcd_segments": [
            BASE / "replicas/R1/AMG315_CB1_prod_5ns_R1.dcd",
            BASE / "replicas/R1/AMG315_CB1_prod_5to10ns_R1.dcd",
            BASE / "replicas/R1/AMG315_CB1_prod_10to20ns_R1.dcd",
            BASE / "replicas/R1/AMG315_CB1_prod_20to30ns_R1.dcd",
            BASE / "replicas/R1/AMG315_CB1_prod_30to40ns_R1.dcd",
            BASE / "replicas/R1/AMG315_CB1_prod_40to50ns_R1.dcd",
            BASE / "replicas/R1/AMG315_CB1_prod_50to60ns_R1.dcd",
            BASE / "replicas/R1/AMG315_CB1_prod_60to70ns_R1.dcd",
            BASE / "replicas/R1/AMG315_CB1_prod_70to80ns_R1.dcd",
        ],
    },
    "R2": {
        "reference": BASE / "replicas/R2/stabilization/AMG315_R2_unrestrained_500ps_final.pdb",
        "dcd_segments": [
            BASE / "replicas/R2/AMG315_CB1_prod_5ns_R2.dcd",
            BASE / "replicas/R2/AMG315_CB1_prod_5to10ns_R2.dcd",
            BASE / "replicas/R2/AMG315_CB1_prod_10to20ns_R2.dcd",
            BASE / "replicas/R2/AMG315_CB1_prod_20to30ns_R2.dcd",
            BASE / "replicas/R2/AMG315_CB1_prod_30to40ns_R2.dcd",
            BASE / "replicas/R2/AMG315_CB1_prod_40to50ns_R2.dcd",
            BASE / "replicas/R2/AMG315_CB1_prod_50to60ns_R2.dcd",
            BASE / "replicas/R2/AMG315_CB1_prod_60to70ns_R2.dcd",
            BASE / "replicas/R2/AMG315_CB1_prod_70to80ns_R2.dcd",
        ],
    },
    "R3": {
        "reference": BASE / "replicas/R3/stabilization/AMG315_R3_unrestrained_500ps_final.pdb",
        "dcd_segments": [
            BASE / "replicas/R3/AMG315_CB1_prod_5ns_R3.dcd",
            BASE / "replicas/R3/AMG315_CB1_prod_5to10ns_R3.dcd",
            BASE / "replicas/R3/AMG315_CB1_prod_10to20ns_R3.dcd",
            BASE / "replicas/R3/AMG315_CB1_prod_20to30ns_R3.dcd",
            BASE / "replicas/R3/AMG315_CB1_prod_30to40ns_R3.dcd",
            BASE / "replicas/R3/AMG315_CB1_prod_40to50ns_R3.dcd",
            BASE / "replicas/R3/AMG315_CB1_prod_50to60ns_R3.dcd",
            BASE / "replicas/R3/AMG315_CB1_prod_60to70ns_R3.dcd",
            BASE / "replicas/R3/AMG315_CB1_prod_70to80ns_R3.dcd",
        ],
    },
}

protein_names = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
    "TYR","VAL","HID","HIE","HIP","CYX","ASH","GLH"
}

modeled_ranges = [(142,147), (254,265), (314,334)]

def modeled(resseq):
    return any(lo <= resseq <= hi for lo, hi in modeled_ranges)

def kabsch(mobile, target):
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)

    u, _, vt = np.linalg.svd(
        (mobile - mobile_center).T @
        (target - target_center)
    )

    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt

    return rotation, mobile_center, target_center

def rmsd(mobile, target):
    difference = mobile - target
    return np.sqrt(np.mean(np.sum(difference * difference, axis=1)))

def unwrap_ligand(xyz, ligand, box):
    anchor = xyz[ligand[0]].copy()
    delta = xyz[ligand] - anchor
    delta -= box * np.round(delta / box)
    xyz[ligand] = anchor + delta

def slope_per_ns(table, column):
    subset = table[table.time_ps > 70000]
    return np.polyfit(
        subset.time_ps.to_numpy() / 1000.0,
        subset[column].to_numpy(),
        1
    )[0]

# Define a single matched TM-core selection from the R1 reference.
r1_reference = md.load(str(CONFIG["R1"]["reference"]))
r1_top = r1_reference.topology
r1_xyz = r1_reference.xyz[0]

r1_protein_ca = np.array([
    atom.index for atom in r1_top.atoms
    if atom.residue.name in protein_names
    and atom.name == "CA"
], dtype=int)

# Fixed cross-ligand CB1 TM-core residue set.
# This exact 147-residue set was defined geometrically from the AEA R1
# reference and is reused unchanged for AMG315 to ensure directly
# comparable RMSD measurements across ligand systems.
FIXED_TM_CORE_RESSEQ = {
    116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 367, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396
}

fixed_tm_core = np.array([
    atom.index for atom in r1_top.atoms
    if atom.residue.name in protein_names
    and atom.name == "CA"
    and atom.residue.resSeq in FIXED_TM_CORE_RESSEQ
], dtype=int)

assert len(FIXED_TM_CORE_RESSEQ) == 147
assert len(fixed_tm_core) == 147

summary_rows = []

print("=" * 78)
print("AMG315-CB1 THREE-REPLICA 80-NS PBC-CORRECTED PRODUCTION ANALYSIS")
print("=" * 78)
print("Fixed R1-defined TM-core CA atoms:", len(fixed_tm_core))
print("Physical trajectory times: 10-80000 ps")

for replica, config in CONFIG.items():
    print("\n" + "=" * 78)
    print(replica)
    print("=" * 78)

    reference = md.load(str(config["reference"]))
    segments = [
        md.load(
            str(dcd),
            top=str(config["reference"])
        )
        for dcd in config["dcd_segments"]
    ]

    assert len(segments) == 9
    assert segments[0].n_frames == 500
    assert segments[1].n_frames == 500
    assert segments[2].n_frames == 1000
    assert segments[3].n_frames == 1000
    assert segments[4].n_frames == 1000
    assert segments[5].n_frames == 1000
    assert segments[6].n_frames == 1000
    assert segments[7].n_frames == 1000
    assert segments[8].n_frames == 1000
    assert (
        segments[8].n_atoms ==
        segments[7].n_atoms ==
        segments[6].n_atoms ==
        segments[5].n_atoms ==
        segments[0].n_atoms
    )
    assert (
        segments[0].n_atoms ==
        segments[1].n_atoms ==
        segments[2].n_atoms ==
        segments[3].n_atoms ==
        segments[4].n_atoms
    )

    trajectory = md.join(
        segments,
        check_topology=True
    )

    assert trajectory.n_frames == 8000
    assert trajectory.n_atoms == reference.n_atoms == 42295
    assert trajectory.unitcell_lengths is not None

    top = reference.topology
    atoms = list(top.atoms)
    ref_xyz = reference.xyz[0].copy()

    protein_ca = np.array([
        atom.index for atom in atoms
        if atom.residue.name in protein_names
        and atom.name == "CA"
    ], dtype=int)

    stable_core_ca = np.array([
        atom.index for atom in atoms
        if atom.residue.name in protein_names
        and atom.name == "CA"
        and not modeled(atom.residue.resSeq)
    ], dtype=int)

    protein_heavy = np.array([
        atom.index for atom in atoms
        if atom.residue.name in protein_names
        and atom.element is not None
        and atom.element.symbol != "H"
    ], dtype=int)

    ligand_heavy = np.array([
        atom.index for atom in atoms
        if atom.residue.name == "UNK"
        and atom.element is not None
        and atom.element.symbol != "H"
    ], dtype=int)

    assert len(protein_ca) == 304
    assert len(stable_core_ca) == 265
    assert len(protein_heavy) == 2426
    assert len(ligand_heavy) == 27

    ref_box = reference.unitcell_lengths[0]
    unwrap_ligand(ref_xyz, ligand_heavy, ref_box)

    ref_distances = np.linalg.norm(
        ref_xyz[ligand_heavy][:, None, :] -
        ref_xyz[protein_heavy][None, :, :],
        axis=2
    )

    pocket_heavy = protein_heavy[
        np.any(ref_distances <= 0.5, axis=0)
    ]

    assert len(pocket_heavy) > 0

    ref_core = ref_xyz[stable_core_ca]
    ref_tm = ref_xyz[fixed_tm_core]
    ref_ca = ref_xyz[protein_ca]
    ref_protein = ref_xyz[protein_heavy]
    ref_ligand = ref_xyz[ligand_heavy]
    ref_ligand_center = ref_ligand.mean(axis=0)

    records = []
    contact_counts = defaultdict(int)

    for frame in range(trajectory.n_frames):
        xyz = trajectory.xyz[frame].copy()
        box = trajectory.unitcell_lengths[frame].copy()

        assert np.isfinite(xyz).all()
        assert np.isfinite(box).all()
        assert np.all(box > 0)

        # Reconstruct AMG315 as one molecule.
        unwrap_ligand(xyz, ligand_heavy, box)

        # Move the complete AMG315 molecule to the periodic image nearest
        # its reference-pocket atoms.
        pocket_center = xyz[pocket_heavy].mean(axis=0)
        ligand_center = xyz[ligand_heavy].mean(axis=0)

        shift = -box * np.round(
            (ligand_center - pocket_center) / box
        )
        xyz[ligand_heavy] += shift

        # Align the complete frame to the replica-specific production
        # start using the matched 265-CA stable receptor core.
        rotation, mobile_center, target_center = kabsch(
            xyz[stable_core_ca],
            ref_core
        )

        aligned = (
            (xyz - mobile_center) @ rotation +
            target_center
        )

        core_rmsd = rmsd(
            aligned[stable_core_ca],
            ref_core
        )

        tm_rmsd = rmsd(
            aligned[fixed_tm_core],
            ref_tm
        )

        ca_rmsd = rmsd(
            aligned[protein_ca],
            ref_ca
        )

        protein_rmsd = rmsd(
            aligned[protein_heavy],
            ref_protein
        )

        ligand_pose_rmsd = rmsd(
            aligned[ligand_heavy],
            ref_ligand
        )

        ligand_centroid_shift = np.linalg.norm(
            aligned[ligand_heavy].mean(axis=0) -
            ref_ligand_center
        )

        # Internal AMG315 RMSD after ligand-only alignment.
        ligand_rotation, ligand_mobile_center, ligand_target_center = kabsch(
            xyz[ligand_heavy],
            ref_ligand
        )

        internally_aligned = (
            (xyz[ligand_heavy] - ligand_mobile_center) @
            ligand_rotation +
            ligand_target_center
        )

        ligand_internal_rmsd = rmsd(
            internally_aligned,
            ref_ligand
        )

        # Minimum-image AMG315-CB1 distances.
        delta = (
            xyz[ligand_heavy][:, None, :] -
            xyz[protein_heavy][None, :, :]
        )
        delta -= box * np.round(delta / box)
        distances = np.sqrt(np.sum(delta * delta, axis=2))

        minimum_contact = distances.min()

        contacted_columns = np.where(
            np.any(distances <= 0.4, axis=0)
        )[0]

        residues_this_frame = set()
        for column in contacted_columns:
            atom = atoms[protein_heavy[column]]
            residues_this_frame.add(
                (atom.residue.name, atom.residue.resSeq)
            )

        for residue in residues_this_frame:
            contact_counts[residue] += 1

        records.append({
            "replica": replica,
            "frame": frame + 1,
            "time_ps": (frame + 1) * 10.0,
            "stable_core_CA_RMSD_A": core_rmsd * 10.0,
            "TM_core_CA_RMSD_A": tm_rmsd * 10.0,
            "all_protein_CA_RMSD_A": ca_rmsd * 10.0,
            "protein_heavy_RMSD_A": protein_rmsd * 10.0,
            "AMG315_pose_RMSD_A": ligand_pose_rmsd * 10.0,
            "AMG315_centroid_displacement_A":
                ligand_centroid_shift * 10.0,
            "AMG315_internal_RMSD_A":
                ligand_internal_rmsd * 10.0,
            "minimum_AMG315_CB1_contact_A":
                minimum_contact * 10.0,
            "box_volume_nm3": float(np.prod(box)),
        })

    table = pd.DataFrame(records)

    metrics_file = OUT / f"AMG315_CB1_80ns_{replica}_frame_metrics.tsv"
    table.to_csv(metrics_file, sep="\t", index=False)

    contacts = pd.DataFrame([
        {
            "replica": replica,
            "residue_name": name,
            "residue_number": number,
            "frames_contacted": count,
            "occupancy_percent":
                100.0 * count / trajectory.n_frames,
        }
        for (name, number), count in contact_counts.items()
    ])

    contacts = contacts.sort_values(
        ["occupancy_percent", "residue_number"],
        ascending=[False, True]
    )

    contacts_file = OUT / f"AMG315_CB1_80ns_{replica}_contact_occupancy.tsv"
    contacts.to_csv(contacts_file, sep="\t", index=False)

    retained = 100.0 * np.mean(
        table.minimum_AMG315_CB1_contact_A <= 4.0
    )
    lost = int(np.sum(
        table.minimum_AMG315_CB1_contact_A > 5.0
    ))

    last10 = table[table.time_ps > 70000]

    summary_rows.append({
        "replica": replica,
        "mean_stable_core_CA_RMSD_A":
            table.stable_core_CA_RMSD_A.mean(),
        "mean_TM_core_CA_RMSD_A":
            table.TM_core_CA_RMSD_A.mean(),
        "mean_all_CA_RMSD_A":
            table.all_protein_CA_RMSD_A.mean(),
        "mean_protein_heavy_RMSD_A":
            table.protein_heavy_RMSD_A.mean(),
        "mean_AMG315_pose_RMSD_A":
            table.AMG315_pose_RMSD_A.mean(),
        "mean_AMG315_centroid_displacement_A":
            table.AMG315_centroid_displacement_A.mean(),
        "mean_AMG315_internal_RMSD_A":
            table.AMG315_internal_RMSD_A.mean(),
        "mean_minimum_contact_A":
            table.minimum_AMG315_CB1_contact_A.mean(),
        "pocket_retention_percent": retained,
        "frames_contact_gt5A": lost,
        "last10ns_mean_core_RMSD_A":
            last10.stable_core_CA_RMSD_A.mean(),
        "last10ns_mean_AMG315_pose_RMSD_A":
            last10.AMG315_pose_RMSD_A.mean(),
        "last10ns_mean_AMG315_centroid_A":
            last10.AMG315_centroid_displacement_A.mean(),
        "last10ns_core_slope_A_per_ns":
            slope_per_ns(table, "stable_core_CA_RMSD_A"),
        "last10ns_AMG315_pose_slope_A_per_ns":
            slope_per_ns(table, "AMG315_pose_RMSD_A"),
        "last10ns_AMG315_centroid_slope_A_per_ns":
            slope_per_ns(
                table,
                "AMG315_centroid_displacement_A"
            ),
    })

    print("Frames:", len(table))
    print("Reference-pocket heavy atoms:", len(pocket_heavy))
    print(
        "Stable-core CA RMSD:",
        f"mean={table.stable_core_CA_RMSD_A.mean():.3f} A,",
        f"final={table.stable_core_CA_RMSD_A.iloc[-1]:.3f} A"
    )
    print(
        "TM-core CA RMSD:",
        f"mean={table.TM_core_CA_RMSD_A.mean():.3f} A,",
        f"final={table.TM_core_CA_RMSD_A.iloc[-1]:.3f} A"
    )
    print(
        "AMG315 pose RMSD:",
        f"mean={table.AMG315_pose_RMSD_A.mean():.3f} A,",
        f"final={table.AMG315_pose_RMSD_A.iloc[-1]:.3f} A"
    )
    print(
        "AMG315 centroid displacement:",
        f"mean={table.AMG315_centroid_displacement_A.mean():.3f} A"
    )
    print(
        "AMG315 internal RMSD:",
        f"mean={table.AMG315_internal_RMSD_A.mean():.3f} A"
    )
    print(
        "Minimum AMG315-CB1 contact:",
        f"mean={table.minimum_AMG315_CB1_contact_A.mean():.3f} A"
    )
    print("Pocket retention <=4 A:", f"{retained:.1f}%")
    print("Frames with minimum contact >5 A:", lost)

    print("\n1-NS BLOCK MEANS")
    print("ns\tcore_A\tTM_A\tAMG315_pose_A\tcentroid_A\tcontact_A")

    for block in range(80):
        lo = block * 1000
        hi = (block + 1) * 1000
        subset = table[
            (table.time_ps > lo) &
            (table.time_ps <= hi)
        ]

        print(
            f"{block}-{block+1}\t"
            f"{subset.stable_core_CA_RMSD_A.mean():.3f}\t"
            f"{subset.TM_core_CA_RMSD_A.mean():.3f}\t"
            f"{subset.AMG315_pose_RMSD_A.mean():.3f}\t"
            f"{subset.AMG315_centroid_displacement_A.mean():.3f}\t"
            f"{subset.minimum_AMG315_CB1_contact_A.mean():.3f}"
        )

    print("\nTOP 15 CONTACT OCCUPANCIES")
    for row in contacts.head(15).itertuples():
        print(
            f"{row.residue_name:3s} "
            f"{row.residue_number:4d}: "
            f"{row.occupancy_percent:6.1f}%"
        )

summary = pd.DataFrame(summary_rows)
summary_file = OUT / "AMG315_CB1_80ns_three_replica_summary.tsv"
summary.to_csv(summary_file, sep="\t", index=False)

print("\n" + "=" * 78)
print("THREE-REPLICA SUMMARY")
print("=" * 78)
print(summary.to_string(index=False))
print("\nSaved:", summary_file)
print("AMG315 THREE-REPLICA 80-NS ANALYSIS: COMPLETE")
