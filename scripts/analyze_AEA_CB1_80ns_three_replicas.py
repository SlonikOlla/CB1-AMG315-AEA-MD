import mdtraj as md
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path("systems/AEA_CB1")
OUT = Path("results/AEA_CB1_80ns_QC")
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "R1": {
        "reference": BASE / "equil/AEA_CB1_NPT_unrestrained_ext1500ps_final.pdb",
        "dcd": BASE / "replicas/R1/AEA_CB1_prod_10ns_R1.dcd",
        "dcd2": BASE / "replicas/R1/AEA_CB1_prod_10to20ns_R1.dcd",
        "dcd3": BASE / "replicas/R1/AEA_CB1_prod_20to30ns_R1.dcd",
        "dcd4": BASE / "replicas/R1/AEA_CB1_prod_30to40ns_R1.dcd",
        "dcd5": BASE / "replicas/R1/AEA_CB1_prod_40to50ns_R1.dcd",
        "dcd6": BASE / "replicas/R1/AEA_CB1_prod_50to60ns_R1.dcd",
        "dcd7": BASE / "replicas/R1/AEA_CB1_prod_60to70ns_R1.dcd",
        "dcd8": BASE / "replicas/R1/AEA_CB1_prod_70to80ns_R1.dcd",
    },
    "R2": {
        "reference": BASE / "replicas/R2/stabilization/AEA_R2_unrestrained_500ps_final.pdb",
        "dcd": BASE / "replicas/R2/AEA_CB1_prod_10ns_R2.dcd",
        "dcd2": BASE / "replicas/R2/AEA_CB1_prod_10to20ns_R2.dcd",
        "dcd3": BASE / "replicas/R2/AEA_CB1_prod_20to30ns_R2.dcd",
        "dcd4": BASE / "replicas/R2/AEA_CB1_prod_30to40ns_R2.dcd",
        "dcd5": BASE / "replicas/R2/AEA_CB1_prod_40to50ns_R2.dcd",
        "dcd6": BASE / "replicas/R2/AEA_CB1_prod_50to60ns_R2.dcd",
        "dcd7": BASE / "replicas/R2/AEA_CB1_prod_60to70ns_R2.dcd",
        "dcd8": BASE / "replicas/R2/AEA_CB1_prod_70to80ns_R2.dcd",
    },
    "R3": {
        "reference": BASE / "replicas/R3/stabilization/AEA_R3_unrestrained_500ps_final.pdb",
        "dcd": BASE / "replicas/R3/AEA_CB1_prod_10ns_R3.dcd",
        "dcd2": BASE / "replicas/R3/AEA_CB1_prod_10to20ns_R3.dcd",
        "dcd3": BASE / "replicas/R3/AEA_CB1_prod_20to30ns_R3.dcd",
        "dcd4": BASE / "replicas/R3/AEA_CB1_prod_30to40ns_R3.dcd",
        "dcd5": BASE / "replicas/R3/AEA_CB1_prod_40to50ns_R3.dcd",
        "dcd6": BASE / "replicas/R3/AEA_CB1_prod_50to60ns_R3.dcd",
        "dcd7": BASE / "replicas/R3/AEA_CB1_prod_60to70ns_R3.dcd",
        "dcd8": BASE / "replicas/R3/AEA_CB1_prod_70to80ns_R3.dcd",
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

r1_phosphorus = np.array([
    atom.index for atom in r1_top.atoms
    if atom.residue.name == "POP"
    and atom.element is not None
    and atom.element.symbol == "P"
], dtype=int)

pz = r1_xyz[r1_phosphorus, 2]
median_p = np.median(pz)
upper_mean = pz[pz >= median_p].mean()
lower_mean = pz[pz < median_p].mean()
membrane_midpoint = 0.5 * (upper_mean + lower_mean)

fixed_tm_core = np.array([
    index for index in r1_protein_ca
    if abs(r1_xyz[index, 2] - membrane_midpoint) <= 1.5
], dtype=int)

assert len(fixed_tm_core) == 147

summary_rows = []

print("=" * 78)
print("AEA-CB1 THREE-REPLICA 50-NS PBC-CORRECTED PRODUCTION ANALYSIS")
print("=" * 78)
print("Fixed R1-defined TM-core CA atoms:", len(fixed_tm_core))
print("Physical trajectory times: 10-80000 ps")

for replica, config in CONFIG.items():
    print("\n" + "=" * 78)
    print(replica)
    print("=" * 78)

    reference = md.load(str(config["reference"]))
    first_segment = md.load(
        str(config["dcd"]),
        top=str(config["reference"])
    )
    second_segment = md.load(
        str(config["dcd2"]),
        top=str(config["reference"])
    )

    third_segment = md.load(
        str(config["dcd3"]),
        top=str(config["reference"])
    )
    fourth_segment = md.load(
        str(config["dcd4"]),
        top=str(config["reference"])
    )
    fifth_segment = md.load(
        str(config["dcd5"]),
        top=str(config["reference"])
    )
    sixth_segment = md.load(
        str(config["dcd6"]),
        top=str(config["reference"])
    )
    seventh_segment = md.load(
        str(config["dcd7"]),
        top=str(config["reference"])
    )
    eighth_segment = md.load(
        str(config["dcd8"]),
        top=str(config["reference"])
    )

    assert first_segment.n_frames == 1000
    assert second_segment.n_frames == 1000
    assert third_segment.n_frames == 1000
    assert fourth_segment.n_frames == 1000
    assert fifth_segment.n_frames == 1000
    assert sixth_segment.n_frames == 1000
    assert seventh_segment.n_frames == 1000
    assert eighth_segment.n_frames == 1000

    trajectory = first_segment.join(second_segment)
    trajectory = trajectory.join(third_segment)
    trajectory = trajectory.join(fourth_segment)
    trajectory = trajectory.join(fifth_segment)
    trajectory = trajectory.join(sixth_segment)
    trajectory = trajectory.join(seventh_segment)
    trajectory = trajectory.join(eighth_segment)

    del first_segment
    del second_segment
    del third_segment
    del fourth_segment
    del fifth_segment
    del sixth_segment
    del seventh_segment
    del eighth_segment

    assert trajectory.n_frames == 8000
    assert trajectory.n_atoms == reference.n_atoms == 42322
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
    assert len(ligand_heavy) == 25

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

        # Reconstruct AEA as one molecule.
        unwrap_ligand(xyz, ligand_heavy, box)

        # Move the complete AEA molecule to the periodic image nearest
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

        # Internal AEA RMSD after ligand-only alignment.
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

        # Minimum-image AEA-CB1 distances.
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
            "AEA_pose_RMSD_A": ligand_pose_rmsd * 10.0,
            "AEA_centroid_displacement_A":
                ligand_centroid_shift * 10.0,
            "AEA_internal_RMSD_A":
                ligand_internal_rmsd * 10.0,
            "minimum_AEA_CB1_contact_A":
                minimum_contact * 10.0,
            "box_volume_nm3": float(np.prod(box)),
        })

    table = pd.DataFrame(records)

    metrics_file = OUT / f"AEA_CB1_80ns_{replica}_frame_metrics.tsv"
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

    contacts_file = OUT / f"AEA_CB1_80ns_{replica}_contact_occupancy.tsv"
    contacts.to_csv(contacts_file, sep="\t", index=False)

    retained = 100.0 * np.mean(
        table.minimum_AEA_CB1_contact_A <= 4.0
    )
    lost = int(np.sum(
        table.minimum_AEA_CB1_contact_A > 5.0
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
        "mean_AEA_pose_RMSD_A":
            table.AEA_pose_RMSD_A.mean(),
        "mean_AEA_centroid_displacement_A":
            table.AEA_centroid_displacement_A.mean(),
        "mean_AEA_internal_RMSD_A":
            table.AEA_internal_RMSD_A.mean(),
        "mean_minimum_contact_A":
            table.minimum_AEA_CB1_contact_A.mean(),
        "pocket_retention_percent": retained,
        "frames_contact_gt5A": lost,
        "last10ns_mean_core_RMSD_A":
            last10.stable_core_CA_RMSD_A.mean(),
        "last10ns_mean_AEA_pose_RMSD_A":
            last10.AEA_pose_RMSD_A.mean(),
        "last10ns_mean_AEA_centroid_A":
            last10.AEA_centroid_displacement_A.mean(),
        "last10ns_core_slope_A_per_ns":
            slope_per_ns(table, "stable_core_CA_RMSD_A"),
        "last10ns_AEA_pose_slope_A_per_ns":
            slope_per_ns(table, "AEA_pose_RMSD_A"),
        "last10ns_AEA_centroid_slope_A_per_ns":
            slope_per_ns(
                table,
                "AEA_centroid_displacement_A"
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
        "AEA pose RMSD:",
        f"mean={table.AEA_pose_RMSD_A.mean():.3f} A,",
        f"final={table.AEA_pose_RMSD_A.iloc[-1]:.3f} A"
    )
    print(
        "AEA centroid displacement:",
        f"mean={table.AEA_centroid_displacement_A.mean():.3f} A"
    )
    print(
        "AEA internal RMSD:",
        f"mean={table.AEA_internal_RMSD_A.mean():.3f} A"
    )
    print(
        "Minimum AEA-CB1 contact:",
        f"mean={table.minimum_AEA_CB1_contact_A.mean():.3f} A"
    )
    print("Pocket retention <=4 A:", f"{retained:.1f}%")
    print("Frames with minimum contact >5 A:", lost)

    print("\n1-NS BLOCK MEANS")
    print("ns\tcore_A\tTM_A\tAEA_pose_A\tcentroid_A\tcontact_A")

    for block in range(30):
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
            f"{subset.AEA_pose_RMSD_A.mean():.3f}\t"
            f"{subset.AEA_centroid_displacement_A.mean():.3f}\t"
            f"{subset.minimum_AEA_CB1_contact_A.mean():.3f}"
        )

    print("\nTOP 15 CONTACT OCCUPANCIES")
    for row in contacts.head(15).itertuples():
        print(
            f"{row.residue_name:3s} "
            f"{row.residue_number:4d}: "
            f"{row.occupancy_percent:6.1f}%"
        )

summary = pd.DataFrame(summary_rows)
summary_file = OUT / "AEA_CB1_80ns_three_replica_summary.tsv"
summary.to_csv(summary_file, sep="\t", index=False)

print("\n" + "=" * 78)
print("THREE-REPLICA SUMMARY")
print("=" * 78)
print(summary.to_string(index=False))
print("\nSaved:", summary_file)
print("AEA THREE-REPLICA 80-NS ANALYSIS: COMPLETE")
