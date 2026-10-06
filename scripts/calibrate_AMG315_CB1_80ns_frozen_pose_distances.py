from pathlib import Path
from collections import defaultdict

import mdtraj as md
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster


BASE = Path("systems/AMG315_CB1")
OUT = Path("results/AMG315_CB1_frozen_pose_distance_calibration_80ns")
OUT.mkdir(parents=True, exist_ok=False)

FRAME_INTERVAL_PS = 10.0
PCA_VARIANCE_TARGET = 0.90
CLUSTER_RANGE = range(2, 7)

CONFIG = {
    "R1": {
        "reference":
            BASE / "equil/AMG315_CB1_NPT_unrestrained_final.pdb",
        "segments": [
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
        "reference":
            BASE / "replicas/R2/stabilization/"
            "AMG315_R2_unrestrained_500ps_final.pdb",
        "segments": [
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
        "reference":
            BASE / "replicas/R3/stabilization/"
            "AMG315_R3_unrestrained_500ps_final.pdb",
        "segments": [
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

PROTEIN_NAMES = {
    "ALA", "ARG", "ASH", "ASN", "ASP", "CYS", "CYX",
    "GLH", "GLN", "GLU", "GLY", "HID", "HIE", "HIP",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

MODELED_RANGES = (
    (142, 147),
    (254, 265),
    (314, 334),
)

CONTACT_RESIDUES = (
    ("HIS", 178),
    ("THR", 197),
    ("TRP", 279),
    ("MET", 363),
    ("SER", 383),
)


def modeled(resseq):
    return any(
        lo <= resseq <= hi
        for lo, hi in MODELED_RANGES
    )


def kabsch(mobile, target):
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)

    covariance = (
        (mobile - mobile_center).T @
        (target - target_center)
    )

    u, _, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)

    rotation = u @ correction @ vt

    return rotation, mobile_center, target_center


def unwrap_ligand(xyz, ligand, box):
    anchor = xyz[ligand[0]].copy()
    delta = xyz[ligand] - anchor
    delta -= box * np.round(delta / box)
    xyz[ligand] = anchor + delta


def dihedral_angle(xyz, indices):
    p0, p1, p2, p3 = xyz[np.asarray(indices, dtype=int)]

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    norm = np.linalg.norm(b1)

    if norm == 0:
        raise RuntimeError("Zero-length central dihedral bond")

    b1 = b1 / norm

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.arctan2(y, x)


def find_residue(topology, name, resseq):
    hits = [
        residue for residue in topology.residues
        if residue.name == name
        and residue.resSeq == resseq
    ]

    if len(hits) != 1:
        raise RuntimeError(
            f"Expected one {name} {resseq}; found {len(hits)}"
        )

    return hits[0]


def find_residue_dihedral(index_array, topology, target):
    matches = []

    for indices in index_array:
        residue_indices = {
            topology.atom(int(index)).residue.index
            for index in indices
        }

        if residue_indices == {target.index}:
            matches.append(
                tuple(int(index) for index in indices)
            )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one dihedral for {target}; "
            f"found {len(matches)}"
        )

    return matches[0]


def ligand_torsions(topology, ligand_heavy):
    ligand_set = set(ligand_heavy)
    adjacency = defaultdict(set)

    for atom1, atom2 in topology.bonds:
        i = atom1.index
        j = atom2.index

        if i in ligand_set and j in ligand_set:
            adjacency[i].add(j)
            adjacency[j].add(i)

    edges = sorted({
        tuple(sorted((i, j)))
        for i in ligand_set
        for j in adjacency[i]
        if i < j
    })

    torsions = []

    for j, k in edges:
        left = sorted(adjacency[j] - {k})
        right = sorted(adjacency[k] - {j})

        if left and right:
            torsions.append(
                (left[0], j, k, right[0])
            )

    return edges, torsions


def calinski_harabasz(matrix, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)

    n_samples = matrix.shape[0]
    n_clusters = len(unique)

    overall = matrix.mean(axis=0)
    between = 0.0
    within = 0.0

    for label in unique:
        group = matrix[labels == label]
        center = group.mean(axis=0)

        between += (
            len(group) *
            np.sum((center - overall) ** 2)
        )

        within += np.sum((group - center) ** 2)

    if within <= 0:
        return np.inf

    return (
        between / (n_clusters - 1)
    ) / (
        within / (n_samples - n_clusters)
    )


def circular_mean_degrees(values):
    radians = np.deg2rad(np.asarray(values))
    mean = np.arctan2(
        np.mean(np.sin(radians)),
        np.mean(np.cos(radians)),
    )
    return np.rad2deg(mean)


print("=" * 78)
print("AMG315-CB1 THREE-REPLICA 20-NS POSE-STATE ANALYSIS")
print("=" * 78)

r1_reference = md.load(str(CONFIG["R1"]["reference"]))
r1_topology = r1_reference.topology
r1_atoms = list(r1_topology.atoms)
r1_xyz = r1_reference.xyz[0].copy()

stable_core_ca = np.array([
    atom.index for atom in r1_atoms
    if atom.residue.name in PROTEIN_NAMES
    and atom.name == "CA"
    and not modeled(atom.residue.resSeq)
], dtype=int)

ligand_heavy = np.array([
    atom.index for atom in r1_atoms
    if atom.residue.name == "UNK"
    and atom.element is not None
    and atom.element.symbol != "H"
], dtype=int)

protein_heavy = np.array([
    atom.index for atom in r1_atoms
    if atom.residue.name in PROTEIN_NAMES
    and atom.element is not None
    and atom.element.symbol != "H"
], dtype=int)

heavy_edges, torsions = ligand_torsions(
    r1_topology,
    ligand_heavy
)

chi1_indices, _ = md.compute_chi1(r1_reference)
chi2_indices, _ = md.compute_chi2(r1_reference)

phe200_chi1 = find_residue_dihedral(
    chi1_indices,
    r1_topology,
    find_residue(r1_topology, "PHE", 200),
)

trp356_chi2 = find_residue_dihedral(
    chi2_indices,
    r1_topology,
    find_residue(r1_topology, "TRP", 356),
)

contact_indices = {}

for name, resseq in CONTACT_RESIDUES:
    target = find_residue(r1_topology, name, resseq)

    contact_indices[(name, resseq)] = np.array([
        atom.index for atom in target.atoms
        if atom.element is not None
        and atom.element.symbol != "H"
    ], dtype=int)

assert len(stable_core_ca) == 265
assert len(ligand_heavy) == 27
assert len(protein_heavy) == 2426
assert len(heavy_edges) == 26
assert len(torsions) == 21
assert len(contact_indices) == 5

r1_box = r1_reference.unitcell_lengths[0].copy()
unwrap_ligand(r1_xyz, ligand_heavy, r1_box)

reference_pocket = protein_heavy[
    np.any(
        np.linalg.norm(
            r1_xyz[ligand_heavy][:, None, :] -
            r1_xyz[protein_heavy][None, :, :],
            axis=2,
        ) <= 0.5,
        axis=0,
    )
]

assert len(reference_pocket) > 0

reference_pocket_center = r1_xyz[
    reference_pocket
].mean(axis=0)

reference_ligand_center = r1_xyz[
    ligand_heavy
].mean(axis=0)

reference_ligand_coordinates = (
    r1_xyz[ligand_heavy] -
    reference_ligand_center
)

reference_core = r1_xyz[stable_core_ca]

feature_rows = []
descriptive_rows = []
frame_rows = []

feature_names = []

for atom_index in ligand_heavy:
    atom = r1_atoms[atom_index]

    for axis in ("x", "y", "z"):
        feature_names.append(
            f"ligand_{atom.name}_{atom_index}_{axis}_nm"
        )

for number in range(1, len(torsions) + 1):
    feature_names.extend([
        f"torsion_{number:02d}_sin",
        f"torsion_{number:02d}_cos",
    ])

for name, resseq in CONTACT_RESIDUES:
    feature_names.append(
        f"contact_{name}_{resseq}_nm"
    )

feature_names.extend([
    "PHE200_chi1_sin",
    "PHE200_chi1_cos",
    "TRP356_chi2_sin",
    "TRP356_chi2_cos",
])

assert len(feature_names) == 132

global_index = 0

for replica, config in CONFIG.items():
    print()
    print("=" * 78)
    print(replica)
    print("=" * 78)

    reference = md.load(str(config["reference"]))

    segments = [
        md.load(
            str(segment),
            top=str(config["reference"]),
        )
        for segment in config["segments"]
    ]

    assert [segment.n_frames for segment in segments] == [
        500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000
    ]

    trajectory = md.join(
        segments,
        check_topology=True,
    )

    assert trajectory.n_frames == 8000
    assert trajectory.n_atoms == 42295
    assert trajectory.unitcell_lengths is not None

    atoms = list(reference.topology.atoms)

    replica_ligand = np.array([
        atom.index for atom in atoms
        if atom.residue.name == "UNK"
        and atom.element is not None
        and atom.element.symbol != "H"
    ], dtype=int)

    replica_core = np.array([
        atom.index for atom in atoms
        if atom.residue.name in PROTEIN_NAMES
        and atom.name == "CA"
        and not modeled(atom.residue.resSeq)
    ], dtype=int)

    assert np.array_equal(replica_ligand, ligand_heavy)
    assert np.array_equal(replica_core, stable_core_ca)

    print("Frames:", trajectory.n_frames)
    print("Extracting 132 pose-state features...")

    for frame in range(trajectory.n_frames):
        xyz = trajectory.xyz[frame].copy()
        box = trajectory.unitcell_lengths[frame].copy()

        if not np.isfinite(xyz).all():
            raise RuntimeError(
                f"{replica} frame {frame}: non-finite coordinates"
            )

        if not np.isfinite(box).all() or np.any(box <= 0):
            raise RuntimeError(
                f"{replica} frame {frame}: invalid box"
            )

        unwrap_ligand(xyz, ligand_heavy, box)

        pocket_center = xyz[reference_pocket].mean(axis=0)
        ligand_center = xyz[ligand_heavy].mean(axis=0)

        shift = -box * np.round(
            (ligand_center - pocket_center) / box
        )

        xyz[ligand_heavy] += shift

        rotation, mobile_center, target_center = kabsch(
            xyz[stable_core_ca],
            reference_core,
        )

        aligned_ligand = (
            (xyz[ligand_heavy] - mobile_center) @
            rotation +
            target_center
        )

        # Preserve both ligand orientation and translational movement.
        # Coordinates are expressed relative to the single fixed R1
        # reference-ligand centroid after stable-core alignment.
        pose_coordinates = (
            aligned_ligand -
            reference_ligand_center
        )

        coordinate_features = pose_coordinates.reshape(-1)

        torsion_angles = np.array([
            dihedral_angle(xyz, indices)
            for indices in torsions
        ])

        torsion_features = np.column_stack([
            np.sin(torsion_angles),
            np.cos(torsion_angles),
        ]).reshape(-1)

        contact_distances = []

        for residue_key in CONTACT_RESIDUES:
            indices = contact_indices[residue_key]

            delta = (
                xyz[ligand_heavy][:, None, :] -
                xyz[indices][None, :, :]
            )

            delta -= box * np.round(delta / box)

            contact_distances.append(
                np.linalg.norm(delta, axis=2).min()
            )

        contact_distances = np.asarray(
            contact_distances,
            dtype=float,
        )

        phe_angle = dihedral_angle(
            xyz,
            phe200_chi1,
        )

        trp_angle = dihedral_angle(
            xyz,
            trp356_chi2,
        )

        switch_features = np.array([
            np.sin(phe_angle),
            np.cos(phe_angle),
            np.sin(trp_angle),
            np.cos(trp_angle),
        ])

        features = np.concatenate([
            coordinate_features,
            torsion_features,
            contact_distances,
            switch_features,
        ])

        if features.shape != (132,):
            raise RuntimeError(
                f"Unexpected feature shape: {features.shape}"
            )

        if not np.isfinite(features).all():
            raise RuntimeError(
                f"{replica} frame {frame}: non-finite features"
            )

        feature_rows.append(features)

        descriptive_rows.append({
            "global_frame": global_index,
            "replica": replica,
            "replica_frame": frame + 1,
            "time_ps": (frame + 1) * FRAME_INTERVAL_PS,
            "HIS178_contact_A":
                contact_distances[0] * 10.0,
            "THR197_contact_A":
                contact_distances[1] * 10.0,
            "TRP279_contact_A":
                contact_distances[2] * 10.0,
            "MET363_contact_A":
                contact_distances[3] * 10.0,
            "SER383_contact_A":
                contact_distances[4] * 10.0,
            "PHE200_chi1_deg":
                np.rad2deg(phe_angle),
            "TRP356_chi2_deg":
                np.rad2deg(trp_angle),
        })

        frame_rows.append({
            "global_frame": global_index,
            "replica": replica,
            "replica_frame": frame + 1,
            "time_ps": (frame + 1) * FRAME_INTERVAL_PS,
        })

        global_index += 1

    print(f"{replica} feature extraction: PASS")

feature_matrix = np.asarray(feature_rows, dtype=float)
descriptive = pd.DataFrame(descriptive_rows)
frames = pd.DataFrame(frame_rows)

assert feature_matrix.shape == (24000, 132)
assert len(descriptive) == 24000
assert len(frames) == 24000

FROZEN = Path("results/AMG315_CB1_pose_states_20ns")
PRIOR_PROJECTION = Path(
    "results/AMG315_CB1_frozen_pose_distance_calibration_70ns"
)

frozen_frames = pd.read_csv(
    FROZEN / "AMG315_CB1_20ns_pose_state_frames.tsv",
    sep="\t",
)

prior_projection_frames = pd.read_csv(
    PRIOR_PROJECTION /
    "AMG315_CB1_60to70ns_frozen_pose_state_frames.tsv",
    sep="\t",
)

assert len(prior_projection_frames) == 3000

frozen_metadata = pd.read_csv(
    FROZEN / "AMG315_CB1_20ns_pose_feature_metadata.tsv",
    sep="\t",
)

frozen_variance = pd.read_csv(
    FROZEN / "AMG315_CB1_20ns_pose_PCA_variance.tsv",
    sep="\t",
)

assert len(frozen_frames) == 6000
assert len(frozen_metadata) == 132
assert frozen_frames.state.value_counts().sort_index().to_dict() == {
    1: 4115,
    2: 1885,
}

training_mask = frames.time_ps <= 20000.0
projection_mask = frames.time_ps > 70000.0

training_features = feature_matrix[training_mask]
projection_features = feature_matrix[projection_mask]

assert training_features.shape == (6000, 132)
assert projection_features.shape == (3000, 132)

training_frames = frames.loc[training_mask].reset_index(drop=True)
projection_frames = frames.loc[projection_mask].reset_index(drop=True)

training_descriptive = (
    descriptive.loc[training_mask].reset_index(drop=True)
)
projection_descriptive = (
    descriptive.loc[projection_mask].reset_index(drop=True)
)

expected_training_identity = frozen_frames[
    ["replica", "replica_frame", "time_ps"]
].reset_index(drop=True)

observed_training_identity = training_frames[
    ["replica", "replica_frame", "time_ps"]
].reset_index(drop=True)

pd.testing.assert_frame_equal(
    observed_training_identity,
    expected_training_identity,
    check_dtype=False,
    check_exact=True,
)

means = training_features.mean(axis=0)
standard_deviations = training_features.std(axis=0, ddof=0)
variable = standard_deviations > 1.0e-12

assert variable.sum() == 132

stored_means = frozen_metadata["mean"].to_numpy(dtype=float)
stored_sd = frozen_metadata["standard_deviation"].to_numpy(dtype=float)
stored_variable = (
    frozen_metadata["variable_for_PCA"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
    .to_numpy(dtype=bool)
)

if not np.allclose(means, stored_means, rtol=0.0, atol=5.0e-13):
    raise RuntimeError(
        "Reconstructed feature means do not match the frozen model"
    )

if not np.allclose(
    standard_deviations,
    stored_sd,
    rtol=0.0,
    atol=5.0e-13,
):
    raise RuntimeError(
        "Reconstructed feature standard deviations do not match "
        "the frozen model"
    )

if not np.array_equal(variable, stored_variable):
    raise RuntimeError(
        "Reconstructed variable-feature mask does not match "
        "the frozen model"
    )

standardized_training = (
    training_features[:, variable] -
    means[variable]
) / standard_deviations[variable]

standardized_projection = (
    projection_features[:, variable] -
    means[variable]
) / standard_deviations[variable]

u, singular_values, vt = np.linalg.svd(
    standardized_training,
    full_matrices=False,
)

variance = singular_values ** 2
variance_ratio = variance / variance.sum()
cumulative_variance = np.cumsum(variance_ratio)

n_components = int(
    np.searchsorted(
        cumulative_variance,
        PCA_VARIANCE_TARGET,
    ) + 1
)

assert n_components == 35

if not np.allclose(
    variance_ratio,
    frozen_variance["variance_fraction"].to_numpy(dtype=float),
    rtol=0.0,
    atol=5.0e-12,
):
    raise RuntimeError(
        "Reconstructed PCA variance fractions do not match"
    )

training_scores = (
    u[:, :n_components] *
    singular_values[:n_components]
)

# Match arbitrary SVD signs to the first five stored components.
for component in range(5):
    stored = frozen_frames[
        f"PC{component + 1}"
    ].to_numpy(dtype=float)

    direct_error = np.max(
        np.abs(training_scores[:, component] - stored)
    )

    flipped_error = np.max(
        np.abs(-training_scores[:, component] - stored)
    )

    if flipped_error < direct_error:
        training_scores[:, component] *= -1.0
        vt[component, :] *= -1.0
        direct_error = flipped_error

    if direct_error > 5.0e-8:
        raise RuntimeError(
            f"PC{component + 1} reconstruction failed: "
            f"maximum difference={direct_error}"
        )

projection_scores = (
    standardized_projection @
    vt[:n_components, :].T
)

assert np.isfinite(training_scores).all()
assert np.isfinite(projection_scores).all()

# Reconstruct the original Ward model and verify every historical label.
ward = linkage(
    training_scores,
    method="ward",
    metric="euclidean",
    optimal_ordering=False,
)

raw_labels = fcluster(
    ward,
    t=2,
    criterion="maxclust",
)

occupancy_order = (
    pd.Series(raw_labels)
    .value_counts()
    .sort_values(ascending=False)
    .index
    .tolist()
)

state_map = {
    old: new
    for new, old in enumerate(
        occupancy_order,
        start=1,
    )
}

reconstructed_labels = np.array([
    state_map[label]
    for label in raw_labels
], dtype=int)

frozen_labels = frozen_frames.state.to_numpy(dtype=int)

if not np.array_equal(reconstructed_labels, frozen_labels):
    disagreements = int(np.sum(
        reconstructed_labels != frozen_labels
    ))
    raise RuntimeError(
        f"Frozen state-label reconstruction failed: "
        f"{disagreements} historical frames disagree"
    )

state_centroids = {}

for state in (1, 2):
    subset = training_scores[
        reconstructed_labels == state
    ]

    state_centroids[state] = subset.mean(axis=0)

centroid_matrix = np.vstack([
    state_centroids[1],
    state_centroids[2],
])

projection_distances = np.linalg.norm(
    projection_scores[:, None, :] -
    centroid_matrix[None, :, :],
    axis=2,
)

projected_labels = (
    np.argmin(projection_distances, axis=1) + 1
)

projection_output = projection_frames.copy()
projection_output["state"] = projected_labels
projection_output["distance_to_state_1"] = projection_distances[:, 0]
projection_output["distance_to_state_2"] = projection_distances[:, 1]
projection_output["distance_to_assigned_state"] = np.min(
    projection_distances,
    axis=1,
)

for component in range(5):
    projection_output[
        f"PC{component + 1}"
    ] = projection_scores[:, component]

projection_output = pd.concat(
    [
        projection_output,
        projection_descriptive.drop(
            columns=[
                "global_frame",
                "replica",
                "replica_frame",
                "time_ps",
            ]
        ),
    ],
    axis=1,
)

# The complete loaded trajectory already carries its physical
# 0-to-80-ns replica-frame and time identities. Therefore, the
# new projection requires no inherited frame or time offset.

replica_global_offsets = {
    "R1": 0,
    "R2": 8000,
    "R3": 16000,
}

projection_output["global_frame"] = (
    projection_output["replica"].map(
        replica_global_offsets
    ).to_numpy(dtype=int) +
    projection_output["replica_frame"].to_numpy(dtype=int) -
    1
)

for replica in ("R1", "R2", "R3"):
    identity_subset = (
        projection_output.loc[
            projection_output.replica == replica
        ]
        .sort_values("replica_frame")
        .reset_index(drop=True)
    )

    assert len(identity_subset) == 1000
    assert np.array_equal(
        identity_subset.replica_frame.to_numpy(dtype=int),
        np.arange(7001, 8001, dtype=int),
    )
    assert np.allclose(
        identity_subset.time_ps.to_numpy(dtype=float),
        np.arange(70010.0, 80000.0 + 0.1, 10.0),
        rtol=0.0,
        atol=1.0e-9,
    )

projection_output.to_csv(
    OUT / "AMG315_CB1_70to80ns_frozen_pose_state_frames.tsv",
    sep="\t",
    index=False,
)

occupancy_rows = []

for replica in ("R1", "R2", "R3"):
    subset = projection_output[
        projection_output.replica == replica
    ]

    assert len(subset) == 1000

    for state in (1, 2):
        count = int(np.sum(subset.state == state))

        occupancy_rows.append({
            "replica": replica,
            "state": state,
            "frames": count,
            "occupancy_percent":
                100.0 * count / len(subset),
            "first5ns_percent":
                100.0 * np.mean(
                    subset[
                        subset.time_ps <= 75000.0
                    ].state == state
                ),
            "last5ns_percent":
                100.0 * np.mean(
                    subset[
                        subset.time_ps > 75000.0
                    ].state == state
                ),
        })

occupancy = pd.DataFrame(occupancy_rows)

occupancy.to_csv(
    OUT / "AMG315_CB1_70to80ns_frozen_pose_state_occupancy.tsv",
    sep="\t",
    index=False,
)

transition_rows = []

for replica in ("R1", "R2", "R3"):
    previous_state = int(
        prior_projection_frames.loc[
            prior_projection_frames.replica == replica,
            "state",
        ].iloc[-1]
    )

    new_states = (
        projection_output.loc[
            projection_output.replica == replica,
            "state",
        ]
        .to_numpy(dtype=int)
    )

    complete_states = np.concatenate([
        np.array([previous_state], dtype=int),
        new_states,
    ])

    for from_state in (1, 2):
        for to_state in (1, 2):
            transition_rows.append({
                "replica": replica,
                "from_state": from_state,
                "to_state": to_state,
                "transition_count": int(np.sum(
                    (complete_states[:-1] == from_state) &
                    (complete_states[1:] == to_state)
                )),
            })

transitions = pd.DataFrame(transition_rows)

transitions.to_csv(
    OUT / "AMG315_CB1_70to80ns_frozen_pose_state_transitions.tsv",
    sep="\t",
    index=False,
)

model_rows = []

for state in (1, 2):
    model_rows.append({
        "state": state,
        "training_frames": int(np.sum(
            reconstructed_labels == state
        )),
        "centroid_dimension": n_components,
        "mean_training_distance_to_centroid":
            np.linalg.norm(
                training_scores[
                    reconstructed_labels == state
                ] - state_centroids[state],
                axis=1,
            ).mean(),
    })

model_summary = pd.DataFrame(model_rows)

model_summary.to_csv(
    OUT / "AMG315_CB1_frozen_pose_model_summary.tsv",
    sep="\t",
    index=False,
)

print()
print("=" * 78)
print("FROZEN 20-NS MODEL RECONSTRUCTION")
print("=" * 78)
print("Training frames:", len(training_scores))
print("Raw features:", training_features.shape[1])
print("Variable features:", int(variable.sum()))
print("PCA components:", n_components)
print("Historical labels reproduced exactly: PASS")
print("Frozen state counts:", {
    1: int(np.sum(reconstructed_labels == 1)),
    2: int(np.sum(reconstructed_labels == 2)),
})

print()
print("=" * 78)
print("70-TO-80-NS FROZEN-STATE OCCUPANCY")
print("=" * 78)
print(occupancy.to_string(index=False))

print()
print("=" * 78)
print("70-TO-80-NS TRANSITIONS")
print("=" * 78)
print(transitions.to_string(index=False))

print()
print("No new-frame scaling, PCA, clustering, or centroid fitting: PASS")
print("Saved output directory:", OUT)
print(
    "AMG315 70-TO-80-NS FROZEN POSE-STATE "
    "PROJECTION: COMPLETE"
)


# ----------------------------------------------------------------------
# Frozen-state distance calibration
# ----------------------------------------------------------------------

training_distance_rows = []
threshold_rows = []
training_distance_lookup = {}

for state in (1, 2):
    state_mask = reconstructed_labels == state

    distances = np.linalg.norm(
        training_scores[state_mask] -
        state_centroids[state],
        axis=1,
    )

    distances = np.asarray(distances, dtype=float)
    training_distance_lookup[state] = np.sort(distances)

    q50, q90, q95, q99 = np.quantile(
        distances,
        [0.50, 0.90, 0.95, 0.99],
    )

    threshold_rows.append({
        "state": state,
        "training_frames": len(distances),
        "distance_median": q50,
        "distance_p90": q90,
        "distance_p95": q95,
        "distance_p99": q99,
        "distance_maximum": distances.max(),
    })

    indices = np.where(state_mask)[0]

    for index, distance in zip(indices, distances):
        training_distance_rows.append({
            "global_frame": int(
                frozen_frames.iloc[index].global_frame
            ),
            "replica": frozen_frames.iloc[index].replica,
            "replica_frame": int(
                frozen_frames.iloc[index].replica_frame
            ),
            "time_ps": float(
                frozen_frames.iloc[index].time_ps
            ),
            "state": state,
            "distance_to_own_state_centroid": distance,
        })

thresholds = pd.DataFrame(threshold_rows)
training_distances = pd.DataFrame(training_distance_rows)

thresholds.to_csv(
    OUT / "AMG315_CB1_frozen_state_distance_thresholds.tsv",
    sep="\t",
    index=False,
)

training_distances.to_csv(
    OUT / "AMG315_CB1_frozen_training_frame_distances.tsv",
    sep="\t",
    index=False,
)

calibrated = projection_output.copy()

percentiles = []
categories = []
distance_ratios = []
distance_margins = []

for row in calibrated.itertuples():
    state = int(row.state)
    distance = float(row.distance_to_assigned_state)
    distribution = training_distance_lookup[state]

    percentile = (
        100.0 *
        np.searchsorted(
            distribution,
            distance,
            side="right",
        ) /
        len(distribution)
    )

    state_thresholds = thresholds[
        thresholds.state == state
    ].iloc[0]

    if distance <= state_thresholds.distance_p95:
        category = "IN_BASIN_95"
    elif distance <= state_thresholds.distance_p99:
        category = "BASIN_EDGE_95_TO_99"
    else:
        category = "OUTSIDE_TRAINING_99"

    d1 = float(row.distance_to_state_1)
    d2 = float(row.distance_to_state_2)

    percentiles.append(percentile)
    categories.append(category)
    distance_ratios.append(
        max(d1, d2) / max(min(d1, d2), 1.0e-15)
    )
    distance_margins.append(abs(d1 - d2))

calibrated["assigned_state_training_percentile"] = percentiles
calibrated["distance_calibration_category"] = categories
calibrated["centroid_distance_ratio"] = distance_ratios
calibrated["centroid_distance_margin"] = distance_margins

calibrated.to_csv(
    OUT / "AMG315_CB1_70to80ns_calibrated_pose_state_frames.tsv",
    sep="\t",
    index=False,
)

summary_rows = []

for replica in ("R1", "R2", "R3"):
    replica_subset = calibrated[
        calibrated.replica == replica
    ]

    for interval, interval_subset in (
        ("70_to_80ns", replica_subset),
        (
            "70_to_75ns",
            replica_subset[
                replica_subset.time_ps <= 75000.0
            ],
        ),
        (
            "75_to_80ns",
            replica_subset[
                replica_subset.time_ps > 75000.0
            ],
        ),
    ):
        for state in (1, 2):
            state_subset = interval_subset[
                interval_subset.state == state
            ]

            for category in (
                "IN_BASIN_95",
                "BASIN_EDGE_95_TO_99",
                "OUTSIDE_TRAINING_99",
            ):
                count = int(np.sum(
                    state_subset.distance_calibration_category ==
                    category
                ))

                summary_rows.append({
                    "replica": replica,
                    "interval": interval,
                    "state": state,
                    "calibration_category": category,
                    "frames": count,
                    "percent_of_interval":
                        100.0 * count / len(interval_subset),
                    "percent_within_assigned_state":
                        (
                            100.0 * count / len(state_subset)
                            if len(state_subset) > 0
                            else 0.0
                        ),
                })

calibration_summary = pd.DataFrame(summary_rows)

calibration_summary.to_csv(
    OUT / "AMG315_CB1_70to80ns_distance_calibration_summary.tsv",
    sep="\t",
    index=False,
)

compact_rows = []

for replica in ("R1", "R2", "R3"):
    for interval, subset in (
        (
            "70_to_80ns",
            calibrated[calibrated.replica == replica],
        ),
        (
            "70_to_75ns",
            calibrated[
                (calibrated.replica == replica) &
                (calibrated.time_ps <= 75000.0)
            ],
        ),
        (
            "75_to_80ns",
            calibrated[
                (calibrated.replica == replica) &
                (calibrated.time_ps > 75000.0)
            ],
        ),
    ):
        compact_rows.append({
            "replica": replica,
            "interval": interval,
            "frames": len(subset),
            "state1_percent":
                100.0 * np.mean(subset.state == 1),
            "state2_percent":
                100.0 * np.mean(subset.state == 2),
            "in_basin_95_percent":
                100.0 * np.mean(
                    subset.distance_calibration_category ==
                    "IN_BASIN_95"
                ),
            "basin_edge_95_to_99_percent":
                100.0 * np.mean(
                    subset.distance_calibration_category ==
                    "BASIN_EDGE_95_TO_99"
                ),
            "outside_training_99_percent":
                100.0 * np.mean(
                    subset.distance_calibration_category ==
                    "OUTSIDE_TRAINING_99"
                ),
            "median_assigned_state_percentile":
                subset[
                    "assigned_state_training_percentile"
                ].median(),
            "median_centroid_distance_ratio":
                subset["centroid_distance_ratio"].median(),
        })

compact = pd.DataFrame(compact_rows)

compact.to_csv(
    OUT / "AMG315_CB1_70to80ns_distance_calibration_compact.tsv",
    sep="\t",
    index=False,
)

print()
print("=" * 78)
print("FROZEN TRAINING-DISTANCE THRESHOLDS")
print("=" * 78)
print(thresholds.to_string(index=False))

print()
print("=" * 78)
print("PROJECTED-FRAME DISTANCE CALIBRATION")
print("=" * 78)
print(compact.to_string(index=False))

print()
print("Training state counts:", {
    state: len(training_distance_lookup[state])
    for state in (1, 2)
})
print("Calibrated projection frames:", len(calibrated))
print("Frozen-model distance calibration: PASS")
print(
    "AMG315 70-TO-80-NS FROZEN POSE-STATE "
    "DISTANCE CALIBRATION: COMPLETE"
)
