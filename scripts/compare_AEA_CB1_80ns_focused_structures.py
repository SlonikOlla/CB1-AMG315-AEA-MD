from pathlib import Path
from collections import defaultdict
import mdtraj as md
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
BASE = Path("systems/AEA_CB1")
OUT = Path("results/AEA_CB1_80ns_focused_structural_comparison")
OUT.mkdir(parents=True, exist_ok=False)
FRAME_INTERVAL_PS = 10.0
PCA_VARIANCE_TARGET = 0.9
CLUSTER_RANGE = range(2, 9)
CONFIG = {
    "R1": {
        "reference": BASE / "equil/AEA_CB1_NPT_unrestrained_ext1500ps_final.pdb",
        "segments": [
            BASE / "replicas/R1/AEA_CB1_prod_10ns_R1.dcd",
            BASE / "replicas/R1/AEA_CB1_prod_10to20ns_R1.dcd",
            BASE / "replicas/R1/AEA_CB1_prod_20to30ns_R1.dcd",
            BASE / "replicas/R1/AEA_CB1_prod_30to40ns_R1.dcd",
            BASE / "replicas/R1/AEA_CB1_prod_40to50ns_R1.dcd",
            BASE / "replicas/R1/AEA_CB1_prod_50to60ns_R1.dcd",
            BASE / "replicas/R1/AEA_CB1_prod_60to70ns_R1.dcd",
            BASE / "replicas/R1/AEA_CB1_prod_70to80ns_R1.dcd",
        ],
    },
    "R2": {
        "reference": BASE / "replicas/R2/stabilization/AEA_R2_unrestrained_500ps_final.pdb",
        "segments": [
            BASE / "replicas/R2/AEA_CB1_prod_10ns_R2.dcd",
            BASE / "replicas/R2/AEA_CB1_prod_10to20ns_R2.dcd",
            BASE / "replicas/R2/AEA_CB1_prod_20to30ns_R2.dcd",
            BASE / "replicas/R2/AEA_CB1_prod_30to40ns_R2.dcd",
            BASE / "replicas/R2/AEA_CB1_prod_40to50ns_R2.dcd",
            BASE / "replicas/R2/AEA_CB1_prod_50to60ns_R2.dcd",
            BASE / "replicas/R2/AEA_CB1_prod_60to70ns_R2.dcd",
            BASE / "replicas/R2/AEA_CB1_prod_70to80ns_R2.dcd",
        ],
    },
    "R3": {
        "reference": BASE / "replicas/R3/stabilization/AEA_R3_unrestrained_500ps_final.pdb",
        "segments": [
            BASE / "replicas/R3/AEA_CB1_prod_10ns_R3.dcd",
            BASE / "replicas/R3/AEA_CB1_prod_10to20ns_R3.dcd",
            BASE / "replicas/R3/AEA_CB1_prod_20to30ns_R3.dcd",
            BASE / "replicas/R3/AEA_CB1_prod_30to40ns_R3.dcd",
            BASE / "replicas/R3/AEA_CB1_prod_40to50ns_R3.dcd",
            BASE / "replicas/R3/AEA_CB1_prod_50to60ns_R3.dcd",
            BASE / "replicas/R3/AEA_CB1_prod_60to70ns_R3.dcd",
            BASE / "replicas/R3/AEA_CB1_prod_70to80ns_R3.dcd",
        ],
    },
}
PROTEIN_NAMES = {'ALA', 'ARG', 'ASH', 'ASN', 'ASP', 'CYS', 'CYX', 'GLH', 'GLN', 'GLU', 'GLY', 'HID', 'HIE', 'HIP', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
MODELED_RANGES = ((142, 147), (254, 265), (314, 334))
CONTACT_RESIDUES = (('HIS', 178), ('THR', 197), ('TRP', 279), ('MET', 363), ('SER', 383))

def modeled(resseq):
    return any((lo <= resseq <= hi for lo, hi in MODELED_RANGES))

def kabsch(mobile, target):
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    return (rotation, mobile_center, target_center)

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
        raise RuntimeError('Zero-length central dihedral bond')
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def find_residue(topology, name, resseq):
    hits = [residue for residue in topology.residues if residue.name == name and residue.resSeq == resseq]
    if len(hits) != 1:
        raise RuntimeError(f'Expected one {name} {resseq}; found {len(hits)}')
    return hits[0]

def find_residue_dihedral(index_array, topology, target):
    matches = []
    for indices in index_array:
        residue_indices = {topology.atom(int(index)).residue.index for index in indices}
        if residue_indices == {target.index}:
            matches.append(tuple((int(index) for index in indices)))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one dihedral for {target}; found {len(matches)}')
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
    edges = sorted({tuple(sorted((i, j))) for i in ligand_set for j in adjacency[i] if i < j})
    torsions = []
    for j, k in edges:
        left = sorted(adjacency[j] - {k})
        right = sorted(adjacency[k] - {j})
        if left and right:
            torsions.append((left[0], j, k, right[0]))
    return (edges, torsions)

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
        between += len(group) * np.sum((center - overall) ** 2)
        within += np.sum((group - center) ** 2)
    if within <= 0:
        return np.inf
    return between / (n_clusters - 1) / (within / (n_samples - n_clusters))

def circular_mean_degrees(values):
    radians = np.deg2rad(np.asarray(values))
    mean = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    return np.rad2deg(mean)
print('=' * 78)
print('AEA-CB1 80-NS FOCUSED REPRESENTATIVE-STRUCTURE ANALYSIS')
print('=' * 78)
r1_reference = md.load(str(CONFIG['R1']['reference']))
r1_topology = r1_reference.topology
r1_atoms = list(r1_topology.atoms)
r1_xyz = r1_reference.xyz[0].copy()
stable_core_ca = np.array([atom.index for atom in r1_atoms if atom.residue.name in PROTEIN_NAMES and atom.name == 'CA' and (not modeled(atom.residue.resSeq))], dtype=int)
ligand_heavy = np.array([atom.index for atom in r1_atoms if atom.residue.name == 'UNK' and atom.element is not None and (atom.element.symbol != 'H')], dtype=int)
protein_heavy = np.array([atom.index for atom in r1_atoms if atom.residue.name in PROTEIN_NAMES and atom.element is not None and (atom.element.symbol != 'H')], dtype=int)
heavy_edges, torsions = ligand_torsions(r1_topology, ligand_heavy)
chi1_indices, _ = md.compute_chi1(r1_reference)
chi2_indices, _ = md.compute_chi2(r1_reference)
phe200_chi1 = find_residue_dihedral(chi1_indices, r1_topology, find_residue(r1_topology, 'PHE', 200))
trp356_chi2 = find_residue_dihedral(chi2_indices, r1_topology, find_residue(r1_topology, 'TRP', 356))
contact_indices = {}
for name, resseq in CONTACT_RESIDUES:
    target = find_residue(r1_topology, name, resseq)
    contact_indices[name, resseq] = np.array([atom.index for atom in target.atoms if atom.element is not None and atom.element.symbol != 'H'], dtype=int)
assert len(stable_core_ca) == 265
assert len(ligand_heavy) == 25
assert len(protein_heavy) == 2426
assert len(heavy_edges) == 24
assert len(torsions) == 21
assert len(contact_indices) == 5
r1_box = r1_reference.unitcell_lengths[0].copy()
unwrap_ligand(r1_xyz, ligand_heavy, r1_box)
reference_pocket = protein_heavy[np.any(np.linalg.norm(r1_xyz[ligand_heavy][:, None, :] - r1_xyz[protein_heavy][None, :, :], axis=2) <= 0.5, axis=0)]
assert len(reference_pocket) > 0
reference_pocket_center = r1_xyz[reference_pocket].mean(axis=0)
reference_ligand_center = r1_xyz[ligand_heavy].mean(axis=0)
reference_ligand_coordinates = r1_xyz[ligand_heavy] - reference_ligand_center
reference_core = r1_xyz[stable_core_ca]
feature_rows = []
descriptive_rows = []
frame_rows = []
feature_names = []
for atom_index in ligand_heavy:
    atom = r1_atoms[atom_index]
    for axis in ('x', 'y', 'z'):
        feature_names.append(f'ligand_{atom.name}_{atom_index}_{axis}_nm')
for number in range(1, len(torsions) + 1):
    feature_names.extend([f'torsion_{number:02d}_sin', f'torsion_{number:02d}_cos'])
for name, resseq in CONTACT_RESIDUES:
    feature_names.append(f'contact_{name}_{resseq}_nm')
feature_names.extend(['PHE200_chi1_sin', 'PHE200_chi1_cos', 'TRP356_chi2_sin', 'TRP356_chi2_cos'])
assert len(feature_names) == 126
global_index = 0
for replica, config in CONFIG.items():
    print()
    print('=' * 78)
    print(replica)
    print('=' * 78)
    reference = md.load(str(config['reference']))
    segments = [md.load(str(segment), top=str(config['reference'])) for segment in config['segments']]
    assert [segment.n_frames for segment in segments] == [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    trajectory = md.join(segments, check_topology=True)
    assert trajectory.n_frames == 8000
    assert trajectory.n_atoms == 42322
    assert trajectory.unitcell_lengths is not None
    atoms = list(reference.topology.atoms)
    replica_ligand = np.array([atom.index for atom in atoms if atom.residue.name == 'UNK' and atom.element is not None and (atom.element.symbol != 'H')], dtype=int)
    replica_core = np.array([atom.index for atom in atoms if atom.residue.name in PROTEIN_NAMES and atom.name == 'CA' and (not modeled(atom.residue.resSeq))], dtype=int)
    assert np.array_equal(replica_ligand, ligand_heavy)
    assert np.array_equal(replica_core, stable_core_ca)
    print('Frames:', trajectory.n_frames)
    print('Extracting 126 pose-state features...')
    for frame in range(trajectory.n_frames):
        xyz = trajectory.xyz[frame].copy()
        box = trajectory.unitcell_lengths[frame].copy()
        if not np.isfinite(xyz).all():
            raise RuntimeError(f'{replica} frame {frame}: non-finite coordinates')
        if not np.isfinite(box).all() or np.any(box <= 0):
            raise RuntimeError(f'{replica} frame {frame}: invalid box')
        unwrap_ligand(xyz, ligand_heavy, box)
        pocket_center = xyz[reference_pocket].mean(axis=0)
        ligand_center = xyz[ligand_heavy].mean(axis=0)
        shift = -box * np.round((ligand_center - pocket_center) / box)
        xyz[ligand_heavy] += shift
        rotation, mobile_center, target_center = kabsch(xyz[stable_core_ca], reference_core)
        aligned_ligand = (xyz[ligand_heavy] - mobile_center) @ rotation + target_center
        pose_coordinates = aligned_ligand - reference_ligand_center
        coordinate_features = pose_coordinates.reshape(-1)
        torsion_angles = np.array([dihedral_angle(xyz, indices) for indices in torsions])
        torsion_features = np.column_stack([np.sin(torsion_angles), np.cos(torsion_angles)]).reshape(-1)
        contact_distances = []
        for residue_key in CONTACT_RESIDUES:
            indices = contact_indices[residue_key]
            delta = xyz[ligand_heavy][:, None, :] - xyz[indices][None, :, :]
            delta -= box * np.round(delta / box)
            contact_distances.append(np.linalg.norm(delta, axis=2).min())
        contact_distances = np.asarray(contact_distances, dtype=float)
        phe_angle = dihedral_angle(xyz, phe200_chi1)
        trp_angle = dihedral_angle(xyz, trp356_chi2)
        switch_features = np.array([np.sin(phe_angle), np.cos(phe_angle), np.sin(trp_angle), np.cos(trp_angle)])
        features = np.concatenate([coordinate_features, torsion_features, contact_distances, switch_features])
        if features.shape != (126,):
            raise RuntimeError(f'Unexpected feature shape: {features.shape}')
        if not np.isfinite(features).all():
            raise RuntimeError(f'{replica} frame {frame}: non-finite features')
        feature_rows.append(features)
        descriptive_rows.append({'global_frame': global_index, 'replica': replica, 'replica_frame': frame + 1, 'time_ps': (frame + 1) * FRAME_INTERVAL_PS, 'HIS178_contact_A': contact_distances[0] * 10.0, 'THR197_contact_A': contact_distances[1] * 10.0, 'TRP279_contact_A': contact_distances[2] * 10.0, 'MET363_contact_A': contact_distances[3] * 10.0, 'SER383_contact_A': contact_distances[4] * 10.0, 'PHE200_chi1_deg': np.rad2deg(phe_angle), 'TRP356_chi2_deg': np.rad2deg(trp_angle)})
        frame_rows.append({'global_frame': global_index, 'replica': replica, 'replica_frame': frame + 1, 'time_ps': (frame + 1) * FRAME_INTERVAL_PS})
        global_index += 1
    print(f'{replica} feature extraction: PASS')
feature_matrix = np.asarray(feature_rows, dtype=float)
descriptive = pd.DataFrame(descriptive_rows)
frames = pd.DataFrame(frame_rows)
assert feature_matrix.shape == (24000, 126)
assert len(descriptive) == 24000
assert len(frames) == 24000
means = feature_matrix.mean(axis=0)
standard_deviations = feature_matrix.std(axis=0, ddof=0)
variable = standard_deviations > 1e-12
assert np.any(variable)
standardized = (feature_matrix[:, variable] - means[variable]) / standard_deviations[variable]
print()
print('=' * 78)
print('PCA')
print('=' * 78)
print('Raw features:', feature_matrix.shape[1])
print('Variable features:', standardized.shape[1])
u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
variance = singular_values ** 2
variance_ratio = variance / variance.sum()
cumulative_variance = np.cumsum(variance_ratio)
n_components = int(np.searchsorted(cumulative_variance, PCA_VARIANCE_TARGET) + 1)
scores = u[:, :n_components] * singular_values[:n_components]
assert np.isfinite(scores).all()
print('PCA components retained:', n_components)
print('Cumulative variance retained:', f'{cumulative_variance[n_components - 1]:.6f}')

print('AEA 80-NS FOCUSED FULL-45-PC REPRESENTATIVE STRUCTURES')
print("=" * 78)

FOCUSED_INPUT = Path(
    "results/AEA_CB1_80ns_focused_pose_state_comparison"
)

FOCUSED_MERGE = (
    FOCUSED_INPUT /
    "AEA_CB1_80ns_focused_complete_frame_merge.tsv"
)

FOCUSED_MANIFEST = (
    FOCUSED_INPUT /
    "AEA_CB1_80ns_focused_ensemble_manifest.tsv"
)

focused_merge = pd.read_csv(FOCUSED_MERGE, sep="\t")
focused_manifest = pd.read_csv(FOCUSED_MANIFEST, sep="\t")

assert len(frames) == 24000
assert scores.shape[0] == 24000
assert n_components == 45
assert scores.shape[1] == n_components
assert len(focused_merge) == 4952
assert len(focused_manifest) == 7

expected_ensembles = {
    "R1_state1_pre_30to40ns": {"replica": "R1", "state": 1, "frames": 1000},
    "R1_state2_excursion_48to52ns": {"replica": "R1", "state": 2, "frames": 391},
    "R1_state1_returned_70to80ns": {"replica": "R1", "state": 1, "frames": 1000},
    "R2_state1_core_run": {"replica": "R2", "state": 1, "frames": 661},
    "R2_state2_late_70to80ns": {"replica": "R2", "state": 2, "frames": 1000},
    "R3_state1_pre_late_70to78ns": {"replica": "R3", "state": 1, "frames": 797},
    "R3_state2_late_78.97to80ns": {"replica": "R3", "state": 2, "frames": 103},
}

assert set(focused_manifest.ensemble) == set(expected_ensembles)
assert focused_merge.groupby("ensemble").size().to_dict() == {
    name: specification["frames"]
    for name, specification in expected_ensembles.items()
}

STRUCTURE_DIR = OUT / "representative_structures"
STRUCTURE_DIR.mkdir(
    parents=True,
    exist_ok=False,
)


def safe_name(value):
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def load_replica_frame(replica, replica_frame):
    replica_frame = int(replica_frame)

    if not 1 <= replica_frame <= 8000:
        raise RuntimeError(
            f"{replica}: invalid replica frame {replica_frame}"
        )

    segment_counts = [1000] * 8

    if len(CONFIG[replica]["segments"]) != len(segment_counts):
        raise RuntimeError(
            f"{replica}: expected eight trajectory segments; "
            f"found {len(CONFIG[replica]['segments'])}"
        )

    segment_start = 1

    for path, count in zip(
        CONFIG[replica]["segments"],
        segment_counts,
    ):
        segment_end = segment_start + count - 1

        if segment_start <= replica_frame <= segment_end:
            local_index = replica_frame - segment_start

            trajectory = md.load_frame(
                str(path),
                index=int(local_index),
                top=str(CONFIG[replica]["reference"]),
            )

            if trajectory.n_frames != 1:
                raise RuntimeError(
                    f"{replica} frame {replica_frame}: "
                    "single-frame load failed"
                )

            if trajectory.n_atoms != 42322:
                raise RuntimeError(
                    f"{replica} frame {replica_frame}: "
                    f"unexpected atom count {trajectory.n_atoms}"
                )

            return trajectory

        segment_start = segment_end + 1

    raise RuntimeError(
        f"{replica}: frame {replica_frame} not located"
    )


def residue_heavy_indices(
    topology,
    residue_number,
):
    indices = [
        atom.index
        for atom in topology.atoms
        if (
            atom.residue.resSeq == residue_number and
            atom.element is not None and
            atom.element.symbol != "H"
        )
    ]

    if not indices:
        raise RuntimeError(
            f"No heavy atoms for residue {residue_number}"
        )

    return np.asarray(
        indices,
        dtype=int,
    )


def atom_index_for_residue(
    topology,
    residue_number,
    atom_name,
):
    matches = [
        atom.index
        for atom in topology.atoms
        if (
            atom.residue.resSeq == residue_number and
            atom.name == atom_name
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {residue_number}:{atom_name}; "
            f"found {len(matches)}"
        )

    return int(matches[0])


def torsion_degrees(
    xyz,
    indices,
):
    p0, p1, p2, p3 = [
        xyz[int(index)]
        for index in indices
    ]

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)

    if b1_norm <= 0.0:
        raise RuntimeError(
            "Degenerate torsion central bond"
        )

    b1 = b1 / b1_norm

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(
        np.cross(b1, v),
        w,
    )

    return float(
        np.degrees(
            np.arctan2(y, x)
        )
    )


def wrapped_difference(second, first):
    return float(
        (second - first + 180.0) % 360.0 - 180.0
    )


def minimum_image_contact_A(
    xyz,
    box,
    ligand_indices,
    residue_indices,
):
    delta = (
        xyz[ligand_indices][:, None, :] -
        xyz[residue_indices][None, :, :]
    )

    delta -= box * np.round(
        delta / box
    )

    return float(
        np.linalg.norm(
            delta,
            axis=2,
        ).min() * 10.0
    )


def centered_internal_rmsd_A(
    first_xyz,
    second_xyz,
):
    first = (
        first_xyz -
        first_xyz.mean(axis=0)
    )

    second = (
        second_xyz -
        second_xyz.mean(axis=0)
    )

    covariance = first.T @ second

    u, singular_values, vt = np.linalg.svd(
        covariance
    )

    rotation = u @ vt

    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    fitted = first @ rotation

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    (fitted - second) ** 2,
                    axis=1,
                )
            )
        ) * 10.0
    )


alignment_reference = md.load(
    str(CONFIG["R1"]["reference"])
)

topology = alignment_reference.topology

assert alignment_reference.n_atoms == 42322
assert len(stable_core_ca) == 265
assert len(ligand_heavy) == 25

phe200_indices = [
    atom_index_for_residue(
        topology,
        200,
        "N",
    ),
    atom_index_for_residue(
        topology,
        200,
        "CA",
    ),
    atom_index_for_residue(
        topology,
        200,
        "CB",
    ),
    atom_index_for_residue(
        topology,
        200,
        "CG",
    ),
]

trp356_indices = [
    atom_index_for_residue(
        topology,
        356,
        "CA",
    ),
    atom_index_for_residue(
        topology,
        356,
        "CB",
    ),
    atom_index_for_residue(
        topology,
        356,
        "CG",
    ),
    atom_index_for_residue(
        topology,
        356,
        "CD1",
    ),
]

contact_residues = [
    ("HIS", 178),
    ("THR", 197),
    ("TRP", 279),
    ("MET", 363),
    ("SER", 383),
]

contact_index_map = {
    f"{residue_name}{residue_number}":
        residue_heavy_indices(
            topology,
            residue_number,
        )
    for residue_name, residue_number in contact_residues
}

representative_rows = []
prepared_structures = {}

for ensemble_name, specification in expected_ensembles.items():
    manifest_row = focused_manifest.loc[
        focused_manifest.ensemble == ensemble_name
    ].iloc[0]

    assert manifest_row.replica == specification["replica"]
    assert int(manifest_row.state) == specification["state"]
    assert int(manifest_row.frames) == specification["frames"]

    ensemble_frames = focused_merge.loc[
        focused_merge.ensemble == ensemble_name
    ].copy()

    assert len(ensemble_frames) == specification["frames"]

    assert ensemble_frames.global_frame.notna().all()

    global_indices = (
        ensemble_frames.global_frame
        .to_numpy(dtype=int)
    )

    assert np.all(global_indices >= 0)
    assert np.all(global_indices < len(frames))

    identity_rows = frames.iloc[global_indices].reset_index(drop=True)
    focused_identity = ensemble_frames.reset_index(drop=True)
    assert np.array_equal(
        identity_rows.replica.to_numpy(),
        focused_identity.replica.to_numpy(),
    )
    assert np.array_equal(
        identity_rows.replica_frame.to_numpy(dtype=int),
        focused_identity.replica_frame.to_numpy(dtype=int),
    )
    assert np.allclose(
        identity_rows.time_ps.to_numpy(dtype=float),
        focused_identity.time_ps.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )

    ensemble_scores = scores[
        global_indices,
        :n_components,
    ]

    centroid = ensemble_scores.mean(axis=0)

    distances = np.linalg.norm(
        ensemble_scores - centroid,
        axis=1,
    )

    selected_position = int(
        np.argmin(distances)
    )

    selected_global_index = int(
        global_indices[selected_position]
    )

    record = frames.iloc[
        selected_global_index
    ]

    if int(record.global_frame) != selected_global_index:
        raise RuntimeError(
            "Global-frame/PCA-row identity mismatch"
        )

    replica = str(record.replica)
    replica_frame = int(record.replica_frame)
    time_ps = float(record.time_ps)

    trajectory = load_replica_frame(
        replica,
        replica_frame,
    )

    xyz = trajectory.xyz[0].copy()
    box = trajectory.unitcell_lengths[0].copy()

    assert np.isfinite(xyz).all()
    assert np.isfinite(box).all()
    assert np.all(box > 0)

    unwrap_ligand(
        xyz,
        ligand_heavy,
        box,
    )

    pocket_center = xyz[
        reference_pocket
    ].mean(axis=0)

    ligand_center = xyz[
        ligand_heavy
    ].mean(axis=0)

    shift = -box * np.round(
        (ligand_center - pocket_center) / box
    )

    xyz[ligand_heavy] += shift
    trajectory.xyz[0] = xyz

    trajectory.superpose(
        alignment_reference,
        atom_indices=np.asarray(
            stable_core_ca,
            dtype=int,
        ),
        ref_atom_indices=np.asarray(
            stable_core_ca,
            dtype=int,
        ),
    )

    aligned_xyz = trajectory.xyz[0].copy()
    aligned_box = trajectory.unitcell_lengths[0].copy()

    assert np.isfinite(aligned_xyz).all()
    assert np.isfinite(aligned_box).all()
    assert np.all(aligned_box > 0)

    pdb_name = (
        f"AEA_CB1_{safe_name(ensemble_name)}_"
        f"frame_{replica_frame}.pdb"
    )

    pdb_path = STRUCTURE_DIR / pdb_name

    trajectory.save_pdb(
        str(pdb_path)
    )

    contact_values = {}

    for residue_key, residue_indices in contact_index_map.items():
        contact_values[
            f"{residue_key}_contact_A"
        ] = minimum_image_contact_A(
            aligned_xyz,
            aligned_box,
            np.asarray(
                ligand_heavy,
                dtype=int,
            ),
            residue_indices,
        )

    phe200 = torsion_degrees(
        aligned_xyz,
        phe200_indices,
    )

    trp356 = torsion_degrees(
        aligned_xyz,
        trp356_indices,
    )

    representative_row = {
        "ensemble": ensemble_name,
        "replica": replica,
        "state": specification["state"],
        "ensemble_frames":
            specification["frames"],
        "global_frame":
            selected_global_index,
        "replica_frame":
            replica_frame,
        "time_ps":
            time_ps,
        "distance_to_45PC_ensemble_centroid":
            float(distances[selected_position]),
        "PHE200_chi1_deg":
            phe200,
        "TRP356_chi2_deg":
            trp356,
        "pdb_file":
            str(
                Path("representative_structures") /
                pdb_name
            ),
    }

    representative_row.update(
        contact_values
    )

    representative_rows.append(
        representative_row
    )

    prepared_structures[ensemble_name] = {
        "xyz": aligned_xyz,
        "box": aligned_box,
        "trajectory": trajectory,
        "replica": replica,
        "replica_frame": replica_frame,
        "time_ps": time_ps,
        "pdb_path": pdb_path,
        "contacts": contact_values,
        "phe200": phe200,
        "trp356": trp356,
    }

representative_table = pd.DataFrame(
    representative_rows
)

assert len(representative_table) == 7
assert representative_table.replica.tolist().count("R1") == 3
assert representative_table.replica.tolist().count("R2") == 2
assert representative_table.replica.tolist().count("R3") == 2

representative_table.to_csv(
    OUT /
    "AEA_CB1_80ns_focused_structural_representatives.tsv",
    sep="\t",
    index=False,
)

comparisons = [
    {"comparison": "R1_excursion_vs_pre", "first": "R1_state1_pre_30to40ns", "second": "R1_state2_excursion_48to52ns"},
    {"comparison": "R1_returned_vs_excursion", "first": "R1_state2_excursion_48to52ns", "second": "R1_state1_returned_70to80ns"},
    {"comparison": "R1_returned_vs_pre", "first": "R1_state1_pre_30to40ns", "second": "R1_state1_returned_70to80ns"},
    {"comparison": "R2_late_state2_vs_early_state1", "first": "R2_state1_core_run", "second": "R2_state2_late_70to80ns"},
    {"comparison": "R3_late_state2_vs_pre_late_state1", "first": "R3_state1_pre_late_70to78ns", "second": "R3_state2_late_78.97to80ns"},
]

comparison_rows = []
atom_rows = []

for comparison_specification in comparisons:
    comparison_name = comparison_specification[
        "comparison"
    ]

    first_name = comparison_specification[
        "first"
    ]

    second_name = comparison_specification[
        "second"
    ]

    first = prepared_structures[first_name]
    second = prepared_structures[second_name]

    first_xyz = first["xyz"]
    second_xyz = second["xyz"]

    ligand_indices = np.asarray(
        ligand_heavy,
        dtype=int,
    )

    ligand_delta = (
        second_xyz[ligand_indices] -
        first_xyz[ligand_indices]
    )

    ligand_atom_displacements = (
        np.linalg.norm(
            ligand_delta,
            axis=1,
        ) * 10.0
    )

    pose_rmsd = float(
        np.sqrt(
            np.mean(
                np.sum(
                    ligand_delta ** 2,
                    axis=1,
                )
            )
        ) * 10.0
    )

    centroid_displacement = float(
        np.linalg.norm(
            second_xyz[
                ligand_indices
            ].mean(axis=0) -
            first_xyz[
                ligand_indices
            ].mean(axis=0)
        ) * 10.0
    )

    internal_rmsd = centered_internal_rmsd_A(
        first_xyz[ligand_indices],
        second_xyz[ligand_indices],
    )

    row = {
        "comparison":
            comparison_name,
        "first_ensemble":
            first_name,
        "second_ensemble":
            second_name,
        "first_replica":
            first["replica"],
        "second_replica":
            second["replica"],
        "first_replica_frame":
            first["replica_frame"],
        "second_replica_frame":
            second["replica_frame"],
        "first_time_ps":
            first["time_ps"],
        "second_time_ps":
            second["time_ps"],
        "AEA_pose_RMSD_A":
            pose_rmsd,
        "AEA_centroid_displacement_A":
            centroid_displacement,
        "AEA_internal_RMSD_A":
            internal_rmsd,
        "first_PHE200_chi1_deg":
            first["phe200"],
        "second_PHE200_chi1_deg":
            second["phe200"],
        "second_minus_first_PHE200_chi1_wrapped_deg":
            wrapped_difference(
                second["phe200"],
                first["phe200"],
            ),
        "first_TRP356_chi2_deg":
            first["trp356"],
        "second_TRP356_chi2_deg":
            second["trp356"],
        "second_minus_first_TRP356_chi2_wrapped_deg":
            wrapped_difference(
                second["trp356"],
                first["trp356"],
            ),
    }

    for residue_key in contact_index_map:
        first_contact = first["contacts"][
            f"{residue_key}_contact_A"
        ]

        second_contact = second["contacts"][
            f"{residue_key}_contact_A"
        ]

        row[
            f"first_{residue_key}_contact_A"
        ] = first_contact

        row[
            f"second_{residue_key}_contact_A"
        ] = second_contact

        row[
            f"second_minus_first_{residue_key}_contact_A"
        ] = (
            second_contact -
            first_contact
        )

    comparison_rows.append(row)

    for atom_index, displacement in zip(
        ligand_indices,
        ligand_atom_displacements,
    ):
        atom = topology.atom(
            int(atom_index)
        )

        atom_rows.append({
            "comparison":
                comparison_name,
            "first_ensemble":
                first_name,
            "second_ensemble":
                second_name,
            "atom_index":
                int(atom_index),
            "atom_name":
                atom.name,
            "residue_name":
                atom.residue.name,
            "first_x_A":
                float(
                    first_xyz[
                        atom_index,
                        0,
                    ] * 10.0
                ),
            "first_y_A":
                float(
                    first_xyz[
                        atom_index,
                        1,
                    ] * 10.0
                ),
            "first_z_A":
                float(
                    first_xyz[
                        atom_index,
                        2,
                    ] * 10.0
                ),
            "second_x_A":
                float(
                    second_xyz[
                        atom_index,
                        0,
                    ] * 10.0
                ),
            "second_y_A":
                float(
                    second_xyz[
                        atom_index,
                        1,
                    ] * 10.0
                ),
            "second_z_A":
                float(
                    second_xyz[
                        atom_index,
                        2,
                    ] * 10.0
                ),
            "second_minus_first_displacement_A":
                float(displacement),
        })

comparison_table = pd.DataFrame(
    comparison_rows
)

atom_displacement_table = pd.DataFrame(
    atom_rows
)

assert len(comparison_table) == 5
assert len(atom_displacement_table) == 125

assert np.isfinite(
    representative_table.select_dtypes(
        include=[np.number]
    ).to_numpy(dtype=float)
).all()

assert np.isfinite(
    comparison_table.select_dtypes(
        include=[np.number]
    ).to_numpy(dtype=float)
).all()

assert np.isfinite(
    atom_displacement_table.select_dtypes(
        include=[np.number]
    ).to_numpy(dtype=float)
).all()

comparison_table.to_csv(
    OUT /
    "AEA_CB1_80ns_focused_structural_comparisons.tsv",
    sep="\t",
    index=False,
)

atom_displacement_table.to_csv(
    OUT /
    "AEA_CB1_80ns_focused_atom_displacements.tsv",
    sep="\t",
    index=False,
)

print()
print("=" * 78)
print("SEVEN FULL-45-PC REPRESENTATIVES")
print("=" * 78)
print(
    representative_table.to_string(
        index=False
    )
)

print()
print("=" * 78)
print("PAIRWISE STRUCTURAL COMPARISONS")
print("=" * 78)
print(
    comparison_table.to_string(
        index=False
    )
)

print()
print("Exact focused ensemble frame assignments reproduced: PASS")
print("Seven focused full-45-PC representatives: PASS")
print("Seven complete 42,322-atom PDB structures: PASS")
print("Five PBC-corrected structural comparisons: PASS")
print("One hundred twenty-five ligand-heavy-atom displacement records: PASS")
print("Twenty-five minimum-image contact pairs: PASS")
print("Ten PHE200/TRP356 torsion pairs: PASS")
print('AEA 80-NS FOCUSED REPRESENTATIVE-STRUCTURE COMPARISON: COMPLETE')
