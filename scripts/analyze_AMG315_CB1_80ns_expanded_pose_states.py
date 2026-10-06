from pathlib import Path
from collections import defaultdict
import mdtraj as md
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
BASE = Path('systems/AMG315_CB1')
OUT = Path('results/AMG315_CB1_pose_states_80ns_expanded')
OUT.mkdir(parents=True, exist_ok=False)
FRAME_INTERVAL_PS = 10.0
PCA_VARIANCE_TARGET = 0.9
CLUSTER_RANGE = range(2, 9)
CONFIG = {'R1': {'reference': BASE / 'equil/AMG315_CB1_NPT_unrestrained_final.pdb', 'segments': [BASE / 'replicas/R1/AMG315_CB1_prod_5ns_R1.dcd', BASE / 'replicas/R1/AMG315_CB1_prod_5to10ns_R1.dcd', BASE / 'replicas/R1/AMG315_CB1_prod_10to20ns_R1.dcd', BASE / 'replicas/R1/AMG315_CB1_prod_20to30ns_R1.dcd', BASE / 'replicas/R1/AMG315_CB1_prod_30to40ns_R1.dcd', BASE / 'replicas/R1/AMG315_CB1_prod_40to50ns_R1.dcd', BASE / 'replicas/R1/AMG315_CB1_prod_50to60ns_R1.dcd', BASE / 'replicas/R1/AMG315_CB1_prod_60to70ns_R1.dcd', BASE / 'replicas/R1/AMG315_CB1_prod_70to80ns_R1.dcd']}, 'R2': {'reference': BASE / 'replicas/R2/stabilization/AMG315_R2_unrestrained_500ps_final.pdb', 'segments': [BASE / 'replicas/R2/AMG315_CB1_prod_5ns_R2.dcd', BASE / 'replicas/R2/AMG315_CB1_prod_5to10ns_R2.dcd', BASE / 'replicas/R2/AMG315_CB1_prod_10to20ns_R2.dcd', BASE / 'replicas/R2/AMG315_CB1_prod_20to30ns_R2.dcd', BASE / 'replicas/R2/AMG315_CB1_prod_30to40ns_R2.dcd', BASE / 'replicas/R2/AMG315_CB1_prod_40to50ns_R2.dcd', BASE / 'replicas/R2/AMG315_CB1_prod_50to60ns_R2.dcd', BASE / 'replicas/R2/AMG315_CB1_prod_60to70ns_R2.dcd', BASE / 'replicas/R2/AMG315_CB1_prod_70to80ns_R2.dcd']}, 'R3': {'reference': BASE / 'replicas/R3/stabilization/AMG315_R3_unrestrained_500ps_final.pdb', 'segments': [BASE / 'replicas/R3/AMG315_CB1_prod_5ns_R3.dcd', BASE / 'replicas/R3/AMG315_CB1_prod_5to10ns_R3.dcd', BASE / 'replicas/R3/AMG315_CB1_prod_10to20ns_R3.dcd', BASE / 'replicas/R3/AMG315_CB1_prod_20to30ns_R3.dcd', BASE / 'replicas/R3/AMG315_CB1_prod_30to40ns_R3.dcd', BASE / 'replicas/R3/AMG315_CB1_prod_40to50ns_R3.dcd', BASE / 'replicas/R3/AMG315_CB1_prod_50to60ns_R3.dcd', BASE / 'replicas/R3/AMG315_CB1_prod_60to70ns_R3.dcd', BASE / 'replicas/R3/AMG315_CB1_prod_70to80ns_R3.dcd']}}
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
print('AMG315-CB1 THREE-REPLICA 80-NS EXPANDED POSE-STATE ANALYSIS')
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
assert len(ligand_heavy) == 27
assert len(protein_heavy) == 2426
assert len(heavy_edges) == 26
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
assert len(feature_names) == 132
global_index = 0
for replica, config in CONFIG.items():
    print()
    print('=' * 78)
    print(replica)
    print('=' * 78)
    reference = md.load(str(config['reference']))
    segments = [md.load(str(segment), top=str(config['reference'])) for segment in config['segments']]
    assert [segment.n_frames for segment in segments] == [500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    trajectory = md.join(segments, check_topology=True)
    assert trajectory.n_frames == 8000
    assert trajectory.n_atoms == 42295
    assert trajectory.unitcell_lengths is not None
    atoms = list(reference.topology.atoms)
    replica_ligand = np.array([atom.index for atom in atoms if atom.residue.name == 'UNK' and atom.element is not None and (atom.element.symbol != 'H')], dtype=int)
    replica_core = np.array([atom.index for atom in atoms if atom.residue.name in PROTEIN_NAMES and atom.name == 'CA' and (not modeled(atom.residue.resSeq))], dtype=int)
    assert np.array_equal(replica_ligand, ligand_heavy)
    assert np.array_equal(replica_core, stable_core_ca)
    print('Frames:', trajectory.n_frames)
    print('Extracting 132 pose-state features...')
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
        if features.shape != (132,):
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
assert feature_matrix.shape == (24000, 132)
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
pca_table = pd.DataFrame({'component': np.arange(1, len(variance_ratio) + 1), 'variance_fraction': variance_ratio, 'cumulative_variance': cumulative_variance})
pca_table.to_csv(OUT / 'AMG315_CB1_80ns_expanded_pose_PCA_variance.tsv', sep='\t', index=False)
print()
print('=' * 78)
print('WARD CLUSTERING')
print('=' * 78)
ward = linkage(scores, method='ward', metric='euclidean', optimal_ordering=False)
diagnostics = []
candidate_labels = {}
for k in CLUSTER_RANGE:
    labels = fcluster(ward, t=k, criterion='maxclust')
    counts = np.bincount(labels)[1:]
    score = calinski_harabasz(scores, labels)
    replica_coverage = []
    for state in range(1, k + 1):
        represented = sum((np.any((labels == state) & (frames.replica.to_numpy() == replica)) for replica in ('R1', 'R2', 'R3')))
        replica_coverage.append(represented)
    diagnostics.append({'states': k, 'calinski_harabasz': score, 'minimum_state_frames': int(counts.min()), 'minimum_state_percent': 100.0 * counts.min() / len(labels), 'states_present_in_all_3_replicas': int(np.sum(np.asarray(replica_coverage) == 3)), 'minimum_replica_coverage': int(min(replica_coverage))})
    candidate_labels[k] = labels
candidate_frame_rows = []
candidate_summary_rows = []
for candidate_k in CLUSTER_RANGE:
    candidate_raw = candidate_labels[candidate_k]
    candidate_order = pd.Series(candidate_raw).value_counts().sort_values(ascending=False).index.tolist()
    candidate_map = {old: new for new, old in enumerate(candidate_order, start=1)}
    candidate_states = np.array([candidate_map[label] for label in candidate_raw], dtype=int)
    for index, state in enumerate(candidate_states):
        record = frames.iloc[index]
        candidate_frame_rows.append({'states_in_model': candidate_k, 'global_frame': int(record.global_frame), 'replica': record.replica, 'replica_frame': int(record.replica_frame), 'time_ps': float(record.time_ps), 'candidate_state': int(state)})
    for state in range(1, candidate_k + 1):
        state_mask = candidate_states == state
        state_frames = frames.loc[state_mask]
        row = {'states_in_model': candidate_k, 'candidate_state': state, 'frames': int(np.sum(state_mask)), 'occupancy_percent': 100.0 * np.mean(state_mask), 'replicas_present': int(state_frames.replica.nunique())}
        for replica in ('R1', 'R2', 'R3'):
            replica_mask = frames.replica.to_numpy() == replica
            row[f'{replica}_frames'] = int(np.sum(state_mask & replica_mask))
            row[f'{replica}_percent'] = 100.0 * np.sum(state_mask & replica_mask) / np.sum(replica_mask)
            for interval_name, time_mask in (('first10ns', frames.time_ps.to_numpy() <= 10000.0), ('second10ns', (frames.time_ps.to_numpy() > 10000.0) & (frames.time_ps.to_numpy() <= 20000.0)), ('third10ns', (frames.time_ps.to_numpy() > 20000.0) & (frames.time_ps.to_numpy() <= 30000.0)), ('fourth10ns', (frames.time_ps.to_numpy() > 30000.0) & (frames.time_ps.to_numpy() <= 40000.0)), ('fifth10ns', (frames.time_ps.to_numpy() > 40000.0) & (frames.time_ps.to_numpy() <= 50000.0)), ('sixth10ns', (frames.time_ps.to_numpy() > 50000.0) & (frames.time_ps.to_numpy() <= 60000.0)), ('seventh10ns', (frames.time_ps.to_numpy() > 60000.0) & (frames.time_ps.to_numpy() <= 70000.0)), ('eighth10ns', frames.time_ps.to_numpy() > 70000.0)):
                denominator = np.sum(replica_mask & time_mask)
                row[f'{replica}_{interval_name}_percent'] = 100.0 * np.sum(state_mask & replica_mask & time_mask) / denominator
        candidate_summary_rows.append(row)
candidate_frames = pd.DataFrame(candidate_frame_rows)
candidate_summary = pd.DataFrame(candidate_summary_rows)
assert len(candidate_frames) == 168000
assert len(candidate_summary) == 35
candidate_frames.to_csv(OUT / 'AMG315_CB1_80ns_expanded_candidate_partitions.tsv', sep='\t', index=False)
candidate_summary.to_csv(OUT / 'AMG315_CB1_80ns_expanded_candidate_state_summary.tsv', sep='\t', index=False)
diagnostics = pd.DataFrame(diagnostics)
eligible = diagnostics[diagnostics.minimum_state_percent >= 1.0]
if eligible.empty:
    eligible = diagnostics.copy()
selected_k = int(eligible.sort_values(['calinski_harabasz', 'states'], ascending=[False, True]).iloc[0].states)
raw_labels = candidate_labels[selected_k]
occupancy_order = pd.Series(raw_labels).value_counts().sort_values(ascending=False).index.tolist()
state_map = {old: new for new, old in enumerate(occupancy_order, start=1)}
labels = np.array([state_map[label] for label in raw_labels], dtype=int)
frames['state'] = labels
for component in range(min(5, n_components)):
    frames[f'PC{component + 1}'] = scores[:, component]
frames = pd.concat([frames, descriptive.drop(columns=['global_frame', 'replica', 'replica_frame', 'time_ps'])], axis=1)
frozen_historical = pd.read_csv(Path('results/AMG315_CB1_pose_states_20ns') / 'AMG315_CB1_20ns_pose_state_frames.tsv', sep='\t')[['replica', 'replica_frame', 'time_ps', 'state']]
frozen_projected_20to30 = pd.read_csv(Path('results/AMG315_CB1_frozen_pose_projection_30ns') / 'AMG315_CB1_20to30ns_frozen_pose_state_frames.tsv', sep='\t')[['replica', 'replica_frame', 'time_ps', 'state']]
frozen_projected_30to40 = pd.read_csv(Path('results/AMG315_CB1_frozen_pose_distance_calibration_40ns') / 'AMG315_CB1_30to40ns_frozen_pose_state_frames.tsv', sep='\t')[['replica', 'replica_frame', 'time_ps', 'state']]
frozen_projected_40to50 = pd.read_csv(Path('results/AMG315_CB1_frozen_pose_distance_calibration_50ns') / 'AMG315_CB1_40to50ns_frozen_pose_state_frames.tsv', sep='\t')[['replica', 'replica_frame', 'time_ps', 'state']]
frozen_projected_50to60 = pd.read_csv(Path('results/AMG315_CB1_frozen_pose_distance_calibration_60ns') / 'AMG315_CB1_50to60ns_frozen_pose_state_frames.tsv', sep='\t')[['replica', 'replica_frame', 'time_ps', 'state']]
frozen_projected_60to70 = pd.read_csv(Path('results/AMG315_CB1_frozen_pose_distance_calibration_70ns') / 'AMG315_CB1_60to70ns_frozen_pose_state_frames.tsv', sep='\t')[['replica', 'replica_frame', 'time_ps', 'state']]
frozen_projected_70to80 = pd.read_csv(Path('results/AMG315_CB1_frozen_pose_distance_calibration_80ns') / 'AMG315_CB1_70to80ns_frozen_pose_state_frames.tsv', sep='\t')[['replica', 'replica_frame', 'time_ps', 'state']]
frozen_reference = pd.concat([frozen_historical, frozen_projected_20to30, frozen_projected_30to40, frozen_projected_40to50, frozen_projected_50to60, frozen_projected_60to70, frozen_projected_70to80], ignore_index=True)
frozen_reference = frozen_reference.rename(columns={'state': 'frozen_state'})
assert len(frozen_reference) == 24000
candidate_comparison = candidate_frames.merge(frozen_reference, on=['replica', 'replica_frame', 'time_ps'], how='left', validate='many_to_one')
assert candidate_comparison.frozen_state.notna().all()
candidate_comparison['frozen_state'] = candidate_comparison.frozen_state.astype(int)
comparison_rows = []
for (candidate_k, candidate_state, frozen_state), group in candidate_comparison.groupby(['states_in_model', 'candidate_state', 'frozen_state']):
    comparison_rows.append({'states_in_model': int(candidate_k), 'expanded_state': int(candidate_state), 'frozen_state': int(frozen_state), 'frames': len(group), 'percent_within_expanded_state': 100.0 * len(group) / np.sum((candidate_comparison.states_in_model == candidate_k) & (candidate_comparison.candidate_state == candidate_state))})
comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(OUT / 'AMG315_CB1_80ns_expanded_vs_frozen_state_comparison.tsv', sep='\t', index=False)
diagnostics['selected'] = diagnostics.states == selected_k
diagnostics.to_csv(OUT / 'AMG315_CB1_80ns_expanded_cluster_diagnostics.tsv', sep='\t', index=False)
frames.to_csv(OUT / 'AMG315_CB1_80ns_expanded_pose_state_frames.tsv', sep='\t', index=False)
print(diagnostics.to_string(index=False))
print()
print('Selected number of states:', selected_k)
occupancy_rows = []
for replica in ('R1', 'R2', 'R3'):
    subset = frames[frames.replica == replica]
    for state in range(1, selected_k + 1):
        count = int(np.sum(subset.state == state))
        occupancy_rows.append({'replica': replica, 'state': state, 'frames': count, 'occupancy_percent': 100.0 * count / len(subset), 'first10ns_percent': 100.0 * np.mean(subset[subset.time_ps <= 10000].state == state), 'second10ns_percent': 100.0 * np.mean(subset[(subset.time_ps > 10000) & (subset.time_ps <= 20000)].state == state), 'third10ns_percent': 100.0 * np.mean(subset[(subset.time_ps > 20000) & (subset.time_ps <= 30000)].state == state), 'fourth10ns_percent': 100.0 * np.mean(subset[(subset.time_ps > 30000) & (subset.time_ps <= 40000)].state == state), 'fifth10ns_percent': 100.0 * np.mean(subset[(subset.time_ps > 40000.0) & (subset.time_ps <= 50000.0)].state == state), 'sixth10ns_percent': 100.0 * np.mean(subset[(subset.time_ps > 50000.0) & (subset.time_ps <= 60000.0)].state == state), 'seventh10ns_percent': 100.0 * np.mean(subset[(subset.time_ps > 60000.0) & (subset.time_ps <= 70000.0)].state == state), 'eighth10ns_percent': 100.0 * np.mean(subset[subset.time_ps > 70000.0].state == state)})
occupancy = pd.DataFrame(occupancy_rows)
occupancy.to_csv(OUT / 'AMG315_CB1_80ns_expanded_pose_state_replica_occupancy.tsv', sep='\t', index=False)
state_rows = []
for state in range(1, selected_k + 1):
    subset = frames[frames.state == state]
    row = {'state': state, 'frames': len(subset), 'occupancy_percent': 100.0 * len(subset) / len(frames), 'replicas_present': subset.replica.nunique(), 'PHE200_chi1_circular_mean_deg': circular_mean_degrees(subset.PHE200_chi1_deg), 'TRP356_chi2_circular_mean_deg': circular_mean_degrees(subset.TRP356_chi2_deg)}
    for name, resseq in CONTACT_RESIDUES:
        column = f'{name}{resseq}_contact_A'
        row[f'mean_{column}'] = subset[column].mean()
    state_rows.append(row)
state_summary = pd.DataFrame(state_rows)
state_summary.to_csv(OUT / 'AMG315_CB1_80ns_expanded_pose_state_summary.tsv', sep='\t', index=False)
transition_rows = []
for replica in ('R1', 'R2', 'R3'):
    replica_states = frames.loc[frames.replica == replica, 'state'].to_numpy(dtype=int)
    for from_state in range(1, selected_k + 1):
        for to_state in range(1, selected_k + 1):
            count = int(np.sum((replica_states[:-1] == from_state) & (replica_states[1:] == to_state)))
            transition_rows.append({'replica': replica, 'from_state': from_state, 'to_state': to_state, 'transition_count': count})
transitions = pd.DataFrame(transition_rows)
transitions.to_csv(OUT / 'AMG315_CB1_80ns_expanded_pose_state_transitions.tsv', sep='\t', index=False)
representative_rows = []
for state in range(1, selected_k + 1):
    indices = np.where(labels == state)[0]
    center = scores[indices].mean(axis=0)
    distances = np.linalg.norm(scores[indices] - center, axis=1)
    representative = indices[np.argmin(distances)]
    record = frames.iloc[representative]
    representative_rows.append({'state': state, 'global_frame': int(record.global_frame), 'replica': record.replica, 'replica_frame': int(record.replica_frame), 'time_ps': record.time_ps, 'distance_to_state_centroid': float(distances.min())})
representatives = pd.DataFrame(representative_rows)
representatives.to_csv(OUT / 'AMG315_CB1_80ns_expanded_pose_state_representatives.tsv', sep='\t', index=False)
feature_metadata = pd.DataFrame({'feature': feature_names, 'mean': means, 'standard_deviation': standard_deviations, 'variable_for_PCA': variable})
feature_metadata.to_csv(OUT / 'AMG315_CB1_80ns_expanded_pose_feature_metadata.tsv', sep='\t', index=False)
print()
print('=' * 78)
print('POSE-STATE SUMMARY')
print('=' * 78)
print(state_summary.to_string(index=False))
print()
print('=' * 78)
print('REPLICA OCCUPANCY')
print('=' * 78)
print(occupancy.to_string(index=False))
print()
print('=' * 78)
print('REPRESENTATIVE FRAMES')
print('=' * 78)
print(representatives.to_string(index=False))
print()
print('Saved output directory:', OUT)
print('AMG315 THREE-REPLICA 80-NS EXPANDED POSE-STATE ANALYSIS: COMPLETE')
