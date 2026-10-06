from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("results/AMG315_AEA_CB1_80ns_comparison")
OUT.mkdir(parents=True, exist_ok=False)

CONFIG = {
    "AMG315": {
        "qc": Path("results/AMG315_CB1_production_QC_80ns"),
        "prefix": "AMG315_CB1_80ns",
        "state": Path("results/AMG315_CB1_pose_states_80ns_expanded"),
        "state_prefix": "AMG315_CB1_80ns_expanded",
    },
    "AEA": {
        "qc": Path("results/AEA_CB1_80ns_QC"),
        "prefix": "AEA_CB1_80ns",
        "state": Path("results/AEA_CB1_pose_states_80ns_expanded"),
        "state_prefix": "AEA_CB1_80ns_expanded",
    },
}

METRICS = [
    "stable_core_CA_RMSD_A",
    "TM_core_CA_RMSD_A",
    "all_protein_CA_RMSD_A",
    "protein_heavy_RMSD_A",
    "ligand_pose_RMSD_A",
    "ligand_centroid_displacement_A",
    "ligand_internal_RMSD_A",
    "minimum_ligand_CB1_contact_A",
    "box_volume_nm3",
]


def finite(table, label):
    numeric = table.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{label}: nonfinite numerical value")


def exact_permutation_p(amg, aea):
    values = np.concatenate([amg, aea])
    observed = abs(aea.mean() - amg.mean())
    extreme = 0
    total = 0
    for chosen in combinations(range(6), 3):
        chosen = set(chosen)
        group_a = np.array([values[i] for i in range(6) if i in chosen])
        group_b = np.array([values[i] for i in range(6) if i not in chosen])
        difference = abs(group_b.mean() - group_a.mean())
        extreme += difference >= observed - 1e-12
        total += 1
    return extreme / total


def hedges_g(amg, aea):
    n1, n2 = len(amg), len(aea)
    pooled_variance = (
        (n1 - 1) * np.var(amg, ddof=1)
        + (n2 - 1) * np.var(aea, ddof=1)
    ) / (n1 + n2 - 2)
    if pooled_variance <= 0:
        return np.nan
    d = (aea.mean() - amg.mean()) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2 - 2) - 1.0)
    return correction * d


frame_tables = []
contact_tables = []
state_frames = []
state_summaries = []
transition_tables = []

for ligand, config in CONFIG.items():
    for replica in ["R1", "R2", "R3"]:
        frame_file = config["qc"] / f'{config["prefix"]}_{replica}_frame_metrics.tsv'
        contact_file = config["qc"] / f'{config["prefix"]}_{replica}_contact_occupancy.tsv'
        if not frame_file.is_file() or not contact_file.is_file():
            raise FileNotFoundError(frame_file if not frame_file.is_file() else contact_file)

        table = pd.read_csv(frame_file, sep="\t")
        assert len(table) == 8000
        assert table["replica"].eq(replica).all()
        assert table["frame"].tolist() == list(range(1, 8001))
        assert np.allclose(table["time_ps"], np.arange(10.0, 80000.1, 10.0))
        table = table.rename(columns={
            f"{ligand}_pose_RMSD_A": "ligand_pose_RMSD_A",
            f"{ligand}_centroid_displacement_A": "ligand_centroid_displacement_A",
            f"{ligand}_internal_RMSD_A": "ligand_internal_RMSD_A",
            f"minimum_{ligand}_CB1_contact_A": "minimum_ligand_CB1_contact_A",
        })
        assert set(METRICS).issubset(table.columns)
        table.insert(0, "ligand", ligand)
        finite(table, f"{ligand} {replica} frame table")
        frame_tables.append(table)

        contacts = pd.read_csv(contact_file, sep="\t")
        assert set(contacts.columns) == {
            "replica", "residue_name", "residue_number",
            "frames_contacted", "occupancy_percent",
        }
        contacts.insert(0, "ligand", ligand)
        finite(contacts, f"{ligand} {replica} contacts")
        contact_tables.append(contacts)

    state_file = config["state"] / f'{config["state_prefix"]}_pose_state_frames.tsv'
    summary_file = config["state"] / f'{config["state_prefix"]}_pose_state_summary.tsv'
    transition_file = config["state"] / f'{config["state_prefix"]}_pose_state_transitions.tsv'
    sf = pd.read_csv(state_file, sep="\t")
    ss = pd.read_csv(summary_file, sep="\t")
    tt = pd.read_csv(transition_file, sep="\t")
    assert len(sf) == 24000
    assert set(sf["replica"]) == {"R1", "R2", "R3"}
    assert set(sf["state"]) == {1, 2}
    assert len(ss) == 2
    assert len(tt) == 12
    for table in [sf, ss, tt]:
        table.insert(0, "ligand", ligand)
        finite(table, f"{ligand} pose-state table")
    state_frames.append(sf)
    state_summaries.append(ss)
    transition_tables.append(tt)

frames = pd.concat(frame_tables, ignore_index=True)
contacts_observed = pd.concat(contact_tables, ignore_index=True)
states = pd.concat(state_frames, ignore_index=True)
state_summary = pd.concat(state_summaries, ignore_index=True)
transitions = pd.concat(transition_tables, ignore_index=True)

assert len(frames) == 48000
assert len(states) == 48000

frames.to_csv(
    OUT / "AMG315_AEA_80ns_harmonized_frame_metrics.tsv",
    sep="\t", index=False,
)

replica_rows = []
interval_rows = []
for (ligand, replica), group in frames.groupby(["ligand", "replica"], sort=True):
    row = {"ligand": ligand, "replica": replica, "frames": len(group)}
    for metric in METRICS:
        row[f"mean_{metric}"] = group[metric].mean()
        row[f"SD_{metric}"] = group[metric].std(ddof=1)
    row["pocket_retention_percent"] = 100.0 * np.mean(
        group["minimum_ligand_CB1_contact_A"] <= 4.0
    )
    row["frames_contact_gt5A"] = int(np.sum(
        group["minimum_ligand_CB1_contact_A"] > 5.0
    ))
    last10 = group[group["time_ps"] > 70000.0]
    for metric in METRICS:
        row[f"last10ns_mean_{metric}"] = last10[metric].mean()
    replica_rows.append(row)

    for block in range(8):
        lo = block * 10000.0
        hi = (block + 1) * 10000.0
        subset = group[(group["time_ps"] > lo) & (group["time_ps"] <= hi)]
        assert len(subset) == 1000
        block_row = {
            "ligand": ligand,
            "replica": replica,
            "interval": f"{block * 10}-{(block + 1) * 10} ns",
            "first_time_ps": subset["time_ps"].min(),
            "last_time_ps": subset["time_ps"].max(),
            "frames": len(subset),
        }
        for metric in METRICS:
            block_row[f"mean_{metric}"] = subset[metric].mean()
        interval_rows.append(block_row)

replica_summary = pd.DataFrame(replica_rows)
interval_summary = pd.DataFrame(interval_rows)
assert len(replica_summary) == 6
assert len(interval_summary) == 48
finite(replica_summary, "replica summary")
finite(interval_summary, "interval summary")
replica_summary.to_csv(OUT / "AMG315_AEA_80ns_replica_summary.tsv", sep="\t", index=False)
interval_summary.to_csv(OUT / "AMG315_AEA_80ns_10ns_interval_summary.tsv", sep="\t", index=False)

ligand_rows = []
effect_rows = []
replica_metrics = [
    column for column in replica_summary.columns
    if column.startswith("mean_") or column.startswith("last10ns_mean_")
] + ["pocket_retention_percent", "frames_contact_gt5A"]

for ligand, group in replica_summary.groupby("ligand", sort=True):
    row = {"ligand": ligand, "replicas": len(group)}
    for metric in replica_metrics:
        row[f"mean_{metric}"] = group[metric].mean()
        row[f"SD_{metric}"] = group[metric].std(ddof=1)
    ligand_rows.append(row)

for metric in replica_metrics:
    amg = replica_summary.loc[replica_summary["ligand"] == "AMG315", metric].to_numpy(float)
    aea = replica_summary.loc[replica_summary["ligand"] == "AEA", metric].to_numpy(float)
    effect_rows.append({
        "metric": metric,
        "AMG315_mean": amg.mean(),
        "AMG315_SD": amg.std(ddof=1),
        "AEA_mean": aea.mean(),
        "AEA_SD": aea.std(ddof=1),
        "AEA_minus_AMG315": aea.mean() - amg.mean(),
        "Hedges_g_AEA_minus_AMG315": hedges_g(amg, aea),
        "exact_unpaired_permutation_p_two_sided": exact_permutation_p(amg, aea),
        "inference_note": "exploratory; three independent replicas per ligand",
    })

ligand_summary = pd.DataFrame(ligand_rows)
effects = pd.DataFrame(effect_rows)
finite(ligand_summary, "ligand summary")
ligand_summary.to_csv(OUT / "AMG315_AEA_80ns_ligand_summary.tsv", sep="\t", index=False)
effects.to_csv(OUT / "AMG315_AEA_80ns_replica_effect_sizes.tsv", sep="\t", index=False)

# Complete the union of contacted residues with explicit zero occupancy.
residue_ids = (
    contacts_observed[["residue_name", "residue_number"]]
    .drop_duplicates()
    .sort_values(["residue_number", "residue_name"])
)
grid = pd.MultiIndex.from_product(
    [["AMG315", "AEA"], ["R1", "R2", "R3"]],
    names=["ligand", "replica"],
).to_frame(index=False).merge(residue_ids, how="cross")
contacts = grid.merge(
    contacts_observed,
    on=["ligand", "replica", "residue_name", "residue_number"],
    how="left",
)
contacts[["frames_contacted", "occupancy_percent"]] = contacts[
    ["frames_contacted", "occupancy_percent"]
].fillna(0.0)
contacts["frames_contacted"] = contacts["frames_contacted"].astype(int)
contacts.to_csv(OUT / "AMG315_AEA_80ns_contact_occupancy_complete.tsv", sep="\t", index=False)

contact_summary = (
    contacts.groupby(["ligand", "residue_name", "residue_number"], as_index=False)
    .agg(
        mean_occupancy_percent=("occupancy_percent", "mean"),
        SD_occupancy_percent=("occupancy_percent", "std"),
        replicas_contacted=("frames_contacted", lambda x: int(np.sum(x > 0))),
    )
)
pivot = contact_summary.pivot(
    index=["residue_name", "residue_number"],
    columns="ligand",
    values=["mean_occupancy_percent", "SD_occupancy_percent", "replicas_contacted"],
).reset_index()
pivot.columns = [
    "_".join(str(part) for part in column if str(part))
    if isinstance(column, tuple) else column
    for column in pivot.columns
]
pivot["AEA_minus_AMG315_mean_occupancy_percent"] = (
    pivot["mean_occupancy_percent_AEA"]
    - pivot["mean_occupancy_percent_AMG315"]
)
pivot.to_csv(OUT / "AMG315_AEA_80ns_contact_comparison.tsv", sep="\t", index=False)

state_summary.to_csv(OUT / "AMG315_AEA_80ns_pose_state_summary.tsv", sep="\t", index=False)

state_replica_rows = []
for (ligand, replica), group in states.groupby(["ligand", "replica"], sort=True):
    counts = group["state"].value_counts()
    trans = transitions[(transitions["ligand"] == ligand) & (transitions["replica"] == replica)]
    switches = int(trans.loc[trans["from_state"] != trans["to_state"], "transition_count"].sum())
    state_replica_rows.append({
        "ligand": ligand,
        "replica": replica,
        "frames": len(group),
        "state1_frames": int(counts.get(1, 0)),
        "state2_frames": int(counts.get(2, 0)),
        "state1_occupancy_percent": 100.0 * counts.get(1, 0) / len(group),
        "state2_occupancy_percent": 100.0 * counts.get(2, 0) / len(group),
        "state_switch_events": switches,
        "state_labels_are_ligand_specific": True,
    })
state_replica = pd.DataFrame(state_replica_rows)
assert len(state_replica) == 6
state_replica.to_csv(OUT / "AMG315_AEA_80ns_pose_state_replica_dynamics.tsv", sep="\t", index=False)

print("=" * 78)
print("AMG315–AEA CB1 THREE-REPLICA 80-NS HARMONIZED COMPARISON")
print("=" * 78)
print()
print("LIGAND SUMMARY")
print(ligand_summary.to_string(index=False))
print()
print("POSE-STATE REPLICA DYNAMICS")
print(state_replica.to_string(index=False))
print()
print("Frames:", len(frames))
print("Replica summaries:", len(replica_summary))
print("Ten-ns interval summaries:", len(interval_summary))
print("Complete contact records:", len(contacts))
print("Contact comparison residues:", len(pivot))
print("State labels are independent ligand-specific model labels.")
print("Permutation tests are exploratory with n=3 replicas per ligand.")
print("AMG315–AEA 80-NS HARMONIZED COMPARISON: COMPLETE")
