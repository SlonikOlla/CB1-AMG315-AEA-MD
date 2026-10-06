from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(
    "results/AMG315_CB1_80ns_focused_candidate_ensemble_comparison"
)
OUT.mkdir(parents=True, exist_ok=False)

CANDIDATES = Path(
    "results/AMG315_CB1_pose_states_80ns_expanded/"
    "AMG315_CB1_80ns_expanded_candidate_partitions.tsv"
)

QC80 = Path("results/AMG315_CB1_production_QC_80ns")

FROZEN70TO80 = Path(
    "results/AMG315_CB1_frozen_pose_distance_calibration_80ns/"
    "AMG315_CB1_70to80ns_calibrated_pose_state_frames.tsv"
)

MIN_BLOCK_FRAMES = 20

ENSEMBLE_ORDER = [
    "R2_k5_state1_70to75ns",
    "R2_k5_state1_75to80ns",
    "R3_k6_state4_70to75ns",
    "R3_k6_state2_75to80ns",
]

COMPARISONS = [
    (
        "R2_k5_state1_late_vs_early",
        "R2_k5_state1_70to75ns",
        "R2_k5_state1_75to80ns",
    ),
    (
        "R3_k6_late_state2_vs_early_state4",
        "R3_k6_state4_70to75ns",
        "R3_k6_state2_75to80ns",
    ),
]


LINEAR_FEATURES = [
    "stable_core_CA_RMSD_A",
    "TM_core_CA_RMSD_A",
    "all_protein_CA_RMSD_A",
    "protein_heavy_RMSD_A",
    "AMG315_pose_RMSD_A",
    "AMG315_centroid_displacement_A",
    "AMG315_internal_RMSD_A",
    "minimum_AMG315_CB1_contact_A",
    "box_volume_nm3",
    "distance_to_state_1",
    "distance_to_state_2",
    "distance_to_assigned_state",
    "assigned_state_training_percentile",
    "centroid_distance_ratio",
    "centroid_distance_margin",
    "HIS178_contact_A",
    "THR197_contact_A",
    "TRP279_contact_A",
    "MET363_contact_A",
    "SER383_contact_A",
]

CIRCULAR_FEATURES = [
    "PHE200_chi1_deg",
    "TRP356_chi2_deg",
]

CONTACT_FEATURES = [
    "HIS178_contact_A",
    "THR197_contact_A",
    "TRP279_contact_A",
    "MET363_contact_A",
    "SER383_contact_A",
]

CONTACT_THRESHOLDS_A = [3.5, 4.0, 4.5, 5.0]


def circular_summary(values):
    values = np.asarray(values, dtype=float)
    radians = np.deg2rad(values)

    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))

    mean_angle = float(
        np.rad2deg(np.arctan2(mean_sin, mean_cos))
    )

    resultant = float(
        np.hypot(mean_sin, mean_cos)
    )

    mean_angle = (
        (mean_angle + 180.0) % 360.0
    ) - 180.0

    return mean_angle, resultant


def circular_difference(second, first):
    return float(
        ((second - first) + 180.0) % 360.0 -
        180.0
    )


def hedges_g(second, first):
    second = np.asarray(second, dtype=float)
    first = np.asarray(first, dtype=float)

    n2 = len(second)
    n1 = len(first)

    if n1 < 2 or n2 < 2:
        return np.nan

    pooled_variance = (
        (n2 - 1) * np.var(second, ddof=1) +
        (n1 - 1) * np.var(first, ddof=1)
    ) / (n1 + n2 - 2)

    if pooled_variance <= 0.0:
        return 0.0

    cohen_d = (
        np.mean(second) -
        np.mean(first)
    ) / np.sqrt(pooled_variance)

    correction = (
        1.0 -
        3.0 / (4.0 * (n1 + n2) - 9.0)
    )

    return float(correction * cohen_d)


candidates = pd.read_csv(CANDIDATES, sep="\t")
frozen = pd.read_csv(FROZEN70TO80, sep="\t")

if "states_in_model" in candidates.columns:
    model_column = "states_in_model"
elif "candidate_states" in candidates.columns:
    model_column = "candidate_states"
else:
    raise RuntimeError("Candidate model-count column is missing")

assert len(candidates) == 168000
assert len(frozen) == 3000

candidate_tables = []

for replica, candidate_k in (("R2", 5), ("R3", 6)):
    table = candidates[
        (candidates.replica == replica) &
        (
            candidates[model_column].astype(int) ==
            candidate_k
        ) &
        (candidates.time_ps > 70000.0) &
        (candidates.time_ps <= 80000.0)
    ].copy()

    assert len(table) == 1000

    table = table.rename(columns={
        model_column: "focused_candidate_k",
        "candidate_state": "focused_candidate_state",
    })

    candidate_tables.append(table)

candidate_focus = pd.concat(
    candidate_tables,
    ignore_index=True,
)

assert len(candidate_focus) == 2000

qc_tables = []

for replica in ("R2", "R3"):
    table = pd.read_csv(
        QC80 /
        f"AMG315_CB1_80ns_{replica}_frame_metrics.tsv",
        sep="\t",
    )

    assert len(table) == 8000

    table = table.rename(
        columns={"frame": "replica_frame"}
    )

    table = table[
        (table.time_ps > 70000.0) &
        (table.time_ps <= 80000.0)
    ].copy()

    assert len(table) == 1000

    qc_tables.append(table)

qc = pd.concat(qc_tables, ignore_index=True)
assert len(qc) == 2000

frozen_focus = frozen[
    frozen.replica.isin(["R2", "R3"])
].copy()

assert len(frozen_focus) == 2000

frozen_focus = frozen_focus.rename(columns={
    "global_frame": "frozen_global_frame",
    "state": "frozen_state",
})

identity = [
    "replica",
    "replica_frame",
    "time_ps",
]

for table, name in (
    (candidate_focus, "candidate"),
    (qc, "QC"),
    (frozen_focus, "frozen"),
):
    if table.duplicated(identity).any():
        raise RuntimeError(
            f"Duplicate {name} frame identities"
        )

merged = candidate_focus.merge(
    qc,
    on=identity,
    how="left",
    validate="one_to_one",
)

merged = merged.merge(
    frozen_focus,
    on=identity,
    how="left",
    validate="one_to_one",
)

assert len(merged) == 2000
assert merged.frozen_state.notna().all()


required_numeric = (
    LINEAR_FEATURES +
    CIRCULAR_FEATURES +
    [
        "focused_candidate_k",
        "focused_candidate_state",
        "frozen_state",
    ]
)

if not np.isfinite(
    merged[required_numeric].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Merged table contains non-finite values"
    )

merged["one_ns_block"] = np.floor(
    (merged.time_ps.to_numpy(dtype=float) - 10.0) /
    1000.0
).astype(int)

merged["ensemble"] = "outside_primary_comparison"

r2_early = (
    (merged.replica == "R2") &
    (merged.focused_candidate_k.astype(int) == 5) &
    (merged.focused_candidate_state.astype(int) == 1) &
    (merged.time_ps > 70000.0) &
    (merged.time_ps <= 75000.0)
)

r2_late = (
    (merged.replica == "R2") &
    (merged.focused_candidate_k.astype(int) == 5) &
    (merged.focused_candidate_state.astype(int) == 1) &
    (merged.time_ps > 75000.0) &
    (merged.time_ps <= 80000.0)
)

r3_early = (
    (merged.replica == "R3") &
    (merged.focused_candidate_k.astype(int) == 6) &
    (merged.focused_candidate_state.astype(int) == 4) &
    (merged.time_ps > 70000.0) &
    (merged.time_ps <= 75000.0)
)

r3_late = (
    (merged.replica == "R3") &
    (merged.focused_candidate_k.astype(int) == 6) &
    (merged.focused_candidate_state.astype(int) == 2) &
    (merged.time_ps > 75000.0) &
    (merged.time_ps <= 80000.0)
)

merged.loc[
    r2_early,
    "ensemble",
] = ENSEMBLE_ORDER[0]

merged.loc[
    r2_late,
    "ensemble",
] = ENSEMBLE_ORDER[1]

merged.loc[
    r3_early,
    "ensemble",
] = ENSEMBLE_ORDER[2]

merged.loc[
    r3_late,
    "ensemble",
] = ENSEMBLE_ORDER[3]

ensembles = {
    name: merged[merged.ensemble == name].copy()
    for name in ENSEMBLE_ORDER
}

expected_counts = {
    ENSEMBLE_ORDER[0]: 495,
    ENSEMBLE_ORDER[1]: 484,
    ENSEMBLE_ORDER[2]: 305,
    ENSEMBLE_ORDER[3]: 500,
}

for name, expected in expected_counts.items():
    assert len(ensembles[name]) == expected


merged.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_complete_frame_merge.tsv",
    sep="\t",
    index=False,
)

manifest_rows = []

for name, table in ensembles.items():
    manifest_rows.append({
        "ensemble": name,
        "replica": str(table.replica.iloc[0]),
        "candidate_k":
            int(table.focused_candidate_k.iloc[0]),
        "candidate_state":
            int(table.focused_candidate_state.iloc[0]),
        "frames": int(len(table)),
        "first_replica_frame":
            int(table.replica_frame.min()),
        "last_replica_frame":
            int(table.replica_frame.max()),
        "first_time_ps":
            float(table.time_ps.min()),
        "last_time_ps":
            float(table.time_ps.max()),
        "one_ns_blocks":
            int(table.one_ns_block.nunique()),
    })

manifest = pd.DataFrame(manifest_rows)

manifest.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_ensemble_manifest.tsv",
    sep="\t",
    index=False,
)

linear_rows = []

for name, table in ensembles.items():
    for feature in LINEAR_FEATURES:
        values = table[feature].to_numpy(dtype=float)
        q1, median, q3 = np.quantile(
            values,
            [0.25, 0.50, 0.75],
        )

        sd = float(np.std(values, ddof=1))

        linear_rows.append({
            "ensemble": name,
            "feature": feature,
            "frames": int(len(values)),
            "mean": float(np.mean(values)),
            "standard_deviation": sd,
            "standard_error":
                float(sd / np.sqrt(len(values))),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "interquartile_range":
                float(q3 - q1),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        })

linear_summary = pd.DataFrame(linear_rows)

linear_summary.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_linear_feature_summary.tsv",
    sep="\t",
    index=False,
)

circular_rows = []

for name, table in ensembles.items():
    for feature in CIRCULAR_FEATURES:
        mean_angle, resultant = circular_summary(
            table[feature].to_numpy(dtype=float)
        )

        circular_rows.append({
            "ensemble": name,
            "feature": feature,
            "frames": int(len(table)),
            "circular_mean_deg": mean_angle,
            "resultant_vector_length": resultant,
            "circular_variance":
                float(1.0 - resultant),
        })

circular_table = pd.DataFrame(circular_rows)

circular_table.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_circular_feature_summary.tsv",
    sep="\t",
    index=False,
)

difference_rows = []

for comparison, first_name, second_name in COMPARISONS:
    for feature in CIRCULAR_FEATURES:
        first = circular_table[
            (circular_table.ensemble == first_name) &
            (circular_table.feature == feature)
        ].iloc[0]

        second = circular_table[
            (circular_table.ensemble == second_name) &
            (circular_table.feature == feature)
        ].iloc[0]

        difference_rows.append({
            "comparison": comparison,
            "feature": feature,
            "first_ensemble": first_name,
            "second_ensemble": second_name,
            "first_circular_mean_deg":
                float(first.circular_mean_deg),
            "second_circular_mean_deg":
                float(second.circular_mean_deg),
            "second_minus_first_deg":
                circular_difference(
                    float(second.circular_mean_deg),
                    float(first.circular_mean_deg),
                ),
        })

circular_differences = pd.DataFrame(
    difference_rows
)

circular_differences.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_circular_mean_differences.tsv",
    sep="\t",
    index=False,
)

contact_rows = []

for name, table in ensembles.items():
    for feature in CONTACT_FEATURES:
        values = table[feature].to_numpy(dtype=float)

        for threshold in CONTACT_THRESHOLDS_A:
            contacted = values <= threshold

            contact_rows.append({
                "ensemble": name,
                "feature": feature,
                "threshold_A": float(threshold),
                "frames": int(len(values)),
                "frames_at_or_below_threshold":
                    int(np.sum(contacted)),
                "occupancy_percent":
                    float(100.0 * np.mean(contacted)),
            })

contact_occupancy = pd.DataFrame(contact_rows)

contact_occupancy.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_contact_threshold_occupancy.tsv",
    sep="\t",
    index=False,
)

block_rows = []

for name, table in ensembles.items():
    for block, block_table in table.groupby(
        "one_ns_block",
        sort=True,
    ):
        row = {
            "ensemble": name,
            "one_ns_block": int(block),
            "block_start_ps":
                float(block_table.time_ps.min()),
            "block_end_ps":
                float(block_table.time_ps.max()),
            "frames": int(len(block_table)),
            "eligible_for_effect_size":
                bool(len(block_table) >= MIN_BLOCK_FRAMES),
        }

        for feature in LINEAR_FEATURES:
            row[f"{feature}_mean"] = float(
                block_table[feature].mean()
            )

        for feature in CIRCULAR_FEATURES:
            mean_angle, resultant = circular_summary(
                block_table[feature].to_numpy(dtype=float)
            )

            row[
                f"{feature}_circular_mean_deg"
            ] = mean_angle

            row[
                f"{feature}_resultant_vector_length"
            ] = resultant

        block_rows.append(row)

block_summary = pd.DataFrame(block_rows)

block_summary.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_one_ns_block_summary.tsv",
    sep="\t",
    index=False,
)

effect_rows = []

for comparison, first_name, second_name in COMPARISONS:
    first_blocks = block_summary[
        (block_summary.ensemble == first_name) &
        block_summary.eligible_for_effect_size
    ]

    second_blocks = block_summary[
        (block_summary.ensemble == second_name) &
        block_summary.eligible_for_effect_size
    ]

    if len(first_blocks) < 2 or len(second_blocks) < 2:
        raise RuntimeError(
            f"Too few eligible blocks for {comparison}"
        )

    for feature in LINEAR_FEATURES:
        first_values = first_blocks[
            f"{feature}_mean"
        ].to_numpy(dtype=float)

        second_values = second_blocks[
            f"{feature}_mean"
        ].to_numpy(dtype=float)

        effect_rows.append({
            "comparison": comparison,
            "feature": feature,
            "first_ensemble": first_name,
            "second_ensemble": second_name,
            "first_eligible_blocks":
                int(len(first_values)),
            "second_eligible_blocks":
                int(len(second_values)),
            "first_block_mean":
                float(np.mean(first_values)),
            "first_block_standard_deviation":
                float(np.std(first_values, ddof=1)),
            "second_block_mean":
                float(np.mean(second_values)),
            "second_block_standard_deviation":
                float(np.std(second_values, ddof=1)),
            "second_minus_first":
                float(
                    np.mean(second_values) -
                    np.mean(first_values)
                ),
            "hedges_g_second_minus_first":
                hedges_g(second_values, first_values),
            "inferential_unit":
                "one_ns_temporal_block",
            "frame_level_p_value":
                "not_calculated",
        })

block_effects = pd.DataFrame(effect_rows)

block_effects.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_block_effect_sizes.tsv",
    sep="\t",
    index=False,
)

frozen_rows = []

for name, table in ensembles.items():
    categories = sorted(
        table.distance_calibration_category
        .astype(str)
        .unique()
    )

    for category in categories:
        count = int(np.sum(
            table.distance_calibration_category
            .astype(str)
            .to_numpy() == category
        ))

        frozen_rows.append({
            "ensemble": name,
            "distance_calibration_category":
                category,
            "frames": int(len(table)),
            "category_frames": count,
            "occupancy_percent":
                float(100.0 * count / len(table)),
            "mean_distance_to_assigned_state":
                float(
                    table.distance_to_assigned_state.mean()
                ),
            "standard_deviation_distance_to_assigned_state":
                float(
                    table.distance_to_assigned_state.std(ddof=1)
                ),
            "median_assigned_state_training_percentile":
                float(
                    table.assigned_state_training_percentile
                    .median()
                ),
            "mean_centroid_distance_ratio":
                float(
                    table.centroid_distance_ratio.mean()
                ),
        })

frozen_summary = pd.DataFrame(frozen_rows)

frozen_summary.to_csv(
    OUT /
    "AMG315_CB1_80ns_focused_frozen_distance_summary.tsv",
    sep="\t",
    index=False,
)

required_outputs = [
    "AMG315_CB1_80ns_focused_complete_frame_merge.tsv",
    "AMG315_CB1_80ns_focused_ensemble_manifest.tsv",
    "AMG315_CB1_80ns_focused_linear_feature_summary.tsv",
    "AMG315_CB1_80ns_focused_circular_feature_summary.tsv",
    "AMG315_CB1_80ns_focused_circular_mean_differences.tsv",
    "AMG315_CB1_80ns_focused_contact_threshold_occupancy.tsv",
    "AMG315_CB1_80ns_focused_one_ns_block_summary.tsv",
    "AMG315_CB1_80ns_focused_block_effect_sizes.tsv",
    "AMG315_CB1_80ns_focused_frozen_distance_summary.tsv",
]

for filename in required_outputs:
    path = OUT / filename

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Missing or empty output: {path}"
        )

selected_effects = block_effects[
    block_effects.feature.isin([
        "AMG315_pose_RMSD_A",
        "AMG315_centroid_displacement_A",
        "AMG315_internal_RMSD_A",
        "HIS178_contact_A",
        "THR197_contact_A",
        "TRP279_contact_A",
        "MET363_contact_A",
        "SER383_contact_A",
        "distance_to_assigned_state",
        "centroid_distance_ratio",
    ])
]

print("=" * 78)
print("AMG315 80-NS FOCUSED CANDIDATE-ENSEMBLE COMPARISON")
print("=" * 78)

print()
print("ENSEMBLE MANIFEST")
print(manifest.to_string(index=False))

print()
print("SELECTED BLOCK-LEVEL EFFECTS")
print(
    selected_effects[[
        "comparison",
        "feature",
        "second_minus_first",
        "hedges_g_second_minus_first",
        "first_eligible_blocks",
        "second_eligible_blocks",
    ]].to_string(index=False)
)

print()
print("Complete 2,000-frame final-interval focused merge: PASS")
print("Exact R2 k5 State-1 70-to-75-ns definition: PASS")
print("Exact R2 k5 State-1 75-to-80-ns definition: PASS")
print("Exact R3 k6 early-State-4/late-State-2 definition: PASS")
print("Linear summaries with SD, SE, median and IQR: PASS")
print("Circular ensemble summaries: PASS")
print("Contact-threshold occupancies: PASS")
print("One-nanosecond temporal blocks: PASS")
print("Minimum 20-frame block-effect safeguard: PASS")
print("Block-level standardized effect sizes: PASS")
print("No pseudoreplicated frame-level P values: PASS")
print("Frozen-distance summaries: PASS")
print()
print("AMG315 80-NS FOCUSED CANDIDATE-ENSEMBLE COMPARISON: COMPLETE")
