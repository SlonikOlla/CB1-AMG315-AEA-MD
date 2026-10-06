from pathlib import Path
import numpy as np
import pandas as pd

STATE_DIR = Path("results/AEA_CB1_pose_states_80ns_expanded")
QC_DIR = Path("results/AEA_CB1_80ns_QC")
OUT = Path("results/AEA_CB1_80ns_focused_pose_state_comparison")
KEY = ["replica", "replica_frame", "time_ps"]

SPECS = {
    "R1_state1_pre_30to40ns": ("R1", 1, "time", 30000, 40000, 1000),
    "R1_state2_excursion_48to52ns": ("R1", 2, "time", 48000, 52000, 391),
    "R1_state1_returned_70to80ns": ("R1", 1, "time", 70000, 80000, 1000),
    "R2_state1_core_run": ("R2", 1, "frame", 673, 1333, 661),
    "R2_state2_late_70to80ns": ("R2", 2, "time", 70000, 80000, 1000),
    "R3_state1_pre_late_70to78ns": ("R3", 1, "time", 70000, 78000, 797),
    "R3_state2_late_78.97to80ns": ("R3", 2, "frame", 7897, 8000, 103),
}

COMPARISONS = [
    ("R1_excursion_vs_pre", "R1_state1_pre_30to40ns", "R1_state2_excursion_48to52ns"),
    ("R1_returned_vs_excursion", "R1_state2_excursion_48to52ns", "R1_state1_returned_70to80ns"),
    ("R1_returned_vs_pre", "R1_state1_pre_30to40ns", "R1_state1_returned_70to80ns"),
    ("R2_late_state2_vs_early_state1", "R2_state1_core_run", "R2_state2_late_70to80ns"),
    ("R3_late_state2_vs_pre_late_state1", "R3_state1_pre_late_70to78ns", "R3_state2_late_78.97to80ns"),
]


def normalize(table, replica=None):
    table = table.copy()
    if "replica" not in table and replica is not None:
        table["replica"] = replica
    if "replica_frame" not in table and "frame" in table:
        table = table.rename(columns={"frame": "replica_frame"})
    missing = set(KEY).difference(table.columns)
    if missing:
        raise RuntimeError(f"Missing identity columns: {sorted(missing)}")
    table["replica"] = table.replica.astype(str)
    table["replica_frame"] = table.replica_frame.astype(int)
    table["time_ps"] = table.time_ps.astype(float)
    return table


def hedges_g(second, first):
    second, first = np.asarray(second, float), np.asarray(first, float)
    n2, n1 = len(second), len(first)
    if min(n1, n2) < 2:
        return np.nan
    pooled = ((n2 - 1) * np.var(second, ddof=1) + (n1 - 1) * np.var(first, ddof=1)) / (n1 + n2 - 2)
    if pooled <= 0:
        return 0.0
    d = (second.mean() - first.mean()) / np.sqrt(pooled)
    return float((1 - 3 / (4 * (n1 + n2) - 9)) * d)


print("=" * 78)
print("AEA-CB1 80-NS FOCUSED POSE-STATE ENSEMBLE COMPARISON")
print("=" * 78)
OUT.mkdir(parents=True, exist_ok=False)

frames = normalize(pd.read_csv(STATE_DIR / "AEA_CB1_80ns_expanded_pose_state_frames.tsv", sep="\t"))
assert len(frames) == 24000
assert not frames.duplicated(KEY).any()

qc_parts = []
for replica in ("R1", "R2", "R3"):
    q = normalize(pd.read_csv(QC_DIR / f"AEA_CB1_80ns_{replica}_frame_metrics.tsv", sep="\t"), replica)
    assert len(q) == 8000
    qc_parts.append(q)
qc = pd.concat(qc_parts, ignore_index=True)
overlap = (set(frames.columns) & set(qc.columns)) - set(KEY)
qc = qc.drop(columns=sorted(overlap))
merged = frames.merge(qc, on=KEY, how="left", validate="one_to_one")

hb = normalize(pd.read_csv(QC_DIR / "AEA_80ns_signature_hbond_frame_metrics.tsv", sep="\t"))
assert len(hb) == 144000 and hb.interaction.nunique() == 6
wide = hb.pivot(index=KEY, columns="interaction", values="hydrogen_bond_present").reset_index()
wide.columns.name = None
hcols0 = sorted(set(wide.columns) - set(KEY))
wide = wide.rename(columns={c: f"HBOND_{c}" for c in hcols0})
hcols = [f"HBOND_{c}" for c in hcols0]
merged = merged.merge(wide, on=KEY, how="left", validate="one_to_one")
assert len(merged) == 24000 and merged[hcols].notna().all().all()

ensembles, manifest_rows, focused = {}, [], []
for name, (replica, state, mode, lower, upper, expected) in SPECS.items():
    t = merged[(merged.replica == replica) & (merged.state == state)].copy()
    if mode == "time":
        t = t[(t.time_ps > lower) & (t.time_ps <= upper)]
    else:
        t = t[(t.replica_frame >= lower) & (t.replica_frame <= upper)]
    t = t.sort_values("replica_frame")
    assert len(t) == expected, (name, expected, len(t))
    v = t.replica_frame.to_numpy(int)
    manifest_rows.append({
        "ensemble": name, "replica": replica, "state": state, "frames": len(t),
        "first_replica_frame": int(v[0]), "last_replica_frame": int(v[-1]),
        "first_time_ps": float(t.time_ps.iloc[0]), "last_time_ps": float(t.time_ps.iloc[-1]),
        "contiguous_runs": 1 + int(np.sum(np.diff(v) != 1)),
        "missing_internal_frames": int(v[-1] - v[0] + 1 - len(v)),
    })
    t.insert(0, "ensemble", name)
    t["half_ns_block"] = np.floor((t.time_ps - 1) / 500).astype(int)
    ensembles[name] = t
    focused.append(t)

manifest = pd.DataFrame(manifest_rows)
focus = pd.concat(focused, ignore_index=True)
assert len(manifest) == 7 and len(focus) == 4952
manifest.to_csv(OUT / "AEA_CB1_80ns_focused_ensemble_manifest.tsv", sep="\t", index=False)
focus.to_csv(OUT / "AEA_CB1_80ns_focused_complete_frame_merge.tsv", sep="\t", index=False)

excluded = {"global_frame", "replica_frame", "time_ps", "state", "half_ns_block"}
numeric = [c for c in focus.select_dtypes(include=[np.number, "bool"]).columns if c not in excluded]
linear = [c for c in numeric if c not in hcols and not c.endswith("_deg")]
circular = [c for c in numeric if c.endswith("_deg")]

rows = []
for name, t in ensembles.items():
    for feature in linear:
        x = t[feature].to_numpy(float)
        q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
        sd = float(np.std(x, ddof=1))
        rows.append({"ensemble": name, "feature": feature, "frames": len(x), "mean": x.mean(),
                     "standard_deviation": sd, "standard_error": sd / np.sqrt(len(x)),
                     "median": med, "q1": q1, "q3": q3, "minimum": x.min(), "maximum": x.max()})
linear_summary = pd.DataFrame(rows)
linear_summary.to_csv(OUT / "AEA_CB1_80ns_focused_linear_feature_summary.tsv", sep="\t", index=False)

hrows = []
for name, t in ensembles.items():
    for feature in hcols:
        x = t[feature].astype(int).to_numpy()
        hrows.append({"ensemble": name, "interaction": feature.removeprefix("HBOND_"),
                      "frames": len(x), "frames_present": int(x.sum()),
                      "occupancy_percent": 100 * x.mean()})
hocc = pd.DataFrame(hrows)
hocc.to_csv(OUT / "AEA_CB1_80ns_focused_hbond_occupancy.tsv", sep="\t", index=False)

contact_features = [c for c in linear if c.endswith("_contact_A") or c == "minimum_AEA_CB1_contact_A"]
crows = []
for name, t in ensembles.items():
    for feature in contact_features:
        x = t[feature].to_numpy(float)
        for threshold in (3.5, 4.0, 5.0):
            present = x <= threshold
            crows.append({"ensemble": name, "feature": feature, "threshold_A": threshold,
                          "frames": len(x), "frames_at_or_below_threshold": int(present.sum()),
                          "occupancy_percent": 100 * present.mean()})
pd.DataFrame(crows).to_csv(OUT / "AEA_CB1_80ns_focused_contact_threshold_occupancy.tsv", sep="\t", index=False)

brows = []
for name, t in ensembles.items():
    for block, b in t.groupby("half_ns_block", sort=True):
        row = {"ensemble": name, "half_ns_block": int(block), "block_start_ps": b.time_ps.min(),
               "block_end_ps": b.time_ps.max(), "frames": len(b), "eligible_for_effect_size": len(b) >= 10}
        for feature in linear:
            row[f"{feature}_mean"] = b[feature].mean()
        for feature in hcols:
            row[f"{feature}_occupancy_percent"] = 100 * b[feature].mean()
        brows.append(row)
blocks = pd.DataFrame(brows)
blocks.to_csv(OUT / "AEA_CB1_80ns_focused_half_ns_block_summary.tsv", sep="\t", index=False)

erows = []
for comparison, first, second in COMPARISONS:
    a = blocks[(blocks.ensemble == first) & blocks.eligible_for_effect_size]
    b = blocks[(blocks.ensemble == second) & blocks.eligible_for_effect_size]
    assert len(a) >= 2 and len(b) >= 2, (comparison, len(a), len(b))
    for feature in linear + hcols:
        column = f"{feature}_mean" if feature in linear else f"{feature}_occupancy_percent"
        x, y = a[column].to_numpy(float), b[column].to_numpy(float)
        erows.append({"comparison": comparison, "feature": feature, "first_ensemble": first,
                      "second_ensemble": second, "first_eligible_blocks": len(x),
                      "second_eligible_blocks": len(y), "first_block_mean": x.mean(),
                      "first_block_standard_deviation": np.std(x, ddof=1), "second_block_mean": y.mean(),
                      "second_block_standard_deviation": np.std(y, ddof=1),
                      "second_minus_first": y.mean() - x.mean(),
                      "hedges_g_second_minus_first": hedges_g(y, x),
                      "inferential_unit": "half_ns_temporal_block", "frame_level_p_value": "not_calculated"})
effects = pd.DataFrame(erows)
effects.to_csv(OUT / "AEA_CB1_80ns_focused_block_effect_sizes.tsv", sep="\t", index=False)

comparison_manifest = pd.DataFrame([
    {"comparison": c, "first_ensemble": a, "second_ensemble": b,
     "first_frames": len(ensembles[a]), "second_frames": len(ensembles[b])}
    for c, a, b in COMPARISONS
])
comparison_manifest.to_csv(OUT / "AEA_CB1_80ns_focused_comparison_manifest.tsv", sep="\t", index=False)

for label, table in {"manifest": manifest, "merge": focus, "linear": linear_summary,
                     "hbond": hocc, "blocks": blocks, "effects": effects,
                     "comparisons": comparison_manifest}.items():
    assert len(table), label
    numbers = table.select_dtypes(include=[np.number])
    assert np.isfinite(numbers.to_numpy(float)).all(), label

print("\nENSEMBLE MANIFEST")
print(manifest.to_string(index=False))
print("\nH-BOND OCCUPANCY")
print(hocc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nFocused frames: {len(focus):,}")
print(f"Linear features: {len(linear)}")
print(f"H-bond interactions: {len(hcols)}")
print(f"Half-ns block records: {len(blocks)}")
print(f"Block-effect records: {len(effects)}")
print("Seven exact focused ensembles: PASS")
print("Exact 4,952-frame focused record: PASS")
print("Five prespecified comparisons: PASS")
print("All stored numerical values finite: PASS")
print("AEA 80-NS FOCUSED POSE-STATE ENSEMBLE COMPARISON: COMPLETE")
