from itertools import combinations
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd


OUT = Path("results/AMG315_AEA_CB1_80ns_microswitch_comparison")
OUT.mkdir(parents=True, exist_ok=True)

FRAME_INTERVAL_PS = 10.0
SEGMENT_FRAMES = 1000

SYSTEMS = {
    "AMG315": {
        "base": Path("systems/AMG315_CB1"),
        "references": {
            "R1": "equil/AMG315_CB1_NPT_unrestrained_final.pdb",
            "R2": "replicas/R2/stabilization/AMG315_R2_unrestrained_500ps_final.pdb",
            "R3": "replicas/R3/stabilization/AMG315_R3_unrestrained_500ps_final.pdb",
        },
        "segments": {
            "R1": [
                "replicas/R1/AMG315_CB1_prod_5ns_R1.dcd",
                "replicas/R1/AMG315_CB1_prod_5to10ns_R1.dcd",
                "replicas/R1/AMG315_CB1_prod_10to20ns_R1.dcd",
                "replicas/R1/AMG315_CB1_prod_20to30ns_R1.dcd",
                "replicas/R1/AMG315_CB1_prod_30to40ns_R1.dcd",
                "replicas/R1/AMG315_CB1_prod_40to50ns_R1.dcd",
                "replicas/R1/AMG315_CB1_prod_50to60ns_R1.dcd",
                "replicas/R1/AMG315_CB1_prod_60to70ns_R1.dcd",
                "replicas/R1/AMG315_CB1_prod_70to80ns_R1.dcd",
            ],
            "R2": [
                "replicas/R2/AMG315_CB1_prod_5ns_R2.dcd",
                "replicas/R2/AMG315_CB1_prod_5to10ns_R2.dcd",
                "replicas/R2/AMG315_CB1_prod_10to20ns_R2.dcd",
                "replicas/R2/AMG315_CB1_prod_20to30ns_R2.dcd",
                "replicas/R2/AMG315_CB1_prod_30to40ns_R2.dcd",
                "replicas/R2/AMG315_CB1_prod_40to50ns_R2.dcd",
                "replicas/R2/AMG315_CB1_prod_50to60ns_R2.dcd",
                "replicas/R2/AMG315_CB1_prod_60to70ns_R2.dcd",
                "replicas/R2/AMG315_CB1_prod_70to80ns_R2.dcd",
            ],
            "R3": [
                "replicas/R3/AMG315_CB1_prod_5ns_R3.dcd",
                "replicas/R3/AMG315_CB1_prod_5to10ns_R3.dcd",
                "replicas/R3/AMG315_CB1_prod_10to20ns_R3.dcd",
                "replicas/R3/AMG315_CB1_prod_20to30ns_R3.dcd",
                "replicas/R3/AMG315_CB1_prod_30to40ns_R3.dcd",
                "replicas/R3/AMG315_CB1_prod_40to50ns_R3.dcd",
                "replicas/R3/AMG315_CB1_prod_50to60ns_R3.dcd",
                "replicas/R3/AMG315_CB1_prod_60to70ns_R3.dcd",
                "replicas/R3/AMG315_CB1_prod_70to80ns_R3.dcd",
            ],
        },
        "atoms": 42295,
    },
    "AEA": {
        "base": Path("systems/AEA_CB1"),
        "references": {
            "R1": "equil/AEA_CB1_NPT_unrestrained_ext1500ps_final.pdb",
            "R2": "replicas/R2/stabilization/AEA_R2_unrestrained_500ps_final.pdb",
            "R3": "replicas/R3/stabilization/AEA_R3_unrestrained_500ps_final.pdb",
        },
        "segments": {
            "R1": [
                "replicas/R1/AEA_CB1_prod_10ns_R1.dcd",
                "replicas/R1/AEA_CB1_prod_10to20ns_R1.dcd",
                "replicas/R1/AEA_CB1_prod_20to30ns_R1.dcd",
                "replicas/R1/AEA_CB1_prod_30to40ns_R1.dcd",
                "replicas/R1/AEA_CB1_prod_40to50ns_R1.dcd",
                "replicas/R1/AEA_CB1_prod_50to60ns_R1.dcd",
                "replicas/R1/AEA_CB1_prod_60to70ns_R1.dcd",
                "replicas/R1/AEA_CB1_prod_70to80ns_R1.dcd",
            ],
            "R2": [
                "replicas/R2/AEA_CB1_prod_10ns_R2.dcd",
                "replicas/R2/AEA_CB1_prod_10to20ns_R2.dcd",
                "replicas/R2/AEA_CB1_prod_20to30ns_R2.dcd",
                "replicas/R2/AEA_CB1_prod_30to40ns_R2.dcd",
                "replicas/R2/AEA_CB1_prod_40to50ns_R2.dcd",
                "replicas/R2/AEA_CB1_prod_50to60ns_R2.dcd",
                "replicas/R2/AEA_CB1_prod_60to70ns_R2.dcd",
                "replicas/R2/AEA_CB1_prod_70to80ns_R2.dcd",
            ],
            "R3": [
                "replicas/R3/AEA_CB1_prod_10ns_R3.dcd",
                "replicas/R3/AEA_CB1_prod_10to20ns_R3.dcd",
                "replicas/R3/AEA_CB1_prod_20to30ns_R3.dcd",
                "replicas/R3/AEA_CB1_prod_30to40ns_R3.dcd",
                "replicas/R3/AEA_CB1_prod_40to50ns_R3.dcd",
                "replicas/R3/AEA_CB1_prod_50to60ns_R3.dcd",
                "replicas/R3/AEA_CB1_prod_60to70ns_R3.dcd",
                "replicas/R3/AEA_CB1_prod_70to80ns_R3.dcd",
            ],
        },
        "atoms": 42322,
    },
}

LINEAR = [
    "ARG214_ASP338_ionic_lock_min_A",
    "ARG214_CA_ALA342_CA_distance_A",
    "ASN393_CG_TYR397_OH_distance_A",
]
CIRCULAR = ["TRP356_chi2_deg", "TYR397_chi1_deg"]


def protein_residue(topology, resseq, name):
    found = [r for r in topology.residues if r.is_protein and r.resSeq == resseq]
    assert len(found) == 1, (resseq, len(found))
    assert found[0].name == name, (resseq, found[0].name, name)
    return found[0]


def atom(residue, name):
    found = [a for a in residue.atoms if a.name == name]
    assert len(found) == 1, (residue, name, len(found))
    return found[0].index


def selections(topology):
    r214 = protein_residue(topology, 214, "ARG")
    d338 = protein_residue(topology, 338, "ASP")
    a342 = protein_residue(topology, 342, "ALA")
    w356 = protein_residue(topology, 356, "TRP")
    n393 = protein_residue(topology, 393, "ASN")
    y397 = protein_residue(topology, 397, "TYR")
    return {
        "ionic_pairs": np.array([
            [atom(r214, n), atom(d338, o)]
            for n in ("NH1", "NH2") for o in ("OD1", "OD2")
        ], dtype=int),
        "tm3_tm6": np.array([[atom(r214, "CA"), atom(a342, "CA")]], dtype=int),
        "npy": np.array([[atom(n393, "CG"), atom(y397, "OH")]], dtype=int),
        "trp_chi2": np.array([[atom(w356, "CA"), atom(w356, "CB"), atom(w356, "CG"), atom(w356, "CD1")]], dtype=int),
        "tyr_chi1": np.array([[atom(y397, "N"), atom(y397, "CA"), atom(y397, "CB"), atom(y397, "CG")]], dtype=int),
    }


def circular_stats(values):
    radians = np.deg2rad(np.asarray(values, dtype=float))
    z = np.mean(np.exp(1j * radians))
    return float(np.rad2deg(np.angle(z))), float(abs(z))


def wrap_degrees(value):
    return float((value + 180.0) % 360.0 - 180.0)


def summarize(subset):
    row = {"frames": len(subset)}
    for metric in LINEAR:
        values = subset[metric].to_numpy(float)
        row[f"mean_{metric}"] = values.mean()
        row[f"SD_{metric}"] = values.std(ddof=1)
        row[f"median_{metric}"] = np.median(values)
    for metric in CIRCULAR:
        mean, resultant = circular_stats(subset[metric])
        row[f"circular_mean_{metric}"] = mean
        row[f"resultant_length_{metric}"] = resultant
    return row


def hedges_g(first, second):
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    n1, n2 = len(first), len(second)
    pooled = np.sqrt(((n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    d = (second.mean() - first.mean()) / pooled
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(correction * d)


def exact_permutation_p(first, second):
    values = np.concatenate([np.asarray(first, float), np.asarray(second, float)])
    observed = abs(np.mean(second) - np.mean(first))
    differences = []
    for chosen in combinations(range(6), 3):
        chosen = set(chosen)
        group1 = [values[i] for i in range(6) if i in chosen]
        group2 = [values[i] for i in range(6) if i not in chosen]
        differences.append(abs(np.mean(group2) - np.mean(group1)))
    return float(np.mean(np.asarray(differences) >= observed - 1e-12))


print("=" * 78)
print("AMG315–AEA CB1 THREE-REPLICA 80-NS MICROSWITCH COMPARISON")
print("=" * 78)

frame_parts = []
reference_rows = []

for ligand, system in SYSTEMS.items():
    for replica in ("R1", "R2", "R3"):
        reference_path = system["base"] / system["references"][replica]
        reference = md.load(str(reference_path))
        assert reference.n_atoms == system["atoms"]
        select = selections(reference.topology)

        reference_metric = {
            "ligand": ligand,
            "replica": replica,
            "reference_file": str(reference_path),
        }
        ref_ionic = md.compute_distances(reference, select["ionic_pairs"], periodic=False)[0]
        reference_metric[LINEAR[0]] = 10.0 * ref_ionic.min()
        reference_metric[LINEAR[1]] = 10.0 * md.compute_distances(reference, select["tm3_tm6"], periodic=False)[0, 0]
        reference_metric[LINEAR[2]] = 10.0 * md.compute_distances(reference, select["npy"], periodic=False)[0, 0]
        reference_metric[CIRCULAR[0]] = np.rad2deg(md.compute_dihedrals(reference, select["trp_chi2"], periodic=False)[0, 0])
        reference_metric[CIRCULAR[1]] = np.rad2deg(md.compute_dihedrals(reference, select["tyr_chi1"], periodic=False)[0, 0])
        reference_rows.append(reference_metric)

        replica_parts = []
        offset = 0
        for segment_name in system["segments"][replica]:
            segment_path = system["base"] / segment_name
            trajectory = md.load(str(segment_path), top=str(reference_path))
            segment_frames = trajectory.n_frames
            if ligand == "AMG315" and offset < 1000:
                assert segment_frames == 500
            else:
                assert segment_frames == SEGMENT_FRAMES
            assert trajectory.n_atoms == system["atoms"]

            ionic = md.compute_distances(trajectory, select["ionic_pairs"], periodic=False)
            tm_distance = md.compute_distances(trajectory, select["tm3_tm6"], periodic=False)[:, 0]
            npy_distance = md.compute_distances(trajectory, select["npy"], periodic=False)[:, 0]
            trp = md.compute_dihedrals(trajectory, select["trp_chi2"], periodic=False)[:, 0]
            tyr = md.compute_dihedrals(trajectory, select["tyr_chi1"], periodic=False)[:, 0]

            frames = np.arange(offset + 1, offset + segment_frames + 1)
            replica_parts.append(pd.DataFrame({
                "ligand": ligand,
                "replica": replica,
                "frame": frames,
                "time_ps": frames * FRAME_INTERVAL_PS,
                LINEAR[0]: 10.0 * ionic.min(axis=1),
                LINEAR[1]: 10.0 * tm_distance,
                LINEAR[2]: 10.0 * npy_distance,
                CIRCULAR[0]: np.rad2deg(trp),
                CIRCULAR[1]: np.rad2deg(tyr),
            }))
            offset += segment_frames
            del trajectory

        replica_table = pd.concat(replica_parts, ignore_index=True)
        assert offset == 8000
        assert len(replica_table) == 8000
        assert replica_table.frame.tolist() == list(range(1, 8001))
        frame_parts.append(replica_table)
        print(f"{ligand} {replica}: 8,000 frames: PASS")

frames = pd.concat(frame_parts, ignore_index=True)
assert len(frames) == 48000
assert np.isfinite(frames[LINEAR + CIRCULAR].to_numpy(float)).all()

replica_rows = []
for (ligand, replica), subset in frames.groupby(["ligand", "replica"], sort=True):
    for period, selected in (
        ("0-80 ns", subset),
        ("70-80 ns", subset[subset.time_ps > 70000.0]),
    ):
        row = {"ligand": ligand, "replica": replica, "period": period}
        row.update(summarize(selected))
        replica_rows.append(row)
replica_summary = pd.DataFrame(replica_rows)
assert len(replica_summary) == 12

interval_rows = []
for (ligand, replica), subset in frames.groupby(["ligand", "replica"], sort=True):
    interval = np.ceil(subset.time_ps.to_numpy(float) / 10000.0).astype(int)
    interval = np.clip(interval, 1, 8)
    work = subset.copy()
    work["interval"] = interval
    for number, selected in work.groupby("interval", sort=True):
        row = {
            "ligand": ligand,
            "replica": replica,
            "period": f"{number * 10 - 10}-{number * 10} ns",
        }
        row.update(summarize(selected))
        interval_rows.append(row)
interval_summary = pd.DataFrame(interval_rows)
assert len(interval_summary) == 48

effect_rows = []
for period in ("0-80 ns", "70-80 ns"):
    period_table = replica_summary[replica_summary.period == period]
    for metric in LINEAR:
        column = f"mean_{metric}"
        amg = period_table[period_table.ligand == "AMG315"].sort_values("replica")[column].to_numpy(float)
        aea = period_table[period_table.ligand == "AEA"].sort_values("replica")[column].to_numpy(float)
        effect_rows.append({
            "period": period,
            "metric": metric,
            "AMG315_mean": amg.mean(),
            "AMG315_SD": amg.std(ddof=1),
            "AEA_mean": aea.mean(),
            "AEA_SD": aea.std(ddof=1),
            "AEA_minus_AMG315": aea.mean() - amg.mean(),
            "Hedges_g_AEA_minus_AMG315": hedges_g(amg, aea),
            "exact_two_sided_permutation_p": exact_permutation_p(amg, aea),
            "replicas_per_ligand": 3,
        })
effects = pd.DataFrame(effect_rows)
assert len(effects) == 6

circular_rows = []
for period in ("0-80 ns", "70-80 ns"):
    subset = frames if period == "0-80 ns" else frames[frames.time_ps > 70000.0]
    for metric in CIRCULAR:
        ligand_stats = {}
        for ligand, selected in subset.groupby("ligand"):
            ligand_stats[ligand] = circular_stats(selected[metric])
        circular_rows.append({
            "period": period,
            "metric": metric,
            "AMG315_circular_mean_deg": ligand_stats["AMG315"][0],
            "AMG315_resultant_length": ligand_stats["AMG315"][1],
            "AEA_circular_mean_deg": ligand_stats["AEA"][0],
            "AEA_resultant_length": ligand_stats["AEA"][1],
            "wrapped_AEA_minus_AMG315_deg": wrap_degrees(ligand_stats["AEA"][0] - ligand_stats["AMG315"][0]),
            "inference": "descriptive_circular_comparison",
        })
circular_comparison = pd.DataFrame(circular_rows)
assert len(circular_comparison) == 4

definitions = pd.DataFrame([
    {"metric": LINEAR[0], "definition": "minimum of ARG214 NH1/NH2 to ASP338 OD1/OD2 distances", "type": "linear_distance", "unit": "A"},
    {"metric": LINEAR[1], "definition": "ARG214 CA to ALA342 CA distance", "type": "linear_distance", "unit": "A"},
    {"metric": CIRCULAR[0], "definition": "TRP356 CA-CB-CG-CD1 chi2 dihedral", "type": "circular_angle", "unit": "degrees"},
    {"metric": CIRCULAR[1], "definition": "TYR397 N-CA-CB-CG chi1 dihedral", "type": "circular_angle", "unit": "degrees"},
    {"metric": LINEAR[2], "definition": "ASN393 CG to TYR397 OH distance", "type": "linear_distance", "unit": "A"},
])

outputs = {
    "AMG315_AEA_80ns_microswitch_frame_metrics.tsv": frames,
    "AMG315_AEA_80ns_microswitch_reference_metrics.tsv": pd.DataFrame(reference_rows),
    "AMG315_AEA_80ns_microswitch_replica_summary.tsv": replica_summary,
    "AMG315_AEA_80ns_microswitch_10ns_interval_summary.tsv": interval_summary,
    "AMG315_AEA_80ns_microswitch_replica_effect_sizes.tsv": effects,
    "AMG315_AEA_80ns_microswitch_circular_comparison.tsv": circular_comparison,
    "AMG315_AEA_80ns_microswitch_metric_definitions.tsv": definitions,
}
for name, table in outputs.items():
    table.to_csv(OUT / name, sep="\t", index=False)

print()
print("LINEAR-DISTANCE REPLICA EFFECTS")
print(effects.to_string(index=False))
print()
print("CIRCULAR COMPARISONS")
print(circular_comparison.to_string(index=False))
print()
print("Exact 48,000-frame matched receptor record: PASS")
print("Exact 12-row full/final replica summary: PASS")
print("Exact 48-row ten-ns interval summary: PASS")
print("Six exploratory replica-level distance comparisons: PASS")
print("Four descriptive circular comparisons: PASS")
print("State labels were not compared between ligand-specific models: PASS")
print("AMG315–AEA CB1 80-NS MICROSWITCH ANALYSIS: COMPLETE")
