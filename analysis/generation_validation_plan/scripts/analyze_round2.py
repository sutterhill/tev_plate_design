#!/usr/bin/env python3
"""Evidence synthesis for a TEV generation-to-validation plan.

This analysis deliberately separates experimental observations from inferred
activity tiers and model-derived quantities. Machine-readable outputs are JSON
so the source CSV files remain untouched.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import log_softmax
from scipy.spatial.distance import jensenshannon
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import roc_auc_score


TEV_REANALYSIS = Path("/opt/dlami/nvme/tev_reanalysis")
PLATE_ROOT = Path("/opt/dlami/nvme/tev_plate_design")
OUT = TEV_REANALYSIS / "round2_generation_validation"
RAW = OUT / "raw" / "ridgey"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
AA = "ACDEFGHIKLMNPQRSTVWY"
TIER_ORDER = {"inactive_or_floor": 0, "somewhat_active": 1, "active": 2}
SEVERE_DDG_IMPROVEMENT = -1.0


def load_gzip_json(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict("records")


def parent_sequence() -> str:
    return "".join(
        line.strip()
        for line in (TEV_REANALYSIS / "dataset" / "TEVd.fasta").read_text().splitlines()
        if not line.startswith(">")
    )


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def structured_positions() -> set[int]:
    """Infer helix/sheet-like positions from 1LVM backbone phi/psi geometry."""
    atoms: dict[int, dict[str, np.ndarray]] = {}
    path = TEV_REANALYSIS / "dataset" / "TEVd_1LVM_A_1-221.pdb"
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM") or line[21].strip() not in ("", "A"):
            continue
        atom = line[12:16].strip()
        if atom not in {"N", "CA", "C"}:
            continue
        position = int(line[22:26])
        atoms.setdefault(position, {})[atom] = np.array([
            float(line[30:38]), float(line[38:46]), float(line[46:54])
        ])
    raw_class: dict[int, str] = {}
    for position in range(2, 221):
        if not all(
            atom in atoms.get(residue, {})
            for residue, atom in (
                (position - 1, "C"), (position, "N"), (position, "CA"),
                (position, "C"), (position + 1, "N"),
            )
        ):
            continue
        phi = dihedral(
            atoms[position - 1]["C"], atoms[position]["N"],
            atoms[position]["CA"], atoms[position]["C"],
        )
        psi = dihedral(
            atoms[position]["N"], atoms[position]["CA"],
            atoms[position]["C"], atoms[position + 1]["N"],
        )
        helix = -105 <= phi <= -25 and -90 <= psi <= 20
        sheet = -180 <= phi <= -55 and (70 <= psi <= 180 or -180 <= psi <= -125)
        if helix:
            raw_class[position] = "H"
        elif sheet:
            raw_class[position] = "E"
    # A loop residue can momentarily occupy a helix/sheet-like Ramachandran
    # region. Keep only contiguous secondary-structure runs, approximating the
    # run logic of DSSP/P-SEA rather than classifying isolated torsions.
    structured: set[int] = set()
    run: list[int] = []
    run_class = None
    for position in range(1, 222):
        current = raw_class.get(position)
        if current is not None and current == run_class and run and position == run[-1] + 1:
            run.append(position)
            continue
        if run and ((run_class == "H" and len(run) >= 3) or (run_class == "E" and len(run) >= 2)):
            structured.update(run)
        run = [position] if current is not None else []
        run_class = current
    if run and ((run_class == "H" and len(run) >= 3) or (run_class == "E" and len(run) >= 2)):
        structured.update(run)
    return structured


def prediction_members(item: dict) -> list[dict]:
    value = item["predictions"]
    return value if isinstance(value, list) else [value]


def scalar(item: dict, key: str) -> float:
    return float(np.mean([float(member[key]) for member in prediction_members(item)]))


def matrix_members(item: dict, key: str, order_key: str) -> list[np.ndarray]:
    members = prediction_members(item)
    order = members[0][order_key]
    reindex = [order.index(aa) for aa in AA]
    return [np.asarray(member[key], dtype=float)[:, reindex] for member in members]


def mutation_scores(
    sequences: pd.Series,
    parent: str,
    member_matrices: list[np.ndarray],
    structured: set[int],
) -> pd.DataFrame:
    wt_index = np.asarray([AA.index(aa) for aa in parent])
    improvements = []
    for matrix in member_matrices:
        wt = matrix[np.arange(len(parent)), wt_index][:, None]
        improvements.append(wt - matrix)
    mean_improvement = np.mean(improvements, axis=0)
    output = []
    for sequence in sequences:
        changed = [i for i, (a, b) in enumerate(zip(parent, sequence)) if a != b]
        values = np.asarray([mean_improvement[i, AA.index(sequence[i])] for i in changed])
        structured_mask = np.asarray([(i + 1) in structured for i in changed], dtype=bool)
        structured_values = values[structured_mask]
        loop_values = values[~structured_mask]
        output.append({
            "ridgey_ddg_sum_improvement_all": float(values.sum()) if len(values) else 0.0,
            "ridgey_ddg_mean_improvement_all": float(values.mean()) if len(values) else 0.0,
            "ridgey_ddg_fraction_improving_all": float((values > 0).mean()) if len(values) else 1.0,
            "ridgey_ddg_structured_sum_improvement": (
                float(structured_values.sum()) if len(structured_values) else 0.0
            ),
            "ridgey_ddg_loop_sum_improvement": float(loop_values.sum()) if len(loop_values) else 0.0,
            "ridgey_ddg_severe_structured_count": int(
                (structured_values < SEVERE_DDG_IMPROVEMENT).sum()
            ),
            "ridgey_ddg_n_structured_mutations": int(structured_mask.sum()),
            "ridgey_ddg_n_loop_mutations": int((~structured_mask).sum()),
        })
    return pd.DataFrame(output, index=sequences.index)


def inverse_fold_scores(sequences: pd.Series, parent: str, model: str = "600m") -> pd.Series:
    result = load_gzip_json(
        TEV_REANALYSIS / "results" / "ridgey" / "inverse_fold" / f"{model}_full.json.gz"
    )
    counts = np.ones((len(parent), len(AA)), dtype=float)
    for sequence in result["sequences"]:
        for position, residue in enumerate(sequence):
            counts[position, AA.index(residue)] += 1
    logp = np.log(counts / counts.sum(axis=1, keepdims=True))
    wt = np.asarray([logp[i, AA.index(aa)] for i, aa in enumerate(parent)])
    values = []
    for sequence in sequences:
        selected = np.asarray([logp[i, AA.index(aa)] for i, aa in enumerate(sequence)])
        values.append(float((selected - wt).sum()))
    return pd.Series(values, index=sequences.index)


def msa_scores(sequences: pd.Series, parent: str) -> pd.DataFrame:
    data = np.load(TEV_REANALYSIS / "results" / "evolution" / "tev_ev_wide.npz")
    order = str(data["aa_order"])
    reindex = [order.index(aa) for aa in AA]
    output = {}
    for label, key in (("independent", "dE_independent"), ("potts", "dE_potts")):
        matrix = np.asarray(data[key], dtype=float)[:, reindex]
        wt = np.asarray([matrix[i, AA.index(aa)] for i, aa in enumerate(parent)])
        values = []
        for sequence in sequences:
            selected = np.asarray([matrix[i, AA.index(aa)] for i, aa in enumerate(sequence)])
            values.append(float((selected - wt).sum()))
        output[f"msa_{label}_sum_improvement"] = values
    return pd.DataFrame(output, index=sequences.index)


def proteinmpnn_wt_scores(metadata: pd.DataFrame) -> dict[str, float]:
    folder = (
        TEV_REANALYSIS / "results" / "proteinmpnn" /
        "score_all_designs_order16" / "score_only"
    )
    names = ["TEVd"] + metadata.design_id.tolist()
    values = []
    for index in range(1, len(names) + 1):
        path = folder / f"TEVd_1LVM_A_1-221_fasta_{index}.npz"
        values.append(float(np.mean(np.load(path)["score"])))
    return {name: values[0] - value for name, value in zip(names, values)}


def load_m3(parent: str, structured: set[int]) -> tuple[pd.DataFrame, list[np.ndarray]]:
    metadata = pd.read_csv(TEV_REANALYSIS / "dataset" / "design_metadata.csv")
    m3 = metadata.loc[metadata.method_code.eq("m3")].copy().reset_index(drop=True)
    tiers = pd.DataFrame(json.loads(
        (PLATE_ROOT / "work" / "m3_controls" / "m3_all_trace_tiers.json").read_text()
    ))
    tiers["trace_slope_relative"] = pd.to_numeric(tiers.trace_slope_relative, errors="coerce").fillna(0.0)
    m3 = m3.merge(tiers[["design_id", "activity_tier", "trace_slope_relative"]], on="design_id")
    m3["activity_ordinal"] = m3.activity_tier.map(TIER_ORDER)
    m3["activity_log_trace"] = np.log10(m3.trace_slope_relative + 0.02)

    long = pd.read_csv(TEV_REANALYSIS / "results" / "tables" / "ridgey_design_predictions_long.csv")
    model_aliases = {"600M": "600m", "600M ensemble": "ensemble", "6B": "6b"}
    for model, alias in model_aliases.items():
        subset = long.loc[long.model.eq(model)].set_index("variant")
        m3[f"ridgey_{alias}_stability_delta"] = m3.design_id.map(subset.stability_delta_vs_TEVD)
        m3[f"ridgey_{alias}_solubility"] = m3.design_id.map(subset.solubility)
        m3[f"ridgey_{alias}_solubility_delta"] = m3.design_id.map(subset.solubility_delta_vs_TEVD)
        if alias == "ensemble":
            m3["ridgey_ensemble_stability_sd"] = m3.design_id.map(subset.stability_member_std)
            m3["ridgey_ensemble_solubility_sd"] = m3.design_id.map(subset.solubility_member_std)
            m3["ridgey_ensemble_stability_lcb_delta"] = (
                m3.ridgey_ensemble_stability_delta - m3.ridgey_ensemble_stability_sd
            )
            m3["ridgey_ensemble_solubility_lcb"] = (
                m3.ridgey_ensemble_solubility - m3.ridgey_ensemble_solubility_sd
            )

    ec = pd.read_csv(TEV_REANALYSIS / "results" / "ridgey" / "ec_class_scores.csv")
    ec = ec.loc[ec.model.eq("600m")].set_index("design_id")
    parent_ec = float(ec.loc["TEVd", "ec_3_4_22_44_probability"])
    m3["ridgey_600m_ec_probability"] = m3.design_id.map(ec.ec_3_4_22_44_probability)
    m3["ridgey_600m_ec_log2_ratio_vs_tevd"] = np.log2(
        (m3.ridgey_600m_ec_probability + 1e-12) / (parent_ec + 1e-12)
    )

    own_if = pd.read_csv(
        TEV_REANALYSIS / "results" / "ridgey" / "own_structure_inverse_fold_scores.csv"
    ).query("model == '600m'").set_index("design_id")
    m3["ridgey_own_structure_if_mean_logp"] = m3.design_id.map(
        own_if.marginal_mean_log_probability
    )
    m3["proteinmpnn_wt_nll_improvement"] = m3.design_id.map(proteinmpnn_wt_scores(metadata))
    m3["ridgey_wt_if_logodds_sum"] = inverse_fold_scores(m3.sequence, parent)
    m3 = pd.concat([m3, msa_scores(m3.sequence, parent)], axis=1)

    parent_item = load_gzip_json(
        TEV_REANALYSIS / "results" / "ridgey" / "predictions_600m_ensemble.json.gz"
    )[0]
    ddg_members = matrix_members(parent_item, "ddg", "ddg_amino_acid_order")
    m3 = pd.concat([m3, mutation_scores(m3.sequence, parent, ddg_members, structured)], axis=1)
    return m3, ddg_members


METRICS = {
    "600M stability Δ": "ridgey_600m_stability_delta",
    "Ensemble stability mean Δ": "ridgey_ensemble_stability_delta",
    "Ensemble stability paired LCB Δ": "ridgey_ensemble_stability_lcb_delta",
    "600M solubility": "ridgey_600m_solubility",
    "Ensemble solubility mean": "ridgey_ensemble_solubility",
    "Ensemble solubility LCB": "ridgey_ensemble_solubility_lcb",
    "6B stability Δ": "ridgey_6b_stability_delta",
    "6B solubility": "ridgey_6b_solubility",
    "600M exact EC log2 ratio": "ridgey_600m_ec_log2_ratio_vs_tevd",
    "ProteinMPNN p(seq|WT) improvement": "proteinmpnn_wt_nll_improvement",
    "Ridgey IF p(seq|WT) log-odds": "ridgey_wt_if_logodds_sum",
    "Ridgey IF p(seq|mutant) mean logp": "ridgey_own_structure_if_mean_logp",
    "Ridgey structured-only ddG sum": "ridgey_ddg_structured_sum_improvement",
    "Ridgey loop-only ddG sum": "ridgey_ddg_loop_sum_improvement",
    "MSA independent score": "msa_independent_sum_improvement",
    "MSA Potts score": "msa_potts_sum_improvement",
    "Identity to TEVd": "identity_to_TEVD_percent",
}


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = []
    for _ in range(2000):
        index = rng.integers(0, len(x), len(x))
        rho = spearmanr(x[index], y[index]).statistic
        if np.isfinite(rho):
            values.append(rho)
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    exceed = 0
    for _ in range(5000):
        value = spearmanr(x, rng.permutation(y)).statistic
        exceed += abs(value) >= abs(observed)
    return (exceed + 1) / 5001


def metric_evidence(m3: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(29)
    rows = []
    active = m3.activity_tier.eq("active").astype(int).to_numpy()
    for display, column in METRICS.items():
        valid = m3[[column, "activity_log_trace", "activity_tier"]].dropna()
        x = valid[column].to_numpy(float)
        y = valid.activity_log_trace.to_numpy(float)
        rho = float(spearmanr(x, y).statistic)
        low, high = bootstrap_spearman(x, y, rng)
        p = permutation_p(x, y, rho, rng)
        auc = float(roc_auc_score(active[m3[column].notna()], x))
        group_medians = valid.groupby("activity_tier")[column].median().to_dict()
        rows.append({
            "metric": display,
            "column": column,
            "n": len(valid),
            "spearman_rho_trace": rho,
            "spearman_ci_low": low,
            "spearman_ci_high": high,
            "permutation_p_two_sided": p,
            "active_vs_rest_auc": auc,
            "floor_median": float(group_medians.get("inactive_or_floor", np.nan)),
            "somewhat_median": float(group_medians.get("somewhat_active", np.nan)),
            "active_median": float(group_medians.get("active", np.nan)),
        })
    return pd.DataFrame(rows).sort_values("spearman_rho_trace", ascending=False)


def bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def mutation_signatures(m3: pd.DataFrame, parent: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = m3.loc[m3.activity_tier.eq("active")]
    floor = m3.loc[m3.activity_tier.eq("inactive_or_floor")]
    rows = []
    position_rows = []
    for position in range(len(parent)):
        active_counts = Counter(sequence[position] for sequence in active.sequence)
        floor_counts = Counter(sequence[position] for sequence in floor.sequence)
        active_probs = np.asarray([(active_counts[aa] + 0.5) for aa in AA], dtype=float)
        floor_probs = np.asarray([(floor_counts[aa] + 0.5) for aa in AA], dtype=float)
        active_probs /= active_probs.sum()
        floor_probs /= floor_probs.sum()
        position_rows.append({
            "position_1indexed": position + 1,
            "wt": parent[position],
            "active_consensus": active_counts.most_common(1)[0][0],
            "floor_consensus": floor_counts.most_common(1)[0][0],
            "jensen_shannon_distance": float(jensenshannon(active_probs, floor_probs, base=2)),
        })
        for residue in sorted(set(active_counts) | set(floor_counts)):
            if residue == parent[position]:
                continue
            a = active_counts[residue]
            b = floor_counts[residue]
            if a + b < 3:
                continue
            table = [[a, len(active) - a], [b, len(floor) - b]]
            odds, p = fisher_exact(table, alternative="two-sided")
            log2_odds = math.log2(
                ((a + 0.5) / (len(active) - a + 0.5)) /
                ((b + 0.5) / (len(floor) - b + 0.5))
            )
            rows.append({
                "position_1indexed": position + 1,
                "wt": parent[position],
                "residue": residue,
                "mutation": f"{parent[position]}{position + 1}{residue}",
                "active_count": int(a),
                "active_frequency": float(a / len(active)),
                "floor_count": int(b),
                "floor_frequency": float(b / len(floor)),
                "frequency_difference": float(a / len(active) - b / len(floor)),
                "fisher_odds_ratio": float(odds),
                "log2_odds_haldane": float(log2_odds),
                "fisher_p": float(p),
            })
    signatures = pd.DataFrame(rows)
    signatures["fisher_fdr_bh"] = bh_adjust(signatures.fisher_p.tolist())
    signatures = signatures.sort_values(
        ["fisher_fdr_bh", "frequency_difference"], ascending=[True, False]
    )
    positions = pd.DataFrame(position_rows).sort_values("jensen_shannon_distance", ascending=False)
    return signatures, positions


def attribution_table(parent: str, fixed: set[int], structured: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for scaffold in ("TEVd", "hyperTEV56", "hyperTEV60", "hyperTEV89"):
        for model in ("600m", "6b"):
            for target in ("stability", "solubility", "ec:3.4.22.44"):
                safe_target = target.replace(":", "_")
                item = load_gzip_json(
                    RAW / "attributions" / scaffold / f"{model}_{safe_target}.json.gz"
                )
                signed = np.asarray(item["attributions"]["signed"], dtype=float)
                positive_rank = pd.Series(signed).rank(pct=True).to_numpy()
                for index, value in enumerate(signed):
                    rows.append({
                        "scaffold": scaffold,
                        "model": model,
                        "target": target,
                        "position_1indexed": index + 1,
                        "residue": item["sequence"][index],
                        "signed_attribution": float(value),
                        "within_map_percentile": float(positive_rank[index]),
                        "positive": bool(value > 0),
                        "m3_fixed": bool(index + 1 in fixed),
                        "structured": bool(index + 1 in structured),
                    })
    long = pd.DataFrame(rows)
    ec = long.loc[long.target.eq("ec:3.4.22.44")]
    consensus_rows = []
    for position, group in ec.groupby("position_1indexed"):
        residue_counts = group.groupby("scaffold").residue.first().value_counts()
        consensus_residue = residue_counts.index[0]
        consensus_count = int(residue_counts.iloc[0])
        consensus_rows.append({
            "position_1indexed": int(position),
            "tevd_residue": parent[position - 1],
            "scaffold_consensus_residue": consensus_residue,
            "scaffolds_with_consensus_residue": consensus_count,
            "positive_maps": int(group.positive.sum()),
            "mean_signed_attribution": float(group.signed_attribution.mean()),
            "mean_within_map_percentile": float(group.within_map_percentile.mean()),
            "m3_fixed": bool(position in fixed),
            "structured": bool(position in structured),
        })
    consensus = pd.DataFrame(consensus_rows)
    consensus["extra_lock_candidate"] = (
        ~consensus.m3_fixed
        & consensus.positive_maps.ge(6)
        & consensus.scaffolds_with_consensus_residue.ge(3)
        & consensus.mean_within_map_percentile.ge(0.65)
    )
    consensus = consensus.sort_values(
        ["extra_lock_candidate", "positive_maps", "mean_within_map_percentile"],
        ascending=[False, False, False],
    )
    return long, consensus


def exact_ec(item: dict) -> float:
    mapping = {
        row["label"]: float(row["probability"])
        for row in item["predictions"]["ec_number"]
    }
    if "3.4.22.44" not in mapping:
        raise ValueError(f"Exact EC missing after low-threshold query: {item['name']}")
    return mapping["3.4.22.44"]


def parse_member_values(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def load_current_plate(
    parent: str,
    structured: set[int],
    ddg_members: list[np.ndarray],
    active_thresholds: dict,
) -> pd.DataFrame:
    plate = pd.read_csv(PLATE_ROOT / "deliverables" / "tev_plate_97.csv")
    pred600 = {
        item["name"]: item
        for item in load_gzip_json(RAW / "plate_predictions_600m.json.gz")
    }
    pred6b = {
        item["name"]: item
        for item in load_gzip_json(RAW / "plate_predictions_6b.json.gz")
    }
    if set(plate.id) != set(pred600) or set(plate.id) != set(pred6b):
        raise ValueError("Current-plate Ridgey prediction names do not match plate table")
    parent_id = "TEVd"
    parent_ec = exact_ec(pred600[parent_id])
    parent_6b_stability = scalar(pred6b[parent_id], "stability")
    parent_6b_solubility = scalar(pred6b[parent_id], "solubility")
    parent_mpnn = float(plate.loc[plate.id.eq(parent_id), "mpnn_wt_structure_nll"].iloc[0])
    parent_stability_members = parse_member_values(
        plate.loc[plate.id.eq(parent_id), "ridgey_ensemble_stability_members"].iloc[0]
    )
    parent_solubility_members = parse_member_values(
        plate.loc[plate.id.eq(parent_id), "ridgey_ensemble_solubility_members"].iloc[0]
    )

    plate["ridgey_600m_ec_probability"] = plate.id.map(lambda name: exact_ec(pred600[name]))
    plate["ridgey_600m_ec_log2_ratio_vs_tevd"] = np.log2(
        (plate.ridgey_600m_ec_probability + 1e-12) / (parent_ec + 1e-12)
    )
    plate["ridgey_6b_stability"] = plate.id.map(lambda name: scalar(pred6b[name], "stability"))
    plate["ridgey_6b_stability_delta"] = plate.ridgey_6b_stability - parent_6b_stability
    plate["ridgey_6b_solubility"] = plate.id.map(lambda name: scalar(pred6b[name], "solubility"))
    plate["ridgey_6b_solubility_delta"] = plate.ridgey_6b_solubility - parent_6b_solubility

    stability_lcb = []
    solubility_lcb = []
    for _, row in plate.iterrows():
        stability_delta = parse_member_values(row.ridgey_ensemble_stability_members) - parent_stability_members
        solubility_values = parse_member_values(row.ridgey_ensemble_solubility_members)
        stability_lcb.append(float(stability_delta.mean() - stability_delta.std(ddof=1)))
        solubility_lcb.append(float(solubility_values.mean() - solubility_values.std(ddof=1)))
    plate["ridgey_ensemble_stability_lcb_delta"] = stability_lcb
    plate["ridgey_ensemble_solubility_lcb"] = solubility_lcb
    plate["proteinmpnn_wt_nll_improvement"] = parent_mpnn - plate.mpnn_wt_structure_nll
    plate["ridgey_wt_if_logodds_sum"] = inverse_fold_scores(plate.sequence, parent)
    plate = pd.concat([plate, msa_scores(plate.sequence, parent)], axis=1)
    plate = pd.concat([plate, mutation_scores(plate.sequence, parent, ddg_members, structured)], axis=1)

    plate["activity_preservation_gate"] = (
        plate.ridgey_600m_ec_probability.ge(active_thresholds["ec_probability_q10"])
        & plate.msa_independent_sum_improvement.ge(active_thresholds["msa_independent_q10"])
        & plate.ridgey_ddg_severe_structured_count.le(active_thresholds["severe_structured_q90"])
    )
    plate["property_floor_gate"] = (
        plate.ridgey_ensemble_solubility_lcb.ge(active_thresholds["solubility_lcb_q10"])
        & plate.ridgey_ensemble_stability_lcb_delta.gt(0)
        & plate.ridgey_6b_stability_delta.gt(0)
    )
    # Compatibility alias retained in the machine-readable table.
    plate["functional_floor_pass"] = plate.activity_preservation_gate
    plate["ready_gate"] = (
        plate.af2_gate_pass.astype(bool)
        & plate.activity_preservation_gate
        & plate.property_floor_gate
    )

    generated = plate.cohort.isin(["ridgey", "proteinmpnn"])
    quality_columns = [
        "ridgey_ensemble_stability_lcb_delta",
        "ridgey_ensemble_solubility_lcb",
        "ridgey_6b_stability_delta",
        "ridgey_600m_ec_log2_ratio_vs_tevd",
        "proteinmpnn_wt_nll_improvement",
        "msa_independent_sum_improvement",
        "ridgey_ddg_structured_sum_improvement",
    ]
    for cohort in ("ridgey", "proteinmpnn"):
        mask = plate.cohort.eq(cohort)
        percentiles = plate.loc[mask, quality_columns].rank(pct=True)
        plate.loc[mask, "quality_mean_percentile"] = percentiles.mean(axis=1)
        plate.loc[mask, "quality_weakest_percentile"] = percentiles.min(axis=1)
        plate.loc[mask, "quality_score"] = (
            0.4 * percentiles.mean(axis=1) + 0.6 * percentiles.min(axis=1)
        )
    plate.loc[~generated, ["quality_mean_percentile", "quality_weakest_percentile", "quality_score"]] = np.nan
    return plate


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def diverse_shortlist(frame: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    candidates = frame.sort_values("quality_score", ascending=False).head(max(n * 2, n)).copy()
    ready = candidates.loc[candidates.ready_gate]
    if len(ready) >= n:
        candidates = ready
    selected = [candidates.index[0]]
    while len(selected) < min(n, len(candidates)):
        remaining = [index for index in candidates.index if index not in selected]
        best = max(
            remaining,
            key=lambda index: (
                float(candidates.loc[index, "quality_score"])
                + 0.5 * min(
                    hamming(candidates.loc[index, "sequence"], candidates.loc[other, "sequence"])
                    for other in selected
                ) / 221.0
            ),
        )
        selected.append(best)
    result = candidates.loc[selected].copy()
    result["shortlist_rank"] = range(1, len(result) + 1)
    result["min_hamming_to_other_shortlist"] = [
        min(hamming(sequence, other) for other in result.sequence if other != sequence)
        if len(result) > 1 else 0
        for sequence in result.sequence
    ]
    return result


def plot_metric_evidence(evidence: pd.DataFrame) -> None:
    ordered = evidence.sort_values("spearman_rho_trace")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    ax.barh(y - 0.18, ordered.spearman_rho_trace, height=0.34, label="Spearman vs m3 trace")
    ax.barh(
        y + 0.18,
        2 * (ordered.active_vs_rest_auc - 0.5),
        height=0.34,
        label="2 × (active AUC − 0.5)",
    )
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y, ordered.metric)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Higher means better association with m3 activity")
    ax.set_title("Exploratory evidence within the paper's m3 arm (48 designs)", fontsize=17)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=12)
    sns.despine()
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(FIGURES / "m3_metric_activity_evidence.png", dpi=220)
    plt.close(fig)


def plot_signature_heatmap(m3: pd.DataFrame, signatures: pd.DataFrame) -> None:
    top = signatures.assign(abs_effect=signatures.frequency_difference.abs()).sort_values(
        ["fisher_fdr_bh", "abs_effect"], ascending=[True, False]
    ).head(22)
    matrix = []
    labels = []
    for _, row in top.iterrows():
        position = int(row.position_1indexed) - 1
        residue = row.residue
        labels.append(row.mutation)
        matrix.append([
            float((m3.loc[m3.activity_tier.eq(tier), "sequence"].str[position] == residue).mean())
            for tier in ("inactive_or_floor", "somewhat_active", "active")
        ])
    fig, ax = plt.subplots(figsize=(7.8, max(6, 0.32 * len(labels))))
    sns.heatmap(
        np.asarray(matrix), cmap="mako", vmin=0, vmax=1, annot=True, fmt=".2f",
        xticklabels=["floor", "somewhat", "active"], yticklabels=labels,
        cbar_kws={"label": "Residue frequency"}, ax=ax,
    )
    ax.set_title("m3 substitution signatures (trace-order inferred tiers)", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "m3_activity_signature_heatmap.png", dpi=220)
    plt.close(fig)


def plot_attributions(long: pd.DataFrame, fixed: set[int]) -> None:
    targets = ("stability", "solubility", "ec:3.4.22.44")
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.08]},
    )
    for ax, target in zip(axes[:3], targets):
        subset = long.loc[long.target.eq(target)]
        values = subset.groupby("position_1indexed").signed_attribution.mean()
        scale = np.quantile(np.abs(values), 0.98)
        ax.imshow(
            values.to_numpy()[None, :], aspect="auto", cmap="coolwarm",
            vmin=-scale, vmax=scale, extent=(0.5, 221.5, 0, 1),
        )
        ax.set_yticks([])
        ax.set_ylabel(target.replace("ec:", "EC "), rotation=0, ha="right", va="center")
    fixed_mask = np.asarray([[1 if position in fixed else 0 for position in range(1, 222)]])
    axes[3].imshow(
        fixed_mask, aspect="auto", cmap="Greys", vmin=0, vmax=1,
        extent=(0.5, 221.5, 0, 1),
    )
    axes[3].set_yticks([])
    axes[3].set_ylabel("m3 fixed", rotation=0, ha="right", va="center", fontsize=10)
    axes[3].set_xlabel("TEV residue number")
    axes[0].set_title(
        "Mean Ridgey integrated-gradient attribution across 600M/6B and four scaffolds",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "ridgey_attribution_map.png", dpi=220)
    plt.close(fig)


def plot_current_plate(plate: pd.DataFrame, shortlist_ids: set[str]) -> None:
    subset = plate.loc[plate.cohort.isin(["ridgey", "proteinmpnn"])].copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = {"ridgey": "#7C3AED", "proteinmpnn": "#0EA5E9"}
    for cohort, group in subset.groupby("cohort"):
        selected = group.id.isin(shortlist_ids)
        ax.scatter(
            group.ridgey_ensemble_stability_lcb_delta,
            group.ridgey_ensemble_solubility_lcb,
            c=palette[cohort], s=38, alpha=0.58, label=cohort,
        )
        ax.scatter(
            group.loc[selected, "ridgey_ensemble_stability_lcb_delta"],
            group.loc[selected, "ridgey_ensemble_solubility_lcb"],
            facecolors="none", edgecolors="#111111", linewidths=1.4, s=92,
        )
    ax.axvline(0, color="#444444", lw=0.8, ls="--")
    ax.set_xlabel("Ridgey ensemble paired stability Δ LCB vs TEVd")
    ax.set_ylabel("Ridgey ensemble absolute solubility LCB")
    ax.set_title("Current high-mutation plate: diagnostic multi-objective shortlist outlined", fontsize=15)
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES / "current_plate_objective_map.png", dpi=220)
    plt.close(fig)


def plot_generation_funnel() -> None:
    stages = [
        (
            "1. Generate four arms",
            "8,192 Ridgey IF • 256 Ridgey-gradient endpoints\n"
            "8,192 ProteinMPNN • 4,096 lead-centered samples",
            "#EDE9FE",
        ),
        (
            "2. Sequence-only triage",
            "m3 + EC-support locks • K45A / retain K45 split • no new Cys\n"
            "68–76% identity • MSA independent ≥ −152.6 • Potts ≥ −198.1",
            "#DBEAFE",
        ),
        (
            "3. Orthogonal structure triage",
            "Fold ~600 with paper-parity AF2 and an MSA-free predictor\n"
            "pLDDT ≥ 87.5 • global Cα RMSD < 2 Å • active-site geometry retained",
            "#CFFAFE",
        ),
        (
            "4. Multi-model Pareto selection",
            "Ridgey ensemble LCB + 6B • solubility floor • EC anti-collapse\n"
            "structured-only ddG veto • ProteinMPNN/IF as plausibility gates • diversity",
            "#DCFCE7",
        ),
        (
            "5. Lab funnel",
            "84 new designs + 12 control wells → expression/solubility/activity screen\n"
            "top 12 full kinetics + Tm → top 6 protein-substrate/specificity validation",
            "#FEF3C7",
        ),
    ]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    heights = np.linspace(0.86, 0.10, len(stages))
    for index, ((title, text, color), y) in enumerate(zip(stages, heights)):
        box = FancyBboxPatch(
            (0.08, y - 0.07), 0.84, 0.14,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            facecolor=color, edgecolor="#334155", linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(0.11, y + 0.027, title, fontsize=15, fontweight="bold", va="center")
        ax.text(0.11, y - 0.027, text, fontsize=11.5, va="center")
        if index < len(stages) - 1:
            ax.annotate(
                "", xy=(0.5, heights[index + 1] + 0.076), xytext=(0.5, y - 0.076),
                arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#475569"},
            )
    ax.set_title("Recommended TEV generation → validation funnel", fontsize=20, pad=16)
    fig.tight_layout()
    fig.savefig(FIGURES / "generation_validation_funnel.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    parent = parent_sequence()
    structured = structured_positions()
    fixed_data = json.loads((TEV_REANALYSIS / "dataset" / "fixed_positions.json").read_text())
    fixed = set(fixed_data["active_site_plus_50pct_conserved"])

    m3, ddg_members = load_m3(parent, structured)
    evidence = metric_evidence(m3)
    signatures, position_signatures = mutation_signatures(m3, parent)
    attribution_long, ec_consensus = attribution_table(parent, fixed, structured)

    active = m3.loc[m3.activity_tier.eq("active")]
    active_thresholds = {
        "solubility_lcb_q10": float(active.ridgey_ensemble_solubility_lcb.quantile(0.10)),
        "stability_lcb_delta_q10_for_ranking": float(
            active.ridgey_ensemble_stability_lcb_delta.quantile(0.10)
        ),
        "stability_6b_delta_q10_for_ranking": float(
            active.ridgey_6b_stability_delta.quantile(0.10)
        ),
        "ec_probability_q10": float(active.ridgey_600m_ec_probability.quantile(0.10)),
        "msa_independent_q10": float(active.msa_independent_sum_improvement.quantile(0.10)),
        "severe_structured_q90": int(math.ceil(active.ridgey_ddg_severe_structured_count.quantile(0.90))),
    }
    plate = load_current_plate(parent, structured, ddg_members, active_thresholds)
    shortlists = []
    for cohort in ("ridgey", "proteinmpnn"):
        shortlists.append(diverse_shortlist(plate.loc[plate.cohort.eq(cohort)], 12))
    shortlist = pd.concat(shortlists, ignore_index=True)

    plot_metric_evidence(evidence)
    plot_signature_heatmap(m3, signatures)
    plot_attributions(attribution_long, fixed)
    plot_current_plate(plate, set(shortlist.id))
    plot_generation_funnel()

    m3_columns = [
        "design_id", "sequence", "n_mutations", "identity_to_TEVD_percent",
        "activity_tier", "trace_slope_relative", "activity_log_trace",
    ] + list(dict.fromkeys(METRICS.values())) + [
        "ridgey_ddg_severe_structured_count", "ridgey_ddg_n_structured_mutations",
        "ridgey_ddg_n_loop_mutations",
    ]
    m3_columns = list(dict.fromkeys(m3_columns))
    plate_columns = [
        "id", "display_name", "cohort", "sequence", "mutations", "mutation_count",
        "identity_to_wt", "af2_mean_plddt", "global_ca_rmsd_vs_1lvm_A", "af2_gate_pass",
        "ridgey_ensemble_stability_lcb_delta", "ridgey_ensemble_solubility_lcb",
        "ridgey_6b_stability_delta", "ridgey_6b_solubility",
        "ridgey_600m_ec_probability", "ridgey_600m_ec_log2_ratio_vs_tevd",
        "proteinmpnn_wt_nll_improvement", "ridgey_wt_if_logodds_sum",
        "msa_independent_sum_improvement", "msa_potts_sum_improvement",
        "ridgey_ddg_structured_sum_improvement", "ridgey_ddg_loop_sum_improvement",
        "ridgey_ddg_severe_structured_count", "activity_preservation_gate",
        "property_floor_gate", "functional_floor_pass", "ready_gate",
        "quality_mean_percentile", "quality_weakest_percentile", "quality_score",
    ]
    shortlist_columns = [
        "shortlist_rank", "id", "display_name", "cohort", "sequence", "mutations",
        "mutation_count", "identity_to_wt", "quality_score", "ready_gate",
        "min_hamming_to_other_shortlist", "ridgey_ensemble_stability_lcb_delta",
        "ridgey_ensemble_solubility_lcb", "ridgey_6b_stability_delta",
        "ridgey_600m_ec_probability", "proteinmpnn_wt_nll_improvement",
        "msa_independent_sum_improvement", "ridgey_ddg_structured_sum_improvement",
        "ridgey_ddg_severe_structured_count",
    ]
    write_json(TABLES / "m3_design_metrics.json", json_records(m3[m3_columns]))
    write_json(TABLES / "m3_metric_activity_evidence.json", json_records(evidence))
    write_json(TABLES / "m3_mutation_signatures.json", json_records(signatures))
    write_json(TABLES / "m3_position_signatures.json", json_records(position_signatures))
    write_json(TABLES / "ridgey_attributions_long.json", json_records(attribution_long))
    write_json(TABLES / "ridgey_ec_attribution_consensus.json", json_records(ec_consensus))
    write_json(TABLES / "current_plate_round2_metrics.json", json_records(plate[plate_columns]))
    write_json(TABLES / "current_plate_shortlist_24.json", json_records(shortlist[shortlist_columns]))

    active_counts = m3.activity_tier.value_counts().to_dict()
    extra_locks = ec_consensus.loc[ec_consensus.extra_lock_candidate, "position_1indexed"].astype(int).tolist()
    summary = {
        "m3_designs": int(len(m3)),
        "m3_tier_counts": {str(key): int(value) for key, value in active_counts.items()},
        "experimental_activity_assignment": "trace-order inferred from Figure S7c except named kinetic leads",
        "active_calibrated_thresholds": active_thresholds,
        "structured_positions_from_1lvm_phi_psi": sorted(structured),
        "loop_positions_excluded_from_structured_ddg_filter": sorted(set(range(1, 222)) - structured),
        "extra_ec_attribution_lock_candidates": extra_locks,
        "current_generated_ready_gate_counts": {
            cohort: int(plate.loc[plate.cohort.eq(cohort), "ready_gate"].sum())
            for cohort in ("ridgey", "proteinmpnn")
        },
        "shortlist_ids": shortlist.groupby("cohort").id.apply(list).to_dict(),
        "top_metric_evidence": json_records(evidence.head(6)),
        "important_caveats": [
            "The 48-design m3 activity tiers are inferred from row-major Figure S7c traces; only hyperTEV56/60/89 have named kinetics.",
            "The paper did not release individual numeric expression yields, so no individual-design expression model is fit.",
            "Ridgey stability and ddG are proteolysis-trained; only phi/psi-structured positions contribute to the ddG hard-risk feature.",
            "Integrated gradients explain a current score and do not identify causal substitutions.",
            "All current-plate candidates remain full high-mutation sequences; no mutations are reverted.",
        ],
    }
    write_json(OUT / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
