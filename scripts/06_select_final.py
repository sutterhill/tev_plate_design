#!/usr/bin/env python3
"""Select 36 diverse Ridgey candidates and 36 diverse ProteinMPNN controls."""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import ROOT, greedy_maximin, read_csv, write_csv, write_fasta, write_json


def as_bool(value: object) -> bool:
    return str(value).lower() in ("1", "true", "yes")


def member_array(value: object) -> np.ndarray:
    """Accept both the JSON arrays written by 05 and legacy pipe lists."""
    text = str(value)
    if text.startswith("["):
        return np.asarray(json.loads(text), dtype=float)
    return np.asarray([float(item) for item in text.split("|")], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=36)
    args = parser.parse_args()
    rows = []
    seen_sequences = set()
    for run_name in ("production", "ridgey_extension", "ridgey_enriched"):
        path = ROOT / "scores" / f"characterized_{run_name}.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            if row["sequence"] in seen_sequences:
                continue
            row["characterization_run"] = run_name
            rows.append(row)
            seen_sequences.add(row["sequence"])
    parent = next(r for r in rows if r["method"] == "parent")
    control_path = ROOT / "work" / "control_characterization" / "released_controls_final_25.csv"
    control_sequences = {row["sequence"] for row in read_csv(control_path)} if control_path.exists() else {parent["sequence"]}
    parent_stability = float(parent["ridgey_600m_stability"])
    parent_solubility = float(parent["ridgey_600m_solubility"])
    wt_stability_members = member_array(parent["ridgey_ensemble_stability_members"])
    wt_solubility_members = member_array(parent["ridgey_ensemble_solubility_members"])
    wt_stability_mean = float(parent["ridgey_ensemble_stability_mean"])
    wt_solubility_mean = float(parent["ridgey_ensemble_solubility_mean"])
    for row in rows:
        row["n_mutations"] = int(row["n_mutations"])
        row["ridgey_600m_stability_delta_vs_wt"] = float(row["ridgey_600m_stability"]) - parent_stability
        row["ridgey_600m_solubility_delta_vs_wt"] = float(row["ridgey_600m_solubility"]) - parent_solubility
        row["ridgey_600m_stability_percent_vs_wt"] = 100.0 * row["ridgey_600m_stability_delta_vs_wt"] / max(abs(parent_stability), 1e-12)
        row["ridgey_600m_solubility_percent_vs_wt"] = 100.0 * row["ridgey_600m_solubility_delta_vs_wt"] / max(abs(parent_solubility), 1e-12)
        stability_members = member_array(row["ridgey_ensemble_stability_members"])
        solubility_members = member_array(row["ridgey_ensemble_solubility_members"])
        if len(stability_members) != 5 or len(solubility_members) != 5:
            raise ValueError(f"expected five Ridgey members for {row['candidate_id']}")
        stability_delta = stability_members - wt_stability_members
        solubility_delta = solubility_members - wt_solubility_members
        stability_relative = stability_delta / max(abs(wt_stability_mean), 1e-12)
        solubility_relative = solubility_delta / max(abs(wt_solubility_mean), 1e-12)
        joint_relative = np.minimum(stability_relative, solubility_relative)
        row["ridgey_ensemble_stability_mean_delta_vs_wt"] = float(row["ridgey_ensemble_stability_mean"]) - wt_stability_mean
        row["ridgey_ensemble_solubility_mean_delta_vs_wt"] = float(row["ridgey_ensemble_solubility_mean"]) - wt_solubility_mean
        row["ridgey_ensemble_stability_mean_percent_vs_wt"] = 100.0 * row["ridgey_ensemble_stability_mean_delta_vs_wt"] / max(abs(wt_stability_mean), 1e-12)
        row["ridgey_ensemble_solubility_mean_percent_vs_wt"] = 100.0 * row["ridgey_ensemble_solubility_mean_delta_vs_wt"] / max(abs(wt_solubility_mean), 1e-12)
        row["ridgey_ensemble_stability_votes_vs_wt"] = int((stability_delta > 0).sum())
        row["ridgey_ensemble_solubility_votes_vs_wt"] = int((solubility_delta > 0).sum())
        row["ridgey_ensemble_joint_votes_vs_wt"] = int(((stability_delta > 0) & (solubility_delta > 0)).sum())
        row["ridgey_ensemble_stability_member_deltas_vs_wt"] = json.dumps(stability_delta.tolist())
        row["ridgey_ensemble_solubility_member_deltas_vs_wt"] = json.dumps(solubility_delta.tolist())
        # The second-worst of five joint member margins is positive exactly
        # when at least four members improve both endpoints.
        row["ridgey_consensus_4of5_margin"] = float(np.sort(joint_relative)[1])

    ridgey_pool = [
        row for row in rows
        if row["method"] == "ridgey"
        and row["sequence"] not in control_sequences
        and as_bool(row["af2_pass"])
        and float(row["ridgey_ensemble_stability_mean"]) > wt_stability_mean
        and float(row["ridgey_ensemble_solubility_mean"]) > wt_solubility_mean
    ]
    mpnn_pool = [
        row for row in rows
        if row["method"] == "proteinmpnn"
        and row["sequence"] not in control_sequences
        and as_bool(row["af2_pass"])
    ]
    if len(ridgey_pool) < args.count:
        raise RuntimeError(f"only {len(ridgey_pool)} Ridgey designs pass AF2 + strict stability/solubility > WT; need {args.count}. Generate/fold another shard.")
    if len(mpnn_pool) < args.count:
        raise RuntimeError(f"only {len(mpnn_pool)} ProteinMPNN designs pass AF2; need {args.count}.")
    # Keep the strongest three-fold overcomplete consensus set, then maximize
    # sequence diversity within it.  This prevents a barely passing outlier
    # from displacing a consistently favorable ensemble design.
    ridgey_quality_pool = sorted(
        ridgey_pool,
        key=lambda row: (
            -int(row["ridgey_ensemble_joint_votes_vs_wt"]),
            -float(row["ridgey_consensus_4of5_margin"]),
            row["candidate_id"],
        ),
    )[: min(len(ridgey_pool), args.count * 3)]
    ridgey_selected = greedy_maximin(ridgey_quality_pool, args.count)
    mpnn_selected = greedy_maximin(mpnn_pool, args.count, seed_key="mpnn_generation_nll")
    selected = [parent] + ridgey_selected + mpnn_selected
    ridgey_selection_rank = {row["candidate_id"]: index + 1 for index, row in enumerate(ridgey_selected)}
    mpnn_selection_rank = {row["candidate_id"]: index + 1 for index, row in enumerate(mpnn_selected)}
    for row in selected:
        row["final_selection"] = "parent" if row["method"] == "parent" else "diverse_ridgey_ensemble_consensus_ranked" if row["method"] == "ridgey" else "diverse_af2_pass_control"
        row["final_selection_rank"] = 0 if row["method"] == "parent" else ridgey_selection_rank.get(row["candidate_id"], mpnn_selection_rank.get(row["candidate_id"]))
    write_csv(ROOT / "selected" / "generated_73.csv", selected)
    write_fasta(ROOT / "selected" / "generated_73.fasta", [(r["candidate_id"], r["sequence"]) for r in selected])
    summary = {
        "parent": 1,
        "ridgey_pool": len(ridgey_pool),
        "ridgey_selected": len(ridgey_selected),
        "proteinmpnn_pool": len(mpnn_pool),
        "proteinmpnn_selected": len(mpnn_selected),
        "ridgey_quality_pool": len(ridgey_quality_pool),
        "released_control_sequences_excluded": len(control_sequences),
        "strict_ridgey_filter": "AF2 pLDDT>85, CA RMSD<2A, Ridgey 600M ensemble mean stability>WT and mean solubility>WT",
        "ridgey_consensus_ranking": "prioritize 5/5, then 4/5, then 3/5 paired members improving both; break ties by the second-worst joint relative member margin; diverse max-min selection within top 3x count",
    }
    write_json(ROOT / "selected" / "generated_73.summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
