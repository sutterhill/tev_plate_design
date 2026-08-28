#!/usr/bin/env python3
"""Select 36 loop-robust Ridgey designs and retain the ProteinMPNN controls."""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import ROOT, hamming, read_csv, write_csv, write_fasta, write_json


def truth(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def members(value: object) -> np.ndarray:
    text = str(value).strip()
    if text.startswith("["):
        return np.asarray(json.loads(text), dtype=float)
    return np.asarray([float(item) for item in text.split("|")], dtype=float)


def percentile(values: np.ndarray) -> np.ndarray:
    """Stable 0..1 ordinal percentiles; larger values are always better."""
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    if len(values) == 1:
        result[0] = 1.0
        return result
    result[order] = np.arange(len(values), dtype=float) / (len(values) - 1)
    return result


def diverse_select(rows: list[dict], count: int) -> list[dict]:
    remaining = sorted(rows, key=lambda row: row["candidate_id"])
    first = max(
        remaining,
        key=lambda row: (
            float(row["loop_robust_balanced_quality"]),
            float(row["fresh_solubility_lcb_delta_vs_wt"]),
            float(row["loop_robust_core_ddg_per_mutation_lcb"]),
            row["candidate_id"],
        ),
    )
    selected = [first]
    remaining.remove(first)
    min_distance = {row["candidate_id"]: hamming(row["sequence"], first["sequence"]) for row in remaining}
    while remaining and len(selected) < count:
        next_row = max(
            remaining,
            key=lambda row: (
                min_distance[row["candidate_id"]],
                float(row["loop_robust_balanced_quality"]),
                float(row["fresh_solubility_lcb_delta_vs_wt"]),
                float(row["loop_robust_core_ddg_per_mutation_lcb"]),
                row["candidate_id"],
            ),
        )
        selected.append(next_row)
        remaining.remove(next_row)
        for row in remaining:
            min_distance[row["candidate_id"]] = min(
                min_distance[row["candidate_id"]],
                hamming(row["sequence"], next_row["sequence"]),
            )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--quality-pool-multiplier", type=int, default=3)
    args = parser.parse_args()

    characterized = read_csv(ROOT / "scores" / "characterized_loop_robust.csv")
    parent = next(row for row in characterized if row["method"] == "parent")
    wt_solubility = members(parent["ridgey_ensemble_solubility_members"])
    candidates: list[dict] = []
    for row in characterized:
        if row["method"] != "ridgey" or not truth(row["af2_pass"]):
            continue
        if int(float(row["structured_destabilizing_mutations_included"])) != 0:
            continue
        solubility = members(row["ridgey_ensemble_solubility_members"])
        delta = solubility - wt_solubility
        delta_mean = float(delta.mean())
        delta_std = float(delta.std(ddof=1))
        delta_lcb = delta_mean - delta_std
        core_count = int(float(row["structured_mutation_count"]))
        core_lcb = float(row["loop_robust_core_ddg_lcb"])
        if delta_mean <= 0 or delta_lcb <= 0 or core_lcb <= 0:
            continue
        row.update({
            "fresh_solubility_delta_members_vs_wt": json.dumps(delta.tolist()),
            "fresh_solubility_delta_mean_vs_wt": delta_mean,
            "fresh_solubility_delta_std_vs_wt": delta_std,
            "fresh_solubility_lcb_delta_vs_wt": delta_lcb,
            "fresh_solubility_improving_votes_vs_wt": int((delta > 0).sum()),
            "loop_robust_core_ddg_per_mutation_lcb": core_lcb / max(core_count, 1),
            "global_proteolysis_stability_used_for_selection": False,
        })
        candidates.append(row)
    if len(candidates) < args.count:
        raise RuntimeError(f"only {len(candidates)} candidates pass AF2 + solubility LCB + loop-robust DDG; need {args.count}")

    sol_percentile = percentile(np.asarray([float(row["fresh_solubility_lcb_delta_vs_wt"]) for row in candidates]))
    ddg_percentile = percentile(np.asarray([float(row["loop_robust_core_ddg_per_mutation_lcb"]) for row in candidates]))
    for row, sol_rank, ddg_rank in zip(candidates, sol_percentile, ddg_percentile):
        # The minimum percentile prevents a high value on one endpoint from
        # compensating for a poor value on the other.
        row["loop_robust_solubility_percentile"] = float(sol_rank)
        row["loop_robust_core_ddg_percentile"] = float(ddg_rank)
        row["loop_robust_balanced_quality"] = float(min(sol_rank, ddg_rank))

    quality_pool_count = min(len(candidates), args.count * args.quality_pool_multiplier)
    quality_pool = sorted(
        candidates,
        key=lambda row: (
            -int(row["fresh_solubility_improving_votes_vs_wt"]),
            -float(row["loop_robust_balanced_quality"]),
            -float(row["fresh_solubility_lcb_delta_vs_wt"]),
            -float(row["loop_robust_core_ddg_per_mutation_lcb"]),
            row["candidate_id"],
        ),
    )[:quality_pool_count]
    selected_ridgey = diverse_select(quality_pool, args.count)
    for rank, row in enumerate(selected_ridgey, 1):
        row["final_selection"] = "loop_excluded_ddg_and_solubility_lcb_then_diversity"
        row["final_selection_rank"] = rank

    previous = read_csv(ROOT / "selected" / "generated_73.csv")
    proteinmpnn = [row for row in previous if row["method"] == "proteinmpnn"]
    if len(proteinmpnn) != args.count:
        raise RuntimeError(f"expected {args.count} retained ProteinMPNN controls, found {len(proteinmpnn)}")
    generated = [parent] + selected_ridgey + proteinmpnn
    if len({row["sequence"] for row in generated}) != 1 + 2 * args.count:
        raise RuntimeError("generated parent/Ridgey/ProteinMPNN sequences are not unique")

    old_ridgey = [row for row in previous if row["method"] == "ridgey"]
    old_sources = {row["candidate_id"] for row in old_ridgey}
    selected_sources = {row["source_candidate_id"] for row in selected_ridgey}
    pairwise = [hamming(a["sequence"], b["sequence"]) for index, a in enumerate(selected_ridgey) for b in selected_ridgey[index + 1:]]
    write_csv(ROOT / "selected" / "ridgey_loop_robust_36.csv", selected_ridgey)
    write_csv(ROOT / "selected" / "generated_73_loop_robust.csv", generated)
    write_fasta(ROOT / "selected" / "generated_73_loop_robust.fasta", [(row["candidate_id"], row["sequence"]) for row in generated])
    summary = {
        "fresh_characterized_ridgey": sum(row["method"] == "ridgey" for row in characterized),
        "eligible_af2_solubility_lcb_and_core_ddg": len(candidates),
        "quality_pool": len(quality_pool),
        "selected_ridgey": len(selected_ridgey),
        "retained_proteinmpnn_controls": len(proteinmpnn),
        "selected_precursors_in_old_ridgey_cohort": len(selected_sources & old_sources),
        "exact_sequence_overlap_with_old_ridgey_cohort": len({row["sequence"] for row in selected_ridgey} & {row["sequence"] for row in old_ridgey}),
        "mean_mutations": float(np.mean([int(float(row["n_mutations"])) for row in selected_ridgey])),
        "mean_structured_mutations": float(np.mean([int(float(row["structured_mutation_count"])) for row in selected_ridgey])),
        "mean_loop_mutations": float(np.mean([int(float(row["loop_mutation_count"])) for row in selected_ridgey])),
        "mean_pairwise_hamming": float(np.mean(pairwise)),
        "min_pairwise_hamming": int(min(pairwise)),
        "solubility_vote_counts": {str(vote): sum(int(row["fresh_solubility_improving_votes_vs_wt"]) == vote for row in selected_ridgey) for vote in range(6)},
        "selection": "AF2 pass; no structured destabilizing mutations; fresh solubility paired-member mean-SD > WT; core DDG mean-SD >0; 5-member vote and balanced-quality pool; max-min sequence diversity",
        "global_proteolysis_stability_used_for_selection": False,
    }
    write_json(ROOT / "selected" / "ridgey_loop_robust_36.summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
