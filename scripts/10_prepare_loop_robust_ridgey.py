#!/usr/bin/env python3
"""Create a loop-excluded, mutation-DDG-filtered Ridgey AF2 candidate pool.

The global Ridgey stability head and the DDG head are both trained on
proteolysis-derived measurements.  To keep loop accessibility from acting as a
surrogate for thermodynamic stability, this stage:

1. assigns TEV residues as structured (PDB HELIX/SHEET records) or loop/coil;
2. evaluates every WT->candidate substitution using the five-member Ridgey DDG
   matrix on the common 1LVM/TEVd structure;
3. leaves loop substitutions untouched but excludes them from stability score;
4. retains a structured-region substitution only when its improving-is-positive
   DDG delta is positive on the ensemble mean and in at least four of five
   members; otherwise it is reverted to WT; and
5. uses pre-edit own-structure solubility only to enrich a tractable AF2 pool.

All edited candidates are refolded and rescored downstream.  The pre-edit
solubility value is never reported as the edited sequence's final value.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from common import (
    REANALYSIS,
    ROOT,
    hamming,
    read_csv,
    sequence_features,
    validate_m3_sequence,
    write_csv,
    write_fasta,
    write_json,
)


RUNS = ("production", "ridgey_extension", "ridgey_enriched")


def parse_members(value: object) -> np.ndarray:
    text = str(value).strip()
    if text.startswith("["):
        return np.asarray(json.loads(text), dtype=float)
    return np.asarray([float(item) for item in text.split("|")], dtype=float)


def structured_positions(pdb_path: Path, chain_id: str = "A") -> set[int]:
    """Read author/DSSP HELIX and SHEET spans from the original PDB header."""
    positions: set[int] = set()
    for line in pdb_path.read_text().splitlines():
        try:
            if line.startswith("HELIX"):
                start_chain, end_chain = line[19], line[31]
                start, end = int(line[21:25]), int(line[33:37])
            elif line.startswith("SHEET"):
                start_chain, end_chain = line[21], line[32]
                start, end = int(line[22:26]), int(line[33:37])
            else:
                continue
        except (ValueError, IndexError):
            continue
        if start_chain == end_chain == chain_id:
            positions.update(range(max(1, start), min(221, end) + 1))
    if not positions:
        raise RuntimeError(f"no HELIX/SHEET assignments found for chain {chain_id} in {pdb_path}")
    return positions


def load_tevd_ensemble() -> dict:
    path = REANALYSIS / "results" / "ridgey" / "predictions_600m_ensemble.json.gz"
    with gzip.open(path, "rt") as handle:
        rows = json.load(handle)
    parent = next(row for row in rows if row["name"] == "TEVd")
    if parent["structure_source"] != "provided":
        raise RuntimeError(f"expected a provided TEVd structure, got {parent['structure_source']}")
    if len(parent["predictions"]) != 5:
        raise RuntimeError("expected five Ridgey ensemble members")
    parent["_source_path"] = str(path)
    return parent


def unique_characterized_rows() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for run_name in RUNS:
        path = ROOT / "scores" / f"characterized_{run_name}.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            if row["sequence"] in seen:
                continue
            row["source_characterization_run"] = run_name
            rows.append(row)
            seen.add(row["sequence"])
    return rows


def maximin_extension(selected: list[dict], remaining: list[dict], count: int) -> list[dict]:
    """Add diverse candidates from a quality-enriched fallback pool."""
    chosen = list(selected)
    candidates = list(remaining)
    while candidates and len(chosen) < count:
        next_row = max(
            candidates,
            key=lambda row: (
                min(hamming(row["sequence"], ref["sequence"]) for ref in chosen),
                float(row["precursor_solubility_lcb_delta_vs_wt"]),
                float(row["loop_robust_core_ddg_lcb"]),
                row["candidate_id"],
            ),
        )
        chosen.append(next_row)
        candidates.remove(next_row)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--min-ddg-member-votes", type=int, default=4)
    parser.add_argument("--lcb-sd-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    source_pdb = REANALYSIS / "source_data" / "1LVM.pdb"
    structured = structured_positions(source_pdb)
    loops = set(range(1, 222)) - structured
    parent = load_tevd_ensemble()
    wt = parent["sequence"]
    models = parent["prediction_models"]
    aa_order = parent["predictions"][0]["ddg_amino_acid_order"]
    aa_index = {aa: index for index, aa in enumerate(aa_order)}
    ddg = np.asarray([member["ddg"] for member in parent["predictions"]], dtype=float)
    if ddg.shape != (5, len(wt), 20):
        raise RuntimeError(f"unexpected DDG shape {ddg.shape}")

    rows = unique_characterized_rows()
    parent_row = next(row for row in rows if row["method"] == "parent")
    wt_solubility = parse_members(parent_row["ridgey_ensemble_solubility_members"])
    wt_solubility_mean = float(wt_solubility.mean())
    released_path = ROOT / "work" / "control_characterization" / "released_controls_final_25.csv"
    released_sequences = {row["sequence"] for row in read_csv(released_path)} if released_path.exists() else {wt}

    edited_by_sequence: dict[str, dict] = {}
    rejection_counts: dict[str, int] = {}
    source_pool = [
        row for row in rows
        if row["method"] == "ridgey"
        and str(row["af2_pass"]).lower() == "true"
        and row["sequence"] not in released_sequences
    ]
    for source in source_pool:
        original = source["sequence"]
        edited = list(original)
        structured_kept: list[str] = []
        loop_kept: list[str] = []
        reverted: list[str] = []
        retained_member_deltas: list[np.ndarray] = []
        mutation_details: list[dict] = []
        for index, (wt_aa, mutant_aa) in enumerate(zip(wt, original)):
            if wt_aa == mutant_aa:
                continue
            position = index + 1
            if position in loops:
                loop_kept.append(f"{wt_aa}{position}{mutant_aa}")
                mutation_details.append({
                    "mutation": f"{wt_aa}{position}{mutant_aa}",
                    "secondary_structure_class": "loop_or_coil",
                    "used_for_stability_selection": False,
                    "action": "retained_without_stability_credit",
                })
                continue
            member_delta = ddg[:, index, aa_index[wt_aa]] - ddg[:, index, aa_index[mutant_aa]]
            mean_delta = float(member_delta.mean())
            votes = int((member_delta > 0).sum())
            keep = mean_delta > 0 and votes >= args.min_ddg_member_votes
            detail = {
                "mutation": f"{wt_aa}{position}{mutant_aa}",
                "secondary_structure_class": "helix_or_sheet",
                "used_for_stability_selection": True,
                "ddg_improvement_members": member_delta.tolist(),
                "ddg_improvement_mean": mean_delta,
                "ddg_improving_member_votes": votes,
                "action": "retained" if keep else "reverted_to_wt",
            }
            mutation_details.append(detail)
            if keep:
                structured_kept.append(f"{wt_aa}{position}{mutant_aa}")
                retained_member_deltas.append(member_delta)
            else:
                edited[index] = wt_aa
                reverted.append(f"{wt_aa}{position}{mutant_aa}")

        sequence = "".join(edited)
        # Reversion may restore TEVd's native cysteines at mutable positions.
        # Permit those WT identities while continuing to reject genuinely new
        # cysteines, which preserves the intent of the no-new-C constraint.
        reasons = validate_m3_sequence(sequence, forbid_c_at_mutable=False)
        novel_cysteines = [
            index + 1 for index, (wt_aa, aa) in enumerate(zip(wt, sequence))
            if aa == "C" and wt_aa != "C"
        ]
        if novel_cysteines:
            reasons.append("novel_C:" + ";".join(map(str, novel_cysteines)))
        if reasons:
            for reason in reasons:
                rejection_counts[reason.split(":", 1)[0]] = rejection_counts.get(reason.split(":", 1)[0], 0) + 1
            continue
        if sequence in released_sequences or sequence == wt:
            rejection_counts["released_or_parent_sequence"] = rejection_counts.get("released_or_parent_sequence", 0) + 1
            continue

        if retained_member_deltas:
            total_members = np.asarray(retained_member_deltas, dtype=float).sum(axis=0)
        else:
            total_members = np.zeros(5, dtype=float)
        total_mean = float(total_members.mean())
        total_std = float(total_members.std(ddof=1))
        total_lcb = total_mean - args.lcb_sd_multiplier * total_std
        precursor_solubility = parse_members(source["ridgey_ensemble_solubility_members"])
        precursor_delta = precursor_solubility - wt_solubility
        precursor_lcb = float(precursor_delta.mean() - args.lcb_sd_multiplier * precursor_delta.std(ddof=1))
        ident = source["candidate_id"].replace("ridgey_", "", 1)
        row = {
            "candidate_id": f"ridgey_lr_{ident}",
            "method": "ridgey",
            "design_strategy": "ridgey_inverse_fold_then_loop_excluded_ddg_filter",
            "source_candidate_id": source["candidate_id"],
            "source_characterization_run": source["source_characterization_run"],
            **sequence_features(sequence),
            "source_sequence": original,
            "source_n_mutations": sum(a != b for a, b in zip(wt, original)),
            "loop_mutations_retained": ";".join(loop_kept),
            "loop_mutation_count": len(loop_kept),
            "structured_mutations_retained": ";".join(structured_kept),
            "structured_mutation_count": len(structured_kept),
            "structured_mutations_reverted": ";".join(reverted),
            "structured_mutations_reverted_count": len(reverted),
            "structured_destabilizing_mutations_included": 0,
            "loop_robust_core_ddg_members": json.dumps(total_members.tolist()),
            "loop_robust_core_ddg_mean": total_mean,
            "loop_robust_core_ddg_std": total_std,
            "loop_robust_core_ddg_lcb": total_lcb,
            "loop_robust_ddg_member_models": json.dumps(models),
            "loop_robust_ddg_min_improving_votes_per_mutation": args.min_ddg_member_votes,
            "loop_robust_ddg_lcb_sd_multiplier": args.lcb_sd_multiplier,
            "loop_robust_mutation_details": json.dumps(mutation_details, separators=(",", ":")),
            "secondary_structure_reference": str(source_pdb),
            "ddg_reference": parent["_source_path"],
            "ddg_reference_structure_source": parent["structure_source"],
            "precursor_solubility_members": json.dumps(precursor_solubility.tolist()),
            "precursor_solubility_mean": float(precursor_solubility.mean()),
            "precursor_solubility_delta_vs_wt": float(precursor_delta.mean()),
            "precursor_solubility_lcb_delta_vs_wt": precursor_lcb,
            "precursor_solubility_improving_votes_vs_wt": int((precursor_delta > 0).sum()),
            "precursor_score_used_for_prefold_enrichment_only": True,
        }
        incumbent = edited_by_sequence.get(sequence)
        if incumbent is None or (
            row["precursor_solubility_lcb_delta_vs_wt"], row["loop_robust_core_ddg_lcb"], row["candidate_id"]
        ) > (
            incumbent["precursor_solubility_lcb_delta_vs_wt"], incumbent["loop_robust_core_ddg_lcb"], incumbent["candidate_id"]
        ):
            edited_by_sequence[sequence] = row

    all_rows = list(edited_by_sequence.values())
    high_solubility = sorted(
        [row for row in all_rows if float(row["precursor_solubility_delta_vs_wt"]) > 0],
        key=lambda row: (
            -int(row["precursor_solubility_improving_votes_vs_wt"]),
            -float(row["precursor_solubility_lcb_delta_vs_wt"]),
            -float(row["loop_robust_core_ddg_lcb"]),
            row["candidate_id"],
        ),
    )
    if len(high_solubility) > args.count:
        high_solubility = high_solubility[: args.count]
    selected_sequences = {row["sequence"] for row in high_solubility}
    need = args.count - len(high_solubility)
    fallback_ranked = sorted(
        [row for row in all_rows if row["sequence"] not in selected_sequences],
        key=lambda row: (
            -float(row["precursor_solubility_lcb_delta_vs_wt"]),
            -int(row["precursor_solubility_improving_votes_vs_wt"]),
            -float(row["loop_robust_core_ddg_lcb"]),
            row["candidate_id"],
        ),
    )[: max(need * 3, need)]
    selected = maximin_extension(high_solubility, fallback_ranked, args.count) if need else high_solubility
    if len(selected) < args.count:
        raise RuntimeError(f"only {len(selected)} loop-robust candidates available; requested {args.count}")
    for rank, row in enumerate(selected, 1):
        row["loop_robust_prefold_rank"] = rank
        row["loop_robust_prefold_selection"] = (
            "all precursor-solubility-mean-above-WT candidates"
            if float(row["precursor_solubility_delta_vs_wt"]) > 0
            else "quality-enriched max-min diversity fallback"
        )

    write_csv(ROOT / "candidates" / "ridgey_loop_robust_all.csv", sorted(all_rows, key=lambda row: row["candidate_id"]))
    write_csv(ROOT / "candidates" / "ridgey_loop_robust_valid_unique.csv", selected)
    write_fasta(ROOT / "candidates" / "ridgey_loop_robust_valid_unique.fasta", [(row["candidate_id"], row["sequence"]) for row in selected])
    write_json(
        ROOT / "manifests" / "ridgey_loop_robust_preparation.json",
        {
            "source_ridgey_af2_pass": len(source_pool),
            "edited_unique": len(all_rows),
            "selected_for_refolding": len(selected),
            "selected_with_precursor_solubility_mean_above_wt": sum(float(row["precursor_solubility_delta_vs_wt"]) > 0 for row in selected),
            "structured_residue_count": len(structured),
            "loop_or_coil_residue_count": len(loops),
            "structured_positions_1_indexed": sorted(structured),
            "loop_or_coil_positions_1_indexed": sorted(loops),
            "structured_mutation_rule": f"ensemble mean DDG improvement >0 and at least {args.min_ddg_member_votes}/5 members >0; otherwise revert to WT",
            "loop_mutation_rule": "retain without stability credit",
            "cysteine_rule": "no novel cysteine; WT cysteines may be restored by DDG reversion",
            "global_proteolysis_stability_used_for_selection": False,
            "prefold_enrichment": "pre-edit own-AF2-structure Ridgey ensemble solubility only; final values require refolding and rescoring",
            "ddg_reference": parent["_source_path"],
            "ddg_models": models,
            "secondary_structure_reference": str(source_pdb),
            "rejection_counts": rejection_counts,
        },
    )
    print(json.dumps({
        "source_ridgey_af2_pass": len(source_pool),
        "edited_unique": len(all_rows),
        "selected_for_refolding": len(selected),
        "structured_positions": len(structured),
        "loop_or_coil_positions": len(loops),
        "mean_final_mutations": float(np.mean([int(row["n_mutations"]) for row in selected])),
        "mean_structured_mutations_retained": float(np.mean([int(row["structured_mutation_count"]) for row in selected])),
        "mean_loop_mutations_retained": float(np.mean([int(row["loop_mutation_count"]) for row in selected])),
    }, indent=2))


if __name__ == "__main__":
    main()
