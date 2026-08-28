#!/usr/bin/env python3
"""Assemble the final 97-protein TEV plate, app payload, and mutation PNG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from common import ROOT


BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYV"
BLOSUM62 = np.asarray([
    [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
    [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
    [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
    [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
    [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
    [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
    [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
    [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
    [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
    [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
    [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
], dtype=int)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: object) -> float | None:
    if value is None or str(value).strip() in ("", "nan", "None"):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def truth(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def parse_values(value: object) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        return [float(item) for item in json.loads(text)]
    separator = "|" if "|" in text else ";"
    return [float(item) for item in text.split(separator) if item]


def mutations(wt: str, sequence: str) -> list[str]:
    return [f"{a}{index}{b}" for index, (a, b) in enumerate(zip(wt, sequence), 1) if a != b]


def metric(value: float, wt: float, unit: str, good_direction: str = "higher", **extra: object) -> dict:
    delta = value - wt
    percent = None if abs(wt) < 1e-12 else 100.0 * delta / abs(wt)
    return {
        "value": value,
        "wt": wt,
        "delta": delta,
        "percent_change": percent,
        "unit": unit,
        "good_direction": good_direction,
        **extra,
    }


def paired_votes(stability: list[float], solubility: list[float], wt_stability: list[float], wt_solubility: list[float]) -> tuple[int, int, int]:
    stable = sum(value > baseline for value, baseline in zip(stability, wt_stability))
    soluble = sum(value > baseline for value, baseline in zip(solubility, wt_solubility))
    joint = sum(s > ws and q > wq for s, ws, q, wq in zip(stability, wt_stability, solubility, wt_solubility))
    return stable, soluble, joint


def copy_structure(source: str, destination: Path) -> None:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def control_record(row: dict, wt: str, baseline: dict, well: str, structure_dir: Path) -> tuple[dict, dict]:
    design_id = row["design_id"]
    sequence = row["sequence"]
    member_stability = parse_values(row["fresh_af2_ridgey_ensemble_stability_members_json"])
    member_solubility = parse_values(row["fresh_af2_ridgey_ensemble_solubility_members_json"])
    stable_votes, soluble_votes, joint_votes = paired_votes(
        member_stability, member_solubility,
        baseline["stability_members"], baseline["solubility_members"],
    )
    filename = f"{design_id}.pdb"
    copy_structure(row["fresh_af2_pdb"], structure_dir / filename)
    mut = mutations(wt, sequence)
    is_parent = design_id == "TEVd"
    tier = row["experimental_activity_tier"]
    exact_efficiency = finite(row.get("published_efficiency_fold_vs_TEVD"))
    activity_label = "parent reference" if is_parent else tier.replace("_", " ")
    if exact_efficiency is not None and not is_parent:
        activity_label += f" · published efficiency {exact_efficiency:.2f}× TEVd"
    app = {
        "id": design_id,
        "name": design_id,
        "well": well,
        "cohort": "TEVd" if is_parent else "released_m3",
        "selection_label": "TEVd parent" if is_parent else "released m3; activity tier inferred from Figure S7 trace order" if not truth(row["published_individual_kinetics_available"]) else "released m3 lead with published individual kinetics",
        "experimental_activity": finite(row.get("figureS7_trace_slope_relative_not_RFU_s")),
        "experimental_activity_label": activity_label,
        "experimental_details": {
            "tier": tier,
            "tier_evidence": row["experimental_tier_evidence"],
            "published_kcat_min_1": finite(row.get("published_kcat_min_1")),
            "published_Km_uM": finite(row.get("published_Km_uM")),
            "published_kcat_over_Km_uM_1_min_1": finite(row.get("published_kcat_over_Km_uM_1_min_1")),
            "published_efficiency_fold_vs_TEVD": exact_efficiency,
        },
        "sequence": sequence,
        "mutations": mut,
        "mutation_positions": [index for index, (a, b) in enumerate(zip(wt, sequence), 1) if a != b],
        "mutation_count": len(mut),
        "identity_to_wt": (len(wt) - len(mut)) / len(wt),
        "structure_key": f"structures/{filename}",
        "af2": {
            "mean_plddt": float(row["fresh_af2_plddt_mean"]),
            "global_ca_rmsd_vs_wt": float(row["fresh_af2_ca_rmsd_global_vs_1LVM_A"]),
            "protected_ca_rmsd_vs_wt": float(row["fresh_af2_ca_rmsd_m3_protected_after_global_align_A"]),
            "active_site_ca_rmsd_vs_wt": float(row["fresh_af2_ca_rmsd_active_site_after_global_align_A"]),
            "gate_pass": truth(row["fresh_af2_pipeline_gate_plddt85_rmsd2_pass"]),
            "failed_gates": [],
        },
        "metrics": {
            "ridgey_stability": metric(
                float(row["fresh_af2_ridgey_ensemble_stability_mean"]), baseline["stability_mean"], "kcal/mol",
                std=float(row["fresh_af2_ridgey_ensemble_stability_sample_sd"]), member_values=member_stability,
                votes_vs_wt=stable_votes, joint_votes_vs_wt=joint_votes, member_count=5,
            ),
            "ridgey_solubility": metric(
                float(row["fresh_af2_ridgey_ensemble_solubility_mean"]), baseline["solubility_mean"], "",
                std=float(row["fresh_af2_ridgey_ensemble_solubility_sample_sd"]), member_values=member_solubility,
                votes_vs_wt=soluble_votes, joint_votes_vs_wt=joint_votes, member_count=5,
            ),
            "mpnn_p_seq_wt_structure": metric(
                float(row["proteinmpnn_wt_structure_geomean_probability"]), baseline["mpnn_wt"], "geom. mean p"
            ),
            "mpnn_p_seq_mutant_structure": metric(
                float(row["fresh_af2_proteinmpnn_own_structure_geomean_probability"]), baseline["mpnn_own"], "geom. mean p"
            ),
        },
    }
    flat = {
        "well": well, "id": design_id, "display_name": design_id, "source_candidate_id": design_id,
        "cohort": app["cohort"], "selection_label": app["selection_label"], "experimental_activity_tier": tier,
        "sequence": sequence, "mutations": ";".join(mut), "mutation_count": len(mut), "identity_to_wt": app["identity_to_wt"],
        "af2_pdb_source": row["fresh_af2_pdb"], "af2_mean_plddt": app["af2"]["mean_plddt"],
        "global_ca_rmsd_vs_1lvm_A": app["af2"]["global_ca_rmsd_vs_wt"],
        "protected_ca_rmsd_vs_1lvm_A": app["af2"]["protected_ca_rmsd_vs_wt"],
        "active_site_ca_rmsd_vs_1lvm_A": app["af2"]["active_site_ca_rmsd_vs_wt"],
        "af2_gate_pass": app["af2"]["gate_pass"],
        "ridgey_600m_stability": float(row["fresh_af2_ridgey_600m_stability"]),
        "ridgey_600m_solubility": float(row["fresh_af2_ridgey_600m_solubility_probability"]),
        "ridgey_ensemble_stability_mean": app["metrics"]["ridgey_stability"]["value"],
        "ridgey_ensemble_stability_sd": app["metrics"]["ridgey_stability"]["std"],
        "ridgey_ensemble_solubility_mean": app["metrics"]["ridgey_solubility"]["value"],
        "ridgey_ensemble_solubility_sd": app["metrics"]["ridgey_solubility"]["std"],
        "ridgey_ensemble_stability_members": json.dumps(member_stability),
        "ridgey_ensemble_solubility_members": json.dumps(member_solubility),
        "ridgey_ensemble_stability_votes_vs_wt": stable_votes,
        "ridgey_ensemble_solubility_votes_vs_wt": soluble_votes,
        "ridgey_ensemble_joint_votes_vs_wt": joint_votes,
        "mpnn_wt_structure_nll": float(row["proteinmpnn_wt_structure_nll_mean_16order"]),
        "mpnn_wt_structure_geomean_probability": app["metrics"]["mpnn_p_seq_wt_structure"]["value"],
        "mpnn_mutant_structure_nll": float(row["fresh_af2_proteinmpnn_own_structure_nll_mean_16order"]),
        "mpnn_mutant_structure_geomean_probability": app["metrics"]["mpnn_p_seq_mutant_structure"]["value"],
        "published_kcat_min_1": finite(row.get("published_kcat_min_1")),
        "published_Km_uM": finite(row.get("published_Km_uM")),
        "published_efficiency_fold_vs_TEVD": exact_efficiency,
    }
    return app, flat


def generated_record(row: dict, wt: str, baseline: dict, cohort: str, number: int, well: str, structure_dir: Path) -> tuple[dict, dict]:
    candidate_id = row["candidate_id"]
    display_name = f"{'Ridgey' if cohort == 'ridgey' else 'ProteinMPNN'}-{number:02d}"
    filename = f"{display_name}.pdb"
    copy_structure(row["af2_pdb"], structure_dir / filename)
    sequence = row["sequence"]
    mut = mutations(wt, sequence)
    member_stability = parse_values(row["ridgey_ensemble_stability_members"])
    member_solubility = parse_values(row["ridgey_ensemble_solubility_members"])
    stable_votes, soluble_votes, joint_votes = paired_votes(
        member_stability, member_solubility,
        baseline["stability_members"], baseline["solubility_members"],
    )
    stability_mean = float(row["ridgey_ensemble_stability_mean"])
    solubility_mean = float(row["ridgey_ensemble_solubility_mean"])
    stability_sd = finite(row.get("ridgey_ensemble_stability_std"))
    solubility_sd = finite(row.get("ridgey_ensemble_solubility_std"))
    if stability_sd is None:
        stability_sd = float(np.std(member_stability, ddof=1))
    if solubility_sd is None:
        solubility_sd = float(np.std(member_solubility, ddof=1))
    failed = []
    if float(row["af2_mean_plddt"]) <= 85:
        failed.append("mean pLDDT <=85")
    if float(row["af2_ca_rmsd_to_1lvm_angstrom"]) >= 2:
        failed.append("global CA RMSD >=2A")
    label = (
        f"ensemble means > WT; paired consensus {joint_votes}/5; diversity-selected"
        if cohort == "ridgey" else "paper-matched ProteinMPNN recipe; AF2-pass diversity control"
    )
    app = {
        "id": candidate_id,
        "name": display_name,
        "well": well,
        "cohort": cohort,
        "selection_label": label,
        "experimental_activity": None,
        "experimental_activity_label": "not experimentally measured",
        "source_candidate_id": candidate_id,
        "sequence": sequence,
        "mutations": mut,
        "mutation_positions": [index for index, (a, b) in enumerate(zip(wt, sequence), 1) if a != b],
        "mutation_count": len(mut),
        "identity_to_wt": (len(wt) - len(mut)) / len(wt),
        "structure_key": f"structures/{filename}",
        "af2": {
            "mean_plddt": float(row["af2_mean_plddt"]),
            "global_ca_rmsd_vs_wt": float(row["af2_ca_rmsd_to_1lvm_angstrom"]),
            "protected_ca_rmsd_vs_wt": float(row["af2_protected_ca_rmsd_to_1lvm_angstrom"]),
            "active_site_ca_rmsd_vs_wt": float(row["af2_active_site_ca_rmsd_to_1lvm_angstrom"]),
            "gate_pass": truth(row["af2_pass"]),
            "failed_gates": failed,
        },
        "metrics": {
            "ridgey_stability": metric(
                stability_mean, baseline["stability_mean"], "kcal/mol", std=stability_sd,
                member_values=member_stability, votes_vs_wt=stable_votes, joint_votes_vs_wt=joint_votes, member_count=5,
            ),
            "ridgey_solubility": metric(
                solubility_mean, baseline["solubility_mean"], "", std=solubility_sd,
                member_values=member_solubility, votes_vs_wt=soluble_votes, joint_votes_vs_wt=joint_votes, member_count=5,
            ),
            "mpnn_p_seq_wt_structure": metric(
                float(row["mpnn_wt_structure_geomean_probability"]), baseline["mpnn_wt"], "geom. mean p"
            ),
            "mpnn_p_seq_mutant_structure": metric(
                float(row["mpnn_mutant_structure_geomean_probability"]), baseline["mpnn_own"], "geom. mean p"
            ),
        },
    }
    flat = {
        "well": well, "id": candidate_id, "display_name": display_name, "source_candidate_id": candidate_id,
        "cohort": cohort, "selection_label": label, "experimental_activity_tier": "not_measured",
        "sequence": sequence, "mutations": ";".join(mut), "mutation_count": len(mut), "identity_to_wt": app["identity_to_wt"],
        "af2_pdb_source": row["af2_pdb"], "af2_mean_plddt": app["af2"]["mean_plddt"],
        "global_ca_rmsd_vs_1lvm_A": app["af2"]["global_ca_rmsd_vs_wt"],
        "protected_ca_rmsd_vs_1lvm_A": app["af2"]["protected_ca_rmsd_vs_wt"],
        "active_site_ca_rmsd_vs_1lvm_A": app["af2"]["active_site_ca_rmsd_vs_wt"],
        "af2_gate_pass": app["af2"]["gate_pass"],
        "ridgey_600m_stability": float(row["ridgey_600m_stability"]), "ridgey_600m_solubility": float(row["ridgey_600m_solubility"]),
        "ridgey_ensemble_stability_mean": stability_mean, "ridgey_ensemble_stability_sd": stability_sd,
        "ridgey_ensemble_solubility_mean": solubility_mean, "ridgey_ensemble_solubility_sd": solubility_sd,
        "ridgey_ensemble_stability_members": json.dumps(member_stability), "ridgey_ensemble_solubility_members": json.dumps(member_solubility),
        "ridgey_ensemble_stability_votes_vs_wt": stable_votes, "ridgey_ensemble_solubility_votes_vs_wt": soluble_votes,
        "ridgey_ensemble_joint_votes_vs_wt": joint_votes,
        "mpnn_wt_structure_nll": float(row["mpnn_wt_structure_nll"]),
        "mpnn_wt_structure_geomean_probability": app["metrics"]["mpnn_p_seq_wt_structure"]["value"],
        "mpnn_mutant_structure_nll": float(row["mpnn_mutant_structure_nll"]),
        "mpnn_mutant_structure_geomean_probability": app["metrics"]["mpnn_p_seq_mutant_structure"]["value"],
    }
    return app, flat


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mutation_matrix(path: Path, designs: list[dict], wt: str) -> None:
    matrix = np.zeros((len(designs), len(wt)), dtype=int)
    for row_index, design in enumerate(designs):
        for column, (parent, mutant) in enumerate(zip(wt, design["sequence"])):
            if parent == mutant:
                continue
            i, j = BLOSUM_ORDER.index(parent), BLOSUM_ORDER.index(mutant)
            matrix[row_index, column] = 1 if BLOSUM62[i, j] >= 0 else 2
    figure, axis = plt.subplots(figsize=(18, 17), constrained_layout=True)
    axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#111827", "#2563eb", "#dc2626"]), vmin=0, vmax=2)
    axis.set_title("TEV redesign plate mutation matrix", loc="left", fontsize=17, fontweight="bold", pad=14)
    axis.set_xlabel("TEVd residue number", fontsize=12)
    axis.set_ylabel("Design", fontsize=12)
    # Keep the terminal 220/221 labels from colliding on a 221-residue axis.
    ticks = [0] + list(range(9, len(wt), 10))
    axis.set_xticks(ticks, [str(index + 1) for index in ticks], fontsize=8)
    axis.set_yticks(range(len(designs)), [f"{row['well']}  {row['name']}" for row in designs], fontsize=6.5)
    axis.tick_params(length=0)
    for boundary in (1, 25, 61):
        axis.axhline(boundary - 0.5, color="white", linewidth=1.5)
    axis.legend(
        handles=[Patch(color="#111827", label="same as TEVd"), Patch(color="#2563eb", label="conservative (BLOSUM62 >= 0)"), Patch(color="#dc2626", label="radical (BLOSUM62 < 0)")],
        loc="upper center", bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False, fontsize=10,
    )
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-csv", required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()

    output = ROOT / "deliverables"
    structure_dir = output / "structures"
    output.mkdir(parents=True, exist_ok=True)
    controls = read_csv(ROOT / "work" / "control_characterization" / "released_controls_final_25.csv")
    generated = read_csv(Path(args.generated_csv))
    wt_row = next(row for row in controls if row["design_id"] == "TEVd")
    wt = wt_row["sequence"]
    wt_stability_members = parse_values(wt_row["fresh_af2_ridgey_ensemble_stability_members_json"])
    wt_solubility_members = parse_values(wt_row["fresh_af2_ridgey_ensemble_solubility_members_json"])
    baseline = {
        "stability_members": wt_stability_members,
        "solubility_members": wt_solubility_members,
        "stability_mean": float(wt_row["fresh_af2_ridgey_ensemble_stability_mean"]),
        "solubility_mean": float(wt_row["fresh_af2_ridgey_ensemble_solubility_mean"]),
        "mpnn_wt": float(wt_row["proteinmpnn_wt_structure_geomean_probability"]),
        "mpnn_own": float(wt_row["fresh_af2_proteinmpnn_own_structure_geomean_probability"]),
    }

    apps: list[dict] = []
    flats: list[dict] = []
    parent_app, parent_flat = control_record(wt_row, wt, baseline, "A01", structure_dir)
    apps.append(parent_app); flats.append(parent_flat)
    released = [row for row in controls if row["design_id"] != "TEVd"]
    if len(released) != 24:
        raise ValueError(f"expected 24 released controls, found {len(released)}")
    for index, row in enumerate(released, 1):
        app, flat = control_record(row, wt, baseline, f"B{index:02d}", structure_dir)
        apps.append(app); flats.append(flat)

    by_method = {
        "ridgey": [row for row in generated if row["method"] == "ridgey"],
        "proteinmpnn": [row for row in generated if row["method"] == "proteinmpnn"],
    }
    if {key: len(value) for key, value in by_method.items()} != {"ridgey": 36, "proteinmpnn": 36}:
        raise ValueError(f"expected 36+36 generated designs, found { {key: len(value) for key, value in by_method.items()} }")
    for cohort, start_row in (("ridgey", "C"), ("proteinmpnn", "E")):
        for index, row in enumerate(by_method[cohort], 1):
            plate_row = start_row if index <= 24 else chr(ord(start_row) + 1)
            column = index if index <= 24 else index - 24
            app, flat = generated_record(row, wt, baseline, cohort, index, f"{plate_row}{column:02d}", structure_dir)
            apps.append(app); flats.append(flat)

    if len(apps) != 97 or len({row["sequence"] for row in apps}) != 97 or any(len(row["sequence"]) != 221 for row in apps):
        raise ValueError("final plate must contain 97 unique 221-aa sequences")
    if not all(row["af2"]["gate_pass"] for row in apps):
        raise ValueError("every selected plate design must pass the AF2 gate")
    ridgey_apps = [row for row in apps if row["cohort"] == "ridgey"]
    if not all(row["metrics"]["ridgey_stability"]["value"] > baseline["stability_mean"] and row["metrics"]["ridgey_solubility"]["value"] > baseline["solubility_mean"] for row in ridgey_apps):
        raise ValueError("every Ridgey design must beat WT ensemble mean for stability and solubility")

    payload = {
        "project": "tev-redesign-plate",
        "schema_version": "1.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "repository": "https://github.com/sutterhill/tev_plate_design",
            "analysis_commit": args.analysis_commit,
            "analysis_commit_url": f"https://github.com/sutterhill/tev_plate_design/commit/{args.analysis_commit}",
            "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10811672/",
        },
        "methods": {
            "ridgey": "Ridgey v2 600M five-member ensemble on each AF2 structure; base 600M also retained in the archival table",
            "proteinmpnn": "ProteinMPNN v_48_020; geometric-mean per-residue probability from 16 decoding orders",
            "folding": "AlphaFold2 model 3, six recycles, parent-MSA query swap; no explicit 1LVM template",
            "constraints": "m3: active site + top 50% conserved fixed (127 fixed, 94 mutable); no introduced cysteine",
            "ridgey_selection_eligibility": "AF2 pass and five-member ensemble mean strictly above TEVd for both stability and solubility.",
            "ridgey_selection_fallback": "No score threshold was relaxed. Eligible designs were ranked by paired-member consensus (5/5, then 4/5, then 3/5, etc.) before deterministic sequence-diversity selection.",
            "released_activity": "Only hyperTEV56/60/89 have individual published kinetics; other active/somewhat/inactive labels are trace-order inferences from Figure S7 and are not RFU/s.",
        },
        "parent": {
            "name": "TEVd", "length": len(wt), "sequence": wt,
            "structure_key": parent_app["structure_key"],
            "metrics": {
                "ridgey_stability": baseline["stability_mean"], "ridgey_solubility": baseline["solubility_mean"],
                "mpnn_p_seq_wt_structure": baseline["mpnn_wt"], "mpnn_p_seq_mutant_structure": baseline["mpnn_own"],
            },
        },
        "designs": apps,
    }
    (output / "plate.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(output / "tev_plate_97.csv", flats)
    with (output / "tev_plate_97.fasta").open("w") as handle:
        for row in apps:
            handle.write(f">{row['well']}|{row['name']}|{row['cohort']}|{row['id']}\n{row['sequence']}\n")
    mutation_matrix(output / "tev_plate_mutation_matrix.png", apps, wt)
    summary = {
        "rows": len(apps),
        "cohorts": {cohort: sum(row["cohort"] == cohort for row in apps) for cohort in ("TEVd", "released_m3", "ridgey", "proteinmpnn")},
        "ridgey_joint_vote_counts": {str(vote): sum(row["metrics"]["ridgey_stability"]["joint_votes_vs_wt"] == vote for row in ridgey_apps) for vote in range(6)},
        "all_af2_pass": all(row["af2"]["gate_pass"] for row in apps),
        "all_ridgey_means_better_than_wt": True,
    }
    (output / "VALIDATION.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("w") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(output)}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
