#!/usr/bin/env python3
"""Build the fully characterized TEVd + 24 released-m3 control table.

Requires numpy. On aws0 this was run with:
  /home/ubuntu/ridgey_solubility_analysis/.venv/bin/python characterize_controls.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


REANALYSIS = Path("/opt/dlami/nvme/tev_reanalysis")
CONTROL_SELECTION = Path("/opt/dlami/nvme/tev_plate_design/work/m3_controls")
OUT = Path("/opt/dlami/nvme/tev_plate_design/work/control_characterization")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_ca(path: Path) -> dict[int, dict[str, object]]:
    residues: dict[int, dict[str, object]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[21].strip() != "A" or line[16] not in (" ", "A"):
            continue
        pos = int(line[22:26])
        if not 1 <= pos <= 221 or pos in residues:
            continue
        residues[pos] = {
            "xyz": np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            ),
            "bfactor": float(line[60:66]),
        }
    if len(residues) != 221:
        raise ValueError(f"expected 221 CA atoms in {path}, got {len(residues)}")
    return residues


def best_fit_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """Kabsch RMSD, independently superposing the supplied selection."""
    p = mobile - mobile.mean(axis=0)
    q = reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    rotation = u @ vt
    d = p @ rotation - q
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def globally_aligned_subset_rmsds(
    mobile: np.ndarray,
    reference: np.ndarray,
    subsets: dict[str, list[int]],
) -> dict[str, float]:
    p_mean = mobile.mean(axis=0)
    q_mean = reference.mean(axis=0)
    p = mobile - p_mean
    q = reference - q_mean
    u, _, vt = np.linalg.svd(p.T @ q)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    aligned = p @ (u @ vt) + q_mean
    out = {}
    for name, positions in subsets.items():
        idx = np.array([x - 1 for x in positions], dtype=int)
        d = aligned[idx] - reference[idx]
        out[name] = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_mpnn_by_sequence(folder: Path) -> tuple[dict[str, dict], dict[str, str]]:
    scores: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for path in sorted(folder.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        seq = str(z["seq_str"].item())
        nlls = np.asarray(z["score"], dtype=float)
        if nlls.size != 16:
            raise ValueError(f"{path}: expected 16 random-order scores, got {nlls.size}")
        rec = {
            "nll_mean": float(nlls.mean()),
            "nll_sd": float(nlls.std(ddof=1)),
            "geomean_probability": float(math.exp(-float(nlls.mean()))),
            "n_orders": int(nlls.size),
        }
        # The WT-backbone folder includes the native sequence twice (PDB and
        # fasta entry). Keep the PDB record for the parent, otherwise require a
        # unique exact-sequence mapping.
        if seq in scores and path.name.endswith("_pdb.npz"):
            scores[seq] = rec
            sources[seq] = str(path)
        elif seq not in scores:
            scores[seq] = rec
            sources[seq] = str(path)
    return scores, sources


def pct_delta(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / abs(reference)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    structure_dir = OUT / "structures"
    structure_dir.mkdir(exist_ok=True)

    controls = read_csv(CONTROL_SELECTION / "proposed_m3_24_controls.csv")
    metadata = {r["design_id"]: r for r in read_csv(REANALYSIS / "dataset/design_metadata.csv")}
    lead_exp = {r["variant"]: r for r in read_csv(REANALYSIS / "dataset/published_lead_experiments.csv")}
    with (REANALYSIS / "dataset/fixed_positions.json").open() as f:
        fixed = json.load(f)
    protected = fixed["active_site_plus_50pct_conserved"]
    active_site = fixed["active_site"]
    position_sets = {
        "global": list(range(1, 222)),
        "m3_protected_active_plus_50pct_conserved": protected,
        "active_site": active_site,
    }
    with (OUT / "rmsd_position_sets.json").open("w") as f:
        json.dump(position_sets, f, indent=2)

    tevd_seq = "".join(
        line.strip()
        for line in (REANALYSIS / "dataset/TEVd.fasta").read_text().splitlines()
        if line and not line.startswith(">")
    )
    rows_in: list[dict[str, str]] = [
        {
            "panel_index": "0",
            "design_id": "TEVd",
            "design_number": "",
            "method_code": "parent",
            "activity_tier": "parent_reference",
            "named_hyperTEV_lead": "False",
            "activity_evidence": "published parent Michaelis-Menten kinetics",
            "trace_slope_relative": "",
            "identity_to_TEVD_percent": "100.0",
            "n_mutations": "0",
            "mutations": "",
            "sequence": tevd_seq,
            "structure_file": str(REANALYSIS / "dataset/TEVd_1LVM_A_1-221.pdb"),
        }
    ] + controls

    ridgey = {
        r["name"]: r
        for r in json.load(gzip.open(REANALYSIS / "results/ridgey/predictions_600m.json.gz", "rt"))
    }
    wt_mpnn, wt_mpnn_sources = load_mpnn_by_sequence(
        REANALYSIS / "results/proteinmpnn/score_all_designs_order16/score_only"
    )
    own_mpnn, own_mpnn_sources = load_mpnn_by_sequence(
        REANALYSIS / "results/proteinmpnn/score_own_structures_order16/score_only"
    )
    # The parent conditional is the 1LVM PDB record in the WT folder and is
    # also its own-structure conditional.
    own_mpnn[tevd_seq] = wt_mpnn[tevd_seq]
    own_mpnn_sources[tevd_seq] = wt_mpnn_sources[tevd_seq]

    reference_path = REANALYSIS / "dataset/TEVd_1LVM_A_1-221.pdb"
    ref_ca = parse_ca(reference_path)
    ref_xyz = np.stack([ref_ca[i]["xyz"] for i in range(1, 222)])

    ridgey_ref = ridgey["TEVd"]["predictions"]
    stability_ref = float(ridgey_ref["stability"])
    solubility_ref = float(ridgey_ref["solubility"])
    mpnn_wt_ref = wt_mpnn[tevd_seq]
    mpnn_own_ref = own_mpnn[tevd_seq]

    output: list[dict[str, object]] = []
    structure_manifest: list[dict[str, object]] = []
    for source in rows_in:
        design_id = source["design_id"]
        seq = source["sequence"]
        path = Path(source["structure_file"])
        if not path.exists():
            raise FileNotFoundError(path)
        ca = parse_ca(path)
        xyz = np.stack([ca[i]["xyz"] for i in range(1, 222)])
        bvals = np.array([float(ca[i]["bfactor"]) for i in range(1, 222)])

        if design_id == "TEVd":
            global_rmsd = protected_rmsd = active_rmsd = 0.0
            protected_after_global = active_after_global = 0.0
            plddt_available = False
            plddt_mean = plddt_median = plddt_min = plddt_frac90 = None
            structure_type = "experimental X-ray structure, PDB 1LVM chain A residues 1-221"
            plddt_note = "1LVM B-factors are crystallographic, not pLDDT; paper reports TEVd AF2 pLDDT=90 but does not release that AF2 file"
        else:
            all_pos = list(range(1, 222))
            global_rmsd = best_fit_rmsd(xyz, ref_xyz)
            pidx = np.array([x - 1 for x in protected])
            aidx = np.array([x - 1 for x in active_site])
            protected_rmsd = best_fit_rmsd(xyz[pidx], ref_xyz[pidx])
            active_rmsd = best_fit_rmsd(xyz[aidx], ref_xyz[aidx])
            after = globally_aligned_subset_rmsds(
                xyz,
                ref_xyz,
                {"protected": protected, "active": active_site},
            )
            protected_after_global = after["protected"]
            active_after_global = after["active"]
            # Released design files zeroed every B-factor; do not convert these
            # zeros into fake pLDDT values.
            plddt_available = bool(np.any(bvals != 0.0))
            if plddt_available:
                plddt_mean = float(bvals.mean())
                plddt_median = float(np.median(bvals))
                plddt_min = float(bvals.min())
                plddt_frac90 = float(np.mean(bvals >= 90.0))
                plddt_note = "read from CA B-factor column"
            else:
                plddt_mean = plddt_median = plddt_min = plddt_frac90 = None
                plddt_note = "released author PDB has all B-factors=0; paper reports all 144 designs >87.5 but no per-design pLDDT"
            structure_type = "paper-supplied AlphaFold2 predicted design structure"

        pred = ridgey[design_id]["predictions"]
        stability = float(pred["stability"])
        solubility = float(pred["solubility"])
        mw = wt_mpnn[seq]
        mo = own_mpnn[seq]
        exp = lead_exp.get(design_id, {})

        link = structure_dir / f"{design_id}.pdb"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(path)
        structure_manifest.append(
            {
                "design_id": design_id,
                "source_structure_file": str(path),
                "linked_structure_file": str(link),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )

        row: dict[str, object] = {
            "panel_index": int(source["panel_index"]),
            "design_id": design_id,
            "design_number": source.get("design_number", ""),
            "design_group": source["method_code"],
            "sequence": seq,
            "sequence_length": len(seq),
            "mutations": source["mutations"],
            "n_mutations": int(source["n_mutations"]),
            "identity_to_TEVD_percent": float(source["identity_to_TEVD_percent"]),
            "structure_file": str(link),
            "structure_source_file": str(path),
            "structure_type": structure_type,
            "structure_sha256": sha256(path),
            "ca_atoms": len(ca),
            "ca_rmsd_global_vs_1LVM_A": global_rmsd,
            "ca_rmsd_m3_protected_vs_1LVM_A": protected_rmsd,
            "ca_rmsd_active_site_vs_1LVM_A": active_rmsd,
            "ca_rmsd_m3_protected_after_global_align_A": protected_after_global,
            "ca_rmsd_active_site_after_global_align_A": active_after_global,
            "m3_protected_positions_n": len(protected),
            "active_site_positions_n": len(active_site),
            "pdb_ca_bfactor_mean": float(bvals.mean()),
            "plddt_available_from_pdb": plddt_available,
            "plddt_mean": plddt_mean,
            "plddt_median": plddt_median,
            "plddt_min": plddt_min,
            "plddt_fraction_ge_90": plddt_frac90,
            "plddt_note": plddt_note,
            "paper_af2_filter_status": "pass: paper reports all 144 designs pLDDT >87.5" if design_id != "TEVd" else "paper reports TEVd AF2 pLDDT=90",
            "ridgey_600m_stability": stability,
            "ridgey_600m_stability_units": pred["stability_units"],
            "ridgey_600m_stability_delta_vs_TEVD": stability - stability_ref,
            "ridgey_600m_stability_percent_delta_vs_TEVD": pct_delta(stability, stability_ref),
            "ridgey_600m_solubility_probability": solubility,
            "ridgey_600m_solubility_target": pred["solubility_target"],
            "ridgey_600m_solubility_delta_vs_TEVD": solubility - solubility_ref,
            "ridgey_600m_solubility_percent_delta_vs_TEVD": pct_delta(solubility, solubility_ref),
            "proteinmpnn_wt_structure_nll_mean_16order": mw["nll_mean"],
            "proteinmpnn_wt_structure_nll_sd_16order": mw["nll_sd"],
            "proteinmpnn_wt_structure_geomean_probability": mw["geomean_probability"],
            "proteinmpnn_wt_structure_probability_delta_vs_TEVD": mw["geomean_probability"] - mpnn_wt_ref["geomean_probability"],
            "proteinmpnn_wt_structure_probability_percent_delta_vs_TEVD": pct_delta(mw["geomean_probability"], mpnn_wt_ref["geomean_probability"]),
            "proteinmpnn_wt_structure_score_source": wt_mpnn_sources[seq],
            "proteinmpnn_own_structure_nll_mean_16order": mo["nll_mean"],
            "proteinmpnn_own_structure_nll_sd_16order": mo["nll_sd"],
            "proteinmpnn_own_structure_geomean_probability": mo["geomean_probability"],
            "proteinmpnn_own_structure_probability_delta_vs_TEVD": mo["geomean_probability"] - mpnn_own_ref["geomean_probability"],
            "proteinmpnn_own_structure_probability_percent_delta_vs_TEVD": pct_delta(mo["geomean_probability"], mpnn_own_ref["geomean_probability"]),
            "proteinmpnn_own_structure_score_source": own_mpnn_sources[seq],
            "experimental_activity_tier": source["activity_tier"],
            "experimental_tier_evidence": source["activity_evidence"],
            "figureS7_trace_slope_relative_not_RFU_s": source.get("trace_slope_relative", ""),
            "published_individual_kinetics_available": bool(exp),
            "published_kcat_min_1": exp.get("kcat_min-1", ""),
            "published_kcat_se": exp.get("kcat_se", ""),
            "published_Km_uM": exp.get("Km_uM", ""),
            "published_Km_se": exp.get("Km_se", ""),
            "published_kcat_over_Km_uM_1_min_1": exp.get("kcat_over_Km_uM-1_min-1", ""),
            "published_efficiency_fold_vs_TEVD": exp.get("fold_efficiency_vs_TEVD", ""),
            "published_Tm_C_approx": exp.get("Tm_C_approx", ""),
            "published_activity_after_4h_30C_percent": exp.get("activity_after_4h_at_30C_percent", ""),
        }
        output.append(row)

    if len(output) != 25 or output[0]["design_id"] != "TEVd":
        raise AssertionError("expected exactly TEVd + 24 controls")
    if len({r["design_id"] for r in output}) != 25:
        raise AssertionError("duplicate design IDs")
    if any(int(r["sequence_length"]) != 221 or int(r["ca_atoms"]) != 221 for r in output):
        raise AssertionError("sequence/structure length mismatch")
    if any(r["plddt_available_from_pdb"] for r in output):
        raise AssertionError("expected no usable pLDDT in released/experimental files")

    csv_path = OUT / "released_controls_characterized_25.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    with (OUT / "released_controls_characterized_25.json").open("w") as f:
        json.dump(output, f, indent=2)
    with (OUT / "structure_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(structure_manifest[0]))
        w.writeheader()
        w.writerows(structure_manifest)

    mapping = {
        "row_count": len(output),
        "input_control_selection": str(CONTROL_SELECTION / "proposed_m3_24_controls.csv"),
        "inputs": {
            "sequences_mutations": str(REANALYSIS / "dataset/design_metadata.csv"),
            "tevd_sequence": str(REANALYSIS / "dataset/TEVd.fasta"),
            "tevd_reference_structure": str(reference_path),
            "released_design_structures": str(REANALYSIS / "dataset/structures"),
            "m3_and_active_site_position_sets": str(REANALYSIS / "dataset/fixed_positions.json"),
            "ridgey_600m_seq_structure_predictions": str(REANALYSIS / "results/ridgey/predictions_600m.json.gz"),
            "proteinmpnn_wt_structure_16order": str(REANALYSIS / "results/proteinmpnn/score_all_designs_order16/score_only"),
            "proteinmpnn_own_structure_16order": str(REANALYSIS / "results/proteinmpnn/score_own_structures_order16/score_only"),
            "published_named_lead_kinetics": str(REANALYSIS / "dataset/published_lead_experiments.csv"),
        },
        "definitions": {
            "proteinmpnn_geomean_probability": "exp(-mean sequence NLL across 16 random decoding orders); a per-residue geometric-mean conditional probability, not a full-sequence joint probability",
            "rmsd_primary": "independent best-fit Kabsch CA RMSD for global, m3-protected, or active-site selection",
            "rmsd_after_global_align": "subset CA RMSD after Kabsch alignment on all 221 CA atoms",
            "m3_protected": "127 active-site-plus-50%-conserved positions fixed in the paper's m3 arm",
            "active_site": "38 active-site/substrate-contact positions fixed in all paper arms",
            "plddt": "null because released design PDB B-factor columns are all zero; paper-level pass statement retained separately",
            "experimental_tier": "non-lead tiers inferred from Figure S7 row-major trace ordering; not a numeric per-design rate",
        },
    }
    with (OUT / "source_mapping.json").open("w") as f:
        json.dump(mapping, f, indent=2)

    print(csv_path)
    print(f"rows={len(output)} unique={len({r['design_id'] for r in output})}")
    print(f"structures={len(structure_manifest)} plddt_available={sum(bool(r['plddt_available_from_pdb']) for r in output)}")
    print(
        "rmsd_global_range=",
        min(float(r["ca_rmsd_global_vs_1LVM_A"]) for r in output),
        max(float(r["ca_rmsd_global_vs_1LVM_A"]) for r in output),
    )


if __name__ == "__main__":
    main()
