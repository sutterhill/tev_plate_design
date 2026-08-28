#!/usr/bin/env python3
"""Characterize TEVd + released controls on fresh, matched AF2 structures.

Primary structure-dependent fields use the task-specific query-swapped-A3M
ColabFold run.  Paper-supplied PDB metrics remain explicit provenance fields.
ProteinMPNN WT-backbone scores are reused by exact sequence. Fresh-own-backbone
ProteinMPNN is intentionally left null while aws0 GPUs are occupied by training;
the parent pipeline fills those fields later.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import requests


ROOT = Path("/opt/dlami/nvme/tev_plate_design")
REANALYSIS = Path("/opt/dlami/nvme/tev_reanalysis")
CONTROL_SELECTION = ROOT / "work/m3_controls/proposed_m3_24_controls.csv"
OUT = ROOT / "work/control_characterization"
RIDGEY_API = "https://shv-internal--ridgey-v2-prod-web.modal.run"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_ca(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords, b = [], []
    seen = set()
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[12:16].strip() != "CA":
            continue
        if line[21].strip() not in ("", "A") or line[16] not in (" ", "A"):
            continue
        pos = int(line[22:26])
        if not 1 <= pos <= 221 or pos in seen:
            continue
        seen.add(pos)
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        b.append(float(line[60:66]))
    if len(coords) != 221:
        raise ValueError(f"expected 221 CA in {path}, got {len(coords)}")
    return np.asarray(coords, float), np.asarray(b, float)


def pdb_sequence(path: Path) -> str:
    residues = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[21].strip() not in ("", "A") or line[16] not in (" ", "A"):
            continue
        pos = int(line[22:26])
        if 1 <= pos <= 221 and pos not in residues:
            residues[pos] = AA3[line[17:20].strip()]
    return "".join(residues[p] for p in range(1, 222))


def align_all(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    rc, mc = reference.mean(0), mobile.mean(0)
    r, m = reference - rc, mobile - mc
    u, _, vh = np.linalg.svd(m.T @ r)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = u @ vh
    return m @ rotation + rc


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def best_fit_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    return rmsd(reference, align_all(reference, mobile))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fold_files(run_name: str, candidate_id: str) -> tuple[Path, Path]:
    folder = ROOT / f"folds/{run_name}/outputs/{candidate_id}"
    pdbs = sorted(folder.glob("*_unrelaxed_rank_001_*.pdb"))
    scores = sorted(folder.glob("*_scores_rank_001_*.json"))
    if len(pdbs) != 1 or len(scores) != 1:
        raise FileNotFoundError(f"fresh AF2 output incomplete for {candidate_id}: {pdbs}, {scores}")
    return pdbs[0], scores[0]


def load_mpnn_by_sequence(folder: Path) -> tuple[dict[str, dict], dict[str, str]]:
    out, sources = {}, {}
    for path in sorted(folder.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        seq = str(z["seq_str"].item())
        nll = np.asarray(z["score"], float)
        if nll.size != 16:
            continue
        rec = {
            "nll": float(nll.mean()),
            "nll_sd": float(nll.std(ddof=1)),
            "probability": float(math.exp(-float(nll.mean()))),
        }
        if seq not in out or path.name.endswith("_pdb.npz"):
            out[seq], sources[seq] = rec, str(path)
    return out, sources


def wait_job(call_id: str) -> list[dict]:
    while True:
        response = requests.get(f"{RIDGEY_API}/jobs/{call_id}", timeout=120)
        if response.status_code == 202:
            time.sleep(5)
            continue
        response.raise_for_status()
        return response.json()


def ridgey_fresh(rows: list[dict], model: str, tag: str) -> dict[str, dict]:
    raw = OUT / f"raw/ridgey_fresh_af2_{tag}"
    raw.mkdir(parents=True, exist_ok=True)
    request_path = raw / "request.json.gz"
    response_path = raw / "response.json.gz"
    manifest_path = OUT / f"ridgey_fresh_af2_{tag}_manifest.json"
    payload = {
        "model": model,
        "structures": [
            {
                "name": r["design_id"],
                "filename": Path(r["fresh_af2_pdb"]).name,
                "content": Path(r["fresh_af2_pdb"]).read_text(),
                "chain_id": "A",
            }
            for r in rows
        ],
        "threshold": 0.5,
    }
    if not response_path.exists():
        with gzip.open(request_path, "wt") as f:
            json.dump(payload, f)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            call_id = manifest["call_id"]
        else:
            response = requests.post(f"{RIDGEY_API}/jobs", json=payload, timeout=120)
            response.raise_for_status()
            call_id = response.json()["call_id"]
            manifest = {
                "call_id": call_id,
                "api": RIDGEY_API,
                "model": model,
                "count": len(rows),
                "status": "submitted",
                "request": str(request_path),
                "response": str(response_path),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        result = wait_job(call_id)
        with gzip.open(response_path, "wt") as f:
            json.dump(result, f)
        manifest["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        result = json.load(gzip.open(response_path, "rt"))
    if len(result) != len(rows):
        raise AssertionError((len(result), len(rows)))
    return {r["name"]: r for r in result}


def pct(value: float, ref: float) -> float:
    return 100.0 * (value - ref) / abs(ref)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    controls = read_csv(CONTROL_SELECTION)
    if len(controls) != 24:
        raise AssertionError("expected 24 selected controls")
    old_rows = {r["design_id"]: r for r in read_csv(OUT / "released_controls_characterized_25.csv")}
    lead_exp = {r["variant"]: r for r in read_csv(REANALYSIS / "dataset/published_lead_experiments.csv")}
    fixed = json.loads((ROOT / "inputs/fixed_positions.json").read_text())
    pidx = np.asarray([p - 1 for p in fixed["active_site_plus_50pct_conserved"]], int)
    aidx = np.asarray([p - 1 for p in fixed["active_site"]], int)
    tevd_seq = "".join(
        x.strip() for x in (ROOT / "inputs/TEVd.fasta").read_text().splitlines()
        if x and not x.startswith(">")
    )
    entries = [{
        "panel_index": "0", "design_id": "TEVd", "design_number": "",
        "method_code": "parent", "activity_tier": "parent_reference",
        "activity_evidence": "published parent Michaelis-Menten kinetics",
        "trace_slope_relative": "", "identity_to_TEVD_percent": "100",
        "n_mutations": "0", "mutations": "", "sequence": tevd_seq,
    }] + controls

    ref, _ = parse_ca(ROOT / "inputs/TEVd_1LVM_A_1-221.pdb")
    wt_scores, wt_sources = load_mpnn_by_sequence(
        REANALYSIS / "results/proteinmpnn/score_all_designs_order16/score_only"
    )
    fresh_own_folder = ROOT / "raw/proteinmpnn_scores/released_controls/own_structure/score_only"
    fresh_own_scores, fresh_own_sources = load_mpnn_by_sequence(fresh_own_folder)
    if len(fresh_own_scores) not in (0, 25):
        raise AssertionError(f"incomplete fresh-own ProteinMPNN scores: {len(fresh_own_scores)}")
    wt_ref = wt_scores[tevd_seq]["probability"]
    fresh_own_ref = fresh_own_scores.get(tevd_seq, {}).get("probability")
    prepared = []
    fresh_structure_manifest = []
    fresh_structure_dir = OUT / "fresh_af2_structures"
    fresh_structure_dir.mkdir(exist_ok=True)
    for r in entries:
        design_id = r["design_id"]
        run = "production" if design_id == "TEVd" else "released_controls"
        pdb, score_file = fold_files(run, design_id)
        score = json.loads(score_file.read_text())
        plddt = np.asarray(score["plddt"], float)
        if plddt.mean() <= 1.5:
            plddt *= 100.0
        coords, pdb_b = parse_ca(pdb)
        if plddt.size != 221 or pdb_sequence(pdb) != r["sequence"]:
            raise AssertionError(f"fresh AF2 sequence/confidence mismatch for {design_id}")
        if abs(float(plddt.mean()) - float(pdb_b.mean())) > 0.02:
            raise AssertionError(f"AF2 JSON/PDB pLDDT mismatch for {design_id}")
        aligned = align_all(ref, coords)
        mw = wt_scores[r["sequence"]]
        mo = fresh_own_scores.get(r["sequence"])
        old = old_rows[design_id]
        exp = lead_exp.get(design_id, {})
        link = fresh_structure_dir / f"{design_id}.pdb"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(pdb)
        fresh_structure_manifest.append({
            "design_id": design_id,
            "fresh_af2_pdb": str(pdb),
            "app_structure_link": str(link),
            "fresh_af2_scores_json": str(score_file),
            "pdb_sha256": sha256(pdb),
            "scores_sha256": sha256(score_file),
            "a3m": str(ROOT / f"folds/{run}/a3m/{design_id}.a3m"),
            "run_name": run,
        })
        prepared.append({
            "panel_index": int(r["panel_index"]),
            "design_id": design_id,
            "design_number": r.get("design_number", ""),
            "design_group": r["method_code"],
            "sequence": r["sequence"],
            "sequence_length": len(r["sequence"]),
            "mutations": r["mutations"],
            "n_mutations": int(r["n_mutations"]),
            "identity_to_TEVD_percent": float(r["identity_to_TEVD_percent"]),
            "fresh_af2_pdb": str(link),
            "fresh_af2_raw_pdb": str(pdb),
            "fresh_af2_scores_json": str(score_file),
            "fresh_af2_pdb_sha256": sha256(pdb),
            "fresh_af2_run": run,
            "fresh_af2_model": "alphafold2 model 3",
            "fresh_af2_recycles": 6,
            "fresh_af2_msa_strategy": "query-swapped current TEVd MMseqs A3M",
            "fresh_af2_plddt_mean": float(plddt.mean()),
            "fresh_af2_plddt_median": float(np.median(plddt)),
            "fresh_af2_plddt_min": float(plddt.min()),
            "fresh_af2_plddt_fraction_ge_90": float(np.mean(plddt >= 90.0)),
            "fresh_af2_ptm": float(score["ptm"]) if score.get("ptm") is not None else None,
            "fresh_af2_ptm_note": "not emitted by this ColabFold AlphaFold2 monomer score JSON" if score.get("ptm") is None else "emitted by ColabFold",
            "fresh_af2_ca_rmsd_global_vs_1LVM_A": rmsd(ref, aligned),
            "fresh_af2_ca_rmsd_m3_protected_after_global_align_A": rmsd(ref[pidx], aligned[pidx]),
            "fresh_af2_ca_rmsd_active_site_after_global_align_A": rmsd(ref[aidx], aligned[aidx]),
            "fresh_af2_ca_rmsd_m3_protected_best_fit_A": best_fit_rmsd(ref[pidx], coords[pidx]),
            "fresh_af2_ca_rmsd_active_site_best_fit_A": best_fit_rmsd(ref[aidx], coords[aidx]),
            "fresh_af2_paper_plddt_gate_87_5_pass": bool(plddt.mean() > 87.5),
            "fresh_af2_pipeline_gate_plddt85_rmsd2_pass": bool(plddt.mean() > 85 and rmsd(ref, aligned) < 2.0),
            "fresh_af2_pdb_ca_bfactor_mean": float(pdb_b.mean()),
            "paper_supplied_pdb": old["structure_source_file"],
            "paper_supplied_pdb_sha256": old["structure_sha256"],
            "paper_supplied_pdb_bfactor_mean": float(old["pdb_ca_bfactor_mean"]),
            "paper_supplied_pdb_plddt_available": False,
            "paper_supplied_pdb_plddt_note": old["plddt_note"],
            "paper_pdb_ca_rmsd_global_vs_1LVM_A": float(old["ca_rmsd_global_vs_1LVM_A"]),
            "paper_pdb_ridgey_600m_stability": float(old["ridgey_600m_stability"]),
            "paper_pdb_ridgey_600m_solubility_probability": float(old["ridgey_600m_solubility_probability"]),
            "proteinmpnn_wt_structure_nll_mean_16order": mw["nll"],
            "proteinmpnn_wt_structure_nll_sd_16order": mw["nll_sd"],
            "proteinmpnn_wt_structure_geomean_probability": mw["probability"],
            "proteinmpnn_wt_structure_probability_delta_vs_TEVD": mw["probability"] - wt_ref,
            "proteinmpnn_wt_structure_probability_percent_delta_vs_TEVD": pct(mw["probability"], wt_ref),
            "proteinmpnn_wt_structure_score_source": wt_sources[r["sequence"]],
            "paper_pdb_proteinmpnn_own_structure_nll_mean_16order": float(old["proteinmpnn_own_structure_nll_mean_16order"]),
            "paper_pdb_proteinmpnn_own_structure_geomean_probability": float(old["proteinmpnn_own_structure_geomean_probability"]),
            "fresh_af2_proteinmpnn_own_structure_nll_mean_16order": mo["nll"] if mo else None,
            "fresh_af2_proteinmpnn_own_structure_nll_sd_16order": mo["nll_sd"] if mo else None,
            "fresh_af2_proteinmpnn_own_structure_geomean_probability": mo["probability"] if mo else None,
            "fresh_af2_proteinmpnn_own_structure_probability_delta_vs_TEVD": mo["probability"] - fresh_own_ref if mo and fresh_own_ref is not None else None,
            "fresh_af2_proteinmpnn_own_structure_probability_percent_delta_vs_TEVD": pct(mo["probability"], fresh_own_ref) if mo and fresh_own_ref is not None else None,
            "fresh_af2_proteinmpnn_own_structure_score_source": fresh_own_sources.get(r["sequence"], ""),
            "fresh_af2_proteinmpnn_own_structure_status": "completed" if mo else "pending",
            "experimental_activity_tier": r["activity_tier"],
            "experimental_tier_evidence": r["activity_evidence"],
            "figureS7_trace_slope_relative_not_RFU_s": r.get("trace_slope_relative", ""),
            "published_individual_kinetics_available": bool(exp),
            "published_kcat_min_1": exp.get("kcat_min-1", ""),
            "published_Km_uM": exp.get("Km_uM", ""),
            "published_kcat_over_Km_uM_1_min_1": exp.get("kcat_over_Km_uM-1_min-1", ""),
            "published_efficiency_fold_vs_TEVD": exp.get("fold_efficiency_vs_TEVD", ""),
            "published_Tm_C_approx": exp.get("Tm_C_approx", ""),
            "published_activity_after_4h_30C_percent": exp.get("activity_after_4h_at_30C_percent", ""),
        })

    ridgey = ridgey_fresh(prepared, "600m", "600m")
    ridgey_ensemble = ridgey_fresh(prepared, "600m_ensemble", "600m_ensemble")
    ridgey_ref = ridgey["TEVd"]["predictions"]
    stab_ref = float(ridgey_ref["stability"])
    sol_ref = float(ridgey_ref["solubility"])
    ensemble_ref_item = ridgey_ensemble["TEVd"]
    ensemble_ref_predictions = ensemble_ref_item["predictions"]
    if len(ensemble_ref_predictions) != 5:
        raise AssertionError("expected five Ridgey ensemble members")
    ensemble_ref_stability = np.asarray([float(x["stability"]) for x in ensemble_ref_predictions])
    ensemble_ref_solubility = np.asarray([float(x["solubility"]) for x in ensemble_ref_predictions])
    ensemble_ref_stability_mean = float(ensemble_ref_stability.mean())
    ensemble_ref_solubility_mean = float(ensemble_ref_solubility.mean())
    for row in prepared:
        item = ridgey[row["design_id"]]
        pred = item["predictions"]
        stab, sol = float(pred["stability"]), float(pred["solubility"])
        ensemble_item = ridgey_ensemble[row["design_id"]]
        ensemble_predictions = ensemble_item["predictions"]
        ensemble_models = ensemble_item["prediction_models"]
        if len(ensemble_predictions) != 5 or len(ensemble_models) != 5:
            raise AssertionError(f"expected five ensemble members for {row['design_id']}")
        ensemble_stability = np.asarray([float(x["stability"]) for x in ensemble_predictions])
        ensemble_solubility = np.asarray([float(x["solubility"]) for x in ensemble_predictions])
        ensemble_stability_mean = float(ensemble_stability.mean())
        ensemble_solubility_mean = float(ensemble_solubility.mean())
        update = {
            "fresh_af2_ridgey_600m_stability": stab,
            "fresh_af2_ridgey_600m_stability_units": pred["stability_units"],
            "fresh_af2_ridgey_600m_stability_delta_vs_TEVD": stab - stab_ref,
            "fresh_af2_ridgey_600m_stability_percent_delta_vs_TEVD": pct(stab, stab_ref),
            "fresh_af2_ridgey_600m_solubility_probability": sol,
            "fresh_af2_ridgey_600m_solubility_target": pred["solubility_target"],
            "fresh_af2_ridgey_600m_solubility_delta_vs_TEVD": sol - sol_ref,
            "fresh_af2_ridgey_600m_solubility_percent_delta_vs_TEVD": pct(sol, sol_ref),
            "fresh_af2_ridgey_600m_call_id": json.loads((OUT / "ridgey_fresh_af2_600m_manifest.json").read_text())["call_id"],
            "fresh_af2_ridgey_ensemble_models_json": json.dumps(ensemble_models),
            "fresh_af2_ridgey_ensemble_stability_members_json": json.dumps(ensemble_stability.tolist()),
            "fresh_af2_ridgey_ensemble_solubility_members_json": json.dumps(ensemble_solubility.tolist()),
            "fresh_af2_ridgey_ensemble_stability_mean": ensemble_stability_mean,
            "fresh_af2_ridgey_ensemble_stability_sample_sd": float(ensemble_stability.std(ddof=1)),
            "fresh_af2_ridgey_ensemble_stability_delta_vs_TEVD_mean": ensemble_stability_mean - ensemble_ref_stability_mean,
            "fresh_af2_ridgey_ensemble_stability_percent_delta_vs_TEVD_mean": pct(ensemble_stability_mean, ensemble_ref_stability_mean),
            "fresh_af2_ridgey_ensemble_stability_paired_member_votes_better_than_TEVD": int(np.sum(ensemble_stability > ensemble_ref_stability)),
            "fresh_af2_ridgey_ensemble_stability_paired_member_deltas_json": json.dumps((ensemble_stability - ensemble_ref_stability).tolist()),
            "fresh_af2_ridgey_ensemble_solubility_mean": ensemble_solubility_mean,
            "fresh_af2_ridgey_ensemble_solubility_sample_sd": float(ensemble_solubility.std(ddof=1)),
            "fresh_af2_ridgey_ensemble_solubility_delta_vs_TEVD_mean": ensemble_solubility_mean - ensemble_ref_solubility_mean,
            "fresh_af2_ridgey_ensemble_solubility_percent_delta_vs_TEVD_mean": pct(ensemble_solubility_mean, ensemble_ref_solubility_mean),
            "fresh_af2_ridgey_ensemble_solubility_paired_member_votes_better_than_TEVD": int(np.sum(ensemble_solubility > ensemble_ref_solubility)),
            "fresh_af2_ridgey_ensemble_solubility_paired_member_deltas_json": json.dumps((ensemble_solubility - ensemble_ref_solubility).tolist()),
            "fresh_af2_ridgey_600m_ensemble_call_id": json.loads((OUT / "ridgey_fresh_af2_600m_ensemble_manifest.json").read_text())["call_id"],
        }
        for member_index, (model_name, stability_value, solubility_value) in enumerate(
            zip(ensemble_models, ensemble_stability, ensemble_solubility), 1
        ):
            update[f"fresh_af2_ridgey_ensemble_member{member_index}_model"] = model_name
            update[f"fresh_af2_ridgey_ensemble_member{member_index}_stability"] = float(stability_value)
            update[f"fresh_af2_ridgey_ensemble_member{member_index}_solubility"] = float(solubility_value)
        row.update(update)

    if len(prepared) != 25 or len({r["design_id"] for r in prepared}) != 25:
        raise AssertionError("expected exactly 25 unique characterized rows")
    if any(r["sequence_length"] != 221 for r in prepared):
        raise AssertionError("bad sequence length")
    if any(not Path(r["fresh_af2_pdb"]).exists() for r in prepared):
        raise AssertionError("missing fresh AF2 PDB")
    tier_counts = {
        tier: sum(r["experimental_activity_tier"] == tier for r in prepared)
        for tier in ("parent_reference", "active", "somewhat_active", "inactive_or_floor")
    }
    if tier_counts != {"parent_reference": 1, "active": 8, "somewhat_active": 8, "inactive_or_floor": 8}:
        raise AssertionError(tier_counts)
    fields = []
    for row in prepared:
        for k in row:
            if k not in fields:
                fields.append(k)
    with (OUT / "released_controls_fresh_af2_characterized_25.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(prepared)
    (OUT / "released_controls_fresh_af2_characterized_25.json").write_text(json.dumps(prepared, indent=2) + "\n")
    with (OUT / "fresh_af2_structure_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fresh_structure_manifest[0]))
        w.writeheader(); w.writerows(fresh_structure_manifest)
    fresh_mapping = {
        "primary_characterized_table": str(OUT / "released_controls_fresh_af2_characterized_25.csv"),
        "row_count": 25,
        "control_selection": str(CONTROL_SELECTION),
        "af2_controls_manifest": str(ROOT / "manifests/af2_released_controls.jsonl"),
        "af2_controls_calls": str(ROOT / "manifests/af2_modal_calls_released_controls.json"),
        "af2_tevd_manifest": str(ROOT / "manifests/af2_production.jsonl"),
        "parent_a3m": str(ROOT / "inputs/tev_parent_current_mmseqs.a3m"),
        "ridgey_base_manifest": str(OUT / "ridgey_fresh_af2_600m_manifest.json"),
        "ridgey_ensemble_manifest": str(OUT / "ridgey_fresh_af2_600m_ensemble_manifest.json"),
        "proteinmpnn_fresh_own_structure_16order": str(ROOT / "raw/proteinmpnn_scores/released_controls/own_structure"),
        "paper_pdb_provenance_table": str(OUT / "released_controls_characterized_25.csv"),
        "fresh_structure_manifest": str(OUT / "fresh_af2_structure_manifest.csv"),
        "definitions": {
            "fresh_af2": "ColabFold 1.6.2 AlphaFold2 monomer model 3, six recycles, seed 0; query-swapped current TEVd MMseqs A3M; no explicit structure template",
            "plddt": "per-residue confidence from ColabFold score JSON; checked against AF2 PDB CA B-factors",
            "primary_rmsd": "subset CA RMSD after global 221-CA Kabsch superposition to 1LVM",
            "best_fit_subset_rmsd": "secondary independent Kabsch fit on protected or active-site subset",
            "ridgey_ensemble_mean_sd": "arithmetic mean and sample SD across five named 600M ensemble members",
            "ridgey_paired_votes": "number of five matched ensemble members for which design score is strictly greater than freshly folded TEVd score",
            "proteinmpnn_probability": "exp(-mean per-residue sequence NLL across 16 random decoding orders)",
            "activity": "non-lead tiers are Figure S7 trace-order inferences; no per-design numeric screen rate is invented",
        },
    }
    (OUT / "fresh_source_mapping.json").write_text(json.dumps(fresh_mapping, indent=2) + "\n")
    summary = {
        "rows": 25,
        "unique_design_ids": 25,
        "fresh_af2_pdbs": 25,
        "paper_plddt_gate_pass": sum(r["fresh_af2_paper_plddt_gate_87_5_pass"] for r in prepared),
        "pipeline_gate_pass": sum(r["fresh_af2_pipeline_gate_plddt85_rmsd2_pass"] for r in prepared),
        "ridgey_fresh_structure_scores": 25,
        "ridgey_fresh_structure_ensemble5_scores": 25,
        "proteinmpnn_wt_structure_scores": 25,
        "proteinmpnn_fresh_own_structure_scores": len(fresh_own_scores),
        "proteinmpnn_fresh_own_status": "completed" if len(fresh_own_scores) == 25 else "pending",
        "activity_tier_counts": tier_counts,
    }
    (OUT / "fresh_characterization_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
