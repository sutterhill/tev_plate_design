#!/usr/bin/env python3
"""Compute AF2 gates, Ridgey mutant-structure scores, and ProteinMPNN likelihoods."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

from common import ROOT, read_jsonl, write_csv, write_fasta, write_json


RIDGEY_API = "https://shv-internal--ridgey-v2-prod-web.modal.run"
MPNN = Path("/home/ubuntu/ml-apps/ProteinMPNN/ProteinMPNN")
PYTHON = Path("/opt/pytorch/bin/python")


def ca_coords(path: Path) -> np.ndarray:
    coords = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA" and line[21].strip() in ("", "A"):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(coords, dtype=float)


def align_mobile(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    if reference.shape != mobile.shape:
        raise ValueError(f"CA shapes differ: {reference.shape} vs {mobile.shape}")
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    ref = reference - ref_center
    mob = mobile - mob_center
    u, _, vh = np.linalg.svd(mob.T @ ref)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = u @ vh
    return mob @ rotation + ref_center


def rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((mobile - reference) ** 2, axis=1))))


def find_fold_files(run_name: str, candidate_id: str) -> tuple[Path, Path]:
    out = ROOT / "folds" / run_name / "outputs" / candidate_id
    pdbs = sorted(out.glob("*_unrelaxed_rank_001_*.pdb"))
    scores = sorted(out.glob("*_scores_rank_001_*.json"))
    if not pdbs or not scores:
        raise FileNotFoundError(f"missing AF2 outputs for {candidate_id}")
    return pdbs[0], scores[0]


def analyze_folds(rows: list[dict], run_name: str) -> list[dict]:
    reference = ca_coords(ROOT / "inputs" / "TEVd_1LVM_A_1-221.pdb")
    fixed = json.loads((ROOT / "inputs" / "fixed_positions.json").read_text())
    protected_index = np.asarray([position - 1 for position in fixed["active_site_plus_50pct_conserved"]], dtype=int)
    active_index = np.asarray([position - 1 for position in fixed["active_site"]], dtype=int)
    output = []
    for row in rows:
        pdb, score_path = find_fold_files(run_name, row["candidate_id"])
        score = json.loads(score_path.read_text())
        plddt = np.asarray(score["plddt"], dtype=float)
        mean_plddt = float(plddt.mean())
        if mean_plddt <= 1.5:
            mean_plddt *= 100.0
        aligned = align_mobile(reference, ca_coords(pdb))
        global_rmsd = rmsd(reference, aligned)
        protected_rmsd = rmsd(reference[protected_index], aligned[protected_index])
        active_rmsd = rmsd(reference[active_index], aligned[active_index])
        output.append(
            {
                **row,
                "af2_pdb": str(pdb),
                "af2_scores_json": str(score_path),
                "af2_mean_plddt": mean_plddt,
                "af2_ca_rmsd_to_1lvm_angstrom": global_rmsd,
                "af2_protected_ca_rmsd_to_1lvm_angstrom": protected_rmsd,
                "af2_active_site_ca_rmsd_to_1lvm_angstrom": active_rmsd,
                "af2_pass": mean_plddt > 85.0 and global_rmsd < 2.0,
            }
        )
    return output


def wait_ridgey(call_id: str) -> list[dict]:
    transient_failures = 0
    while True:
        try:
            response = requests.get(f"{RIDGEY_API}/jobs/{call_id}", timeout=120)
        except requests.RequestException:
            transient_failures += 1
            if transient_failures > 20:
                raise
            time.sleep(min(5 * transient_failures, 30))
            continue
        if response.status_code == 202:
            time.sleep(5)
            continue
        if response.status_code in (408, 425, 429) or 500 <= response.status_code < 600:
            transient_failures += 1
            if transient_failures > 20:
                response.raise_for_status()
            time.sleep(min(5 * transient_failures, 30))
            continue
        response.raise_for_status()
        return response.json()


def score_ridgey(rows: list[dict], run_name: str, model: str = "600m", batch_size: int = 30) -> dict[str, dict]:
    raw_dir = ROOT / "raw" / f"ridgey_structure_scores_{model}" / run_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifests" / f"ridgey_structure_scores_{model}_{run_name}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "api": RIDGEY_API, "model": model, "jobs": []
    }
    by_batch = {int(record["batch"]): record for record in manifest["jobs"]}
    by_name: dict[str, dict] = {}
    batches = [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]

    # Submit every missing batch before polling so the hosted service can fan
    # out the work.  The normal pool is fewer than 20 calls, well below 50.
    for batch_number, batch in enumerate(batches):
        response_path = raw_dir / f"batch_{batch_number:03d}.response.json.gz"
        request_path = raw_dir / f"batch_{batch_number:03d}.request.json"
        if response_path.exists() or batch_number in by_batch:
            continue
        payload = {
            "model": model,
            "structures": [
                {"name": row["candidate_id"], "filename": Path(row["af2_pdb"]).name, "content": Path(row["af2_pdb"]).read_text(), "chain_id": "A"}
                for row in batch
            ],
            "threshold": 0.5,
        }
        write_json(request_path, payload)
        response = requests.post(f"{RIDGEY_API}/jobs", json=payload, timeout=120)
        response.raise_for_status()
        record = {
            "batch": batch_number,
            "call_id": response.json()["call_id"],
            "status": "submitted",
            "request": str(request_path),
            "response": str(response_path),
        }
        manifest["jobs"].append(record)
        by_batch[batch_number] = record
        write_json(manifest_path, manifest)

    def collect(batch_number: int) -> tuple[int, list[dict]]:
        response_path = raw_dir / f"batch_{batch_number:03d}.response.json.gz"
        if response_path.exists():
            with gzip.open(response_path, "rt") as handle:
                return batch_number, json.load(handle)
        result = wait_ridgey(by_batch[batch_number]["call_id"])
        with gzip.open(response_path, "wt") as handle:
            json.dump(result, handle)
        return batch_number, result

    with ThreadPoolExecutor(max_workers=min(20, len(batches))) as executor:
        futures = [executor.submit(collect, batch_number) for batch_number in range(len(batches))]
        for future in as_completed(futures):
            batch_number, result = future.result()
            record = by_batch[batch_number]
            record["status"] = "completed"
            for item in result:
                if model == "600m_ensemble":
                    members = item["predictions"]
                    stability = np.asarray([float(member["stability"]) for member in members], dtype=float)
                    solubility = np.asarray([float(member["solubility"]) for member in members], dtype=float)
                    by_name[item["name"]] = {
                        "ridgey_ensemble_stability_mean": float(stability.mean()),
                        "ridgey_ensemble_stability_std": float(stability.std(ddof=1)),
                        "ridgey_ensemble_stability_members": json.dumps(stability.tolist()),
                        "ridgey_ensemble_solubility_mean": float(solubility.mean()),
                        "ridgey_ensemble_solubility_std": float(solubility.std(ddof=1)),
                        "ridgey_ensemble_solubility_members": json.dumps(solubility.tolist()),
                        "ridgey_ensemble_member_models": json.dumps(item.get("prediction_models", [])),
                        "ridgey_ensemble_size": len(members),
                        "ridgey_ensemble_structure_source": item["structure_source"],
                        "ridgey_ensemble_raw_response": record["response"],
                        "ridgey_ensemble_call_id": record["call_id"],
                    }
                else:
                    by_name[item["name"]] = {
                        "ridgey_600m_stability": float(item["predictions"]["stability"]),
                        "ridgey_600m_solubility": float(item["predictions"]["solubility"]),
                        "ridgey_model": item["model"],
                        "ridgey_structure_source": item["structure_source"],
                        "ridgey_raw_response": record["response"],
                        "ridgey_call_id": record["call_id"],
                    }
            write_json(manifest_path, manifest)
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    return by_name


def run_mpnn(command: list[str], gpu: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=MPNN, env=env, check=True)


def prepare_parsed(pdb_dir: Path, out_dir: Path, gpu: str) -> tuple[Path, Path]:
    parsed = out_dir / "parsed.jsonl"
    assigned = out_dir / "assigned.jsonl"
    if not parsed.exists():
        run_mpnn([str(PYTHON), "helper_scripts/parse_multiple_chains.py", "--input_path", str(pdb_dir), "--output_path", str(parsed)], gpu)
    if not assigned.exists():
        run_mpnn([str(PYTHON), "helper_scripts/assign_fixed_chains.py", "--input_path", str(parsed), "--output_path", str(assigned), "--chain_list", "A"], gpu)
    return parsed, assigned


def parse_npz_scores(folder: Path) -> dict[str, dict]:
    by_sequence: dict[str, list[dict]] = {}
    for path in sorted((folder / "score_only").glob("*.npz")):
        value = np.load(path)
        sequence = str(value["seq_str"])
        nll = float(np.asarray(value["score"], dtype=float).mean())
        by_sequence.setdefault(sequence, []).append({"nll": nll, "npz": str(path)})
    result = {}
    for sequence, values in by_sequence.items():
        nll = float(np.mean([x["nll"] for x in values]))
        result[sequence] = {"nll": nll, "geomean_probability": math.exp(-nll), "npz_files": "|".join(x["npz"] for x in values)}
    return result


def score_mpnn(rows: list[dict], run_name: str, gpu: str, orders: int = 16) -> tuple[dict[str, dict], dict[str, dict]]:
    base = ROOT / "raw" / "proteinmpnn_scores" / run_name
    wt_input = base / "wt_input"
    wt_pdbs = wt_input / "pdbs"
    wt_pdbs.mkdir(parents=True, exist_ok=True)
    wt_target = wt_pdbs / "TEVd.pdb"
    if not wt_target.exists():
        shutil.copy2(ROOT / "inputs" / "TEVd_1LVM_A_1-221.pdb", wt_target)
    wt_parsed, wt_assigned = prepare_parsed(wt_pdbs, wt_input, gpu)
    candidate_fasta = wt_input / "candidates.fasta"
    write_fasta(candidate_fasta, [(row["candidate_id"], row["sequence"]) for row in rows])
    wt_out = base / "wt_structure"
    if not (wt_out / "score_only").exists():
        command = [str(PYTHON), "protein_mpnn_run.py", "--jsonl_path", str(wt_parsed), "--chain_id_jsonl", str(wt_assigned), "--out_folder", str(wt_out), "--score_only", "1", "--path_to_fasta", str(candidate_fasta), "--num_seq_per_target", str(orders), "--batch_size", str(orders), "--sampling_temp", "0.1", "--seed", "99173"]
        write_json(wt_out / "command.json", {"command": command, "gpu": gpu})
        run_mpnn(command, gpu)
    wt_scores = parse_npz_scores(wt_out)

    own_input = base / "own_input"
    own_pdbs = own_input / "pdbs"
    own_pdbs.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = own_pdbs / f"{row['candidate_id']}.pdb"
        if not target.exists():
            shutil.copy2(row["af2_pdb"], target)
    own_parsed, own_assigned = prepare_parsed(own_pdbs, own_input, gpu)
    own_out = base / "own_structure"
    if not (own_out / "score_only").exists():
        command = [str(PYTHON), "protein_mpnn_run.py", "--jsonl_path", str(own_parsed), "--chain_id_jsonl", str(own_assigned), "--out_folder", str(own_out), "--score_only", "1", "--num_seq_per_target", str(orders), "--batch_size", str(orders), "--sampling_temp", "0.1", "--seed", "99173"]
        write_json(own_out / "command.json", {"command": command, "gpu": gpu})
        run_mpnn(command, gpu)
    own_scores = parse_npz_scores(own_out)
    return wt_scores, own_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="production")
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--mpnn-orders", type=int, default=16)
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifests" / f"af2_{args.run_name}.jsonl")
    characterized = analyze_folds(rows, args.run_name)
    write_csv(ROOT / "scores" / f"fold_metrics_{args.run_name}.csv", characterized)
    ridgey = score_ridgey(characterized, args.run_name, model="600m")
    ridgey_ensemble = score_ridgey(characterized, args.run_name, model="600m_ensemble")
    wt_scores, own_scores = score_mpnn(characterized, args.run_name, args.gpu, args.mpnn_orders)
    for row in characterized:
        row.update(ridgey[row["candidate_id"]])
        row.update(ridgey_ensemble[row["candidate_id"]])
        wt_score = wt_scores[row["sequence"]]
        own_score = own_scores[row["sequence"]]
        row.update(
            {
                "mpnn_wt_structure_nll": wt_score["nll"],
                "mpnn_wt_structure_geomean_probability": wt_score["geomean_probability"],
                "mpnn_wt_structure_npz": wt_score["npz_files"],
                "mpnn_mutant_structure_nll": own_score["nll"],
                "mpnn_mutant_structure_geomean_probability": own_score["geomean_probability"],
                "mpnn_mutant_structure_npz": own_score["npz_files"],
            }
        )
    write_csv(ROOT / "scores" / f"characterized_{args.run_name}.csv", characterized)
    write_json(ROOT / "manifests" / f"characterized_{args.run_name}.json", {"count": len(characterized), "ridgey_api": RIDGEY_API, "ridgey_models": ["600m", "600m_ensemble"], "ridgey_ensemble_members": 5, "mpnn_orders": args.mpnn_orders})
    print(json.dumps({"characterized": len(characterized), "af2_pass": sum(bool(r["af2_pass"]) for r in characterized)}, indent=2))


if __name__ == "__main__":
    main()
