#!/usr/bin/env python3
"""WT-coordinate Ridgey prescore and diverse enriched AF2 pool preparation."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from common import ROOT, greedy_maximin, read_csv, sha256, write_csv, write_fasta, write_json, write_jsonl


API = "https://shv-internal--ridgey-v2-prod-web.modal.run"
AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def wait(call_id: str) -> list[dict]:
    while True:
        response = requests.get(f"{API}/jobs/{call_id}", timeout=120)
        if response.status_code == 202:
            time.sleep(5)
            continue
        response.raise_for_status()
        return response.json()


def mutate_pdb(parent_text: str, sequence: str) -> str:
    lines = []
    seen: list[tuple[str, str]] = []
    index_by_residue: dict[tuple[str, str], int] = {}
    for line in parent_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[21].strip() in ("", "A"):
            key = (line[22:26], line[26:27])
            if key not in index_by_residue:
                index_by_residue[key] = len(seen)
                seen.append(key)
            index = index_by_residue[key]
            if index >= len(sequence):
                raise ValueError("PDB has more residues than sequence")
            line = line[:17] + AA3[sequence[index]] + line[20:]
        lines.append(line)
    if len(seen) != len(sequence):
        raise ValueError(f"PDB residues={len(seen)} sequence={len(sequence)}")
    return "\n".join(lines) + "\n"


def swap_query(parent_a3m: str, sequence: str, name: str) -> str:
    lines = parent_a3m.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(">"))
    query = next(i for i in range(header + 1, len(lines)) if lines[i] and not lines[i].startswith(">"))
    lines[header] = f">{name}"
    lines[query] = sequence
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", default="additional")
    parser.add_argument("--output-run", default="ridgey_enriched")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--enrichment-pool", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()

    candidates = read_csv(ROOT / "candidates" / f"ridgey_{args.input_run}_valid_unique.csv")
    previous = {row["sequence"] for row in read_csv(ROOT / "candidates" / "ridgey_production_valid_unique.csv")}
    candidates = [row for row in candidates if row["sequence"] not in previous]
    for row in candidates:
        row["n_mutations"] = int(row["n_mutations"])
    parent_pdb = (ROOT / "inputs" / "TEVd_1LVM_A_1-221.pdb").read_text()
    raw_dir = ROOT / "raw" / "ridgey_wt_coordinate_prescore" / args.input_run
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifests" / f"ridgey_wt_coordinate_prescore_{args.input_run}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"api": API, "model": "600m_ensemble", "jobs": []}
    job_index = {row["batch"]: row for row in manifest["jobs"]}

    wt = "".join(line.strip() for line in (ROOT / "inputs" / "TEVd.fasta").read_text().splitlines() if not line.startswith(">"))
    items = [{"candidate_id": "TEVd", "sequence": wt}]
    items.extend(candidates)
    jobs = []
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        batch_number = start // args.batch_size
        request_path = raw_dir / f"batch_{batch_number:03d}.request.json.gz"
        response_path = raw_dir / f"batch_{batch_number:03d}.response.json.gz"
        payload = {
            "model": "600m_ensemble",
            "structures": [
                {"name": row["candidate_id"], "filename": f"{safe_name(row['candidate_id'])}_on_1LVM.pdb", "content": mutate_pdb(parent_pdb, row["sequence"]), "chain_id": "A"}
                for row in batch
            ],
            "threshold": 0.5,
        }
        if not request_path.exists():
            with gzip.open(request_path, "wt") as handle:
                json.dump(payload, handle)
        record = job_index.get(batch_number)
        if response_path.exists():
            call_id = record["call_id"] if record else "cached"
        elif record:
            call_id = record["call_id"]
        else:
            response = requests.post(f"{API}/jobs", json=payload, timeout=120)
            if not response.ok:
                raise RuntimeError(
                    f"Ridgey submission failed for batch {batch_number}: "
                    f"HTTP {response.status_code}: {response.text[:1000]}"
                )
            call_id = response.json()["call_id"]
            record = {"batch": batch_number, "call_id": call_id, "status": "submitted", "request": str(request_path), "response": str(response_path)}
            manifest["jobs"].append(record)
            job_index[batch_number] = record
        jobs.append({"batch": batch_number, "call_id": call_id, "response": response_path})
    write_json(manifest_path, manifest)
    missing = [job for job in jobs if not job["response"].exists()]
    if len(missing) > 50:
        raise RuntimeError(f"{len(missing)} prescore jobs exceeds the 50-call concurrency guard")
    with ThreadPoolExecutor(max_workers=max(1, len(missing))) as executor:
        results = list(executor.map(lambda job: wait(job["call_id"]), missing))
    for job, result in zip(missing, results):
        with gzip.open(job["response"], "wt") as handle:
            json.dump(result, handle)
        job_index[job["batch"]]["status"] = "completed"
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)

    scores: dict[str, dict] = {}
    for job in jobs:
        for item in json.load(gzip.open(job["response"], "rt")):
            members = item["predictions"]
            stability = [float(member["stability"]) for member in members]
            solubility = [float(member["solubility"]) for member in members]
            scores[item["name"]] = {
                "prescore_stability": sum(stability) / len(stability),
                "prescore_solubility": sum(solubility) / len(solubility),
                "prescore_stability_members": "|".join(map(str, stability)),
                "prescore_solubility_members": "|".join(map(str, solubility)),
                "prescore_models": "|".join(item.get("prediction_models", [])),
                "prescore_raw_response": str(job["response"]),
            }
    wt_score = scores["TEVd"]
    for row in candidates:
        row.update(scores[row["candidate_id"]])
        row["prescore_stability_delta_vs_wt"] = row["prescore_stability"] - wt_score["prescore_stability"]
        row["prescore_solubility_delta_vs_wt"] = row["prescore_solubility"] - wt_score["prescore_solubility"]
        stable = [float(x) for x in row["prescore_stability_members"].split("|")]
        soluble = [float(x) for x in row["prescore_solubility_members"].split("|")]
        wt_stable = [float(x) for x in wt_score["prescore_stability_members"].split("|")]
        wt_soluble = [float(x) for x in wt_score["prescore_solubility_members"].split("|")]
        row["prescore_paired_members_improving_both"] = sum(s > ws and q > wq for s, ws, q, wq in zip(stable, wt_stable, soluble, wt_soluble))
        row["prescore_mean_both_improve"] = row["prescore_stability_delta_vs_wt"] > 0 and row["prescore_solubility_delta_vs_wt"] > 0
    candidates.sort(
        key=lambda row: (
            row["prescore_mean_both_improve"],
            row["prescore_paired_members_improving_both"],
            row["prescore_solubility_delta_vs_wt"],
            row["prescore_stability_delta_vs_wt"],
        ),
        reverse=True,
    )
    enriched_pool = candidates[: min(args.enrichment_pool, len(candidates))]
    selected = greedy_maximin(enriched_pool, min(args.count, len(enriched_pool)))
    write_csv(ROOT / "scores" / f"ridgey_wt_coordinate_prescore_{args.input_run}.csv", candidates)

    parent_a3m_path = ROOT / "inputs" / "tev_parent_current_mmseqs.a3m"
    parent_a3m = parent_a3m_path.read_text()
    a3m_dir = ROOT / "folds" / args.output_run / "a3m"
    a3m_dir.mkdir(parents=True, exist_ok=True)
    af2_manifest = []
    for row in selected:
        candidate_id = safe_name(row["candidate_id"])
        a3m_path = a3m_dir / f"{candidate_id}.a3m"
        a3m_path.write_text(swap_query(parent_a3m, row["sequence"], candidate_id))
        af2_manifest.append({**row, "candidate_id": candidate_id, "a3m_path": str(a3m_path), "a3m_sha256": sha256(a3m_path), "parent_a3m": str(parent_a3m_path), "parent_a3m_sha256": sha256(parent_a3m_path), "af2_model_type": "alphafold2", "af2_model_order": 3, "af2_recycles": 6, "explicit_structure_template": False, "prefold_selection_method": f"ensemble_mean_both_then_joint_votes_top_{len(enriched_pool)}_then_greedy_maximin"})
    write_jsonl(ROOT / "manifests" / f"af2_{args.output_run}.jsonl", af2_manifest)
    write_csv(ROOT / "candidates" / f"prefold_{args.output_run}.csv", af2_manifest)
    write_csv(ROOT / "candidates" / f"ridgey_{args.output_run}_valid_unique.csv", af2_manifest)
    write_fasta(
        ROOT / "candidates" / f"ridgey_{args.output_run}_valid_unique.fasta",
        [(row["candidate_id"], row["sequence"]) for row in af2_manifest],
    )
    print(json.dumps({"new_unique_strict": len(candidates), "prescore_jobs": len(jobs), "enrichment_pool": len(enriched_pool), "selected_for_af2": len(af2_manifest), "wt_prescore": wt_score}, indent=2))


if __name__ == "__main__":
    main()
