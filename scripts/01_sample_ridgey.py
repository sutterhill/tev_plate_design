#!/usr/bin/env python3
"""Sample Ridgey 600M sequences on the 1LVM backbone under the exact m3 mask."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from common import ROOT, sequence_features, validate_m3_sequence, write_csv, write_fasta, write_json


API = "https://shv-internal--ridgey-v2-prod-web.modal.run"


def wait(call_id: str, poll_seconds: int = 5) -> dict:
    while True:
        response = requests.get(f"{API}/jobs/{call_id}", timeout=120)
        if response.status_code == 202:
            time.sleep(poll_seconds)
            continue
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperatures", default="0.7,1.0")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--num-per-job", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--run-name", default="production")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    temperatures = [float(x) for x in args.temperatures.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    if args.smoke:
        temperatures, seeds, args.num_per_job = [1.0], [99173], 3

    pdb = (ROOT / "inputs" / "TEVd_1LVM_A_1-221.pdb").read_text()
    aligned = (ROOT / "inputs" / "m3_aligned_sequence.txt").read_text().strip()
    run_name = "smoke" if args.smoke else args.run_name
    raw_dir = ROOT / "raw" / "ridgey_inverse_fold" / run_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifests" / f"ridgey_inverse_fold_{run_name}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"api": API, "jobs": []}
    job_index = {row["token"]: row for row in manifest["jobs"]}
    all_rows: list[dict] = []
    jobs: list[dict] = []
    for temperature in temperatures:
        for seed in seeds:
            token = f"t{temperature:g}_seed{seed}_n{args.num_per_job}"
            request_path = raw_dir / f"{token}.request.json"
            response_path = raw_dir / f"{token}.response.json.gz"
            payload = {
                "model": "600m",
                "mmcif": pdb,
                "filename": "TEVd_1LVM_A_1-221.pdb",
                "chain_id": "A",
                "aligned_sequence": aligned,
                "temperature": temperature,
                "steps": args.steps,
                "num_seqs": args.num_per_job,
                "seed": seed,
            }
            write_json(request_path, payload)
            record = job_index.get(token)
            if response_path.exists():
                call_id = record["call_id"] if record else "cached"
            elif record:
                # Reuse an already-submitted job after a driver restart.
                call_id = record["call_id"]
            else:
                submitted = requests.post(f"{API}/inverse-fold/jobs", json=payload, timeout=120)
                submitted.raise_for_status()
                call_id = submitted.json()["call_id"]
                record = {"token": token, "call_id": call_id, "request": str(request_path), "response": str(response_path), "status": "submitted"}
                manifest["jobs"].append(record)
                job_index[token] = record
            jobs.append({"token": token, "temperature": temperature, "seed": seed, "call_id": call_id, "response_path": response_path})

    # All missing jobs have now been submitted; poll them concurrently.
    write_json(manifest_path, manifest)
    missing = [job for job in jobs if not job["response_path"].exists()]
    with ThreadPoolExecutor(max_workers=min(50, max(1, len(missing)))) as executor:
        results = list(executor.map(lambda job: wait(job["call_id"]), missing))
    for job, result in zip(missing, results):
        response_path = job["response_path"]
        with gzip.open(response_path, "wt") as handle:
            json.dump(result, handle)
        job_index[job["token"]]["status"] = "completed"
    write_json(manifest_path, manifest)

    for job in jobs:
        with gzip.open(job["response_path"], "rt") as handle:
            result = json.load(handle)
        token = job["token"]
        temperature = job["temperature"]
        seed = job["seed"]
        call_id = job["call_id"]
        response_path = job["response_path"]
        for index, sequence in enumerate(result["sequences"]):
            reasons = validate_m3_sequence(sequence, forbid_c_at_mutable=True)
            row = {
                "candidate_id": f"ridgey_{token}_{index:04d}",
                "method": "ridgey",
                "temperature": temperature,
                "seed": seed,
                "sample_index": index,
                "valid_for_paper_parity": not reasons,
                "rejection_reasons": "|".join(reasons),
                **sequence_features(sequence),
                "raw_response": str(response_path),
                "ridgey_call_id": call_id,
            }
            all_rows.append(row)

    dedup: dict[str, dict] = {}
    for row in all_rows:
        if row["valid_for_paper_parity"] and row["sequence"] not in dedup:
            dedup[row["sequence"]] = row
    rows = list(dedup.values())
    table = ROOT / "candidates" / f"ridgey_{run_name}_valid_unique.csv"
    fasta = ROOT / "candidates" / f"ridgey_{run_name}_valid_unique.fasta"
    rejection_table = ROOT / "candidates" / f"ridgey_{run_name}_all_with_rejections.csv"
    write_csv(rejection_table, all_rows)
    write_csv(table, rows)
    write_fasta(fasta, [(r["candidate_id"], r["sequence"]) for r in rows])
    manifest["summary"] = {
        "raw_samples": len(all_rows),
        "paper_parity_samples": sum(bool(r["valid_for_paper_parity"]) for r in all_rows),
        "paper_parity_unique": len(rows),
        "table": str(table),
        "fasta": str(fasta),
        "all_with_rejections": str(rejection_table),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
