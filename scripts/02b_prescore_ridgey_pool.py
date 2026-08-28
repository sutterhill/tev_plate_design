#!/usr/bin/env python3
"""Pre-score the Ridgey pool on the frozen 1LVM coordinates.

This is an enrichment-only stage before expensive AF2 folding.  Every candidate
sequence is written into the residue-name fields of the same 1LVM coordinate
set, then evaluated with Ridgey 600M.  Final selection still uses Ridgey scores
on each candidate's own AF2 structure.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from common import ROOT, read_fasta, write_csv, write_json


API = "https://shv-internal--ridgey-v2-prod-web.modal.run"
AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def mutate_pdb_sequence(pdb_text: str, sequence: str) -> str:
    """Replace chain-A residue names without altering any coordinates."""
    output: list[str] = []
    seen: dict[tuple[str, str, str], int] = {}
    next_index = 0
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27 and line[21].strip() in ("", "A"):
            key = (line[21], line[22:26], line[26])
            if key not in seen:
                seen[key] = next_index
                next_index += 1
            index = seen[key]
            if index >= len(sequence):
                raise ValueError(f"PDB contains more than {len(sequence)} residues")
            line = f"{line[:17]}{AA3[sequence[index]]:>3}{line[20:]}"
        output.append(line)
    if next_index != len(sequence):
        raise ValueError(f"PDB has {next_index} residues; sequence has {len(sequence)}")
    return "\n".join(output) + "\n"


def wait(call_id: str) -> list[dict]:
    while True:
        response = requests.get(f"{API}/jobs/{call_id}", timeout=120)
        if response.status_code == 202:
            time.sleep(5)
            continue
        response.raise_for_status()
        return response.json()


def collect_one(record: dict, response_path: Path) -> tuple[dict, list[dict]]:
    if response_path.exists():
        with gzip.open(response_path, "rt") as handle:
            result = json.load(handle)
    else:
        result = wait(record["call_id"])
        with gzip.open(response_path, "wt") as handle:
            json.dump(result, handle)
    return record, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()

    candidate_path = ROOT / "candidates" / "ridgey_production_valid_unique.csv"
    with candidate_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    wt = read_fasta(ROOT / "inputs" / "TEVd.fasta")[0][1]
    rows.insert(0, {"candidate_id": "TEVd", "sequence": wt, "method": "parent"})
    pdb_text = (ROOT / "inputs" / "TEVd_1LVM_A_1-221.pdb").read_text()

    raw_dir = ROOT / "raw" / "ridgey_wt_coordinate_prescore"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifests" / "ridgey_wt_coordinate_prescore.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "api": API,
        "model": "600m",
        "purpose": "AF2 pre-fold enrichment only; final scores use own AF2 structure",
        "jobs": [],
    }
    by_batch = {int(record["batch"]): record for record in manifest["jobs"]}
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

    # Submit all missing batches before waiting, keeping total calls well below 50.
    for batch_number, batch in enumerate(batches):
        response_path = raw_dir / f"batch_{batch_number:03d}.response.json.gz"
        if response_path.exists() or batch_number in by_batch:
            continue
        payload = {
            "model": "600m",
            "structures": [
                {
                    "name": row["candidate_id"],
                    "filename": f"{row['candidate_id']}_on_1LVM_coords.pdb",
                    "content": mutate_pdb_sequence(pdb_text, row["sequence"]),
                    "chain_id": "A",
                }
                for row in batch
            ],
            "threshold": 0.5,
        }
        request_path = raw_dir / f"batch_{batch_number:03d}.request.json.gz"
        with gzip.open(request_path, "wt") as handle:
            json.dump(payload, handle)
        response = requests.post(f"{API}/jobs", json=payload, timeout=120)
        response.raise_for_status()
        record = {
            "batch": batch_number,
            "call_id": response.json()["call_id"],
            "count": len(batch),
            "request": str(request_path),
            "response": str(response_path),
            "status": "submitted",
        }
        manifest["jobs"].append(record)
        by_batch[batch_number] = record
        write_json(manifest_path, manifest)

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(batches))) as executor:
        futures = {
            executor.submit(
                collect_one,
                by_batch[batch_number],
                raw_dir / f"batch_{batch_number:03d}.response.json.gz",
            ): batch_number
            for batch_number in range(len(batches))
        }
        for future in as_completed(futures):
            record, items = future.result()
            record["status"] = "completed"
            for item in items:
                prediction = item["predictions"]
                results[item["name"]] = {
                    "ridgey_wt_coordinate_stability": float(prediction["stability"]),
                    "ridgey_wt_coordinate_solubility": float(prediction["solubility"]),
                    "ridgey_wt_coordinate_raw_response": record["response"],
                    "ridgey_wt_coordinate_call_id": record["call_id"],
                }
            write_json(manifest_path, manifest)

    if len(results) != len(rows):
        raise RuntimeError(f"expected {len(rows)} Ridgey scores, received {len(results)}")
    output = [{**row, **results[row["candidate_id"]]} for row in rows]
    write_csv(ROOT / "scores" / "ridgey_wt_coordinate_prescore.csv", output)
    manifest["status"] = "completed"
    manifest["count"] = len(output)
    write_json(manifest_path, manifest)
    print(json.dumps({"scored": len(output), "batches": len(batches)}, indent=2))


if __name__ == "__main__":
    main()
