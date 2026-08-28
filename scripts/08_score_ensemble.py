#!/usr/bin/env python3
"""Score all folded candidates with Ridgey 600M ensemble and paired-member gates."""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from common import ROOT, read_csv, write_csv, write_json


API = "https://shv-internal--ridgey-v2-prod-web.modal.run"


def wait(call_id: str) -> list[dict]:
    while True:
        response = requests.get(f"{API}/jobs/{call_id}", timeout=120)
        if response.status_code == 202:
            time.sleep(5)
            continue
        response.raise_for_status()
        return response.json()


def main() -> None:
    runs = ["production", "ridgey_extension", "ridgey_enriched"]
    rows = []
    seen = set()
    for run in runs:
        path = ROOT / "scores" / f"characterized_{run}.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            if row["sequence"] not in seen:
                row["characterization_run"] = run
                rows.append(row)
                seen.add(row["sequence"])
    raw_dir = ROOT / "raw" / "ridgey_ensemble_structure_scores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifests" / "ridgey_ensemble_structure_scores.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"api": API, "model": "600m_ensemble", "jobs": []}
    job_index = {row["batch"]: row for row in manifest["jobs"]}
    jobs = []
    batch_size = 30
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_number = start // batch_size
        request_path = raw_dir / f"batch_{batch_number:03d}.request.json.gz"
        response_path = raw_dir / f"batch_{batch_number:03d}.response.json.gz"
        payload = {"model": "600m_ensemble", "structures": [{"name": row["candidate_id"], "filename": row["candidate_id"] + ".pdb", "content": open(row["af2_pdb"]).read(), "chain_id": "A"} for row in batch], "threshold": 0.5}
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
            response.raise_for_status()
            call_id = response.json()["call_id"]
            record = {"batch": batch_number, "call_id": call_id, "status": "submitted", "request": str(request_path), "response": str(response_path)}
            manifest["jobs"].append(record)
            job_index[batch_number] = record
        jobs.append({"batch": batch_number, "call_id": call_id, "response": response_path})
    write_json(manifest_path, manifest)
    missing = [job for job in jobs if not job["response"].exists()]
    if len(missing) > 50:
        raise RuntimeError(f"{len(missing)} ensemble jobs exceeds 50-call guard")
    with ThreadPoolExecutor(max_workers=max(1, len(missing))) as executor:
        results = list(executor.map(lambda job: wait(job["call_id"]), missing))
    for job, result in zip(missing, results):
        with gzip.open(job["response"], "wt") as handle:
            json.dump(result, handle)
        job_index[job["batch"]]["status"] = "completed"
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)

    predictions = {}
    for job in jobs:
        for item in json.load(gzip.open(job["response"], "rt")):
            members = item["predictions"]
            predictions[item["name"]] = {
                "models": item["prediction_models"],
                "stability": [float(member["stability"]) for member in members],
                "solubility": [float(member["solubility"]) for member in members],
                "raw": str(job["response"]),
            }
    wt = next(row for row in rows if row["method"] == "parent")
    wt_pred = predictions[wt["candidate_id"]]
    for row in rows:
        pred = predictions[row["candidate_id"]]
        stable = pred["stability"]
        soluble = pred["solubility"]
        paired = [s > ws and q > wq for s, ws, q, wq in zip(stable, wt_pred["stability"], soluble, wt_pred["solubility"])]
        mean_stability = sum(stable) / len(stable)
        mean_solubility = sum(soluble) / len(soluble)
        wt_mean_stability = sum(wt_pred["stability"]) / len(wt_pred["stability"])
        wt_mean_solubility = sum(wt_pred["solubility"]) / len(wt_pred["solubility"])
        row.update({
            "ridgey_ensemble_models": "|".join(pred["models"]),
            "ridgey_ensemble_stability_members": "|".join(map(str, stable)),
            "ridgey_ensemble_solubility_members": "|".join(map(str, soluble)),
            "ridgey_ensemble_stability_mean": mean_stability,
            "ridgey_ensemble_solubility_mean": mean_solubility,
            "ridgey_ensemble_stability_delta_vs_wt": mean_stability - wt_mean_stability,
            "ridgey_ensemble_solubility_delta_vs_wt": mean_solubility - wt_mean_solubility,
            "ridgey_ensemble_paired_members_improving_both": sum(paired),
            "ridgey_ensemble_strict_pass": mean_stability > wt_mean_stability and mean_solubility > wt_mean_solubility and sum(paired) >= 4,
            "ridgey_ensemble_raw_response": pred["raw"],
        })
    write_csv(ROOT / "scores" / "characterized_all_with_ensemble.csv", rows)
    strict = sum(str(row["ridgey_ensemble_strict_pass"]).lower() == "true" and row["method"] == "ridgey" and str(row["af2_pass"]).lower() == "true" for row in rows)
    summary = {"rows": len(rows), "ridgey_strict_af2_and_ensemble": strict, "wt_stability_mean": sum(wt_pred["stability"]) / 5, "wt_solubility_mean": sum(wt_pred["solubility"]) / 5, "required_paired_members": 4}
    write_json(ROOT / "scores" / "characterized_all_with_ensemble.summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
