#!/usr/bin/env python3
"""Targeted Ridgey runs for the TEV generation/validation analysis.

The script submits every missing job before polling so the hosted jobs can run
concurrently. Raw responses, job IDs, exact model names, source hashes, and
derived combined files are retained under the round-2 NVMe analysis directory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import requests


API = "https://shv-internal--ridgey-v2-prod-web.modal.run"
TEV_REANALYSIS = Path("/opt/dlami/nvme/tev_reanalysis")
PLATE_ROOT = Path("/opt/dlami/nvme/tev_plate_design")
OUT = TEV_REANALYSIS / "round2_generation_validation"
RAW = OUT / "raw" / "ridgey"
MANIFEST_PATH = OUT / "ridgey_jobs.json"
PREDICTION_BATCH_SIZE = 25
ATTRIBUTION_STEPS = 32
ATTRIBUTION_TOP_N = 100
ATTRIBUTION_TARGETS = ("stability", "solubility", "ec:3.4.22.44")
ATTRIBUTION_SCAFFOLDS = ("TEVd", "hyperTEV56", "hyperTEV60", "hyperTEV89")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def dump_gzip_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump(value, handle)


def load_gzip_json(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {
        "api": API,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
        "attribution_steps": ATTRIBUTION_STEPS,
        "jobs": [],
    }


def save_manifest(manifest: dict) -> None:
    dump_json(MANIFEST_PATH, manifest)


def submit(path: str, payload: dict) -> str:
    response = requests.post(f"{API}{path}", json=payload, timeout=600)
    response.raise_for_status()
    call_id = response.json()["call_id"]
    print(f"submitted {path} {call_id}", flush=True)
    return call_id


def wait(call_id: str, poll_seconds: int = 5):
    while True:
        response = requests.get(f"{API}/jobs/{call_id}", timeout=600)
        if response.status_code == 202:
            time.sleep(poll_seconds)
            continue
        response.raise_for_status()
        return response.json()


def plate_structures() -> list[dict]:
    table = PLATE_ROOT / "deliverables" / "tev_plate_97.csv"
    structure_dir = PLATE_ROOT / "deliverables" / "structures"
    items = []
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = structure_dir / f"{row['display_name']}.pdb"
            if not path.exists():
                raise FileNotFoundError(path)
            items.append({
                "name": row["id"],
                "filename": path.name,
                "content": path.read_text(),
                "chain_id": "A",
            })
    if len(items) != 97 or len({item["name"] for item in items}) != 97:
        raise ValueError("Expected 97 unique plate structures")
    return items


def attribution_structures() -> dict[str, dict]:
    paths = {
        "TEVd": TEV_REANALYSIS / "dataset" / "TEVd_1LVM_A_1-221.pdb",
        "hyperTEV56": TEV_REANALYSIS / "dataset" / "structures" / "hyperTEV56.pdb",
        "hyperTEV60": TEV_REANALYSIS / "dataset" / "structures" / "hyperTEV60.pdb",
        "hyperTEV89": TEV_REANALYSIS / "dataset" / "structures" / "hyperTEV89.pdb",
    }
    return {
        name: {
            "name": name,
            "filename": path.name,
            "content": path.read_text(),
            "chain_id": "A",
        }
        for name, path in paths.items()
    }


def source_manifest() -> dict:
    files = [
        PLATE_ROOT / "deliverables" / "tev_plate_97.csv",
        PLATE_ROOT / "deliverables" / "plate.json",
        TEV_REANALYSIS / "dataset" / "TEVd_1LVM_A_1-221.pdb",
        TEV_REANALYSIS / "dataset" / "design_metadata.csv",
    ]
    return {
        "sources": {str(path): sha256(path) for path in files},
        "ridgey_api": API,
        "prediction_models": {
            "600m": {"threshold": 1e-12, "purpose": "exact TEV EC plus scalar heads"},
            "6b": {"threshold": 0.5, "purpose": "strong-model scalar confirmation"},
        },
        "attribution_models": ["600m", "6b"],
        "attribution_targets": list(ATTRIBUTION_TARGETS),
        "attribution_scaffolds": list(ATTRIBUTION_SCAFFOLDS),
    }


def prediction_key(model: str, batch_index: int) -> str:
    return f"prediction:{model}:{batch_index}"


def attribution_key(scaffold: str, model: str, target: str) -> str:
    return f"attribution:{scaffold}:{model}:{target}"


def submit_missing_predictions(manifest: dict, structures: list[dict]) -> None:
    index = {job["key"]: job for job in manifest["jobs"]}
    batches = [
        structures[i : i + PREDICTION_BATCH_SIZE]
        for i in range(0, len(structures), PREDICTION_BATCH_SIZE)
    ]
    for model, threshold in (("600m", 1e-12), ("6b", 0.5)):
        combined_path = RAW / f"plate_predictions_{model}.json.gz"
        if combined_path.exists():
            continue
        for batch_index, batch in enumerate(batches):
            raw_path = RAW / f"plate_predictions_{model}_batch{batch_index}.json.gz"
            key = prediction_key(model, batch_index)
            if raw_path.exists() or key in index:
                continue
            call_id = submit("/jobs", {
                "model": model,
                "structures": batch,
                "threshold": threshold,
            })
            record = {
                "key": key,
                "kind": "prediction",
                "model": model,
                "threshold": threshold,
                "batch": batch_index,
                "names": [item["name"] for item in batch],
                "call_id": call_id,
            }
            manifest["jobs"].append(record)
            index[key] = record
            save_manifest(manifest)


def submit_missing_attributions(manifest: dict, structures: dict[str, dict]) -> None:
    index = {job["key"]: job for job in manifest["jobs"]}
    for scaffold in ATTRIBUTION_SCAFFOLDS:
        for model in ("600m", "6b"):
            for target in ATTRIBUTION_TARGETS:
                safe_target = target.replace(":", "_")
                raw_path = RAW / "attributions" / scaffold / f"{model}_{safe_target}.json.gz"
                key = attribution_key(scaffold, model, target)
                if raw_path.exists() or key in index:
                    continue
                call_id = submit("/attribution/jobs", {
                    "model": model,
                    "structure": structures[scaffold],
                    "target": target,
                    "steps": ATTRIBUTION_STEPS,
                    "top_n": ATTRIBUTION_TOP_N,
                })
                record = {
                    "key": key,
                    "kind": "attribution",
                    "scaffold": scaffold,
                    "model": model,
                    "target": target,
                    "steps": ATTRIBUTION_STEPS,
                    "call_id": call_id,
                }
                manifest["jobs"].append(record)
                index[key] = record
                save_manifest(manifest)


def collect_predictions(manifest: dict, structures: list[dict]) -> None:
    index = {job["key"]: job for job in manifest["jobs"]}
    batches = [
        structures[i : i + PREDICTION_BATCH_SIZE]
        for i in range(0, len(structures), PREDICTION_BATCH_SIZE)
    ]
    for model in ("600m", "6b"):
        combined_path = RAW / f"plate_predictions_{model}.json.gz"
        if combined_path.exists():
            print(f"skip existing {combined_path}", flush=True)
            continue
        combined = []
        for batch_index, batch in enumerate(batches):
            raw_path = RAW / f"plate_predictions_{model}_batch{batch_index}.json.gz"
            if raw_path.exists():
                result = load_gzip_json(raw_path)
            else:
                job = index[prediction_key(model, batch_index)]
                try:
                    result = wait(job["call_id"])
                except requests.HTTPError as exc:
                    code = exc.response.status_code if exc.response is not None else None
                    if code is None or code < 500 or job.get("retry_call_id"):
                        raise
                    batch_payload = {
                        "model": model,
                        "structures": batch,
                        "threshold": job["threshold"],
                    }
                    job["retry_call_id"] = submit("/jobs", batch_payload)
                    save_manifest(manifest)
                    result = wait(job["retry_call_id"])
                dump_gzip_json(raw_path, result)
            if [item["name"] for item in result] != [item["name"] for item in batch]:
                raise ValueError(f"Prediction name mismatch for {model} batch {batch_index}")
            combined.extend(result)
            print(f"collected prediction {model} batch {batch_index}", flush=True)
        if len(combined) != len(structures):
            raise ValueError(f"Incomplete predictions for {model}: {len(combined)}")
        dump_gzip_json(combined_path, combined)


def collect_attributions(manifest: dict) -> None:
    index = {job["key"]: job for job in manifest["jobs"]}
    for scaffold in ATTRIBUTION_SCAFFOLDS:
        for model in ("600m", "6b"):
            for target in ATTRIBUTION_TARGETS:
                safe_target = target.replace(":", "_")
                raw_path = RAW / "attributions" / scaffold / f"{model}_{safe_target}.json.gz"
                if raw_path.exists():
                    continue
                job = index[attribution_key(scaffold, model, target)]
                result = wait(job.get("retry_call_id", job["call_id"]))
                dump_gzip_json(raw_path, result)
                print(f"collected attribution {scaffold} {model} {target}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    health = requests.get(f"{API}/health", timeout=60)
    health.raise_for_status()
    dump_json(OUT / "ridgey_health.json", health.json())
    dump_json(OUT / "source_manifest.json", source_manifest())

    structures = plate_structures()
    attribution_inputs = attribution_structures()
    manifest = load_manifest()
    submit_missing_predictions(manifest, structures)
    submit_missing_attributions(manifest, attribution_inputs)
    collect_predictions(manifest, structures)
    collect_attributions(manifest)
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_manifest(manifest)
    print("round-2 Ridgey jobs complete", flush=True)


if __name__ == "__main__":
    main()
