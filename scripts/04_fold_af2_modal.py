#!/usr/bin/env python3
"""Run paper-matched query-swapped-A3M AF2 in up to 50 Modal containers.

Invoke from aws0:
  /opt/pytorch/bin/modal run 04_fold_af2_modal.py --run-name production

The local entrypoint runs on aws0, submits all folds before waiting, downloads
each complete raw tarball to NVMe, and extracts it under folds/<run>/outputs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal


APP_NAME = "tev-m3-af2-a3m"
BUCKET = "shvaibackups"
IMAGE_NAME = "ghcr.io/sokrypton/colabfold:1.6.2-cuda12"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("af-multimer", create_if_missing=False)
image = modal.Image.from_registry(IMAGE_NAME).pip_install("boto3")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    secrets=[modal.Secret.from_name("shv-internal-modal-aws")],
    timeout=30 * 60,
    max_containers=50,
)
@modal.concurrent(max_inputs=1)
def fold_candidate(candidate_id: str, a3m_text: str, run_name: str) -> dict:
    import boto3

    candidate_id = safe_name(candidate_id)
    with tempfile.TemporaryDirectory(prefix=f"tev_{candidate_id}_") as tmp:
        work = Path(tmp)
        input_path = work / f"{candidate_id}.a3m"
        output_dir = work / "output"
        output_dir.mkdir()
        input_path.write_text(a3m_text)
        command = [
            "colabfold_batch",
            "--disable-unified-memory",
            "--data", "/data",
            "--model-type", "alphafold2",
            "--model-order", "3",
            "--num-models", "1",
            "--num-recycle", "6",
            "--num-ensemble", "1",
            "--num-seeds", "1",
            "--random-seed", "0",
            "--rank", "plddt",
            "--stop-at-score", "100",
            "--overwrite-existing-results",
            str(input_path),
            str(output_dir),
        ]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output_dir / "modal_command.json").write_text(json.dumps({"command": command, "image": IMAGE_NAME}, indent=2) + "\n")
        (output_dir / "colabfold.log").write_text(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"ColabFold failed for {candidate_id}:\n{completed.stdout[-4000:]}")
        pdbs = sorted(output_dir.glob("*_unrelaxed_rank_001_*.pdb"))
        scores = sorted(output_dir.glob("*_scores_rank_001_*.json"))
        if not pdbs or not scores:
            raise RuntimeError(f"missing rank-001 outputs for {candidate_id}: {[p.name for p in output_dir.iterdir()]}")

        tar_path = work / f"{candidate_id}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as archive:
            archive.add(output_dir, arcname=".")
        key = f"spandrel/projects/tev_plate_design/af2/{run_name}/{candidate_id}.tar.gz"
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.upload_file(str(tar_path), BUCKET, key)
        presigned = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=3600)
        score = json.loads(scores[0].read_text())
        return {
            "candidate_id": candidate_id,
            "s3_path": f"s3://{BUCKET}/{key}",
            "presigned_url": presigned,
            "pdb_name": pdbs[0].name,
            "scores_name": scores[0].name,
            "plddt": score.get("plddt"),
            "ptm": score.get("ptm"),
            "command": command,
        }


def safe_extract(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (out_dir / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe tar member: {member.name}")
        archive.extractall(out_dir)


@app.local_entrypoint()
def main(
    run_name: str = "production",
    root: str = "/opt/dlami/nvme/tev_plate_design",
    limit: int = 0,
) -> None:
    import requests

    root_path = Path(root)
    manifest_path = root_path / "manifests" / f"af2_{run_name}.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if limit > 0:
        rows = rows[:limit]
    calls_path = root_path / "manifests" / f"af2_modal_calls_{run_name}.json"
    calls_manifest = json.loads(calls_path.read_text()) if calls_path.exists() else {"app": APP_NAME, "max_containers": 50, "calls": []}
    by_candidate = {row["candidate_id"]: row for row in calls_manifest["calls"]}

    # Submit every missing call before blocking on results.
    calls = []
    for row in rows:
        candidate_id = safe_name(row["candidate_id"])
        output_dir = root_path / "folds" / run_name / "outputs" / candidate_id
        existing = sorted(output_dir.glob("*_unrelaxed_rank_001_*.pdb"))
        if existing:
            continue
        record = by_candidate.get(candidate_id)
        if record and record.get("function_call_id"):
            call = modal.FunctionCall.from_id(record["function_call_id"])
        else:
            a3m_text = Path(row["a3m_path"]).read_text()
            call = fold_candidate.spawn(candidate_id, a3m_text, run_name)
            record = {"candidate_id": candidate_id, "function_call_id": call.object_id, "status": "submitted", "a3m_path": row["a3m_path"]}
            calls_manifest["calls"].append(record)
            by_candidate[candidate_id] = record
            calls_path.write_text(json.dumps(calls_manifest, indent=2, sort_keys=True) + "\n")
        calls.append((candidate_id, call, record))

    raw_dir = root_path / "folds" / run_name / "modal_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for candidate_id, call, record in calls:
        try:
            result = call.get()
            result_path = raw_dir / f"{candidate_id}.result.json"
            result_path.write_text(json.dumps({k: v for k, v in result.items() if k != "presigned_url"}, indent=2, sort_keys=True) + "\n")
            tar_path = raw_dir / f"{candidate_id}.tar.gz"
            if not tar_path.exists():
                with requests.get(result["presigned_url"], stream=True, timeout=600) as response:
                    response.raise_for_status()
                    with tar_path.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
            output_dir = root_path / "folds" / run_name / "outputs" / candidate_id
            safe_extract(tar_path, output_dir)
            record.update({"status": "completed", "s3_path": result["s3_path"], "tar_path": str(tar_path), "result_path": str(result_path)})
        except Exception as error:
            record.update({"status": "failed", "error": repr(error)})
        calls_path.write_text(json.dumps(calls_manifest, indent=2, sort_keys=True) + "\n")
        print(candidate_id, record["status"], flush=True)

    summary = {status: sum(row.get("status") == status for row in calls_manifest["calls"]) for status in ("submitted", "completed", "failed")}
    print(json.dumps(summary, indent=2))
