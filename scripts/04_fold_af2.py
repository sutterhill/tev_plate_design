#!/usr/bin/env python3
"""Fold query-swapped TEV A3Ms with local AF2 model 3 and six recycles."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ROOT, read_jsonl, write_json


IMAGE = "ghcr.io/sokrypton/colabfold:1.6.2-cuda12"
CACHE = Path("/home/ubuntu/colabfold_cache")


def fold_one(row: dict, gpu: str, run_name: str) -> dict:
    candidate_id = row["candidate_id"]
    out_dir = ROOT / "folds" / run_name / "outputs" / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "DONE.json"
    if done.exists() and list(out_dir.glob("*_unrelaxed_rank_001_*.pdb")):
        return json.loads(done.read_text())
    relative_a3m = Path(row["a3m_path"]).relative_to(ROOT)
    relative_out = out_dir.relative_to(ROOT)
    command = [
        "docker", "run", "--rm", "--gpus", f"device={gpu}",
        "-v", f"{ROOT}:/work",
        "-v", f"{CACHE}:/cache:ro",
        IMAGE,
        "colabfold_batch",
        "--data", "/cache",
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
        f"/work/{relative_a3m}",
        f"/work/{relative_out}",
    ]
    write_json(out_dir / "command.json", {"command": command, "gpu": gpu, "candidate": row})
    log_path = ROOT / "logs" / f"af2_{run_name}_{candidate_id}.log"
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    pdbs = sorted(out_dir.glob("*_unrelaxed_rank_001_*.pdb"))
    scores = sorted(out_dir.glob("*_scores_rank_001_*.json"))
    result = {
        "candidate_id": candidate_id,
        "gpu": gpu,
        "returncode": completed.returncode,
        "pdb": str(pdbs[0]) if pdbs else None,
        "scores_json": str(scores[0]) if scores else None,
        "log": str(log_path),
    }
    if completed.returncode != 0 or not pdbs:
        write_json(out_dir / "FAILED.json", result)
        raise RuntimeError(f"AF2 failed for {candidate_id}; see {log_path}")
    write_json(done, result)
    return result


def run_gpu_shard(rows: list[dict], gpu: str, run_name: str) -> list[dict]:
    return [fold_one(row, gpu, run_name) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", choices=["smoke", "production"], default="production")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "manifests" / f"af2_{args.run_name}.jsonl")
    if args.limit is not None:
        rows = rows[: args.limit]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    shards = [rows[i::len(gpus)] for i in range(len(gpus))]
    outputs: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_gpu_shard, shard, gpu, args.run_name) for gpu, shard in zip(gpus, shards) if shard]
        for future in futures:
            outputs.extend(future.result())
    write_json(ROOT / "manifests" / f"af2_{args.run_name}_completed.json", {"image": IMAGE, "outputs": outputs})
    print(json.dumps({"completed": len(outputs), "run_name": args.run_name}, indent=2))


if __name__ == "__main__":
    main()

