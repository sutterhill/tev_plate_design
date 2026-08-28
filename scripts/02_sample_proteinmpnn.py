#!/usr/bin/env python3
"""Generate ProteinMPNN m3 candidates using the paper's exact command recipe."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from common import ROOT, sequence_features, validate_m3_sequence, write_csv, write_fasta, write_json


MPNN = Path("/home/ubuntu/ml-apps/ProteinMPNN/ProteinMPNN")
PYTHON = Path("/opt/pytorch/bin/python")


def run(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=MPNN, env=env, check=True)


def parse_generated(path: Path) -> list[dict]:
    rows: list[dict] = []
    header: str | None = None
    sequence: list[str] = []
    records: list[tuple[str, str]] = []
    for raw in path.read_text().splitlines():
        if raw.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header, sequence = raw[1:], []
        else:
            sequence.append(raw.strip())
    if header is not None:
        records.append((header, "".join(sequence)))
    for sample_index, (description, seq) in enumerate(records[1:]):
        values = dict(re.findall(r"([A-Za-z_]+)=\s*([^,]+)", description))
        rows.append(
            {
                "sample_index": sample_index,
                "temperature": float(values.get("T", "nan")),
                "mpnn_generation_nll": float(values.get("score", "nan")),
                "mpnn_generation_global_nll": float(values.get("global_score", "nan")),
                "mpnn_header": description,
                "sequence": seq,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-temperature", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.per_temperature, args.batch_size, args.seed = 1, 1, 99173
    run_name = "smoke" if args.smoke else "production"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    raw_dir = ROOT / "raw" / "proteinmpnn" / run_name
    input_dir = raw_dir / "input"
    pdb_dir = input_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_target = pdb_dir / "TEVd.pdb"
    if not pdb_target.exists():
        pdb_target.write_bytes((ROOT / "inputs" / "TEVd_1LVM_A_1-221.pdb").read_bytes())
    parsed = input_dir / "parsed.jsonl"
    assigned = input_dir / "assigned.jsonl"
    fixed_jsonl = input_dir / "m3_fixed.jsonl"
    if not parsed.exists():
        run([str(PYTHON), "helper_scripts/parse_multiple_chains.py", "--input_path", str(pdb_dir), "--output_path", str(parsed)], env)
    if not assigned.exists():
        run([str(PYTHON), "helper_scripts/assign_fixed_chains.py", "--input_path", str(parsed), "--output_path", str(assigned), "--chain_list", "A"], env)
    if not fixed_jsonl.exists():
        fixed = json.loads((ROOT / "inputs" / "m3_mask.json").read_text())["fixed_positions"]
        run(
            [str(PYTHON), "helper_scripts/make_fixed_positions_dict.py", "--input_path", str(parsed), "--output_path", str(fixed_jsonl), "--chain_list", "A", "--position_list", " ".join(map(str, fixed))],
            env,
        )

    out_dir = raw_dir / "output"
    fasta_path = out_dir / "seqs" / "TEVd.fa"
    command = [
        str(PYTHON),
        "protein_mpnn_run.py",
        "--jsonl_path", str(parsed),
        "--chain_id_jsonl", str(assigned),
        "--fixed_positions_jsonl", str(fixed_jsonl),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(args.per_temperature),
        "--sampling_temp", "0.1 0.2 0.3",
        "--batch_size", str(args.batch_size),
        "--omit_AAs", "XC",
        # Current ProteinMPNN CLI spells the paper's explicit checkpoint as
        # model-name + weights directory rather than --checkpoint_path.
        "--model_name", "v_48_020",
        "--path_to_model_weights", str(MPNN / "vanilla_model_weights"),
        "--seed", str(args.seed),
        "--save_score", "1",
        "--save_probs", "1",
    ]
    manifest_path = ROOT / "manifests" / f"proteinmpnn_generation_{run_name}.json"
    write_json(manifest_path, {"command": command, "cwd": str(MPNN), "gpu": args.gpu, "status": "prepared"})
    if not fasta_path.exists():
        run(command, env)
    parsed_rows = parse_generated(fasta_path)
    all_rows: list[dict] = []
    for index, parsed_row in enumerate(parsed_rows):
        sequence = parsed_row.pop("sequence")
        reasons = validate_m3_sequence(sequence, forbid_c_at_mutable=True)
        all_rows.append(
            {
                "candidate_id": f"mpnn_{run_name}_{index:04d}",
                "method": "proteinmpnn",
                "seed": args.seed,
                "valid_for_paper_parity": not reasons,
                "rejection_reasons": "|".join(reasons),
                **parsed_row,
                **sequence_features(sequence),
                "raw_fasta": str(fasta_path),
            }
        )
    dedup: dict[str, dict] = {}
    for row in all_rows:
        if row["valid_for_paper_parity"] and row["sequence"] not in dedup:
            dedup[row["sequence"]] = row
    rows = list(dedup.values())
    table = ROOT / "candidates" / f"proteinmpnn_{run_name}_valid_unique.csv"
    fasta = ROOT / "candidates" / f"proteinmpnn_{run_name}_valid_unique.fasta"
    write_csv(table, rows)
    write_fasta(fasta, [(r["candidate_id"], r["sequence"]) for r in rows])
    summary = {
        "raw_samples": len(all_rows),
        "paper_parity_unique": len(rows),
        "expected_samples": 3 * args.per_temperature,
        "table": str(table),
        "fasta": str(fasta),
    }
    write_json(manifest_path, {"command": command, "cwd": str(MPNN), "gpu": args.gpu, "status": "completed", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
