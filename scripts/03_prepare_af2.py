#!/usr/bin/env python3
"""Choose diverse pre-fold pools and create paper-style query-swapped A3Ms."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import ROOT, greedy_maximin, read_csv, read_fasta, sha256, write_csv, write_jsonl


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def swap_query(parent_a3m: str, sequence: str, name: str) -> str:
    lines = parent_a3m.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith(">"))
    query_index = next(i for i in range(header_index + 1, len(lines)) if lines[i] and not lines[i].startswith(">"))
    lines[header_index] = f">{name}"
    lines[query_index] = sequence
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", choices=["smoke", "production"], default="production")
    parser.add_argument("--per-method", type=int, default=144)
    parser.add_argument("--ridgey-table")
    parser.add_argument("--proteinmpnn-table")
    args = parser.parse_args()
    if args.run_name == "smoke":
        args.per_method = min(args.per_method, 1)

    ridgey_table = Path(args.ridgey_table) if args.ridgey_table else ROOT / "candidates" / f"ridgey_{args.run_name}_valid_unique.csv"
    mpnn_table = Path(args.proteinmpnn_table) if args.proteinmpnn_table else ROOT / "candidates" / f"proteinmpnn_{args.run_name}_valid_unique.csv"
    tables = {"ridgey": ridgey_table, "proteinmpnn": mpnn_table}
    selected: list[dict] = []
    already_seen: set[str] = set()
    for method, path in tables.items():
        rows = read_csv(path) if path.exists() and path.stat().st_size else []
        for row in rows:
            row["n_mutations"] = int(row["n_mutations"])
        rows = [r for r in rows if r["sequence"] not in already_seen]
        seed_key = "mpnn_generation_nll" if method == "proteinmpnn" and rows else None
        chosen = greedy_maximin(rows, min(args.per_method, len(rows)), seed_key=seed_key) if rows else []
        for row in chosen:
            row["prefold_selection_method"] = "greedy_sequence_maximin"
        selected.extend(chosen)
        already_seen.update(r["sequence"] for r in chosen)

    wt = read_fasta(ROOT / "inputs" / "TEVd.fasta")[0][1]
    selected.insert(
        0,
        {
            "candidate_id": "TEVd",
            "method": "parent",
            "sequence": wt,
            "n_mutations": 0,
            "identity_to_tevd_percent": 100.0,
            "mutations": "",
            "prefold_selection_method": "required_parent_control",
        },
    )
    parent_path = ROOT / "inputs" / "tev_parent_current_mmseqs.a3m"
    parent_text = parent_path.read_text()
    a3m_dir = ROOT / "folds" / args.run_name / "a3m"
    a3m_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for row in selected:
        candidate_id = safe_name(row["candidate_id"])
        a3m_path = a3m_dir / f"{candidate_id}.a3m"
        a3m_path.write_text(swap_query(parent_text, row["sequence"], candidate_id))
        manifest.append(
            {
                **row,
                "candidate_id": candidate_id,
                "a3m_path": str(a3m_path),
                "a3m_sha256": sha256(a3m_path),
                "parent_a3m": str(parent_path),
                "parent_a3m_sha256": sha256(parent_path),
                "af2_model_type": "alphafold2",
                "af2_model_order": 3,
                "af2_recycles": 6,
                "explicit_structure_template": False,
            }
        )
    write_jsonl(ROOT / "manifests" / f"af2_{args.run_name}.jsonl", manifest)
    write_csv(ROOT / "candidates" / f"prefold_{args.run_name}.csv", manifest)
    print(json.dumps({"run_name": args.run_name, "total": len(manifest), "by_method": {m: sum(r["method"] == m for r in manifest) for m in ("parent", "ridgey", "proteinmpnn")}}, indent=2))


if __name__ == "__main__":
    main()

