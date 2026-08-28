#!/usr/bin/env python3
"""Prepare AF2 inputs for every strict Ridgey sequence not in the first pool."""

from __future__ import annotations

import json
import re

from common import ROOT, read_csv, read_jsonl, sha256, write_csv, write_jsonl


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def swap_query(parent_a3m: str, sequence: str, name: str) -> str:
    lines = parent_a3m.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(">"))
    query = next(i for i in range(header + 1, len(lines)) if lines[i] and not lines[i].startswith(">"))
    lines[header] = f">{name}"
    lines[query] = sequence
    return "\n".join(lines) + "\n"


def main() -> None:
    all_rows = read_csv(ROOT / "candidates" / "ridgey_production_valid_unique.csv")
    initial = read_jsonl(ROOT / "manifests" / "af2_production.jsonl")
    used = {row["sequence"] for row in initial if row["method"] == "ridgey"}
    remaining = [row for row in all_rows if row["sequence"] not in used]
    parent_path = ROOT / "inputs" / "tev_parent_current_mmseqs.a3m"
    parent = parent_path.read_text()
    a3m_dir = ROOT / "folds" / "ridgey_extension" / "a3m"
    a3m_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in remaining:
        candidate_id = safe_name(row["candidate_id"])
        a3m_path = a3m_dir / f"{candidate_id}.a3m"
        a3m_path.write_text(swap_query(parent, row["sequence"], candidate_id))
        manifest.append(
            {
                **row,
                "candidate_id": candidate_id,
                "n_mutations": int(row["n_mutations"]),
                "a3m_path": str(a3m_path),
                "a3m_sha256": sha256(a3m_path),
                "parent_a3m": str(parent_path),
                "parent_a3m_sha256": sha256(parent_path),
                "af2_model_type": "alphafold2",
                "af2_model_order": 3,
                "af2_recycles": 6,
                "explicit_structure_template": False,
                "prefold_selection_method": "all_remaining_strict_ridgey",
            }
        )
    write_jsonl(ROOT / "manifests" / "af2_ridgey_extension.jsonl", manifest)
    write_csv(ROOT / "candidates" / "prefold_ridgey_extension.csv", manifest)
    print(json.dumps({"all_strict_ridgey": len(all_rows), "initial_ridgey": len(used), "extension": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
