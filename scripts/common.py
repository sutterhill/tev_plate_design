#!/usr/bin/env python3
"""Shared helpers for the TEV m3 plate-design pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


ROOT = Path(os.environ.get("TEV_PLATE_ROOT", "/opt/dlami/nvme/tev_plate_design"))
REANALYSIS = Path(
    os.environ.get("TEV_REANALYSIS_ROOT", "/opt/dlami/nvme/tev_reanalysis")
)
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks).upper()))
            name = line[1:].split()[0]
            chunks = []
        else:
            if name is None:
                raise ValueError(f"sequence before FASTA header in {path}")
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks).upper()))
    return records


def write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol() -> tuple[str, set[int], list[int]]:
    wt = read_fasta(ROOT / "inputs" / "TEVd.fasta")[0][1]
    fixed = set(
        json.loads((ROOT / "inputs" / "fixed_positions.json").read_text())[
            "active_site_plus_50pct_conserved"
        ]
    )
    mutable = [i for i in range(len(wt)) if i + 1 not in fixed]
    return wt, fixed, mutable


def validate_m3_sequence(sequence: str, forbid_c_at_mutable: bool = True) -> list[str]:
    wt, fixed, mutable = load_protocol()
    reasons: list[str] = []
    if len(sequence) != len(wt):
        return [f"length_{len(sequence)}_not_{len(wt)}"]
    bad = sorted(set(sequence) - CANONICAL)
    if bad:
        reasons.append("noncanonical:" + "".join(bad))
    changed_fixed = [i for i in fixed if sequence[i - 1] != wt[i - 1]]
    if changed_fixed:
        reasons.append("changed_fixed:" + ";".join(map(str, changed_fixed)))
    if forbid_c_at_mutable:
        c_positions = [i + 1 for i in mutable if sequence[i] == "C"]
        if c_positions:
            reasons.append("C_at_mutable:" + ";".join(map(str, c_positions)))
    return reasons


def sequence_features(sequence: str) -> dict:
    wt, _, _ = load_protocol()
    mutations = [f"{a}{i}{b}" for i, (a, b) in enumerate(zip(wt, sequence), 1) if a != b]
    return {
        "sequence": sequence,
        "length": len(sequence),
        "n_mutations": len(mutations),
        "identity_to_tevd_percent": 100.0 * (len(wt) - len(mutations)) / len(wt),
        "mutations": ";".join(mutations),
    }


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length sequences")
    return sum(x != y for x, y in zip(a, b))


def greedy_maximin(rows: list[dict], count: int, seed_key: str | None = None) -> list[dict]:
    """Greedy sequence max-min selection with deterministic tie breaking."""
    if count >= len(rows):
        return list(rows)
    remaining = sorted(rows, key=lambda r: r["candidate_id"])
    if seed_key and seed_key in remaining[0]:
        first = min(remaining, key=lambda r: (float(r[seed_key]), r["candidate_id"]))
    else:
        wt, _, _ = load_protocol()
        target = 60
        first = min(
            remaining,
            key=lambda r: (abs(hamming(wt, r["sequence"]) - target), r["candidate_id"]),
        )
    selected = [first]
    remaining.remove(first)
    min_dist = {r["candidate_id"]: hamming(r["sequence"], first["sequence"]) for r in remaining}
    while remaining and len(selected) < count:
        nxt = max(
            remaining,
            key=lambda r: (min_dist[r["candidate_id"]], -abs(int(r["n_mutations"]) - 60), r["candidate_id"]),
        )
        selected.append(nxt)
        remaining.remove(nxt)
        for row in remaining:
            ident = row["candidate_id"]
            min_dist[ident] = min(min_dist[ident], hamming(row["sequence"], nxt["sequence"]))
    return selected

