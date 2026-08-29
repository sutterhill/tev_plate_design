#!/usr/bin/env python3
"""Compute true sequence-only masked pseudo-perplexity for TEV designs.

Every residue is masked once and scored conditional on all other residues.
The Ridgey encoder receives ``structure=None`` on every forward pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from ridgey.serving import RidgeyPredictor, normalize_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--design-metadata", type=Path, required=True)
    parser.add_argument("--tevd-fasta", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_fasta(path: Path) -> str:
    return normalize_sequence(
        "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith(">"))
    )


def file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    metadata = pd.read_csv(args.design_metadata)
    records = [
        {
            "design_id": "TEVd",
            "method_code": "parent",
            "identity_to_TEVD_percent": 100.0,
            "sequence": read_fasta(args.tevd_fasta),
        }
    ]
    records.extend(
        metadata[
            ["design_id", "method_code", "identity_to_TEVD_percent", "sequence"]
        ].to_dict(orient="records")
    )
    sequences = [normalize_sequence(str(record["sequence"])) for record in records]
    lengths = {len(sequence) for sequence in sequences}
    if lengths != {221}:
        raise ValueError(f"expected 221-residue TEV sequences, observed lengths {lengths}")

    started = time.time()
    predictor = RidgeyPredictor(
        checkpoint_path=args.checkpoint,
        artifact_dir=args.artifact_dir,
        device="cuda:0",
        model_name=args.model_label,
    )
    model = predictor.model.encoder
    tokenizer = predictor.tokenizer
    device = predictor.device
    mask_id = int(tokenizer.mask_token_id)

    token_rows = []
    for sequence in sequences:
        residue_ids = tokenizer.encode(sequence, add_special_tokens=False)
        token_rows.append([int(tokenizer.bos_token_id), *map(int, residue_ids)])
    base_tokens = torch.tensor(token_rows, dtype=torch.long, device=device)
    n_sequences, token_length = base_tokens.shape
    residue_length = token_length - 1
    jobs = [(sequence_index, position) for sequence_index in range(n_sequences) for position in range(residue_length)]
    per_residue_logp = torch.empty((n_sequences, residue_length), dtype=torch.float32)

    torch.set_float32_matmul_precision("high")
    with torch.inference_mode():
        for start in range(0, len(jobs), args.batch_size):
            chunk = jobs[start : start + args.batch_size]
            sequence_indices = torch.tensor(
                [item[0] for item in chunk], dtype=torch.long, device=device
            )
            residue_positions = torch.tensor(
                [item[1] for item in chunk], dtype=torch.long, device=device
            )
            batch = base_tokens.index_select(0, sequence_indices).clone()
            rows = torch.arange(len(chunk), device=device)
            token_positions = residue_positions + 1
            targets = batch[rows, token_positions].clone()
            batch[rows, token_positions] = mask_id
            sequence_id = torch.ones_like(batch, dtype=torch.bool)
            position_ids = torch.arange(token_length, device=device).expand(len(chunk), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    sequence_tokens=batch,
                    sequence_id=sequence_id,
                    position_ids=position_ids,
                    structure=None,
                    return_logits=True,
                )
            selected_logits = output["logits"][rows, token_positions].float()
            values = selected_logits.log_softmax(dim=-1)[rows, targets].cpu()
            per_residue_logp[
                sequence_indices.cpu(), residue_positions.cpu()
            ] = values
            if start == 0 or (start // args.batch_size) % 100 == 0:
                completed = min(start + len(chunk), len(jobs))
                print(
                    f"{args.model_label}: {completed}/{len(jobs)} masked residues",
                    flush=True,
                )

    output_records = []
    for index, record in enumerate(records):
        values = per_residue_logp[index]
        mean_log_probability = float(values.mean())
        output_records.append(
            {
                **record,
                "mean_masked_log_probability": mean_log_probability,
                "pseudo_perplexity": float(math.exp(-mean_log_probability)),
                "pseudo_nll_sum": float(-values.sum()),
                "per_residue_log_probability": values.tolist(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": args.model_label,
        "ridgey_reported_model_name": predictor.model_name,
        "definition": "exp(-mean_i log p(x_i | x_without_i, structure=None))",
        "structure_argument": None,
        "masking": "one residue per sequence row; every residue scored exactly once",
        "checkpoint": file_fingerprint(args.checkpoint),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "batch_size": args.batch_size,
        "n_sequences": n_sequences,
        "sequence_length": residue_length,
        "elapsed_seconds": time.time() - started,
        "records": output_records,
    }
    with gzip.open(args.output, "wt") as handle:
        json.dump(payload, handle)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": args.model_label,
                "n_sequences": n_sequences,
                "elapsed_seconds": payload["elapsed_seconds"],
                "tevd_pseudo_perplexity": output_records[0]["pseudo_perplexity"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
