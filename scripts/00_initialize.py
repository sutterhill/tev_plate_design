#!/usr/bin/env python3
"""Initialize immutable inputs and record the paper-matched TEV m3 protocol."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from common import REANALYSIS, ROOT, read_fasta, sha256, write_json


def main() -> None:
    inputs = ROOT / "inputs"
    for directory in (
        inputs,
        ROOT / "raw",
        ROOT / "candidates",
        ROOT / "folds",
        ROOT / "scores",
        ROOT / "selected",
        ROOT / "logs",
        ROOT / "manifests",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    copies = {
        REANALYSIS / "dataset" / "TEVd.fasta": inputs / "TEVd.fasta",
        REANALYSIS / "dataset" / "TEVd_1LVM_A_1-221.pdb": inputs / "TEVd_1LVM_A_1-221.pdb",
        REANALYSIS / "dataset" / "fixed_positions.json": inputs / "fixed_positions.json",
        REANALYSIS / "results" / "evolution" / "tev_ev_wide.a3m": inputs / "tev_parent_current_mmseqs.a3m",
    }
    for source, target in copies.items():
        if not source.exists():
            raise FileNotFoundError(source)
        if not target.exists():
            shutil.copy2(source, target)

    wt = read_fasta(inputs / "TEVd.fasta")[0][1]
    fixed_data = json.loads((inputs / "fixed_positions.json").read_text())
    fixed = fixed_data["active_site_plus_50pct_conserved"]
    fixed_set = set(fixed)
    mutable = [i for i in range(1, len(wt) + 1) if i not in fixed_set]
    if len(wt) != 221 or len(fixed) != 127 or len(mutable) != 94:
        raise RuntimeError(
            f"unexpected TEV m3 dimensions: L={len(wt)} fixed={len(fixed)} mutable={len(mutable)}"
        )
    aligned = "".join(aa if i in fixed_set else "-" for i, aa in enumerate(wt, 1))
    (inputs / "m3_aligned_sequence.txt").write_text(aligned + "\n")
    write_json(
        inputs / "m3_mask.json",
        {
            "name": "paper_m3_active_site_plus_50pct_conserved",
            "indexing": "1-indexed",
            "sequence_length": len(wt),
            "fixed_positions": fixed,
            "mutable_positions": mutable,
            "n_fixed": len(fixed),
            "n_mutable": len(mutable),
            "aligned_sequence": aligned,
        },
    )

    a3m = inputs / "tev_parent_current_mmseqs.a3m"
    a3m_records = read_fasta(a3m)
    if not a3m_records or a3m_records[0][1] != wt:
        raise RuntimeError("parent MSA query row does not exactly match TEVd")

    write_json(
        ROOT / "PROTOCOL.json",
        {
            "paper": {
                "proteinmpnn_backbone": "PDB 1LVM chain A, TEVd residues 1-221",
                "proteinmpnn_checkpoint": "v_48_020 (0.2 A training-backbone noise)",
                "proteinmpnn_temperatures": [0.1, 0.2, 0.3],
                "proteinmpnn_omit_amino_acids": ["X", "C"],
                "m3_fixed_scheme": "active site plus top 50% MSA-conserved positions",
                "af2_model": 3,
                "af2_recycles": 6,
                "af2_msa_protocol": "replace parent TEV query row with each design sequence",
                "af2_pass_mean_plddt": ">85.0",
                "af2_pass_ca_rmsd_angstrom": "<2.0",
            },
            "replication": {
                "af2_runtime": "ghcr.io/sokrypton/colabfold:1.6.2-cuda12 on aws0",
                "af2_model_type": "alphafold2",
                "af2_model_order": [3],
                "msa_substitute": "current MMseqs TEVd MSA; paper's 2020 UniRef30 HHblits MSA was not released",
                "explicit_af2_structure_template": False,
                "rationale": "Paper specifies query-swapped parent MSA but does not specify supplying 1LVM as an AF2 template. 1LVM is the design backbone and RMSD reference.",
                "ridgey_model": "600m",
                "ridgey_no_c_parity_filter": "reject C at every mutable m3 position",
            },
            "input_sha256": {target.name: sha256(target) for target in copies.values()},
        },
    )
    print(f"initialized {ROOT}")
    print(f"TEVd length={len(wt)} m3 fixed={len(fixed)} mutable={len(mutable)} MSA rows={len(a3m_records)}")


if __name__ == "__main__":
    main()

