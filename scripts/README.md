# TEV m3 candidate-generation pipeline

This directory reproduces the paper's TEV **m3** design and AF2 checks while adding Ridgey-generated candidates and mutant-structure Ridgey scoring. Everything is orchestrated and retained under `/opt/dlami/nvme/tev_plate_design` on `aws0`.

## Exact paper settings encoded

- Backbone and RMSD reference: TEVd, PDB 1LVM chain A, residues 1-221.
- m3 mask: catalytic/substrate-contact positions plus the paper's top 50% conserved positions fixed: 127 fixed and 94 mutable residues.
- ProteinMPNN: `v_48_020`, temperatures `0.1 0.2 0.3`, no X/C at mutable positions, no backbone noise at inference.
- AF2: monomer model 3, six recycles, one model/seed, parent-TEV MSA with the query row replaced by the candidate.
- AF2 gates: mean pLDDT >85.0 and globally aligned C-alpha RMSD to 1LVM <2.0 A.

The exact 2020 UniRef30 HHblits MSA used by the authors was not released. The pipeline freezes and hashes the current TEVd-seeded MMseqs A3M from the earlier reanalysis and applies the same query-row replacement protocol. It does **not** supply 1LVM as an AF2 structural template or initial guess because the paper does not state that it did so; 1LVM is used as the design backbone and RMSD reference.

Ridgey inverse folding has no request-level amino-acid omission field. Raw outputs are therefore preserved, then candidates containing C at any mutable m3 position are rejected to match ProteinMPNN's `--omit_AAs XC` behavior exactly.

## Stages

```bash
export TEV_PLATE_ROOT=/opt/dlami/nvme/tev_plate_design
cd "$TEV_PLATE_ROOT/scripts"

python3 00_initialize.py

# Service smoke tests; small and safe.
python3 01_sample_ridgey.py --smoke
python3 02_sample_proteinmpnn.py --smoke --gpu 6
python3 03_prepare_af2.py --run-name smoke
/opt/pytorch/bin/modal run 04_fold_af2_modal.py --run-name smoke
python3 05_characterize.py --run-name smoke --gpu 6 --mpnn-orders 2

# Recommended production pools.
python3 01_sample_ridgey.py                         # 2,048 raw: 2 temps x 2 seeds x 512
python3 02_sample_proteinmpnn.py --per-temperature 128 --gpu 6  # 384 raw, exact temp proportions
python3 03_prepare_af2.py --per-method 144
/opt/pytorch/bin/modal run 04_fold_af2_modal.py --run-name production  # up to 40 hosted A100s
python3 05_characterize.py --gpu 6 --mpnn-orders 16
python3 03_prepare_ridgey_extension.py              # all strict Ridgey not in first 144
/opt/pytorch/bin/modal run 04_fold_af2_modal.py --run-name ridgey_extension
python3 05_characterize.py --run-name ridgey_extension --gpu 6 --mpnn-orders 16
# When the first two Ridgey waves do not yield 36 mean-both improvements:
python3 01_sample_ridgey.py --run-name additional --temperatures 1.0 --seeds 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --num-per-job 512
python3 07_prescore_and_prepare_enriched.py --input-run additional --output-run ridgey_enriched --count 1000 --enrichment-pool 1000 --batch-size 30
/opt/pytorch/bin/modal run 04_fold_af2_modal.py --run-name ridgey_enriched
python3 05_characterize.py --run-name ridgey_enriched --gpu 6 --mpnn-orders 16
python3 06_select_final.py --count 36
```

If fewer than 36 Ridgey designs pass all strict gates, `06_select_final.py` fails loudly with the exact passing count. Generate an additional Ridgey seed shard, append/deduplicate the candidate table, and fold another diverse pre-fold shard rather than relaxing the requested stability/solubility-above-WT criteria.

## Expected throughput and batching

- Ridgey inverse fold: request 512 sequences/job. The first four jobs produced 372 strict unique candidates from 2,048 raw samples; the additional sixteen jobs produced 1,265 new strict uniques from 8,192 raw samples.
- ProteinMPNN: 128 sequences per each of 0.1/0.2/0.3 = 384 raw; local batch size 32.
- AF2: start with 144 diverse sequences/method, then fold all remaining first-wave Ridgey sequences and an ensemble-prescore-enriched set of 1,000 additional Ridgey candidates because solubility-above-WT is rare. The aws0 GPUs are reserved by unrelated training, so a task-specific Modal app fans out up to 40 hosted A100 containers while the aws0 driver preserves calls, raw tarballs, and extracted outputs on NVMe. Each input is its own frozen query-swapped A3M.
- Ridgey mutant-structure prediction: batches of 30 structures/request.
- ProteinMPNN likelihood: average 16 random decoding orders for both WT-backbone and own-AF2-backbone scores.

All API requests, call IDs, raw compressed responses, A3Ms, AF2 commands/logs/tar-free raw outputs, ProteinMPNN NPZ files, and derived CSVs are retained under `raw/`, `folds/`, `scores/`, and `manifests/`.
