# TEV m3 redesign plate

Reproducible computational replication and extension of the TEV protease redesign experiment in Sumida *et al.*, ["Improving Protein Expression, Stability, and Function with ProteinMPNN"](https://pmc.ncbi.nlm.nih.gov/articles/PMC10811672/).

The final plate contains:

- TEVd, the S219D parent represented by PDB 1LVM chain A (221 residues)
- 24 released m3 designs spanning the paper's activity screen, including hyperTEV56, hyperTEV60, and hyperTEV89
- 36 Ridgey 600M inverse-fold designs
- 36 ProteinMPNN control designs

Both generated cohorts use the paper's m3 constraint set: active-site positions and the top 50% MSA-conserved positions are fixed (127 fixed, 94 mutable), with no novel cysteine allowed. ProteinMPNN uses checkpoint `v_48_020` and temperatures 0.1, 0.2, and 0.3. Ridgey conditions on the same 1LVM backbone. The loop-robust Ridgey cleanup may restore a native TEVd cysteine when reverting a rejected substitution; it never introduces cysteine at a non-cysteine WT position.

Candidates are folded with AlphaFold2 model 3 and six recycles. Following the paper, each input is a frozen parent-TEV MSA whose query row is replaced by the candidate sequence. The paper's 2020 HHblits/UniRef30 alignment was not released, so this replication uses the frozen current MMseqs parent alignment in `inputs/tev_parent_current_mmseqs.a3m`. No 1LVM structural template or initial guess is supplied to AlphaFold2. The paper's gates are mean pLDDT >85 and global C-alpha RMSD to 1LVM <2.0 A.

The Ridgey stability selector is loop robust. HELIX/SHEET records from the original 1LVM chain A define 124 structured residues; the other 97 residues are treated as loop/coil. Every proposed structured-region substitution is scored with the five-member Ridgey DDG matrix on the common TEVd/1LVM structure and retained only when its improving-is-positive mean is above zero and at least four of five members agree. Rejected structured substitutions are reverted to WT. Loop/coil substitutions are retained but receive no stability credit. The global MGnify proteolysis-stability output remains in the table for reference and is not used for gating or ranking.

The edited candidates are refolded from scratch. Eligibility requires the paper's AF2 gate, positive loop-excluded core-DDG mean-minus-ensemble-SD, no retained structured mutation that failed the 4/5 rule, and positive fresh own-structure solubility paired-delta mean-minus-ensemble-SD versus the matched TEVd parent. Of 240 refolded candidates, 134 passed. Balanced DDG/solubility ranking followed by deterministic max-min sequence diversity selected 36 designs: 32 have 5/5 solubility votes and four have 4/5. The selected cohort averages 35 mutations: 10.3 scored structured substitutions and 24.7 loop/coil substitutions excluded from stability scoring. ProteinMPNN likelihoods are reported as geometric-mean per-residue probabilities, `exp(-mean NLL)`, averaged over 16 random decoding orders; these are not raw joint probabilities.

## Final artifacts

- [`outputs/tev_plate_97.csv`](outputs/tev_plate_97.csv): fully characterized 97-row plate table
- [`outputs/tev_plate_97.fasta`](outputs/tev_plate_97.fasta): plate-ordered sequences
- [`outputs/tev_plate_mutation_matrix.png`](outputs/tev_plate_mutation_matrix.png): black/blue/red mutation map
- [`outputs/plate.json`](outputs/plate.json): PlayGod application payload
- [`outputs/structures/`](outputs/structures/): 97 AlphaFold2 PDB files
- [`outputs/VALIDATION.json`](outputs/VALIDATION.json) and [`outputs/SHA256SUMS`](outputs/SHA256SUMS): invariant checks and file hashes

All raw requests, call IDs, candidate rejections, sampled sequences, frozen A3Ms, AlphaFold2 tarballs and logs, Ridgey responses, ProteinMPNN NPZ outputs, selected tables, plots, and checksums are retained in the project archive on aws0 NVMe. See [scripts/README.md](scripts/README.md) for the pipeline commands and exact settings.

The interactive result is published as the [TEV Redesign Plate](https://playgod.bio/tev-redesign-plate) experiment on PlayGod.
