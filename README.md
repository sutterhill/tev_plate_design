# TEV m3 redesign plate

Reproducible computational replication and extension of the TEV protease redesign experiment in Sumida *et al.*, ["Improving Protein Expression, Stability, and Function with ProteinMPNN"](https://pmc.ncbi.nlm.nih.gov/articles/PMC10811672/).

The final plate contains:

- TEVd, the S219D parent represented by PDB 1LVM chain A (221 residues)
- 24 released m3 designs spanning the paper's activity screen, including hyperTEV56, hyperTEV60, and hyperTEV89
- 36 Ridgey 600M inverse-fold designs
- 36 ProteinMPNN control designs

Both generated cohorts use the paper's m3 constraint set: active-site positions and the top 50% MSA-conserved positions are fixed (127 fixed, 94 mutable), with no new cysteine allowed. ProteinMPNN uses checkpoint `v_48_020` and temperatures 0.1, 0.2, and 0.3. Ridgey conditions on the same 1LVM backbone.

Candidates are folded with AlphaFold2 model 3 and six recycles. Following the paper, each input is a frozen parent-TEV MSA whose query row is replaced by the candidate sequence. The paper's 2020 HHblits/UniRef30 alignment was not released, so this replication uses the frozen current MMseqs parent alignment in `inputs/tev_parent_current_mmseqs.a3m`. No 1LVM structural template or initial guess is supplied to AlphaFold2. The paper's gates are mean pLDDT >85 and global C-alpha RMSD to 1LVM <2.0 A.

Ridgey candidates additionally must pass the AF2 gate and score strictly above the matched AF2-folded TEVd parent for both five-member Ridgey 600M ensemble-mean stability and ensemble-mean solubility on their own predicted structures. Paired-member consensus is used for ranking (5/5 ahead of 4/5, then 3/5, and so on) rather than as an extra hard threshold; exact member predictions and votes are retained. Final designs are chosen by deterministic sequence-diversity selection within the consensus-ranked quality pool. ProteinMPNN likelihoods are reported as geometric-mean per-residue probabilities, `exp(-mean NLL)`, averaged over 16 random decoding orders; these are not raw joint probabilities.

All raw requests, call IDs, candidate rejections, sampled sequences, frozen A3Ms, AlphaFold2 tarballs and logs, Ridgey responses, ProteinMPNN NPZ outputs, selected tables, plots, and checksums are retained in the project archive on aws0 NVMe. See [scripts/README.md](scripts/README.md) for the pipeline commands and exact settings.

The interactive result is published as the [TEV Redesign Plate](https://playgod.bio/tev-redesign-plate) experiment on PlayGod.
