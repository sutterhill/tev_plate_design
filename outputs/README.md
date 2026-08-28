# Final TEV redesign plate

This directory is the portable 97-protein result bundle:

- 1 TEVd parent
- 24 released m3 controls spanning the paper's activity trace
- 36 Ridgey inverse-fold designs
- 36 paper-recipe ProteinMPNN controls

`tev_plate_97.csv` contains sequences, mutations, AlphaFold2 pLDDT and three RMSDs, loop-excluded Ridgey core-DDG scores, Ridgey base-model and five-member ensemble global stability/solubility predictions, exact member values and votes, and 16-order ProteinMPNN `P(seq | WT structure)` and `P(seq | mutant structure)` scores. `plate.json` is the normalized mini-app payload. `structures/` contains the corresponding 97 fresh AlphaFold2 structures.

For Ridgey selection, HELIX/SHEET substitutions must have positive mean single-mutation DDG improvement with at least 4/5 ensemble members supporting them; rejected substitutions are reverted. Loop/coil substitutions are excluded from the stability score. After fresh AF2, every selected Ridgey design has positive core-DDG mean-minus-SD and solubility paired-delta mean-minus-SD versus the matched TEVd parent. The final cohort contains 32 designs with 5/5 solubility votes and four with 4/5. Global proteolysis stability is reported only and was not used for selection.

Run `sha256sum -c SHA256SUMS` from this directory to verify the bundle.
