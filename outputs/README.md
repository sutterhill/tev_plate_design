# Final TEV redesign plate

This directory is the portable 97-protein result bundle:

- 1 TEVd parent
- 24 released m3 controls spanning the paper's activity trace
- 36 Ridgey inverse-fold designs
- 36 paper-recipe ProteinMPNN controls

`tev_plate_97.csv` contains sequences, mutations, AlphaFold2 pLDDT and three RMSDs, Ridgey base-model and five-member ensemble stability/solubility predictions, exact member values and paired votes, and 16-order ProteinMPNN `P(seq | WT structure)` and `P(seq | mutant structure)` scores. `plate.json` is the normalized mini-app payload. `structures/` contains the corresponding 97 fresh AlphaFold2 structures.

The Ridgey cohort uses a hard eligibility rule of AF2 pass plus ensemble-mean stability and solubility both strictly above TEVd. Consensus is a ranking criterion: the final cohort contains 9 designs at 5/5 paired-member improvement, 24 at 4/5, and 3 at 3/5.

Run `sha256sum -c SHA256SUMS` from this directory to verify the bundle.
