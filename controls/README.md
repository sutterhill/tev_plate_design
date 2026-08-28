# Released m3 control characterization

This directory contains the fully characterized **25-row control set**:
TEVd plus 24 released m3 designs (8 active, 8 somewhat active, and 8
inactive/assay-floor controls).

## Canonical table

`released_controls_fresh_af2_characterized_25.csv` is the canonical app input.
It contains sequence, mutations, fresh AlphaFold2 structure/confidence and
RMSDs, Ridgey base-600M and five-member-ensemble stability/solubility, and
ProteinMPNN WT- and fresh-mutant-structure probabilities.  All absolute model
values also have signed absolute and percent differences from the freshly
folded TEVd reference where applicable.

## Matched structure protocol

The released designs were refolded because the author-supplied PDB files have
all-zero B-factor columns and therefore do not preserve per-residue pLDDT.
The fresh folds use the project protocol:

- ColabFold 1.6.2 / AlphaFold2 monomer
- model 3, one model, one seed (seed 0)
- six recycles
- query-swapped current TEVd MMseqs A3M
- no explicit structure template

TEVd uses the identically configured fold already present in the production
run. All 25 sequences pass mean pLDDT >87.5, and all pass the project gate of
mean pLDDT >85 plus global CA RMSD <2 A versus 1LVM.

Primary protected and active-site RMSDs are computed after a global 221-CA
Kabsch superposition to 1LVM. Independent subset-best-fit RMSDs are retained as
secondary columns. The m3 protected set comprises the 127 active-site plus
50%-conserved positions fixed in the paper; the active-site set has 38
positions.

## Model-score definitions

- ProteinMPNN probability is `exp(-mean NLL)` across 16 random decoding orders:
  a per-residue geometric-mean conditional probability, not a full sequence
  joint probability.
- Ridgey ensemble mean and sample SD are calculated across the five named 600M
  members. All five member values and their names are retained in columns and
  JSON fields.
- A paired Ridgey vote counts members for which the design value is strictly
  greater than that same member's freshly folded TEVd value.
- Ridgey solubility is a binary soluble-expression probability, not expected
  purified SEC-monomer yield.

## Experimental activity caveat

The paper did not release a numeric per-design activity-screen table. Exact
individual Michaelis-Menten kinetics are joined only for TEVd, hyperTEV56,
hyperTEV60, and hyperTEV89. Other activity tiers come from the Figure S7
row-major raw-trace inference described in `../m3_controls/NOTES.md`. No
per-design RFU/s values were invented.

## Provenance

- `fresh_source_mapping.json` maps every primary source and metric definition.
- `fresh_af2_structure_manifest.csv` maps app-ready structure links to raw AF2
  PDB/score files and A3Ms.
- `released_controls_characterized_25.csv` is the older paper-PDB provenance
  table. Its pLDDT fields are null by design because the released B-factors are
  zero.
- `raw/` preserves compressed Ridgey requests/responses for base 600M and the
  five-member ensemble.
- The reproducible scripts are `prepare_released_control_af2.py`,
  `score_controls_mpnn_fresh.py`, and
  `characterize_released_controls_fresh.py`.

## Validation

- 25 rows, 25 unique design IDs
- 25 sequence-matched fresh AF2 structures, each 221 residues
- 25 pLDDT arrays of length 221, cross-checked against PDB CA B-factors
- 25 Ridgey base-600M results
- 25 Ridgey five-member-ensemble results; member-derived means and sample SDs
  numerically rechecked
- 25 ProteinMPNN WT-structure and 25 fresh-own-structure 16-order scores
- activity tier counts: 1 parent, 8 active, 8 somewhat active, 8 floor/inactive
