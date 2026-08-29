# TEV generation → validation plan

## Recommendation

Use the paper's **m3 functional constraint regime** as the common foundation, but do not optimize or select on any single model score. The next library should contain separate Ridgey inverse-fold, Ridgey multi-objective, ProteinMPNN, and lead-centered arms; every arm should be filtered through the same MSA, structure, function-preservation, and uncertainty gates. Keep every accepted sequence intact—if a candidate fails, reject the whole candidate rather than reverting individual mutations.

The key change from the current plate is **MSA-gated Ridgey generation**. The current Ridgey designs generally look soluble, stable, and enzyme-like to Ridgey, but only 1/36 clears the active-m3-calibrated MSA/function gate. The corresponding rate is 17/36 for the ProteinMPNN controls. This is not evidence that ProteinMPNN directly predicts activity; it says the Ridgey sampling distribution needs an explicit evolutionary plausibility constraint.

## What the experiment actually establishes

The paper generated 144 TEV designs using four conservation regimes. It reports 134/144 soluble monomeric designs, 129/144 with higher soluble recovery than TEVd, and mean recovered yield of 20.1 mg/L versus 1.0 mg/L for TEVd. Sixty-four designs had detectable peptide turnover. The m3 arm—active-site residues plus the top 50% most conserved residues fixed—produced the highest activity. HyperTEV56, hyperTEV60, and hyperTEV89 improved catalytic efficiency 20×, 26×, and 6.2×, respectively; hyperTEV60 also reached an approximate 84 °C melting temperature and retained 90% activity after 4 h at 30 °C. Source: [Sumida et al., JACS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10811672/).

The expression measurement is recovered monomeric protein after expression, IMAC, and SEC, reported per culture volume. It conflates total expression, soluble folding, purification recovery, and monomericity. The next experiment should measure those components separately.

## New analysis results

### Activity cannot be reduced to one score

The full m3 arm contains 48 designs: 10 active, 17 somewhat active, and 21 at the assay floor under the Figure S7c trace-order interpretation. Except for the three named kinetic leads, these labels are inferred from the row-major trace layout and are not publication-provided numeric per-design measurements.

Within these 48 designs, no positive scalar metric has a convincing association with activity:

| Metric | Spearman vs inferred m3 trace | Active-vs-rest AUC | Interpretation |
| --- | ---: | ---: | --- |
| ProteinMPNN p(seq \| WT) improvement | +0.189 | 0.497 | Weak positive rank trend, no useful classifier |
| Ridgey structured-only ddG sum | +0.133 | 0.584 | Suitable as a risk guardrail, not an activity predictor |
| MSA independent score | +0.096 | 0.597 | Useful anti-drift constraint |
| Ridgey ensemble stability mean Δ | -0.015 | 0.450 | Stability is not an activity ranker within m3 |
| Ridgey exact EC 3.4.22.44 | -0.012 | 0.321 | Use only to reject function collapse |
| Ridgey p(seq \| mutant structure) | **-0.346** | 0.211 | Significant inverse trend (permutation p = 0.017); do not optimize this for activity |

This agrees with the broad experimental lesson: m3 constraints make activity possible, but the remaining activity variation is not captured by a generic folding or stability score.

### One activity-associated mutation hypothesis

K45A occurs in 7/10 active m3 designs and 1/21 floor designs (frequency difference +0.652, Fisher exact p = 0.00033, BH FDR = 0.033). K45L shows the opposite pattern (3/10 active versus 15/21 floor), although it does not survive multiple-testing correction. K45 is adjacent to catalytic H46, so this is biologically plausible but still observational.

Treat K45 as a controlled hypothesis:

- half of each de novo generation arm should force **K45A**;
- the matched half should retain **K45**;
- do not use K45L in the primary library;
- analyze the K45 pair as a prespecified factor in the lab.

### Ridgey EC attribution suggests a small extra protection set

Integrated gradients were run with Ridgey 600M and 6B for stability, solubility, and EC 3.4.22.44 on TEVd and hyperTEV56/60/89. Seven m3-mutable positions repeatedly support the TEV EC score across models and scaffolds:

- retain the TEVd/lead-consensus residue at **S15, P39, R49, M87, E102, and V164** in the primary exploit arms;
- at position **155**, allow the observed L/M pair rather than forcing one identity;
- leave 25% of the exploration arm unconstrained at these positions to test whether the attribution rule is too conservative.

Integrated gradients explain current model scores; they do not establish causal activity effects.

### Loop-aware ddG result

The Ridgey ddG feature was split using contiguous 1LVM helix/strand-like backbone runs. Loop positions are excluded from the individual-mutation veto. Structured-only ddG is mildly more informative than the unqualified stability quantities, while loop-only ddG provides little active-vs-rest discrimination. This supports using ddG to reject severe structured-region risks without penalizing loop substitutions or modifying accepted sequences.

### What this says about the existing plate

Against floors calibrated from active m3 designs:

- all 36 current Ridgey designs pass the solubility floor and 35/36 pass the exact-EC floor;
- only **1/36** Ridgey designs passes the MSA independent-site floor;
- Ridgey-35 is the only current Ridgey design clearing the activity-preservation gate, but it fails the conservative property floor;
- **0/36** Ridgey designs and **17/36** ProteinMPNN controls pass the strict combined activity-preservation/property gate;
- the old plate remains useful as an exploratory head-to-head test and was not mutation-reverted.

The Ridgey cohort's median MSA score is -202.9, compared with an active-m3 10th-percentile floor of -152.6. ProteinMPNN's median is -153.1. The next Ridgey pool should therefore be sampled much more broadly and rejected early by the MSA filter.

## Recommended generation funnel

### Common sequence rules

Apply these to every arm:

1. Start from TEVd/1LVM and preserve the original 127 m3-fixed positions: active-site residues plus the top 50% conserved positions.
2. Preserve S219D, length 221, and all catalytic/substrate-contact residues. Do not introduce new cysteine.
3. Target 68–76% identity to TEVd for the main library (roughly 53–71 substitutions). A small exploration tail may span 65–80% identity.
4. Apply the EC-support residues above in the exploit arms and the matched K45A/K45 split.
5. Reject whole sequences that fail. Do not post hoc revert individual mutations.

### Generation arms

| Arm | Raw generation | Final lab designs | Purpose |
| --- | ---: | ---: | --- |
| Ridgey inverse fold on 1LVM | 8,192 sequences across temperatures 0.6, 0.8, 1.0, 1.2 | 24 | Test Ridgey's native backbone prior after explicit MSA gating |
| Ridgey multi-objective gradient design | 64 seeds × 4 weight schedules = 256 endpoints | 20 | Jointly improve stability, solubility, EC preservation, LM naturalness, and MSA plausibility |
| ProteinMPNN paper-recipe control | 8,192 sequences at 0.1, 0.2, 0.3 | 16 | Direct methodological control under identical constraints and filters |
| Lead-centered Ridgey redesign | 4,096 samples split across hyperTEV56/60/89 structures | 16 | Lower-risk exploitation of experimentally active high-mutation scaffolds |
| Uncertainty/exploration | Pareto-near candidates with high ensemble disagreement | 8 | Calibrate model uncertainty and avoid a purely exploitative library |

This yields **84 new designs**. Use the remaining 12 wells of each primary assay plate for distributed controls.

Generate from the experimental 1LVM backbone for the de novo arms, then fold every surviving sequence and score the exact `{mutant sequence, mutant structure}` pair. WT-backbone inverse-fold likelihood is a prior; mutant-backbone likelihood is a structural plausibility check, not an activity objective.

### Cheap pre-fold filters

Use active-m3 empirical floors rather than optimizing toward WT identity:

- MSA independent score ≥ **-152.6**;
- MSA Potts score ≥ **-198.1**;
- no severe structured-region Ridgey mutation with improvement score < **-1.0**;
- structured-region summed ddG improvement ≥ **-1.35**;
- ProteinMPNN p(seq \| WT) is a soft quality rank, not a hard gate;
- deduplicate and enforce the identity window before folding.

Because only about 3% of the current Ridgey designs clear the MSA floor, oversampling Ridgey by thousands is intentional.

### Structural filters

Fold approximately 600 prefiltered candidates, using 50 concurrent jobs:

1. Paper-parity AlphaFold2: parent-MSA query replacement, model 3, six recycles, no explicit 1LVM template.
2. An orthogonal MSA-free monomer prediction to make sure the parent MSA is not carrying an implausible sequence.
3. Hard gates: AF2 mean pLDDT ≥ 87.5 and global Cα RMSD to 1LVM < 2.0 Å.
4. Retain active-site and m3-protected RMSD only if no worse than the 90th percentile of the experimentally active m3 controls.
5. Reject backbone disagreements concentrated around H46, D81, C151, or the substrate-binding shell.

### Ridgey/function filters after folding

Use matched TEVd predictions under the same fold protocol:

- Ridgey ensemble solubility LCB ≥ **0.926**; do not require solubility above TEVd because TEVd is near the model ceiling and the active leads can score slightly below it.
- Ridgey paired-member stability LCB Δ > 0; rank toward the active-m3 10th percentile of **+0.185**.
- Ridgey 6B stability Δ > 0; rank toward the active-m3 10th percentile of **+1.81**.
- Ridgey exact EC 3.4.22.44 probability ≥ **9.7 × 10⁻⁵** as an anti-collapse gate only.
- No severe structured-region ddG mutation; ignore loop ddG in this veto.
- Preserve all five ensemble-member values and rank by mean and one-SD LCB.
- Do not reward high Ridgey p(seq \| mutant structure) beyond a basic plausibility floor.

### Final computational selection

Do not collapse the objectives into a single fitted “activity score.” Select Pareto fronts within each generation arm over:

- ensemble stability LCB and 6B stability;
- ensemble solubility LCB;
- MSA independent and Potts scores;
- structured-only ddG risk;
- exact EC anti-collapse score;
- ProteinMPNN/Ridgey WT-backbone plausibility;
- AF2 and MSA-free structural confidence.

Then use deterministic max-min Hamming selection with a minimum pairwise distance target of 12 substitutions. Enforce arm, K45, generation-temperature, and lead-scaffold quotas. Reserve eight designs with high ensemble disagreement for active-learning value.

## Recommended lab plate

### Controls distributed across the assay plate

- TEVd: 3 wells
- hyperTEV56, hyperTEV60, hyperTEV89: 2 wells each
- two inactive m3 controls (for example hyperTEV64 and hyperTEV79): 1 well each
- catalytically dead C151A control: 1 well

These are assay replicates; duplicate genes do not need to be synthesized twice.

### Phase 1: separate expression from solubility

Express every construct in the same vector, tag context, E. coli BL21(DE3) background, and autoinduction regime used for the paper. Apply one fixed codon-design policy so nucleotide-level optimization does not become an uncontrolled arm effect.

For three independent culture/induction replicates, measure:

1. OD-normalized total cellular TEV;
2. OD-normalized soluble TEV in clarified lysate;
3. soluble fraction = soluble / total;
4. IMAC recovery in mg/L culture;
5. SEC monomer fraction and monomeric yield in mg/L.

This deconvolves the paper's combined “expression” phenotype.

### Phase 2: high-throughput activity

Run the paper-comparable peptide assay with purified enzyme: 500 nM TEV, 10 µM Ac-ENLYFQ-coumarin substrate, pH 8, 2 mM DTT, room temperature. Fit the initial linear window and retain the full progress curve. Include substrate-only, enzyme-only, and C151A blanks.

Report two primary activity quantities:

- **specific activity**, normalized to measured soluble/purified enzyme concentration;
- **volumetric activity**, calculated per culture equivalent, which captures expression × soluble recovery × catalytic rate.

Use three technical wells per purification and at least three independent expression/purification replicates for candidates that advance.

### Phase 3: assay-independent stability

Do not use proteolysis as the primary stability measurement. For the top 24 by expression/activity Pareto rank:

- measure Tm by nanoDSF or CD;
- measure residual peptide-cleavage activity after 0, 1, 4, and 24 h at 30 °C;
- include freeze-thaw and short bench-hold challenges if those match the intended use.

### Phase 4: finalist validation

For the top 12:

- determine kcat, Km, and kcat/Km from at least eight substrate concentrations spanning below and above Km;
- repeat from two independent purifications;
- measure SEC-MALS or an equivalent monomer/oligomer assessment;
- quantify autolysis over time.

For the top six:

- run the MBP-TEVcs-FKBP-EGFP protein-substrate cleavage assay used in the paper;
- test a small P1/P1′ and near-consensus substrate panel to ensure activity gains do not come with unacceptable specificity broadening;
- perform intact-mass confirmation and peptide mapping if the enzyme will be used as a reagent.

## Prespecified advancement rules

A primary three-axis hit should satisfy all of:

- monomeric soluble yield ≥ 5× TEVd;
- specific peptide activity ≥ 2× TEVd;
- Tm ≥ TEVd + 10 °C;
- SEC monomer fraction ≥ 90%;
- no material loss on the protein-substrate assay or specificity panel.

Also retain Pareto specialists when one property is exceptional and the other two are non-inferior (≥80% of the relevant control). Compare elite candidates with hyperTEV60, not only TEVd.

Randomize wells, block controls across plates, blind sample identities during rate fitting, and model culture/purification batch as a random effect. Predeclare the primary endpoints above before looking at the data.

## Round-two learning

The first true individual-design expression/activity table will be more valuable than further fitting to the paper's overplotted figure. After the first lab round:

1. fit separate within-arm models for total expression, soluble fraction, monomer yield, specific activity, and stability;
2. retain one complete generation arm as a holdout;
3. test K45A versus K45 directly;
4. recalibrate Ridgey ensemble uncertainty and the MSA floors;
5. generate the second library from measured Pareto improvements, not from a pooled macro correlation.

## Reproducibility and outputs

The canonical analysis is on aws0 at:

`/opt/dlami/nvme/tev_reanalysis/round2_generation_validation/`

It contains all 32 Ridgey job IDs, raw 600M/6B full-plate predictions, 24 integrated-gradient responses, source hashes, derived per-design JSON tables, and figures. The complete earlier paper reanalysis remains at `/opt/dlami/nvme/tev_reanalysis/`; the restored 97-design plate remains at `/opt/dlami/nvme/tev_plate_design/`.
