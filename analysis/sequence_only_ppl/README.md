# Ridgey sequence-only pseudo-perplexity versus TEV yield

## Bottom line

Ridgey sequence likelihood is **not a robust general predictor** of the paper's recovered-monomer yield. The apparently useful 600M pooled association is mostly a design-arm effect. Ridgey 6B has one interesting, prespecified-arm result in m3, but the public data support only identity-bin averages rather than individual sequence-to-yield pairs.

| Model | Pooled Spearman | Pooled within-arm rank correlation | Within-arm permutation p | m3 Spearman | m3 exact p | m3 FDR q | m3 partial Spearman controlling identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridgey 600M | +0.514 | +0.090 | 0.613 | +0.267 | 0.493 | 0.977 | +0.580 |
| Ridgey 6B | +0.087 | +0.182 | 0.299 | **+0.833** | **0.0083** | 0.066 | +0.659 |

The 6B m3 result is promising enough to retain as a soft ranking feature, but not strong enough to use as a hard expression/solubility gate. It is based on nine m3 identity bins; several bins contain multiple sequences whose individual experimental points cannot be assigned from the released figure.

## What was scored

For TEVd plus all 144 released designs, every one of 221 residues was masked once and scored as:

`PPL = exp(-mean_i log p(x_i | x_without_i, structure=None))`

The encoder received `structure=None` in every forward pass. This is masked pseudo-perplexity, not the unmasked LM matrix returned by an ordinary Ridgey prediction and not a sequence endpoint that silently fetches or folds a structure.

- 32,045 masked-residue examples per checkpoint
- Ridgey 600M runtime: 60.95 s on one H100
- Ridgey 6B runtime: 420.67 s on one H100
- Complete per-residue log probabilities are retained in the compressed raw outputs

The checkpoints disagree substantially:

- 600M TEVd PPL is 13.05; all 144 designs have lower PPL.
- 6B TEVd PPL is 6.06; all 144 designs have higher PPL.
- The 600M-versus-6B per-design likelihood correlation is -0.049.

This checkpoint disagreement is another reason not to treat sequence PPL as a standalone experimental-fitness estimate.

## Experimental endpoint and limitation

The tested outcome is the digitized Figure 3 recovered monomeric yield in mg/L, joined at the paper-arm plus exact-sequence-identity-bin level (37 bins). It combines expression, soluble recovery, purification, and SEC monomer selection.

The paper reports 134/144 soluble monomers and 129/144 above TEVd, but does not release the identities of the failures or a machine-readable individual-design yield table. Therefore an individual-design binary-solubility AUC cannot be computed honestly from the public release.

## Recommended use

- Keep 6B sequence-only PPL as a soft Pareto feature for m3 generation.
- Do not gate candidates on 600M PPL alone.
- Preserve designs that disagree between 600M and 6B as uncertainty/calibration examples.
- Refit this analysis immediately when the lab produces an individual-design table separating total expression, soluble fraction, monomeric yield, and specific activity.

## Outputs

- `raw/ridgey_600m_sequence_only_ppl.json.gz`
- `raw/ridgey_6b_sequence_only_ppl.json.gz`
- `results/sequence_only_ppl_yield_analysis.json`
- `results/sequence_only_ppl_vs_yield.png`
- `results/sequence_only_ppl_yield_correlations.png`
- `results/sequence_only_ppl_within_arm_heatmap.png`
