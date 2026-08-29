#!/usr/bin/env python3
"""Relate Ridgey sequence-only pseudo-perplexity to digitized TEV yield."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata, spearmanr


ARM_ORDER = ["m1", "m2", "m3", "m4"]
ARM_COLORS = {"m1": "#4477AA", "m2": "#66CCEE", "m3": "#228833", "m4": "#CC6677"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-metadata", type=Path, required=True)
    parser.add_argument("--fitness-bins", type=Path, required=True)
    parser.add_argument("--ppl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_ppl(path: Path) -> tuple[str, pd.DataFrame, dict]:
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    frame = pd.DataFrame(payload["records"])
    frame = frame[frame.design_id != "TEVd"].copy()
    return str(payload["model"]), frame, payload


def correlation(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def pooled_within_arm_rank_correlation(frame: pd.DataFrame) -> float:
    ranked = []
    for _, sub in frame.groupby("method_code", sort=True):
        local = sub[["sequence_log_probability", "experimental_mean"]].copy()
        local["score_rank"] = local.sequence_log_probability.rank(pct=True, method="average")
        local["yield_rank"] = local.experimental_mean.rank(pct=True, method="average")
        ranked.append(local[["score_rank", "yield_rank"]])
    combined = pd.concat(ranked, ignore_index=True)
    return float(np.corrcoef(combined.score_rank, combined.yield_rank)[0, 1])


def partial_spearman_controlling_identity(frame: pd.DataFrame) -> float:
    x = rankdata(frame.sequence_log_probability.to_numpy())
    y = rankdata(frame.experimental_mean.to_numpy())
    z = rankdata(frame.identity_bin_percent.to_numpy())
    design = np.column_stack([np.ones(len(z)), z])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def exact_permutation_p_if_small(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) > 9:
        return None
    x_rank = rankdata(x.to_numpy()).astype(float)
    y_rank = rankdata(y.to_numpy()).astype(float)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    x_rank /= np.linalg.norm(x_rank)
    y_rank /= np.linalg.norm(y_rank)
    observed = float(x_rank @ y_rank)
    extreme = 0
    total = 0
    for permuted in itertools.permutations(y_rank):
        value = float(x_rank @ np.asarray(permuted))
        extreme += abs(value) >= abs(observed) - 1e-12
        total += 1
    return float(extreme / total)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, float(p_values[index]) * len(p_values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def stratified_permutation_p(frame: pd.DataFrame, observed: float, seed: int = 7, n: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    scores = frame.sequence_log_probability.to_numpy()
    outcomes = frame.experimental_mean.to_numpy()
    indices = [np.flatnonzero(frame.method_code.to_numpy() == arm) for arm in ARM_ORDER]
    extreme = 0
    for _ in range(n):
        shuffled = outcomes.copy()
        for arm_indices in indices:
            shuffled[arm_indices] = rng.permutation(shuffled[arm_indices])
        value = spearmanr(scores, shuffled).statistic
        extreme += abs(value) >= abs(observed)
    return float((extreme + 1) / (n + 1))


def within_arm_rank_permutation_p(frame: pd.DataFrame, observed: float, seed: int = 17, n: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    score_ranks = frame.groupby("method_code").sequence_log_probability.rank(pct=True).to_numpy()
    yield_ranks = frame.groupby("method_code").experimental_mean.rank(pct=True).to_numpy()
    score_ranks = (score_ranks - score_ranks.mean()) / score_ranks.std()
    indices = [np.flatnonzero(frame.method_code.to_numpy() == arm) for arm in ARM_ORDER]
    extreme = 0
    for _ in range(n):
        shuffled = yield_ranks.copy()
        for arm_indices in indices:
            shuffled[arm_indices] = rng.permutation(shuffled[arm_indices])
        value = float(np.corrcoef(score_ranks, shuffled)[0, 1])
        extreme += abs(value) >= abs(observed)
    return float((extreme + 1) / (n + 1))


def stratified_bootstrap_ci(frame: pd.DataFrame, seed: int = 11, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    arms = [sub.reset_index(drop=True) for _, sub in frame.groupby("method_code", sort=True)]
    values = []
    for _ in range(n):
        sample = pd.concat(
            [sub.iloc[rng.integers(0, len(sub), size=len(sub))] for sub in arms],
            ignore_index=True,
        )
        value = correlation(sample.sequence_log_probability, sample.experimental_mean)
        if np.isfinite(value):
            values.append(value)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def model_result(model: str, design_scores: pd.DataFrame, fitness: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    design_scores = design_scores.copy()
    design_scores["identity_bin_percent"] = design_scores.identity_to_TEVD_percent.round(6)
    aggregated = (
        design_scores.groupby(["method_code", "identity_bin_percent"], as_index=False)
        .agg(
            sequence_log_probability=("mean_masked_log_probability", "mean"),
            pseudo_perplexity=("pseudo_perplexity", "mean"),
            n_scored_designs=("design_id", "size"),
        )
    )
    observed = fitness[fitness.metric.eq("yield_mg_L")].copy()
    observed["identity_bin_percent"] = observed.identity_bin_percent.round(6)
    joined = observed.merge(
        aggregated,
        on=["method_code", "identity_bin_percent"],
        how="inner",
        validate="one_to_one",
    )
    joined = joined[
        np.isfinite(joined.experimental_mean)
        & np.isfinite(joined.sequence_log_probability)
    ].copy()
    pooled = correlation(joined.sequence_log_probability, joined.experimental_mean)
    arm_results = []
    for arm in ARM_ORDER:
        sub = joined[joined.method_code.eq(arm)]
        spearman = spearmanr(sub.sequence_log_probability, sub.experimental_mean)
        arm_results.append(
            {
                "arm": arm,
                "n_bins": len(sub),
                "spearman_rho": float(spearman.statistic),
                "spearman_p_two_sided_asymptotic": float(spearman.pvalue),
                "exact_permutation_p_two_sided": exact_permutation_p_if_small(
                    sub.sequence_log_probability, sub.experimental_mean
                ),
                "partial_spearman_controlling_identity": partial_spearman_controlling_identity(sub),
            }
        )
    finite_arm = [row["spearman_rho"] for row in arm_results if np.isfinite(row["spearman_rho"])]
    within_arm_rank = pooled_within_arm_rank_correlation(joined)
    result = {
        "model": model,
        "score_direction": "higher mean masked log probability / lower PPL is better",
        "n_method_identity_bins": len(joined),
        "pooled_spearman_rho": pooled,
        "pooled_stratified_bootstrap_95pct_ci": stratified_bootstrap_ci(joined),
        "within_arm_correlations": arm_results,
        "mean_within_arm_spearman": float(np.mean(finite_arm)),
        "pooled_within_arm_rank_correlation": within_arm_rank,
        "pooled_within_arm_rank_permutation_p_two_sided": within_arm_rank_permutation_p(
            joined, within_arm_rank
        ),
        "stratified_permutation_p_two_sided": stratified_permutation_p(joined, pooled),
        "likelihood_vs_identity_spearman_all_144": correlation(
            design_scores.mean_masked_log_probability,
            design_scores.identity_to_TEVD_percent,
        ),
        "arm_score_summary": [
            {
                "arm": str(arm),
                "n_designs": int(len(sub)),
                "mean_masked_log_probability": float(sub.mean_masked_log_probability.mean()),
                "median_pseudo_perplexity": float(sub.pseudo_perplexity.median()),
            }
            for arm, sub in design_scores.groupby("method_code", sort=True)
        ],
    }
    joined["model"] = model
    return result, joined


def plot_scatter(joined: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    models = list(dict.fromkeys(joined.model))
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 6.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for axis, model in zip(axes, models):
        sub = joined[joined.model.eq(model)]
        for arm in ARM_ORDER:
            arm_data = sub[sub.method_code.eq(arm)]
            axis.scatter(
                arm_data.sequence_log_probability,
                arm_data.experimental_mean,
                s=55 + 12 * arm_data.n_released_designs,
                alpha=0.8,
                color=ARM_COLORS[arm],
                edgecolor="white",
                linewidth=0.8,
                label=arm,
            )
            if len(arm_data) >= 3:
                fit = np.polyfit(arm_data.sequence_log_probability, arm_data.experimental_mean, 1)
                grid = np.linspace(arm_data.sequence_log_probability.min(), arm_data.sequence_log_probability.max(), 50)
                axis.plot(grid, np.polyval(fit, grid), color=ARM_COLORS[arm], alpha=0.7, linewidth=2)
        axis.set_title(f"Ridgey {model}: no-structure pseudo-PPL")
        axis.set_xlabel("Mean masked log probability (higher = lower PPL)")
    axes[0].set_ylabel("Recovered monomeric yield (mg/L; digitized bin mean)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], title="Paper arm", loc="center right")
    fig.suptitle("Sequence-only Ridgey likelihood versus TEV recovered yield", y=1.01)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(results: list[dict], output: Path) -> None:
    rows = []
    for result in results:
        rows.append({"model": result["model"], "estimate": "Pooled", "rho": result["pooled_spearman_rho"]})
        rows.append({"model": result["model"], "estimate": "Mean within arm", "rho": result["mean_within_arm_spearman"]})
    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="talk")
    fig, axis = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=frame, x="model", y="rho", hue="estimate", palette=["#4477AA", "#EE8866"], ax=axis)
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_ylim(-1, 1)
    axis.set_xlabel("Ridgey checkpoint")
    axis.set_ylabel("Spearman correlation with recovered yield")
    axis.set_title("Does no-structure sequence likelihood predict TEV yield?")
    axis.legend(title=None, frameon=False)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_within_arm_heatmap(results: list[dict], output: Path) -> None:
    models = [result["model"] for result in results]
    values = np.asarray(
        [
            [arm_result["spearman_rho"] for arm_result in result["within_arm_correlations"]]
            for result in results
        ]
    )
    annotations = np.empty_like(values, dtype=object)
    for row, result in enumerate(results):
        for column, arm_result in enumerate(result["within_arm_correlations"]):
            annotations[row, column] = (
                f"ρ={arm_result['spearman_rho']:+.2f}\n"
                f"q={arm_result['bh_fdr_across_2_models_x_4_arms']:.3f}"
            )
    sns.set_theme(style="white", context="talk")
    fig, axis = plt.subplots(figsize=(9, 4.2))
    sns.heatmap(
        values,
        annot=annotations,
        fmt="",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=ARM_ORDER,
        yticklabels=models,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ"},
        ax=axis,
    )
    axis.set_xlabel("Paper design arm")
    axis.set_ylabel("Ridgey checkpoint")
    axis.set_title("Within-arm sequence likelihood vs recovered TEV yield")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fitness = pd.read_csv(args.fitness_bins)
    metadata = pd.read_csv(args.design_metadata)
    results = []
    joined_frames = []
    provenance = []
    for path in args.ppl:
        model, scores, payload = load_ppl(path)
        result, joined = model_result(model, scores, fitness)
        tevd = next(record for record in payload["records"] if record["design_id"] == "TEVd")
        result["tevd_pseudo_perplexity"] = float(tevd["pseudo_perplexity"])
        result["n_of_144_designs_with_lower_ppl_than_tevd"] = int(
            (scores.pseudo_perplexity < float(tevd["pseudo_perplexity"])).sum()
        )
        results.append(result)
        joined_frames.append(joined)
        provenance.append(
            {
                "model": model,
                "input": str(path.resolve()),
                "checkpoint": payload["checkpoint"],
                "definition": payload["definition"],
                "structure_argument": payload["structure_argument"],
            }
        )
    arm_tests = [
        arm_result
        for result in results
        for arm_result in result["within_arm_correlations"]
    ]
    raw_p_values = [
        arm_result["exact_permutation_p_two_sided"]
        if arm_result["exact_permutation_p_two_sided"] is not None
        else arm_result["spearman_p_two_sided_asymptotic"]
        for arm_result in arm_tests
    ]
    for arm_result, adjusted in zip(arm_tests, benjamini_hochberg(raw_p_values)):
        arm_result["bh_fdr_across_2_models_x_4_arms"] = float(adjusted)
    joined = pd.concat(joined_frames, ignore_index=True)
    score_frames = []
    for path in args.ppl:
        model, scores, _ = load_ppl(path)
        score_frames.append(
            scores[["design_id", "mean_masked_log_probability"]].rename(
                columns={"mean_masked_log_probability": model}
            )
        )
    agreement = score_frames[0]
    for frame in score_frames[1:]:
        agreement = agreement.merge(frame, on="design_id", validate="one_to_one")
    model_agreement = None
    if len(score_frames) == 2:
        left, right = [column for column in agreement if column != "design_id"]
        model_agreement = {
            "models": [left, right],
            "spearman_rho_across_144_designs": correlation(agreement[left], agreement[right]),
        }
    plot_scatter(joined, args.output_dir / "sequence_only_ppl_vs_yield.png")
    plot_correlations(results, args.output_dir / "sequence_only_ppl_yield_correlations.png")
    plot_within_arm_heatmap(results, args.output_dir / "sequence_only_ppl_within_arm_heatmap.png")
    output = {
        "analysis_level": "paper method_code plus exact sequence-identity bin",
        "experimental_endpoint": "digitized recovered monomeric yield in mg/L",
        "n_released_designs": len(metadata),
        "individual_binary_solubility_available": False,
        "binary_solubility_limitation": (
            "The paper reports 134/144 soluble monomers and 129/144 above TEVd, "
            "but does not identify the individual failures in a machine-readable table."
        ),
        "results": results,
        "sequence_likelihood_model_agreement": model_agreement,
        "provenance": provenance,
        "joined_bins": joined.to_dict(orient="records"),
    }
    with (args.output_dir / "sequence_only_ppl_yield_analysis.json").open("w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
