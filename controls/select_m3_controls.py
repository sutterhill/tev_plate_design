#!/usr/bin/env python3
"""Select a diverse 24-design experimental-control panel from released m3 TEVs.

Activity tiering uses the Figure S7 panel-C raw-trace ordering.  The panel has
96 plots arranged row-major; released designs 49--96 are m3 and 97--144 are
m4.  This gives a robust relative slope ranking but not a publication-supplied
numeric per-design rate table.  Figure 3 identity-bin values are attached as a
cross-check and are never represented as exact per-design assignments unless
the source table marks them identifiable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REANALYSIS = Path("/opt/dlami/nvme/tev_reanalysis")
OUT = Path("/opt/dlami/nvme/tev_plate_design/work/m3_controls")

# Relative image slopes extracted from Figure S7c, row-major, designs 49--96.
# None means the trace was visually flat / a blue slope could not be fit.
TRACE_SLOPES = [
    None, 0.044, 0.032, 0.061, 0.219, 0.103, None, 6.297,
    0.137, 0.261, 0.202, 4.497, 0.436, 0.068, 0.037, 0.079,
    0.077, 3.814, 0.084, 0.093, 1.241, 0.129, 0.227, 0.296,
    0.107, 0.128, 0.066, 0.124, None, 0.015, 0.000, 0.448,
    0.223, 0.564, 0.149, 0.019, 2.376, 0.127, 0.311, None,
    4.328, 0.346, 0.230, 0.349, 0.101, 4.725, 0.259, -0.002,
]


def hamming(a: str, b: str) -> int:
    assert len(a) == len(b)
    return sum(x != y for x, y in zip(a, b))


def tier(score: float | None) -> str:
    if score is not None and score >= 0.40:
        return "active"
    if score is not None and score >= 0.12:
        return "somewhat_active"
    return "inactive_or_floor"


def farthest_add(
    pool: list[dict[str, str]], selected: list[dict[str, str]], n_add: int
) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    remaining = list(pool)
    for _ in range(n_add):
        anchors = selected + chosen
        if anchors:
            key = lambda r: (
                min(hamming(r["sequence"], x["sequence"]) for x in anchors),
                sum(hamming(r["sequence"], x["sequence"]) for x in remaining),
                float(r["trace_slope_relative"] or -1),
                -int(r["design_number"]),
            )
        else:
            key = lambda r: (
                sum(hamming(r["sequence"], x["sequence"]) for x in remaining),
                float(r["trace_slope_relative"] or -1),
                -int(r["design_number"]),
            )
        best = max(remaining, key=key)
        chosen.append(best)
        remaining.remove(best)
    return chosen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (REANALYSIS / "dataset/design_metadata.csv").open(newline="") as f:
        metadata = list(csv.DictReader(f))
    m3 = [r for r in metadata if r["method_code"] == "m3"]
    assert [int(r["design_number"]) for r in m3] == list(range(49, 97))

    with (REANALYSIS / "dataset/figure3_identity_bin_fitness.csv").open(newline="") as f:
        bins = list(csv.DictReader(f))
    activity_bins = {
        (r["method_code"], r["identity_bin_percent"]): r
        for r in bins
        if r["metric"] == "apparent_rate_RFU_s"
    }

    slope_by_number = {i: s for i, s in enumerate(TRACE_SLOPES, 49)}
    for r in m3:
        score = slope_by_number[int(r["design_number"])]
        r["trace_slope_relative"] = "" if score is None else f"{score:.3f}"
        r["activity_tier"] = tier(score)

    # Three named leads are mandatory.  For the active tier, retain the eight
    # strongest traces so that this is a true positive-control set (this also
    # includes all three leads).  Use farthest-point selection for the broader
    # middle and floor tiers to avoid redundant negative controls.
    leads = [r for r in m3 if int(r["design_number"]) in {56, 60, 89}]
    active_pool = [r for r in m3 if r["activity_tier"] == "active"]
    active_pool.sort(key=lambda r: float(r["trace_slope_relative"]), reverse=True)
    selected = active_pool[:8]
    assert all(x in selected for x in leads)
    for label in ["somewhat_active", "inactive_or_floor"]:
        pool = [r for r in m3 if r["activity_tier"] == label]
        selected.extend(farthest_add(pool, selected, 8))

    tier_order = {"active": 0, "somewhat_active": 1, "inactive_or_floor": 2}
    selected.sort(key=lambda r: (tier_order[r["activity_tier"]], -float(r["trace_slope_relative"] or -1)))
    assert len(selected) == 24
    assert {x["activity_tier"] for x in selected} == {"active", "somewhat_active", "inactive_or_floor"}
    assert {x["design_id"] for x in selected} >= {"hyperTEV56", "hyperTEV60", "hyperTEV89"}

    output_rows = []
    for panel_index, r in enumerate(selected, 1):
        b = activity_bins[("m3", r["identity_to_TEVD_percent"])]
        is_lead = r["design_id"] in {"hyperTEV56", "hyperTEV60", "hyperTEV89"}
        output_rows.append(
            {
                "panel_index": panel_index,
                "design_id": r["design_id"],
                "design_number": r["design_number"],
                "method_code": r["method_code"],
                "activity_tier": r["activity_tier"],
                "named_hyperTEV_lead": str(is_lead),
                "activity_evidence": (
                    "named lead with published Michaelis-Menten kinetics; Figure S7c trace tier"
                    if is_lead
                    else "tier inferred from Figure S7c row-major trace ordering"
                ),
                "trace_slope_relative": r["trace_slope_relative"],
                "figure3_identity_bin_percent": b["identity_bin_percent"],
                "figure3_bin_rate_min_RFU_s": b["experimental_min"],
                "figure3_bin_rate_median_RFU_s": b["experimental_median"],
                "figure3_bin_rate_max_RFU_s": b["experimental_max"],
                "figure3_bin_n_released_designs": b["n_released_designs"],
                "figure3_bin_n_visible_markers": b["n_visible_markers"],
                "figure3_individual_assignment_identifiable": b["individual_design_assignment_identifiable"],
                "identity_to_TEVD_percent": r["identity_to_TEVD_percent"],
                "n_mutations": r["n_mutations"],
                "mutations": r["mutations"],
                "sequence": r["sequence"],
                "structure_file": str(REANALYSIS / r["analysis_structure_file"]),
            }
        )

    csv_path = OUT / "proposed_m3_24_controls.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        w.writeheader()
        w.writerows(output_rows)

    all_scores = []
    for r in m3:
        all_scores.append(
            {
                "design_id": r["design_id"],
                "design_number": r["design_number"],
                "activity_tier": r["activity_tier"],
                "trace_slope_relative": r["trace_slope_relative"],
                "selected": r in selected,
            }
        )
    with (OUT / "m3_all_trace_tiers.json").open("w") as f:
        json.dump(all_scores, f, indent=2)

    counts = {}
    for r in output_rows:
        counts[r["activity_tier"]] = counts.get(r["activity_tier"], 0) + 1
    print(csv_path)
    print(counts)
    for r in output_rows:
        print(r["activity_tier"], r["design_id"], r["trace_slope_relative"])


if __name__ == "__main__":
    main()
