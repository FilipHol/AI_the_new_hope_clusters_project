from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CLUSTERS,
    DATA_DIR,
    ISO_AG_OVER_AV,
    ISO_ALLOWED_PHASES,
    ISO_AV,
    ISO_AV_SENS_EXTREME,
    ISO_AV_SENS_PLAUSIBLE,
    ISO_D_CAP,
    ISO_D_CAP_SENS,
    ISO_DIST_SENS_FRAC,
    ISO_EBPRP_OVER_AV,
    ISO_FEH_BASELINE,
    ISO_FEH_PRAESEPE_SENS,
    ISO_FIT_MG_MAX,
    ISO_FIT_MG_MAX_SENS,
    ISO_LIT_AGE_MYR,
    ISO_LOGAGE_MAX,
    ISO_LOGAGE_MIN,
    ISO_LOGAGE_STEP,
    ISO_LOSS_INTERVAL_FRAC,
    ISO_OFFSET_DROP_FRAC,
    ISO_OFFSET_MG_RANGE,
    ISO_OFFSET_SENS,
    ISO_SIGMA_COLOUR,
    ISO_SIGMA_G,
    OUT_DIR,
)
from iso_lib import (
    distance_modulus,
    logage_to_myr,
    prepare_members,
    read_mist,
    read_mist_interpolated,
    run_isochrone_fit_cycle,
)


SUMMARY_PATH = OUT_DIR / "summary.csv"
OUTPUT_CSV = OUT_DIR / "isochrone_sensitivity.csv"
OUTPUT_PNG = OUT_DIR / "isochrone_sensitivity.png"
EXTREME_COLOUR = "tab:red"
DEFAULT_COLOUR = "black"
BASELINE_COLOUR = "tab:blue"


def load_members(cluster_key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{cluster_key}_members.parquet")


def load_distance_map() -> dict[str, float]:
    summary = pd.read_csv(SUMMARY_PATH)
    if "cluster_key" not in summary.columns or "median_distance_pc_zp017" not in summary.columns:
        raise ValueError("summary.csv is missing required columns: cluster_key, median_distance_pc_zp017")
    return dict(zip(summary["cluster_key"], summary["median_distance_pc_zp017"]))


def add_case_row(
    rows: list[dict[str, float | int | str]],
    cluster_key: str,
    case: str,
    parameter: str,
    cycle: dict[str, object],
) -> None:
    fit_2 = cycle["fit_2"]
    rows.append(
        {
            "cluster": cluster_key,
            "case": case,
            "parameter": parameter,
            "colour_offset": float(cycle["colour_offset_2"]),
            "best_age_grid_myr": float(fit_2["best_grid_age_myr"]),
            "best_age_parabola_myr": float(fit_2["best_age_myr"]),
            "loss_min": float(fit_2["best_loss"]),
            "n_stars_fit": int(len(cycle["fit_members"])),
            "frac_capped": float(fit_2["frac_capped"][int(fit_2["best_index"])]),
        }
    )


def descriptive_loss_interval(fit_result: dict[str, object]) -> tuple[float, float, float]:
    losses = np.asarray(fit_result["losses"], dtype=float)
    logage_grid = np.asarray(fit_result["logage_grid"], dtype=float)
    threshold = (1.0 + ISO_LOSS_INTERVAL_FRAC) * float(losses.min())
    kept = logage_grid[losses <= threshold]
    min_age = logage_to_myr(float(kept.min()))
    max_age = logage_to_myr(float(kept.max()))
    return min_age, max_age, max_age - min_age


def make_case_label(case: str, parameter: str) -> str:
    if not parameter:
        return case
    return f"{case}\n{parameter}"


def save_sensitivity_plot(results_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, constrained_layout=True)

    for ax, cluster_key in zip(axes, CLUSTERS.keys(), strict=False):
        cluster_df = results_df.loc[results_df["cluster"] == cluster_key].reset_index(drop=True)
        x = np.arange(len(cluster_df))
        colours = []
        for _, row in cluster_df.iterrows():
            if row["case"] == "extreme_av":
                colours.append(EXTREME_COLOUR)
            elif row["case"] == "baseline":
                colours.append(BASELINE_COLOUR)
            else:
                colours.append(DEFAULT_COLOUR)

        ax.scatter(x, cluster_df["best_age_grid_myr"], c=colours, s=35)
        ax.plot(x, cluster_df["best_age_grid_myr"], color="0.7", linewidth=1.0, zorder=0)
        ax.set_yscale("log")
        ax.set_title(CLUSTERS[cluster_key]["name"])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [make_case_label(str(case), str(parameter)) for case, parameter in zip(cluster_df["case"], cluster_df["parameter"], strict=False)],
            rotation=90,
        )
        ax.set_ylabel("Best grid age (Myr)")

    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    distances = load_distance_map()
    baseline_isochrones = read_mist(ISO_FEH_BASELINE)
    praesepe_isochrones = read_mist_interpolated(ISO_FEH_PRAESEPE_SENS)
    rows: list[dict[str, float | int | str]] = []
    plausible_cases = {"av_plausible", "distance", "feh_praesepe", "offset_fixed", "fit_window", "loss_cap"}

    for cluster_key, cluster in CLUSTERS.items():
        distance_pc = float(distances[cluster_key])
        base_a_v = float(ISO_AV[cluster_key])
        base_a_g = float(ISO_AG_OVER_AV * base_a_v)
        base_e_bprp = float(ISO_EBPRP_OVER_AV * base_a_v)
        members = prepare_members(load_members(cluster_key), distance_modulus(distance_pc), base_a_g)

        baseline_cycle = run_isochrone_fit_cycle(
            members,
            baseline_isochrones,
            distance_modulus(distance_pc),
            base_a_g,
            base_e_bprp,
            literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
            fit_mg_max=ISO_FIT_MG_MAX,
            offset_mg_range=ISO_OFFSET_MG_RANGE,
            offset_drop_frac=ISO_OFFSET_DROP_FRAC,
            logage_min=ISO_LOGAGE_MIN,
            logage_max=ISO_LOGAGE_MAX,
            logage_step=ISO_LOGAGE_STEP,
            allowed_phases=ISO_ALLOWED_PHASES,
            sigma_colour=ISO_SIGMA_COLOUR,
            sigma_g=ISO_SIGMA_G,
            d_cap=ISO_D_CAP,
        )
        add_case_row(rows, cluster_key, "baseline", "baseline", baseline_cycle)

        baseline_offset = float(baseline_cycle["colour_offset_2"])

        for av_value in ISO_AV_SENS_PLAUSIBLE:
            a_g = float(ISO_AG_OVER_AV * av_value)
            e_bprp = float(ISO_EBPRP_OVER_AV * av_value)
            cycle = run_isochrone_fit_cycle(
                members,
                baseline_isochrones,
                distance_modulus(distance_pc),
                a_g,
                e_bprp,
                literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
                fit_mg_max=ISO_FIT_MG_MAX,
                offset_mg_range=ISO_OFFSET_MG_RANGE,
                offset_drop_frac=ISO_OFFSET_DROP_FRAC,
                logage_min=ISO_LOGAGE_MIN,
                logage_max=ISO_LOGAGE_MAX,
                logage_step=ISO_LOGAGE_STEP,
                allowed_phases=ISO_ALLOWED_PHASES,
                sigma_colour=ISO_SIGMA_COLOUR,
                sigma_g=ISO_SIGMA_G,
                d_cap=ISO_D_CAP,
            )
            add_case_row(rows, cluster_key, "av_plausible", f"A_V={av_value:.2f}", cycle)

        for factor in (1.0 - ISO_DIST_SENS_FRAC, 1.0 + ISO_DIST_SENS_FRAC):
            dm = distance_modulus(distance_pc * factor)
            cycle = run_isochrone_fit_cycle(
                members,
                baseline_isochrones,
                dm,
                base_a_g,
                base_e_bprp,
                literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
                fit_mg_max=ISO_FIT_MG_MAX,
                offset_mg_range=ISO_OFFSET_MG_RANGE,
                offset_drop_frac=ISO_OFFSET_DROP_FRAC,
                logage_min=ISO_LOGAGE_MIN,
                logage_max=ISO_LOGAGE_MAX,
                logage_step=ISO_LOGAGE_STEP,
                allowed_phases=ISO_ALLOWED_PHASES,
                sigma_colour=ISO_SIGMA_COLOUR,
                sigma_g=ISO_SIGMA_G,
                d_cap=ISO_D_CAP,
            )
            add_case_row(rows, cluster_key, "distance", f"x{factor:.2f}", cycle)

        if cluster_key == "praesepe":
            cycle = run_isochrone_fit_cycle(
                members,
                praesepe_isochrones,
                distance_modulus(distance_pc),
                base_a_g,
                base_e_bprp,
                literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
                fit_mg_max=ISO_FIT_MG_MAX,
                offset_mg_range=ISO_OFFSET_MG_RANGE,
                offset_drop_frac=ISO_OFFSET_DROP_FRAC,
                logage_min=ISO_LOGAGE_MIN,
                logage_max=ISO_LOGAGE_MAX,
                logage_step=ISO_LOGAGE_STEP,
                allowed_phases=ISO_ALLOWED_PHASES,
                sigma_colour=ISO_SIGMA_COLOUR,
                sigma_g=ISO_SIGMA_G,
                d_cap=ISO_D_CAP,
            )
            add_case_row(rows, cluster_key, "feh_praesepe", f"[Fe/H]={ISO_FEH_PRAESEPE_SENS:.2f}", cycle)

        for fixed_offset in (0.0, baseline_offset - ISO_OFFSET_SENS, baseline_offset + ISO_OFFSET_SENS):
            cycle = run_isochrone_fit_cycle(
                members,
                baseline_isochrones,
                distance_modulus(distance_pc),
                base_a_g,
                base_e_bprp,
                literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
                fit_mg_max=ISO_FIT_MG_MAX,
                offset_mg_range=ISO_OFFSET_MG_RANGE,
                offset_drop_frac=ISO_OFFSET_DROP_FRAC,
                logage_min=ISO_LOGAGE_MIN,
                logage_max=ISO_LOGAGE_MAX,
                logage_step=ISO_LOGAGE_STEP,
                allowed_phases=ISO_ALLOWED_PHASES,
                sigma_colour=ISO_SIGMA_COLOUR,
                sigma_g=ISO_SIGMA_G,
                d_cap=ISO_D_CAP,
                fixed_offset=float(fixed_offset),
            )
            add_case_row(rows, cluster_key, "offset_fixed", f"offset={fixed_offset:.4f}", cycle)

        cycle = run_isochrone_fit_cycle(
            members,
            baseline_isochrones,
            distance_modulus(distance_pc),
            base_a_g,
            base_e_bprp,
            literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
            fit_mg_max=ISO_FIT_MG_MAX_SENS,
            offset_mg_range=ISO_OFFSET_MG_RANGE,
            offset_drop_frac=ISO_OFFSET_DROP_FRAC,
            logage_min=ISO_LOGAGE_MIN,
            logage_max=ISO_LOGAGE_MAX,
            logage_step=ISO_LOGAGE_STEP,
            allowed_phases=ISO_ALLOWED_PHASES,
            sigma_colour=ISO_SIGMA_COLOUR,
            sigma_g=ISO_SIGMA_G,
            d_cap=ISO_D_CAP,
        )
        add_case_row(rows, cluster_key, "fit_window", f"M_G<{ISO_FIT_MG_MAX_SENS:.1f}", cycle)

        cycle = run_isochrone_fit_cycle(
            members,
            baseline_isochrones,
            distance_modulus(distance_pc),
            base_a_g,
            base_e_bprp,
            literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
            fit_mg_max=ISO_FIT_MG_MAX,
            offset_mg_range=ISO_OFFSET_MG_RANGE,
            offset_drop_frac=ISO_OFFSET_DROP_FRAC,
            logage_min=ISO_LOGAGE_MIN,
            logage_max=ISO_LOGAGE_MAX,
            logage_step=ISO_LOGAGE_STEP,
            allowed_phases=ISO_ALLOWED_PHASES,
            sigma_colour=ISO_SIGMA_COLOUR,
            sigma_g=ISO_SIGMA_G,
            d_cap=ISO_D_CAP_SENS,
        )
        add_case_row(rows, cluster_key, "loss_cap", f"D_CAP={ISO_D_CAP_SENS:.1f}", cycle)

        extreme_a_g = float(ISO_AG_OVER_AV * ISO_AV_SENS_EXTREME)
        extreme_e_bprp = float(ISO_EBPRP_OVER_AV * ISO_AV_SENS_EXTREME)
        cycle = run_isochrone_fit_cycle(
            members,
            baseline_isochrones,
            distance_modulus(distance_pc),
            extreme_a_g,
            extreme_e_bprp,
            literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
            fit_mg_max=ISO_FIT_MG_MAX,
            offset_mg_range=ISO_OFFSET_MG_RANGE,
            offset_drop_frac=ISO_OFFSET_DROP_FRAC,
            logage_min=ISO_LOGAGE_MIN,
            logage_max=ISO_LOGAGE_MAX,
            logage_step=ISO_LOGAGE_STEP,
            allowed_phases=ISO_ALLOWED_PHASES,
            sigma_colour=ISO_SIGMA_COLOUR,
            sigma_g=ISO_SIGMA_G,
            d_cap=ISO_D_CAP,
        )
        add_case_row(rows, cluster_key, "extreme_av", f"A_V={ISO_AV_SENS_EXTREME:.2f}", cycle)

        cluster_rows = [row for row in rows if row["cluster"] == cluster_key]
        plausible_grid_ages = [float(row["best_age_grid_myr"]) for row in cluster_rows if str(row["case"]) in plausible_cases]
        plausible_min = min(plausible_grid_ages)
        plausible_max = max(plausible_grid_ages)
        extreme_age = next(float(row["best_age_grid_myr"]) for row in cluster_rows if row["case"] == "extreme_av")
        interval_min, interval_max, interval_width = descriptive_loss_interval(baseline_cycle["fit_2"])

        print(f"{cluster['name']} ({cluster_key}):")
        print(f"  plausible grid-age range over cases b-g: {plausible_min:.3f} to {plausible_max:.3f} Myr")
        print(f"  extreme-case grid age: {extreme_age:.3f} Myr")
        print(
            f"  baseline descriptive loss-width interval (not a confidence interval): "
            f"{interval_min:.3f} to {interval_max:.3f} Myr, width={interval_width:.3f} Myr"
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    save_sensitivity_plot(results_df)


if __name__ == "__main__":
    main()
