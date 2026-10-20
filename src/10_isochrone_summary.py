from __future__ import annotations

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    CLUSTERS,
    DATA_DIR,
    ISO_AG_OVER_AV,
    ISO_AV,
    ISO_BOOTSTRAP_N,
    ISO_CONSTRAINED_LOSS_RATIO,
    ISO_EBPRP_OVER_AV,
    ISO_FEH_BASELINE,
    ISO_LIT_AGE_MYR,
    ISO_SEED,
    OUT_DIR,
)
from iso_lib import distance_modulus, prepare_members, read_mist, run_isochrone_fit_cycle


SUMMARY_PATH = OUT_DIR / "summary.csv"
SENSITIVITY_PATH = OUT_DIR / "isochrone_sensitivity.csv"
SUMMARY_OUTPUT_PATH = OUT_DIR / "isochrone_summary.csv"
ORDER_OUTPUT_PATH = OUT_DIR / "isochrone_order.csv"
PLOT_OUTPUT_PATH = OUT_DIR / "isochrone_summary.png"
PAIR_ORDER = [("pleiades", "praesepe"), ("praesepe", "m67"), ("pleiades", "m67")]


def load_members(cluster_key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{cluster_key}_members.parquet")


def load_distance_map() -> dict[str, float]:
    summary = pd.read_csv(SUMMARY_PATH)
    if "cluster_key" not in summary.columns or "median_distance_pc_zp017" not in summary.columns:
        raise ValueError("summary.csv is missing required columns: cluster_key, median_distance_pc_zp017")
    return dict(zip(summary["cluster_key"], summary["median_distance_pc_zp017"]))


def load_sensitivity_table() -> pd.DataFrame:
    return pd.read_csv(SENSITIVITY_PATH)


def extract_sensitivity_values(sensitivity_df: pd.DataFrame, cluster_key: str) -> tuple[float, float, float, float | None]:
    cluster_df = sensitivity_df.loc[sensitivity_df["cluster"] == cluster_key].copy()
    plausible_df = cluster_df.loc[~cluster_df["case"].isin(["baseline", "extreme_av"])]
    if plausible_df.empty:
        raise ValueError(f"No plausible sensitivity rows found for {cluster_key}")

    age_min = float(plausible_df["best_age_parabola_myr"].min())
    age_max = float(plausible_df["best_age_parabola_myr"].max())

    extreme_df = cluster_df.loc[cluster_df["case"] == "extreme_av"]
    if extreme_df.empty:
        raise ValueError(f"No extreme_av row found for {cluster_key}")
    extreme_age = float(extreme_df.iloc[0]["best_age_parabola_myr"])

    alt_feh = None
    alt_feh_df = cluster_df.loc[cluster_df["case"] == "feh_praesepe"]
    if not alt_feh_df.empty:
        alt_feh = float(alt_feh_df.iloc[0]["best_age_parabola_myr"])

    return age_min, age_max, extreme_age, alt_feh


def time_one_fit(
    members: pd.DataFrame,
    isochrones: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    literature_age_myr: float,
) -> tuple[dict[str, object], float]:
    t0 = time.perf_counter()
    cycle = run_isochrone_fit_cycle(members, isochrones, dm, a_g, e_bprp, literature_age_myr=literature_age_myr)
    elapsed = time.perf_counter() - t0
    return cycle, elapsed


def bootstrap_parabola_ages(
    members: pd.DataFrame,
    isochrones: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    literature_age_myr: float,
    seed: int,
    n_boot: int,
    use_vectorized_resampling: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_members = len(members)
    ages = np.empty(n_boot, dtype=float)

    if use_vectorized_resampling:
        all_indices = rng.integers(0, n_members, size=(n_boot, n_members))
        iterator = all_indices
    else:
        iterator = (rng.integers(0, n_members, size=n_members) for _ in range(n_boot))

    for draw_index, sample_indices in enumerate(iterator):
        sample = members.iloc[np.asarray(sample_indices)].reset_index(drop=True)
        cycle = run_isochrone_fit_cycle(sample, isochrones, dm, a_g, e_bprp, literature_age_myr=literature_age_myr)
        ages[draw_index] = float(cycle["fit_2"]["best_age_myr"])

    return ages


def make_summary_row(
    cluster_key: str,
    distance_pc: float,
    a_v: float,
    baseline_cycle: dict[str, object],
    age_min: float,
    age_max: float,
    age_extreme: float,
    alt_feh_age: float | None,
    bootstrap_ages: np.ndarray,
) -> dict[str, float | int | str | bool]:
    fit_2 = baseline_cycle["fit_2"]
    loss_min = float(fit_2["best_loss"])
    loss_ratio_minus = float(fit_2["loss_minus_0p2"] / loss_min)
    loss_ratio_plus = float(fit_2["loss_plus_0p2"] / loss_min)

    return {
        "cluster": cluster_key,
        "feh": float(ISO_FEH_BASELINE),
        "av": float(a_v),
        "distance_pc": float(distance_pc),
        "colour_offset": float(baseline_cycle["colour_offset_2"]),
        "best_age_myr": float(fit_2["best_age_myr"]),
        "age_min_myr": float(age_min),
        "age_max_myr": float(age_max),
        "age_extreme_av_myr": float(age_extreme),
        "boot_median_myr": float(np.median(bootstrap_ages)),
        "boot_p16_myr": float(np.percentile(bootstrap_ages, 16)),
        "boot_p84_myr": float(np.percentile(bootstrap_ages, 84)),
        "n_stars_fit": int(len(baseline_cycle["fit_members"])),
        "loss_ratio_minus02": loss_ratio_minus,
        "loss_ratio_plus02": loss_ratio_plus,
        "age_constrained": bool(
            (loss_ratio_minus >= ISO_CONSTRAINED_LOSS_RATIO) and (loss_ratio_plus >= ISO_CONSTRAINED_LOSS_RATIO)
        ),
        "age_alt_feh015_myr": np.nan if alt_feh_age is None else float(alt_feh_age),
    }


def make_order_rows(summary_df: pd.DataFrame, bootstrap_map: dict[str, np.ndarray]) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    summary_indexed = summary_df.set_index("cluster")

    for first, second in PAIR_ORDER:
        first_row = summary_indexed.loc[first]
        second_row = summary_indexed.loc[second]
        overlap = not (
            float(first_row["age_max_myr"]) < float(second_row["age_min_myr"])
            or float(second_row["age_max_myr"]) < float(first_row["age_min_myr"])
        )
        first_younger = float(np.mean(bootstrap_map[first] < bootstrap_map[second]))
        rows.append(
            {
                "first_cluster": first,
                "second_cluster": second,
                "plausible_ranges_overlap": bool(overlap),
                "frac_bootstrap_first_younger": first_younger,
            }
        )

    return rows


def save_summary_plot(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)

    for ax, cluster_key in zip(axes, CLUSTERS.keys(), strict=False):
        row = summary_df.loc[summary_df["cluster"] == cluster_key].iloc[0]
        x = np.array([0, 1, 2])
        y = np.array([
            float(row["best_age_myr"]),
            float(row["boot_median_myr"]),
            float(row["age_extreme_av_myr"]),
        ])
        yerr_low = np.array([
            float(row["best_age_myr"] - row["boot_p16_myr"]),
            float(row["boot_median_myr"] - row["boot_p16_myr"]),
            0.0,
        ])
        yerr_high = np.array([
            float(row["boot_p84_myr"] - row["best_age_myr"]),
            float(row["boot_p84_myr"] - row["boot_median_myr"]),
            0.0,
        ])
        colours = ["tab:blue", "black", "tab:red"]

        for idx in range(len(x)):
            ax.scatter([x[idx]], [y[idx]], color=colours[idx], s=40)
        ax.errorbar([x[0], x[1]], y[:2], yerr=[yerr_low[:2], yerr_high[:2]], fmt="none", ecolor="0.4", capsize=3)
        ax.axhspan(float(row["age_min_myr"]), float(row["age_max_myr"]), color="0.9", zorder=0)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["baseline", "bootstrap", "extreme A_V"], rotation=20)
        ax.set_title(CLUSTERS[cluster_key]["name"])
        ax.set_ylabel("Age (Myr)")

    fig.savefig(PLOT_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    distances = load_distance_map()
    sensitivity_df = load_sensitivity_table()
    isochrones = read_mist(ISO_FEH_BASELINE)

    summary_rows: list[dict[str, float | int | str | bool]] = []
    bootstrap_map: dict[str, np.ndarray] = {}

    for cluster_index, cluster_key in enumerate(CLUSTERS.keys()):
        distance_pc = float(distances[cluster_key])
        a_v = float(ISO_AV[cluster_key])
        a_g = float(ISO_AG_OVER_AV * a_v)
        e_bprp = float(ISO_EBPRP_OVER_AV * a_v)
        dm = distance_modulus(distance_pc)
        members = prepare_members(load_members(cluster_key), dm, a_g)

        baseline_cycle, one_fit_seconds = time_one_fit(
            members,
            isochrones,
            dm,
            a_g,
            e_bprp,
            literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
        )
        use_vectorized_resampling = one_fit_seconds > 1.0
        mode_text = "vectorized resampling" if use_vectorized_resampling else "simple resampling"
        print(f"{cluster_key}: one baseline fit took {one_fit_seconds:.3f} s; using {mode_text} for bootstrap")

        bootstrap_ages = bootstrap_parabola_ages(
            members,
            isochrones,
            dm,
            a_g,
            e_bprp,
            literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
            seed=int(ISO_SEED + cluster_index),
            n_boot=int(ISO_BOOTSTRAP_N),
            use_vectorized_resampling=use_vectorized_resampling,
        )
        bootstrap_map[cluster_key] = bootstrap_ages

        age_min, age_max, age_extreme, alt_feh_age = extract_sensitivity_values(sensitivity_df, cluster_key)
        summary_rows.append(
            make_summary_row(
                cluster_key,
                distance_pc,
                a_v,
                baseline_cycle,
                age_min,
                age_max,
                age_extreme,
                alt_feh_age,
                bootstrap_ages,
            )
        )

    summary_df = pd.DataFrame(summary_rows)[
        [
            "cluster",
            "feh",
            "av",
            "distance_pc",
            "colour_offset",
            "best_age_myr",
            "age_min_myr",
            "age_max_myr",
            "age_extreme_av_myr",
            "boot_median_myr",
            "boot_p16_myr",
            "boot_p84_myr",
            "n_stars_fit",
            "loss_ratio_minus02",
            "loss_ratio_plus02",
            "age_constrained",
            "age_alt_feh015_myr",
        ]
    ]
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    order_rows = make_order_rows(summary_df, bootstrap_map)
    order_df = pd.DataFrame(order_rows)[
        ["first_cluster", "second_cluster", "plausible_ranges_overlap", "frac_bootstrap_first_younger"]
    ]
    order_df.to_csv(ORDER_OUTPUT_PATH, index=False)

    save_summary_plot(summary_df)

    print("\nIsochrone summary:")
    print(summary_df.to_string(index=False))
    print("\nIsochrone order:")
    print(order_df.to_string(index=False))


if __name__ == "__main__":
    main()
