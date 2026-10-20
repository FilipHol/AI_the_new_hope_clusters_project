from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLUSTERS, DATA_DIR, ISO_AG_OVER_AV, ISO_ALLOWED_PHASES, ISO_AV, ISO_EBPRP_OVER_AV, ISO_FEH_BASELINE, ISO_LIT_AGE_MYR, OUT_DIR
from iso_lib import build_mist_observed_segments, distance_modulus, logage_to_myr, prepare_members, read_mist, run_isochrone_fit_cycle


SUMMARY_PATH = OUT_DIR / "summary.csv"
SUMMARY_OUTPUT_PATH = OUT_DIR / "isochrone_fit_baseline.csv"
POINT_SIZE = 4
ALL_MEMBER_COLOUR = "0.75"
FIT_MEMBER_COLOUR = "0.35"
PHASE_02_COLOUR = "red"
PHASE_3_COLOUR = "blue"


def load_members(cluster_key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{cluster_key}_members.parquet")


def load_distance_map() -> dict[str, float]:
    summary = pd.read_csv(SUMMARY_PATH)
    if "cluster_key" not in summary.columns or "median_distance_pc_zp017" not in summary.columns:
        raise ValueError("summary.csv is missing required columns: cluster_key, median_distance_pc_zp017")
    return dict(zip(summary["cluster_key"], summary["median_distance_pc_zp017"]))


def save_loss_plot(cluster_key: str, cluster_name: str, fit_result: dict[str, object]) -> None:
    output_path = OUT_DIR / f"isochrone_loss_{cluster_key}.png"
    logage_grid = np.asarray(fit_result["logage_grid"], dtype=float)
    losses = np.asarray(fit_result["losses"], dtype=float)
    ages_myr = np.array([logage_to_myr(value) for value in logage_grid], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ages_myr, losses, color="black", linewidth=1.5)
    ax.scatter([float(fit_result["best_age_myr"])], [float(fit_result["best_loss"])], color=PHASE_02_COLOUR, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel("Loss")
    ax.set_title(f"{cluster_name} isochrone loss")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_fit_plot(
    cluster_key: str,
    cluster_name: str,
    members: pd.DataFrame,
    fit_members: pd.DataFrame,
    isochrone: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float,
    fit_result: dict[str, object],
) -> None:
    output_path = OUT_DIR / f"isochrone_fit_{cluster_key}.png"
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(members["colour"], members["phot_g_mean_mag"], s=POINT_SIZE, c=ALL_MEMBER_COLOUR, linewidths=0)
    ax.scatter(fit_members["colour"], fit_members["phot_g_mean_mag"], s=POINT_SIZE, c=FIT_MEMBER_COLOUR, linewidths=0)

    segments = build_mist_observed_segments(
        isochrone,
        ISO_ALLOWED_PHASES,
        dm,
        a_g,
        e_bprp,
        colour_offset=colour_offset,
    )
    for segment_name, segment in segments:
        if segment_name == "phase_3":
            colour = PHASE_3_COLOUR
            label = "phase 3"
        else:
            colour = PHASE_02_COLOUR
            label = "phases 0/2"
        ax.plot(
            segment["colour_model"],
            segment["g_model"],
            color=colour,
            linewidth=2.0,
            label=label,
        )

    y_min = float(min(members["phot_g_mean_mag"].min(), fit_members["phot_g_mean_mag"].min()))
    y_max = float(fit_members["phot_g_mean_mag"].max())
    y_margin = max(0.2, 0.05 * (y_max - y_min))

    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G")
    ax.set_title(
        f"{cluster_name} fit (grid {float(fit_result['best_grid_age_myr']):.1f} Myr, "
        f"parabola {float(fit_result['best_age_myr']):.1f} Myr)"
    )
    ax.set_xlim(float(members["colour"].min()), float(members["colour"].max()))
    ax.set_ylim(y_max + y_margin, y_min - y_margin)
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    distances = load_distance_map()
    isochrones = read_mist(ISO_FEH_BASELINE)
    summary_rows: list[dict[str, float | int | str]] = []

    for cluster_key, cluster in CLUSTERS.items():
        distance_pc = float(distances[cluster_key])
        dm = distance_modulus(distance_pc)
        a_v = float(ISO_AV[cluster_key])
        a_g = float(ISO_AG_OVER_AV * a_v)
        e_bprp = float(ISO_EBPRP_OVER_AV * a_v)

        members = prepare_members(load_members(cluster_key), dm, a_g)
        cycle = run_isochrone_fit_cycle(
            members,
            isochrones,
            dm,
            a_g,
            e_bprp,
            literature_age_myr=float(ISO_LIT_AGE_MYR[cluster_key]),
        )
        fit_members = cycle["fit_members"]
        fit_1 = cycle["fit_1"]
        fit_2 = cycle["fit_2"]
        colour_offset_1 = float(cycle["colour_offset_1"])
        colour_offset_2 = float(cycle["colour_offset_2"])
        n_offset_2 = int(cycle["n_offset_2"])
        dropped_2 = int(cycle["dropped_2"])
        best_grid_isochrone_2 = cycle["best_grid_isochrone_2"]

        save_loss_plot(cluster_key, str(cluster["name"]), fit_2)
        save_fit_plot(
            cluster_key,
            str(cluster["name"]),
            members,
            fit_members,
            best_grid_isochrone_2,
            dm,
            a_g,
            e_bprp,
            colour_offset_2,
            fit_2,
        )

        print(f"{cluster['name']} ({cluster_key}):")
        print(f"  n_fit_stars={len(fit_members)}")
        print(
            f"  colour_offset={colour_offset_2:.4f} "
            f"(offset stars {n_offset_2}, dropped {dropped_2}; first pass {colour_offset_1:.4f})"
        )
        print(
            f"  first_fit_age_myr: grid={float(fit_1['best_grid_age_myr']):.3f}, "
            f"parabola={float(fit_1['best_age_myr']):.3f}"
        )
        print(
            f"  refit_age_myr: grid={float(fit_2['best_grid_age_myr']):.3f}, "
            f"parabola={float(fit_2['best_age_myr']):.3f}"
        )
        if fit_1["warning"] is not None:
            print(f"  warning first fit: {fit_1['warning']}")
        if fit_2["warning"] is not None:
            print(f"  warning refit: {fit_2['warning']}")
        print(
            f"  loss_min={float(fit_2['best_loss']):.3f}, "
            f"loss_minus_0.2dex={float(fit_2['loss_minus_0p2']):.3f}, "
            f"loss_plus_0.2dex={float(fit_2['loss_plus_0p2']):.3f}"
        )
        print(f"  frac_capped={float(fit_2['frac_capped'][int(fit_2['best_index'])]):.4f}")

        summary_rows.append(
            {
                "cluster": cluster_key,
                "colour_offset": float(colour_offset_2),
                "best_age_myr": float(fit_2["best_age_myr"]),
                "n_stars_fit": int(len(fit_members)),
                "frac_capped": float(fit_2["frac_capped"][int(fit_2["best_index"])]),
                "loss_min": float(fit_2["best_loss"]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
