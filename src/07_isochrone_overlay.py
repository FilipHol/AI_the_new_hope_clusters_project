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
    ISO_EBPRP_OVER_AV,
    ISO_LIT_AGE_MYR,
    OUT_DIR,
)
from iso_lib import build_mist_segments, logage_to_myr, nearest_mist_logage, read_mist, select_mist_logage


SUMMARY_PATH = OUT_DIR / "summary.csv"
POINT_SIZE = 4
LINE_COLOUR = "red"
MEMBER_COLOUR = "0.65"


def load_members(cluster_key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{cluster_key}_members.parquet")


def load_distance_map() -> dict[str, float]:
    summary = pd.read_csv(SUMMARY_PATH)
    if "cluster_key" not in summary.columns or "median_distance_pc_zp017" not in summary.columns:
        raise ValueError("summary.csv is missing required columns: cluster_key, median_distance_pc_zp017")
    return dict(zip(summary["cluster_key"], summary["median_distance_pc_zp017"]))


def prepare_members(frame: pd.DataFrame) -> pd.DataFrame:
    members = frame[
        frame["phot_g_mean_mag"].notna()
        & frame["phot_bp_mean_mag"].notna()
        & frame["phot_rp_mean_mag"].notna()
    ].copy()
    members["colour"] = members["phot_bp_mean_mag"] - members["phot_rp_mean_mag"]
    return members


def distance_modulus(distance_pc: float) -> float:
    return float(5.0 * np.log10(distance_pc) - 5.0)


def lookup_grid_logages(isochrones: pd.DataFrame, literature_age_myr: float) -> dict[str, float]:
    target_logage = float(np.log10(literature_age_myr * 1.0e6))
    return {
        "younger": nearest_mist_logage(isochrones, target_logage - 0.2),
        "central": nearest_mist_logage(isochrones, target_logage),
        "older": nearest_mist_logage(isochrones, target_logage + 0.2),
    }


def transform_segment(segment: pd.DataFrame, dm: float, a_g: float, e_bprp: float) -> pd.DataFrame:
    transformed = segment.copy()
    transformed["g_model"] = transformed["g"] + dm + a_g
    transformed["colour_model"] = (transformed["bp"] - transformed["rp"]) + e_bprp
    return transformed


def plot_segments(
    ax: plt.Axes,
    segments: list[tuple[str, pd.DataFrame]],
    dm: float,
    a_g: float,
    e_bprp: float,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    first = True
    for _, segment in segments:
        transformed = transform_segment(segment, dm, a_g, e_bprp)
        ax.plot(
            transformed["colour_model"],
            transformed["g_model"],
            color=LINE_COLOUR,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if first else None,
        )
        first = False


def median_absolute_deviation(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def delta_mode(values: np.ndarray, bin_width: float = 0.01) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute mode for an empty array")
    if values.size == 1:
        return float(values[0])

    lower = float(np.floor(values.min() / bin_width) * bin_width)
    upper = float(np.ceil(values.max() / bin_width) * bin_width)
    if np.isclose(lower, upper):
        upper = lower + bin_width
    edges = np.arange(lower, upper + bin_width, bin_width)
    if edges.size < 2:
        edges = np.array([lower, lower + bin_width])
    counts, edges = np.histogram(values, bins=edges)
    peak_index = int(np.argmax(counts))
    return float(0.5 * (edges[peak_index] + edges[peak_index + 1]))


def prepare_colour_diagnostic_data(
    members: pd.DataFrame,
    central_isochrone: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
) -> pd.DataFrame:
    phase0 = central_isochrone.loc[central_isochrone["phase"] == 0].sort_values("eep")
    if phase0.empty:
        raise ValueError("Phase-0 isochrone segment is empty")

    transformed = transform_segment(phase0, dm, a_g, e_bprp).sort_values(["g_model", "eep"])
    unique_g, unique_indices = np.unique(transformed["g_model"].to_numpy(), return_index=True)
    unique_colour = transformed["colour_model"].to_numpy()[unique_indices]
    if unique_g.size < 2:
        raise ValueError("Phase-0 isochrone has fewer than two distinct G values")

    diagnostic_members = members.copy()
    diagnostic_members["M_G"] = diagnostic_members["phot_g_mean_mag"] - dm - a_g
    diagnostic_members = diagnostic_members[
        diagnostic_members["M_G"].gt(4.0)
        & diagnostic_members["M_G"].lt(7.0)
        & diagnostic_members["phot_g_mean_mag"].between(float(unique_g.min()), float(unique_g.max()))
    ].copy()
    if diagnostic_members.empty:
        raise ValueError("No members available for the 4 < M_G < 7 colour diagnostic")

    diagnostic_members["iso_colour"] = np.interp(
        diagnostic_members["phot_g_mean_mag"],
        unique_g,
        unique_colour,
    )
    diagnostic_members["delta_colour"] = diagnostic_members["colour"] - diagnostic_members["iso_colour"]
    return diagnostic_members


def summarise_delta(values: np.ndarray) -> tuple[int, float, float, float]:
    if values.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    median = float(np.median(values))
    mad = median_absolute_deviation(values)
    mode = delta_mode(values)
    return int(values.size), median, mad, mode


def summarise_trimmed_delta(values: np.ndarray, trim_fraction: float = 0.2) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("No values available for trimmed summary")
    overall = float(np.median(values))
    keep_count = max(1, int(np.ceil((1.0 - trim_fraction) * values.size)))
    trimmed = np.sort(values)[:keep_count]
    trimmed_median = float(np.median(trimmed))
    return overall, trimmed_median


def save_colour_offset_plot(cluster_key: str, diagnostic_members: pd.DataFrame, bin_summaries: list[dict[str, float]]) -> None:
    output_path = OUT_DIR / f"colour_offset_{cluster_key}.png"
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        diagnostic_members["delta_colour"],
        diagnostic_members["M_G"],
        s=POINT_SIZE,
        c=MEMBER_COLOUR,
        linewidths=0,
    )
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0)

    for summary in bin_summaries:
        if summary["n"] == 0:
            continue
        mid_mg = 0.5 * (summary["mg_min"] + summary["mg_max"])
        ax.scatter(summary["median"], mid_mg, s=90, c=LINE_COLOUR, marker="o", zorder=3)

    ax.set_title(f"{cluster_key} colour offset")
    ax.set_xlabel("delta(BP-RP)")
    ax.set_ylabel("M_G")
    ax.set_ylim(7.0, 4.0)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cluster_plot(
    cluster_key: str,
    cluster_name: str,
    members: pd.DataFrame,
    age_frames: dict[str, pd.DataFrame],
    grid_logages: dict[str, float],
    dm: float,
    a_g: float,
    e_bprp: float,
) -> None:
    output_path = OUT_DIR / f"isochrone_overlay_{cluster_key}.png"
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(members["colour"], members["phot_g_mean_mag"], s=POINT_SIZE, c=MEMBER_COLOUR, linewidths=0)

    plot_segments(
        ax,
        build_mist_segments(age_frames["younger"], ISO_ALLOWED_PHASES),
        dm,
        a_g,
        e_bprp,
        linestyle="--",
        linewidth=1.5,
        label=f"younger ({logage_to_myr(grid_logages['younger']):.1f} Myr)",
    )
    plot_segments(
        ax,
        build_mist_segments(age_frames["central"], ISO_ALLOWED_PHASES),
        dm,
        a_g,
        e_bprp,
        linestyle="-",
        linewidth=2.0,
        label=f"literature ({logage_to_myr(grid_logages['central']):.1f} Myr)",
    )
    plot_segments(
        ax,
        build_mist_segments(age_frames["older"], ISO_ALLOWED_PHASES),
        dm,
        a_g,
        e_bprp,
        linestyle="--",
        linewidth=1.5,
        label=f"older ({logage_to_myr(grid_logages['older']):.1f} Myr)",
    )

    ax.set_title(f"{cluster_name} isochrone overlay (grid age {logage_to_myr(grid_logages['central']):.1f} Myr)")
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G")
    ax.set_xlim(float(members["colour"].min()), float(members["colour"].max()))
    ax.set_ylim(float(members["phot_g_mean_mag"].max()), float(members["phot_g_mean_mag"].min()))
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    distances = load_distance_map()
    isochrones = read_mist(0.00)

    for cluster_key, cluster in CLUSTERS.items():
        members = prepare_members(load_members(cluster_key))
        distance_pc = float(distances[cluster_key])
        dm = distance_modulus(distance_pc)
        a_v = float(ISO_AV[cluster_key])
        a_g = float(ISO_AG_OVER_AV * a_v)
        e_bprp = float(ISO_EBPRP_OVER_AV * a_v)

        grid_logages = lookup_grid_logages(isochrones, float(ISO_LIT_AGE_MYR[cluster_key]))
        age_frames = {name: select_mist_logage(isochrones, logage) for name, logage in grid_logages.items()}

        save_cluster_plot(
            cluster_key,
            str(cluster["name"]),
            members,
            age_frames,
            grid_logages,
            dm,
            a_g,
            e_bprp,
        )

        diagnostic_members = prepare_colour_diagnostic_data(
            members,
            age_frames["central"],
            dm,
            a_g,
            e_bprp,
        )

        bin_edges = [(4.0, 5.0), (5.0, 6.0), (6.0, 7.0)]
        bin_summaries: list[dict[str, float]] = []
        for mg_min, mg_max in bin_edges:
            in_bin = diagnostic_members[
                diagnostic_members["M_G"].ge(mg_min) & diagnostic_members["M_G"].lt(mg_max)
            ]
            n_bin, median_bin, mad_bin, mode_bin = summarise_delta(in_bin["delta_colour"].to_numpy())
            bin_summaries.append(
                {
                    "mg_min": mg_min,
                    "mg_max": mg_max,
                    "n": n_bin,
                    "median": median_bin,
                    "mad": mad_bin,
                    "mode": mode_bin,
                }
            )

        overall_median, trimmed_median = summarise_trimmed_delta(
            diagnostic_members["delta_colour"].to_numpy(),
            trim_fraction=0.2,
        )
        save_colour_offset_plot(cluster_key, diagnostic_members, bin_summaries)

        print(f"{cluster['name']} ({cluster_key}):")
        print(f"  distance_pc={distance_pc:.3f}")
        print(f"  DM={dm:.3f}")
        print(f"  A_G={a_g:.3f}")
        print(f"  E(BP-RP)={e_bprp:.3f}")
        print(
            f"  grid_ages_myr: younger={logage_to_myr(grid_logages['younger']):.3f}, "
            f"central={logage_to_myr(grid_logages['central']):.3f}, "
            f"older={logage_to_myr(grid_logages['older']):.3f}"
        )
        print("  delta sign convention: member colour minus isochrone colour at the same G (positive = member redder)")
        for summary in bin_summaries:
            print(
                f"  M_G {summary['mg_min']:.0f}-{summary['mg_max']:.0f}: "
                f"n={int(summary['n'])}, median={summary['median']:.4f}, "
                f"MAD={summary['mad']:.4f}, mode={summary['mode']:.4f}"
            )
        print(
            f"  overall 4<M_G<7: median={overall_median:.4f}, "
            f"median_without_top20pct_delta={trimmed_median:.4f}"
        )


if __name__ == "__main__":
    main()
