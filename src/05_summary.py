import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLUSTERS, DATA_DIR, OUT_DIR


BIN_WIDTH = 0.25
MG_MIN = -1.0
MG_MAX = 6.0
MIN_STARS_PER_BIN = 5
BOOTSTRAP_SAMPLES = 200
RANDOM_SEED = 12345
PARALLAX_ZEROPOINT_MAS = 0.017
X_LIMITS = (-0.5, 4.0)
Y_LIMITS = (14.0, -4.0)
EXPECTED_ORDER = ["Pleiades", "Praesepe", "M67"]
BOOTSTRAP_ORDER_PATH = OUT_DIR / "bootstrap_order.csv"


def load_members(cluster_key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{cluster_key}_members.parquet")


def prepare_hr_data(frame: pd.DataFrame) -> pd.DataFrame:
    hr_frame = frame[
        frame["parallax"].gt(0)
        & frame["phot_g_mean_mag"].notna()
        & frame["phot_bp_mean_mag"].notna()
        & frame["phot_rp_mean_mag"].notna()
    ].copy()
    hr_frame["bp_rp"] = hr_frame["phot_bp_mean_mag"] - hr_frame["phot_rp_mean_mag"]
    hr_frame["M_G"] = hr_frame["phot_g_mean_mag"] + 5.0 * np.log10(hr_frame["parallax"] / 100.0)
    return hr_frame


def find_turnoff(hr_frame: pd.DataFrame) -> tuple[float, float]:
    turnoff_frame = hr_frame[hr_frame["M_G"].gt(MG_MIN) & hr_frame["M_G"].lt(MG_MAX)].copy()
    if turnoff_frame.empty:
        raise RuntimeError("No stars available in the turn-off magnitude range.")

    bins = np.arange(MG_MIN, MG_MAX + BIN_WIDTH, BIN_WIDTH)
    turnoff_frame["mg_bin"] = pd.cut(turnoff_frame["M_G"], bins=bins, right=False)

    grouped = (
        turnoff_frame.groupby("mg_bin", observed=True)["bp_rp"]
        .agg(["count", "median"])
        .reset_index()
    )
    grouped = grouped[grouped["count"] >= MIN_STARS_PER_BIN].copy()
    if grouped.empty:
        raise RuntimeError("No turn-off bins have at least 5 stars.")

    grouped["bin_center_mg"] = grouped["mg_bin"].apply(lambda interval: float(interval.mid))
    best_row = grouped.loc[grouped["median"].idxmin()]
    return float(best_row["bin_center_mg"]), float(best_row["median"])


def find_fifth_brightest_main_sequence_mg(
    hr_frame: pd.DataFrame,
    cluster_key: str,
) -> float:
    if cluster_key == "m67":
        main_sequence = hr_frame[(hr_frame["bp_rp"] < 0.95) & (hr_frame["M_G"] > 2.5)].copy()
    else:
        main_sequence = hr_frame[hr_frame["bp_rp"] < 1.0].copy()

    if len(main_sequence) < 5:
        raise RuntimeError(f"Not enough main-sequence stars to find the 5th-brightest star for {cluster_key}.")

    return float(main_sequence["M_G"].nsmallest(5).iloc[-1])


def bootstrap_metrics(
    hr_frame: pd.DataFrame,
    cluster_key: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    samples = []
    sample_size = len(hr_frame)
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_indices = rng.integers(0, sample_size, size=sample_size)
        sample = hr_frame.iloc[sample_indices].copy()
        turnoff_mg, turnoff_bp_rp = find_turnoff(sample)
        fifth_brightest_mg = find_fifth_brightest_main_sequence_mg(sample, cluster_key)
        samples.append(
            {
                "turnoff_M_G": turnoff_mg,
                "turnoff_bp_rp": turnoff_bp_rp,
                "fifth_brightest_ms_M_G": fifth_brightest_mg,
            }
        )
    return pd.DataFrame(samples)


def build_summary_row(
    cluster_key: str,
    cluster: dict[str, float | str],
    rng: np.random.Generator,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    members = load_members(cluster_key)
    hr_frame = prepare_hr_data(members)
    turnoff_mg, turnoff_bp_rp = find_turnoff(hr_frame)
    fifth_brightest_mg = find_fifth_brightest_main_sequence_mg(hr_frame, cluster_key)
    bootstrap_df = bootstrap_metrics(hr_frame, cluster_key, rng)
    summary = {
        "cluster": str(cluster["name"]),
        "cluster_key": cluster_key,
        "n_members": int(len(members)),
        "median_distance_pc": float((1000.0 / members["parallax"]).median()),
        "median_distance_pc_zp017": float((1000.0 / (members["parallax"] + PARALLAX_ZEROPOINT_MAS)).median()),
        "turnoff_M_G": turnoff_mg,
        "turnoff_bp_rp": turnoff_bp_rp,
        "bootstrap_turnoff_M_G_median": float(bootstrap_df["turnoff_M_G"].median()),
        "bootstrap_turnoff_M_G_std": float(bootstrap_df["turnoff_M_G"].std(ddof=1)),
        "bootstrap_turnoff_bp_rp_median": float(bootstrap_df["turnoff_bp_rp"].median()),
        "bootstrap_turnoff_bp_rp_std": float(bootstrap_df["turnoff_bp_rp"].std(ddof=1)),
        "fifth_brightest_ms_M_G": fifth_brightest_mg,
        "bootstrap_fifth_brightest_ms_M_G_median": float(bootstrap_df["fifth_brightest_ms_M_G"].median()),
        "bootstrap_fifth_brightest_ms_M_G_std": float(bootstrap_df["fifth_brightest_ms_M_G"].std(ddof=1)),
    }
    return summary, hr_frame, bootstrap_df


def save_summary_csv(summary_df: pd.DataFrame) -> None:
    output_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(output_path, index=False)


def build_bootstrap_order_table(bootstrap_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pleiades = bootstrap_results["pleiades"]
    praesepe = bootstrap_results["praesepe"]
    m67 = bootstrap_results["m67"]

    colour_all = (
        (pleiades["turnoff_bp_rp"] < praesepe["turnoff_bp_rp"])
        & (praesepe["turnoff_bp_rp"] < m67["turnoff_bp_rp"])
    )
    luminosity_all = (
        (pleiades["fifth_brightest_ms_M_G"] < praesepe["fifth_brightest_ms_M_G"])
        & (praesepe["fifth_brightest_ms_M_G"] < m67["fifth_brightest_ms_M_G"])
    )

    m67_blue = m67["turnoff_bp_rp"] < 0.5
    m67_blue_median_mg = float(m67.loc[m67_blue, "turnoff_M_G"].median()) if m67_blue.any() else np.nan

    rows = [
        {
            "measure": "colour_turnoff_bp_rp",
            "fraction_pleiades_lt_praesepe": float((pleiades["turnoff_bp_rp"] < praesepe["turnoff_bp_rp"]).mean()),
            "fraction_praesepe_lt_m67": float((praesepe["turnoff_bp_rp"] < m67["turnoff_bp_rp"]).mean()),
            "fraction_pleiades_lt_m67": float((pleiades["turnoff_bp_rp"] < m67["turnoff_bp_rp"]).mean()),
            "fraction_all_three_hold": float(colour_all.mean()),
            "m67_fraction_bp_rp_lt_0p5": float(m67_blue.mean()),
            "m67_median_turnoff_M_G_when_bp_rp_lt_0p5": m67_blue_median_mg,
        },
        {
            "measure": "fifth_brightest_ms_M_G",
            "fraction_pleiades_lt_praesepe": float((pleiades["fifth_brightest_ms_M_G"] < praesepe["fifth_brightest_ms_M_G"]).mean()),
            "fraction_praesepe_lt_m67": float((praesepe["fifth_brightest_ms_M_G"] < m67["fifth_brightest_ms_M_G"]).mean()),
            "fraction_pleiades_lt_m67": float((pleiades["fifth_brightest_ms_M_G"] < m67["fifth_brightest_ms_M_G"]).mean()),
            "fraction_all_three_hold": float(luminosity_all.mean()),
            "m67_fraction_bp_rp_lt_0p5": np.nan,
            "m67_median_turnoff_M_G_when_bp_rp_lt_0p5": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def save_bootstrap_order_csv(order_df: pd.DataFrame) -> None:
    order_df.to_csv(BOOTSTRAP_ORDER_PATH, index=False)


def print_bootstrap_order_table(order_df: pd.DataFrame) -> None:
    print("Bootstrap order fractions:")
    print(order_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("M_G")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)


def save_turnoff_plot(hr_frames: dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, constrained_layout=True)

    for ax, (cluster_key, cluster) in zip(axes, CLUSTERS.items(), strict=False):
        hr_frame = hr_frames[cluster_key]
        row = summary_df.loc[summary_df["cluster_key"] == cluster_key].iloc[0]
        ax.scatter(hr_frame["bp_rp"], hr_frame["M_G"], s=4, c="black", linewidths=0)
        ax.scatter(
            row["turnoff_bp_rp"],
            row["turnoff_M_G"],
            s=80,
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
        )
        style_axes(ax, str(cluster["name"]))

    fig.savefig(OUT_DIR / "hr_turnoff.png", dpi=200)
    plt.close(fig)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    print(
        "cluster      N members  median distance (pc)  zp+0.017 mas (pc)  "
        "turn-off M_G  turn-off BP-RP  boot M_G med±std  boot BP-RP med±std  "
        "5th-brightest M_G  boot 5th M_G med±std"
    )
    for row in summary_df.itertuples(index=False):
        print(
            f"{row.cluster:<12}"
            f"{row.n_members:>10}"
            f"{row.median_distance_pc:>22.1f}"
            f"{row.median_distance_pc_zp017:>19.1f}"
            f"{row.turnoff_M_G:>14.3f}"
            f"{row.turnoff_bp_rp:>16.3f}"
            f"{row.bootstrap_turnoff_M_G_median:>10.3f}±{row.bootstrap_turnoff_M_G_std:<6.3f}"
            f"{row.bootstrap_turnoff_bp_rp_median:>11.3f}±{row.bootstrap_turnoff_bp_rp_std:<6.3f}"
            f"{row.fifth_brightest_ms_M_G:>18.3f}"
            f"{row.bootstrap_fifth_brightest_ms_M_G_median:>11.3f}±{row.bootstrap_fifth_brightest_ms_M_G_std:<6.3f}"
        )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    hr_frames: dict[str, pd.DataFrame] = {}
    bootstrap_results: dict[str, pd.DataFrame] = {}
    for cluster_key, cluster in CLUSTERS.items():
        summary_row, hr_frame, bootstrap_df = build_summary_row(cluster_key, cluster, rng)
        summary_rows.append(summary_row)
        hr_frames[cluster_key] = hr_frame
        bootstrap_results[cluster_key] = bootstrap_df

    summary_df = pd.DataFrame(summary_rows)
    sorted_df = summary_df.sort_values("turnoff_bp_rp", ascending=True).reset_index(drop=True)

    bootstrap_order_df = build_bootstrap_order_table(bootstrap_results)

    print_summary_table(sorted_df)
    print_bootstrap_order_table(bootstrap_order_df)

    sorted_order = sorted_df["cluster"].tolist()
    print("Youngest to oldest:", " < ".join(sorted_order))
    print("Matches expected order:", sorted_order == EXPECTED_ORDER)
    print(
        "Bootstrap fraction with colour order Pleiades < Praesepe < M67:",
        f"{bootstrap_order_df.loc[bootstrap_order_df['measure'] == 'colour_turnoff_bp_rp', 'fraction_all_three_hold'].iloc[0]:.3f}",
    )
    print(
        "Bootstrap fraction with 5th-brightest M_G order Pleiades < Praesepe < M67:",
        f"{bootstrap_order_df.loc[bootstrap_order_df['measure'] == 'fifth_brightest_ms_M_G', 'fraction_all_three_hold'].iloc[0]:.3f}",
    )

    save_summary_csv(sorted_df)
    save_bootstrap_order_csv(bootstrap_order_df)
    save_turnoff_plot(hr_frames, summary_df)


if __name__ == "__main__":
    main()
