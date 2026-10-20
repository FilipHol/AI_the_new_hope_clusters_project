import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLUSTERS, DATA_DIR, OUT_DIR


X_LIMITS = (-0.5, 4.0)
Y_LIMITS = (14.0, -4.0)
POINT_SIZE = 4


def load_members(cluster_key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{cluster_key}_members.parquet")


def prepare_hr_data(frame: pd.DataFrame) -> pd.DataFrame:
    hr_frame = frame[
        frame["parallax"].gt(0)
        & frame["phot_g_mean_mag"].notna()
        & frame["phot_bp_mean_mag"].notna()
        & frame["phot_rp_mean_mag"].notna()
    ].copy()
    hr_frame["colour"] = hr_frame["phot_bp_mean_mag"] - hr_frame["phot_rp_mean_mag"]
    hr_frame["M_G"] = hr_frame["phot_g_mean_mag"] + 5.0 * np.log10(hr_frame["parallax"] / 100.0)
    return hr_frame


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("M_G")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)


def save_single_plot(cluster_key: str, cluster_name: str, hr_frame: pd.DataFrame) -> None:
    output_path = OUT_DIR / f"hr_{cluster_key}.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(hr_frame["colour"], hr_frame["M_G"], s=POINT_SIZE, c="black", linewidths=0)
    style_axes(ax, cluster_name)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_combined_plot(hr_frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, constrained_layout=True)

    for ax, (cluster_key, cluster) in zip(axes, CLUSTERS.items(), strict=False):
        hr_frame = hr_frames[cluster_key]
        ax.scatter(hr_frame["colour"], hr_frame["M_G"], s=POINT_SIZE, c="black", linewidths=0)
        style_axes(ax, str(cluster["name"]))

    fig.savefig(OUT_DIR / "hr_all.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    hr_frames: dict[str, pd.DataFrame] = {}
    for cluster_key, cluster in CLUSTERS.items():
        members = load_members(cluster_key)
        hr_frame = prepare_hr_data(members)
        hr_frames[cluster_key] = hr_frame
        save_single_plot(cluster_key, str(cluster["name"]), hr_frame)
        print(f"{cluster['name']} ({cluster_key}): stars_plotted={len(hr_frame)}")

    save_combined_plot(hr_frames)


if __name__ == "__main__":
    main()
