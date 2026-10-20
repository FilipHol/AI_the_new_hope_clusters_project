import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

from config import CLUSTERS, DATA_DIR, OUT_DIR


WINDOW_HALF_WIDTH = 40.0
BINS = 300
OUTPUT_PATH = OUT_DIR / "pm_all.png"


def load_cluster_frame(cluster_key: str) -> pd.DataFrame:
    data_path = DATA_DIR / f"{cluster_key}.parquet"
    return pd.read_parquet(data_path)


def count_stars_in_window(frame: pd.DataFrame, cluster: dict[str, float | str]) -> int:
    pmra_center = float(cluster["expected_pmra"])
    pmdec_center = float(cluster["expected_pmdec"])
    in_window = (
        frame["pmra"].between(pmra_center - WINDOW_HALF_WIDTH, pmra_center + WINDOW_HALF_WIDTH)
        & frame["pmdec"].between(pmdec_center - WINDOW_HALF_WIDTH, pmdec_center + WINDOW_HALF_WIDTH)
    )
    return int(in_window.sum())


def plot_cluster(ax: plt.Axes, frame: pd.DataFrame, cluster: dict[str, float | str]) -> int:
    pmra_center = float(cluster["expected_pmra"])
    pmdec_center = float(cluster["expected_pmdec"])
    x_min = pmra_center - WINDOW_HALF_WIDTH
    x_max = pmra_center + WINDOW_HALF_WIDTH
    y_min = pmdec_center - WINDOW_HALF_WIDTH
    y_max = pmdec_center + WINDOW_HALF_WIDTH

    window_frame = frame[
        frame["pmra"].between(x_min, x_max) & frame["pmdec"].between(y_min, y_max)
    ].copy()

    ax.hist2d(
        window_frame["pmra"],
        window_frame["pmdec"],
        bins=BINS,
        range=[[x_min, x_max], [y_min, y_max]],
        norm=LogNorm(),
        cmap="viridis",
    )
    ax.plot(pmra_center, pmdec_center, marker="x", color="red", markersize=6, mew=1.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(str(cluster["name"]))
    ax.set_xlabel("pmra (mas/yr)")
    ax.set_ylabel("pmdec (mas/yr)")

    return len(window_frame)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    counts: dict[str, int] = {}
    for ax, (cluster_key, cluster) in zip(axes, CLUSTERS.items(), strict=False):
        frame = load_cluster_frame(cluster_key)
        counts[cluster_key] = plot_cluster(ax, frame, cluster)

    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {OUTPUT_PATH.name}")
    for cluster_key, cluster in CLUSTERS.items():
        print(f"{cluster['name']} ({cluster_key}): stars_in_window={counts[cluster_key]}")


if __name__ == "__main__":
    main()
