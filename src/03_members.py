import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from config import CLUSTERS, DATA_DIR, OUT_DIR


FEATURE_COLUMNS = ["pmra", "pmdec", "parallax"]
MIN_CLUSTER_SIZE = 50
MIN_SAMPLES = 10
PARALLAX_FRACTION = 0.3


def expected_parallax_mas(cluster: dict[str, float | str]) -> float:
    return 1000.0 / float(cluster["expected_distance_pc"])


def load_cluster_frame(cluster_key: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{cluster_key}.parquet")


def preselect_subset(frame: pd.DataFrame, cluster: dict[str, float | str]) -> pd.DataFrame:
    pmra_center = float(cluster["expected_pmra"])
    pmdec_center = float(cluster["expected_pmdec"])
    pm_window = float(cluster["pm_window_masyr"])
    parallax_center = expected_parallax_mas(cluster)
    parallax_window = PARALLAX_FRACTION * parallax_center

    subset = frame[
        frame["pmra"].notna()
        & frame["pmdec"].notna()
        & frame["parallax"].notna()
        & frame["pmra"].between(pmra_center - pm_window, pmra_center + pm_window)
        & frame["pmdec"].between(pmdec_center - pm_window, pmdec_center + pm_window)
        & frame["parallax"].between(
            parallax_center - parallax_window,
            parallax_center + parallax_window,
        )
    ].copy()
    return subset


def find_member_label(subset: pd.DataFrame, cluster: dict[str, float | str]) -> int:
    if len(subset) < MIN_CLUSTER_SIZE:
        raise RuntimeError(
            f"Subset for {cluster['name']} has only {len(subset)} rows; need at least {MIN_CLUSTER_SIZE}."
        )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(subset[FEATURE_COLUMNS])
    labels = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        copy=True,
    ).fit_predict(scaled_features)
    subset["cluster_label"] = labels

    non_noise = subset[subset["cluster_label"] != -1]
    if non_noise.empty:
        raise RuntimeError(f"HDBSCAN found only noise for {cluster['name']}.")

    expected_parallax = expected_parallax_mas(cluster)
    parallax_medians = non_noise.groupby("cluster_label")["parallax"].median()
    best_label = (parallax_medians - expected_parallax).abs().idxmin()
    return int(best_label)


def save_member_plot(cluster_key: str, cluster: dict[str, float | str], subset: pd.DataFrame, members: pd.DataFrame) -> None:
    output_path = OUT_DIR / f"pm_members_{cluster_key}.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(subset["pmra"], subset["pmdec"], s=2, c="lightgrey", alpha=0.5, linewidths=0)
    ax.scatter(members["pmra"], members["pmdec"], s=4, c="tab:blue", alpha=0.8, linewidths=0)
    ax.set_title(str(cluster["name"]))
    ax.set_xlabel("pmra (mas/yr)")
    ax.set_ylabel("pmdec (mas/yr)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_members(cluster_key: str, members: pd.DataFrame) -> None:
    output_path = DATA_DIR / f"{cluster_key}_members.parquet"
    members.drop(columns=["cluster_label"], errors="ignore").to_parquet(output_path, index=False)


def print_summary(cluster_key: str, cluster: dict[str, float | str], subset: pd.DataFrame, members: pd.DataFrame) -> None:
    median_distance_pc = (1000.0 / members["parallax"]).median()
    median_pmra = members["pmra"].median()
    median_pmdec = members["pmdec"].median()
    print(
        f"{cluster['name']} ({cluster_key}): subset={len(subset)}, members={len(members)}, "
        f"median_distance_pc={median_distance_pc:.1f}, median_pmra={median_pmra:.3f}, "
        f"median_pmdec={median_pmdec:.3f}"
    )


def process_cluster(cluster_key: str, cluster: dict[str, float | str]) -> None:
    frame = load_cluster_frame(cluster_key)
    subset = preselect_subset(frame, cluster)
    print(f"{cluster['name']} subset size: {len(subset)}")

    member_label = find_member_label(subset, cluster)
    members = subset[subset["cluster_label"] == member_label].copy()

    save_members(cluster_key, members)
    save_member_plot(cluster_key, cluster, subset, members)
    print_summary(cluster_key, cluster, subset, members)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    for cluster_key, cluster in CLUSTERS.items():
        process_cluster(cluster_key, cluster)


if __name__ == "__main__":
    main()
