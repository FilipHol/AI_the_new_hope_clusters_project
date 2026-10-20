import pandas as pd
import pyvo as vo

from config import CLUSTERS, DATA_DIR, DATA_SOURCE, DATA_SOURCES, OUT_DIR, STANDARD_COLUMNS


JOB_TIMEOUT_SECONDS = 1800.0
REQUESTED_MAXREC = 2_000_000


def get_source_config() -> dict[str, object]:
    if DATA_SOURCE not in DATA_SOURCES:
        raise RuntimeError(f"Unknown DATA_SOURCE: {DATA_SOURCE}")
    return DATA_SOURCES[DATA_SOURCE]


def get_available_columns(service: vo.dal.TAPService, table_name: str) -> set[str]:
    result = service.run_sync(f"SELECT TOP 1 * FROM {table_name}")
    return {column_name.lower() for column_name in result.fieldnames}


def print_required_columns(available_columns: set[str], column_mapping: dict[str, str]) -> None:
    print(f"Using data source: {DATA_SOURCE}")
    print("Required source columns:")
    for standard_name in STANDARD_COLUMNS:
        source_name = column_mapping[standard_name]
        exists = source_name.lower() in available_columns
        print(f"- {source_name} -> {standard_name}: {'yes' if exists else 'no'}")


def validate_required_columns(available_columns: set[str], column_mapping: dict[str, str]) -> None:
    missing_columns = [
        source_name
        for source_name in column_mapping.values()
        if source_name.lower() not in available_columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise RuntimeError(
            "Missing required source columns: "
            f"{missing_text}. Update the selected TAP column mapping before running the fetch stage."
        )


def get_effective_maxrec(service: vo.dal.TAPService) -> int:
    hardlimit = getattr(service, "hardlimit", None)
    if hardlimit is None:
        return REQUESTED_MAXREC
    return min(REQUESTED_MAXREC, int(hardlimit))


def build_select_clause(column_mapping: dict[str, str]) -> str:
    select_parts = []
    for standard_name in STANDARD_COLUMNS:
        source_name = column_mapping[standard_name]
        if source_name.lower() == standard_name.lower():
            select_parts.append(source_name)
        else:
            select_parts.append(f"{source_name} AS {standard_name}")
    return ",\n    ".join(select_parts)


def build_query(cluster: dict[str, float | str], table_name: str, column_mapping: dict[str, str]) -> str:
    ra_column = column_mapping["ra"]
    dec_column = column_mapping["dec"]
    parallax_column = column_mapping["parallax"]
    parallax_error_column = column_mapping["parallax_error"]
    ruwe_column = column_mapping["ruwe"]
    g_column = column_mapping["phot_g_mean_mag"]
    bp_column = column_mapping["phot_bp_mean_mag"]
    rp_column = column_mapping["phot_rp_mean_mag"]

    return f"""
SELECT
    {build_select_clause(column_mapping)}
FROM {table_name}
WHERE CONTAINS(
    POINT('ICRS', {ra_column}, {dec_column}),
    CIRCLE('ICRS', {cluster['ra_deg']}, {cluster['dec_deg']}, {cluster['radius_deg']})
) = 1
AND {parallax_column} / {parallax_error_column} > 5
AND {ruwe_column} < 1.4
AND {g_column} IS NOT NULL
AND {bp_column} IS NOT NULL
AND {rp_column} IS NOT NULL
"""


def query_cluster(
    service: vo.dal.TAPService,
    cluster_key: str,
    cluster: dict[str, float | str],
    table_name: str,
    column_mapping: dict[str, str],
    maxrec: int,
) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{cluster_key}.parquet"
    if cache_path.exists():
        print(f"Using cache for {cluster['name']}: {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"Querying {cluster['name']}...")
    job = service.submit_job(build_query(cluster, table_name, column_mapping), maxrec=maxrec)
    try:
        job.run()
        job.wait(timeout=JOB_TIMEOUT_SECONDS)
        job.raise_if_error()
        results = job.fetch_result()
    finally:
        job.delete()

    frame = results.to_table().to_pandas()
    frame = frame[STANDARD_COLUMNS]
    frame.to_parquet(cache_path, index=False)

    row_count = len(frame)
    default_cap = getattr(service, "maxrec", None)
    if results.query_status == "OVERFLOW" or row_count == maxrec or row_count == default_cap:
        print(
            f"Warning: {cluster['name']} returned {row_count} rows, which may indicate a TAP row cap."
        )

    print(f"Saved {cluster['name']} cache to {cache_path.name}")
    return frame


def print_summary(cluster_key: str, cluster: dict[str, float | str], frame: pd.DataFrame) -> None:
    row_count = len(frame)
    median_parallax = frame["parallax"].median()
    print(
        f"{cluster['name']} ({cluster_key}): rows={row_count}, "
        f"median_parallax={median_parallax:.4f} mas"
    )


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    source_config = get_source_config()
    table_name = str(source_config["table_name"])
    column_mapping = dict(source_config["column_mapping"])
    service = vo.dal.TAPService(str(source_config["url"]))

    available_columns = get_available_columns(service, table_name)
    print_required_columns(available_columns, column_mapping)
    validate_required_columns(available_columns, column_mapping)

    maxrec = get_effective_maxrec(service)
    cluster_frames: dict[str, pd.DataFrame] = {}
    for cluster_key, cluster in CLUSTERS.items():
        cluster_frames[cluster_key] = query_cluster(
            service,
            cluster_key,
            cluster,
            table_name,
            column_mapping,
            maxrec,
        )

    print("Summary:")
    for cluster_key, cluster in CLUSTERS.items():
        print_summary(cluster_key, cluster, cluster_frames[cluster_key])


if __name__ == "__main__":
    main()
