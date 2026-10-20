from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import OUT_DIR


SUMMARY_PATH = OUT_DIR / "summary.csv"
BOOTSTRAP_ORDER_PATH = OUT_DIR / "bootstrap_order.csv"
REPORT_PATH = OUT_DIR / "report.html"
ISOCHRONE_PATH = OUT_DIR / "isochrone_summary.csv"
IMAGE_PATHS = [
    ("pm_all", OUT_DIR / "pm_all.png"),
    ("pm_members_pleiades", OUT_DIR / "pm_members_pleiades.png"),
    ("pm_members_praesepe", OUT_DIR / "pm_members_praesepe.png"),
    ("pm_members_m67", OUT_DIR / "pm_members_m67.png"),
    ("hr_all", OUT_DIR / "hr_all.png"),
    ("hr_turnoff", OUT_DIR / "hr_turnoff.png"),
]
REQUIRED_SUMMARY_COLUMNS = [
    "cluster",
    "cluster_key",
    "n_members",
    "median_distance_pc",
    "median_distance_pc_zp017",
    "turnoff_M_G",
    "turnoff_bp_rp",
    "fifth_brightest_ms_M_G",
]
OPTIONAL_SUMMARY_COLUMNS = [
    "bootstrap_turnoff_M_G_median",
    "bootstrap_turnoff_M_G_std",
    "bootstrap_turnoff_bp_rp_median",
    "bootstrap_turnoff_bp_rp_std",
    "bootstrap_fifth_brightest_ms_M_G_median",
    "bootstrap_fifth_brightest_ms_M_G_std",
]
EXPECTED_ORDER = ["Pleiades", "Praesepe", "M67"]


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)
    print(f"Warning: {message}")


def read_summary_csv(warnings: list[str]) -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    summary_df = pd.read_csv(SUMMARY_PATH)
    print("summary.csv columns:", ", ".join(summary_df.columns))

    for column_name in REQUIRED_SUMMARY_COLUMNS:
        if column_name not in summary_df.columns:
            warn(warnings, f"summary.csv is missing required column '{column_name}'.")

    for column_name in OPTIONAL_SUMMARY_COLUMNS:
        if column_name not in summary_df.columns:
            warn(warnings, f"summary.csv is missing optional column '{column_name}'; related text will be omitted.")

    return summary_df


def read_bootstrap_order_csv(warnings: list[str]) -> pd.DataFrame | None:
    if not BOOTSTRAP_ORDER_PATH.exists():
        warn(warnings, f"Missing bootstrap order file: {BOOTSTRAP_ORDER_PATH.name}")
        return None

    bootstrap_df = pd.read_csv(BOOTSTRAP_ORDER_PATH)
    print("bootstrap_order.csv columns:", ", ".join(bootstrap_df.columns))
    return bootstrap_df


def encode_image(path: Path, warnings: list[str]) -> str | None:
    if not path.exists():
        warn(warnings, f"Image not found: {path.name}")
        return None

    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_images(warnings: list[str]) -> dict[str, str]:
    image_map: dict[str, str] = {}
    for image_key, image_path in IMAGE_PATHS:
        encoded = encode_image(image_path, warnings)
        if encoded is not None:
            image_map[image_key] = encoded
    return image_map


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def format_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value)}"


def compute_order(summary_df: pd.DataFrame, warnings: list[str]) -> list[str]:
    if "cluster" not in summary_df.columns or "turnoff_bp_rp" not in summary_df.columns:
        warn(warnings, "Cannot compute age order because 'cluster' or 'turnoff_bp_rp' is missing.")
        return []

    ordered = summary_df.sort_values("turnoff_bp_rp", ascending=True)["cluster"].tolist()
    return [str(name) for name in ordered]


def make_badges(order: list[str]) -> str:
    if not order:
        return "<p class=\"muted\">Age order unavailable.</p>"

    parts: list[str] = []
    for index, cluster_name in enumerate(order):
        parts.append(f"<span class=\"badge\">{html.escape(cluster_name)}</span>")
        if index < len(order) - 1:
            parts.append("<span class=\"arrow\">&rarr;</span>")
    return "".join(parts)


def make_summary_table(summary_df: pd.DataFrame, warnings: list[str]) -> str:
    needed_columns = [
        "cluster",
        "n_members",
        "median_distance_pc",
        "median_distance_pc_zp017",
        "turnoff_M_G",
        "turnoff_bp_rp",
        "fifth_brightest_ms_M_G",
    ]
    missing = [column for column in needed_columns if column not in summary_df.columns]
    if missing:
        warn(warnings, "Summary table omitted columns: " + ", ".join(missing))

    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('cluster', '—')))}</td>"
            f"<td>{format_int(row.get('n_members'))}</td>"
            f"<td>{format_float(row.get('median_distance_pc'), 1)}</td>"
            f"<td>{format_float(row.get('median_distance_pc_zp017'), 1)}</td>"
            f"<td>{format_float(row.get('turnoff_M_G'), 3)}</td>"
            f"<td>{format_float(row.get('turnoff_bp_rp'), 3)}</td>"
            f"<td>{format_float(row.get('fifth_brightest_ms_M_G'), 3)}</td>"
            "</tr>"
        )

    return (
        "<div class=\"table-wrap\">"
        "<table>"
        "<thead><tr>"
        "<th>Cluster</th>"
        "<th>N members</th>"
        "<th>Median distance (pc)</th>"
        "<th>Distance with +0.017 mas offset (pc)</th>"
        "<th>Turn-off M_G</th>"
        "<th>Turn-off BP-RP</th>"
        "<th>5th-brightest main-sequence M_G</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def format_turnoff_mg_label(cluster_name: str, value: float | int | None) -> str:
    base = format_float(value, 3)
    if cluster_name in {"Pleiades", "Praesepe"} and base != "—":
        return f"{base} (no clear knee)"
    return base


def make_turnoff_metrics_table(summary_df: pd.DataFrame, warnings: list[str]) -> str:
    required = [
        "cluster",
        "turnoff_bp_rp",
        "turnoff_M_G",
        "bootstrap_turnoff_bp_rp_median",
        "bootstrap_turnoff_bp_rp_std",
        "bootstrap_turnoff_M_G_median",
        "bootstrap_turnoff_M_G_std",
        "fifth_brightest_ms_M_G",
        "bootstrap_fifth_brightest_ms_M_G_median",
        "bootstrap_fifth_brightest_ms_M_G_std",
    ]
    missing = [column for column in required if column not in summary_df.columns]
    if missing:
        warn(warnings, "Turn-off metrics table omitted columns: " + ", ".join(missing))
        return "<p class=\"muted\">Turn-off metrics table unavailable.</p>"

    rows = []
    for _, row in summary_df.iterrows():
        cluster_name = str(row["cluster"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(cluster_name)}</td>"
            f"<td>{format_float(row['turnoff_bp_rp'], 3)}</td>"
            f"<td>{html.escape(format_turnoff_mg_label(cluster_name, row['turnoff_M_G']))}</td>"
            f"<td>{format_float(row['bootstrap_turnoff_bp_rp_median'], 3)} ± {format_float(row['bootstrap_turnoff_bp_rp_std'], 3)}</td>"
            f"<td>{format_float(row['bootstrap_turnoff_M_G_median'], 3)} ± {format_float(row['bootstrap_turnoff_M_G_std'], 3)}</td>"
            f"<td>{format_float(row['fifth_brightest_ms_M_G'], 3)}</td>"
            f"<td>{format_float(row['bootstrap_fifth_brightest_ms_M_G_median'], 3)} ± {format_float(row['bootstrap_fifth_brightest_ms_M_G_std'], 3)}</td>"
            "</tr>"
        )

    return (
        "<div class=\"table-wrap\">"
        "<table>"
        "<thead><tr>"
        "<th>Cluster</th>"
        "<th>Main-run turn-off BP-RP</th>"
        "<th>Main-run turn-off M_G</th>"
        "<th>Bootstrap turn-off BP-RP median ± std</th>"
        "<th>Bootstrap turn-off M_G median ± std</th>"
        "<th>5th-brightest main-sequence M_G</th>"
        "<th>Bootstrap 5th-brightest M_G median ± std</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def make_bootstrap_order_table(bootstrap_order_df: pd.DataFrame | None, warnings: list[str]) -> str:
    if bootstrap_order_df is None:
        return "<p class=\"muted\">Bootstrap order table unavailable.</p>"

    required = [
        "measure",
        "fraction_pleiades_lt_praesepe",
        "fraction_praesepe_lt_m67",
        "fraction_pleiades_lt_m67",
        "fraction_all_three_hold",
    ]
    missing = [column for column in required if column not in bootstrap_order_df.columns]
    if missing:
        warn(warnings, "Bootstrap order table omitted columns: " + ", ".join(missing))
        return "<p class=\"muted\">Bootstrap order table unavailable.</p>"

    measure_labels = {
        "colour_turnoff_bp_rp": "Colour turn-off BP-RP",
        "fifth_brightest_ms_M_G": "5th-brightest main-sequence M_G",
    }

    rows = []
    for _, row in bootstrap_order_df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(measure_labels.get(str(row['measure']), str(row['measure'])))}</td>"
            f"<td>{format_float(row['fraction_pleiades_lt_praesepe'], 3)}</td>"
            f"<td>{format_float(row['fraction_praesepe_lt_m67'], 3)}</td>"
            f"<td>{format_float(row['fraction_pleiades_lt_m67'], 3)}</td>"
            f"<td>{format_float(row['fraction_all_three_hold'], 3)}</td>"
            "</tr>"
        )

    return (
        "<div class=\"table-wrap\">"
        "<table>"
        "<thead><tr>"
        "<th>Measure</th>"
        "<th>Pleiades &lt; Praesepe</th>"
        "<th>Praesepe &lt; M67</th>"
        "<th>Pleiades &lt; M67</th>"
        "<th>All three hold</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def make_image_block(
    section_id: str,
    title: str,
    caption: str,
    image_sources: list[tuple[str, str]],
    warnings: list[str],
) -> str:
    images_html = []
    for label, src in image_sources:
        images_html.append(
            "<figure class=\"figure-card\">"
            f"<img src=\"{src}\" alt=\"{html.escape(label)}\">"
            f"<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )

    if not images_html:
        warn(warnings, f"No images available for section '{title}'.")
        return (
            f"<section id=\"{html.escape(section_id)}\" class=\"card section\">"
            f"<h2>{html.escape(title)}</h2>"
            f"<p class=\"caption\">{html.escape(caption)}</p>"
            "<p class=\"muted\">No figures available.</p>"
            "</section>"
        )

    return (
        f"<section id=\"{html.escape(section_id)}\" class=\"card section\">"
        f"<h2>{html.escape(title)}</h2>"
        f"<p class=\"caption\">{html.escape(caption)}</p>"
        f"<div class=\"figure-grid\">{''.join(images_html)}</div>"
        "</section>"
    )


def make_isochrone_section(warnings: list[str]) -> str:
    if not ISOCHRONE_PATH.exists():
        return (
            "<section id=\"isochrones\" class=\"card section\">"
            "<h2>Isochrone ages: not done yet</h2>"
            "<p class=\"muted\">No isochrone summary file is present.</p>"
            "</section>"
        )

    try:
        isochrone_df = pd.read_csv(ISOCHRONE_PATH)
    except Exception as exc:
        warn(warnings, f"Could not read isochrone_summary.csv: {exc}")
        return (
            "<section id=\"isochrones\" class=\"card section\">"
            "<h2>Isochrone ages: not done yet</h2>"
            "<p class=\"muted\">Isochrone summary exists but could not be read.</p>"
            "</section>"
        )

    return (
        "<section id=\"isochrones\" class=\"card section\">"
        "<h2>Isochrone ages</h2>"
        "<div class=\"table-wrap\">"
        f"{isochrone_df.to_html(index=False, classes='dataframe', border=0)}"
        "</div>"
        "</section>"
    )


def build_html(
    summary_df: pd.DataFrame,
    bootstrap_order_df: pd.DataFrame | None,
    images: dict[str, str],
    warnings: list[str],
) -> str:
    generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    order = compute_order(summary_df, warnings)
    order_text = " < ".join(order) if order else None

    badges_html = make_badges(order)
    summary_table_html = make_summary_table(summary_df, warnings)
    turnoff_metrics_table_html = make_turnoff_metrics_table(summary_df, warnings)
    bootstrap_order_table_html = make_bootstrap_order_table(bootstrap_order_df, warnings)

    age_order_bits = []
    if order_text is not None:
        age_order_bits.append(f"<p>The CSV-based age order is <strong>{html.escape(order_text)}</strong>.</p>")
    else:
        age_order_bits.append("<p class=\"muted\">The CSV-based age order is unavailable.</p>")

    age_order_bits.append("<p>The bootstrap order fractions for the colour and luminosity measures are listed below.</p>")
    age_order_bits.append(
        "<p>M67 is clearly the oldest; the Pleiades-Praesepe step is not statistically resolved by the simple turn-off measures. "
        "Between the Pleiades and Praesepe, red-clump giants and a white-dwarf sequence appear in Praesepe but not in the Pleiades (one possible white-dwarf candidate). "
        "M67 also has a red clump, and the Pleiades retain the bluer and brighter main-sequence end.</p>"
    )

    proper_motion_images = []
    if "pm_all" in images:
        proper_motion_images.append(("All-field proper motions", images["pm_all"]))
    caption_names = {"pleiades": "Pleiades", "praesepe": "Praesepe", "m67": "M67"}
    for cluster_key in ["pleiades", "praesepe", "m67"]:
        image_key = f"pm_members_{cluster_key}"
        if image_key in images:
            proper_motion_images.append((f"Member selection: {caption_names[cluster_key]}", images[image_key]))

    hr_images = []
    if "hr_all" in images:
        hr_images.append(("Combined HR diagrams", images["hr_all"]))

    turnoff_images = []
    if "hr_turnoff" in images:
        turnoff_images.append(("Turn-off markers", images["hr_turnoff"]))

    isochrone_section = make_isochrone_section(warnings)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Open clusters with Gaia DR3</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f6f7fb;
      --surface: #ffffff;
      --surface-alt: #f0f3f8;
      --text: #18202a;
      --muted: #5a6472;
      --border: #d9dee7;
      --accent: #2f6feb;
      --accent-soft: #e8f0ff;
      --shadow: rgba(15, 23, 42, 0.06);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f141b;
        --surface: #161d26;
        --surface-alt: #111821;
        --text: #ecf2f8;
        --muted: #a9b4c0;
        --border: #2b3642;
        --accent: #7cb3ff;
        --accent-soft: #18263a;
        --shadow: rgba(0, 0, 0, 0.3);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(10px);
      background: color-mix(in srgb, var(--surface) 88%, transparent);
      border-bottom: 1px solid var(--border);
    }}
    .nav-inner {{
      max-width: 1000px;
      margin: 0 auto;
      padding: 0.9rem 1rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.9rem;
    }}
    .container {{
      max-width: 1000px;
      margin: 0 auto;
      padding: 1.5rem 1rem 3rem;
    }}
    .hero {{ margin-bottom: 1.5rem; }}
    .hero h1 {{ margin: 0 0 0.6rem; font-size: clamp(1.9rem, 4vw, 2.7rem); line-height: 1.15; }}
    .hero p {{ margin: 0.25rem 0; color: var(--muted); }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 30px var(--shadow);
      padding: 1.25rem;
    }}
    .section {{ margin-top: 1.25rem; }}
    .eyebrow {{ margin: 1rem 0 0.4rem; color: var(--muted); font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .badge-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin-top: 0.25rem; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 0.45rem 0.8rem;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--border));
      font-weight: 600;
    }}
    .arrow {{ color: var(--muted); font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; margin-top: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 0.75rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ background: var(--surface-alt); font-weight: 700; }}
    .caption {{ color: var(--muted); margin-top: 0.2rem; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem; }}
    .figure-card {{ margin: 0; padding: 0.75rem; border: 1px solid var(--border); border-radius: 14px; background: var(--surface-alt); }}
    .figure-card img {{ display: block; width: 100%; max-width: 100%; height: auto; border-radius: 10px; }}
    .figure-card figcaption {{ margin-top: 0.6rem; color: var(--muted); font-size: 0.95rem; }}
    .metric-list {{ margin: 1rem 0 0; padding-left: 1.2rem; }}
    .muted {{ color: var(--muted); }}
    footer {{ margin-top: 1.5rem; color: var(--muted); font-size: 0.95rem; }}
  </style>
</head>
<body>
  <nav class=\"nav\">
    <div class=\"nav-inner\">
      <a href=\"#results\">Key results</a>
      <a href=\"#proper-motions\">Proper motions</a>
      <a href=\"#hr-diagrams\">HR diagrams</a>
      <a href=\"#turnoff\">Turn-off markers</a>
      <a href=\"#age-ordering\">Age ordering</a>
      <a href=\"#method\">Method</a>
      <a href=\"#limitations\">Limitations</a>
      <a href=\"#isochrones\">Isochrone ages</a>
    </div>
  </nav>
  <main class=\"container\">
    <header class=\"hero\">
      <h1>Open clusters with Gaia DR3: Pleiades, Praesepe, M67</h1>
      <p>Generated on {html.escape(generation_date)}.</p>
      <p>Data source: ARI Heidelberg TAP, <code>gaiadr3.gaia_source_lite</code>.</p>
    </header>

    <section id=\"results\" class=\"card section\">
      <h2>Key results</h2>
      {summary_table_html}
      <p class=\"eyebrow\">Youngest to oldest</p>
      <div class=\"badge-row\">{badges_html}</div>
      {turnoff_metrics_table_html}
    </section>

    {make_image_block('proper-motions', 'Proper motions', 'The field-level proper-motion map shows the wide Gaia selection, while the member panels show the clustered subset adopted for each open cluster.', proper_motion_images, warnings)}

    {make_image_block('hr-diagrams', 'HR diagrams', 'The member-selected HR diagrams show the different main-sequence and evolved-star morphologies across the three clusters.', hr_images, warnings)}

    {make_image_block('turnoff', 'Turn-off markers', 'The red circles mark the simple turn-off proxy defined from the bluest median colour in binned absolute magnitude.', turnoff_images, warnings)}

    <section id=\"age-ordering\" class=\"card section\">
      <h2>Age ordering</h2>
      {''.join(age_order_bits)}
      {bootstrap_order_table_html}
    </section>

    <section id=\"method\" class=\"card section\">
      <h2>Method</h2>
      <p>
        Gaia rows were fetched with the filters <code>parallax/parallax_error &gt; 5</code>, <code>ruwe &lt; 1.4</code>, and non-null Gaia G, BP, and RP photometry,
        using cone radii of 10, 5, and 1 deg for Pleiades, Praesepe, and M67. Member finding started from a pre-selection within proper-motion windows of 15, 15,
        and 10 mas/yr for Pleiades, Praesepe, and M67, together with a parallax window within 30% of the expected value. The final member choice used HDBSCAN on standardized
        <code>(pmra, pmdec, parallax)</code> with <code>min_cluster_size=50</code> and <code>min_samples=10</code>. Absolute magnitude was defined as
        <code>M_G = phot_g_mean_mag + 5*log10(parallax/100)</code> with parallax in mas. The turn-off proxy used 0.25 mag bins, at least 5 stars per bin, and the bluest
        median BP-RP in the range <code>-1 &lt; M_G &lt; 6</code>. The 5th-brightest luminosity measure used stars with <code>BP-RP &lt; 1.0</code> for Pleiades and Praesepe,
        and <code>BP-RP &lt; 0.95</code> plus <code>M_G &gt; 2.5</code> for M67.
      </p>
    </section>

    <section id=\"limitations\" class=\"card section\">
      <h2>Limitations</h2>
      <ul class=\"metric-list\">
        <li>Bootstrap resampling captures sampling noise only.</li>
        <li>No reddening correction or binary-star correction is applied.</li>
        <li>The +0.017 mas parallax shift is an approximate global zero-point correction.</li>
        <li>Distances are based on the simple <code>1/parallax</code> conversion.</li>
        <li>The turn-off proxy is weak when the cluster HR diagram does not show a clean knee.</li>
        <li>The BP-RP cuts used for the luminosity measure were chosen by eye.</li>
      </ul>
    </section>

    {isochrone_section}

    <footer>
      Literature ages for context only, not a result of this analysis: Pleiades about 110 Myr, Praesepe about 650 Myr, and M67 about 4 Gyr.
    </footer>
  </main>
</body>
</html>
"""


def main() -> None:
    warnings: list[str] = []
    summary_df = read_summary_csv(warnings)
    bootstrap_order_df = read_bootstrap_order_csv(warnings)
    images = load_images(warnings)
    html_text = build_html(summary_df, bootstrap_order_df, images, warnings)
    REPORT_PATH.write_text(html_text, encoding="utf-8")

    file_size_mb = REPORT_PATH.stat().st_size / (1024 * 1024)
    print(f"Output path: {REPORT_PATH}")
    print(f"File size: {file_size_mb:.2f} MB")
    if warnings:
        print("Warnings:")
        for message in warnings:
            print(f"- {message}")
    else:
        print("Warnings: none")


if __name__ == "__main__":
    main()
