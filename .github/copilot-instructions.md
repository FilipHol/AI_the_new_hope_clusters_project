Project: Gaia DR3 open clusters (Pleiades, Praesepe, M67).
- Python 3.11+, venv in .venv, dependencies in requirements.txt.
- One script per stage in src/, numbered: 01_fetch.py, 02_pm_plot.py, ...
- Shared constants (cluster names, coords, radii, expected distances) live in src/config.py only.
- Raw data cached in data/*.parquet. Never re-query Gaia if the file exists.
- Figures go to out/. Each script prints sanity numbers at the end (row counts, medians).
- Only edit the file(s) I name. Do not refactor or reformat other stages.
- Use only gaia_source_lite. Never query gaiadr3.gaia_source (it times out).
- Keep code simple and commented. No extra features I didn't ask for.

Isochrone stage (scripts 07 and up):
- Isochrone files live in data/isochrones/. Never re-download if the file exists. Print the source URL and file size before any download.
- Do not modify src/01 to src/06. Only add new keys to src/config.py, never rename or remove existing ones.
- Fit in apparent G vs BP-RP with one distance modulus per cluster (out/summary.csv, column median_distance_pc_zp017). Gaia EDR3 passbands, Vega system.
- Every fit script saves a figure with the isochrone overlaid on the members in out/.
- Print best-fit values and the number of stars used.