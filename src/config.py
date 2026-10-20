from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"

STANDARD_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "ruwe",
]

DATA_SOURCE = "ari_heidelberg"
DATA_SOURCES = {
    "ari_heidelberg": {
        "url": "https://gaia.ari.uni-heidelberg.de/tap",
        "table_name": "gaiadr3.gaia_source_lite",
        "column_mapping": {
            "source_id": "source_id",
            "ra": "ra",
            "dec": "dec",
            "parallax": "parallax",
            "parallax_error": "parallax_error",
            "pmra": "pmra",
            "pmdec": "pmdec",
            "phot_g_mean_mag": "phot_g_mean_mag",
            "phot_bp_mean_mag": "phot_bp_mean_mag",
            "phot_rp_mean_mag": "phot_rp_mean_mag",
            "ruwe": "ruwe",
        },
    },
    "vizier": {
        "url": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
        "table_name": '"I/355/gaiadr3"',
        "column_mapping": {
            "source_id": "Source",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "parallax": "Plx",
            "parallax_error": "e_Plx",
            "pmra": "pmRA",
            "pmdec": "pmDE",
            "phot_g_mean_mag": "Gmag",
            "phot_bp_mean_mag": "BPmag",
            "phot_rp_mean_mag": "RPmag",
            "ruwe": "RUWE",
        },
    },
}

CLUSTERS = {
    "pleiades": {
        "name": "Pleiades",
        "ra_deg": 56.75,
        "dec_deg": 24.12,
        "radius_deg": 10.0,
        "expected_distance_pc": 136.0,
        "expected_pmra": 20.0,
        "expected_pmdec": -45.0,
        "pm_window_masyr": 15.0,
    },
    "praesepe": {
        "name": "Praesepe",
        "ra_deg": 130.05,
        "dec_deg": 19.69,
        "radius_deg": 5.0,
        "expected_distance_pc": 185.0,
        "expected_pmra": -36.0,
        "expected_pmdec": -13.0,
        "pm_window_masyr": 15.0,
    },
    "m67": {
        "name": "M67",
        "ra_deg": 132.825,
        "dec_deg": 11.8,
        "radius_deg": 1.0,
        "expected_distance_pc": 850.0,
        "expected_pmra": -11.0,
        "expected_pmdec": -3.0,
        "pm_window_masyr": 10.0,
    },
}

ISO_ROTATION_VVCRIT = 0.4  # MIST v1.2 grid, v/vcrit = 0.4
ISO_FEH_BASELINE = 0.0  # Baseline metallicity for the main fits.
ISO_FEH_PRAESEPE_SENS = 0.15  # Praesepe metallicity sensitivity test value.
ISO_FEH_GRID_FILES = [0.00, 0.25]  # Packaged grid metallicities used for interpolation.
ISO_AV = {"pleiades": 0.10, "praesepe": 0.05, "m67": 0.10}  # Approximate literature values, editable.
ISO_AV_SENS_PLAUSIBLE = [0.0, 0.05, 0.10, 0.15]  # Plausible extinction sensitivity grid.
ISO_AV_SENS_EXTREME = 0.30  # Extreme extinction sensitivity check.
ISO_DIST_SENS_FRAC = 0.01  # Fractional distance sensitivity perturbation.
ISO_AG_OVER_AV = 0.80  # Approximate A_G / A_V for Gaia; Riello et al. (2021) handling still to be checked.
ISO_EBPRP_OVER_AV = 0.42  # Approximate E(BP-RP) / A_V for Gaia; Riello et al. (2021) handling still to be checked.
ISO_LIT_AGE_MYR = {"pleiades": 110, "praesepe": 650, "m67": 4000}  # Literature ages for the pre-fit overlay only.
ISO_ALLOWED_PHASES = [0, 2, 3]  # MIST phase codes to use; verify exact meanings from the grid README when downloaded.
ISO_FIT_MG_MAX = 6.0  # Fit window reduced because the member-MIST colour offset rises for M_G > 6 in all three clusters.
ISO_FIT_MG_MAX_SENS = 7.0  # Sensitivity value for the fit-window upper limit.
ISO_OFFSET_MG_RANGE = (4.0, 6.0)  # Offset window reduced because the member-MIST colour offset rises for M_G > 6 in all three clusters.
ISO_OFFSET_DROP_FRAC = 0.2  # Drop the reddest 20% of stars when estimating the colour offset.
ISO_OFFSET_SENS = 0.02  # Sensitivity step for the fixed colour-offset experiments.
ISO_SIGMA_COLOUR = 0.05  # Colour scatter used in the fit metric.
ISO_SIGMA_G = 0.15  # G-band scatter used in the fit metric.
ISO_D_CAP = 4.0  # Per-star loss cap in sigma units.
ISO_D_CAP_SENS = 2.0  # Sensitivity value for the per-star loss cap.
ISO_LOSS_INTERVAL_FRAC = 0.10  # Descriptive width threshold for the baseline loss curve.
ISO_CONSTRAINED_LOSS_RATIO = 1.3  # Age is constrained only if both baseline +/-0.2 dex loss ratios exceed this value.
ISO_LOGAGE_MIN = 7.5  # Minimum log-age grid value.
ISO_LOGAGE_MAX = 10.0  # Maximum log-age grid value.
ISO_LOGAGE_STEP = 0.05  # Log-age grid step.
ISO_BOOTSTRAP_N = 200  # Number of bootstrap resamples.
ISO_SEED = 42  # Random seed for reproducibility.


if __name__ == "__main__":
    print(f"ISO_ROTATION_VVCRIT = {ISO_ROTATION_VVCRIT}")
    print(f"ISO_FEH_BASELINE = {ISO_FEH_BASELINE}")
    print(f"ISO_FEH_PRAESEPE_SENS = {ISO_FEH_PRAESEPE_SENS}")
    print(f"ISO_FEH_GRID_FILES = {ISO_FEH_GRID_FILES}")
    print(f"ISO_AV = {ISO_AV}")
    print(f"ISO_AV_SENS_PLAUSIBLE = {ISO_AV_SENS_PLAUSIBLE}")
    print(f"ISO_AV_SENS_EXTREME = {ISO_AV_SENS_EXTREME}")
    print(f"ISO_DIST_SENS_FRAC = {ISO_DIST_SENS_FRAC}")
    print(
        f"ISO_AG_OVER_AV = {ISO_AG_OVER_AV}  # approximate A_G / A_V for Gaia; source: Riello et al. (2021) still to be checked"
    )
    print(
        f"ISO_EBPRP_OVER_AV = {ISO_EBPRP_OVER_AV}  # approximate E(BP-RP) / A_V for Gaia; source: Riello et al. (2021) still to be checked"
    )
    print(f"ISO_LIT_AGE_MYR = {ISO_LIT_AGE_MYR}")
    print(f"ISO_ALLOWED_PHASES = {ISO_ALLOWED_PHASES}")
    print(f"ISO_FIT_MG_MAX = {ISO_FIT_MG_MAX}")
    print(f"ISO_FIT_MG_MAX_SENS = {ISO_FIT_MG_MAX_SENS}")
    print(f"ISO_OFFSET_MG_RANGE = {ISO_OFFSET_MG_RANGE}")
    print(f"ISO_OFFSET_DROP_FRAC = {ISO_OFFSET_DROP_FRAC}")
    print(f"ISO_OFFSET_SENS = {ISO_OFFSET_SENS}")
    print(f"ISO_SIGMA_COLOUR = {ISO_SIGMA_COLOUR}")
    print(f"ISO_SIGMA_G = {ISO_SIGMA_G}")
    print(f"ISO_D_CAP = {ISO_D_CAP}")
    print(f"ISO_D_CAP_SENS = {ISO_D_CAP_SENS}")
    print(f"ISO_LOSS_INTERVAL_FRAC = {ISO_LOSS_INTERVAL_FRAC}")
    print(f"ISO_CONSTRAINED_LOSS_RATIO = {ISO_CONSTRAINED_LOSS_RATIO}")
    print(f"ISO_LOGAGE_MIN = {ISO_LOGAGE_MIN}")
    print(f"ISO_LOGAGE_MAX = {ISO_LOGAGE_MAX}")
    print(f"ISO_LOGAGE_STEP = {ISO_LOGAGE_STEP}")
    print(f"ISO_BOOTSTRAP_N = {ISO_BOOTSTRAP_N}")
    print(f"ISO_SEED = {ISO_SEED}")
