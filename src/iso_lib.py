from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    ISO_ALLOWED_PHASES,
    ISO_D_CAP,
    ISO_FIT_MG_MAX,
    ISO_LOGAGE_MAX,
    ISO_LOGAGE_MIN,
    ISO_LOGAGE_STEP,
    ISO_OFFSET_DROP_FRAC,
    ISO_OFFSET_MG_RANGE,
    ISO_SIGMA_COLOUR,
    ISO_SIGMA_G,
)


ISO_DIR = DATA_DIR / "isochrones"
TARGET_FILES = {
    0.00: "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
    0.25: "MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
}
CACHE_FILES = {
    0.00: ISO_DIR / "mist_feh_p0.00.parquet",
    0.25: ISO_DIR / "mist_feh_p0.25.parquet",
}
OUTPUT_COLUMNS = ["eep", "logage", "mass_init", "logteff", "logl", "feh", "g", "bp", "rp", "phase"]
SOURCE_COLUMNS = {
    "EEP": "eep",
    "log10_isochrone_age_yr": "logage",
    "initial_mass": "mass_init",
    "log_Teff": "logteff",
    "log_L": "logl",
    "[Fe/H]": "feh",
    "Gaia_G_EDR3": "g",
    "Gaia_BP_EDR3": "bp",
    "Gaia_RP_EDR3": "rp",
    "phase": "phase",
}
INTERP_COLUMNS = ["mass_init", "logteff", "logl", "g", "bp", "rp"]


def _normalize_feh(feh: float) -> float:
    value = round(float(feh), 2)
    if value not in TARGET_FILES:
        raise ValueError(f"Unsupported [Fe/H]: {feh}. Expected one of {sorted(TARGET_FILES)}")
    return value


def _source_path(feh: float) -> Path:
    value = _normalize_feh(feh)
    return ISO_DIR / TARGET_FILES[value]


def _cache_path(feh: float) -> Path:
    value = _normalize_feh(feh)
    return CACHE_FILES[value]


def _parse_mist_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"MIST file not found: {path}")

    header_tokens: list[str] | None = None
    rows: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# EEP"):
                header_tokens = line[1:].split()
                continue
            if line.startswith("#"):
                continue
            if header_tokens is None:
                raise ValueError(f"Data row encountered before '# EEP' header in {path}")

            parts = line.split()
            if len(parts) != len(header_tokens):
                raise ValueError(
                    f"Column count mismatch in {path}: expected {len(header_tokens)}, got {len(parts)}"
                )
            rows.append([float(value) for value in parts])

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    frame = pd.DataFrame(rows, columns=header_tokens)
    missing = [name for name in SOURCE_COLUMNS if name not in frame.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns in {path.name}: {missing_str}")

    frame = frame[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    frame["eep"] = frame["eep"].astype(int)
    frame["phase"] = frame["phase"].astype(int)
    return frame[OUTPUT_COLUMNS]


def read_mist(feh: float) -> pd.DataFrame:
    value = _normalize_feh(feh)
    cache_path = _cache_path(value)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    frame = _parse_mist_file(_source_path(value))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


def _interpolate_mist_frame(feh: float) -> tuple[pd.DataFrame, int]:
    value = round(float(feh), 2)
    if value < 0.0 or value > 0.25:
        raise ValueError(f"Interpolated [Fe/H] must lie in [0.00, 0.25], got {feh}")
    if np.isclose(value, 0.0):
        return read_mist(0.00), 0
    if np.isclose(value, 0.25):
        return read_mist(0.25), 0

    low = read_mist(0.00)
    high = read_mist(0.25)
    merged = low.merge(high, on=["logage", "eep"], suffixes=("_low", "_high"), how="inner")
    phase_match = merged["phase_low"] == merged["phase_high"]
    dropped = int((~phase_match).sum())
    shared = merged.loc[phase_match].copy()

    weight = value / 0.25
    result = pd.DataFrame()
    result["eep"] = shared["eep"].astype(int)
    result["logage"] = shared["logage"]
    for column in INTERP_COLUMNS:
        result[column] = (1.0 - weight) * shared[f"{column}_low"] + weight * shared[f"{column}_high"]
    result["feh"] = value
    result["phase"] = shared["phase_low"].astype(int)
    return result[OUTPUT_COLUMNS].sort_values(["logage", "eep"]).reset_index(drop=True), dropped


def read_mist_interpolated(feh: float) -> pd.DataFrame:
    frame, _ = _interpolate_mist_frame(feh)
    return frame


def mist_logage_grid(frame: pd.DataFrame) -> np.ndarray:
    return np.sort(frame["logage"].unique())


def nearest_mist_logage(frame: pd.DataFrame, target_logage: float) -> float:
    logages = mist_logage_grid(frame)
    if logages.size == 0:
        raise ValueError("No logage values found")
    index = int(np.argmin(np.abs(logages - target_logage)))
    return float(logages[index])


def select_mist_logage(frame: pd.DataFrame, logage: float) -> pd.DataFrame:
    return frame.loc[np.isclose(frame["logage"], logage)].copy()


def build_mist_segments(frame: pd.DataFrame, allowed_phases: list[int] | tuple[int, ...]) -> list[tuple[str, pd.DataFrame]]:
    allowed = frame.loc[frame["phase"].isin(allowed_phases)].copy()
    segments: list[tuple[str, pd.DataFrame]] = []

    phase_02 = allowed.loc[allowed["phase"].isin([0, 2])].sort_values("eep")
    if not phase_02.empty:
        segments.append(("phase_0_2", phase_02))

    phase_3 = allowed.loc[allowed["phase"] == 3].sort_values("eep")
    if not phase_3.empty:
        segments.append(("phase_3", phase_3))

    return segments


def logage_to_myr(logage: float) -> float:
    return float(10**logage / 1.0e6)


def transform_mist_to_observed(
    frame: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float = 0.0,
) -> pd.DataFrame:
    transformed = frame.copy()
    transformed["g_model"] = transformed["g"] + dm + a_g
    transformed["colour_model"] = (transformed["bp"] - transformed["rp"]) + e_bprp + colour_offset
    return transformed


def build_mist_observed_segments(
    frame: pd.DataFrame,
    allowed_phases: list[int] | tuple[int, ...],
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float = 0.0,
) -> list[tuple[str, pd.DataFrame]]:
    segments: list[tuple[str, pd.DataFrame]] = []
    for name, segment in build_mist_segments(frame, allowed_phases):
        segments.append((name, transform_mist_to_observed(segment, dm, a_g, e_bprp, colour_offset)))
    return segments


def interpolate_phase0_colour(
    frame: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    g_values: np.ndarray,
    colour_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    phase0 = frame.loc[frame["phase"] == 0].sort_values("eep")
    if phase0.empty:
        raise ValueError("Phase-0 isochrone segment is empty")

    transformed = transform_mist_to_observed(phase0, dm, a_g, e_bprp, colour_offset).sort_values(["g_model", "eep"])
    unique_g, unique_indices = np.unique(transformed["g_model"].to_numpy(), return_index=True)
    unique_colour = transformed["colour_model"].to_numpy()[unique_indices]
    if unique_g.size < 2:
        raise ValueError("Phase-0 isochrone has fewer than two distinct G values")

    g_values = np.asarray(g_values, dtype=float)
    in_range = (g_values >= float(unique_g.min())) & (g_values <= float(unique_g.max()))
    interp_colour = np.full(g_values.shape, np.nan, dtype=float)
    interp_colour[in_range] = np.interp(g_values[in_range], unique_g, unique_colour)
    return interp_colour, in_range


def polyline_min_distance2(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if polyline.ndim != 2 or polyline.shape[1] != 2:
        raise ValueError("polyline must have shape (m, 2)")
    if polyline.shape[0] == 0:
        raise ValueError("polyline must contain at least one point")

    if polyline.shape[0] == 1:
        delta = points - polyline[0]
        return np.sum(delta * delta, axis=1)

    best = np.full(points.shape[0], np.inf, dtype=float)
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        seg_norm2 = float(np.dot(segment, segment))
        if seg_norm2 <= 0.0:
            delta = points - start
            dist2 = np.sum(delta * delta, axis=1)
        else:
            rel = points - start
            t = np.clip((rel @ segment) / seg_norm2, 0.0, 1.0)
            projection = start + t[:, None] * segment
            delta = points - projection
            dist2 = np.sum(delta * delta, axis=1)
        best = np.minimum(best, dist2)
    return best


def min_distance2_to_segments(points: np.ndarray, polylines: list[np.ndarray]) -> np.ndarray:
    if not polylines:
        raise ValueError("At least one polyline segment is required")

    best = np.full(points.shape[0], np.inf, dtype=float)
    for polyline in polylines:
        best = np.minimum(best, polyline_min_distance2(points, polyline))
    return best


def refine_parabolic_minimum(logages: np.ndarray, losses: np.ndarray, index: int) -> tuple[float | None, float | None, str | None]:
    if index <= 0 or index >= len(logages) - 1:
        return None, None, "minimum is on the grid edge"

    x = logages[index - 1 : index + 2]
    y = losses[index - 1 : index + 2]
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0.0:
        return None, None, "parabola fit is not convex"

    vertex_x = float(-b / (2.0 * a))
    if vertex_x < float(x[0]) or vertex_x > float(x[-1]):
        return None, None, "parabola minimum lies outside the neighbour bracket"

    vertex_y = float(a * vertex_x * vertex_x + b * vertex_x + c)
    return vertex_x, vertex_y, None


def prepare_members(frame: pd.DataFrame, dm: float, a_g: float) -> pd.DataFrame:
    members = frame[
        frame["phot_g_mean_mag"].notna()
        & frame["phot_bp_mean_mag"].notna()
        & frame["phot_rp_mean_mag"].notna()
    ].copy()
    members["colour"] = members["phot_bp_mean_mag"] - members["phot_rp_mean_mag"]
    members["M_G"] = members["phot_g_mean_mag"] - dm - a_g
    return members


def distance_modulus(distance_pc: float) -> float:
    return float(5.0 * np.log10(distance_pc) - 5.0)


def _make_fit_logage_grid(logage_min: float, logage_max: float, logage_step: float) -> np.ndarray:
    return np.round(np.arange(logage_min, logage_max + 0.5 * logage_step, logage_step), 10)


def make_fit_logage_grid() -> np.ndarray:
    return _make_fit_logage_grid(ISO_LOGAGE_MIN, ISO_LOGAGE_MAX, ISO_LOGAGE_STEP)


def _compute_colour_offset(
    members: pd.DataFrame,
    isochrone: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    mg_range: tuple[float, float],
    drop_frac: float,
) -> tuple[float, int, int]:
    mg_min, mg_max = mg_range
    offset_members = members.loc[members["M_G"].gt(mg_min) & members["M_G"].lt(mg_max)].copy()
    if offset_members.empty:
        raise ValueError("No members in the colour-offset M_G window")

    interp_colour, in_range = interpolate_phase0_colour(
        isochrone,
        dm,
        a_g,
        e_bprp,
        offset_members["phot_g_mean_mag"].to_numpy(),
        colour_offset=0.0,
    )
    offset_members = offset_members.loc[in_range].copy()
    if offset_members.empty:
        raise ValueError("No colour-offset members overlap the phase-0 isochrone in G")

    offset_members["delta_colour"] = offset_members["colour"].to_numpy() - interp_colour[in_range]
    n_offset = len(offset_members)
    drop_n = min(int(np.floor(drop_frac * n_offset)), max(0, n_offset - 1))
    keep_n = n_offset - drop_n
    kept = offset_members.nsmallest(keep_n, columns="delta_colour")
    return float(kept["delta_colour"].median()), int(n_offset), int(drop_n)


def compute_colour_offset(
    members: pd.DataFrame,
    isochrone: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
) -> tuple[float, int, int]:
    return _compute_colour_offset(members, isochrone, dm, a_g, e_bprp, ISO_OFFSET_MG_RANGE, ISO_OFFSET_DROP_FRAC)


def compute_colour_offset_custom(
    members: pd.DataFrame,
    isochrone: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    mg_range: tuple[float, float],
    drop_frac: float,
) -> tuple[float, int, int]:
    return _compute_colour_offset(members, isochrone, dm, a_g, e_bprp, mg_range, drop_frac)


def _evaluate_loss_for_age(
    fit_members: pd.DataFrame,
    isochrone: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float,
    allowed_phases: list[int] | tuple[int, ...],
    sigma_colour: float,
    sigma_g: float,
    d_cap: float,
) -> tuple[float, float]:
    segments = build_mist_observed_segments(
        isochrone,
        allowed_phases,
        dm,
        a_g,
        e_bprp,
        colour_offset=colour_offset,
    )
    polylines = [
        np.column_stack([segment["colour_model"].to_numpy() / sigma_colour, segment["g_model"].to_numpy() / sigma_g])
        for _, segment in segments
        if not segment.empty
    ]
    if not polylines:
        raise ValueError("No isochrone segments available for the loss evaluation")

    points = np.column_stack([fit_members["colour"].to_numpy() / sigma_colour, fit_members["phot_g_mean_mag"].to_numpy() / sigma_g])
    distance2 = min_distance2_to_segments(points, polylines)
    cap2 = d_cap**2
    losses = np.minimum(distance2, cap2)
    frac_capped = float(np.mean(distance2 >= cap2))
    return float(losses.sum()), frac_capped


def evaluate_loss_for_age(
    fit_members: pd.DataFrame,
    isochrone: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float,
) -> tuple[float, float]:
    return _evaluate_loss_for_age(
        fit_members,
        isochrone,
        dm,
        a_g,
        e_bprp,
        colour_offset,
        ISO_ALLOWED_PHASES,
        ISO_SIGMA_COLOUR,
        ISO_SIGMA_G,
        ISO_D_CAP,
    )


def evaluate_loss_for_age_custom(
    fit_members: pd.DataFrame,
    isochrone: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float,
    allowed_phases: list[int] | tuple[int, ...],
    sigma_colour: float,
    sigma_g: float,
    d_cap: float,
) -> tuple[float, float]:
    return _evaluate_loss_for_age(
        fit_members,
        isochrone,
        dm,
        a_g,
        e_bprp,
        colour_offset,
        allowed_phases,
        sigma_colour,
        sigma_g,
        d_cap,
    )


def _fit_age_grid(
    fit_members: pd.DataFrame,
    isochrones: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float,
    logage_min: float,
    logage_max: float,
    logage_step: float,
    allowed_phases: list[int] | tuple[int, ...],
    sigma_colour: float,
    sigma_g: float,
    d_cap: float,
) -> dict[str, object]:
    logage_grid = _make_fit_logage_grid(logage_min, logage_max, logage_step)
    losses: list[float] = []
    frac_capped: list[float] = []

    for logage in logage_grid:
        age_frame = select_mist_logage(isochrones, logage)
        if age_frame.empty:
            raise ValueError(f"No isochrone rows found at logage={logage:.2f}")
        loss, frac = _evaluate_loss_for_age(
            fit_members,
            age_frame,
            dm,
            a_g,
            e_bprp,
            colour_offset,
            allowed_phases,
            sigma_colour,
            sigma_g,
            d_cap,
        )
        losses.append(loss)
        frac_capped.append(frac)

    loss_array = np.array(losses, dtype=float)
    frac_array = np.array(frac_capped, dtype=float)
    best_index = int(np.argmin(loss_array))
    best_grid_logage = float(logage_grid[best_index])
    best_grid_loss = float(loss_array[best_index])
    best_grid_age_myr = logage_to_myr(best_grid_logage)

    refined_logage, refined_loss, warning = refine_parabolic_minimum(logage_grid, loss_array, best_index)
    if refined_logage is None or refined_loss is None:
        best_logage = best_grid_logage
        best_loss = best_grid_loss
    else:
        best_logage = refined_logage
        best_loss = refined_loss

    minus_index = int(np.argmin(np.abs(logage_grid - (best_grid_logage - 0.2))))
    plus_index = int(np.argmin(np.abs(logage_grid - (best_grid_logage + 0.2))))

    return {
        "logage_grid": logage_grid,
        "losses": loss_array,
        "frac_capped": frac_array,
        "best_index": best_index,
        "best_grid_logage": best_grid_logage,
        "best_grid_age_myr": best_grid_age_myr,
        "best_logage": float(best_logage),
        "best_age_myr": logage_to_myr(float(best_logage)),
        "best_loss": float(best_loss),
        "warning": warning,
        "loss_minus_0p2": float(loss_array[minus_index]),
        "loss_plus_0p2": float(loss_array[plus_index]),
        "loss_minus_0p2_logage": float(logage_grid[minus_index]),
        "loss_plus_0p2_logage": float(logage_grid[plus_index]),
    }


def fit_age_grid(
    fit_members: pd.DataFrame,
    isochrones: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float,
) -> dict[str, object]:
    return _fit_age_grid(
        fit_members,
        isochrones,
        dm,
        a_g,
        e_bprp,
        colour_offset,
        ISO_LOGAGE_MIN,
        ISO_LOGAGE_MAX,
        ISO_LOGAGE_STEP,
        ISO_ALLOWED_PHASES,
        ISO_SIGMA_COLOUR,
        ISO_SIGMA_G,
        ISO_D_CAP,
    )


def fit_age_grid_custom(
    fit_members: pd.DataFrame,
    isochrones: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    colour_offset: float,
    logage_min: float,
    logage_max: float,
    logage_step: float,
    allowed_phases: list[int] | tuple[int, ...],
    sigma_colour: float,
    sigma_g: float,
    d_cap: float,
) -> dict[str, object]:
    return _fit_age_grid(
        fit_members,
        isochrones,
        dm,
        a_g,
        e_bprp,
        colour_offset,
        logage_min,
        logage_max,
        logage_step,
        allowed_phases,
        sigma_colour,
        sigma_g,
        d_cap,
    )


def run_isochrone_fit_cycle(
    members: pd.DataFrame,
    isochrones: pd.DataFrame,
    dm: float,
    a_g: float,
    e_bprp: float,
    literature_age_myr: float,
    fit_mg_max: float = ISO_FIT_MG_MAX,
    offset_mg_range: tuple[float, float] = ISO_OFFSET_MG_RANGE,
    offset_drop_frac: float = ISO_OFFSET_DROP_FRAC,
    logage_min: float = ISO_LOGAGE_MIN,
    logage_max: float = ISO_LOGAGE_MAX,
    logage_step: float = ISO_LOGAGE_STEP,
    allowed_phases: list[int] | tuple[int, ...] = ISO_ALLOWED_PHASES,
    sigma_colour: float = ISO_SIGMA_COLOUR,
    sigma_g: float = ISO_SIGMA_G,
    d_cap: float = ISO_D_CAP,
    fixed_offset: float | None = None,
) -> dict[str, object]:
    fit_members = members.loc[members["M_G"] < fit_mg_max].copy()
    if fit_members.empty:
        raise ValueError("No fit stars remain after applying the M_G fit window")

    literature_logage = nearest_mist_logage(isochrones, float(np.log10(literature_age_myr * 1.0e6)))
    literature_isochrone = select_mist_logage(isochrones, literature_logage)

    if fixed_offset is None:
        colour_offset_1, n_offset_1, dropped_1 = compute_colour_offset_custom(
            members,
            literature_isochrone,
            dm,
            a_g,
            e_bprp,
            offset_mg_range,
            offset_drop_frac,
        )
        fit_1 = fit_age_grid_custom(
            fit_members,
            isochrones,
            dm,
            a_g,
            e_bprp,
            colour_offset_1,
            logage_min,
            logage_max,
            logage_step,
            allowed_phases,
            sigma_colour,
            sigma_g,
            d_cap,
        )
        best_grid_isochrone_1 = select_mist_logage(isochrones, float(fit_1["best_grid_logage"]))
        colour_offset_2, n_offset_2, dropped_2 = compute_colour_offset_custom(
            members,
            best_grid_isochrone_1,
            dm,
            a_g,
            e_bprp,
            offset_mg_range,
            offset_drop_frac,
        )
        fit_2 = fit_age_grid_custom(
            fit_members,
            isochrones,
            dm,
            a_g,
            e_bprp,
            colour_offset_2,
            logage_min,
            logage_max,
            logage_step,
            allowed_phases,
            sigma_colour,
            sigma_g,
            d_cap,
        )
    else:
        colour_offset_1 = float(fixed_offset)
        colour_offset_2 = float(fixed_offset)
        n_offset_1 = 0
        n_offset_2 = 0
        dropped_1 = 0
        dropped_2 = 0
        fit_1 = fit_age_grid_custom(
            fit_members,
            isochrones,
            dm,
            a_g,
            e_bprp,
            colour_offset_1,
            logage_min,
            logage_max,
            logage_step,
            allowed_phases,
            sigma_colour,
            sigma_g,
            d_cap,
        )
        fit_2 = fit_1

    best_grid_isochrone_2 = select_mist_logage(isochrones, float(fit_2["best_grid_logage"]))
    return {
        "fit_members": fit_members,
        "literature_logage": literature_logage,
        "literature_isochrone": literature_isochrone,
        "colour_offset_1": float(colour_offset_1),
        "colour_offset_2": float(colour_offset_2),
        "n_offset_1": int(n_offset_1),
        "n_offset_2": int(n_offset_2),
        "dropped_1": int(dropped_1),
        "dropped_2": int(dropped_2),
        "fit_1": fit_1,
        "fit_2": fit_2,
        "best_grid_isochrone_2": best_grid_isochrone_2,
    }


def _logage_summary(frame: pd.DataFrame) -> tuple[int, float, float, float]:
    ages = np.sort(frame["logage"].unique())
    if ages.size == 0:
        raise ValueError("No logage values found")
    if ages.size == 1:
        step = float("nan")
    else:
        diffs = np.diff(ages)
        step = float(diffs[0])
    return int(ages.size), float(ages.min()), float(ages.max()), step


def _phase_counts_at_age(frame: pd.DataFrame, logage: float) -> pd.Series:
    subset = frame.loc[np.isclose(frame["logage"], logage), "phase"]
    return subset.value_counts().sort_index()


def _print_phase_counts(counts: pd.Series) -> None:
    if counts.empty:
        print("  none")
        return
    for phase, count in counts.items():
        print(f"  phase {int(phase)}: {int(count)}")


def _self_test() -> None:
    frames = {feh: read_mist(feh) for feh in sorted(TARGET_FILES)}

    for feh, frame in frames.items():
        source_name = TARGET_FILES[feh]
        n_age, min_age, max_age, step = _logage_summary(frame)
        print(f"[Fe/H]={feh:.2f} file: {source_name}")
        print(f"  rows: {len(frame)}")
        print(f"  logage distinct: {n_age}, min={min_age:.2f}, max={max_age:.2f}, step={step:.2f}")
        print("  phase counts at logage 9.60:")
        _print_phase_counts(_phase_counts_at_age(frame, 9.60))

    eep_000 = set(frames[0.00].loc[np.isclose(frames[0.00]["logage"], 9.60), "eep"].astype(int))
    eep_025 = set(frames[0.25].loc[np.isclose(frames[0.25]["logage"], 9.60), "eep"].astype(int))
    print("EEP counts at logage 9.60:")
    print(f"  [Fe/H]=0.00: {len(eep_000)}")
    print(f"  [Fe/H]=0.25: {len(eep_025)}")
    print(f"  shared: {len(eep_000 & eep_025)}")

    target = frames[0.00].loc[np.isclose(frames[0.00]["logage"], 9.65)].copy()
    if target.empty:
        raise ValueError("No rows found at logage 9.65 for [Fe/H]=0.00")
    target["mass_delta"] = (target["mass_init"] - 1.00).abs()
    row = target.sort_values(["mass_delta", "mass_init"]).iloc[0]
    colour = float(row["bp"] - row["rp"])
    print("Closest row to mass_init=1.00 at logage 9.65 for [Fe/H]=0.00:")
    print(
        f"  eep={int(row['eep'])}, mass_init={row['mass_init']:.6f}, "
        f"g={row['g']:.6f}, bp-rp={colour:.6f}"
    )

    interp_000, dropped_000 = _interpolate_mist_frame(0.00)
    interp_025, dropped_025 = _interpolate_mist_frame(0.25)
    diff_000 = (interp_000[["g", "bp", "rp"]] - frames[0.00][["g", "bp", "rp"]]).abs().to_numpy().max()
    diff_025 = (interp_025[["g", "bp", "rp"]] - frames[0.25][["g", "bp", "rp"]]).abs().to_numpy().max()
    interp_015, dropped_015 = _interpolate_mist_frame(0.15)
    shared_015 = interp_015.merge(
        frames[0.00][["logage", "eep", "bp", "rp"]].rename(columns={"bp": "bp_000", "rp": "rp_000"}),
        on=["logage", "eep"],
        how="inner",
    ).merge(
        frames[0.25][["logage", "eep", "bp", "rp"]].rename(columns={"bp": "bp_025", "rp": "rp_025"}),
        on=["logage", "eep"],
        how="inner",
    )
    colour_000 = shared_015["bp_000"] - shared_015["rp_000"]
    colour_015 = shared_015["bp"] - shared_015["rp"]
    colour_025 = shared_015["bp_025"] - shared_015["rp_025"]
    tol = 1.0e-12
    between = (
        (colour_015 >= np.minimum(colour_000, colour_025) - tol)
        & (colour_015 <= np.maximum(colour_000, colour_025) + tol)
    ).all()
    print("Interpolation self-test:")
    print(f"  feh=0.00 max abs difference on g,bp,rp: {float(diff_000):.6e}")
    print(f"  feh=0.25 max abs difference on g,bp,rp: {float(diff_025):.6e}")
    print(f"  feh=0.15 bp-rp between endpoints at every shared EEP: {bool(between)}")
    print(f"  dropped EEPs because phases differ: {dropped_015} (endpoints: {dropped_000}, {dropped_025})")


if __name__ == "__main__":
    _self_test()
