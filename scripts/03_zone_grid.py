"""
Assign each Statcast pitch to a 5x5 zone grid (2024), producing statcast_zones_2024.csv.

This version fixes edge-handling AND supports a "majority of baseball area" rule:

- Default / recommended behavior:
    Assign ONE zone per pitch (no double counting).
- If USE_BALL_MAJORITY = True:
    Treat (plate_x, plate_z) as the BALL CENTER and assign the zone where the
    largest fraction of the baseball's cross-sectional area falls, using a
    fast Monte Carlo overlap method (only runs near boundaries for speed).

Outputs columns:
- x_bin, z_bin in [0..4]
- zone_id in [0..24] row-major (z first): zone_id = z_bin*5 + x_bin
- in_zone_3x3 flag (middle 3x3 block)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

# -----------------------------
# Grid definition (5x5)
# -----------------------------
# Full analysis region (outer bounds): x in [-2, 2], z in [0, 5]
# Strike zone sits inside the middle 3x3 using x in [-0.83, 0.83] and z in [1.5, 3.5]
X_EDGES = np.array([-2.00, -0.83, -0.28, 0.28, 0.83, 2.00], dtype=float)  # 5 columns
Z_EDGES = np.array([0.00,  1.50,  2.17, 2.83, 3.50, 5.00], dtype=float)   # 5 rows

NX = 5
NZ = 5

# -----------------------------
# Behavior toggles
# -----------------------------
USE_BALL_MAJORITY = True      # True = choose zone with majority ball overlap area
DROP_OUTSIDE_REGION = True    # True = drop pitches whose CENTER is outside [-2,2]x[0,5]
N_SAMPLES_NEAR_EDGE = 250     # Monte Carlo samples used near boundaries (250-400 is plenty)

# -----------------------------
# Baseball size (inches -> feet)
# -----------------------------
# MLB baseball diameter is about ~2.9 inches
BASEBALL_DIAMETER_IN = 2.9
BASEBALL_RADIUS_FT = (BASEBALL_DIAMETER_IN / 2.0) / 12.0


def _bin_index_left_closed(value: float, edges: np.ndarray) -> int:
    """
    Bin index for [edges[i], edges[i+1]) with clamping to valid range.
    """
    idx = np.searchsorted(edges, value, side="right") - 1
    return int(np.clip(idx, 0, len(edges) - 2))


def _seed_from_row(row: pd.Series, fallback_i: int) -> int:
    """
    Deterministic per-pitch seed. Uses game_pk + pitch_number if present; otherwise fallback index.
    """
    if "game_pk" in row and "pitch_number" in row and pd.notna(row["game_pk"]) and pd.notna(row["pitch_number"]):
        key = (int(row["game_pk"]), int(row["pitch_number"]))
    elif "pitch_id" in row and pd.notna(row["pitch_id"]):
        key = (int(row["pitch_id"]),)
    else:
        key = (int(fallback_i),)

    # Make stable 32-bit seed
    return abs(hash(key)) % (2**32)


def zone_by_ball_area_majority(
    xc: float,
    zc: float,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    radius_ft: float,
    n_samples: int,
    seed: int,
) -> tuple[int, int]:
    """
    Monte Carlo: sample points uniformly in the circle representing the baseball cross-section.
    Assign to zone with largest sampled area overlap.

    Returns: (x_bin, z_bin)
    """
    rng = np.random.default_rng(seed)

    # Uniform points in circle:
    # r = R * sqrt(u), theta = 2*pi*v
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    r = radius_ft * np.sqrt(u)
    theta = 2.0 * np.pi * v

    xs = xc + r * np.cos(theta)
    zs = zc + r * np.sin(theta)

    bx = np.searchsorted(x_edges, xs, side="right") - 1
    bz = np.searchsorted(z_edges, zs, side="right") - 1
    bx = np.clip(bx, 0, len(x_edges) - 2)
    bz = np.clip(bz, 0, len(z_edges) - 2)

    # Count samples per cell using packed ids
    nx = len(x_edges) - 1
    ids = (bz * nx + bx).astype(int)
    best_id = np.bincount(ids).argmax()

    best_bz = best_id // nx
    best_bx = best_id % nx
    return int(best_bx), int(best_bz)


def assign_zone_row(row: pd.Series, i: int) -> tuple[int, int]:
    """
    Assign (x_bin, z_bin) for one pitch.
    If USE_BALL_MAJORITY:
        use center-bin unless within 1 ball radius of a grid line, then Monte Carlo majority area.
    Otherwise:
        simple center-bin via [low, high) bins (left-closed/right-open).
    """
    xc = float(row["plate_x"])
    zc = float(row["plate_z"])

    # First: center bin (single-assignment, no overlaps)
    x_bin = _bin_index_left_closed(xc, X_EDGES)
    z_bin = _bin_index_left_closed(zc, Z_EDGES)

    if not USE_BALL_MAJORITY:
        return x_bin, z_bin

    # Only do Monte Carlo if we're within one ball radius of any relevant edge
    R = BASEBALL_RADIUS_FT

    left, right = X_EDGES[x_bin], X_EDGES[x_bin + 1]
    bottom, top = Z_EDGES[z_bin], Z_EDGES[z_bin + 1]

    near_x_edge = (xc - left) < R or (right - xc) < R
    near_z_edge = (zc - bottom) < R or (top - zc) < R

    if not (near_x_edge or near_z_edge):
        return x_bin, z_bin

    seed = _seed_from_row(row, i)
    return zone_by_ball_area_majority(
        xc=xc,
        zc=zc,
        x_edges=X_EDGES,
        z_edges=Z_EDGES,
        radius_ft=R,
        n_samples=N_SAMPLES_NEAR_EDGE,
        seed=seed,
    )


def main() -> None:
    in_path = PROCESSED_DIR / "statcast_cleaned_2024.csv"
    out_path = PROCESSED_DIR / "statcast_zones_2024.csv"

    print(f"Loading cleaned dataset: {in_path}")
    df = pd.read_csv(in_path)

    # Basic column check
    required = {"plate_x", "plate_z"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    before = len(df)

    # Optionally drop pitches whose CENTER is outside your 5x5 analysis region
    if DROP_OUTSIDE_REGION:
        x_min, x_max = X_EDGES[0], X_EDGES[-1]
        z_min, z_max = Z_EDGES[0], Z_EDGES[-1]
        mask = (df["plate_x"] >= x_min) & (df["plate_x"] <= x_max) & (df["plate_z"] >= z_min) & (df["plate_z"] <= z_max)
        df = df.loc[mask].copy()

    after_region = len(df)

    print("Assigning zones...")
    # Assign zones row-wise (fast enough, especially with the 'near boundary only' Monte Carlo)
    xb = np.empty(len(df), dtype=int)
    zb = np.empty(len(df), dtype=int)

    # Use itertuples for speed
    # We'll still construct a Series-like access via row._asdict() minimally; faster approach:
    # keep row as Series by iterating over df.iterrows() (slower) OR do a hybrid:
    # Here: iterate over positions and access df.iloc[i] (OK for moderate sizes)
    for j in range(len(df)):
        row = df.iloc[j]
        x_bin, z_bin = assign_zone_row(row, j)
        xb[j] = x_bin
        zb[j] = z_bin

    df["x_bin"] = xb
    df["z_bin"] = zb

    # zone_id in [0..24], row-major (z first then x)
    df["zone_id"] = df["z_bin"] * NX + df["x_bin"]

    # Flag inside the middle 3x3 strike-zone block (bins 1,2,3 for both x and z)
    df["in_zone_3x3"] = df["x_bin"].between(1, 3) & df["z_bin"].between(1, 3)

    print(f"Rows before region filter: {before}")
    print(f"Rows after region filter : {after_region}")
    print("Zone assignment complete.")
    print(f"USE_BALL_MAJORITY={USE_BALL_MAJORITY}, R={BASEBALL_RADIUS_FT:.5f} ft, N_SAMPLES_NEAR_EDGE={N_SAMPLES_NEAR_EDGE}")

    df.to_csv(out_path, index=False)
    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()