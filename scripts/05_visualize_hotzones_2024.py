"""
Script 5 — Visualize 2024 Hot Zones with MLB-style visuals

INPUT : data/processed/zone_weights_2024.csv  (built by Script 4)
OUTPUT: PNG files saved to /visualizations (VIZ_DIR)

What it does:
- Loads per-player/per-zone metrics (zone_weight, xwoba_shrunk, xwoba_contact, bip_count)
- Builds a GLOBAL league baseline map by zone_id (0..24)
- For any selected player, ensures a full 25-zone grid:
    - Missing zones filled neutrally using league baseline:
        bip_count = 0
        xwoba_shrunk = league_xwoba_contact
        zone_weight = 0
- Renders a 5×5 heatmap with:
    - Thin light-gray zone gridlines
    - Thick black strike-zone box (middle 3×3)
    - Home plate marker for orientation

Color rules:
- zone_weight: diverging blue-white-red centered at 0
- xwoba_shrunk / xwoba_contact: diverging blue-white-red centered at LEAGUE AVG xwOBA
  (blue below league avg, red above league avg)
- bip_count: sequential Blues

Run examples (from project root):
  python scripts/05_visualize_hotzones_2024.py --player "Aaron Judge" --metric zone_weight --annotate-values --annotate-counts
  python scripts/05_visualize_hotzones_2024.py --player "Aaron Judge" --metric xwoba_shrunk --annotate-values --annotate-counts
  python scripts/05_visualize_hotzones_2024.py --batter 592450 --metric zone_weight
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Polygon

# NOTE:
# Your config.py lives in /scripts (based on how it defines PROJECT_ROOT) :contentReference[oaicite:2]{index=2}
from config import PROCESSED_DIR, VIZ_DIR

NX = 5
NZ = 5
ALL_ZONES = list(range(NX * NZ))  # 0..24


# -----------------------------
# Loading + validation
# -----------------------------
def load_weights(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "batter",
        "zone_id",
        "bip_count",
        "league_xwoba_contact",
        "xwoba_shrunk",
        "zone_weight",
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if "player_name" not in df.columns:
        df["player_name"] = pd.NA

    # xwoba_contact is optional (script 4 outputs it, but we don’t require it)
    if "xwoba_contact" not in df.columns:
        df["xwoba_contact"] = np.nan

    if not df["zone_id"].between(0, 24).all():
        bad = df.loc[~df["zone_id"].between(0, 24), "zone_id"].unique()
        raise ValueError(f"zone_id out of 0..24 found: {bad}")

    return df


def build_league_map(df: pd.DataFrame) -> pd.Series:
    """
    Global baseline: zone_id -> league_xwoba_contact.
    IMPORTANT: must be computed from full df (not player subset).
    """
    league_map = df.groupby("zone_id")["league_xwoba_contact"].mean().reindex(ALL_ZONES)
    if league_map.isna().any():
        missing = league_map[league_map.isna()].index.tolist()
        raise ValueError(
            f"League baseline missing for zone_id(s): {missing}. "
            "This indicates Script 4 output does not include league baselines for all zones."
        )
    return league_map


def compute_league_avg_xwoba(df: pd.DataFrame) -> float:
    """
    Compute a single league-average xwOBA baseline for centering absolute xwOBA colormaps.
    If league_bip exists, use it for a weighted average. Otherwise use simple mean across zones.
    """
    cols = ["zone_id", "league_xwoba_contact"]
    d = df[cols].drop_duplicates("zone_id").copy()

    # Weighted average if league_bip exists (not guaranteed)
    if "league_bip" in df.columns:
        w = df[["zone_id", "league_bip"]].drop_duplicates("zone_id").set_index("zone_id")["league_bip"]
        d = d.set_index("zone_id")
        d["league_bip"] = w.reindex(d.index).fillna(0)
        if d["league_bip"].sum() > 0:
            return float((d["league_xwoba_contact"] * d["league_bip"]).sum() / d["league_bip"].sum())

    # Fallback: simple mean across zones
    return float(d["league_xwoba_contact"].mean())


# -----------------------------
# Fill missing zones
# -----------------------------
def fill_missing_zones_for_player(p: pd.DataFrame, league_map: pd.Series) -> pd.DataFrame:
    """
    Ensure the player has all 25 zones.
    Missing zones filled neutrally:
      bip_count = 0
      xwoba_shrunk = league_xwoba_contact
      zone_weight = 0
      xwoba_contact = NaN
    """
    batter_id = int(p["batter"].iloc[0])
    player_name = p["player_name"].iloc[0] if "player_name" in p.columns else pd.NA

    base = pd.DataFrame({"zone_id": ALL_ZONES})
    base["batter"] = batter_id
    base["player_name"] = player_name
    base["league_xwoba_contact"] = league_map.values

    keep = ["zone_id", "bip_count", "xwoba_shrunk", "zone_weight", "xwoba_contact"]
    filled = base.merge(p[keep], on="zone_id", how="left")

    filled["bip_count"] = filled["bip_count"].fillna(0).astype(int)
    filled["xwoba_shrunk"] = filled["xwoba_shrunk"].fillna(filled["league_xwoba_contact"])
    filled["zone_weight"] = filled["zone_weight"].fillna(0.0)
    # xwoba_contact can stay NaN

    return filled.sort_values("zone_id").reset_index(drop=True)


# -----------------------------
# Plot helpers
# -----------------------------
def zones_to_matrix(values_by_zone: pd.Series) -> np.ndarray:
    """
    zone_id -> (z, x) using zone_id = z*5 + x.
    Flip vertically so z=4 appears at the top.
    """
    arr = np.full((NZ, NX), np.nan, dtype=float)
    for zone_id, val in values_by_zone.items():
        z = int(zone_id) // NX
        x = int(zone_id) % NX
        arr[z, x] = float(val) if pd.notna(val) else np.nan
    return np.flipud(arr)


def add_strike_zone_box(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle(
            (0.5, 0.5),
            3.0,
            3.0,
            fill=False,
            edgecolor="black",
            linewidth=3.0,
            zorder=6,
        )
    )


def add_home_plate(ax: plt.Axes) -> None:
    cx = 2.0
    y = -0.85
    plate = Polygon(
        [
            (cx - 0.35, y + 0.20),
            (cx + 0.35, y + 0.20),
            (cx + 0.35, y - 0.05),
            (cx,        y - 0.35),
            (cx - 0.35, y - 0.05),
        ],
        closed=True,
        facecolor="white",
        edgecolor="black",
        linewidth=2.0,
        zorder=7,
        clip_on=False,
    )
    ax.add_patch(plate)


def safe_player_stub(player_name: str | float, batter_id: int) -> str:
    if player_name is None or pd.isna(player_name) or str(player_name).strip() == "":
        return f"batter_{batter_id}"
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "" for ch in str(player_name)).strip()
    safe = safe.replace(" ", "_")
    return f"{safe}_{batter_id}"


def choose_cmap_and_norm(
    metric: str,
    value_mat: np.ndarray,
    fixed_limit: float | None,
    league_avg_xwoba: float,
):
    """
    Returns (cmap, norm, label)
    """
    # Diverging centered at 0 for relative weights
    if metric == "zone_weight":
        if fixed_limit is not None and fixed_limit > 0:
            limit = float(fixed_limit)
        else:
            vmin = float(np.nanmin(value_mat))
            vmax = float(np.nanmax(value_mat))
            limit = max(abs(vmin), abs(vmax))
            if not np.isfinite(limit) or limit == 0:
                limit = 0.01

        cmap = "bwr"
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        label = "zone_weight (player − league zone baseline), centered at 0"
        return cmap, norm, label

    # Absolute xwOBA metrics but colored relative to league average
    if metric in ("xwoba_shrunk", "xwoba_contact"):
        vmin = float(np.nanmin(value_mat))
        vmax = float(np.nanmax(value_mat))

        # Spread around league avg so it is centered
        spread = max(abs(vmax - league_avg_xwoba), abs(league_avg_xwoba - vmin))
        if not np.isfinite(spread) or spread == 0:
            spread = 0.01

        vmin = league_avg_xwoba - spread
        vmax = league_avg_xwoba + spread

        cmap = "bwr"
        norm = TwoSlopeNorm(vmin=vmin, vcenter=league_avg_xwoba, vmax=vmax)
        label = f"{metric} (absolute), centered at league avg {league_avg_xwoba:.3f}"
        return cmap, norm, label

    # Counts
    if metric == "bip_count":
        cmap = "Blues"
        vmin = 0
        vmax = float(np.nanmax(value_mat))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        label = "bip_count"
        return cmap, norm, label

    # Fallback
    cmap = "viridis"
    norm = None
    label = metric
    return cmap, norm, label


def plot_player_heatmap(
    player_df: pd.DataFrame,
    metric: str,
    annotate_values: bool,
    annotate_counts: bool,
    out_dir: Path,
    fixed_limit: float | None,
    league_avg_xwoba: float,
) -> Path:
    player_name = player_df["player_name"].iloc[0]
    batter_id = int(player_df["batter"].iloc[0])

    title_name = str(player_name).strip() if pd.notna(player_name) and str(player_name).strip() else f"batter {batter_id}"
    file_stub = safe_player_stub(player_name, batter_id)

    if metric not in player_df.columns:
        raise ValueError(f"Metric '{metric}' not in data. Available: {list(player_df.columns)}")

    value_mat = zones_to_matrix(player_df.set_index("zone_id")[metric])
    count_mat = zones_to_matrix(player_df.set_index("zone_id")["bip_count"])

    cmap, norm, cbar_label = choose_cmap_and_norm(metric, value_mat, fixed_limit, league_avg_xwoba)

    fig, ax = plt.subplots(figsize=(6.6, 6.8))
    im = ax.imshow(
        value_mat,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
        zorder=1,
    )

    ax.set_title(f"{title_name} — 2024 {metric}")
    ax.set_xlabel("x_bin (0→4)")
    ax.set_ylabel("z_bin (4→0 shown top→bottom)")

    ax.set_xticks(range(NX))
    ax.set_yticks(range(NZ))
    ax.set_xticklabels([str(i) for i in range(NX)])
    ax.set_yticklabels([str(i) for i in range(NZ)][::-1])

    # Thin zone gridlines
    ax.set_xticks(np.arange(-0.5, NX, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NZ, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=1.0, zorder=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Strike zone + plate
    add_strike_zone_box(ax)
    add_home_plate(ax)
    ax.set_ylim(NZ - 0.5, -1.3)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    # Annotations
    if annotate_values or annotate_counts:
        for r in range(NZ):
            for c in range(NX):
                v = value_mat[r, c]
                n = int(round(count_mat[r, c])) if not np.isnan(count_mat[r, c]) else 0

                lines = []
                if annotate_values:
                    if np.isnan(v):
                        lines.append("NA")
                    else:
                        if metric == "bip_count":
                            lines.append(f"{int(round(v))}")
                        else:
                            lines.append(f"{v:.3f}")

                if annotate_counts and metric != "bip_count":
                    lines.append(f"n={n}")

                if lines:
                    ax.text(c, r, "\n".join(lines), ha="center", va="center", fontsize=8, color="black", zorder=5)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hotzone_2024_{metric}_{file_stub}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


# -----------------------------
# Player selection
# -----------------------------
def select_player(df: pd.DataFrame, player: str | None, batter: int | None) -> pd.DataFrame:
    if batter is not None:
        sub = df[df["batter"] == batter].copy()
        if sub.empty:
            raise ValueError(f"No rows found for batter id {batter}")
        return sub

    if player is None or player.strip() == "":
        raise ValueError('Provide --player "First Last" or --batter <id>.')

    q = player.strip().lower()

    # exact match
    sub = df[df["player_name"].fillna("").str.lower() == q].copy()
    if not sub.empty:
        return sub

    # contains match
    sub = df[df["player_name"].fillna("").str.lower().str.contains(q, na=False)].copy()
    if sub.empty:
        raise ValueError(f"No rows found matching '{player}'. Try different spelling or use --batter.")
    return sub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=str, default=None, help='Player name (e.g., "Aaron Judge")')
    parser.add_argument("--batter", type=int, default=None, help="MLBAM batter id (e.g., 592450)")
    parser.add_argument("--metric", type=str, default="zone_weight",
                        help="zone_weight | xwoba_shrunk | xwoba_contact | bip_count")
    parser.add_argument("--annotate-values", action="store_true", help="Write metric value in each cell")
    parser.add_argument("--annotate-counts", action="store_true", help="Write n=BIP count in each cell")
    parser.add_argument("--top", type=int, default=None, help="Batch mode: plot top N hitters by mean zone_weight")
    parser.add_argument("--fixed-limit", type=float, default=0.15,
                        help="For zone_weight only: symmetric limit. Use 0 for auto per player.")
    args = parser.parse_args()

    in_path = PROCESSED_DIR / "zone_weights_2024.csv"
    df = load_weights(in_path)

    league_map = build_league_map(df)
    league_avg_xwoba = compute_league_avg_xwoba(df)

    fixed_limit = None if (args.fixed_limit is not None and args.fixed_limit <= 0) else args.fixed_limit

    # Batch mode
    if args.top is not None:
        if args.top <= 0:
            raise ValueError("--top must be positive")

        rank = (
            df.groupby(["batter", "player_name"], as_index=False)["zone_weight"]
            .mean()
            .sort_values("zone_weight", ascending=False)
            .head(args.top)
        )

        print(f"Plotting top {args.top} hitters by mean zone_weight...")
        for _, row in rank.iterrows():
            batter_id = int(row["batter"])
            p = df[df["batter"] == batter_id].copy()
            p = fill_missing_zones_for_player(p, league_map)

            out = plot_player_heatmap(
                p,
                metric=args.metric,
                annotate_values=args.annotate_values,
                annotate_counts=args.annotate_counts,
                out_dir=VIZ_DIR,
                fixed_limit=fixed_limit if args.metric == "zone_weight" else None,
                league_avg_xwoba=league_avg_xwoba,
            )
            print("Saved:", out)
        return

    # Single player mode
    p = select_player(df, args.player, args.batter)
    p = fill_missing_zones_for_player(p, league_map)

    out = plot_player_heatmap(
        p,
        metric=args.metric,
        annotate_values=args.annotate_values,
        annotate_counts=args.annotate_counts,
        out_dir=VIZ_DIR,
        fixed_limit=fixed_limit if args.metric == "zone_weight" else None,
        league_avg_xwoba=league_avg_xwoba,
    )
    print("Saved:", out)


if __name__ == "__main__":
    main()