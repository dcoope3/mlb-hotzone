"""
Script 5 — Visualize 2024 Hot Zones (5x5 grid)

INPUT : data/processed/zone_weights_2024.csv
OUTPUT: PNG files saved to /visualizations (VIZ_DIR)

Supported metrics:
- zone_weight
- xwoba_shrunk
- xwoba_contact
- bip_count
- pitches_seen
- babip
- contact_rate

CLI (matches app.py):
  python scripts/05_visualize_hotzones_2024.py --batter 592450 --metric contact_rate --season 2024 --annotate-values --annotate-counts --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import PROCESSED_DIR, VIZ_DIR
from utils_paths import heatmap_path

NX = 5
NZ = 5
ALL_ZONES = list(range(NX * NZ))  # 0..24

VALID_METRICS = {
    "zone_weight",
    "xwoba_shrunk",
    "xwoba_contact",
    "bip_count",
    "pitches_seen",
    "babip",
    "contact_rate",
}


def load_weights(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "batter",
        "zone_id",
        "player_name",

        # league baselines
        "league_xwoba_contact",
        "league_babip",
        "league_contact_rate",

        # xwOBA contact surface
        "bip_count",
        "xwoba_contact",
        "xwoba_shrunk",
        "zone_weight",

        # volume
        "pitches_seen",

        # BABIP
        "bip",
        "babip",

        # Contact%
        "swings",
        "contact_rate",
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(
            f"zone_weights_2024.csv missing columns: {missing}\n"
            "Rebuild with Script 4 (the Contact% + BABIP version)."
        )

    df["batter"] = df["batter"].astype(int)
    df["zone_id"] = df["zone_id"].astype(int)

    if not df["zone_id"].between(0, 24).all():
        bad = df.loc[~df["zone_id"].between(0, 24), "zone_id"].unique()
        raise ValueError(f"zone_id out of 0..24 found: {bad}")

    return df


def zones_to_matrix(values_by_zone: pd.Series) -> np.ndarray:
    arr = np.full((NZ, NX), np.nan, dtype=float)
    for zone_id, val in values_by_zone.items():
        z = int(zone_id) // NX
        x = int(zone_id) % NX
        arr[z, x] = float(val) if pd.notna(val) else np.nan
    return np.flipud(arr)


def add_strike_zone_box(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle((0.5, 0.5), 3.0, 3.0, fill=False, edgecolor="black", linewidth=3.0, zorder=6)
    )


def add_home_plate(ax: plt.Axes) -> None:
    cx = 2.0
    y = -0.85
    plate = Polygon(
        [
            (cx - 0.35, y + 0.20),
            (cx + 0.35, y + 0.20),
            (cx + 0.35, y - 0.05),
            (cx, y - 0.35),
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


def league_zone_map(df: pd.DataFrame, col: str) -> pd.Series:
    s = df.groupby("zone_id")[col].mean().reindex(ALL_ZONES)
    if s.isna().any():
        missing = s[s.isna()].index.tolist()
        raise ValueError(f"League map missing '{col}' for zones: {missing}")
    return s


def fill_missing_zones_for_player(
    p: pd.DataFrame,
    league_xwoba_map: pd.Series,
    league_babip_map: pd.Series,
    league_contact_map: pd.Series,
) -> pd.DataFrame:
    batter_id = int(p["batter"].iloc[0])
    player_name = str(p["player_name"].iloc[0] or "")

    base = pd.DataFrame({"zone_id": ALL_ZONES})
    base["batter"] = batter_id
    base["player_name"] = player_name

    base["league_xwoba_contact"] = league_xwoba_map.values
    base["league_babip"] = league_babip_map.values
    base["league_contact_rate"] = league_contact_map.values

    keep = [
        "zone_id",

        # volume
        "pitches_seen",

        # BABIP
        "bip",
        "babip",

        # xwOBA contact surface
        "bip_count",
        "xwoba_contact",
        "xwoba_shrunk",
        "zone_weight",

        # Contact%
        "swings",
        "contact_rate",
    ]
    filled = base.merge(p[keep], on="zone_id", how="left")

    # volume
    filled["pitches_seen"] = filled["pitches_seen"].fillna(0).astype(int)

    # xwOBA contact surface
    filled["bip_count"] = filled["bip_count"].fillna(0).astype(int)
    filled["xwoba_contact"] = filled["xwoba_contact"].fillna(filled["league_xwoba_contact"])
    filled["xwoba_shrunk"] = filled["xwoba_shrunk"].fillna(filled["league_xwoba_contact"])
    filled["zone_weight"] = filled["zone_weight"].fillna(0.0)

    # BABIP
    filled["bip"] = filled["bip"].fillna(0).astype(int)
    filled["babip"] = filled["babip"].fillna(filled["league_babip"])

    # Contact%
    filled["swings"] = filled["swings"].fillna(0).astype(int)
    filled["contact_rate"] = filled["contact_rate"].fillna(filled["league_contact_rate"])

    return filled.sort_values("zone_id").reset_index(drop=True)


def display_metric_name(metric: str) -> str:
    mapping = {
        "zone_weight": "Zone Weight",
        "xwoba_shrunk": "xwOBA Shrunk",
        "xwoba_contact": "xwOBA Contact",
        "bip_count": "BIP Count",
        "pitches_seen": "Pitches Seen",
        "babip": "BABIP",
        "contact_rate": "Contact Rate",
    }
    return mapping.get(metric, metric)


def display_player_name(player_name: str, batter_id: int) -> str:
    clean = str(player_name or "").strip()
    if not clean:
        return f"Batter {batter_id}"
    return clean.title()


def choose_cmap_norm_label(
    metric: str,
    value_mat: np.ndarray,
    league_avg_xwoba: float,
    league_avg_babip: float,
    league_avg_contact: float,
    fixed_limit: float | None,
):
    if metric == "zone_weight":
        if fixed_limit is not None and fixed_limit > 0:
            limit = float(fixed_limit)
        else:
            vmin = float(np.nanmin(value_mat))
            vmax = float(np.nanmax(value_mat))
            limit = max(abs(vmin), abs(vmax))
            if not np.isfinite(limit) or limit == 0:
                limit = 0.01
        return (
            "bwr",
            TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            "Zone Weight (Player − League), Centered at 0",
        )

    if metric in ("xwoba_shrunk", "xwoba_contact"):
        vmin = float(np.nanmin(value_mat))
        vmax = float(np.nanmax(value_mat))
        spread = max(abs(vmax - league_avg_xwoba), abs(league_avg_xwoba - vmin))
        if not np.isfinite(spread) or spread == 0:
            spread = 0.01
        vmin = league_avg_xwoba - spread
        vmax = league_avg_xwoba + spread
        metric_label = "xwOBA Shrunk" if metric == "xwoba_shrunk" else "xwOBA Contact"
        return (
            "bwr",
            TwoSlopeNorm(vmin=vmin, vcenter=league_avg_xwoba, vmax=vmax),
            f"{metric_label} (Centered at League Avg {league_avg_xwoba:.3f})",
        )

    if metric == "babip":
        vmin = float(np.nanmin(value_mat))
        vmax = float(np.nanmax(value_mat))
        spread = max(abs(vmax - league_avg_babip), abs(league_avg_babip - vmin))
        if not np.isfinite(spread) or spread == 0:
            spread = 0.02
        vmin = league_avg_babip - spread
        vmax = league_avg_babip + spread
        return (
            "bwr",
            TwoSlopeNorm(vmin=vmin, vcenter=league_avg_babip, vmax=vmax),
            f"BABIP (Centered at League Avg {league_avg_babip:.3f})",
        )

    if metric == "contact_rate":
        vmin = float(np.nanmin(value_mat))
        vmax = float(np.nanmax(value_mat))
        spread = max(abs(vmax - league_avg_contact), abs(league_avg_contact - vmin))
        if not np.isfinite(spread) or spread == 0:
            spread = 0.02
        vmin = league_avg_contact - spread
        vmax = league_avg_contact + spread
        return (
            "bwr",
            TwoSlopeNorm(vmin=vmin, vcenter=league_avg_contact, vmax=vmax),
            f"Contact Rate (Centered at League Avg {league_avg_contact:.3f})",
        )

    if metric in ("bip_count", "pitches_seen"):
        vmax = float(np.nanmax(value_mat))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        label = "BIP Count" if metric == "bip_count" else "Pitches Seen"
        return "Blues", plt.Normalize(vmin=0.0, vmax=vmax), label

    return "viridis", None, display_metric_name(metric)


def get_count_matrix_for_metric(player_df: pd.DataFrame, metric: str) -> np.ndarray:
    """
    What sample size should `--annotate-counts` show?
    - xwOBA metrics: bip_count
    - BABIP: bip
    - contact_rate: swings
    """
    if metric in ("xwoba_shrunk", "xwoba_contact", "zone_weight"):
        return zones_to_matrix(player_df.set_index("zone_id")["bip_count"])
    if metric == "babip":
        return zones_to_matrix(player_df.set_index("zone_id")["bip"])
    if metric == "contact_rate":
        return zones_to_matrix(player_df.set_index("zone_id")["swings"])
    return np.full((NZ, NX), np.nan, dtype=float)


def load_batter_handedness(batter_id: int) -> str | None:
    """
    Visualization-only helper.
    Uses the zoned Statcast file if available.
    Does not change any calculations or metric definitions.
    """
    zoned_path = PROCESSED_DIR / "statcast_zones_2024.csv"
    if not zoned_path.exists():
        return None

    try:
        z = pd.read_csv(zoned_path, usecols=["batter", "stand"])
    except Exception:
        return None

    if "batter" not in z.columns or "stand" not in z.columns:
        return None

    z = z.dropna(subset=["batter"]).copy()
    z["batter"] = z["batter"].astype(int)

    batter_rows = (
        z.loc[z["batter"] == int(batter_id), "stand"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if batter_rows.empty:
        return None

    mode_vals = batter_rows.mode()
    if mode_vals.empty:
        return None

    hand = mode_vals.iloc[0]
    if hand in {"L", "R"}:
        return hand

    return None


def handedness_text_and_side(hand: str | None) -> tuple[str, str]:
    """
    Returns:
    - label text
    - visual side of chart

    Viewer-facing chart convention:
    - Right-handed batter label on left side
    - Left-handed batter label on right side
    - Unknown handedness goes in the right-handed batter box position (left side)
    """
    if hand == "L":
        return "Left-Handed Batter", "right"
    if hand == "R":
        return "Right-Handed Batter", "left"
    return "Unknown Handedness", "left"


def add_handedness_text(ax: plt.Axes, hand: str | None) -> str:
    label, side = handedness_text_and_side(hand)

    x_pos = -0.95 if side == "left" else 4.95
    ha = "right" if side == "left" else "left"

    ax.text(
        x_pos,
        2.0,
        label,
        rotation=90,
        ha=ha,
        va="center",
        fontsize=11,
        fontweight="bold",
        color="black",
        clip_on=False,
        zorder=8,
    )
    return side


def add_colorbar_opposite_side(fig: plt.Figure, ax: plt.Axes, im, batter_side: str, label: str):
    cbar_side = "right" if batter_side == "left" else "left"

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(cbar_side, size="4.5%", pad=0.30)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label)

    if cbar_side == "left":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    else:
        cax.yaxis.set_ticks_position("right")
        cax.yaxis.set_label_position("right")

    return cbar


def plot_heatmap(
    player_df: pd.DataFrame,
    metric: str,
    season: int,
    annotate_values: bool,
    annotate_counts: bool,
    fixed_limit: float | None,
) -> Path:
    raw_player_name = str(player_df["player_name"].iloc[0] or "")
    batter_id = int(player_df["batter"].iloc[0])
    title_name = display_player_name(raw_player_name, batter_id)
    metric_display = display_metric_name(metric)

    value_mat = zones_to_matrix(player_df.set_index("zone_id")[metric])

    league_avg_xwoba = float(player_df["league_xwoba_contact"].drop_duplicates().mean())
    league_avg_babip = float(player_df["league_babip"].drop_duplicates().mean())
    league_avg_contact = float(player_df["league_contact_rate"].drop_duplicates().mean())

    cmap, norm, label = choose_cmap_norm_label(
        metric,
        value_mat,
        league_avg_xwoba,
        league_avg_babip,
        league_avg_contact,
        fixed_limit,
    )

    batter_hand = load_batter_handedness(batter_id)

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    im = ax.imshow(value_mat, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest", zorder=1)

    ax.set_title(f"{title_name} — {season} {metric_display}", fontsize=14, pad=12)

    # Remove axis labels and numbered bin labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    # Keep gridlines only
    ax.set_xticks(np.arange(-0.5, NX, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NZ, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=1.0, zorder=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    add_strike_zone_box(ax)
    add_home_plate(ax)

    batter_side = add_handedness_text(ax, batter_hand)

    ax.set_ylim(NZ - 0.5, -1.3)
    ax.set_xlim(-0.5, NX - 0.5)

    add_colorbar_opposite_side(fig, ax, im, batter_side, label)

    count_mat = get_count_matrix_for_metric(player_df, metric)

    if annotate_values or annotate_counts:
        for r in range(NZ):
            for c in range(NX):
                v = value_mat[r, c]
                n = count_mat[r, c]
                n_int = int(round(n)) if (annotate_counts and not np.isnan(n)) else None

                lines = []

                if annotate_values:
                    if np.isnan(v):
                        lines.append("NA")
                    else:
                        if metric in ("bip_count", "pitches_seen"):
                            lines.append(f"{int(round(v))}")
                        else:
                            lines.append(f"{v:.3f}")

                if annotate_counts:
                    if metric not in ("bip_count", "pitches_seen") and n_int is not None:
                        lines.append(f"n={n_int}")

                if lines:
                    ax.text(
                        c,
                        r,
                        "\n".join(lines),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                        zorder=5,
                    )

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = heatmap_path(VIZ_DIR, season, metric, raw_player_name, batter_id)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=str, default=None)
    parser.add_argument("--batter", type=int, default=None)
    parser.add_argument("--metric", type=str, default="zone_weight", choices=sorted(VALID_METRICS))
    parser.add_argument("--annotate-values", action="store_true")
    parser.add_argument("--annotate-counts", action="store_true")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fixed-limit", type=float, default=0.15)
    args = parser.parse_args()

    df = load_weights(PROCESSED_DIR / "zone_weights_2024.csv")

    if args.batter is not None:
        p = df[df["batter"] == int(args.batter)].copy()
        if p.empty:
            raise ValueError(f"No rows found for batter id {args.batter}")
    else:
        if not args.player or not args.player.strip():
            raise ValueError('Provide --player "First Last" or --batter <id>.')
        q = args.player.strip().lower()
        p = df[df["player_name"].fillna("").str.lower() == q].copy()
        if p.empty:
            p = df[df["player_name"].fillna("").str.lower().str.contains(q, na=False)].copy()
        if p.empty:
            raise ValueError(f"No rows found matching '{args.player}'")

    league_xwoba_map = league_zone_map(df, "league_xwoba_contact")
    league_babip_map = league_zone_map(df, "league_babip")
    league_contact_map = league_zone_map(df, "league_contact_rate")

    p = fill_missing_zones_for_player(p, league_xwoba_map, league_babip_map, league_contact_map)

    batter_id = int(p["batter"].iloc[0])
    player_name = str(p["player_name"].iloc[0] or "")
    out_path = heatmap_path(VIZ_DIR, int(args.season), args.metric, player_name, batter_id)

    if out_path.exists() and not args.overwrite:
        print(f"Exists (skip, use --overwrite to replace): {out_path}")
        return

    fixed_limit = None if args.fixed_limit <= 0 else args.fixed_limit
    out = plot_heatmap(
        player_df=p,
        metric=args.metric,
        season=int(args.season),
        annotate_values=bool(args.annotate_values),
        annotate_counts=bool(args.annotate_counts),
        fixed_limit=fixed_limit if args.metric == "zone_weight" else None,
    )
    print("Saved:", out)


if __name__ == "__main__":
    main()