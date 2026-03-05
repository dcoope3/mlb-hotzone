"""
Build 2024 player-by-zone value surfaces (xwOBA on contact) with shrinkage,
AND attach player names (MLBAM id -> name) to the output.

INPUT  : data/processed/statcast_zones_2024.csv
OUTPUT : data/processed/zone_weights_2024.csv

Key choices:
- Eligibility filter: keep only hitters with >= 500 pitches seen in 2024
  (pitches seen = number of rows in statcast_zones_2024.csv for that batter)
- Contact/BIP proxy: rows where `estimated_woba_using_speedangle` is NOT null
- Shrinkage: toward league zone baseline with stabilization constant K (default 50)
- zone_weight = xwoba_shrunk - league_xwoba_contact

Notes:
- Output includes one row per (batter, zone_id) that had at least 1 BIP/contact in that zone.
  This means some hitters will have < 25 rows (missing zones with 0 BIP).
  If you later want a full 25-zone grid per hitter for cleaner heatmaps, tell me and I’ll
  modify this to “complete” zones 0..24 per hitter and fill missing with league baseline.
"""

from __future__ import annotations

import pandas as pd
from config import PROCESSED_DIR

# pybaseball is in your requirements.txt, so this should be available
from pybaseball import cache
from pybaseball import playerid_reverse_lookup

# -----------------------------
# Files
# -----------------------------
INPUT_ZONED_2024 = "statcast_zones_2024.csv"
OUTPUT_WEIGHTS_2024 = "zone_weights_2024.csv"

# -----------------------------
# Parameters
# -----------------------------
MIN_PITCHES_SEEN = 500

# Stabilization constant for shrinkage (tune later; 30–60 is common for contact metrics)
K = 50


def main() -> None:
    # Enable caching so name lookups don’t keep re-downloading in future runs
    cache.enable()

    in_path = PROCESSED_DIR / INPUT_ZONED_2024
    out_path = PROCESSED_DIR / OUTPUT_WEIGHTS_2024

    print("Loading:", in_path)
    df = pd.read_csv(in_path)
    print("Rows loaded:", len(df))

    # Required columns
    needed = ["batter", "zone_id", "estimated_woba_using_speedangle"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found columns: {list(df.columns)}")

    # -----------------------------
    # 0) Eligibility filter: >= 500 pitches seen in 2024
    # -----------------------------
    pitches_seen = (
        df.groupby("batter", as_index=False)
        .size()
        .rename(columns={"size": "pitches_seen"})
    )

    eligible_hitters = pitches_seen.loc[
        pitches_seen["pitches_seen"] >= MIN_PITCHES_SEEN, "batter"
    ]

    print("Hitters total in zoned file:", df["batter"].nunique())
    print(f"Eligible hitters (>= {MIN_PITCHES_SEEN} pitches):", eligible_hitters.nunique())

    df = df[df["batter"].isin(eligible_hitters)].copy()
    print("Rows after eligibility filter:", len(df))

    # -----------------------------
    # 1) Contact/BIP proxy
    # -----------------------------
    bip = df.dropna(subset=["estimated_woba_using_speedangle"]).copy()
    print("Contact/BIP rows (eligible hitters):", len(bip))

    if len(bip) == 0:
        raise ValueError("No contact/BIP rows found after filtering. Check your input data.")

    # -----------------------------
    # 2) Player-zone raw (2024)
    # -----------------------------
    player_zone = (
        bip.groupby(["batter", "zone_id"], as_index=False)
        .agg(
            bip_count=("estimated_woba_using_speedangle", "size"),
            xwoba_contact=("estimated_woba_using_speedangle", "mean"),
        )
    )

    # -----------------------------
    # 3) League-zone baseline (2024)
    # -----------------------------
    league_zone = (
        bip.groupby("zone_id", as_index=False)
        .agg(
            league_bip=("estimated_woba_using_speedangle", "size"),
            league_xwoba_contact=("estimated_woba_using_speedangle", "mean"),
        )
    )

    m = player_zone.merge(league_zone, on="zone_id", how="left")

    # -----------------------------
    # 4) Shrinkage toward league zone baseline
    # -----------------------------
    m["xwoba_shrunk"] = (
        (m["bip_count"] * m["xwoba_contact"] + K * m["league_xwoba_contact"])
        / (m["bip_count"] + K)
    )

    # -----------------------------
    # 5) Continuous zone weight (relative to league at that zone)
    # -----------------------------
    m["zone_weight"] = m["xwoba_shrunk"] - m["league_xwoba_contact"]

    out = m[
        [
            "batter",
            "zone_id",
            "bip_count",
            "xwoba_contact",
            "league_xwoba_contact",
            "xwoba_shrunk",
            "zone_weight",
        ]
    ].sort_values(["batter", "zone_id"])

    # -----------------------------
    # 6) Add player names (MLBAM -> first/last)
    # -----------------------------
    print("Adding player names via pybaseball lookup...")
    ids = out["batter"].dropna().unique()

    # Reverse lookup expects a list/array of MLBAM ids
    players = playerid_reverse_lookup(ids, key_type="mlbam")

    # Build a single display name
    players["player_name"] = players["name_first"].fillna("") + " " + players["name_last"].fillna("")
    players["player_name"] = players["player_name"].str.strip()

    # Merge into output
    out = out.merge(
        players[["key_mlbam", "player_name"]],
        left_on="batter",
        right_on="key_mlbam",
        how="left",
    ).drop(columns=["key_mlbam"])

    # Put player_name first for readability
    cols = ["player_name"] + [c for c in out.columns if c != "player_name"]
    out = out[cols]

    # -----------------------------
    # 7) Sanity checks + summary
    # -----------------------------
    if not out["zone_id"].between(0, 24).all():
        bad = out.loc[~out["zone_id"].between(0, 24), "zone_id"].unique()
        raise ValueError(f"zone_id out of 0..24 found: {bad}")

    zones_per_hitter = out.groupby("batter")["zone_id"].nunique()

    print("---- OUTPUT SUMMARY ----")
    print("Eligible hitters (>= pitches seen):", eligible_hitters.nunique())
    print("Hitters in weight table:", out["batter"].nunique())
    print("Rows in weight table:", len(out))
    print("Zones per hitter (min/median/max):",
          int(zones_per_hitter.min()),
          float(zones_per_hitter.median()),
          int(zones_per_hitter.max()))
    print("Missing names (should be small):", int(out["player_name"].isna().sum()))
    print(out.head(10))

    out.to_csv(out_path, index=False)
    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()