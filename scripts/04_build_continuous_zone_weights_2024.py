"""
Script 4 — Build 2024 player-by-zone metrics (value + volume + BABIP + Contact%) with shrinkage.

INPUT  : data/processed/statcast_zones_2024.csv
OUTPUT : data/processed/zone_weights_2024.csv

Adds:
- pitches_seen (all pitches in zone)

CONTACT%:
- swings   (# swings in zone)
- contacts (# contact outcomes on swings: foul or ball in play)
- whiffs   (# whiffs on swings)
- contact_rate = contacts / swings  (NaN if swings == 0)
- league baselines by zone: league_swings, league_contacts, league_whiffs, league_contact_rate

BABIP:
- bip (balls in play count, including hits-in-play + outs-in-play + ROE + SF)
- bip_hits (hits on balls in play: 1B/2B/3B; HR excluded)
- babip = bip_hits / bip  (NaN if bip == 0)
- league baselines by zone: league_bip, league_bip_hits, league_babip

xwOBA contact surface:
- bip_count (contact proxy via estimated_woba_using_speedangle not null)
- xwoba_contact
- league_xwoba_contact
- xwoba_shrunk
- zone_weight = xwoba_shrunk - league_xwoba_contact
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from config import PROCESSED_DIR

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
K = 50  # shrinkage constant for xwOBA contact

# -----------------------------
# BABIP definitions
# -----------------------------
BIP_HIT_EVENTS = {"single", "double", "triple"}

BIP_EVENTS = {
    "single", "double", "triple",
    "field_out", "force_out",
    "grounded_into_double_play", "double_play", "triple_play",
    "fielders_choice", "fielders_choice_out",
    "reached_on_error",
    "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play",
}

# -----------------------------
# Contact% definitions (Statcast "description")
# -----------------------------
# We interpret "contact" as: foul OR ball put in play (regardless of result).
CONTACT_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",              # optional; keep or remove if you want "swings" to exclude bunts
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",            # optional; keep or remove depending on how you treat bunts
}

SWING_DESCRIPTIONS = CONTACT_DESCRIPTIONS | WHIFF_DESCRIPTIONS


def main() -> None:
    cache.enable()

    in_path = PROCESSED_DIR / INPUT_ZONED_2024
    out_path = PROCESSED_DIR / OUTPUT_WEIGHTS_2024

    print("Loading:", in_path)
    df = pd.read_csv(in_path)
    print("Rows loaded:", len(df))

    needed = ["batter", "zone_id", "estimated_woba_using_speedangle", "events", "description"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            "Did you rerun Script 02 and Script 03 after adding 'description'?"
        )

    # -----------------------------
    # 0) Eligibility: >= MIN_PITCHES_SEEN pitches (rows) in zoned file
    # -----------------------------
    pitches_seen_total = (
        df.groupby("batter", as_index=False)
        .size()
        .rename(columns={"size": "pitches_seen_total"})
    )
    eligible = pitches_seen_total.loc[pitches_seen_total["pitches_seen_total"] >= MIN_PITCHES_SEEN, "batter"]

    print("Hitters total in zoned file:", df["batter"].nunique())
    print(f"Eligible hitters (>= {MIN_PITCHES_SEEN} pitches):", eligible.nunique())

    df = df[df["batter"].isin(eligible)].copy()
    print("Rows after eligibility filter:", len(df))

    if not df["zone_id"].between(0, 24).all():
        bad = df.loc[~df["zone_id"].between(0, 24), "zone_id"].unique()
        raise ValueError(f"zone_id out of 0..24 found: {bad}")

    # -----------------------------
    # 1) Volume metrics on ALL pitches
    # -----------------------------
    player_zone_pitches = (
        df.groupby(["batter", "zone_id"], as_index=False)
        .agg(pitches_seen=("zone_id", "size"))
    )

    league_zone_pitches = (
        df.groupby("zone_id", as_index=False)
        .agg(league_pitches_seen=("zone_id", "size"))
    )

    # -----------------------------
    # 2) Contact% on swings
    # -----------------------------
    desc = df["description"].fillna("").astype(str)

    df["is_swing"] = desc.isin(SWING_DESCRIPTIONS)
    df["is_contact"] = desc.isin(CONTACT_DESCRIPTIONS)
    df["is_whiff"] = desc.isin(WHIFF_DESCRIPTIONS)

    player_zone_contact_pct = (
        df.groupby(["batter", "zone_id"], as_index=False)
        .agg(
            swings=("is_swing", "sum"),
            contacts=("is_contact", "sum"),
            whiffs=("is_whiff", "sum"),
        )
    )
    player_zone_contact_pct["contact_rate"] = np.where(
        player_zone_contact_pct["swings"] > 0,
        player_zone_contact_pct["contacts"] / player_zone_contact_pct["swings"],
        np.nan,
    )

    league_zone_contact_pct = (
        df.groupby("zone_id", as_index=False)
        .agg(
            league_swings=("is_swing", "sum"),
            league_contacts=("is_contact", "sum"),
            league_whiffs=("is_whiff", "sum"),
        )
    )
    league_zone_contact_pct["league_contact_rate"] = np.where(
        league_zone_contact_pct["league_swings"] > 0,
        league_zone_contact_pct["league_contacts"] / league_zone_contact_pct["league_swings"],
        np.nan,
    )

    # -----------------------------
    # 3) BABIP by zone
    # -----------------------------
    df["is_bip"] = df["events"].isin(BIP_EVENTS)
    df["is_bip_hit"] = df["events"].isin(BIP_HIT_EVENTS)

    player_zone_babip = (
        df.groupby(["batter", "zone_id"], as_index=False)
        .agg(
            bip=("is_bip", "sum"),
            bip_hits=("is_bip_hit", "sum"),
        )
    )
    player_zone_babip["babip"] = np.where(
        player_zone_babip["bip"] > 0,
        player_zone_babip["bip_hits"] / player_zone_babip["bip"],
        np.nan,
    )

    league_zone_babip = (
        df.groupby("zone_id", as_index=False)
        .agg(
            league_bip=("is_bip", "sum"),
            league_bip_hits=("is_bip_hit", "sum"),
        )
    )
    league_zone_babip["league_babip"] = np.where(
        league_zone_babip["league_bip"] > 0,
        league_zone_babip["league_bip_hits"] / league_zone_babip["league_bip"],
        np.nan,
    )

    # -----------------------------
    # 4) xwOBA on contact (your existing approach)
    # -----------------------------
    bip_contact = df.dropna(subset=["estimated_woba_using_speedangle"]).copy()
    print("Contact/BIP rows (eligible hitters):", len(bip_contact))
    if len(bip_contact) == 0:
        raise ValueError("No contact/BIP rows found after filtering. Check your input data.")

    player_zone_xwoba = (
        bip_contact.groupby(["batter", "zone_id"], as_index=False)
        .agg(
            bip_count=("estimated_woba_using_speedangle", "size"),
            xwoba_contact=("estimated_woba_using_speedangle", "mean"),
        )
    )

    league_zone_xwoba = (
        bip_contact.groupby("zone_id", as_index=False)
        .agg(
            league_bip_contact=("estimated_woba_using_speedangle", "size"),
            league_xwoba_contact=("estimated_woba_using_speedangle", "mean"),
        )
    )

    # -----------------------------
    # 5) Merge into one player-zone table
    # -----------------------------
    m = player_zone_pitches.merge(player_zone_contact_pct, on=["batter", "zone_id"], how="left")
    m = m.merge(player_zone_babip, on=["batter", "zone_id"], how="left")
    m = m.merge(player_zone_xwoba, on=["batter", "zone_id"], how="left")

    m = m.merge(league_zone_xwoba, on="zone_id", how="left")
    m = m.merge(league_zone_pitches, on="zone_id", how="left")
    m = m.merge(league_zone_babip, on="zone_id", how="left")
    m = m.merge(league_zone_contact_pct, on="zone_id", how="left")

    # -----------------------------
    # 6) Fill missing / neutral defaults
    # -----------------------------
    # Swings/contact/whiff
    m["swings"] = m["swings"].fillna(0).astype(int)
    m["contacts"] = m["contacts"].fillna(0).astype(int)
    m["whiffs"] = m["whiffs"].fillna(0).astype(int)

    # For zones with 0 swings, set contact_rate to league_contact_rate (neutral)
    m["contact_rate"] = m["contact_rate"].fillna(m["league_contact_rate"])

    # xwOBA contact fields for zones with 0 contact
    m["bip_count"] = m["bip_count"].fillna(0).astype(int)
    m["xwoba_contact"] = m["xwoba_contact"].fillna(m["league_xwoba_contact"])

    # BABIP components for zones with 0 BIP
    m["bip"] = m["bip"].fillna(0).astype(int)
    m["bip_hits"] = m["bip_hits"].fillna(0).astype(int)
    m["babip"] = m["babip"].fillna(m["league_babip"])

    # -----------------------------
    # 7) Shrinkage toward league zone baseline (xwOBA contact)
    # -----------------------------
    m["xwoba_shrunk"] = (
        (m["bip_count"] * m["xwoba_contact"] + K * m["league_xwoba_contact"])
        / (m["bip_count"] + K)
    )
    m["zone_weight"] = m["xwoba_shrunk"] - m["league_xwoba_contact"]

    # -----------------------------
    # 8) Add player names
    # -----------------------------
    print("Adding player names via pybaseball lookup...")
    ids = m["batter"].dropna().unique()
    players = playerid_reverse_lookup(ids, key_type="mlbam")
    players["player_name"] = (players["name_first"].fillna("") + " " + players["name_last"].fillna("")).str.strip()

    m = m.merge(
        players[["key_mlbam", "player_name"]],
        left_on="batter",
        right_on="key_mlbam",
        how="left",
    ).drop(columns=["key_mlbam"])

    # -----------------------------
    # 9) Output
    # -----------------------------
    cols = [
        "player_name",
        "batter",
        "zone_id",

        # Volume
        "pitches_seen",
        "league_pitches_seen",

        # Contact%
        "swings",
        "contacts",
        "whiffs",
        "contact_rate",
        "league_swings",
        "league_contacts",
        "league_whiffs",
        "league_contact_rate",

        # BABIP
        "bip",
        "bip_hits",
        "babip",
        "league_bip",
        "league_bip_hits",
        "league_babip",

        # xwOBA contact surface
        "bip_count",
        "xwoba_contact",
        "league_xwoba_contact",
        "xwoba_shrunk",
        "zone_weight",
    ]

    out = m[cols].sort_values(["batter", "zone_id"]).reset_index(drop=True)

    zones_per_hitter = out.groupby("batter")["zone_id"].nunique()

    print("---- OUTPUT SUMMARY ----")
    print("Eligible hitters (>= pitches seen):", eligible.nunique())
    print("Hitters in table:", out["batter"].nunique())
    print("Rows in table:", len(out))
    print("Zones per hitter (min/median/max):",
          int(zones_per_hitter.min()),
          float(zones_per_hitter.median()),
          int(zones_per_hitter.max()))
    print("Missing names:", int(out["player_name"].isna().sum()))
    print(out.head(10))

    out.to_csv(out_path, index=False)
    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()