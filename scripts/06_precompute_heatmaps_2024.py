from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import PROCESSED_DIR, VIZ_DIR  # noqa: E402
from utils_paths import heatmap_path  # noqa: E402

SCRIPT5_PATH = SCRIPTS_DIR / "05_visualize_hotzones_2024.py"
WEIGHTS_PATH = PROCESSED_DIR / "zone_weights_2024.csv"

DEFAULT_METRICS = [
    "zone_weight",
    "xwoba_shrunk",
    "xwoba_contact",
    "bip_count",
    "pitches_seen",
    "babip",
    "contact_rate",
]


def run_script5(
    batter_id: int,
    metric: str,
    season: int,
    annotate_values: bool,
    annotate_counts: bool,
    overwrite: bool,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT5_PATH),
        "--batter",
        str(int(batter_id)),
        "--metric",
        metric,
        "--season",
        str(int(season)),
    ]
    if annotate_values:
        cmd.append("--annotate-values")
    if annotate_counts:
        cmd.append("--annotate-counts")
    if overwrite:
        cmd.append("--overwrite")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"Script 5 failed for batter={batter_id}, metric={metric}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument("--annotate-values", action="store_true")
    parser.add_argument("--annotate-counts", action="store_true")
    args = parser.parse_args()

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing {WEIGHTS_PATH}. Run Script 4 first.")

    if args.metric is None:
        metrics = DEFAULT_METRICS
    else:
        if args.metric not in DEFAULT_METRICS:
            raise ValueError(f"--metric must be one of: {DEFAULT_METRICS}")
        metrics = [args.metric]

    df = pd.read_csv(WEIGHTS_PATH)
    if "batter" not in df.columns:
        raise ValueError("zone_weights_2024.csv missing 'batter'")
    if "player_name" not in df.columns:
        df["player_name"] = ""

    players = df[["batter", "player_name"]].dropna(subset=["batter"]).drop_duplicates().copy()
    players["batter"] = players["batter"].astype(int)
    players = players.sort_values("batter").reset_index(drop=True)

    if args.max_players is not None:
        players = players.head(int(args.max_players))

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    total = len(players) * len(metrics)
    done = 0
    skipped = 0

    print(f"Players: {len(players)} | Metrics: {metrics} | only_missing={args.only_missing} overwrite={args.overwrite}")

    for row in players.itertuples(index=False):
        batter_id = int(row.batter)
        player_name = str(row.player_name or "")

        for metric in metrics:
            out_path = heatmap_path(VIZ_DIR, int(args.season), metric, player_name, batter_id)

            if args.only_missing and out_path.exists() and not args.overwrite:
                skipped += 1
                done += 1
                continue

            run_script5(
                batter_id=batter_id,
                metric=metric,
                season=int(args.season),
                annotate_values=args.annotate_values,
                annotate_counts=args.annotate_counts,
                overwrite=args.overwrite,
            )

            done += 1
            if done % 100 == 0 or done == total:
                print(f"Progress: {done}/{total} (skipped {skipped})")

    print("✅ Precompute complete.")
    print(f"Attempted: {done} | Skipped: {skipped} | Output: {VIZ_DIR.resolve()}")


if __name__ == "__main__":
    main()