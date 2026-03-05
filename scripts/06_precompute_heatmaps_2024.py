import sys
import subprocess
from pathlib import Path

import pandas as pd

# -----------------------------------------------------
# Paths (match your app.py structure)
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# script 5 path
SCRIPT5_PATH = SCRIPTS_DIR / "05_visualize_hotzones_2024.py"

# your processed weights file (made by script 4)
WEIGHTS_PATH = PROJECT_ROOT / "data" / "processed" / "zone_weights_2024.csv"


def run_script5(batter_id: int, metric: str, annotate_values: bool, annotate_counts: bool) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT5_PATH),
        "--batter",
        str(batter_id),
        "--metric",
        metric,
    ]

    if annotate_values:
        cmd.append("--annotate-values")
    if annotate_counts:
        cmd.append("--annotate-counts")

    # Run from project root so config paths match
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Script 5 failed for batter={batter_id}, metric={metric}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def main():
    df = pd.read_csv(WEIGHTS_PATH)

    # Get all batters that exist in zone_weights_2024.csv
    batters = sorted(df["batter"].dropna().unique().astype(int).tolist())
    print(f"Found {len(batters)} batters in zone_weights_2024.csv")

    metrics = ["zone_weight", "xwoba_shrunk", "xwoba_contact", "bip_count"]

    # Choose your default annotation style for all images
    annotate_values = True
    annotate_counts = True

    n_done = 0
    for batter_id in batters:
        for metric in metrics:
            run_script5(
                batter_id=batter_id,
                metric=metric,
                annotate_values=annotate_values,
                annotate_counts=annotate_counts,
            )
        n_done += 1
        if n_done % 25 == 0:
            print(f"Done {n_done}/{len(batters)} players...")

    print("✅ Precompute complete. All images should be in /visualizations")


if __name__ == "__main__":
    main()