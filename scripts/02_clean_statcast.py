import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

print("Loading Statcast data...")

file = RAW_DIR / "statcast_2024.csv"
df = pd.read_csv(file)

print("Original rows:", len(df))

# Keep only columns needed for hot-zone analysis
columns = [
    "batter",
    "pitch_type",
    "plate_x",
    "plate_z",
    "events",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "game_date"
]

df = df[columns]

# Remove pitches with missing location
df = df.dropna(subset=["plate_x", "plate_z"])

print("Rows after cleaning:", len(df))

# Save cleaned dataset
out = PROCESSED_DIR / "statcast_cleaned_2024.csv"
df.to_csv(out, index=False)

print("Saved cleaned dataset to:", out)