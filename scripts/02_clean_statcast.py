import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

print("Loading Statcast data...")

file = RAW_DIR / "statcast_2024.csv"
df = pd.read_csv(file)

print("Original rows:", len(df))

# Keep only columns needed for hot-zone analysis + contact%
columns = [
    "batter",
    "pitch_type",
    "plate_x",
    "plate_z",

    # outcome fields
    "description",   # REQUIRED for swing/contact/whiff
    "type",          # optional but useful ("X" in-play, "S" strike, "B" ball)
    "events",

    # batted-ball / value fields
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",

    "game_date",
]

# Keep only columns that actually exist in the downloaded file (defensive)
existing = [c for c in columns if c in df.columns]
missing = [c for c in columns if c not in df.columns]
if missing:
    print("Warning: missing columns in raw Statcast file:", missing)

df = df[existing]

# Remove pitches with missing location
df = df.dropna(subset=["plate_x", "plate_z"])

print("Rows after cleaning:", len(df))

# Save cleaned dataset
out = PROCESSED_DIR / "statcast_cleaned_2024.csv"
df.to_csv(out, index=False)

print("Saved cleaned dataset to:", out)