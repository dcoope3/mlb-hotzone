from pybaseball import statcast
from pybaseball import cache
from config import RAW_DIR

# enable caching
cache.enable()

print("Downloading Statcast data...")

data = statcast(start_dt="2024-03-28", end_dt="2024-10-01")

file = RAW_DIR / "statcast_2024.csv"
data.to_csv(file, index=False)

print("Saved to:", file)