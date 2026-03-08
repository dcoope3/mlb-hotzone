# app.py — sidebar controls + auto-updating heatmap
#
# Run:
#   streamlit run app.py

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------
# Project paths
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import PROCESSED_DIR, VIZ_DIR  # noqa: E402
from utils_paths import heatmap_path  # noqa: E402

WEIGHTS_PATH = PROCESSED_DIR / "zone_weights_2024.csv"
SCRIPT5_PATH = SCRIPTS_DIR / "05_visualize_hotzones_2024.py"
SEASON = 2024

# -----------------------------------------------------
# Available metrics (internal values must match Script 5)
# -----------------------------------------------------
METRICS = [
    "zone_weight",
    "xwoba_shrunk",
    "xwoba_contact",
    "bip_count",
    "pitches_seen",
    "babip",
    "contact_rate",
]

METRIC_DISPLAY = {
    "zone_weight": "Zone Weight",
    "xwoba_shrunk": "xwOBA Shrunk",
    "xwoba_contact": "xwOBA Contact",
    "bip_count": "BIP Count",
    "pitches_seen": "Pitches Seen",
    "babip": "BABIP",
    "contact_rate": "Contact Rate",
}

# -----------------------------------------------------
# Streamlit settings
# -----------------------------------------------------
st.set_page_config(page_title="MLB Hot Zones", layout="wide")
st.title("MLB Hot Zones (2024)")

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
@st.cache_data(show_spinner=False)
def load_players(weights_path: Path) -> pd.DataFrame:
    usecols = ["player_name", "batter"]
    try:
        df = pd.read_csv(weights_path, usecols=usecols, engine="pyarrow")
    except Exception:
        df = pd.read_csv(weights_path, usecols=usecols)

    df = df.dropna(subset=["batter"]).copy()
    df["batter"] = df["batter"].astype(int)

    if "player_name" not in df.columns:
        df["player_name"] = ""

    df["player_name"] = df["player_name"].fillna("").astype(str)

    return (
        df.drop_duplicates(subset=["player_name", "batter"])
        .sort_values("player_name")
        .reset_index(drop=True)
    )


def run_script5(
    batter_id: int,
    metric: str,
    annotate_values: bool,
    annotate_counts: bool,
) -> tuple[int, str, str]:
    """
    Always overwrites so current visual settings are reflected immediately.
    """
    if metric not in METRICS:
        return 1, "", f"Invalid metric requested by app.py: {metric}"

    cmd = [
        sys.executable,
        str(SCRIPT5_PATH),
        "--batter",
        str(int(batter_id)),
        "--metric",
        metric,
        "--season",
        str(SEASON),
        "--overwrite",
    ]

    if annotate_values:
        cmd.append("--annotate-values")
    if annotate_counts:
        cmd.append("--annotate-counts")

    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return p.returncode, p.stdout, p.stderr


def format_player_name(name: str, batter_id: int) -> str:
    clean = str(name or "").strip()
    if not clean:
        return f"Batter {batter_id}"
    return clean.title()


# -----------------------------------------------------
# Guardrails
# -----------------------------------------------------
if not WEIGHTS_PATH.exists():
    st.error(f"Missing weights file: {WEIGHTS_PATH}")
    st.info("Run Script 4 first.")
    st.stop()

if not SCRIPT5_PATH.exists():
    st.error(f"Missing Script 5: {SCRIPT5_PATH}")
    st.stop()

VIZ_DIR.mkdir(parents=True, exist_ok=True)
players = load_players(WEIGHTS_PATH)

if players.empty:
    st.error("No players found in zone_weights_2024.csv")
    st.stop()

# -----------------------------------------------------
# Sidebar UI
# -----------------------------------------------------
with st.sidebar:
    st.header("Controls")

    name_query = st.text_input("Search Player", "")
    q = name_query.strip().lower()

    if q:
        matches = players[players["player_name"].str.lower().str.contains(q, na=False)].head(100)
    else:
        matches = players.head(100)

    if matches.empty:
        st.warning("No players found.")
        st.stop()

    options = list(matches.itertuples(index=False, name=None))

    def format_player(opt) -> str:
        return f"{format_player_name(opt[0], int(opt[1]))} ({int(opt[1])})"

    choice = st.selectbox("Player", options=options, format_func=format_player)

    metric_display_options = [METRIC_DISPLAY[m] for m in METRICS]
    selected_metric_display = st.selectbox("Metric", metric_display_options, index=0)
    metric = next(k for k, v in METRIC_DISPLAY.items() if v == selected_metric_display)

    annotate_values = st.checkbox("Annotate Values", value=True)
    annotate_counts = st.checkbox("Annotate Sample Size (n)", value=True)

player_name = str(choice[0] or "")
batter_id = int(choice[1])

# -----------------------------------------------------
# Image path
# -----------------------------------------------------
image_path = heatmap_path(
    viz_dir=VIZ_DIR,
    season=SEASON,
    metric=metric,
    player_name=player_name,
    batter_id=batter_id,
)

# -----------------------------------------------------
# Automatic regeneration
# -----------------------------------------------------
request_key = (batter_id, metric, annotate_values, annotate_counts)

if "last_request_key" not in st.session_state:
    st.session_state.last_request_key = None

if "last_gen_result" not in st.session_state:
    st.session_state.last_gen_result = None

if request_key != st.session_state.last_request_key:
    st.session_state.last_request_key = request_key
    with st.spinner("Generating heatmap..."):
        code, out, err = run_script5(
            batter_id=batter_id,
            metric=metric,
            annotate_values=annotate_values,
            annotate_counts=annotate_counts,
        )
    st.session_state.last_gen_result = (code, out, err)

# -----------------------------------------------------
# Display
# -----------------------------------------------------
if st.session_state.last_gen_result is not None:
    code, out, err = st.session_state.last_gen_result
    if code != 0:
        st.error("Heatmap generation failed.")
        st.code(err if err else out)

display_player = format_player_name(player_name, batter_id)
display_metric = METRIC_DISPLAY.get(metric, metric)

if image_path.exists():
    st.image(str(image_path), caption=f"{display_player} — {display_metric}", width="stretch")
else:
    st.warning("Heatmap file not found. Generation may have failed.")