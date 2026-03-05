import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------
# Allow importing config.py from scripts folder
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import PROCESSED_DIR, VIZ_DIR


# -----------------------------------------------------
# File locations
# -----------------------------------------------------
WEIGHTS_PATH = PROCESSED_DIR / "zone_weights_2024.csv"
SCRIPT5_PATH = SCRIPTS_DIR / "05_visualize_hotzones_2024.py"


# -----------------------------------------------------
# Streamlit page settings
# -----------------------------------------------------
st.set_page_config(
    page_title="MLB Hot Zones",
    layout="centered"
)

st.title("MLB Hot Zones (2024)")


# -----------------------------------------------------
# Load data
# -----------------------------------------------------
@st.cache_data
def load_weights():
    df = pd.read_csv(WEIGHTS_PATH)

    if "player_name" not in df.columns:
        df["player_name"] = ""

    return df


df = load_weights()


# -----------------------------------------------------
# Player list
# -----------------------------------------------------
players = (
    df[["player_name", "batter"]]
    .dropna()
    .drop_duplicates()
    .sort_values("player_name")
)


# -----------------------------------------------------
# Search box
# -----------------------------------------------------
name_query = st.text_input(
    "Type a player name (example: Aaron Judge):",
    ""
)

if name_query.strip():
    matches = players[
        players["player_name"].str.lower().str.contains(
            name_query.lower(),
            na=False
        )
    ]
else:
    matches = players


if len(matches) == 0:
    st.warning("No players found.")
    st.stop()


# -----------------------------------------------------
# Player selector
# -----------------------------------------------------
options = list(matches.itertuples(index=False, name=None))


def format_player(opt):
    return f"{opt[0]} ({int(opt[1])})"


choice = st.selectbox(
    "Select player:",
    options=options,
    format_func=format_player
)


player_name = choice[0]
batter_id = int(choice[1])


# -----------------------------------------------------
# Metric selector
# -----------------------------------------------------
metric = st.selectbox(
    "Metric:",
    [
        "zone_weight",
        "xwoba_shrunk",
        "xwoba_contact",
        "bip_count"
    ]
)


# -----------------------------------------------------
# Annotation options
# -----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    annotate_values = st.checkbox(
        "Annotate values",
        value=True
    )

with col2:
    annotate_counts = st.checkbox(
        "Annotate BIP counts",
        value=True
    )


# -----------------------------------------------------
# Expected image path
# -----------------------------------------------------
def safe_name(s):
    return "".join(
        ch if ch.isalnum() or ch in (" ", "-", "_")
        else ""
        for ch in s
    ).strip().replace(" ", "_")


image_path = (
    VIZ_DIR /
    f"hotzone_2024_{metric}_{safe_name(player_name)}_{batter_id}.png"
)


# -----------------------------------------------------
# Run Script 5
# -----------------------------------------------------
def generate_heatmap():
    cmd = [
        sys.executable,
        str(SCRIPT5_PATH),
        "--batter",
        str(batter_id),
        "--metric",
        metric
    ]

    if annotate_values:
        cmd.append("--annotate-values")

    if annotate_counts:
        cmd.append("--annotate-counts")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )

    return result.returncode, result.stdout, result.stderr


# -----------------------------------------------------
# Generate button
# -----------------------------------------------------
if st.button("Generate / Update Heatmap"):

    with st.spinner("Generating heatmap..."):

        code, out, err = generate_heatmap()

    if code == 0:
        st.success("Heatmap generated.")
    else:
        st.error("Script 5 failed.")
        st.code(err if err else out)


# -----------------------------------------------------
# Display image
# -----------------------------------------------------
if image_path.exists():

    st.image(
        str(image_path),
        caption=f"{player_name} — {metric}",
        use_container_width=True
    )

else:

    st.info(
        "No heatmap yet for this player. Click 'Generate / Update Heatmap'."
    )