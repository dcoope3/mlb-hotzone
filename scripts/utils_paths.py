from pathlib import Path


def sanitize_name(name: str) -> str:
    """
    Make a safe filename version of a player name.
    """
    if not name:
        return "unknown"

    name = name.strip().replace(" ", "_")
    name = name.replace(".", "")
    name = name.replace("'", "")
    return name


def heatmap_path(viz_dir: Path, season: int, metric: str, player_name: str, batter_id: int) -> Path:
    """
    Build the output PNG path for a player's heatmap.
    """
    safe_name = sanitize_name(player_name)

    filename = f"{season}_{metric}_{safe_name}_{batter_id}.png"

    return viz_dir / filename