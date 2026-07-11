"""Club-performance side of the data store (this project's analog of
moneyball/getPlayerStats.py).

Data source: JaseZiv/worldfootballR_data (GitHub, archived Sep 2025) — a
pre-scraped mirror of FBref's Big-5-leagues advanced stats, shipped as .rds
files. FBref itself now sits behind a Cloudflare bot-challenge that blocks
even a real headless browser from this sandbox (verified: soccerdata's
Selenium-based reader times out against fbref.com's "Just a moment..." wall,
and a plain curl gets a 403 challenge page), so this static mirror is what
actually works without needing a residential IP or paid scraping service.

Ceiling of the mirror is the 2022-23 season (Season_End_Year 2023) — the repo
stopped updating player-level tables well before archival. So "current" here
means the 2022-23 season, not real-time squads. A fresher source can be
swapped in later by pointing WFR_BASE elsewhere; nothing downstream depends on
this being live data.

Two kinds of tables:
  - player_<stat>.parquet / team_<stat>.parquet — one FBref stat page each
    (standard, shooting, passing, defense, possession, misc, keepers,
    keepers_adv), Big-5 leagues, all seasons in the mirror.
  - match_results.parquet — actual W/D/L/GF/GA per match, Big-5 top flights
    only (Tier=='1st', Gender=='M') — this is the ground truth the KPI panel
    is built from; the stat tables above are features, not KPIs.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pandas as pd
import pyreadr
import requests

DATA_DIR = Path(__file__).parent / "data"

WFR_BASE = "https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master"
STAT_TYPES = ["standard", "shooting", "passing", "defense", "possession", "misc",
              "keepers", "keepers_adv"]

# match_results/<COUNTRY>_match_results.rds -> our Comp name (must match the
# `Comp` column used in the big5 stat tables) so KPI and factor tables can be
# joined on Squad + Season_End_Year + Comp.
LEAGUE_COUNTRY = {
    "ENG": "Premier League",
    "ESP": "La Liga",
    "FRA": "Ligue 1",
    "GER": "Bundesliga",
    "ITA": "Serie A",
}


def _pq(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def _read_rds(url: str, retries: int = 4) -> pd.DataFrame:
    """Download an .rds file and load it as a DataFrame (pyreadr needs a real
    file on disk, no stream support). raw.githubusercontent.com throws
    occasional transient 429s -- retry with backoff rather than fail the
    whole build over one flaky request."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            break
        except requests.exceptions.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    with tempfile.NamedTemporaryFile(suffix=".rds") as tmp:
        tmp.write(r.content)
        tmp.flush()
        result = pyreadr.read_r(tmp.name)
    return result[None]


# --------------------------------------------------------------------------- #
# stat tables (player + team, big-5 leagues, all seasons in the mirror)
# --------------------------------------------------------------------------- #
def fetch_player_stats(stat_types: list[str] = STAT_TYPES) -> dict[str, pd.DataFrame]:
    out = {}
    for stat in stat_types:
        print(f"[football]   player_{stat}")
        out[stat] = _read_rds(f"{WFR_BASE}/data/fb_big5_advanced_season_stats/"
                              f"big5_player_{stat}.rds")
    return out


def fetch_team_stats(stat_types: list[str] = STAT_TYPES) -> dict[str, pd.DataFrame]:
    out = {}
    for stat in stat_types:
        print(f"[football]   team_{stat}")
        out[stat] = _read_rds(f"{WFR_BASE}/data/fb_big5_advanced_season_stats/"
                              f"big5_team_{stat}.rds")
    return out


# --------------------------------------------------------------------------- #
# match results -> KPI ground truth
# --------------------------------------------------------------------------- #
def fetch_match_results(countries: dict[str, str] = LEAGUE_COUNTRY) -> pd.DataFrame:
    """Top-flight men's matches only, one row per match, tagged with the Comp
    name used in the stat tables so the two sides join cleanly."""
    frames = []
    for country, comp in countries.items():
        print(f"[football]   match_results {country}")
        df = _read_rds(f"{WFR_BASE}/data/match_results/{country}_match_results.rds")
        df = df[(df["Tier"] == "1st") & (df["Gender"] == "M")].copy()
        df["Comp"] = comp
        frames.append(df[["Comp", "Season_End_Year", "Home", "HomeGoals",
                          "Away", "AwayGoals"]])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# build (no incremental logic -- the source is a frozen, archived mirror, so
# there is nothing new to fetch on a re-run; skip work that's already on disk)
# --------------------------------------------------------------------------- #
def build(force: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    player_missing = [s for s in STAT_TYPES if force or not _pq(f"player_{s}").exists()]
    if player_missing:
        for stat, df in fetch_player_stats(player_missing).items():
            df.to_parquet(_pq(f"player_{stat}"), index=False)

    team_missing = [s for s in STAT_TYPES if force or not _pq(f"team_{s}").exists()]
    if team_missing:
        for stat, df in fetch_team_stats(team_missing).items():
            df.to_parquet(_pq(f"team_{stat}"), index=False)

    if force or not _pq("match_results").exists():
        fetch_match_results().to_parquet(_pq("match_results"), index=False)

    for stat in STAT_TYPES:
        print(f"[football]   player_{stat}: {len(pd.read_parquet(_pq(f'player_{stat}')))} rows")
        print(f"[football]   team_{stat}: {len(pd.read_parquet(_pq(f'team_{stat}')))} rows")
    print(f"[football]   match_results: {len(pd.read_parquet(_pq('match_results')))} rows")


if __name__ == "__main__":
    build()
