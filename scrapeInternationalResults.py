"""National-team side of the data store — validation layer, not the primary
training signal (see factorPerformanceFootball.py for why).

Source: martj42/international_results (GitHub, public domain CSV) — every
men's full international since 1872. A straight download, no scraping.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

RESULTS_CSV = ("https://raw.githubusercontent.com/martj42/international_results/"
              "master/results.csv")


def _pq(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def fetch_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df


def build(force: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if force or not _pq("international_results").exists():
        fetch_results().to_parquet(_pq("international_results"), index=False)
    print(f"[intl]   international_results: "
          f"{len(pd.read_parquet(_pq('international_results')))} rows")


if __name__ == "__main__":
    build()
