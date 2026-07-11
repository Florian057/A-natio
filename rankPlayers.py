"""The actual "Team zusammenstellen" step -- moneyball itself never built this
(see Readme.md there: roster optimization was a TODO). Ranks every
Germany-eligible player (FBref Nation == 'GER') in the Big-5 leagues by a
position-specific "win contribution" score.

Score = z-scored factor values (z relative to peers at the same position in
the same season, all nations) dotted with the standardized OLS coefficients
learned in factorPerformanceFootball.ols_block() for pts_per_game -- i.e.
"how many standard deviations above a typical starter is this player, on the
factors that actually move a team's points-per-game". This is a ranking, not
a squad optimizer: the user chose position-ranking over an ILP-based squad
selector for this iteration (no formation/quota constraints applied).

Data ceiling is the 2022-23 season (see scrapeFootballData.py) -- treat this
as "who ranked well as of 2022-23", not a live current-form snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorPerformanceFootball import (
    POSITION_BLOCKS,
    WEIGHT_COL,
    build_feature_blocks,
    build_player_features,
    ols_block,
)

MIN_MINUTES = 900          # ~10 full matches -- drop small-sample noise
TARGET_KPI = "pts_per_game"  # differentiates draws from losses, unlike win_pct
TOP_N = 15


def _pick_pool_season(player_features: pd.DataFrame, spike_threshold: float = 0.9,
                      prior_threshold: float = 0.3) -> int:
    """Most recent season that isn't a broken scrape.

    Several factor columns (e.g. keepers_adv-based ones) are structurally
    only available from ~2018 onward -- that's fine, dropna() handles it.
    But the source mirror's last season (2022-23) has two columns
    (Prog_Carries/CPA_Carries, feeding prog_carries_p90/carries_into_box_p90)
    that are 100% empty even though the season right before it is ~0% empty
    -- a broken scrape right before the repo was archived, not a structural
    gap. Detect that spike (season vs. the season right before it) rather
    than a flat threshold, so structurally-late columns don't reject every
    season.
    """
    feature_union = sorted({f for cols in POSITION_BLOCKS.values() for f in cols})
    seasons = sorted(player_features["Season_End_Year"].unique(), reverse=True)
    for i, season in enumerate(seasons):
        sub = player_features[player_features["Season_End_Year"] == season]
        if sub.empty:
            continue
        missing = sub[feature_union].isna().mean()
        if i + 1 < len(seasons):
            prev = player_features[player_features["Season_End_Year"] == seasons[i + 1]]
            prev_missing = prev[feature_union].isna().mean()
            broken = (missing > spike_threshold) & (prev_missing < prior_threshold)
        else:
            broken = missing > spike_threshold
        if not broken.any():
            return int(season)
    return int(seasons[0])


def score_bucket(player_features: pd.DataFrame, blocks: dict[str, pd.DataFrame],
                 kpi: pd.DataFrame, bucket: str, season: int) -> pd.DataFrame | None:
    feature_cols = POSITION_BLOCKS[bucket]
    weights_df = ols_block(blocks[bucket], kpi, feature_cols, TARGET_KPI)
    if weights_df.empty:
        return None
    weights = weights_df.set_index("factor")["coef_std"]

    pool = player_features[(player_features["pos_bucket"] == bucket) &
                            (player_features["Season_End_Year"] == season) &
                            (player_features[WEIGHT_COL] >= MIN_MINUTES)].copy()
    if pool.empty:
        return None

    # standardize each factor against the full peer population at this
    # position (all nations, same season) -- a player's z-score reads as
    # "std devs above/below a typical starter at this position"
    z = pd.DataFrame(index=pool.index)
    for f in feature_cols:
        mu, sd = pool[f].mean(), pool[f].std()
        z[f] = (pool[f] - mu) / sd if sd and not np.isnan(sd) else 0.0

    pool["score"] = (z[feature_cols] * weights[feature_cols]).sum(axis=1)

    ger = pool[pool["Nation"] == "GER"].copy()
    cols = ["Player", "Squad", "Comp", "Pos", WEIGHT_COL, "score"] + feature_cols
    return ger.sort_values("score", ascending=False)[cols]


def main() -> None:
    player_features = build_player_features()
    blocks, kpi = build_feature_blocks()
    season = _pick_pool_season(player_features)
    print(f"[rank] Datenstand: Saison {season - 1}-{str(season)[-2:]} "
          f"(Mindestminuten: {MIN_MINUTES}, Ziel-KPI: {TARGET_KPI})")

    for bucket in POSITION_BLOCKS:
        print(f"\n=== {bucket}: deutsche Spieler-Rangliste ===")
        ranked = score_bucket(player_features, blocks, kpi, bucket, season)
        if ranked is None or ranked.empty:
            print("  keine Daten")
            continue
        print(f"  ({len(ranked)} Kandidaten über {MIN_MINUTES} Minuten)")
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(ranked.head(TOP_N).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
