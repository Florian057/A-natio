"""Factor investing applied to football (this project's analog of
moneyball/factorPerformance.py).

Idea (same as moneyball): from a "factor zoo" of player stats, find the ones
that best explain team-level winning. Football's factor zoo is split into
four *position blocks* (GK/DF/MF/FW) instead of moneyball's two (batting/
pitching) — a center-back's tackle rate and a striker's xG live in different
spaces and we want them ranked within their own position, not against each
other.

Analysis lives at the club-team-season level (Big-5 European leagues,
2010-11 through 2022-23 — see scrapeFootballData.py for why the ceiling is
2022-23): every top-flight match played by a full squad, not just Germany's
national team, which only plays a handful of games a year and can't support
a regression on its own. rankPlayers.py is what turns the factor weights
learned here into a ranking of individual German-eligible players.

Output:
  - Pearson correlations of each factor vs each KPI (ranked), per position block.
  - OLS regressions per block per KPI with standardized coefficients.
  - PCA per block: the raw factor zoo collapsed into a few "super factors"
    (principal components) with loadings, so correlated stats (e.g. tackles
    and interceptions) don't get counted as independent signal twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"

TEAM_KEY = ["Comp", "Season_End_Year", "Squad"]
STAT_TYPES = ["standard", "shooting", "passing", "defense", "possession", "misc",
              "keepers", "keepers_adv"]
KPI_COLS = ["win_pct", "pts_per_game", "goal_diff_per_game"]

# name -> (source stat table, source column, "rate" as-is or "count" -> per90)
FEATURES: dict[str, tuple[str, str, str]] = {
    "save_pct":            ("keepers", "Save_percent", "rate"),
    "ga90":                ("keepers", "GA90", "rate"),
    "cs_pct":               ("keepers", "CS_percent", "rate"),
    "psxg_per_sot":        ("keepers_adv", "PSxG_per_SoT_Expected", "rate"),
    "launch_cmp_pct":      ("keepers_adv", "Cmp_percent_Launched", "rate"),
    "cross_stop_pct":      ("keepers_adv", "Stp_percent_Crosses", "rate"),
    "pass_cmp_pct":        ("passing", "Cmp_percent_Total", "rate"),
    "aerial_won_pct":      ("misc", "Won_percent_Aerial", "rate"),
    "tackles_p90":         ("defense", "Tkl_Tackles", "count"),
    "interceptions_p90":   ("defense", "Int", "count"),
    "blocks_p90":          ("defense", "Blocks_Blocks", "count"),
    "clearances_p90":      ("defense", "Clr", "count"),
    "prog_passes_p90":     ("passing", "Prog", "count"),
    "key_passes_p90":      ("passing", "KP", "count"),
    "passes_into_box_p90": ("passing", "PPA", "count"),
    "prog_carries_p90":    ("possession", "Prog_Carries", "count"),
    "tkl_int_p90":         ("defense", "Tkl+Int", "count"),
    "recoveries_p90":      ("misc", "Recov", "count"),
    "goals_p90":           ("standard", "Gls_Per", "rate"),
    "assists_p90":         ("standard", "Ast_Per", "rate"),
    "xg_p90":              ("standard", "xG_Per", "rate"),
    "xag_p90":             ("standard", "xAG_Per", "rate"),
    "sot_pct":             ("shooting", "SoT_percent_Standard", "rate"),
    "dribble_succ_pct":    ("possession", "Succ_percent_Dribbles", "rate"),
    "carries_into_box_p90": ("possession", "CPA_Carries", "count"),
}

POSITION_BLOCKS: dict[str, list[str]] = {
    "GK": ["save_pct", "ga90", "cs_pct", "psxg_per_sot", "launch_cmp_pct",
           "cross_stop_pct"],
    "DF": ["pass_cmp_pct", "aerial_won_pct", "tackles_p90", "interceptions_p90",
           "blocks_p90", "clearances_p90", "prog_passes_p90"],
    "MF": ["pass_cmp_pct", "key_passes_p90", "passes_into_box_p90",
           "prog_passes_p90", "prog_carries_p90", "tkl_int_p90", "recoveries_p90"],
    "FW": ["goals_p90", "assists_p90", "xg_p90", "xag_p90", "sot_pct",
           "dribble_succ_pct", "carries_into_box_p90"],
}

WEIGHT_COL = "Min_Playing"   # minutes played -- analog of PA/IP in moneyball


def _pq(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


def _primary_pos(pos) -> str | None:
    """'FW,MF' -> 'FW' -- first-listed FBref position is the primary one."""
    if pos is None or (isinstance(pos, float) and np.isnan(pos)) or pos == "":
        return None
    return str(pos).split(",")[0]


def _per90(df: pd.DataFrame, count_col: str, minutes_col: str = "Mins_Per_90") -> pd.Series:
    m = pd.to_numeric(df[minutes_col], errors="coerce")
    c = pd.to_numeric(df[count_col], errors="coerce")
    return c / m.replace(0, np.nan)


# --------------------------------------------------------------------------- #
# player-season wide table: merge the 8 stat pages into one row per player
# --------------------------------------------------------------------------- #
def build_player_features() -> pd.DataFrame:
    stats = {s: _pq(f"player_{s}") for s in STAT_TYPES}
    key = TEAM_KEY + ["Url"]

    base = stats["standard"][key + ["Player", "Nation", "Pos", WEIGHT_COL]].copy()
    base[WEIGHT_COL] = pd.to_numeric(base[WEIGHT_COL], errors="coerce")
    base["pos_bucket"] = base["Pos"].map(_primary_pos)

    for out_col, (table, src_col, kind) in FEATURES.items():
        src = stats[table]
        val = (pd.to_numeric(src[src_col], errors="coerce") if kind == "rate"
               else _per90(src, src_col))
        tmp = src[key].copy()
        tmp[out_col] = val
        base = base.merge(tmp, on=key, how="left")

    return base


# --------------------------------------------------------------------------- #
# team-season KPI panel, from actual match results (not from the stat pages)
# --------------------------------------------------------------------------- #
def build_kpis() -> pd.DataFrame:
    mr = _pq("match_results").dropna(subset=["HomeGoals", "AwayGoals"])
    home = mr.rename(columns={"Home": "Squad", "HomeGoals": "GF", "AwayGoals": "GA"})
    away = mr.rename(columns={"Away": "Squad", "AwayGoals": "GF", "HomeGoals": "GA"})
    both = pd.concat([home[TEAM_KEY + ["GF", "GA"]], away[TEAM_KEY + ["GF", "GA"]]],
                     ignore_index=True)

    both["W"] = (both["GF"] > both["GA"]).astype(int)
    both["D"] = (both["GF"] == both["GA"]).astype(int)
    both["L"] = (both["GF"] < both["GA"]).astype(int)
    both["Pts"] = both["W"] * 3 + both["D"]

    kpi = (both.groupby(TEAM_KEY)
           .agg(MP=("GF", "size"), W=("W", "sum"), D=("D", "sum"), L=("L", "sum"),
                GF=("GF", "sum"), GA=("GA", "sum"), Pts=("Pts", "sum"))
           .reset_index())
    kpi["win_pct"] = kpi["W"] / kpi["MP"]
    kpi["pts_per_game"] = kpi["Pts"] / kpi["MP"]
    kpi["goal_diff_per_game"] = (kpi["GF"] - kpi["GA"]) / kpi["MP"]
    return kpi


# --------------------------------------------------------------------------- #
# position-block team-season aggregation (minutes-weighted, like moneyball's
# _weighted_mean/_aggregate_team_season)
# --------------------------------------------------------------------------- #
def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(0)
    v = values.fillna(0)
    mask = (w > 0) & values.notna()
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def aggregate_block(player_features: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """Team-season rollup of one position block: minutes-weighted mean of
    every factor in that block, over players whose primary position is
    `bucket`."""
    feature_cols = POSITION_BLOCKS[bucket]
    df = player_features[player_features["pos_bucket"] == bucket]

    def agg(g: pd.DataFrame) -> pd.Series:
        out = {f: _weighted_mean(g[f], g[WEIGHT_COL]) for f in feature_cols}
        out[WEIGHT_COL] = g[WEIGHT_COL].sum(min_count=1)
        return pd.Series(out)

    return df.groupby(TEAM_KEY).apply(agg, include_groups=False).reset_index()


# --------------------------------------------------------------------------- #
# factor <-> KPI analytics (unchanged logic from moneyball/factorPerformance.py,
# just parameterized on the football join key)
# --------------------------------------------------------------------------- #
def correlation_table(features: pd.DataFrame, kpis: pd.DataFrame,
                      feature_cols: list[str], kpi_cols: list[str]) -> pd.DataFrame:
    merged = features.merge(kpis, on=TEAM_KEY, how="inner")
    rows = []
    for k in kpi_cols:
        for f in feature_cols:
            sub = merged[[f, k]].dropna()
            if len(sub) < 10:
                continue
            r, p = sp_stats.pearsonr(sub[f], sub[k])
            rows.append({"kpi": k, "factor": f, "r": r, "p": p, "n": len(sub)})
    out = pd.DataFrame(rows)
    out["abs_r"] = out["r"].abs()
    return out.sort_values(["kpi", "abs_r"], ascending=[True, False])


def ols_block(features: pd.DataFrame, kpis: pd.DataFrame,
              feature_cols: list[str], kpi: str) -> pd.DataFrame:
    merged = features.merge(kpis[TEAM_KEY + [kpi]], on=TEAM_KEY, how="inner")
    merged = merged[feature_cols + [kpi]].dropna()
    if len(merged) < len(feature_cols) + 5:
        return pd.DataFrame(columns=["factor", "coef_std", "abs_coef_std"])

    X = StandardScaler().fit_transform(merged[feature_cols])
    y = StandardScaler().fit_transform(merged[[kpi]]).ravel()
    reg = LinearRegression().fit(X, y)

    out = pd.DataFrame({"factor": feature_cols, "coef_std": reg.coef_})
    out["abs_coef_std"] = out["coef_std"].abs()
    out = out.sort_values("abs_coef_std", ascending=False)
    out.attrs["r2"] = reg.score(X, y)
    out.attrs["n"] = len(merged)
    return out


# --------------------------------------------------------------------------- #
# PCA: break the factor zoo down into a handful of "super factors" per block
# ("Kohaerenzen runterbrechen" -- a statistical factor model, PCA-style,
# instead of moneyball's purely fundamental factor picks)
# --------------------------------------------------------------------------- #
def pca_block(features: pd.DataFrame, feature_cols: list[str],
              n_components: int = 3) -> tuple[pd.DataFrame, np.ndarray] | None:
    X = features[feature_cols].dropna()
    if len(X) < len(feature_cols) + 5:
        return None
    Xs = StandardScaler().fit_transform(X)
    n = min(n_components, len(feature_cols))
    pca = PCA(n_components=n).fit(Xs)
    loadings = pd.DataFrame(pca.components_.T, index=feature_cols,
                            columns=[f"PC{i + 1}" for i in range(n)])
    return loadings, pca.explained_variance_ratio_


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_feature_blocks() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return ({bucket: team_season_block_df}, kpi_panel)."""
    player_features = build_player_features()
    blocks = {b: aggregate_block(player_features, b) for b in POSITION_BLOCKS}
    kpi = build_kpis()
    return blocks, kpi


def _print_top(df: pd.DataFrame, kpi_cols: list[str], top: int = 5) -> None:
    for k in kpi_cols:
        sub = df[df["kpi"] == k].head(top)
        if sub.empty:
            continue
        print(f"\n  {k}:")
        for _, row in sub.iterrows():
            sign = "+" if row["r"] >= 0 else "-"
            print(f"    {sign} {row['factor']:20s} r={row['r']:+.3f} "
                  f"p={row['p']:.3g} n={row['n']}")


def main() -> None:
    blocks, kpi = build_feature_blocks()
    print(f"[factor] KPI panel rows (team-seasons): {len(kpi)}")
    for b, df in blocks.items():
        print(f"[factor] {b} block team-seasons: {len(df)}")

    for bucket, feature_cols in POSITION_BLOCKS.items():
        print(f"\n\n=========== {bucket} block ===========")
        block_df = blocks[bucket]

        print(f"\n=== {bucket}: top Pearson correlations per KPI ===")
        corr = correlation_table(block_df, kpi, feature_cols, KPI_COLS)
        _print_top(corr, KPI_COLS)

        print(f"\n=== {bucket}: OLS (standardized) per KPI ===")
        for k in KPI_COLS:
            r = ols_block(block_df, kpi, feature_cols, k)
            if r.empty:
                continue
            print(f"\n  {k}  R^2={r.attrs['r2']:.3f}  n={r.attrs['n']}")
            for _, row in r.head(5).iterrows():
                print(f"    {row['factor']:20s} beta_std={row['coef_std']:+.3f}")

        print(f"\n=== {bucket}: PCA super-factors ===")
        pca_res = pca_block(block_df, feature_cols)
        if pca_res is None:
            print("  SKIP (not enough data)")
            continue
        loadings, explained = pca_res
        for i, col in enumerate(loadings.columns):
            print(f"\n  {col}  (explains {explained[i] * 100:.1f}% of variance)")
            top_loadings = loadings[col].abs().sort_values(ascending=False).head(4)
            for f in top_loadings.index:
                print(f"    {f:20s} loading={loadings.loc[f, col]:+.3f}")


if __name__ == "__main__":
    main()
