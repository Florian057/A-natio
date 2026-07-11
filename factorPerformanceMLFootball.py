"""ML layer on top of factorPerformanceFootball.py (this project's analog of
moneyball/factorPerformanceML.py).

factorPerformanceFootball.py runs linear correlation + OLS to find factors
that move each KPI within a position block. This file runs the "predictor"
pass on the same blocks:

  - Random Forest    - non-linear feature importance, robust to correlated
                       features (pass completion and progressive passes are
                       highly collinear, same as batting rate stats in
                       moneyball).
  - Lasso (CV)       - L1 regularization -> sparse shortlist of factors that
                       carry information after collinearity is squeezed out.
  - Ridge  (CV)      - L2 regularization -> keeps everything but shrinks noise.
  - Voting ensemble  - average of the three above.

Structure is copied 1:1 from factorPerformanceML.py; only the feature blocks
and KPI list are football's instead of baseball's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from factorPerformanceFootball import (
    KPI_COLS,
    POSITION_BLOCKS,
    TEAM_KEY,
    build_feature_blocks,
)

RANDOM_STATE = 42
CV_FOLDS = 5
RF_ESTIMATORS = 400


def _prepare(features: pd.DataFrame, kpi: pd.DataFrame,
             feature_cols: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    merged = (features.merge(kpi[TEAM_KEY + [target]], on=TEAM_KEY, how="inner")
              [feature_cols + [target]].dropna())
    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged[target].to_numpy(dtype=float)
    return X, y


def _cv_r2(estimator, X: np.ndarray, y: np.ndarray) -> float:
    if len(X) < CV_FOLDS * 2:
        return float("nan")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def _build_models() -> dict[str, object]:
    lasso = Pipeline([("scale", StandardScaler()),
                      ("lasso", LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE,
                                        max_iter=20000, alphas=100))])
    ridge = Pipeline([("scale", StandardScaler()),
                      ("ridge", RidgeCV(cv=CV_FOLDS,
                                        alphas=np.logspace(-3, 3, 25)))])
    rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS,
                               random_state=RANDOM_STATE, n_jobs=-1)
    ensemble = VotingRegressor([("lasso", lasso), ("ridge", ridge), ("rf", rf)])
    return {"RandomForest": rf, "Lasso": lasso, "Ridge": ridge, "Ensemble": ensemble}


def _feature_importances(model, feature_cols: list[str]) -> pd.Series | None:
    if isinstance(model, RandomForestRegressor):
        return pd.Series(model.feature_importances_, index=feature_cols)
    if isinstance(model, Pipeline):
        last = model.steps[-1][1]
        if hasattr(last, "coef_"):
            return pd.Series(np.abs(last.coef_), index=feature_cols)
    return None


def evaluate_block(features: pd.DataFrame, kpi: pd.DataFrame,
                   feature_cols: list[str], block_name: str) -> pd.DataFrame:
    print(f"\n=== {block_name} block ===")
    all_rows = []
    for target in KPI_COLS:
        X, y = _prepare(features, kpi, feature_cols, target)
        if len(X) < CV_FOLDS * 2:
            print(f"\n  {target}: SKIP (n={len(X)} too small)")
            continue

        models = _build_models()
        print(f"\n  KPI: {target}  n={len(X)}")

        scores = {}
        for name, m in models.items():
            r2 = _cv_r2(m, X, y)
            scores[name] = r2
            print(f"    {name:14s} CV R^2 = {r2:+.3f}")

        for name, m in models.items():
            if name == "Ensemble":
                continue
            m.fit(X, y)
            imp = _feature_importances(m, feature_cols)
            if imp is None:
                continue
            imp = imp.sort_values(ascending=False).head(5)
            print(f"    top-5 by {name}:")
            for f, v in imp.items():
                print(f"      {f:20s} {v:+.4f}")

        for name, r2 in scores.items():
            all_rows.append({"block": block_name, "kpi": target,
                             "model": name, "cv_r2": r2})

    return pd.DataFrame(all_rows)


def main() -> None:
    blocks, kpi = build_feature_blocks()
    print(f"[ml] KPI panel rows (team-seasons): {len(kpi)}")
    for b, df in blocks.items():
        print(f"[ml] {b} block team-seasons: {len(df)}")

    all_scores = [evaluate_block(blocks[b], kpi, cols, b)
                 for b, cols in POSITION_BLOCKS.items()]

    print("\n\n=== SUMMARY: CV R^2 by block / KPI / model ===")
    summary = pd.concat(all_scores, ignore_index=True)
    if not summary.empty:
        wide = summary.pivot_table(index=["block", "kpi"], columns="model",
                                    values="cv_r2")
        print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
