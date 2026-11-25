"""AI components:
1. train_runtime_model - RandomForest regressor that predicts runtime from a config.
2. recommend_config    - scores the whole config space with the model.
3. Anomaly detector    - IsolationForest that flags abnormal pipeline runs.
"""
import itertools

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config

FEATURES = list(config.SEARCH_SPACE.keys())
ANOMALY_COLS = ["runtime_sec", "peak_mem_mb", "throughput_rps"]


def train_runtime_model() -> RandomForestRegressor:
    df = pd.read_csv(config.RUN_LOG_PATH)
    X, y = df[FEATURES], df["runtime_sec"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=config.RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_STATE)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Runtime model  MAE={mean_absolute_error(y_te, pred):.3f}s  R2={r2_score(y_te, pred):.3f}")

    model.fit(X, y)  # refit on all data
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    return model


def recommend_config(model=None, top_k: int = 5) -> pd.DataFrame:
    """Score every config in the search space and return the top_k fastest."""
    model = model or joblib.load(config.MODEL_PATH)
    keys = list(config.SEARCH_SPACE)
    grid = pd.DataFrame(list(itertools.product(*config.SEARCH_SPACE.values())), columns=keys)
    grid["predicted_runtime_sec"] = model.predict(grid[FEATURES])
    return grid.sort_values("predicted_runtime_sec").head(top_k).reset_index(drop=True)


def train_anomaly_detector() -> IsolationForest:
    df = pd.read_csv(config.RUN_LOG_PATH)
    det = IsolationForest(contamination=0.05, random_state=config.RANDOM_STATE).fit(df[ANOMALY_COLS])
    joblib.dump(det, config.ANOMALY_MODEL_PATH)
    return det


def is_anomalous(metrics: dict) -> bool:
    det = joblib.load(config.ANOMALY_MODEL_PATH)
    row = pd.DataFrame([{k: metrics[k] for k in ANOMALY_COLS}])
    return bool(det.predict(row)[0] == -1)
