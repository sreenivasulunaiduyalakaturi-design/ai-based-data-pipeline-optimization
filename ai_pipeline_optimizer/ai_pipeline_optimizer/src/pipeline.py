"""A configurable ETL pipeline: Extract -> Clean -> Transform -> Load.

The pipeline's performance depends on the config (batch size, workers, etc.),
which is what the AI optimizer learns to tune.
"""
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import config


def extract(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _clean_batch(df: pd.DataFrame, vectorized: bool) -> pd.DataFrame:
    df = df.drop_duplicates(subset="transaction_id").copy()
    df["quantity"] = df["quantity"].fillna(df["quantity"].median())
    if vectorized:
        df = df[df["unit_price"] > 0]
    else:  # deliberately slow row-wise path
        keep = [p > 0 for p in df["unit_price"].tolist()]
        df = df[np.array(keep, dtype=bool)]
    return df


def _transform_batch(df: pd.DataFrame, vectorized: bool) -> pd.DataFrame:
    if vectorized:
        df["revenue"] = df["quantity"] * df["unit_price"]
    else:
        df["revenue"] = [q * p for q, p in zip(df["quantity"], df["unit_price"])]
    df["revenue_band"] = pd.cut(df["revenue"], bins=[-np.inf, 100, 500, np.inf],
                                labels=["low", "mid", "high"])
    return df


def _process_batch(batch: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    batch = _clean_batch(batch, bool(cfg["use_vectorized"]))
    return _transform_batch(batch, bool(cfg["use_vectorized"]))


def run_pipeline(cfg: dict, df: pd.DataFrame | None = None, load: bool = False) -> dict:
    """Run the pipeline once with `cfg` and return performance metrics."""
    if df is None:
        df = extract()

    tracemalloc.start()
    t0 = time.perf_counter()

    bs = int(cfg["batch_size"])
    batches = [df.iloc[i:i + bs] for i in range(0, len(df), bs)]

    if int(cfg["num_workers"]) > 1:
        with ThreadPoolExecutor(max_workers=int(cfg["num_workers"])) as ex:
            results = list(ex.map(lambda b: _process_batch(b, cfg), batches))
    else:
        results = [_process_batch(b, cfg) for b in batches]

    out = pd.concat(results, ignore_index=True)
    # Duplicates may span batches, so de-duplicate globally (always, for correctness)
    out = out.drop_duplicates(subset="transaction_id")

    if load:
        config.DATA_DIR.mkdir(exist_ok=True)
        out.to_csv(config.DATA_DIR / "clean_data.csv", index=False)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        **cfg,
        "runtime_sec": runtime,
        "peak_mem_mb": peak / 1e6,
        "rows_in": len(df),
        "rows_out": len(out),
        "throughput_rps": len(df) / runtime if runtime else 0,
    }
