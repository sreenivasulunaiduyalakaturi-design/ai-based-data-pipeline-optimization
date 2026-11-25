"""Run the pipeline under many random configs to build a training log for the AI model."""
import random

import pandas as pd

import config
from src.pipeline import extract, run_pipeline


def random_config() -> dict:
    return {k: random.choice(v) for k, v in config.SEARCH_SPACE.items()}


def main(n_runs: int = config.N_TRAINING_RUNS):
    random.seed(config.RANDOM_STATE)
    df = extract()
    records = []
    for i in range(n_runs):
        cfg = random_config()
        rec = run_pipeline(cfg, df)
        records.append(rec)
        print(f"[{i + 1:>3}/{n_runs}] {cfg} -> {rec['runtime_sec']:.2f}s")
    pd.DataFrame(records).to_csv(config.RUN_LOG_PATH, index=False)
    print(f"Saved run log -> {config.RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
