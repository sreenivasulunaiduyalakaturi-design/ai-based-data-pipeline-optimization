"""Central configuration for the AI-Based Data Pipeline Optimization project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "raw_data.csv"
RUN_LOG_PATH = DATA_DIR / "pipeline_runs.csv"
MODEL_PATH = MODEL_DIR / "runtime_model.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.joblib"

# Synthetic dataset size
N_ROWS = 200_000

# Search space the optimizer is allowed to explore
SEARCH_SPACE = {
    "batch_size": [500, 1_000, 5_000, 10_000, 25_000, 50_000],
    "num_workers": [1, 2, 4],
    "use_vectorized": [0, 1],      # 1 = pandas vectorized ops, 0 = slow row-wise ops
}

# Number of random configurations executed to build the training log
N_TRAINING_RUNS = 40
RANDOM_STATE = 42
