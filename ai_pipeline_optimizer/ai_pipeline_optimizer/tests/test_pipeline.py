import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generator import generate_raw_data
from src.pipeline import run_pipeline


def test_pipeline_cleans_data():
    df = generate_raw_data(2_000)
    cfg = {"batch_size": 500, "num_workers": 2, "use_vectorized": 1}
    res = run_pipeline(cfg, df)
    assert res["rows_out"] < res["rows_in"]
    assert res["runtime_sec"] > 0
