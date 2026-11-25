# AI-Based Data Pipeline Optimization

A simple base project showing how machine learning can tune a data pipeline automatically.

## Idea
1. A configurable ETL pipeline (extract → clean → transform → load) runs on synthetic data.
2. The pipeline is executed under many random configurations (batch size, workers, vectorized vs row-wise ops);
   runtime, memory and throughput are logged.
3. A **RandomForest** model learns `config -> runtime`.
4. The optimizer scores every possible config and recommends the fastest one.
5. An **IsolationForest** flags abnormal pipeline runs (monitoring).
6. Baseline vs optimized results are compared and plotted.

## Structure
```
main.py               # run the whole demo
config.py             # paths, search space, settings
src/data_generator.py # synthetic dirty dataset
src/pipeline.py       # configurable ETL pipeline + metrics
src/collect_runs.py   # builds the training log
src/ai_optimizer.py   # runtime predictor, recommender, anomaly detector
src/visualize.py      # comparison chart
tests/test_pipeline.py
```

## Run
```
pip install -r requirements.txt
python main.py
pytest tests            # optional
```
Outputs: `data/`, `models/`, `reports/comparison.png`, `reports/summary.json`.

## Ideas to extend
- Replace synthetic data with a real CSV/database source.
- Use Bayesian optimization (e.g. Optuna) instead of grid scoring.
- Add real parallelism with multiprocessing / Spark / Dask.
- Add more stages, data-quality checks, and a scheduler (Airflow) with live anomaly alerts.
