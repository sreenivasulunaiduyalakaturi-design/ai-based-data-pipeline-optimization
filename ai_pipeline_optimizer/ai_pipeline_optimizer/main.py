"""End-to-end demo: generate data -> collect runs -> train AI -> optimize -> compare."""
import json

import config
from src import ai_optimizer, collect_runs, data_generator
from src.pipeline import extract, run_pipeline
from src.visualize import plot_comparison

# A deliberately naive default configuration
BASELINE_CFG = {"batch_size": 500, "num_workers": 1, "use_vectorized": 0}


def main():
    print("== 1. Generating data ==")
    data_generator.main()

    print("\n== 2. Collecting pipeline runs ==")
    collect_runs.main()

    print("\n== 3. Training AI models ==")
    model = ai_optimizer.train_runtime_model()
    ai_optimizer.train_anomaly_detector()

    print("\n== 4. AI recommendations (top 5) ==")
    recs = ai_optimizer.recommend_config(model)
    print(recs.to_string(index=False))
    best_cfg = {k: int(recs.loc[0, k]) for k in config.SEARCH_SPACE}

    print("\n== 5. Baseline vs optimized ==")
    df = extract()
    baseline = run_pipeline(BASELINE_CFG, df)
    optimized = run_pipeline(best_cfg, df)
    run_pipeline(best_cfg, df, load=True)  # write clean_data.csv (not timed in comparison)
    speedup = baseline["runtime_sec"] / optimized["runtime_sec"]
    print(f"Baseline : {baseline['runtime_sec']:.2f}s  config={BASELINE_CFG}")
    print(f"Optimized: {optimized['runtime_sec']:.2f}s  config={best_cfg}")
    print(f"Speedup  : {speedup:.1f}x")
    print("Optimized run anomalous?", ai_optimizer.is_anomalous(optimized))

    config.REPORT_DIR.mkdir(exist_ok=True)
    (config.REPORT_DIR / "summary.json").write_text(json.dumps(
        {"baseline": baseline, "optimized": optimized, "speedup": speedup}, indent=2, default=float))
    print("Plot saved ->", plot_comparison(baseline, optimized))


if __name__ == "__main__":
    main()
