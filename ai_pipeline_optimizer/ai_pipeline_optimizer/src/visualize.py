"""Plot baseline vs optimized results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def plot_comparison(baseline: dict, optimized: dict, path=None):
    config.REPORT_DIR.mkdir(exist_ok=True)
    path = path or config.REPORT_DIR / "comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(axes, ["runtime_sec", "peak_mem_mb"], ["Runtime (s)", "Peak memory (MB)"]):
        ax.bar(["Baseline", "AI-optimized"], [baseline[key], optimized[key]], color=["#c0392b", "#27ae60"])
        ax.set_title(title)
    fig.suptitle("Pipeline: Baseline vs AI-Optimized")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
