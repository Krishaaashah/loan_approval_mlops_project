"""Create the compact model-results chart sized for Section 6 of the poster."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
metrics_path = ROOT / "artifacts" / "metrics.json"
output_path = ROOT / "poster_screenshots" / "section6_results_chart.png"

with metrics_path.open(encoding="utf-8") as file:
    metrics = json.load(file)

labels = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [
    metrics["accuracy"] * 100,
    metrics["precision"] * 100,
    metrics["recall"] * 100,
    metrics["f1_score"] * 100,
]

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(4.96, 1.0), dpi=100)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(labels, values, color=["#2f78bd", "#3d8ec9", "#2371b8", "#16538f"], width=0.62)
ax.set_ylim(0, 105)
ax.set_yticks([0, 50, 100])
ax.set_yticklabels(["0", "50", "100%"], fontsize=6, color="#4e6680")
ax.tick_params(axis="x", labelsize=7, colors="#173e70", length=0, pad=2)
ax.tick_params(axis="y", length=0, pad=1)
ax.grid(axis="y", color="#d8e5f2", linewidth=0.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="#173e70",
    )

ax.text(
    0.01,
    1.12,
    "Test-set model metrics | 5 Hyperopt candidates tracked in MLflow",
    transform=ax.transAxes,
    fontsize=7,
    fontweight="bold",
    color="#173e70",
    va="bottom",
)

fig.subplots_adjust(left=0.07, right=0.99, bottom=0.28, top=0.74)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=100, facecolor="white")
plt.close(fig)
print(output_path)
