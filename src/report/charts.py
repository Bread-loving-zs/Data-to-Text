import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from pathlib import Path
from typing import Optional

from src.config import setup_logging

logger = setup_logging(__name__)


FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS", "sans-serif"
]


def _setup_chinese_font():
    for font_name in FONT_CANDIDATES:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font_name, "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


class ChartGenerator:
    def __init__(self, output_dir: Optional[Path] = None):
        from src.config import OUTPUT_DIR
        self.output_dir = output_dir or OUTPUT_DIR / "charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def bar_chart(self, labels: list[str], values: list[float],
                  title: str = "不合格率对比", xlabel: str = "品类",
                  ylabel: str = "不合格率 (%)", filename: str = "bar_chart.png",
                  color: str = "#E74C3C") -> Path:
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            x = np.arange(len(labels))
            bars = ax.bar(x, values, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                        f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            return filepath
        finally:
            plt.close(fig)

    def trend_line_chart(self, periods: list[str], values: list[float],
                         title: str = "不合格率趋势", xlabel: str = "时间",
                         ylabel: str = "不合格率 (%)", filename: str = "trend_chart.png",
                         color: str = "#3498DB") -> Path:
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            x = np.arange(len(periods))

            ax.plot(x, values, color=color, marker="o", linewidth=2.5, markersize=8,
                    markerfacecolor="white", markeredgewidth=2, markeredgecolor=color)

            for i, (xi, yi) in enumerate(zip(x, values)):
                ax.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=9)

            ax.fill_between(x, values, alpha=0.1, color=color)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(periods, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            return filepath
        finally:
            plt.close(fig)

    def pie_chart(self, labels: list[str], values: list[float],
                  title: str = "不合格项目分布", filename: str = "pie_chart.png") -> Path:
        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

            wedges, texts, autotexts = ax.pie(
                values, labels=None, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                textprops={"fontsize": 10}
            )

            for autotext in autotexts:
                autotext.set_fontweight("bold")

            ax.legend(wedges, [f"{l} ({v})" for l, v in zip(labels, values)],
                      title="类别", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=9)

            ax.set_title(title, fontsize=14, fontweight="bold")

            plt.tight_layout()
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            return filepath
        finally:
            plt.close(fig)

    def rate_comparison_chart(self, categories: list[str], current_rates: list[float],
                              previous_rates: list[float], title: str = "不合格率同比对比",
                              filename: str = "comparison_chart.png") -> Path:
        fig, ax = plt.subplots(figsize=(12, 7))
        try:
            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax.bar(x - width / 2, current_rates, width, label="今年",
                           color="#3498DB", alpha=0.85, edgecolor="white")
            bars2 = ax.bar(x + width / 2, previous_rates, width, label="去年",
                           color="#95A5A6", alpha=0.85, edgecolor="white")

            ax.set_xlabel("品类", fontsize=12)
            ax.set_ylabel("不合格率 (%)", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=9)
            ax.legend(fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            return filepath
        finally:
            plt.close(fig)