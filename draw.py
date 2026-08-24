"""绘制消融实验的答案提取质量与确定性对比图。

fig5：100% 堆叠柱，展示 S1–S6 各配置的答案提取质量构成。
fig6：100% 堆叠柱，展示 S1–S6 各配置的答案确定性构成（数据见 DEFINITENESS_ROWS）。

默认读取 results/figures/plot_data.json，也可手动传入数据。

用法:
    python draw.py
    python draw.py --data results/figures/plot_data.json --out results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ---------- 配色：由好到差 ----------
EXTRACT_COLORS = {
    "strict_rate": "#1565C0",      # 严格格式 — 深蓝
    "standard_rate": "#66BB6A",    # 标准格式 — 可接受
    "fallback_rate": "#FB8C00",    # 兜底 — 警告
    "unknown_rate": "#CCCCCC",     # 失败 — 浅灰
}
EXTRACT_LABELS = {
    "strict_rate": "严格格式率",
    "standard_rate": "标准格式率",
    "fallback_rate": "兜底率",
    "unknown_rate": "失败率",
}
EXTRACT_ORDER = ["strict_rate", "standard_rate", "fallback_rate", "unknown_rate"]

DEFINITENESS_COLORS = {
    "definite_rate": "#1565C0",
    "normal_rate": "#CCCCCC",
    "uncertain_rate": "#FFB300",
    "ambiguous_rate": "#E65100",
}
DEFINITENESS_LABELS = {
    "definite_rate": "明确率",
    "normal_rate": "其他正常输出",
    "uncertain_rate": "不确定率",
    "ambiguous_rate": "模糊率",
}
DEFINITENESS_ORDER = [
    "definite_rate",
    "normal_rate",
    "uncertain_rate",
    "ambiguous_rate",
]

SHORT_LABELS = ["S1", "S2", "S3", "S4", "S5", "S6"]

# 答案确定性数据（单位：比例 0~1）
DEFINITENESS_ROWS: List[Dict[str, float]] = [
    {"definite_rate": 0.0000, "uncertain_rate": 0.4201, "ambiguous_rate": 0.2145, "normal_rate": 0.3654},
    {"definite_rate": 0.0000, "uncertain_rate": 0.3784, "ambiguous_rate": 0.1572, "normal_rate": 0.4644},
    {"definite_rate": 0.8634, "uncertain_rate": 0.0131, "ambiguous_rate": 0.0145, "normal_rate": 0.1090},
    {"definite_rate": 0.9271, "uncertain_rate": 0.0154, "ambiguous_rate": 0.0118, "normal_rate": 0.0457},
    {"definite_rate": 0.9453, "uncertain_rate": 0.0115, "ambiguous_rate": 0.0093, "normal_rate": 0.0339},
    {"definite_rate": 0.9738, "uncertain_rate": 0.0090, "ambiguous_rate": 0.0028, "normal_rate": 0.0144},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制答案提取与确定性对比图")
    parser.add_argument(
        "--data",
        default="results/figures/plot_data.json",
        help="plot_data.json 路径",
    )
    parser.add_argument(
        "--out",
        default="results/figures",
        help="输出目录",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


def load_rows(data_path: Path) -> List[Dict]:
    with data_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["ablation_rows"]
    for row in rows:
        row["normal_rate"] = max(
            0.0,
            1.0
            - row["definite_rate"]
            - row["ambiguous_rate"]
            - row["uncertain_rate"],
        )
    return rows


def pct(value: float) -> float:
    return value * 100.0


def annotate_segment(ax, x: float, bottom: float, height: float, text: str) -> None:
    if height < 4.0:
        return
    ax.text(
        x,
        bottom + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
        color="#000000",
        fontweight="bold" if height >= 15 else "normal",
    )


def draw_stacked_100(
    ax,
    rows: List[Dict],
    fields: List[str],
    colors: Dict[str, str],
    labels: Dict[str, str],
) -> None:
    """100% 堆叠柱：每根柱总量为 100%，便于横向对比构成。"""
    n = len(rows)
    x = np.arange(n)
    bottom = np.zeros(n)

    for field in fields:
        values = np.array([pct(row[field]) for row in rows])
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            color=colors[field],
            label=labels[field],
            edgecolor="none",
            width=0.62,
        )
        for i, (bar, val) in enumerate(zip(bars, values)):
            annotate_segment(
                ax,
                bar.get_x() + bar.get_width() / 2,
                bottom[i],
                val,
                f"{val:.1f}%",
            )
        bottom += values

    ax.set_ylim(0, 100)
    ax.set_ylabel("占比（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def plot_extractability(rows: List[Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    draw_stacked_100(
        ax,
        rows,
        EXTRACT_ORDER,
        EXTRACT_COLORS,
        EXTRACT_LABELS,
    )
    ax.set_title("图5 答案提取质量分布", fontsize=12, pad=10)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    path = out_dir / "fig5_answer_extractability.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {path}")


def plot_definiteness(rows: List[Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    draw_stacked_100(
        ax,
        rows,
        DEFINITENESS_ORDER,
        DEFINITENESS_COLORS,
        DEFINITENESS_LABELS,
    )
    ax.set_title("图6 答案确定性分布", fontsize=12, pad=10)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    plt.tight_layout()

    path = out_dir / "fig6_answer_definiteness.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}，请先运行 run_ablation.py 和 plot_results.py")

    rows = load_rows(data_path)
    plot_extractability(rows, out_dir)
    plot_definiteness(DEFINITENESS_ROWS, out_dir)
    print(f"\n绘图完成，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
