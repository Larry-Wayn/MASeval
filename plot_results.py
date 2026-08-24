"""绘制 MAS 对比实验、消融实验与任务完成度图像。

默认读取:
    results/ablation_summary.csv
    results/checkpoints/S1_SingleReasoner.jsonl

输出:
    results/figures/*.png

用法:
    python plot_results.py
    python plot_results.py --results results --data data/dev_rand_split.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

from data_loader import load_commonsenseqa_data
from evaluator import MASEvaluator


CONFIG_LABELS = {
    "S1_SingleReasoner": "S1\nSingle",
    "S2_Reasoner+Analyst": "S2\n+Analyst",
    "S3_Reasoner+Validator": "S3\n+Validator",
    "S4_Reasoner+Validator+Revise": "S4\n+Validator\n+Revise",
    "S5_FullMAS_NoRevise": "S5\nFull\nNoRevise",
    "S6_FullMAS": "S6\nFull MAS",
}

PERCENT_FIELDS = {
    "accuracy",
    "strict_rate",
    "standard_rate",
    "fallback_rate",
    "unknown_rate",
    "definite_rate",
    "ambiguous_rate",
    "uncertain_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 MAS 实验结果图像")
    parser.add_argument(
        "--results",
        default="results",
        help="结果目录，默认 results",
    )
    parser.add_argument(
        "--data",
        default="data/dev_rand_split.jsonl",
        help="CommonsenseQA 数据文件，默认 data/dev_rand_split.jsonl",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="图片输出目录，默认写入 <results>/figures",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    """设置中文字体与基础图像风格。"""
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


def load_ablation_summary(results_dir: Path) -> List[Dict]:
    csv_path = results_dir / "ablation_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 {csv_path}")

    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = dict(row)
            for key, value in row.items():
                if key in {
                    "accuracy",
                    "avg_rounds",
                    "avg_chars",
                    "avg_prompt_tokens",
                    "avg_completion_tokens",
                    "avg_tokens",
                    "avg_revisions",
                    "strict_rate",
                    "standard_rate",
                    "fallback_rate",
                    "unknown_rate",
                    "definite_rate",
                    "ambiguous_rate",
                    "uncertain_rate",
                    "elapsed_sec",
                }:
                    converted[key] = float(value)
                elif key in {"enable_analyst", "enable_validator", "enable_revision"}:
                    converted[key] = value == "True"
            converted["label"] = CONFIG_LABELS.get(converted["config"], converted["config"])
            rows.append(converted)
    return rows


def pct(value: float) -> float:
    return value * 100.0


def savefig(out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()
    print(f"已保存: {out_dir / filename}")


def bar_with_values(
    labels: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    out_dir: Path,
    filename: str,
    ylim: Tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    savefig(out_dir, filename)


def unique_checkpoint_records(path: Path) -> List[Dict]:
    """读取 checkpoint，并按 __idx 去重。

    断点文件中可能因续跑出现重复样本；此处保留每个样本最后一次结果。
    """
    by_idx: Dict[int, Dict] = {}
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_idx[int(record["__idx"])] = record
    return [by_idx[idx] for idx in sorted(by_idx)]


def compute_single_agent_process_metrics(
    results_dir: Path,
    data_path: Path,
) -> Dict[str, float]:
    """从 S1 checkpoint 重新计算推理可行性与推理覆盖质量。"""
    ckpt_path = results_dir / "checkpoints" / "S1_SingleReasoner.jsonl"
    records = unique_checkpoint_records(ckpt_path)
    if not records:
        print(f"未找到或无法读取 {ckpt_path}，跳过推理过程质量指标。")
        return {}

    data = load_commonsenseqa_data(str(data_path), max_samples=len(records))
    if not data:
        print(f"无法加载数据 {data_path}，跳过推理过程质量指标。")
        return {}

    evaluator = MASEvaluator()
    single_results = []
    for record in records:
        result = record["result"]
        final_state = result.get("final_state", {})
        single_results.append(
            {
                "response": final_state.get("reasoner_output", ""),
                "answer": result.get("answer", "UNKNOWN"),
            }
        )

    feasibility_scores = [
        evaluator.eval_reasoning_feasibility(item) for item in single_results
    ]
    coverage_scores = [
        evaluator.eval_reasoning_coverage(
            result,
            choices=item["choices"],
            num_choices=len(item["choices"]),
        )
        for result, item in zip(single_results, data)
    ]

    return {
        "feasibility": sum(feasibility_scores) / len(feasibility_scores),
        "coverage": sum(coverage_scores) / len(coverage_scores),
    }


def row_by_config(rows: Iterable[Dict]) -> Dict[str, Dict]:
    return {row["config"]: row for row in rows}


def plot_single_agent_metrics(
    rows: List[Dict],
    process_metrics: Dict[str, float],
    out_dir: Path,
) -> None:
    by_name = row_by_config(rows)
    s1 = by_name["S1_SingleReasoner"]

    metrics = {
        "准确率": pct(s1["accuracy"]),
        "推理可行性": pct(process_metrics.get("feasibility", 0.0)),
        "推理覆盖质量": pct(process_metrics.get("coverage", 0.0)),
        "标准格式率": pct(s1["standard_rate"]),
        "失败率": pct(s1["unknown_rate"]),
        "不确定率": pct(s1["uncertain_rate"]),
    }

    bar_with_values(
        list(metrics.keys()),
        list(metrics.values()),
        ylabel="比例（%）",
        title="图1 单智能体能力指标",
        out_dir=out_dir,
        filename="fig1_single_agent_metrics.png",
        ylim=(0, 105),
    )


def plot_accuracy_comparison(rows: List[Dict], out_dir: Path) -> None:
    by_name = row_by_config(rows)
    configs = ["S1_SingleReasoner", "S2_Reasoner+Analyst", "S6_FullMAS"]
    labels = [CONFIG_LABELS[c] for c in configs]
    values = [pct(by_name[c]["accuracy"]) for c in configs]

    bar_with_values(
        labels,
        values,
        ylabel="准确率（%）",
        title="图2 不同系统配置准确率对比",
        out_dir=out_dir,
        filename="fig2_accuracy_comparison.png",
        ylim=(75, 90),
    )


def plot_collaboration_gain(rows: List[Dict], out_dir: Path) -> None:
    by_name = row_by_config(rows)
    s1 = by_name["S1_SingleReasoner"]["accuracy"]
    s2 = by_name["S2_Reasoner+Analyst"]["accuracy"]
    s6 = by_name["S6_FullMAS"]["accuracy"]

    gains = {
        "原始协作增益": pct(s6 - s1),
        "上下文增益": pct(s2 - s1),
        "纯协作增益": pct((s6 - s1) - (s2 - s1)),
    }

    bar_with_values(
        list(gains.keys()),
        list(gains.values()),
        ylabel="增益（百分点）",
        title="图3 协作增益分解",
        out_dir=out_dir,
        filename="fig3_collaboration_gain.png",
    )


def plot_ablation_accuracy(rows: List[Dict], out_dir: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [pct(row["accuracy"]) for row in rows]

    bar_with_values(
        labels,
        values,
        ylabel="准确率（%）",
        title="图4 消融实验准确率对比",
        out_dir=out_dir,
        filename="fig4_ablation_accuracy.png",
        ylim=(75, 90),
    )


def plot_extractability_stacked(rows: List[Dict], out_dir: Path) -> None:
    labels = [row["label"] for row in rows]
    series = [
        ("strict_rate", "严格格式率"),
        ("standard_rate", "标准格式率"),
        ("fallback_rate", "兜底率"),
        ("unknown_rate", "失败率"),
    ]

    plt.figure(figsize=(9.5, 5.2))
    bottom = [0.0 for _ in rows]
    for field, name in series:
        values = [pct(row[field]) for row in rows]
        plt.bar(labels, values, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, values)]

    plt.ylabel("比例（%）")
    plt.title("图5 答案提取质量分布")
    plt.legend(ncol=2)
    savefig(out_dir, "fig5_answer_extractability.png")


def plot_definiteness_stacked(rows: List[Dict], out_dir: Path) -> None:
    labels = [row["label"] for row in rows]
    series = [
        ("definite_rate", "明确率"),
        ("ambiguous_rate", "模糊率"),
        ("uncertain_rate", "不确定率"),
        ("normal_rate", "其他正常输出"),
    ]

    enriched = []
    for row in rows:
        copied = dict(row)
        copied["normal_rate"] = max(
            0.0,
            1.0
            - copied["definite_rate"]
            - copied["ambiguous_rate"]
            - copied["uncertain_rate"],
        )
        enriched.append(copied)

    plt.figure(figsize=(9.5, 5.2))
    bottom = [0.0 for _ in enriched]
    for field, name in series:
        values = [pct(row[field]) for row in enriched]
        plt.bar(labels, values, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, values)]

    plt.ylabel("比例（%）")
    plt.title("图6 答案确定性分布")
    plt.legend(ncol=2)
    savefig(out_dir, "fig6_answer_definiteness.png")


def plot_communication_chars(rows: List[Dict], out_dir: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [row["avg_chars"] for row in rows]

    bar_with_values(
        labels,
        values,
        ylabel="平均输出字符数",
        title="图7 不同配置的通信开销",
        out_dir=out_dir,
        filename="fig7_communication_chars.png",
    )


def plot_rounds_and_revisions(rows: List[Dict], out_dir: Path) -> None:
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))

    plt.figure(figsize=(9.5, 5.0))
    plt.plot(x, [row["avg_rounds"] for row in rows], marker="o", label="平均轮次")
    plt.plot(
        x,
        [row["avg_revisions"] for row in rows],
        marker="s",
        label="平均返工次数",
    )
    plt.xticks(x, labels)
    plt.ylabel("次数")
    plt.title("图8 运行轮次与返工次数")
    plt.legend()
    savefig(out_dir, "fig8_rounds_revisions.png")


def plot_output_stability_proxy(rows: List[Dict], out_dir: Path) -> None:
    """基于已有 ablation 字段绘制输出稳定性代理指标。"""
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.35

    plt.figure(figsize=(9.5, 5.0))
    plt.bar(
        [i - width / 2 for i in x],
        [pct(row["unknown_rate"]) for row in rows],
        width=width,
        label="失败率",
    )
    plt.bar(
        [i + width / 2 for i in x],
        [pct(row["uncertain_rate"]) for row in rows],
        width=width,
        label="不确定率",
    )
    plt.xticks(x, labels)
    plt.ylabel("比例（%）")
    plt.title("图9 输出稳定性代理指标")
    plt.legend()
    savefig(out_dir, "fig9_output_stability_proxy.png")


def maybe_plot_full_stability(results_dir: Path, out_dir: Path) -> None:
    """如果存在完整四维评估汇总，则绘制真正的 I3 稳定性图。"""
    summary_path = results_dir / "full_evaluation_summary.json"
    if not summary_path.exists():
        print(
            "未找到 full_evaluation_summary.json，跳过 I3 专项稳定性图。"
            "如需绘制，请先运行 main.py 完整四维评估。"
        )
        return

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    i1 = summary.get("I1", {})
    i3 = summary.get("I3", {})
    baseline_acc = i1.get("single_acc")
    perm_accs = i3.get("perm_accs", [])

    if baseline_acc is not None and perm_accs:
        labels = ["原始"] + [f"排列{i + 1}" for i in range(len(perm_accs))]
        values = [pct(baseline_acc)] + [pct(v) for v in perm_accs]
        bar_with_values(
            labels,
            values,
            ylabel="准确率（%）",
            title="图10 选项扰动前后准确率",
            out_dir=out_dir,
            filename="fig10_position_bias_accuracy.png",
            ylim=(0, 100),
        )

    stability = i3.get("stability", {})
    stability_metrics = {
        "答案翻转率": pct(i3.get("flip_rate", 0.0)),
        "位置偏好RStd": i3.get("rstd", 0.0),
        "重复一致率": pct(stability.get("avg_stability", 0.0)),
    }
    if any(stability_metrics.values()):
        bar_with_values(
            list(stability_metrics.keys()),
            list(stability_metrics.values()),
            ylabel="指标值",
            title="图11 系统稳定性专项指标",
            out_dir=out_dir,
            filename="fig11_stability_metrics.png",
        )


def write_plot_data(rows: List[Dict], process_metrics: Dict[str, float], out_dir: Path) -> None:
    """保存绘图所用关键数据，便于论文制表或复核。"""
    by_name = row_by_config(rows)
    plot_data = {
        "single_agent_process_metrics": process_metrics,
        "accuracy_comparison": {
            key: by_name[key]["accuracy"]
            for key in ["S1_SingleReasoner", "S2_Reasoner+Analyst", "S6_FullMAS"]
        },
        "collaboration_gain": {
            "raw_gain": by_name["S6_FullMAS"]["accuracy"]
            - by_name["S1_SingleReasoner"]["accuracy"],
            "context_gain": by_name["S2_Reasoner+Analyst"]["accuracy"]
            - by_name["S1_SingleReasoner"]["accuracy"],
            "pure_collaboration_gain": (
                by_name["S6_FullMAS"]["accuracy"]
                - by_name["S1_SingleReasoner"]["accuracy"]
            )
            - (
                by_name["S2_Reasoner+Analyst"]["accuracy"]
                - by_name["S1_SingleReasoner"]["accuracy"]
            ),
        },
        "ablation_rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "plot_data.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(plot_data, f, ensure_ascii=False, indent=2)
    print(f"绘图数据已保存: {path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    results_dir = Path(args.results)
    data_path = Path(args.data)
    out_dir = Path(args.out) if args.out else results_dir / "figures"

    rows = load_ablation_summary(results_dir)
    process_metrics = compute_single_agent_process_metrics(results_dir, data_path)

    plot_single_agent_metrics(rows, process_metrics, out_dir)
    plot_accuracy_comparison(rows, out_dir)
    plot_collaboration_gain(rows, out_dir)
    plot_ablation_accuracy(rows, out_dir)
    plot_extractability_stacked(rows, out_dir)
    plot_definiteness_stacked(rows, out_dir)
    plot_communication_chars(rows, out_dir)
    plot_rounds_and_revisions(rows, out_dir)
    plot_output_stability_proxy(rows, out_dir)
    maybe_plot_full_stability(results_dir, out_dir)
    write_plot_data(rows, process_metrics, out_dir)

    print(f"\n绘图完成，所有图片位于: {out_dir}")


if __name__ == "__main__":
    main()
