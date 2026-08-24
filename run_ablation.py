"""
消融实验入口：
    python run_ablation.py --n 12 --workers 4 --no-resume
    python run_ablation.py --resume

实验配置：
    S1 SingleReasoner            : 单智能体基线（仅 Reasoner）
    S2 Reasoner+Analyst          : 上下文等价基线（Analyst 上下文 + Reasoner）
    S3 Reasoner+Validator        : 仅验证不修正（Reasoner + 一次性 Validator）
    S4 Reasoner+Validator+Revise : 验证 + 修正反馈（无 Analyst）
    S5 FullMAS_NoRevise          : 完整三体但无修正循环
    S6 FullMAS                   : 完整三智能体协作系统

输出：
    1. 终端打印对比/消融汇总表 + 协作增益分解
    2. results/ablation_summary.csv
    3. results/ablation_summary.json
    4. results/ablation_per_sample.json

协作增益：
    原始协作增益 = S6 − S1        （总提升）
    上下文增益   = S2 − S1        （Analyst 的信息贡献）
    纯协作增益   = S6 − S2        （Validator + Revision 机制净贡献）

模块消融贡献
    移除 Analyst:
    delta = S6_acc - S4_acc
    移除 Validator:
    delta = S6_acc - S2_acc
    移除修正机制:
    delta = S6_acc - S5_acc
    仅保留 Reasoner:
    delta = S6_acc - S1_acc
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List

import numpy as np

from ablation import DEFAULT_CONFIGS, MASConfig, run_with_config
from concurrent_runner import count_errors, run_in_parallel
from config import SYSTEM_CONFIG
from data_loader import load_commonsenseqa_data
from evaluator import MASEvaluator


_FAILED_PLACEHOLDER: Dict = {
    "answer": "UNKNOWN",
    "final_state": {"messages": []},
    "metrics": {
        "total_rounds": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_chars": 0,
        "num_validations": 0,
    },
}


def _accuracy(results: List[Dict], gts: List[str]) -> float:
    if not results:
        return 0.0
    return sum(r["answer"] == gt for r, gt in zip(results, gts)) / len(results)


def _avg(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _to_eval_format(results: List[Dict]) -> List[Dict]:
    """将 ablation 输出适配为 evaluator 期望的结构

    evaluator.eval_answer_extractability / eval_answer_definiteness 通过
    `validator_output` 或 `response` 字段判断格式合规度，因此需要把对应
    文本注入到这两个字段中。Validator 被关闭时，回退使用 Reasoner 文本。
    """
    out: List[Dict] = []
    for r in results:
        fs = r["final_state"]
        validator_text = fs.get("validator_output", "") or ""
        if validator_text == "[disabled]":
            validator_text = ""
        response_text = validator_text or fs.get("reasoner_output", "") or ""
        out.append(
            {
                "answer": r["answer"],
                "response": response_text,
                "validator_output": validator_text,
                "conversation": fs.get("messages", []),
                "final_state": fs,
                "metrics": r["metrics"],
            }
        )
    return out


def run_one_config(
    cfg: MASConfig,
    data: List[Dict],
    gts: List[str],
    evaluator: MASEvaluator,
    out_dir: str = "results",
    max_workers: int = 8,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
) -> Dict:
    """运行单个配置并汇总指标（并发 + 断点续跑）"""
    print(f"\n>>> 运行配置: {cfg.name}")
    t0 = time.time()

    ckpt_path = os.path.join(out_dir, "checkpoints", f"{cfg.name}.jsonl")

    def _task(item: Dict, _idx: int) -> Dict:
        return run_with_config(cfg, item["question"], item["choices"])

    raw_results = run_in_parallel(
        _task,
        data,
        max_workers=max_workers,
        checkpoint_path=ckpt_path,
        desc=cfg.name,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        progress_every=max(5, len(data) // 50),
    )

    n_err = count_errors(raw_results)
    if n_err:
        print(f"  [{cfg.name}] 警告：{n_err}/{len(raw_results)} 条样本最终失败，已用占位填充")

    results: List[Dict] = []
    for r in raw_results:
        if not isinstance(r, dict) or "_error" in r or "answer" not in r:
            results.append(dict(_FAILED_PLACEHOLDER, final_state={"messages": []}))
        else:
            results.append(r)

    acc = _accuracy(results, gts)
    avg_rounds = _avg([r["metrics"]["total_rounds"] for r in results])
    avg_chars = _avg([
        r["metrics"].get("total_chars", r["metrics"].get("total_tokens", 0))
        for r in results
    ])
    avg_prompt_tokens = _avg([r["metrics"].get("prompt_tokens", 0) for r in results])
    avg_completion_tokens = _avg([r["metrics"].get("completion_tokens", 0) for r in results])
    avg_tokens = _avg([r["metrics"].get("total_tokens", 0) for r in results])
    avg_revs = _avg([r["metrics"]["num_validations"] for r in results])

    eval_payload = _to_eval_format(results)
    ext = evaluator.eval_answer_extractability(eval_payload)
    defi = evaluator.eval_answer_definiteness(eval_payload)

    return {
        "config": cfg.name,
        "enable_analyst": cfg.enable_analyst,
        "enable_validator": cfg.enable_validator,
        "enable_revision": cfg.enable_revision,
        "accuracy": acc,
        "avg_rounds": avg_rounds,
        "avg_chars": avg_chars,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_tokens": avg_tokens,
        "avg_revisions": avg_revs,
        "strict_rate": ext.get("strict_rate", 0.0),
        "standard_rate": ext.get("standard_rate", 0.0),
        "fallback_rate": ext.get("fallback_rate", 0.0),
        "unknown_rate": ext.get("unknown_rate", 0.0),
        "definite_rate": defi.get("definite_rate", 0.0),
        "ambiguous_rate": defi.get("ambiguous_rate", 0.0),
        "uncertain_rate": defi.get("uncertain_rate", 0.0),
        "elapsed_sec": round(time.time() - t0, 2),
        "_per_sample": [
            {
                "answer": r["answer"],
                "rounds": r["metrics"]["total_rounds"],
                "tokens": r["metrics"].get("total_tokens", 0),
                "revisions": r["metrics"]["num_validations"],
            }
            for r in results
        ],
    }


def print_summary_table(summary: List[Dict]) -> None:
    print("\n" + "=" * 110)
    print("对比 / 消融实验结果汇总")
    print("=" * 110)
    header = (
        f"{'Config':<32}{'A':>3}{'V':>3}{'R':>3}"
        f"{'Acc':>9}{'Strict':>9}{'Definite':>10}"
        f"{'Rounds':>9}{'Chars':>9}{'Tokens':>9}{'Revs':>7}{'Time(s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for s in summary:
        a = "√" if s["enable_analyst"] else "×"
        v = "√" if s["enable_validator"] else "×"
        r = "√" if s["enable_revision"] else "×"
        print(
            f"{s['config']:<32}{a:>3}{v:>3}{r:>3}"
            f"{s['accuracy']:>8.2%} {s['strict_rate']:>8.2%} {s['definite_rate']:>9.2%}"
            f"{s['avg_rounds']:>9.2f}{s['avg_chars']:>9.0f}{s['avg_tokens']:>9.0f}{s['avg_revisions']:>7.2f}"
            f"{s['elapsed_sec']:>9.1f}"
        )
    print("-" * len(header))
    print("注: A=Analyst, V=Validator, R=Revision; Strict=严格格式率; Definite=答案明确率")


def print_collaboration_gain(summary: List[Dict], evaluator: MASEvaluator) -> None:
    by_name = {s["config"]: s for s in summary}
    needed = ("S1_SingleReasoner", "S2_Reasoner+Analyst", "S6_FullMAS")
    if not all(k in by_name for k in needed):
        return
    single_acc = by_name["S1_SingleReasoner"]["accuracy"]
    ctx_acc = by_name["S2_Reasoner+Analyst"]["accuracy"]
    full_acc = by_name["S6_FullMAS"]["accuracy"]
    gain = evaluator.eval_collaboration_gain(single_acc, full_acc, ctx_acc)
    print("\n=== 协作增益分解 (基于 S1 / S2 / S6) ===")
    print(f"  单智能体基线 (S1):   {single_acc:.2%}")
    print(f"  上下文等价基线 (S2): {ctx_acc:.2%}")
    print(f"  完整 MAS (S6):       {full_acc:.2%}")
    print(f"  原始协作增益:        {gain['raw_gain']:+.2%}")
    print(f"  上下文增益:          {gain.get('context_gain', 0.0):+.2%}")
    print(f"  纯协作增益:          {gain.get('pure_collab_gain', 0.0):+.2%}")


def print_module_ablation(summary: List[Dict]) -> None:
    """逐模块消融贡献：以 S6 完整系统为参照点反推各模块贡献"""
    by_name = {s["config"]: s for s in summary}
    if "S6_FullMAS" not in by_name:
        return
    full = by_name["S6_FullMAS"]["accuracy"]

    print("\n=== 模块消融贡献 (Δ = S6 完整 MAS - 移除某模块后的配置) ===")

    pairs = [
        ("移除 Analyst", "S4_Reasoner+Validator+Revise"),
        ("移除 Validator", "S2_Reasoner+Analyst"),
        ("移除修正机制", "S5_FullMAS_NoRevise"),
        ("仅保留 Reasoner", "S1_SingleReasoner"),
    ]
    for label, key in pairs:
        if key in by_name:
            delta = full - by_name[key]["accuracy"]
            print(f"  {label:<14}{key:<32} Acc={by_name[key]['accuracy']:.2%}  Δ={delta:+.2%}")


def save_results(summary: List[Dict], out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)

    per_sample = {s["config"]: s.pop("_per_sample") for s in summary}

    json_path = os.path.join(out_dir, "ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    sample_path = os.path.join(out_dir, "ablation_per_sample.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {json_path} / {csv_path} / {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAS 对比/消融实验")
    parser.add_argument(
        "--data",
        default="data/dev_rand_split.jsonl",
        help="CommonsenseQA 数据文件路径",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="样本数量（默认 None 表示跑完整数据集；可指定整数限定前 N 条）",
    )
    parser.add_argument(
        "--out", default="results", help="结果输出目录（默认 results/）"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并发线程数（默认读 .env 中的 MAS_MAX_WORKERS=8）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有断点强制重跑并覆盖 summary（默认会自动从 checkpoints/ 续跑，与 main.py 一致）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="[已弃用] 兼容旧脚本，等同于默认行为（自动续跑）；改用 --no-resume 显式重跑",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== 加载数据 ===")
    data = load_commonsenseqa_data(args.data, max_samples=args.n)
    if not data:
        print("数据为空，请检查数据文件路径")
        return

    gts = [item["answer"] for item in data]
    print(f"实际使用样本数: {len(data)}")

    max_workers = args.workers or SYSTEM_CONFIG["max_workers"]
    max_retries = SYSTEM_CONFIG["max_retries"]
    retry_base_delay = SYSTEM_CONFIG["retry_base_delay"]
    print(f"并发配置: workers={max_workers}  max_retries={max_retries}")

    if args.no_resume:
        ckpt_dir = os.path.join(args.out, "checkpoints")
        if os.path.isdir(ckpt_dir):
            removed = 0
            for fn in os.listdir(ckpt_dir):
                if fn.endswith(".jsonl"):
                    os.remove(os.path.join(ckpt_dir, fn))
                    removed += 1
            if removed:
                print(f"[--no-resume] 已清空断点目录: {ckpt_dir}（删除 {removed} 个 .jsonl）")
        for fn in (
            "ablation_summary.json",
            "ablation_summary.csv",
            "ablation_per_sample.json",
        ):
            fp = os.path.join(args.out, fn)
            if os.path.exists(fp):
                os.remove(fp)
    else:
        print("默认续跑模式：复用 checkpoints/ 中的断点（如需强制重跑加 --no-resume）")

    evaluator = MASEvaluator()

    summary: List[Dict] = []
    for cfg in DEFAULT_CONFIGS:
        summary.append(
            run_one_config(
                cfg,
                data,
                gts,
                evaluator,
                out_dir=args.out,
                max_workers=max_workers,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
        )

    print_summary_table(summary)
    print_collaboration_gain(summary, evaluator)
    print_module_ablation(summary)
    save_results(summary, args.out)


if __name__ == "__main__":
    main()
