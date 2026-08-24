"""完整四维评估实验

覆盖四个维度：
  I1 个体智能水平 → 单 Agent 推理能力（更严格的过程质量指标）
  I2 协作效率     → 多 Agent 协作增益（含上下文等价基线）与一致性（Cohen's κ）
  I3 系统稳定性   → 多排列 position bias 测试 + 无偏 pass@k 稳定性
  I4 任务完成度   → 三级格式率 + 模糊/不确定性惩罚

用法：
    from experiments import run_experiments
    run_experiments(data_path="data/dev_rand_split.jsonl", n=None)

或通过 main.py 命令行：
    python main.py --n 200
    python main.py                    # 跑 dev 全集
    python main.py --workers 16
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from concurrent_runner import count_errors, run_in_parallel
from config import SYSTEM_CONFIG
from data_loader import load_commonsenseqa_data
from evaluator import MASEvaluator
from utils import shuffle_choices
from workflow import MASQuestionAnswering, run_context_agent


# ----------------------------------------------------------------
# 失败占位（保持 schema 与正常结果一致）
# ----------------------------------------------------------------
def _placeholder_single_result() -> Dict:
    return {"response": "", "answer": "UNKNOWN"}


def _placeholder_multi_result() -> Dict:
    return {
        "conversation": [],
        "answer": "UNKNOWN",
        "metrics": {
            "total_rounds": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_chars": 0,
            "num_validations": 0,
        },
        "final_state": {
            "messages": [],
            "analyst_output": "",
            "reasoner_output": "",
            "validator_output": "",
            "final_answer": "UNKNOWN",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _normalize(results: List[Dict], placeholder_fn) -> List[Dict]:
    """把 runner 返回的 _error 占位转换成 schema 兼容结构"""
    out = []
    for r in results:
        if not isinstance(r, dict) or "_error" in r or "answer" not in r:
            out.append(placeholder_fn())
        else:
            out.append(r)
    return out


# ----------------------------------------------------------------
# 同时打印 + 写入文件的简易日志器
# ----------------------------------------------------------------
class ReportLogger:
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.lines: List[str] = []
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            # 清空旧日志
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(msg)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


# ----------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------
def run_experiments(
    data_path: str = "data/dev_rand_split.jsonl",
    n: Optional[int] = 50,
    max_workers: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_base_delay: Optional[float] = None,
    out_dir: str = "results",
    num_perms: int = 3,
    stability_k: int = 5,
    stability_n: int = 10,
    no_resume: bool = False,
) -> Dict:
    """运行完整四维评估实验。

    Args:
        data_path: 数据文件路径
        n: 样本数；None 表示跑全集
        max_workers: 并发线程数（默认读 SYSTEM_CONFIG）
        max_retries: 单条样本最大重试次数
        retry_base_delay: 退避基数（秒）
        out_dir: 结果输出目录
        num_perms: I3.1 多排列实验的排列数
        stability_k: I3.2 pass@k 的 k（每题重复次数）
        stability_n: I3.2 pass@k 的 N（参与重复的题数）
        no_resume: True 时清空断点强制重跑

    Returns:
        包含所有指标的字典（同时已写入 out_dir）
    """
    max_workers = max_workers or SYSTEM_CONFIG["max_workers"]
    max_retries = max_retries or SYSTEM_CONFIG["max_retries"]
    retry_base_delay = retry_base_delay or SYSTEM_CONFIG["retry_base_delay"]

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if no_resume and os.path.isdir(ckpt_dir):
        for fn in os.listdir(ckpt_dir):
            if fn.endswith(".jsonl"):
                os.remove(os.path.join(ckpt_dir, fn))
        print(f"已清空断点目录: {ckpt_dir}")

    log_path = os.path.join(out_dir, "full_evaluation_report.txt")
    log = ReportLogger(log_path)

    # ----------------------------------------------------------------
    # 加载数据
    # ----------------------------------------------------------------
    log("=== 加载数据集 ===")
    data = load_commonsenseqa_data(data_path, max_samples=n)
    if not data:
        log("无法加载数据集，请检查文件路径")
        return {}

    test_data = data
    ground_truths = [item["answer"] for item in test_data]
    log(f"使用 {len(test_data)} 条数据进行测试  (data={data_path})")
    log(
        f"并发配置: workers={max_workers}  max_retries={max_retries}  "
        f"num_perms={num_perms}  pass@k 配置: K={stability_k}, N={stability_n}"
    )
    log("")

    mas_system = MASQuestionAnswering()
    evaluator = MASEvaluator()

    runner_kwargs = dict(
        max_workers=max_workers,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        progress_every=max(5, len(test_data) // 50),
    )
    t_total = time.time()

    # ----------------------------------------------------------------
    # 阶段 1 — I1：单 Agent 基线
    # ----------------------------------------------------------------
    log("=== I1: 单智能体基线测试（个体智能水平）===")

    def _task_single(item: Dict, _idx: int) -> Dict:
        return mas_system.run_single_agent(item["question"], item["choices"])

    single_raw = run_in_parallel(
        _task_single,
        test_data,
        checkpoint_path=os.path.join(ckpt_dir, "I1_single.jsonl"),
        desc="I1_single",
        **runner_kwargs,
    )
    single_results = _normalize(single_raw, _placeholder_single_result)
    log(f"  [I1] 失败样本: {count_errors(single_raw)}/{len(test_data)}")

    feasibility_scores = [
        evaluator.eval_reasoning_feasibility(r) for r in single_results
    ]
    avg_feasibility = float(np.mean(feasibility_scores))

    coverage_scores = [
        evaluator.eval_reasoning_coverage(
            r,
            choices=item["choices"],
            num_choices=len(item["choices"]),
        )
        for r, item in zip(single_results, test_data)
    ]
    avg_coverage = float(np.mean(coverage_scores))

    single_acc = evaluator.eval_single_agent_accuracy(single_results, ground_truths)
    log(f"  I1.1 推理可行性均值（v2）: {avg_feasibility:.2%}")
    log(f"  I1.2 推理覆盖质量均值（v2）:{avg_coverage:.2%}")
    log(f"  I1.3 单体准确率:            {single_acc:.2%}")
    log("")

    # ----------------------------------------------------------------
    # 阶段 2 — I2：多 Agent + 上下文等价基线
    # ----------------------------------------------------------------
    log("=== I2: 多智能体协作测试（协作效率）===")

    def _task_multi_with_ctx(item: Dict, _idx: int) -> Dict:
        """一条样本：先跑多 Agent，再跑上下文等价基线（依赖 multi 的 analyst_output）"""
        multi_res = mas_system.run_multi_agent(item["question"], item["choices"])
        analyst_out = multi_res["final_state"].get("analyst_output", "")
        ctx_res = run_context_agent(
            mas_system, item["question"], item["choices"], analyst_out
        )
        return {"multi": multi_res, "context": ctx_res}

    multi_combo_raw = run_in_parallel(
        _task_multi_with_ctx,
        test_data,
        checkpoint_path=os.path.join(ckpt_dir, "I2_multi_ctx.jsonl"),
        desc="I2_multi+ctx",
        **runner_kwargs,
    )

    multi_results: List[Dict] = []
    context_results: List[Dict] = []
    multi_final_states: List[Dict] = []
    n_combo_err = 0
    for r in multi_combo_raw:
        if not isinstance(r, dict) or "_error" in r or "multi" not in r:
            n_combo_err += 1
            multi_results.append(_placeholder_multi_result())
            context_results.append(_placeholder_single_result())
            multi_final_states.append(multi_results[-1]["final_state"])
        else:
            multi_results.append(r["multi"])
            context_results.append(r["context"])
            multi_final_states.append(r["multi"]["final_state"])
    log(f"  [I2] 失败样本: {n_combo_err}/{len(test_data)}")

    multi_acc = evaluator.eval_task_accuracy(multi_results, ground_truths)
    context_acc = evaluator.eval_single_agent_accuracy(context_results, ground_truths)
    gain_result = evaluator.eval_collaboration_gain(single_acc, multi_acc, context_acc)

    consistency_results = [
        evaluator.eval_coordination_consistency(fs) for fs in multi_final_states
    ]
    avg_pipeline_cons = float(
        np.mean([r["pipeline_consistency"] for r in consistency_results])
    )
    avg_kappa = float(np.mean([r["cohens_kappa"] for r in consistency_results]))

    overhead_list = [
        evaluator.eval_communication_overhead(r["metrics"]) for r in multi_results
    ]
    avg_rounds = float(np.mean([o["rounds"] for o in overhead_list]))
    avg_chars = float(np.mean([o["total_chars"] for o in overhead_list]))
    avg_prompt_tokens = float(np.mean([o["prompt_tokens"] for o in overhead_list]))
    avg_completion_tokens = float(np.mean([o["completion_tokens"] for o in overhead_list]))
    avg_tokens = float(np.mean([o["total_tokens"] for o in overhead_list]))
    avg_rework = float(np.mean([o["rework_count"] for o in overhead_list]))

    log(f"  I2.1 原始协作增益:        {gain_result['raw_gain']:+.2%}")
    log(f"  I2.1 上下文增益:          {gain_result.get('context_gain', 0.0):+.2%}")
    log(
        f"  I2.1 纯协作增益:          {gain_result.get('pure_collab_gain', 0.0):+.2%}"
        f"  ← 排除信息量后的真实协作贡献"
    )
    log(
        f"  I2.2 流水线一致性:        {avg_pipeline_cons:.2%}  "
        f"Cohen's κ: {avg_kappa:.3f}"
    )
    log(
        f"  I2.3 通信开销 — 轮次: {avg_rounds:.1f}  字符: {avg_chars:.0f}"
        f"  API token: {avg_tokens:.0f}  返工: {avg_rework:.1f}"
    )
    log("")

    # ----------------------------------------------------------------
    # 阶段 3 — I3.1：多排列 position bias
    # ----------------------------------------------------------------
    log("=== I3: 系统稳定性测试 ===")
    log(f"  I3.1 准备多排列（{num_perms} 个 seed）...")

    # 扁平化为 (perm_idx, item_idx, item) 一次性并发跑完所有排列
    perm_data_per_seed: List[List[Dict]] = []
    flat_perm_inputs: List[Dict] = []
    for seed in range(num_perms):
        perm_items = [shuffle_choices(item, seed=seed * 17 + 3) for item in test_data]
        perm_data_per_seed.append(perm_items)
        for j, perm_item in enumerate(perm_items):
            flat_perm_inputs.append(
                {"perm_idx": seed, "sample_idx": j, "item": perm_item}
            )

    def _task_perm(record: Dict, _idx: int) -> Dict:
        item = record["item"]
        return mas_system.run_single_agent(item["question"], item["choices"])

    flat_perm_raw = run_in_parallel(
        _task_perm,
        flat_perm_inputs,
        checkpoint_path=os.path.join(ckpt_dir, "I3_perm.jsonl"),
        desc="I3.1_perm",
        **runner_kwargs,
    )

    all_perm_results: List[List[Dict]] = [[] for _ in range(num_perms)]
    all_perm_gts: List[List[str]] = [
        [it["answer"] for it in perm] for perm in perm_data_per_seed
    ]
    for record, raw in zip(flat_perm_inputs, flat_perm_raw):
        if not isinstance(raw, dict) or "_error" in raw or "answer" not in raw:
            normalized = _placeholder_single_result()
        else:
            normalized = raw
        all_perm_results[record["perm_idx"]].append(normalized)

    bias_result = evaluator.eval_position_bias(
        baseline_results=single_results,
        all_perm_results=all_perm_results,
        ground_truths=all_perm_gts[0],
    )
    perm_accs = [
        sum(r["answer"] == gt for r, gt in zip(perm_res, perm_gts)) / len(perm_gts)
        for perm_res, perm_gts in zip(all_perm_results, all_perm_gts)
    ]
    avg_perm_acc = float(np.mean(perm_accs))
    avg_degradation = single_acc - avg_perm_acc

    log(
        f"  I3.1 多排列平均退化:     {avg_degradation:+.2%}  "
        f"(基线: {single_acc:.2%}，各排列均值: {avg_perm_acc:.2%})"
    )
    log(f"  I3.1 答案翻转率:         {bias_result['flip_rate']:.2%}")
    log(f"  I3.1 RStd（位置偏好度）: {bias_result['rstd']:.4f}")

    # ----------------------------------------------------------------
    # 阶段 3 — I3.2：pass@k 重复一致性
    # ----------------------------------------------------------------
    log(f"  I3.2 pass@k 重复一致性（前 {stability_n} 题 × {stability_k} 次）...")

    stability_subset = test_data[:stability_n]
    stability_gts = ground_truths[:stability_n]

    flat_pak_inputs = [
        {"item_idx": i, "run_idx": k, "item": item}
        for i, item in enumerate(stability_subset)
        for k in range(stability_k)
    ]

    flat_pak_raw = run_in_parallel(
        _task_perm,  # 同样是单 Agent 调用
        flat_pak_inputs,
        checkpoint_path=os.path.join(ckpt_dir, "I3_passk.jsonl"),
        desc="I3.2_passk",
        **runner_kwargs,
    )

    repeated_results: List[List[Dict]] = [[] for _ in range(stability_n)]
    for record, raw in zip(flat_pak_inputs, flat_pak_raw):
        if not isinstance(raw, dict) or "_error" in raw or "answer" not in raw:
            normalized = _placeholder_single_result()
        else:
            normalized = raw
        repeated_results[record["item_idx"]].append(normalized)

    stab_result = evaluator.eval_answer_consistency(repeated_results, stability_gts)
    log(
        f"  I3.2 稳定性均值（一致率）: {stab_result['avg_stability']:.2%}"
        f"  pass@1: {stab_result.get('avg_pass1', 'N/A')}"
        f"  95%CI宽度: {stab_result.get('avg_ci_width_95', 'N/A')}"
    )
    log("")

    # ----------------------------------------------------------------
    # 阶段 4 — I4：任务完成度
    # ----------------------------------------------------------------
    log("=== I4: 任务完成度 ===")
    task_acc = evaluator.eval_task_accuracy(multi_results, ground_truths)
    extract_result = evaluator.eval_answer_extractability(multi_results)
    definiteness_result = evaluator.eval_answer_definiteness(multi_results)

    log(f"  I4.1 多体任务准确率:  {task_acc:.2%}")
    log(
        f"  I4.2 严格格式率:      {extract_result['strict_rate']:.2%}  "
        f"标准格式率: {extract_result['standard_rate']:.2%}  "
        f"兜底率: {extract_result['fallback_rate']:.2%}  "
        f"失败率: {extract_result['unknown_rate']:.2%}"
    )
    log(
        f"  I4.3 明确率:          {definiteness_result['definite_rate']:.2%}  "
        f"模糊率: {definiteness_result['ambiguous_rate']:.2%}  "
        f"不确定率: {definiteness_result['uncertain_rate']:.2%}"
    )
    log("")

    # ----------------------------------------------------------------
    # 综合报告
    # ----------------------------------------------------------------
    log("=" * 60)
    log("综合评估报告（v2）")
    log("=" * 60)
    log(f"""
【I1 个体智能水平】
  推理可行性均值（v2）:  {avg_feasibility:.2%}  — 强推理词×2 + 选项关联验证
  推理覆盖质量均值（v2）:{avg_coverage:.2%}  — 选项文本语义匹配 + 排除词语境共现
  单体准确率:             {single_acc:.2%}

【I2 协作效率】
  原始协作增益:           {gain_result['raw_gain']:+.2%}
  上下文增益:             {gain_result.get('context_gain', 0.0):+.2%}
  纯协作增益:             {gain_result.get('pure_collab_gain', 0.0):+.2%}
  流水线一致性:           {avg_pipeline_cons:.2%}  Cohen's κ = {avg_kappa:.3f}
  平均通信轮次:           {avg_rounds:.1f}  API token: {avg_tokens:.0f}  返工: {avg_rework:.1f}

【I3 系统稳定性】
  多排列平均退化:         {avg_degradation:+.2%}  ({num_perms} 种排列均值)
  答案翻转率:             {bias_result['flip_rate']:.2%}
  RStd（位置偏好）:       {bias_result['rstd']:.4f}
  稳定性均值:             {stab_result['avg_stability']:.2%}  pass@1: {stab_result.get('avg_pass1', 'N/A')}  CI宽度: {stab_result.get('avg_ci_width_95', 'N/A')}

【I4 任务完成度】
  多体任务准确率:         {task_acc:.2%}
  严格/标准/兜底/失败:    {extract_result['strict_rate']:.2%} / {extract_result['standard_rate']:.2%} / {extract_result['fallback_rate']:.2%} / {extract_result['unknown_rate']:.2%}
  明确/模糊/不确定:       {definiteness_result['definite_rate']:.2%} / {definiteness_result['ambiguous_rate']:.2%} / {definiteness_result['uncertain_rate']:.2%}
""")

    elapsed_total = time.time() - t_total
    log(f"全部实验耗时: {elapsed_total/60:.1f} min ({elapsed_total:.0f} s)")

    # ----------------------------------------------------------------
    # 汇总指标 JSON
    # ----------------------------------------------------------------
    summary = {
        "data_path": data_path,
        "n_samples": len(test_data),
        "I1": {
            "feasibility": avg_feasibility,
            "coverage": avg_coverage,
            "single_acc": single_acc,
        },
        "I2": {
            **gain_result,
            "pipeline_consistency": avg_pipeline_cons,
            "cohens_kappa": avg_kappa,
            "avg_rounds": avg_rounds,
            "avg_chars": avg_chars,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_tokens": avg_tokens,
            "avg_rework": avg_rework,
        },
        "I3": {
            "degradation": avg_degradation,
            "perm_accs": perm_accs,
            "flip_rate": bias_result["flip_rate"],
            "rstd": bias_result["rstd"],
            "stability": stab_result,
        },
        "I4": {
            "task_acc": task_acc,
            "extract": extract_result,
            "definiteness": definiteness_result,
        },
        "elapsed_sec": elapsed_total,
    }
    summary_path = os.path.join(out_dir, "full_evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log(f"汇总指标已保存到: {summary_path}")
    log(f"完整文本报告已保存到: {log_path}")

    return summary
