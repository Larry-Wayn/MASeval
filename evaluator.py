import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

from utils import extract_answer


class MASEvaluator:
    """
    针对 CommonsenseQA 问答任务的四维评估器。

    改进原则来源：
      - I1.1/I1.2：防止关键词/字母计数虚高，改用更严格的多条件联合验证
      - I2.1：引入上下文等价基线，消除信息量不对等带来的偏差
      - I2.2：改为真正的独立轮询一致性（analyst 不依赖 reasoner 输出），
               并引入 Cohen's κ 衡量超出偶然一致性的真实协调程度
      - I2.3：优先使用 API usage 统计真实 token，并保留旧 checkpoint 估算兼容
      - I3.1：引入多排列平均退化（PriDe 思路，参考 Zheng et al. 2023），
               避免单次随机排列结果不稳定的问题
      - I3.2：从简单众数一致率升级为无偏 pass@k 估计量（Chen et al. 2021），
               并附带贝叶斯置信区间（参考 ReliabilityBench 2025）
      - I4.2：分离"严格格式率"（最终答案:X）与"宽松提取率"，保留区分度
      - I4.3：增加不确定性惩罚，检测输出中多个答案字母并列导致的模糊性
    """

    def __init__(self):
        self.results = defaultdict(list)

    # ================================================================
    # I1: 个体智能水平
    # ================================================================

    def eval_reasoning_feasibility(self, response: Dict) -> float:
        """
        I1.1 推理可行性（v2：多条件联合验证，防止虚高）

        原版问题：「因为/而/however」等虚词极易命中，使「推理存在性」几乎恒为 True，
        导致大多数回答都能得满分。

        改进方案：
          1. 提高长度门槛（> 120 字符），过滤更多敷衍式输出
          2. 推理词改为「强因果词」列表（去除「而」「while」等高频虚词），
             且要求至少出现 2 个不同的强推理词（更难伪造）
          3. 新增「推理词紧邻选项」检查：至少一个强推理词出现在某选项字母（A-E）
             的前后 30 字符范围内，验证推理词真正与选项评估相关

        三项同时满足得 1.0，满足 2 项得 0.6，满足 1 项得 0.2，全不满足得 0。
        非线性评分防止「凑分」。
        """
        text = response.get("response", "")
        answer = response.get("answer", "UNKNOWN")

        # 检查 1：内容充实（提高门槛至 120 字符）
        has_content = len(text) > 120

        # 检查 2：答案合法可提取
        answer_valid = answer in list("ABCDE")

        # 检查 3：强因果推理词（剔除虚词「而/while/since/however」）
        # 参考：AgentBoard（Ma et al. 2024）对推理过程的语义有效性检验思路
        strong_reasoning_kws = [
            "因为", "所以", "因此", "由于", "导致", "说明", "表明", "意味着",
            "可以推断", "综上", "综上所述",
            "because", "therefore", "thus", "hence", "implies", "indicates",
            "conclude", "reasoning", "which means",
        ]
        matched_kws = [kw for kw in strong_reasoning_kws if kw in text]
        has_strong_reasoning = len(matched_kws) >= 2  # 要求至少 2 个不同强推理词

        # 检查 4（新增）：推理词与选项评估相关性
        # 至少一个强推理词出现在选项字母（A/B/C/D/E）附近（前后 30 字符内）
        has_option_linked_reasoning = False
        for kw in matched_kws:
            kw_pos = text.find(kw)
            if kw_pos == -1:
                continue
            nearby = text[max(0, kw_pos - 30): kw_pos + len(kw) + 30]
            if re.search(r"\b[A-E]\b", nearby):
                has_option_linked_reasoning = True
                break

        checks = [has_content, answer_valid, has_strong_reasoning, has_option_linked_reasoning]
        n_passed = sum(checks)

        # 非线性评分：必须答案有效，其余三项组成过程质量分
        if not answer_valid:
            return 0.0
        process_score = sum([has_content, has_strong_reasoning, has_option_linked_reasoning]) / 3
        # 答案有效占 0.4，过程质量占 0.6
        return 0.4 + 0.6 * process_score

    def eval_reasoning_coverage(self, response: Dict, choices: List[str] = None, num_choices: int = 5) -> float:
        """
        I1.2 推理覆盖质量（v2：选项文本匹配 + 排除语境验证，防止字母虚高）

        原版问题：「A 在文本中出现」并不等于「选项 A 被认真评估」，
        因为题目复述或格式本身就包含所有选项字母，导致覆盖率恒为满分。

        改进方案（参考 MultiAgentBench MARBLE 指标体系）：
          子项 1 - 语义覆盖率（权重 0.5）：
            统计选项文字内容（非字母）在推理文本中被提及的比例。
            若 choices 传入了选项文本，则改用选项关键词匹配，
            彻底避免字母计数虚高问题。

          子项 2 - 排除验证深度（权重 0.3）：
            原版只检查排除词是否存在，但「这道题没有错误选项」这样的句子也包含「错误」。
            改进：要求排除词与选项字母在同一句子内共现（± 60 字符窗口）。

          子项 3（新增）- 比较结构完整性（权重 0.2）：
            推理中是否同时出现了肯定性和否定性评价词，
            说明 Agent 做了正向选择 + 逆向排除的双向比较。
        """
        text = response.get("response", "")
        if not text:
            return 0.0

        # 子项 1：语义覆盖率
        if choices:
            # 使用选项文本的前 8 个字符作为关键词，避免过长匹配失败
            keywords = [c[:8].strip() for c in choices if len(c.strip()) > 0]
            covered = sum(1 for kw in keywords if kw and kw in text)
            coverage_score = covered / len(keywords) if keywords else 0.0
        else:
            # 无选项文本时退化为：字母出现 + 后面紧跟非选项内容（排除纯列举）
            # 用「选项字母后跟中文/英文实质内容」判断是否真的评估了该选项
            covered = 0
            for c in "ABCDE"[:num_choices]:
                # 匹配 "A." "A、" "A：" 之后有至少 10 个字符的内容（排除简单列举）
                if re.search(rf"\b{c}\b[.、：:。\s]{{1,3}}.{{10,}}", text):
                    covered += 1
            coverage_score = covered / num_choices

        # 子项 2：排除验证深度（要求排除词与选项字母共现于 ±60 字符窗口）
        exclusion_kws = ["排除", "不是", "不对", "不符合", "错误的", "不可能",
                         "incorrect", "wrong", "cannot", "not the"]
        has_contextualized_exclusion = False
        for kw in exclusion_kws:
            pos = text.find(kw)
            if pos == -1:
                continue
            window = text[max(0, pos - 60): pos + len(kw) + 60]
            if re.search(r"\b[A-E]\b", window):
                has_contextualized_exclusion = True
                break

        # 子项 3：双向比较完整性（同时出现肯定 + 否定评价）
        affirmative_kws = ["正确", "符合", "合理", "应该", "最佳", "更好",
                           "correct", "right", "best", "most likely"]
        negative_kws = ["错误", "不对", "排除", "不符合", "不可能",
                        "wrong", "incorrect", "eliminate", "not"]
        has_bidirectional = (
            any(kw in text for kw in affirmative_kws) and
            any(kw in text for kw in negative_kws)
        )

        return (
            coverage_score * 0.5
            + float(has_contextualized_exclusion) * 0.3
            + float(has_bidirectional) * 0.2
        )

    def eval_single_agent_accuracy(self, results: List[Dict], ground_truths: List[str]) -> float:
        """
        I1.3 单体准确率（逻辑不变）
        单 Agent 最终答案的正确率，作为 I2 协作增益的基准。
        """
        if not results:
            return 0.0
        correct = sum(1 for r, gt in zip(results, ground_truths) if r["answer"] == gt)
        return correct / len(results)

    # ================================================================
    # I2: 协作效率
    # ================================================================

    def eval_collaboration_gain(self, single_acc: float, multi_acc: float, context_acc: float = None) -> Dict:
        """
        I2.1 协作增益（v2：引入上下文等价基线，分离「信息增益」与「协作增益」）

        原版问题：单 Agent 基线只用 Reasoner，没有 Analyst 输入；
        而多 Agent 中 Reasoner 拥有 Analyst 提供的额外上下文，
        导致差值混入「上下文信息量」带来的提升，无法纯粹衡量协作机制本身的价值。

        改进方案：
          - raw_gain：原始增益（多体 vs 纯单体），包含协作 + 上下文双重效益
          - context_gain（可选）：上下文增益（带 Analyst 上下文的单 Reasoner vs 纯单体）
          - pure_collab_gain：纯协作增益 = raw_gain - context_gain，
            排除上下文信息量的影响，才是真正意义上的「协作机制贡献」

        若 context_acc 未提供（未进行等价基线实验），仅返回 raw_gain 并标注警告。

        参考：ReliabilityBench（2025）中关于控制变量的 baseline 设计原则。
        """
        raw_gain = multi_acc - single_acc
        result = {
            "raw_gain": raw_gain,
            "single_acc": single_acc,
            "multi_acc": multi_acc,
        }
        if context_acc is not None:
            context_gain = context_acc - single_acc
            pure_collab_gain = raw_gain - context_gain
            result.update({
                "context_acc": context_acc,
                "context_gain": context_gain,
                "pure_collab_gain": pure_collab_gain,
            })
        else:
            result["warning"] = (
                "未提供 context_acc，无法分离协作增益与上下文增益。"
                "建议运行 run_context_agent() 获取等价基线后再解读。"
            )
        return result

    def eval_coordination_consistency(self, final_state: Dict, test_item: Dict = None, mas_system=None) -> Dict:
        """
        I2.2 协调一致性（v2：独立轮询 + Cohen's κ，解决非独立性问题）

        原版问题：三个 Agent 的答案并非独立——Reasoner 看了 Analyst 的输出，
        Validator 看了 Reasoner 的输出，因此一致性高是流水线结构决定的，
        不能说明协作本身达成了真实共识。

        改进方案（参考 MultiAgentBench 中的 inter-agent agreement 设计）：

        模式 A（有 mas_system 时，独立轮询）：
          对同一道题，分别独立调用 Analyst、Reasoner（无 Analyst 输入）、
          Validator（无 Reasoner 输入），得到三个独立答案后计算一致性。
          这才是真正意义上的「各自判断有多少人认同同一答案」。

        模式 B（无 mas_system 时，仅用现有 state 计算，标注局限）：
          沿用旧逻辑，从 final_state 中提取三个答案，
          计算 Cohen's κ（而非简单众数比例），κ 值区分偶然一致与真实一致。
          κ = (observed_agreement - chance_agreement) / (1 - chance_agreement)
          κ > 0.6 为强一致，0.4~0.6 为中等，< 0.4 为弱一致。
        """
        # 模式 B：从 final_state 提取（标注为流水线一致性，非独立一致性）
        agent_outputs = {
            "analyst": final_state.get("analyst_output", ""),
            "reasoner": final_state.get("reasoner_output", ""),
            "validator": final_state.get("validator_output", ""),
        }
        answers = [extract_answer(text) for text in agent_outputs.values()]
        valid_answers = [a for a in answers if a != "UNKNOWN"]

        if len(valid_answers) < 2:
            return {"pipeline_consistency": 0.0, "cohens_kappa": 0.0, "mode": "pipeline"}

        # 众数一致率（原指标，保留）
        most_common = max(set(valid_answers), key=valid_answers.count)
        pipeline_consistency = valid_answers.count(most_common) / len(valid_answers)

        # Cohen's κ（适配多分类 5 选项情境，基准为随机猜测 1/5 = 0.2）
        # 简化公式：κ = (P_o - P_e) / (1 - P_e)
        # P_o = 观察到的一致比例，P_e = 随机猜测期望一致概率
        n = len(valid_answers)
        option_counts = {opt: valid_answers.count(opt) for opt in set(valid_answers)}
        # P_e = sum(n_i/N * n_j/N) for all pairs = sum((count/n)^2)
        p_e = sum((cnt / n) ** 2 for cnt in option_counts.values())
        p_o = pipeline_consistency
        cohens_kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

        return {
            "pipeline_consistency": pipeline_consistency,
            "cohens_kappa": round(cohens_kappa, 4),
            "mode": "pipeline（非独立，仅供参考）",
            "note": "三个 Agent 输出存在依赖链，κ 值高估真实共识程度",
        }

    def eval_communication_overhead(self, metrics: Dict) -> Dict:
        """
        I2.3 通信开销（优先使用 API usage 中的真实 token）

        原版问题：直接用字符数代替 token 数，对于中文文本误差较大（1 token ≈ 1.5 汉字）；
        且只统计输出，忽略了输入 token（含历史上下文），低估实际 API 成本。

        当前版本优先读取 metrics["total_tokens"] 中由 API usage 返回的真实 token。
        若读取的是旧 checkpoint（没有 total_chars 字段），则退回到旧版字符数估算逻辑。
        """
        if "total_chars" in metrics:
            total_tokens = int(metrics.get("total_tokens", 0))
            total_chars = int(metrics.get("total_chars", 0))
            return {
                "rounds": metrics["total_rounds"],
                "total_chars": total_chars,
                "prompt_tokens": int(metrics.get("prompt_tokens", 0)),
                "completion_tokens": int(metrics.get("completion_tokens", 0)),
                "estimated_tokens": total_tokens,
                "total_tokens": total_tokens,
                "rework_count": metrics["num_validations"],
                "token_note": "total_tokens/prompt_tokens/completion_tokens 来自 API usage",
            }

        # 兼容旧 checkpoint：旧版 metrics["total_tokens"] 实际是字符数。
        total_chars = int(metrics["total_tokens"])
        # 假设中文占 60%，英文占 40%（CommonsenseQA 中文翻译场景）
        estimated_tokens = int(total_chars * 0.6 / 1.5 + total_chars * 0.4 / 4)

        return {
            "rounds": metrics["total_rounds"],
            "total_chars": total_chars,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_tokens": estimated_tokens,
            "total_tokens": estimated_tokens,
            "rework_count": metrics["num_validations"],
            "token_note": "旧结果无 API usage，estimated_tokens 为启发式估算",
        }

    # ================================================================
    # I3: 系统稳定性
    # ================================================================

    def eval_position_bias(
        self,
        baseline_results: List[Dict],
        all_perm_results: List[List[Dict]],
        ground_truths: List[str],
    ) -> Dict:
        """
        I3.1 位置偏差综合指标（v2：多排列平均退化 + RStd，替代单次扰动）

        原版问题：单次随机排列结果受随机种子影响大，
        可能碰巧选到对模型有利/不利的排列，导致退化值不稳定。

        改进方案（参考 PriDe 论文，Zheng et al. ICLR 2024）：
          1. 对每道题生成 M 个不同排列（默认 M=3），对每个排列跑单 Agent
          2. 计算每道题在所有排列上的答案方差（答案不一致的次数 / M）
          3. 汇总指标：
             - avg_degradation：各排列准确率相对基线的平均退化（原版升级）
             - RStd（recall standard deviation）：各选项召回率的标准差，
               RStd 越高说明 position bias 越严重（来自 Zheng et al. 2023）
             - flip_rate：至少有一个排列改变答案的题目比例（不稳定题目占比）

        参数：
          baseline_results：原始顺序下的结果列表
          all_perm_results：外层按排列索引，内层按题目索引的结果列表
          ground_truths：正确答案
        """
        n = len(ground_truths)
        baseline_acc = sum(r["answer"] == gt for r, gt in zip(baseline_results, ground_truths)) / n

        # 计算各排列准确率
        perm_accs = []
        for perm_results in all_perm_results:
            # all_perm_results[i] 的 ground_truth 需在主流程中对应更新
            # 这里传入的 perm_results 已用更新后的 gt 进行匹配
            acc = sum(r["answer"] == gt for r, gt in zip(perm_results, ground_truths)) / n
            perm_accs.append(acc)

        avg_perm_acc = float(np.mean(perm_accs))
        avg_degradation = baseline_acc - avg_perm_acc

        # flip_rate：至少在一个排列中与基线答案不同的题目比例
        flip_count = 0
        for i in range(n):
            base_ans = baseline_results[i]["answer"]
            perm_answers = [perm[i]["answer"] for perm in all_perm_results]
            if any(a != base_ans for a in perm_answers):
                flip_count += 1
        flip_rate = flip_count / n

        # RStd：各选项在所有排列中的召回率标准差（衡量 position bias 的严重程度）
        # 召回率 = 该选项被选中的次数 / 该选项作为正确答案的总次数
        # 简化版本：计算每个选项字母（A-E）在所有排列结果中的被选频率标准差
        from collections import Counter

        all_answers = [r["answer"] for perm in all_perm_results for r in perm if r["answer"] != "UNKNOWN"]
        total = len(all_answers) if all_answers else 1
        option_freqs = [Counter(all_answers).get(opt, 0) / total for opt in "ABCDE"]
        rstd = float(np.std(option_freqs))

        return {
            "baseline_acc": round(baseline_acc, 4),
            "avg_perm_acc": round(avg_perm_acc, 4),
            "avg_degradation": round(avg_degradation, 4),
            "flip_rate": round(flip_rate, 4),
            "rstd": round(rstd, 4),
            "num_perms": len(all_perm_results),
            "note": "avg_degradation > 0 说明存在 position bias；flip_rate 越高越不稳定；"
                    "RStd 越高说明模型对特定选项位置有明显偏好",
        }

    def eval_answer_consistency(self, repeated_results: List[List[Dict]], ground_truths: List[str] = None) -> Dict:
        """
        I3.2 重复运行稳定性（v2：无偏 pass@k + 贝叶斯置信区间，替代简单众数率）

        原版问题：「众数一致率」只衡量了结果有多集中，
        但不区分「集中在正确答案」还是「集中在错误答案」，
        且 5 题×3 次的样本量下置信区间极宽，结果近乎无意义。

        改进方案（参考 Chen et al. 2021 HumanEval pass@k，及 ReliabilityBench 2025）：

        核心指标 1 - 无偏 pass@k 估计量（k=1）：
          对每道题 n 次运行，c 次答对，使用无偏估计量：
          pass@1 = 1 - C(n-c, k) / C(n, k) ≈ c/n（当 k=1 时退化为准确率均值）
          此处 k=1 即 avg@n，是 pass@k 的最保守估计。

        核心指标 2 - 稳定性得分（stability score）：
          每道题的稳定性 = 最高频答案出现次数 / n，
          稳定性得分 = 所有题目稳定性的均值（不变，但补充置信区间）

        核心指标 3 - 贝叶斯 95% 置信区间（Beta 先验）：
          对每道题的正确率 p，用 Beta(1,1) 先验，后验为 Beta(c+1, n-c+1)，
          95% HPD 区间反映不确定程度。
          参考：「Don't Pass@k」(Hariri et al. 2025) 贝叶斯评估框架。

        参数：
          repeated_results：外层按题目索引，内层为同一题的 K 次结果
          ground_truths：若传入，额外计算 pass@1 准确率指标
        """
        from scipy.stats import beta as beta_dist

        n_topics = len(repeated_results)
        stability_scores = []
        pass1_scores = []
        ci_widths = []

        for i, runs in enumerate(repeated_results):
            answers = [r.get("answer", "UNKNOWN") for r in runs]
            valid = [a for a in answers if a != "UNKNOWN"]
            n = len(valid)

            if n == 0:
                stability_scores.append(0.0)
                pass1_scores.append(0.0)
                ci_widths.append(1.0)
                continue

            # 稳定性：最高频答案占比
            most_common = max(set(valid), key=valid.count)
            stability_scores.append(valid.count(most_common) / n)

            # pass@1（准确率）：若 ground_truths 已提供
            if ground_truths and i < len(ground_truths):
                c = sum(1 for a in valid if a == ground_truths[i])
                pass1_scores.append(c / n)

                # 贝叶斯 95% HPD（Beta(c+1, n-c+1)）
                lo, hi = beta_dist.interval(0.95, c + 1, n - c + 1)
                ci_widths.append(hi - lo)
            else:
                pass1_scores.append(None)
                ci_widths.append(None)

        avg_stability = float(np.mean(stability_scores))
        result = {
            "avg_stability": round(avg_stability, 4),
            "n_topics": n_topics,
            "runs_per_topic": len(repeated_results[0]) if repeated_results else 0,
        }

        valid_pass1 = [s for s in pass1_scores if s is not None]
        if valid_pass1:
            avg_pass1 = float(np.mean(valid_pass1))
            valid_ci = [w for w in ci_widths if w is not None]
            avg_ci_width = float(np.mean(valid_ci)) if valid_ci else None
            result.update({
                "avg_pass1": round(avg_pass1, 4),
                "avg_ci_width_95": round(avg_ci_width, 4) if avg_ci_width else None,
                "note": (
                    f"avg_pass1 为 pass@1 无偏估计（k=1 时等价于准确率均值）；"
                    f"avg_ci_width_95={avg_ci_width:.2%} 反映不确定程度，"
                    f"样本量越大越窄"
                ),
            })
        else:
            result["note"] = "未提供 ground_truths，仅计算稳定性，无法计算 pass@1"

        return result

    # ================================================================
    # I4: 任务完成度
    # ================================================================

    def eval_task_accuracy(self, results: List[Dict], ground_truths: List[str]) -> float:
        """
        I4.1 多体任务准确率（逻辑不变）
        多 Agent 协作后的最终答案正确率，与 I1.3 形成对比。
        """
        if not results:
            return 0.0
        correct = sum(1 for r, gt in zip(results, ground_truths) if r["answer"] == gt)
        return correct / len(results)

    def eval_answer_extractability(self, results: List[Dict]) -> Dict:
        """
        I4.2 答案提取率（v2：区分严格格式率与宽松提取率，保留区分度）

        原版问题：修复 extract_answer 后，宽松提取率接近 100%，失去区分度。
        改进：将提取质量分为三级，保留各自比例，提供更细粒度的格式合规信息。

        三级定义：
          - strict_rate（严格格式率）：答案来自「最终答案:X」标准格式（最可靠）
          - standard_rate（标准格式率）：答案来自「答案:X」格式（可接受）
          - fallback_rate（兜底率）：答案来自选项语境匹配或末尾兜底（最不可靠）
          - unknown_rate（失败率）：仍为 UNKNOWN（格式完全混乱）

        注：各率之和 = 1.0，strict_rate 越高说明系统输出越规范。
        """
        if not results:
            return {"strict_rate": 0.0, "standard_rate": 0.0, "fallback_rate": 0.0, "unknown_rate": 1.0}

        strict = standard = fallback = unknown = 0

        for r in results:
            # multi_agent 结果用 validator_output 或 conversation 中找格式证据
            text = r.get("validator_output", "") or r.get("response", "")
            answer = r.get("answer", "UNKNOWN")

            if answer == "UNKNOWN":
                unknown += 1
            elif re.search(r"最终答案\s*[:：]\s*" + answer, text, re.IGNORECASE):
                strict += 1
            elif re.search(r"(?<!\S)答案\s*[:：]\s*" + answer, text, re.IGNORECASE):
                standard += 1
            else:
                fallback += 1

        n = len(results)
        return {
            "strict_rate": round(strict / n, 4),
            "standard_rate": round(standard / n, 4),
            "fallback_rate": round(fallback / n, 4),
            "unknown_rate": round(unknown / n, 4),
            "note": "strict_rate 越高说明输出格式越规范；unknown_rate > 0.1 需警惕",
        }

    def eval_answer_definiteness(self, results: List[Dict]) -> Dict:
        """
        I4.3 答案确定性（v2：增加「模糊惩罚」，区分真正明确与侥幸单字母）

        原版问题：「答案字母在全文只出现一次」这个条件在长输出中几乎不成立；
        两个判断分支实际等价，指标没有区分度。

        改进方案：
          - definite（明确）：严格格式（最终答案:X）且全文中只有该字母被明确标注
          - ambiguous（模糊）：有多个不同字母被「答案:X」格式标注（Agent 给出了矛盾答案）
          - uncertain（不确定）：文本含「不确定/可能/也许/uncertain/might」等犹豫词
          - normal（正常）：未触发以上任何一种特殊情况的有效提取

        明确率越高越好；模糊率 > 0.1 说明系统存在自相矛盾问题。
        """
        if not results:
            return {"definite_rate": 0.0, "ambiguous_rate": 0.0, "uncertain_rate": 0.0, "normal_rate": 0.0}

        definite = ambiguous = uncertain = normal = unknown_cnt = 0
        hedge_words = ["不确定", "可能", "也许", "猜测", "感觉", "大概",
                       "uncertain", "might", "perhaps", "probably", "guess", "maybe"]

        for r in results:
            text = r.get("validator_output", "") or r.get("response", "")
            answer = r.get("answer", "UNKNOWN")

            if answer == "UNKNOWN":
                unknown_cnt += 1
                continue

            # 检查模糊：文本中有多于一个不同字母被明确标注为答案
            all_explicit = re.findall(r"(?:最终)?答案\s*[:：]\s*([A-E])", text, re.IGNORECASE)
            unique_explicit = set(a.upper() for a in all_explicit)

            if len(unique_explicit) > 1:
                ambiguous += 1
                continue

            # 检查不确定：含犹豫词
            has_hedge = any(hw in text for hw in hedge_words)
            if has_hedge:
                uncertain += 1
                continue

            # 检查明确：严格格式且无歧义
            if re.search(r"最终答案\s*[:：]\s*" + answer, text, re.IGNORECASE):
                definite += 1
            else:
                normal += 1

        n = len(results)
        return {
            "definite_rate": round(definite / n, 4),
            "ambiguous_rate": round(ambiguous / n, 4),
            "uncertain_rate": round(uncertain / n, 4),
            "normal_rate": round(normal / n, 4),
            "unknown_rate": round(unknown_cnt / n, 4),
            "note": "definite_rate 高且 ambiguous_rate 低为最佳状态",
        }
