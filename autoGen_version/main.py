import autogen
import json
import numpy as np
from typing import List, Dict
from collections import defaultdict

# ============ 配置部分 ============
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key-here"
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

# ============ 系统配置 ============
SYSTEM_CONFIG = {
    "num_agents": 3,
    "max_rounds": 5,
    "enable_validation": True,
}


# ============ 智能体定义 ============

class MASQuestionAnswering:
    def __init__(self, config=SYSTEM_CONFIG):
        self.config = config
        self.setup_agents()

    def setup_agents(self):
        """初始化三个核心智能体"""

        # Agent 1: Analyst (分析者)
        self.analyst = autogen.AssistantAgent(
            name="Analyst",
            system_message="""你是问题分析专家。你的职责:
1. 仔细阅读问题,识别关键信息
2. 分解问题的推理路径
3. 指出需要的常识知识类型
请用简洁的语言输出你的分析。""",
            llm_config=llm_config,
        )

        # Agent 2: Reasoner (推理者)
        self.reasoner = autogen.AssistantAgent(
            name="Reasoner",
            system_message="""你是逻辑推理专家。你的职责:
1. 基于Analyst的分析进行推理
2. 逐一评估每个选项的合理性
3. 给出答案及置信度
格式: 推理过程 → 答案:X (置信度:Y%)""",
            llm_config=llm_config,
        )

        # Agent 3: Validator (验证者)
        self.validator = autogen.AssistantAgent(
            name="Validator",
            system_message="""你是逻辑验证专家。你的职责:
1. 检查Reasoner的推理是否有逻辑漏洞
2. 确认答案是否符合常识
3. 如有问题,要求Reasoner修正;如无问题,确认最终答案
格式: 验证意见 → 最终答案:X 或 需要修正:原因""",
            llm_config=llm_config,
        )

        # 用户代理(启动对话)
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

    def run_single_agent(self, question: str, choices: List[str]) -> Dict:
        """单智能体基线测试 (仅Reasoner)"""
        prompt = self._format_question(question, choices)

        self.user_proxy.initiate_chat(
            self.reasoner,
            message=prompt,
            max_turns=1,
        )

        response = self.user_proxy.last_message()["content"]
        return {
            "response": response,
            "answer": self._extract_answer(response),
        }

    def run_multi_agent(self, question: str, choices: List[str]) -> Dict:
        """多智能体协作测试"""
        prompt = self._format_question(question, choices)

        # 设置GroupChat
        groupchat = autogen.GroupChat(
            agents=[self.analyst, self.reasoner, self.validator, self.user_proxy],
            messages=[],
            max_round=self.config["max_rounds"],
            speaker_selection_method="round_robin",
        )

        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        self.user_proxy.initiate_chat(
            manager,
            message=prompt,
        )

        # 收集对话历史
        conversation = groupchat.messages
        final_answer = self._extract_final_answer(conversation)

        return {
            "conversation": conversation,
            "answer": final_answer,
            "metrics": self._compute_metrics(conversation),
        }

    def _format_question(self, question: str, choices: List[str]) -> str:
        """格式化问题"""
        choices_str = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
        return f"""问题: {question}

选项:
{choices_str}

请通过协作讨论,给出最合理的答案(A/B/C/D/E)。"""

    def _extract_answer(self, text: str) -> str:
        """从文本中提取答案"""
        import re
        match = re.search(r'答案[::]?\s*([A-E])', text, re.IGNORECASE)
        return match.group(1).upper() if match else "UNKNOWN"

    def _extract_final_answer(self, conversation: List[Dict]) -> str:
        """从对话历史中提取最终答案"""
        # 从后往前找Validator的最终确认
        for msg in reversed(conversation):
            if msg.get("name") == "Validator":
                return self._extract_answer(msg["content"])
        return "UNKNOWN"

    def _compute_metrics(self, conversation: List[Dict]) -> Dict:
        """计算通信开销等指标"""
        return {
            "total_rounds": len(conversation),
            "total_tokens": sum(len(msg["content"]) for msg in conversation),
            "num_validations": sum(1 for msg in conversation if "需要修正" in msg["content"]),
        }


# ============ 评估器 ============

class MASEvaluator:
    def __init__(self):
        self.results = defaultdict(list)

    # ---- I1: 个体智能水平 ----
    def eval_planning_feasibility(self, response: Dict) -> float:
        """I1.1 规划可行性"""
        checks = [
            len(response["response"]) > 50,  # 有实质内容
            response["answer"] in "ABCDE",  # 答案有效
            "推理" in response["response"] or "because" in response["response"].lower(),
        ]
        return sum(checks) / len(checks)

    def eval_planning_quality(self, response: Dict, use_llm_judge=False) -> float:
        """I1.2 规划质量 (简化版)"""
        # 定量指标
        reasoning_steps = response["response"].count("。") + response["response"].count(".")
        has_comparison = sum(1 for c in "ABCDE" if c in response["response"])

        quantitative = min(reasoning_steps / 5.0, 1.0) * 0.5 + min(has_comparison / 3.0, 1.0) * 0.5

        # 可选: LLM评分 (需要额外API调用)
        if use_llm_judge:
            # qualitative = self._llm_judge(response["response"])
            # return 0.5 * quantitative + 0.5 * qualitative
            pass

        return quantitative

    def eval_single_agent_success(self, results: List[Dict], ground_truths: List[str]) -> float:
        """I1.3 单体成功率"""
        correct = sum(1 for r, gt in zip(results, ground_truths) if r["answer"] == gt)
        return correct / len(results)

    # ---- I2: 协作效率 ----
    def eval_collaboration_gain(self, single_acc: float, multi_acc: float) -> float:
        """I2.1 协作增益"""
        return multi_acc - single_acc

    def eval_coordination_consistency(self, conversation: List[Dict]) -> float:
        """I2.2 协调一致性"""
        conflicts = sum(1 for msg in conversation if "矛盾" in msg["content"] or "错误" in msg["content"])
        total_exchanges = len([m for m in conversation if m.get("name") != "User"])
        return 1 - (conflicts / max(total_exchanges, 1))

    def eval_communication_overhead(self, metrics: Dict) -> Dict:
        """I2.3 通信开销"""
        return {
            "rounds": metrics["total_rounds"],
            "tokens": metrics["total_tokens"],
            "rework": metrics["num_validations"],
        }

    # ---- I3: 系统稳定性 ----
    def eval_performance_degradation(self, baseline_acc: float, perturbed_acc: float) -> float:
        """I3.1 性能退化幅度"""
        return baseline_acc - perturbed_acc

    def eval_outcome_variance(self, results: List[List[Dict]]) -> float:
        """I3.3 结果波动性 (同一问题多次运行)"""
        variances = []
        for question_results in results:
            answers = [r["answer"] for r in question_results]
            # 计算答案的一致性
            most_common = max(set(answers), key=answers.count)
            consistency = answers.count(most_common) / len(answers)
            variances.append(1 - consistency)  # 波动性 = 1 - 一致性
        return np.mean(variances)

    # ---- I4: 任务完成度 ----
    def eval_task_success_rate(self, results: List[Dict], ground_truths: List[str]) -> float:
        """I4.1 任务成功率"""
        correct = sum(1 for r, gt in zip(results, ground_truths) if r["answer"] == gt)
        return correct / len(results)

    def eval_constraint_satisfaction(self, results: List[Dict]) -> float:
        """I4.2 约束满足率"""
        valid = sum(1 for r in results if r["answer"] in "ABCDE" and r["answer"] != "UNKNOWN")
        return valid / len(results)

    def eval_solution_completeness(self, response: Dict) -> float:
        """I4.3 方案完整性"""
        checks = [
            "问题" in response["response"] or "Question" in response["response"],
            sum(1 for c in "ABCDE" if c in response["response"]) >= 2,  # 考虑了多个选项
            response["answer"] != "UNKNOWN",
        ]
        return sum(checks) / len(checks)


# ============ 主实验流程 ============

def run_experiments():
    """运行完整评估实验"""

    # 1. 加载CommonsenseQA数据 (示例)
    test_data = [
        {
            "question": "Where would you find a jellyfish that has not been captured?",
            "choices": ["sea water", "store", "tank", "movie", "internet"],
            "answer": "A"
        },
        # ... 添加更多测试数据
    ]

    mas_system = MASQuestionAnswering()
    evaluator = MASEvaluator()

    # 2. I1基线测试: 单智能体
    print("=== I1: 单智能体基线测试 ===")
    single_results = []
    for item in test_data[:10]:  # 测试前10题
        result = mas_system.run_single_agent(item["question"], item["choices"])
        single_results.append(result)

    single_acc = evaluator.eval_single_agent_success(
        single_results,
        [item["answer"] for item in test_data[:10]]
    )
    print(f"单体成功率 (I1.3): {single_acc:.2%}")

    # 3. 多智能体协作测试
    print("\n=== I2: 多智能体协作测试 ===")
    multi_results = []
    for item in test_data[:10]:
        result = mas_system.run_multi_agent(item["question"], item["choices"])
        multi_results.append(result)

    multi_acc = evaluator.eval_task_success_rate(
        multi_results,
        [item["answer"] for item in test_data[:10]]
    )
    print(f"多智能体成功率 (I4.1): {multi_acc:.2%}")

    collab_gain = evaluator.eval_collaboration_gain(single_acc, multi_acc)
    print(f"协作增益 (I2.1): {collab_gain:+.2%}")

    # 4. I3稳定性测试: 扰动实验
    print("\n=== I3: 稳定性测试 ===")
    perturbed_config = SYSTEM_CONFIG.copy()
    perturbed_config["max_rounds"] = 3  # 限制讨论轮次

    mas_perturbed = MASQuestionAnswering(perturbed_config)
    perturbed_results = []
    for item in test_data[:10]:
        result = mas_perturbed.run_multi_agent(item["question"], item["choices"])
        perturbed_results.append(result)

    perturbed_acc = evaluator.eval_task_success_rate(
        perturbed_results,
        [item["answer"] for item in test_data[:10]]
    )
    degradation = evaluator.eval_performance_degradation(multi_acc, perturbed_acc)
    print(f"性能退化幅度 (I3.1): {degradation:.2%}")

    # 5. 输出综合报告
    print("\n=== 综合评估报告 ===")
    print(f"""
个体智能水平 (I1):
  - 单体成功率: {single_acc:.2%}

协作效率 (I2):
  - 协作增益: {collab_gain:+.2%}
  - 平均通信轮次: {np.mean([r["metrics"]["total_rounds"] for r in multi_results]):.1f}

系统稳定性 (I3):
  - 性能退化: {degradation:.2%}

任务完成度 (I4):
  - 多智能体成功率: {multi_acc:.2%}
  - 约束满足率: {evaluator.eval_constraint_satisfaction(multi_results):.2%}
    """)


if __name__ == "__main__":
    # 注意: 需要先配置OpenAI API Key
    # run_experiments()
    print("系统框架已构建,请配置API Key后运行 run_experiments()")