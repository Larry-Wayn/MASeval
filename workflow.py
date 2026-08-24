from typing import Dict, List

from langgraph.graph import END, StateGraph

from config import SYSTEM_CONFIG
from llm_client import call_llm_with_usage
from schemas import AgentState
from utils import extract_answer


def _add_usage(state: AgentState, usage: Dict) -> Dict[str, int]:
    """将本次 API usage 累加到 LangGraph 状态中。"""
    return {
        "prompt_tokens": state.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0),
        "completion_tokens": state.get("completion_tokens", 0) + usage.get("completion_tokens", 0),
        "total_tokens": state.get("total_tokens", 0) + usage.get("total_tokens", 0),
    }


def analyst_node(state: AgentState) -> AgentState:
    """分析者智能体节点"""
    system_message = """你是问题分析专家。你的职责:
1. 仔细阅读问题，识别关键信息
2. 分解问题的推理路径
3. 指出需要的常识知识类型
4. 给出你的初步倾向答案（格式：初步倾向:X）
请用简洁的语言输出你的分析。"""

    question_text = state["question"]
    choices = state["choices"]
    choices_str = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])

    user_message = f"""问题: {question_text}

选项:
{choices_str}

请分析这个问题。"""

    llm_result = call_llm_with_usage(system_message, user_message)
    response = llm_result["content"]

    return {
        "analyst_output": response,
        "messages": [{"role": "Analyst", "content": response}],
        **_add_usage(state, llm_result["usage"]),
        # "round_count": state.get("round_count", 0) + 1,
    }


def reasoner_node(state: AgentState) -> AgentState:
    """推理者智能体节点"""
    system_message = """你是逻辑推理专家。你的职责:
1. 基于Analyst的分析进行推理
2. 逐一评估每个选项的合理性
3. 给出答案及置信度
格式: 推理过程 → 答案:X (置信度:Y%)"""

    analyst_output = state.get("analyst_output", "")
    question_text = state["question"]
    choices = state["choices"]
    choices_str = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])

    # 如果需要修正,包含验证者的反馈
    validator_feedback = ""
    if state.get("needs_revision") and state.get("validator_output"):
        validator_feedback = f"\n\n验证者反馈:\n{state['validator_output']}\n\n请根据反馈修正你的推理。"

    user_message = f"""问题: {question_text}

选项:
{choices_str}

分析者的分析:
{analyst_output}
{validator_feedback}

请进行推理并给出答案。"""

    llm_result = call_llm_with_usage(system_message, user_message)
    response = llm_result["content"]

    return {
        "reasoner_output": response,
        "messages": [{"role": "Reasoner", "content": response}],
        **_add_usage(state, llm_result["usage"]),
        # "round_count": state.get("round_count", 0) + 1,
    }


def validator_node(state: AgentState) -> AgentState:
    """验证者智能体节点"""
    import re

    system_message = """你是逻辑验证专家。你的职责:
1. 检查Reasoner的推理是否有逻辑漏洞
2. 确认答案是否符合常识
3. 必须以以下两种格式之一结尾（这是硬性要求，不可省略）：
    - 若推理正确：最终答案:X   （X为A/B/C/D/E中的一个字母，不加其他字符）
    - 若推理有误：需要修正:（说明原因）
注意：不要在"需要修正"之外的语境使用"需要修正"这个短语。
注意：若推理正确，最后一行必须严格写成"最终答案:X"格式，例如"最终答案:B"。"""

    reasoner_output = state.get("reasoner_output", "")
    question_text = state["question"]
    choices = state["choices"]
    choices_str = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])

    user_message = f"""问题: {question_text}

选项:
{choices_str}

推理者的推理:
{reasoner_output}

请验证推理的正确性。"""

    llm_result = call_llm_with_usage(system_message, user_message)
    response = llm_result["content"]

    # 只匹配明确要求修正的语句
    needs_revision = bool(re.search(r"需要修正|请.*重新.*推理|推理.*有误.*请.*修正", response))

    # 提取答案：优先从 validator 的输出提取，失败时从 reasoner 的输出中兜底
    answer = extract_answer(response)
    if answer == "UNKNOWN" and not needs_revision:
        # Validator 认可推理但未明确写出最终答案时，直接沿用 Reasoner 的答案
        answer = extract_answer(reasoner_output)

    return {
        "validator_output": response,
        "messages": [{"role": "Validator", "content": response}],
        "needs_revision": needs_revision,
        "final_answer": answer if not needs_revision else state.get("final_answer", "UNKNOWN"),
        "round_count": state.get("round_count", 0) + 1,
        "revision_count": state.get("revision_count", 0) + (1 if needs_revision else 0),
        **_add_usage(state, llm_result["usage"]),
    }


def should_continue(state: AgentState) -> str:
    """决定是否继续迭代或结束"""
    # 检查是否达到最大轮次
    if state.get("round_count", 0) >= SYSTEM_CONFIG["max_rounds"]:
        return "end"

    # 检查是否需要修正且未超过修正次数
    if state.get("needs_revision", False) and state.get("revision_count", 0) < 2:
        return "revise"

    # 如果有最终答案且不需要修正,结束
    if state.get("final_answer") and state["final_answer"] != "UNKNOWN" and not state.get("needs_revision", False):
        return "end"

    return "end"


class MASQuestionAnswering:
    def __init__(self, config=SYSTEM_CONFIG):
        self.config = config
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建LangGraph工作流"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("reasoner", reasoner_node)
        workflow.add_node("validator", validator_node)

        # 设置入口点
        workflow.set_entry_point("analyst")

        # 添加边
        workflow.add_edge("analyst", "reasoner")
        workflow.add_edge("reasoner", "validator")

        # 添加条件边
        workflow.add_conditional_edges(
            "validator",
            should_continue,
            {
                "revise": "reasoner",  # 需要修正,回到推理者
                "end": END,
            },
        )

        return workflow.compile()

    def run_single_agent(self, question: str, choices: List[str]) -> Dict:
        """单智能体基线测试 (仅Reasoner)"""
        system_message = """你是逻辑推理专家。你的职责:
1. 仔细分析问题
2. 逐一评估每个选项的合理性
3. 给出答案及置信度
格式: 推理过程 → 答案:X (置信度:Y%)"""

        choices_str = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
        user_message = f"""问题: {question}

选项:
{choices_str}

请直接给出答案。"""

        llm_result = call_llm_with_usage(system_message, user_message)
        response = llm_result["content"]
        answer = extract_answer(response)

        return {
            "response": response,
            "answer": answer,
            "usage": llm_result["usage"],
        }

    def run_multi_agent(self, question: str, choices: List[str]) -> Dict:
        """多智能体协作测试"""
        # 初始化状态
        initial_state = {
            "question": question,
            "choices": choices,
            "messages": [],
            "analyst_output": "",
            "reasoner_output": "",
            "validator_output": "",
            "final_answer": "UNKNOWN",
            "round_count": 0,
            "needs_revision": False,
            "revision_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # 运行图
        final_state = self.graph.invoke(initial_state)

        # 如果没有最终答案,依次从 validator、reasoner、analyst 的输出中兜底提取
        if final_state["final_answer"] == "UNKNOWN":
            for key in ("validator_output", "reasoner_output", "analyst_output"):
                if final_state.get(key):
                    fallback = extract_answer(final_state[key])
                    if fallback != "UNKNOWN":
                        final_state["final_answer"] = fallback
                        break

        return {
            "conversation": final_state["messages"],
            "answer": final_state["final_answer"],
            "metrics": {
                "total_rounds": final_state["round_count"],
                "prompt_tokens": final_state.get("prompt_tokens", 0),
                "completion_tokens": final_state.get("completion_tokens", 0),
                "total_tokens": final_state.get("total_tokens", 0),
                "total_chars": sum(len(msg["content"]) for msg in final_state["messages"]),
                "num_validations": final_state.get("revision_count", 0),
            },
            "final_state": final_state,
        }


def run_context_agent(mas_system, question: str, choices: List[str], analyst_output: str) -> Dict:
    """
    I2.1 辅助函数：上下文等价基线。
    给单 Agent（Reasoner）注入 Analyst 上下文，使信息量与多 Agent 中的 Reasoner 等价，
    从而分离「协作机制」与「上下文信息增量」对增益的贡献。
    """
    system_message = """你是逻辑推理专家。你的职责:
1. 仔细分析问题
2. 逐一评估每个选项的合理性
3. 给出答案及置信度
格式: 推理过程 → 答案:X (置信度:Y%)"""

    choices_str = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
    user_message = f"""问题: {question}

选项:
{choices_str}

参考分析:
{analyst_output}

请基于以上分析直接给出答案。"""

    llm_result = call_llm_with_usage(system_message, user_message)
    response = llm_result["content"]
    answer = extract_answer(response)
    return {"response": response, "answer": answer, "usage": llm_result["usage"]}
