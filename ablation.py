"""
对比实验与消融实验：可配置的多智能体系统 (Configurable MAS)

通过三个开关控制系统结构：
    enable_analyst   是否启用 Analyst 节点
    enable_validator 是否启用 Validator 节点
    enable_revision  是否启用「需要修正 -> 回到 Reasoner」反馈循环

复用 workflow.py 中已定义的 analyst_node / reasoner_node / validator_node，
当某节点被关闭时，使用对应的「空操作节点」占位，保持 LangGraph 图结构一致。
"""

from dataclasses import asdict, dataclass
from typing import Dict, List

from langgraph.graph import END, StateGraph

from schemas import AgentState
from utils import extract_answer
from workflow import analyst_node, reasoner_node, validator_node


@dataclass
class MASConfig:
    """多智能体系统消融配置"""

    name: str
    enable_analyst: bool = True
    enable_validator: bool = True
    enable_revision: bool = True
    max_revisions: int = 2


def _noop_analyst(state: AgentState) -> Dict:
    """Analyst 关闭时的占位节点：不调用 LLM，输出空上下文"""
    return {
        "analyst_output": "",
        "messages": [{"role": "Analyst", "content": "[disabled]"}],
    }


def _noop_validator(state: AgentState) -> Dict:
    """Validator 关闭时的占位节点：不调用 LLM，直接以 Reasoner 输出抽取最终答案"""
    reasoner_output = state.get("reasoner_output", "")
    return {
        "validator_output": "[disabled]",
        "messages": [{"role": "Validator", "content": "[disabled]"}],
        "needs_revision": False,
        "final_answer": extract_answer(reasoner_output),
        "round_count": state.get("round_count", 0) + 1,
        "revision_count": state.get("revision_count", 0),
    }


def build_configurable_graph(cfg: MASConfig):
    """根据消融配置构建 LangGraph 工作流"""
    workflow = StateGraph(AgentState)

    workflow.add_node(
        "analyst", analyst_node if cfg.enable_analyst else _noop_analyst
    )
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node(
        "validator", validator_node if cfg.enable_validator else _noop_validator
    )

    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "reasoner")
    workflow.add_edge("reasoner", "validator")

    def _should_continue(state: AgentState) -> str:
        if not cfg.enable_validator or not cfg.enable_revision:
            return "end"
        if state.get("revision_count", 0) >= cfg.max_revisions:
            return "end"
        if state.get("needs_revision", False):
            return "revise"
        return "end"

    workflow.add_conditional_edges(
        "validator",
        _should_continue,
        {"revise": "reasoner", "end": END},
    )

    return workflow.compile()


def run_with_config(cfg: MASConfig, question: str, choices: List[str]) -> Dict:
    """根据消融配置运行一次问答任务"""
    graph = build_configurable_graph(cfg)

    initial_state: AgentState = {
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

    final_state = graph.invoke(initial_state)

    if final_state["final_answer"] == "UNKNOWN":
        for key in ("validator_output", "reasoner_output", "analyst_output"):
            text = final_state.get(key) or ""
            if not text or text == "[disabled]":
                continue
            fb = extract_answer(text)
            if fb != "UNKNOWN":
                final_state["final_answer"] = fb
                break

    real_messages = [
        m for m in final_state["messages"] if m.get("content") != "[disabled]"
    ]
    total_chars = sum(len(m["content"]) for m in real_messages)

    return {
        "config": asdict(cfg),
        "answer": final_state["final_answer"],
        "final_state": final_state,
        "metrics": {
            "total_rounds": final_state.get("round_count", 0),
            "prompt_tokens": final_state.get("prompt_tokens", 0),
            "completion_tokens": final_state.get("completion_tokens", 0),
            "total_tokens": final_state.get("total_tokens", 0),
            "total_chars": total_chars,
            "num_validations": final_state.get("revision_count", 0),
        },
    }


DEFAULT_CONFIGS: List[MASConfig] = [
    MASConfig(
        name="S1_SingleReasoner",
        enable_analyst=False,
        enable_validator=False,
        enable_revision=False,
    ),
    MASConfig(
        name="S2_Reasoner+Analyst",
        enable_analyst=True,
        enable_validator=False,
        enable_revision=False,
    ),
    MASConfig(
        name="S3_Reasoner+Validator",
        enable_analyst=False,
        enable_validator=True,
        enable_revision=False,
    ),
    MASConfig(
        name="S4_Reasoner+Validator+Revise",
        enable_analyst=False,
        enable_validator=True,
        enable_revision=True,
    ),
    MASConfig(
        name="S5_FullMAS_NoRevise",
        enable_analyst=True,
        enable_validator=True,
        enable_revision=False,
    ),
    MASConfig(
        name="S6_FullMAS",
        enable_analyst=True,
        enable_validator=True,
        enable_revision=True,
    ),
]
