import operator
from typing import Annotated, Dict, List, TypedDict


class AgentState(TypedDict):
    """多智能体系统的状态"""

    question: str
    choices: List[str]
    messages: Annotated[List[Dict], operator.add]
    analyst_output: str
    reasoner_output: str
    validator_output: str
    final_answer: str
    round_count: int
    needs_revision: bool
    revision_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
