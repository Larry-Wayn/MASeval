import re
from typing import Dict, List


def extract_answer(text: str) -> str:
    """
    从文本中提取答案，按优先级依次尝试以下模式：
    1. 明确的最终答案标记（最高优先级，最可靠）
    2. 普通答案标记
    3. 选项字母后跟标点/空格（如 "A." "A、" "A："）
    4. 退化兜底：文本末尾最后出现的孤立字母（最不可靠，慎用）
    """
    if not text:
        return "UNKNOWN"

    # 优先级1：最终答案:X 或 最终答案：X（Validator 的标准输出格式）
    match = re.search(r"最终答案\s*[:：]\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 优先级2：答案:X 或 答案：X（Reasoner 的标准输出格式）
    match = re.search(r"(?<![^\s\(（])答案\s*[:：]\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 优先级3：选项字母后紧跟标点，如 "选A。" "答案是A。" "应该是A"
    match = re.search(r"(?:选|是|为|答案是|应该是|应为)\s*([A-E])\s*[。.）\)）\s]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 优先级4：括号内的单个字母，如 (A) 或 （A）
    match = re.search(r"[（\(]\s*([A-E])\s*[）\)]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 优先级5（兜底）：文本最后出现的孤立大写字母（仅在前面都失败时才用）
    # 限制在末尾50字符内搜索，避免误取推理过程中提到的选项字母
    tail = text[-50:] if len(text) > 50 else text
    match = re.search(r"\b([A-E])\b", tail[::-1])  # 从尾部反向搜索
    if match:
        return match.group(1).upper()

    return "UNKNOWN"


def format_question(question: str, choices: List[str]) -> str:
    """格式化问题"""
    choices_str = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
    return f"""问题: {question}

选项:
{choices_str}

请通过协作讨论,给出最合理的答案(A/B/C/D/E)。"""


def shuffle_choices(item: Dict, seed: int = None) -> Dict:
    """
    I3.1 扰动辅助函数：打乱选项顺序并更新 ground truth 标签。
    用于测试模型对选项位置偏差（position bias）的敏感程度。

    Args:
        item: 包含 "question", "choices", "answer" 的数据条目
        seed: 随机种子（用于复现）
    Returns:
        新的数据条目，choices 顺序已打乱，answer 标签已同步更新
    """
    import random

    rng = random.Random(seed)

    original_choices = item["choices"]
    original_answer_key = item["answer"]  # "A", "B", "C" 等
    original_answer_idx = ord(original_answer_key) - ord("A")
    original_answer_text = original_choices[original_answer_idx]

    shuffled_choices = original_choices[:]
    rng.shuffle(shuffled_choices)

    new_answer_idx = shuffled_choices.index(original_answer_text)
    new_answer_key = chr(ord("A") + new_answer_idx)

    return {
        "question": item["question"],
        "choices": shuffled_choices,
        "answer": new_answer_key,
    }
