import json
from typing import Dict, List


def load_commonsenseqa_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """
    加载CommonsenseQA数据集

    Args:
        file_path: JSONL文件路径
        max_samples: 最大加载样本数，None表示加载全部

    Returns:
        包含问题、选项和答案的字典列表
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                item = json.loads(line.strip())

                # 提取问题文本
                question = item.get("question", {}).get("stem", "")

                # 提取选项
                choices_raw = item.get("question", {}).get("choices", [])
                choices = [choice.get("text", "") for choice in choices_raw]

                # 提取正确答案标签
                answer_key = item.get("answerKey", "")

                data.append({
                    "question": question,
                    "choices": choices,
                    "answer": answer_key,
                })

        print(f"成功加载 {len(data)} 条数据从 {file_path}")
        return data

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        print("请确保已将数据集下载到正确的位置")
        return []
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return []
