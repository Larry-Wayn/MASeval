"""DeepSeek API 客户端

设计原则：
  1. **失败抛异常**：超时、429 限流、5xx 等可重试错误必须 raise，让上层
     concurrent_runner 的 retry 机制接管。
     不要"吞掉异常返回错误字符串"——那样会让错误样本被当作正常完成写入断点，
     S1/S2 这种没有 validator 兜底的配置会被严重污染。
  2. **超时配置化**：默认 120s（V4-Flash 偶尔会冷启动到 30+ s）。
  3. **429 限流尊重 Retry-After**：DeepSeek 在并发过高时返回 429，
     必须 sleep 服务端要求的秒数再 raise，让 runner 的指数退避之上再叠一层。
"""

from __future__ import annotations

import time
from typing import Dict

import requests

from config import LLM_CONFIG


# 可被环境变量覆盖的默认值
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 2048


class LLMTimeoutError(Exception):
    """请求超时；可重试"""


class LLMRateLimitError(Exception):
    """429 限流；强制 sleep 后再重试"""


class LLMServerError(Exception):
    """5xx 服务端错误；可重试"""


class LLMClientError(Exception):
    """4xx 客户端错误（401/400 等）；不可重试"""


def _normalize_usage(raw_usage: Dict | None) -> Dict[str, int]:
    """把 API 返回的 usage 统一成项目内部使用的三个字段。"""
    raw_usage = raw_usage or {}
    return {
        "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
        "total_tokens": int(raw_usage.get("total_tokens") or 0),
    }


def call_llm(
    system_message: str,
    user_message: str,
    config: Dict = LLM_CONFIG,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """统一的 LLM 调用接口（DeepSeek 兼容 OpenAI ChatCompletions）

    Returns:
        模型返回的 content 文本

    Raises:
        LLMTimeoutError: 网络超时
        LLMRateLimitError: 429 限流（已自动 sleep Retry-After）
        LLMServerError: 5xx
        LLMClientError: 4xx（401/400 等通常不可重试）
    """
    return call_llm_with_usage(
        system_message,
        user_message,
        config=config,
        timeout=timeout,
    )["content"]


def call_llm_with_usage(
    system_message: str,
    user_message: str,
    config: Dict = LLM_CONFIG,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict:
    """统一的 LLM 调用接口，并返回 API 真实 token usage。"""
    headers = {
        "Authorization": f"Bearer {config.get('api_key')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.get("model", "deepseek-v4-flash"),
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stream": False,
    }
    url = f"{config.get('base_url', 'https://api.deepseek.com/v1')}/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout as e:
        raise LLMTimeoutError(f"请求超时（>{timeout}s）") from e
    except requests.exceptions.ConnectionError as e:
        # 网络抖动也按超时处理（可重试）
        raise LLMTimeoutError(f"连接错误: {e}") from e

    status = response.status_code
    if status == 200:
        try:
            body = response.json()
            return {
                "content": body["choices"][0]["message"]["content"],
                "usage": _normalize_usage(body.get("usage")),
            }
        except (KeyError, ValueError) as e:
            raise LLMServerError(f"响应格式异常: {e}; body={response.text[:200]}") from e

    if status == 429:
        # 尊重服务端的 Retry-After（秒），缺省按 5s 等
        retry_after_raw = response.headers.get("Retry-After", "5")
        try:
            retry_after = float(retry_after_raw)
        except ValueError:
            retry_after = 5.0
        time.sleep(min(retry_after, 30))  # 上限 30s 避免卡太久
        raise LLMRateLimitError(
            f"429 限流，已 sleep {retry_after}s 后重抛供 runner 重试"
        )

    if 500 <= status < 600:
        raise LLMServerError(f"{status}: {response.text[:200]}")

    # 4xx：401/400 通常是配置错误，不可重试
    raise LLMClientError(f"{status}: {response.text[:200]}")
