"""通用并发执行器

为长时间 LLM 实验提供：
  1. 线程池并发（DeepSeek API 是 IO bound，线程池足够）
  2. JSONL 断点续跑：每条任务完成后立即落盘，崩溃后下次启动自动跳过已完成
  3. 失败重试 + 指数退避（API 限流 / 网络抖动）
  4. 进度打印（每 N 条 + 每条完成都更新 ETA）
  5. 顺序保持：返回列表与输入 items 索引对齐

使用方式：
    from concurrent_runner import run_in_parallel

    def task(item, idx):
        return some_llm_call(item)

    results = run_in_parallel(
        task, items,
        max_workers=8,
        checkpoint_path="results/checkpoints/my_exp.jsonl",
        desc="MyExp",
    )
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple


def _load_checkpoint(path: Optional[str]) -> Dict[int, Any]:
    """读取 JSONL 断点文件，返回 {idx: result}。

    JSONL 每行格式：{"__idx": <int>, "result": <任意 JSON>}
    """
    if not path or not os.path.exists(path):
        return {}
    done: Dict[int, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("__idx")
            if isinstance(idx, int):
                done[idx] = rec.get("result")
    return done


def _append_checkpoint(
    path: Optional[str], idx: int, result: Any, lock: threading.Lock
) -> None:
    if not path:
        return
    payload = json.dumps(
        {"__idx": idx, "result": result}, ensure_ascii=False, default=str
    )
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload + "\n")
            f.flush()


def run_in_parallel(
    task_fn: Callable[[Any, int], Dict],
    items: List[Any],
    max_workers: int = 8,
    checkpoint_path: Optional[str] = None,
    desc: str = "Running",
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
    progress_every: int = 10,
) -> List[Dict]:
    """并发执行 task_fn(item, idx)，结果按 idx 顺序返回。

    Args:
        task_fn: 任务函数，签名 ``(item, idx) -> Dict``。
            必须线程安全（不要写共享可变全局状态）。
        items: 输入列表
        max_workers: 并发线程数
        checkpoint_path: JSONL 断点文件；None 关闭断点续跑
        desc: 进度行前缀（用于区分多个实验阶段）
        max_retries: 单条任务最多尝试次数（含首次）
        retry_base_delay: 退避基数（秒），第 n 次失败等待 base * 2^(n-1)
        progress_every: 每完成多少条输出一次进度

    Returns:
        长度等于 items 的列表，与输入索引对齐。
        失败的样本占位为 ``{"_error": "<msg>", "_idx": i}``。
    """
    if checkpoint_path:
        ckpt_dir = os.path.dirname(checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    done_results = _load_checkpoint(checkpoint_path)
    total = len(items)

    if done_results:
        print(
            f"  [{desc}] 断点续跑：复用已完成 {len(done_results)}/{total} 条"
        )

    file_lock = threading.Lock()
    results: List[Optional[Dict]] = [None] * total
    for idx, r in done_results.items():
        if 0 <= idx < total:
            results[idx] = r

    pending: List[Tuple[int, Any]] = [
        (i, item) for i, item in enumerate(items) if i not in done_results
    ]

    if not pending:
        print(f"  [{desc}] 所有 {total} 条都已在断点中，直接复用")
        return [r if r is not None else {"_error": "missing", "_idx": i}
                for i, r in enumerate(results)]

    print(
        f"  [{desc}] 启动并发：剩余 {len(pending)}/{total}  workers={max_workers}  "
        f"max_retries={max_retries}"
    )

    t0 = time.time()
    completed_lock = threading.Lock()
    completed = [len(done_results)]  # 用列表实现可变闭包

    def _worker(i: int, item: Any) -> Tuple[int, Optional[Dict], Optional[str]]:
        last_err: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                res = task_fn(item, i)
                return i, res, None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < max_retries:
                    delay = retry_base_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
        return i, None, last_err

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, i, item) for i, item in pending]
        for fut in as_completed(futures):
            i, res, err = fut.result()
            if err is not None:
                rec: Dict = {"_error": err, "_idx": i}
            else:
                rec = res if isinstance(res, dict) else {"value": res}
            results[i] = rec
            _append_checkpoint(checkpoint_path, i, rec, file_lock)

            with completed_lock:
                completed[0] += 1
                done_now = completed[0]

            if done_now % progress_every == 0 or done_now == total:
                elapsed = time.time() - t0
                new_done = done_now - len(done_results)
                rate = new_done / max(elapsed, 1e-3)
                eta = (total - done_now) / max(rate, 1e-3)
                print(
                    f"  [{desc}] {done_now}/{total}  "
                    f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s"
                )

    return [r if r is not None else {"_error": "unfilled", "_idx": i}
            for i, r in enumerate(results)]


def count_errors(results: List[Dict]) -> int:
    """统计失败样本数（用于实验后期检查）"""
    return sum(1 for r in results if isinstance(r, dict) and "_error" in r)
