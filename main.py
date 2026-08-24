"""评估实验入口：解析参数并调用 run_experiments()

    python main.py                        # dev 全集，使用 .env 中默认并发度
    python main.py --n 200                # 只跑前 200 条
    python main.py --workers 16           # 指定 16 路并发
    python main.py --no-resume            # 忽略已有断点强制重跑
    python main.py --n 12 --workers 4 --num-perms 2 --stability-k 2
"""

from __future__ import annotations

import argparse

from experiments import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAS 四维评估实验（I1~I4）")
    parser.add_argument(
        "--data",
        default="data/dev_rand_split.jsonl",
        help="CommonsenseQA 数据文件路径（默认 dev 集）",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="样本数量；不传或传 0 表示跑完整数据集",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并发线程数（默认读 .env 的 MAS_MAX_WORKERS=8）",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="结果输出目录（默认 results/）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有断点文件强制重跑",
    )
    parser.add_argument(
        "--num-perms",
        type=int,
        default=3,
        help="I3.1 多排列实验的排列数",
    )
    parser.add_argument(
        "--stability-k",
        type=int,
        default=5,
        help="I3.2 pass@k 中每题重复次数",
    )
    parser.add_argument(
        "--stability-n",
        type=int,
        default=10,
        help="I3.2 pass@k 中参与重复的题数",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        data_path=args.data,
        n=args.n if args.n and args.n > 0 else None,
        max_workers=args.workers,
        out_dir=args.out,
        no_resume=args.no_resume,
        num_perms=args.num_perms,
        stability_k=args.stability_k,
        stability_n=args.stability_n,
    )
