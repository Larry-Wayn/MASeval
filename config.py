import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_env(key: str, default=None, required: bool = False, cast=None):
    value = os.environ.get(key, default)
    if required and (value is None or value == ""):
        raise RuntimeError(
            f"环境变量 {key} 未设置。请在 shell 中 export，或在项目根目录创建 .env 文件 "
            f"（可参考 .env.example）。"
        )
    if cast is not None and value is not None:
        try:
            value = cast(value)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"环境变量 {key} 类型转换失败: {e}") from e
    return value


LLM_CONFIG = {
    "model": _get_env("DEEPSEEK_MODEL", default="deepseek-v4-flash"),
    "api_key": _get_env("DEEPSEEK_API_KEY", required=True),
    "temperature": _get_env("DEEPSEEK_TEMPERATURE", default=0.7, cast=float),
    "base_url": _get_env("DEEPSEEK_BASE_URL", default="https://api.deepseek.com/v1"),
}


SYSTEM_CONFIG = {
    "num_agents": _get_env("MAS_NUM_AGENTS", default=3, cast=int),
    "max_rounds": _get_env("MAS_MAX_ROUNDS", default=5, cast=int),
    "enable_validation": _get_env("MAS_ENABLE_VALIDATION", default="true").lower()
    in ("1", "true", "yes", "y", "on"),
    "max_workers": _get_env("MAS_MAX_WORKERS", default=8, cast=int),
    "max_retries": _get_env("MAS_MAX_RETRIES", default=3, cast=int),
    "retry_base_delay": _get_env("MAS_RETRY_BASE_DELAY", default=2.0, cast=float),
}
