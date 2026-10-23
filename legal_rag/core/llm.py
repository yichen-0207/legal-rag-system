"""统一 LLM 客户端工厂：进程级复用 OpenAI / ChatOpenAI，并挂接本地用量记账。"""

import time
from functools import lru_cache
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from openai import OpenAI

from core import llm_usage
from core.config import settings

_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class _UsageCallbackHandler(BaseCallbackHandler):
    """LangChain 回调：为 ChatOpenAI（Agent 链路）记录用量，按 run_id 隔离并发计时。"""

    def __init__(self) -> None:
        self._starts: dict[Any, float] = {}

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        start = self._starts.pop(run_id, None)
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or {}
        llm_usage.record_usage(
            model=llm_output.get("model_name", ""),
            latency_s=(time.perf_counter() - start) if start else 0.0,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=usage.get("total_tokens"),
        )

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        start = self._starts.pop(run_id, None)
        llm_usage.record_usage(
            latency_s=(time.perf_counter() - start) if start else 0.0,
            ok=False,
            error=f"{type(error).__name__}: {error}",
        )


_USAGE_CALLBACK = _UsageCallbackHandler()


@lru_cache(maxsize=32)
def get_openai_client(
    base_url: str,
    api_key: str,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> OpenAI:
    """按参数缓存 OpenAI 客户端（进程级复用连接池）；None 参数保留 SDK 默认值。"""
    kwargs: dict = {"api_key": api_key, "base_url": base_url}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return llm_usage.wrap_openai_client(OpenAI(**kwargs))


def get_default_client(
    timeout: Optional[float] = None, max_retries: Optional[int] = None
) -> OpenAI:
    """使用默认 LLM 配置（settings.llm_api_key / llm_api_base_url）的客户端。"""
    return get_openai_client(
        base_url=settings.llm_api_base_url or _DEFAULT_BASE_URL,
        api_key=settings.llm_api_key or "",
        timeout=timeout,
        max_retries=max_retries,
    )


def get_deepseek_client(
    timeout: Optional[float] = None, max_retries: Optional[int] = None
) -> OpenAI:
    """使用独立 DeepSeek 配置（settings.llm_deepseek_*）的客户端。"""
    return get_openai_client(
        base_url=settings.llm_deepseek_api_base_url or _DEFAULT_BASE_URL,
        api_key=settings.llm_deepseek_api_key or "",
        timeout=timeout,
        max_retries=max_retries,
    )


@lru_cache(maxsize=32)
def get_chat_model(
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> ChatOpenAI:
    """返回进程级复用的 ChatOpenAI（LangChain 原生入口）；None 参数保留 SDK 默认值。"""
    kwargs: dict = {
        "model": model,
        "api_key": settings.llm_api_key or "",
        "base_url": settings.llm_api_base_url or _DEFAULT_BASE_URL,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    extra_body = settings.llm_extra_body()
    if extra_body:
        kwargs["extra_body"] = extra_body
    kwargs["callbacks"] = [_USAGE_CALLBACK]
    return ChatOpenAI(**kwargs)
