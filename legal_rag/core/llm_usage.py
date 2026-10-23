# -*- coding: utf-8 -*-
"""LLM 调用用量账本：每次调用追加一行 NDJSON 到 <log_dir>/llm_usage_YYYY-MM-DD.jsonl。

仅用标准库；配置走环境变量（LEGAL_LLM_USAGE_LOG_ENABLED / LEGAL_LLM_USAGE_LOG_DIR），
purpose 由调用栈自动推断；落盘失败只告警，不影响主流程。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENABLED_ENV = "LEGAL_LLM_USAGE_LOG_ENABLED"
_DIR_ENV = "LEGAL_LLM_USAGE_LOG_DIR"
_DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)

# 推断用途时跳过的框架/传输帧，不代表业务语义
_SKIP_FUNCS = {
    "create",
    "record_usage",
    "_call_llm",
    "_call_llm_with_model",
    "_call_llm_stream",
    "_call_llm_stream_with_messages",
}
_SKIP_MODULE_PREFIXES = (
    "openai",
    "httpx",
    "httpcore",
    "anyio",
    "core.llm",
    "core.llm_usage",
    "langchain",
    "langgraph",
    "tenacity",
    "ragas",
)

_lock = threading.Lock()


def enabled() -> bool:
    """是否启用用量记账；默认开启，可用 LEGAL_LLM_USAGE_LOG_ENABLED=false 关闭。"""
    raw = os.environ.get(_ENABLED_ENV, "true").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def log_dir() -> str:
    """账本目录；未配置时落到 <仓库根>/logs。"""
    return os.environ.get(_DIR_ENV) or _DEFAULT_DIR


def current_log_path() -> Path:
    """当日账本文件路径（供报表脚本读取）。"""
    return Path(log_dir()) / f"llm_usage_{datetime.now():%Y-%m-%d}.jsonl"


def _infer_purpose() -> str:
    """从调用栈推断业务用途，形如 "qa_service:ask_stream"。"""
    frame = sys._getframe(2)  # 0=_infer_purpose, 1=record_usage, 2=调用 record_usage 处
    while frame is not None:
        name = frame.f_code.co_name
        module = frame.f_globals.get("__name__", "")
        if not (name in _SKIP_FUNCS or module.startswith(_SKIP_MODULE_PREFIXES)):
            label = module.rsplit(".", 1)[-1]
            # __main__ 无信息量时改用脚本文件名
            if label == "__main__":
                label = os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
            return f"{label}:{name}"
        frame = frame.f_back
    return "unknown"


def record_usage(
    *,
    model: str = "",
    purpose: Optional[str] = None,
    latency_s: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: Optional[int] = None,
    ok: bool = True,
    error: str = "",
    stream: bool = False,
) -> None:
    """追加一条用量记录；未启用或落盘失败均静默跳过。"""
    if not enabled():
        return
    if total_tokens is None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    row: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "purpose": purpose or _infer_purpose(),
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": total_tokens,
        "latency_s": round(latency_s, 2),
        "stream": bool(stream),
        "ok": bool(ok),
    }
    if error:
        row["error"] = str(error)[:200]
    try:
        with _lock:
            path = current_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 - 记账失败不影响主流程
        logger.warning("[LLM用量] 落盘失败: %s", e)


def _record_from_response(response: Any, model: str, start: float) -> None:
    usage = getattr(response, "usage", None)
    record_usage(
        model=model or getattr(response, "model", ""),
        latency_s=time.perf_counter() - start,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _wrap_stream(stream: Any, model: str, start: float):
    """包裹流式响应：原样透传分块，迭代结束时按末块 usage 记账。"""

    def generator():
        prompt_tokens = completion_tokens = 0
        total_tokens = None
        resolved_model = model
        ok = True
        err = ""
        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    total_tokens = getattr(usage, "total_tokens", None)
                if getattr(chunk, "model", None):
                    resolved_model = chunk.model
                yield chunk
        except Exception as e:  # noqa: BLE001 - 记账后原样抛出
            ok = False
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            record_usage(
                model=resolved_model or model,
                latency_s=time.perf_counter() - start,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ok=ok,
                error=err,
                stream=True,
            )

    return generator()


def wrap_openai_client(client: Any) -> Any:
    """以鸭子类型包裹 chat.completions.create 自动记账；流式在末块截获 usage，幂等且失败不影响主流程。"""
    completions = client.chat.completions
    if getattr(completions, "_llm_usage_wrapped", False):
        return client
    original_create = completions.create

    def create(*args: Any, **kwargs: Any):
        model = kwargs.get("model", "")
        stream = bool(kwargs.get("stream"))
        injected = False
        if stream:
            options = dict(kwargs.get("stream_options") or {})
            if not options.get("include_usage"):
                options["include_usage"] = True
                kwargs["stream_options"] = options
                injected = True
        start = time.perf_counter()
        try:
            try:
                result = original_create(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if not (injected and getattr(e, "status_code", None) == 400):
                    raise
                # 服务端不支持 stream_options.include_usage：去掉后重试（token 降级为 0）
                kwargs.pop("stream_options", None)
                result = original_create(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - 记账后原样抛出
            record_usage(
                model=model,
                latency_s=time.perf_counter() - start,
                ok=False,
                error=f"{type(e).__name__}: {e}",
                stream=stream,
            )
            raise
        if stream:
            return _wrap_stream(result, model, start)
        _record_from_response(result, model, start)
        return result

    completions.create = create
    completions._llm_usage_wrapped = True
    return client
