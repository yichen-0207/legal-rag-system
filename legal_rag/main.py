import logging
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import threading

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from api.v1.search import router as search_router
from api.v1.qa import router as qa_router
from api.v1.agent import router as agent_router
from api.v1.admin import router as admin_router
from api.v1.compare import router as compare_router
from api.v1.topic_analysis import router as topic_analysis_router
from api.v1.mapping import router as mapping_router
from api.v1.structured_analysis import router as structured_analysis_router
from api.v1.stats import router as stats_router
from api.v1.kg import router as kg_router

import torch

# 限制 PyTorch inter-op 线程池为 1，避免与 intra-op 叠加导致 CPU 过度订阅（须在使用前调用）
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _configure_logging() -> None:
    """配置 root logger（uvicorn 启动方式下 INFO 也可用）；幂等。"""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


_configure_logging()


app = FastAPI(
    title="Legal RAG API",
    version="1.0.0",
    description="境外法规文本检索问答系统"
)


def _warm_reranker(svc) -> None:
    """用一次与真实检索同构的批量推理预热 cross-encoder（只预热 1 个 batch 覆盖主导 shape）。"""
    from core.config import settings

    long_text = ("劳动争议与劳动关系的法律适用及当事人权利义务条款。" * 40)[:500]
    batch = max(1, settings.reranker_batch_size)
    candidates = [
        {"content": long_text, "id": f"warmup-{i}"}
        for i in range(batch)
    ]
    svc.rerank(query="warmup 劳动关系", candidates=candidates, top_k=batch)


def _warm_embedding(model) -> None:
    """用一次真实编码预热嵌入模型（参数与检索链路一致）。"""
    import torch
    from core.config import settings

    with torch.no_grad():
        model.encode(
            ["warmup 劳动关系"],
            batch_size=settings.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def _load_and_warm_reranker(_log) -> None:
    """加载 Re-ranker 并做一次同构推理预热；不抛异常，失败只降级并记日志。"""
    import time
    from core.config import settings

    if not settings.reranker_enabled:
        _log("Re-ranker 已禁用（LEGAL_RERANKER_ENABLED=false），跳过预热")
        return

    t0 = time.time()
    _log("开始加载 Re-ranker 模型...")
    try:
        from services.reranker_service import ReRankerService
        svc = ReRankerService()
        if not svc._lazy_load():
            _log(f"Re-ranker 加载失败（耗时 {time.time() - t0:.1f}s），已降级为不做精排，服务继续提供检索")
            return
        _log(f"Re-ranker 加载完成，耗时 {time.time() - t0:.1f}s")

        t0 = time.time()
        _log("开始 Re-ranker 推理预热...")
        try:
            _warm_reranker(svc)
            _log(f"Re-ranker 推理预热完成，耗时 {time.time() - t0:.1f}s")
        except Exception as e:
            # 推理预热失败不影响可用性（真实请求仍能跑），只损失提前量
            _log(f"Re-ranker 推理预热失败（耗时 {time.time() - t0:.1f}s）：{type(e).__name__}: {e}")
    except Exception as e:
        _log(f"Re-ranker 加载异常（耗时 {time.time() - t0:.1f}s）：{type(e).__name__}: {e}")


def _load_and_warm_embedding(_log) -> None:
    """加载 bge-m3 并做一次真实编码预热；不抛异常，失败状态由 /health/ready 暴露。"""
    import time

    t0 = time.time()
    _log("开始加载嵌入模型 (bge-m3)...")
    try:
        from core.model_loader import ModelLoader
        model = ModelLoader.get_model()
        _log(f"嵌入模型加载完成，耗时 {time.time() - t0:.1f}s")

        t0 = time.time()
        _log("开始嵌入模型推理预热...")
        _warm_embedding(model)
        _log(f"嵌入模型推理预热完成，耗时 {time.time() - t0:.1f}s")
    except Exception as e:
        _log(f"嵌入模型加载/预热失败（耗时 {time.time() - t0:.1f}s）：{type(e).__name__}: {e}")


def _warmup_models() -> None:
    """后台预热模型与法域向量：独立线程执行，reranker 与 bge-m3 并行加载。"""
    global _warmup_done
    import time

    # 预热阶段用 print：立即 flush，启动早期不依赖日志配置
    def _log(msg: str) -> None:
        print(f"[warmup] {msg}", flush=True)

    try:
        # ---- 并行加载 Re-ranker 与 bge-m3 ----
        from concurrent.futures import ThreadPoolExecutor
        from core.config import settings

        # 提前设置线程数，避免两个加载线程并发重配置全局状态（PyTorch 不保证安全）
        torch.set_num_threads(settings.torch_num_threads)

        t_all = time.time()
        # 两个 worker 内部已吞掉异常，result() 仅用于等待完成
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="warmup-model") as pool:
            f_reranker = pool.submit(_load_and_warm_reranker, _log)
            f_embedding = pool.submit(_load_and_warm_embedding, _log)
            f_reranker.result()
            f_embedding.result()
        _log(f"模型加载与推理预热完成（并行），总耗时 {time.time() - t_all:.1f}s")

        # ---- 法域语义向量（问答用，依赖已加载的 bge-m3）----
        # 必须串行排在模型预热之后：否则会与用户请求抢 CPU，把就绪后的请求拖慢
        t0 = time.time()
        _log("开始预热法域语义向量...")
        try:
            from services.qa_service import QAService
            QAService().warmup_jurisdiction_vectors()
            _log(f"法域语义向量预热完成，耗时 {time.time() - t0:.1f}s")
        except Exception as e:
            # 不抛出：问答会降级为纯关键词法域识别，标记为已完成避免就绪探针永远 503
            _log(f"法域语义向量预热失败（耗时 {time.time() - t0:.1f}s）：{type(e).__name__}: {e}")
    finally:
        # 无论成败都置位，否则就绪探针永远 503；
        # 模型是否可用由 /health/ready 的 load_state 判定
        _warmup_done = True
        _log("预热流程结束")


_warmup_started = False
_warmup_done = False


@app.on_event("startup")
def start_warmup():
    """startup 钩子只负责拉起后台预热线程并立即返回，不阻塞 uvicorn 开始服务。"""
    global _warmup_started
    if _warmup_started:
        return
    _warmup_started = True
    threading.Thread(target=_warmup_models, name="model-warmup", daemon=True).start()

# 跨域白名单：正常部署走 nginx 同源反代不触发 CORS，仅保留本地调试场景；
# 鉴权走 Authorization 头，不依赖 cookie，故 allow_credentials=False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router, prefix="/api/v1")
app.include_router(qa_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(compare_router, prefix="/api/v1")
app.include_router(topic_analysis_router, prefix="/api/v1")
app.include_router(mapping_router, prefix="/api/v1")
app.include_router(structured_analysis_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")
app.include_router(kg_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """存活探针：进程能响应即存活（不反映模型状态，后者见 /health/ready）。"""
    return {"status": "healthy", "service": "legal-rag"}


@app.get("/health/ready")
async def readiness_check(response: Response):
    """就绪探针：预热完成且模型可用返回 200，否则 503（供 docker healthcheck 使用）。

    判定直接读取各服务真实状态：
      reranker: loaded/failed(降级不精排)/disabled 均视为就绪，pending 未就绪；
      embedding 必须 loaded（不可降级）；jurisdiction_vectors 需 ready；warmup_done 需完成。
    """
    from core.config import settings

    if not settings.reranker_enabled:
        reranker_state = "disabled"
    else:
        from services.reranker_service import ReRankerService
        reranker_state = ReRankerService().load_state

    from core.model_loader import ModelLoader
    embedding_state = ModelLoader.load_state()

    from services.qa_service import QAService
    jvector_state = QAService().jurisdiction_warmup_state

    ready = (
        _warmup_done
        and embedding_state == "loaded"
        and reranker_state in ("loaded", "failed", "disabled")
        and jvector_state == "ready"
    )
    if not ready:
        response.status_code = 503
    return {
        "status": "ready" if ready else "warming_up",
        "service": "legal-rag",
        "models": {
            "reranker": reranker_state,
            "embedding": embedding_state,
            "jurisdiction_vectors": jvector_state,
        },
        "warmup": "done" if _warmup_done else "running",
    }


if __name__ == "__main__":
    import uvicorn
    import logging
    import sys
    import signal

    # 日志已在模块级 _configure_logging() 配置，此处不再重复
    logger = logging.getLogger("legal_rag")

    def _handle_signal(signum, frame):
        logger.warning(f"收到信号 {signum}，正在退出...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        logger.info("Legal RAG 服务启动中...")
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    except Exception as e:
        logger.exception(f"服务异常退出: {type(e).__name__}: {e}")
        sys.exit(1)
