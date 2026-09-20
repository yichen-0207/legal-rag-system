import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.search import router as search_router
from api.v1.qa import router as qa_router
from api.v1.admin import router as admin_router
from api.v1.compare import router as compare_router
from api.v1.topic_analysis import router as topic_analysis_router
from api.v1.mapping import router as mapping_router
from api.v1.structured_analysis import router as structured_analysis_router
from api.v1.stats import router as stats_router
from api.v1.kg import router as kg_router

app = FastAPI(
    title="Legal RAG API",
    version="1.0.0",
    description="境外法规文本检索问答系统"
)

# 预加载模型（缩短首次搜索耗时）
@app.on_event("startup")
async def warmup():
    import logging
    from core.config import settings
    logger = logging.getLogger("legal_rag")
    if not settings.reranker_enabled:
        logger.info("Re-ranker 已禁用（LEGAL_RERANKER_ENABLED=false），跳过预加载")
        return
    logger.info("预加载 Re-ranker 模型...")
    from services.reranker_service import ReRankerService
    svc = ReRankerService()
    # 加载失败不阻塞启动：服务继续提供检索/问答，仅降级为不做精排
    if svc._lazy_load():
        logger.info("Re-ranker 模型预加载完成")
    else:
        logger.warning("Re-ranker 模型预加载失败，已降级为不做精排，服务继续启动")

# 跨域支持（前后端分离）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router, prefix="/api/v1")
app.include_router(qa_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(compare_router, prefix="/api/v1")
app.include_router(topic_analysis_router, prefix="/api/v1")
app.include_router(mapping_router, prefix="/api/v1")
app.include_router(structured_analysis_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")
app.include_router(kg_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "legal-rag"}


if __name__ == "__main__":
    import uvicorn
    import logging
    import sys
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
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
