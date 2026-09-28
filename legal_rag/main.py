import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import threading

from fastapi import FastAPI, Response
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

import torch

# 限制 PyTorch inter-op 线程池为 1。该池默认大小等于 CPU 核数（本机 16），与 intra-op
# 的 8 路叠加后在 16 核上过度订阅：实测冷启动后前几个请求会烧掉 ~300 CPU秒（占满
# ~14/16 核），而同样的工作稳态只需 ~44.6 CPU秒。必须在任何并行工作开始前调用
# （此处位于所有模型推理之前，安全）；重复调用会抛 RuntimeError，忽略即可。
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

app = FastAPI(
    title="Legal RAG API",
    version="1.0.0",
    description="境外法规文本检索问答系统"
)


def _warm_reranker(svc) -> None:
    """用一次与真实检索同构的批量推理预热 cross-encoder。

    为什么要做「推理预热」而不只是「加载模型」：PyTorch 的 CPU 后端会在首次前向时
    按张量 shape 编译 oneDNN/MKL 原语并扩张内存池。实测只加载模型时，/health/ready
    变 200 之后的前 4~5 个请求仍持续偏慢（46.9 → 23.6 → 25.2 → 14.1 → 7.8s），
    就是因为这些开销被推迟到了真实请求上。

    预热规模对齐真实路径：候选数取 reranker_window，content 填满 500 字符
    （rerank 内部按 content[:500] 截断），这样覆盖的是 shape 最大、代价最高的那批前向。
    """
    from core.config import settings

    long_text = ("劳动争议与劳动关系的法律适用及当事人权利义务条款。" * 40)[:500]
    candidates = [
        {"content": long_text, "id": f"warmup-{i}"}
        for i in range(max(1, settings.reranker_window))
    ]
    svc.rerank(query="warmup 劳动关系", candidates=candidates, top_k=settings.reranker_window)


def _warm_embedding(model) -> None:
    """用一次真实编码预热嵌入模型。

    参数与检索链路保持一致（见 ElasticsearchRepository 的 encode 调用）：
    真实检索只编码查询文本本身，故这里同样只喂一条短文本。
    """
    import torch
    from core.config import settings

    with torch.no_grad():
        model.encode(
            ["warmup 劳动关系"],
            batch_size=settings.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def _warmup_models() -> None:
    """后台预热模型与法域向量（在独立线程中执行）。

    关键点：
    1. 必须放在独立线程里。模型加载与首次推理都是阻塞的 CPU/IO 重活，原先直接写在
       async startup handler 中会占住事件循环，uvicorn 在完成前无法处理任何请求。
    2. 不能只预热 reranker。嵌入模型原先走「首个请求触发的懒加载」，重启后第一个搜索
       请求要自己扛下整个 bge-m3 加载（实测加载耗时 36.7s）。
    3. 加载完成后各跑一次真实推理，把首次前向的 oneDNN/MKL 初始化开销留在启动阶段
       （见 _warm_reranker / _warm_embedding）。
    4. 法域语义向量也纳入本流程串行执行（见下）。它原先在 QAService 构造时以独立线程
       启动、且每个实例各算一份，实测 2 份并发把 16 核占满约 90 秒，又不受就绪探针门控，
       才是「就绪后前几个请求 20 秒以上」的真正原因——并非首次前向开销。
    沿用「加载失败不阻塞启动」的策略：reranker 失败降级为不精排；bge-m3 是必需组件无法降级，
    失败时记录日志并由 /health/ready 报未就绪。
    """
    global _warmup_done
    import time
    from core.config import settings

    # 用 print 而非 logger.info：本项目只在 __main__ 分支里调用了 logging.basicConfig，
    # 容器以 `uvicorn main:app` 启动时不会执行该分支，root logger 没有 handler，
    # "legal_rag" 的 INFO 级别日志实际落不进 docker logs（实测确认）。
    # print 直接写 stdout，在 docker logs 里稳定可见。
    def _log(msg: str) -> None:
        print(f"[warmup] {msg}", flush=True)

    try:
        # ---- Re-ranker（较小，先加载）----
        if not settings.reranker_enabled:
            _log("Re-ranker 已禁用（LEGAL_RERANKER_ENABLED=false），跳过预热")
        else:
            t0 = time.time()
            _log("开始加载 Re-ranker 模型...")
            from services.reranker_service import ReRankerService
            svc = ReRankerService()
            if svc._lazy_load():
                _log(f"Re-ranker 加载完成，耗时 {time.time() - t0:.1f}s")
                t0 = time.time()
                _log("开始 Re-ranker 推理预热...")
                try:
                    _warm_reranker(svc)
                    _log(f"Re-ranker 推理预热完成，耗时 {time.time() - t0:.1f}s")
                except Exception as e:
                    # 推理预热失败不影响可用性（真实请求仍能跑），只损失提前量
                    _log(f"Re-ranker 推理预热失败（耗时 {time.time() - t0:.1f}s）：{type(e).__name__}: {e}")
            else:
                _log(f"Re-ranker 加载失败（耗时 {time.time() - t0:.1f}s），已降级为不做精排，服务继续提供检索")

        # ---- bge-m3 嵌入模型（较大，耗时主要在加载）----
        t0 = time.time()
        _log("开始加载嵌入模型 (bge-m3)...")
        from core.model_loader import ModelLoader
        try:
            model = ModelLoader.get_model()
            _log(f"嵌入模型加载完成，耗时 {time.time() - t0:.1f}s")
            t0 = time.time()
            _log("开始嵌入模型推理预热...")
            _warm_embedding(model)
            _log(f"嵌入模型推理预热完成，耗时 {time.time() - t0:.1f}s")
        except Exception as e:
            # 不抛出：抛出会中断预热线程。检索链路缺少嵌入模型必然报错，由 /health/ready 暴露
            _log(f"嵌入模型加载/预热失败（耗时 {time.time() - t0:.1f}s）：{type(e).__name__}: {e}")

        # ---- 法域语义向量（问答用，依赖已加载的 bge-m3）----
        # 必须串行放在统一预热流程里：它是一次性重活（实测占满多核数十秒）。若像原先
        # 那样放在后台线程异步跑，会在 /health/ready 变绿之后继续与用户请求抢 CPU，
        # 把就绪后的前几个请求拖慢到 20 秒以上。QAService 已改为单例，这里只算一份。
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
        # 无论成败都置位：否则就绪探针永远 503，前端（depends_on: service_healthy）会一直被挡在门外。
        # 模型是否可用由 /health/ready 里的 load_state 判定，本标记只表示"预热流程已跑完"。
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
    """存活探针：只要进程能响应即视为存活。

    语义保持不变（不反映模型/依赖状态），避免影响既有调用方。
    「能否高效服务」的判断请用 /health/ready。
    """
    return {"status": "healthy", "service": "legal-rag"}


@app.get("/health/ready")
async def readiness_check(response: Response):
    """就绪探针：模型加载完成后返回 200，预热期间返回 503。

    用于 docker healthcheck 与部署等待。预热期间返回 503 而不是 200，是为了让调用方
    推迟发送检索/问答这类重活请求——否则会排在模型加载后面长时间等待（与 /health 的区别
    就在于此：/health 是「进程活着」，本接口是「能高效服务」）。

    就绪判定直接读取各服务实例的真实加载状态，不另外维护一份状态，避免与实际不一致：
      - reranker：loaded=可用；failed=已降级为不精排（仍算就绪）；disabled=配置关闭（就绪）；
        pending=仍在加载（未就绪）
      - embedding：必须 loaded。它是检索链路的必需组件，失败无法降级（未就绪）
      - jurisdiction_vectors：法域语义向量预热是否完成。它是启动阶段最重的一次性计算，
        若不等它跑完就放行，用户请求会与它抢 CPU（实测被拖慢到 20 秒以上）
      - warmup_done：预热线程是否跑完（含推理预热）。模型"加载完成"不等于"首次推理已预热"，
        若不等这一步，请求会赶在推理预热之前到达，仍然要付首次前向的开销
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
