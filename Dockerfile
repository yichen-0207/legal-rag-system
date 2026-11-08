# ===== 构建阶段：安装 Python 依赖 =====
FROM python:3.10-slim AS builder

# 配置 PyPI 镜像（国内加速）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /build
COPY legal_rag/requirements.txt .

# 安装 CUDA 11.8 版 PyTorch（需在 sentence-transformers 之前安装，避免自动下载默认 GPU 版）：
# cross-encoder 精排跑 GPU，embedding 仍固定 CPU（见 core/model_loader.py device='cpu'）。
# 版本与原先 CPU 版保持 2.0.0，零版本漂移；cu118 向下兼容本机驱动 577.00（CUDA 12.9）。
# 必须显式指定 +cu118：PyPI 当前默认 torch 为 cu13 构建，在 CUDA 12.9 驱动上会报
#   "The NVIDIA driver on your system is too old (found version 12090)"。
# 走阿里云 pytorch-wheels 镜像：download.pytorch.org 在容器网络下 TLS 握手不稳定（SSLEOF）
# 该 wheel 2.27GB，容器内实测约 946KB/s（约 40 分钟），必须放大 pip 的读超时并多重重试，
# 否则默认 15s 读超时会在长下载中途 ReadTimeoutError 直接失败（实测已发生一次）。
RUN pip install --no-cache-dir --timeout 120 --retries 10 \
        "torch==2.0.0+cu118" --find-links https://mirrors.aliyun.com/pytorch-wheels/cu118/

# 安装其余依赖（numpy<2 避免与 sentence-transformers 冲突）
# 显式使用清华 PyPI 镜像：阿里云 pypi 镜像实测仅 ~100KB/s，清华 ~3.6MB/s
# 注意：每个包规格都必须加引号——否则 shell 会把 `>` / `<` 当作重定向符：
#   未加引号的 `fastapi>=0.104.0` 会被拆成 `fastapi > =0.104.0`（版本约束静默失效），
#   而 `networkx>=3.1,<4` 里的 `<4` 会触发输入重定向并直接报 "cannot open 4"。
# fastapi 及其强耦合框架件必须与旧可用镜像锁死（旧镜像 e3c477b9366c 内 pip show 实测）：
#   fastapi 0.143.3 / starlette 1.7.0 / pydantic 2.14.1 / pydantic-settings 2.15.0
# 起因：fastapi>=0.104.0 重建时被拉到 0.162.0，该版移除了 Query(enum=...)，
# 导致 api/v1/stats.py 启动即崩（TypeError: Query() got an unexpected keyword argument 'enum'）。
# 注意：本 RUN 的显式列表才是镜像实际安装依据，requirements.txt 只被 COPY 未被 -r 安装，
# 两处版本约束须保持一致（requirements.txt 供宿主机本地开发对齐）。
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "numpy<2" \
        "elasticsearch>=8.0.0,<9.0.0" \
        "fastapi==0.143.3" \
        "starlette==1.7.0" \
        "pydantic==2.14.1" \
        "pydantic-settings==2.15.0" \
        "uvicorn>=0.24.0" \
        "streamlit>=1.28.0" \
        "requests>=2.31.0" \
        "pandas>=2.0.0" \
        "plotly>=5.17.0" \
        "tqdm>=4.66.0" \
        "python-multipart>=0.0.6" \
        "sentence-transformers>=2.2.0" \
        "transformers==4.36.2" \
        "accelerate>=0.20.0" \
        "openai>=1.0.0" \
        "openpyxl>=3.1.0" \
        "networkx>=3.1,<4" \
        "langchain>=1.0.0" \
        "langchain-openai>=1.0.0" \
        "langgraph>=1.0.0"

# 答案层 Redis 客户端：单独成层，避免改动上方大依赖层触发全量重装（torch 层 2.27GB 已缓存）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "redis>=5.0.0"


# ===== 运行阶段 =====
FROM python:3.10-slim

WORKDIR /app

# 从构建阶段复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制后端源代码
COPY legal_rag/ .

# 运行用户（非 root，容器逃逸防护基线；UID 固定便于宿主机卷对账）
RUN useradd -m -u 10001 appuser

# 创建数据目录（用于挂载宿主机数据）
RUN mkdir -p /app/data/raw /models /app/logs && \
    chown -R appuser:appuser /app/data /app/logs

# 非 root 运行。端口 8001 > 1024，监听无特权要求；
# /app/data 与 /app/logs 显式授权：代码目录 root 只读即可，运行时写入仅限这两处
#（用量日志、管理端写入）。
USER appuser

# 暴露端口
EXPOSE 8001

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]