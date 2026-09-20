# ===== 构建阶段：安装 Python 依赖 =====
FROM python:3.10-slim AS builder

# 配置 PyPI 镜像（国内加速）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /build
COPY legal_rag/requirements.txt .

# 安装 CPU 版 PyTorch（需在 sentence-transformers 之前安装，避免自动下载 GPU 版）
# 走阿里云 pytorch-wheels 镜像：download.pytorch.org 在容器网络下 TLS 握手不稳定（SSLEOF）
RUN pip install --no-cache-dir "torch==2.0.0+cpu" --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/

# 安装其余依赖（numpy<2 避免与 sentence-transformers 冲突）
# 显式使用清华 PyPI 镜像：阿里云 pypi 镜像实测仅 ~100KB/s，清华 ~3.6MB/s
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "numpy<2" \
        "elasticsearch>=8.0.0,<9.0.0" \
        fastapi>=0.104.0 \
        uvicorn>=0.24.0 \
        streamlit>=1.28.0 \
        requests>=2.31.0 \
        pandas>=2.0.0 \
        plotly>=5.17.0 \
        tqdm>=4.66.0 \
        pydantic>=2.0.0 \
        pydantic-settings>=2.0.0 \
        python-multipart>=0.0.6 \
        sentence-transformers>=2.2.0 \
        transformers==4.36.2 \
        accelerate>=0.20.0 \
        openai>=1.0.0


# ===== 运行阶段 =====
FROM python:3.10-slim

WORKDIR /app

# 从构建阶段复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制后端源代码
COPY legal_rag/ .

# 创建数据目录（用于挂载宿主机数据）
RUN mkdir -p /app/data/raw /models

# 暴露端口
EXPOSE 8001

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]