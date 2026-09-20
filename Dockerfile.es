FROM elasticsearch:8.11.0

# 安装 IK 中文分词插件（供 chinese_analyzer 使用）
# 直接使用仓库内自带的插件包，不再从 GitHub 下载：
#   1. 构建不再依赖外网，避免网络波动导致构建失败；
#   2. 插件固化进镜像层，容器重建后不会丢失。
#      （此前插件是通过 docker exec 手工装进容器可写层的，一旦重建 ES 容器，
#         legal_documents 索引所依赖的 ik_max_word 分析器缺失，会导致索引无法打开。）
# 注意：官方镜像以非 root 的 elasticsearch 用户运行，COPY 的文件默认属 root，
# 在带 sticky 位的 /tmp 下无法删除，故此处显式 chown。
COPY --chown=elasticsearch:elasticsearch elasticsearch-analysis-ik-8.11.0.zip /tmp/
RUN elasticsearch-plugin install --batch file:///tmp/elasticsearch-analysis-ik-8.11.0.zip && \
    rm -f /tmp/elasticsearch-analysis-ik-8.11.0.zip
