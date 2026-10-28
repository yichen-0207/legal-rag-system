"""评测脚本公共引导：统一模块搜索路径与数据/结果目录。

所有评测脚本只依赖本模块定位文件，不再各自推导 BASE_DIR；
宿主机与容器（/app）内均可直接运行：python legal_rag/evals/scripts/<脚本>.py
"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EVALS_DIR = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(EVALS_DIR)  # legal_rag（含 core/ services/）

for _path in (PROJECT_ROOT, "/app"):
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)

DATASETS_DIR = os.path.join(EVALS_DIR, "datasets")
RESULTS_DIR = os.path.join(EVALS_DIR, "results")
