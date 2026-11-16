"""生成 4 张评估图表：①四策略分组柱状图 ②Recall 折线图 ③问答三指标概览 ④评分分布堆叠柱状图。"""
import os, json, datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from _bootstrap import RESULTS_DIR
from eval_metrics import NDCG_DEFINITION

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

OUTPUT_DIR = os.path.join(RESULTS_DIR, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIZ_PATH = os.path.join(RESULTS_DIR, "eval_viz_data.json")
QA_PATH = os.path.join(RESULTS_DIR, "eval_qa_summary.json")
PER_QUERY_PATH = os.path.join(RESULTS_DIR, "eval_per_query.csv")

with open(VIZ_PATH, "r", encoding="utf-8") as f:
    viz_data = json.load(f)
with open(QA_PATH, "r", encoding="utf-8") as f:
    qa_summary = json.load(f)


def report_vintage() -> None:
    """打印两组输入数据的产出时点，相差超过 1 天时告警（①② 来自 run_enhanced_eval，③④ 来自 run_qa_eval，均无内部时间戳，以文件 mtime 为准）。"""
    def fmt(p: str) -> str:
        return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")

    print(f"  数据时点: 检索基线 {fmt(VIZ_PATH)} | 问答结果 {fmt(QA_PATH)}")
    gap_days = abs(os.path.getmtime(VIZ_PATH) - os.path.getmtime(QA_PATH)) / 86400
    if gap_days > 1:
        print(f"  [告警] 两组数据相差 {gap_days:.1f} 天，①② 与 ③④ 并非同一批次；")
        print("         请先重跑 run_enhanced_eval.py 与 run_qa_eval.py，再引用这四张图。")


STRATEGY_LABELS = ['仅BM25', '纯向量', '混合(RRF)', '混合+重排']
STRATEGY_COLORS = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']
STRATEGY_KEYS = ['bm25', 'vector', 'hybrid', 'hybrid+reranker']
# viz overall 键 → eval_per_query.csv 的 strategy 列取值
CSV_STRATEGY_ALIAS = {
    'bm25': '仅BM25检索', 'vector': '纯向量检索',
    'hybrid': '混合检索(RRF)', 'hybrid+reranker': '混合+Re-ranker',
}


def check_ndcg_definition(viz: dict) -> None:
    """图表的 NDCG@10 必须来自 eval_metrics 口径；旧结果文件没有该字段，需明确告警。"""
    declared = viz.get("ndcg_definition")
    if declared is None:
        print("  [告警] eval_viz_data.json 缺少 ndcg_definition：图中 NDCG@10 为 v1 旧口径"
              "（三 k 共用 IDCG@20，gold<k 归零），不可与新口径结果比较。")
        print("         请重跑 run_four_strategies.py 生成新结果后再引用此图。")
    elif declared != NDCG_DEFINITION:
        print(f"  [告警] NDCG 口径不一致：图表数据为 {declared}，当前实现为 {NDCG_DEFINITION}。")


def check_viz_matches_csv(viz: dict) -> None:
    """MRR / Recall@10 / Precision@5 可由逐查询 CSV 精确复算，对不上说明图表与 CSV 不是同一批结果。"""
    if not os.path.exists(PER_QUERY_PATH):
        print("  [告警] 未找到 eval_per_query.csv，无法校验图表与逐查询结果是否同源。")
        return
    import csv
    rows = {}
    with open(PER_QUERY_PATH, "r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r.get("strategy", ""), []).append(r)

    overall = viz.get("overall", {})
    checked = 0
    for key, label in CSV_STRATEGY_ALIAS.items():
        items = rows.get(label) or []
        if not items or key not in overall:
            continue
        n = len(items)
        recomputed = {
            "MRR": sum(float(x["MRR"]) for x in items) / n,
            "Recall@10": sum(float(x["R@10"]) for x in items) / n,
            "Precision@5": sum(float(x["P@5"]) for x in items) / n,
        }
        for m, v in recomputed.items():
            plotted = overall[key].get(m)
            if plotted is None:
                continue
            checked += 1
            if abs(plotted - v) > 1e-3:
                print(f"  [告警] {key}.{m} 图表={plotted:.4f} CSV={v:.4f}："
                      "图表数据与逐查询结果不同源，请重跑 run_four_strategies.py。")
    if checked:
        print(f"  同源校验: {checked} 项（MRR / Recall@10 / Precision@5 × 4 策略）通过")


def _metric(overall: dict, sk: str, name: str):
    """取指标值；缺失或 None（该指标无有效查询）时明确报错，避免格式化崩溃。"""
    v = overall.get(sk, {}).get(name)
    if v is None:
        raise SystemExit(f"  [错误] {sk}.{name} 缺失或为 None（无有效查询），"
                         "请先重跑 run_four_strategies.py 再出图。")
    return v


# ========== ① 四策略分组柱状图 ==========
def chart1_grouped_bar():
    overall = viz_data["overall"]
    metrics = ['MRR', 'Recall@10', 'Precision@5', 'NDCG@10']
    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, sk in enumerate(STRATEGY_KEYS):
        vals = [_metric(overall, sk, m) for m in metrics]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=STRATEGY_LABELS[i],
                      color=STRATEGY_COLORS[i], edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    rotation=0, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('得分', fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.set_title('① 四检索策略核心指标对比', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper right', borderaxespad=1.2, handlelength=1.5,
              handletextpad=0.8, labelspacing=0.8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUTPUT_DIR, '01_grouped_bar.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 01_grouped_bar.png")


# ========== ② Recall@k 趋势折线图 ==========
def chart2_recall_curve():
    overall = viz_data["overall"]
    k_values = [5, 10, 20]
    markers = ['o', 's', 'D', '^']
    linestyles = ['-', '--', '-', '-']  # 纯向量用虚线，强调平台期

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, sk in enumerate(STRATEGY_KEYS):
        recalls = [overall[sk][f'Recall@{k}'] for k in k_values]
        ax.plot(k_values, recalls, marker=markers[i], linestyle=linestyles[i],
                label=STRATEGY_LABELS[i], color=STRATEGY_COLORS[i],
                linewidth=2.5, markersize=9)

        # 数据标签：黑色加白色背景框，提高对比度
        for k, v in zip(k_values, recalls):
            ax.annotate(f'{v:.3f}', (k, v), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=8.5,
                        fontweight='bold', color='#1a1a1a',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='#cccccc', alpha=0.9))

    # 纯向量平台期标注 (K=20)
    vec_recall_20 = overall['vector']['Recall@20']
    ax.annotate('Recall 停滞\n(纯向量平台期)', xy=(20, vec_recall_20),
                xytext=(18, vec_recall_20 + 0.06),
                fontsize=8.5, fontweight='bold', color='#D35400',
                arrowprops=dict(arrowstyle='->', color='#D35400', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                          edgecolor='#FFBE7A', alpha=0.95))

    # BM25 与纯向量在 K=10 处交叉标注
    bm25_r10 = overall['bm25']['Recall@10']
    vec_r10 = overall['vector']['Recall@10']
    ax.annotate(f'BM25={bm25_r10:.3f}\n向量={vec_r10:.3f}', xy=(10, (bm25_r10 + vec_r10)/2),
                xytext=(8, (bm25_r10 + vec_r10)/2 + 0.04),
                fontsize=8, fontweight='bold', color='#555555',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#cccccc', alpha=0.95))

    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Recall@K', fontsize=12)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 0.48)
    ax.set_title('② 四策略 Recall@K 趋势折线图', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower right', borderaxespad=1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUTPUT_DIR, '02_recall_curve.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 02_recall_curve.png")


# ========== ③ 三指标概览柱状图 ==========
def chart3_overview():
    """三指标并列柱状图，每个指标标注原始量纲"""
    has_ans = qa_summary['has_ans_accuracy']
    faith = qa_summary['avg_faithfulness'] / 5.0
    relevance = qa_summary['avg_relevance'] / 5.0

    labels = ['HasAns@5', 'Faithfulness\n(归一化)', 'Relevance\n(归一化)']
    values = [has_ans, faith, relevance]
    raw_labels = [f'{has_ans:.1%}', f'{qa_summary["avg_faithfulness"]:.2f}/5.0', f'{qa_summary["avg_relevance"]:.2f}/5.0']
    colors = ['#82B0D2', '#8ECFC9', '#FA7F6F']

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.8, width=0.5)
    for bar, v, r in zip(bars, values, raw_labels):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                r, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('得分', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title('③ 问答评估三核心指标概览', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUTPUT_DIR, '03_overview.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 03_overview.png")


# ========== ④ 评分分布堆叠柱状图 ==========
def chart4_stacked_distribution():
    faith = qa_summary['faithfulness_distribution']
    rel = qa_summary['relevance_distribution']

    faith_vals = [faith['1-2分'], faith['3分'], faith['4-5分']]
    rel_vals = [rel['1-2分'], rel['3分'], rel['4-5分']]

    x = np.arange(2)
    width = 0.45
    # 改用蓝→紫→深蓝单一色相梯度，色盲友好
    colors = ['#7FB3D8', '#5B8DB8', '#2C5F8A']
    labels = ['1-2分 (低)', '3分 (中)', '4-5分 (高)']

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # 堆叠 — Faithfulness
    bottom = 0
    for i in range(3):
        ax.bar(x[0], faith_vals[i], width, bottom=bottom, color=colors[i],
               edgecolor='white', linewidth=0.5, label=labels[i])
        if faith_vals[i] > 0:
            pct = faith_vals[i] / sum(faith_vals) * 100
            ax.text(x[0], bottom + faith_vals[i]/2, f'{faith_vals[i]}条\n({pct:.0f}%)',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if i < 2 else 'white')
        else:
            ax.text(x[0], bottom + 2, '0 条\n(0%)', ha='center', va='center',
                    fontsize=8, color='#999999', fontstyle='italic')
        bottom += faith_vals[i]

    # 堆叠 — Relevance
    bottom = 0
    for i in range(3):
        ax.bar(x[1], rel_vals[i], width, bottom=bottom, color=colors[i],
               edgecolor='white', linewidth=0.5)
        if rel_vals[i] > 0:
            pct = rel_vals[i] / sum(rel_vals) * 100
            ax.text(x[1], bottom + rel_vals[i]/2, f'{rel_vals[i]}条\n({pct:.0f}%)',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white')
        else:
            # 显式标注 0 条，避免误读为遗漏
            ax.text(x[1], bottom + 3, '0 条\n(0%)', ha='center', va='center',
                    fontsize=8, color='#999999', fontstyle='italic')
        bottom += rel_vals[i]

    ax.set_xticks(x)
    ax.set_xticklabels(['Faithfulness\n(忠实度)', 'Relevance\n(相关性)'], fontsize=11)
    ax.set_ylabel('查询数量', fontsize=12)
    ax.set_title('④ 问答评分分布堆叠柱状图', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper right', borderaxespad=1.0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUTPUT_DIR, '04_stacked_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 04_stacked_distribution.png")


if __name__ == "__main__":
    print(f"\n  {'='*45}")
    print(f"  生成4张优化图表 -> {OUTPUT_DIR}")
    print(f"  {'='*45}\n")

    report_vintage()
    check_ndcg_definition(viz_data)
    check_viz_matches_csv(viz_data)

    print("  [检索评估]")
    chart1_grouped_bar()
    chart2_recall_curve()

    print("\n  [问答评估]")
    chart3_overview()
    chart4_stacked_distribution()

    n = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    print(f"\n  {'='*45}")
    print(f"  完成! 共生成 {n} 张图表")
    print(f"  {'='*45}")