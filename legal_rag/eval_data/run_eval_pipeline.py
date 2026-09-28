"""
检索评估全流程管道
===================
Step 2: 为80条查询检索Top-20候选文档 → 填入Excel L列
Step 3: 基于K列ground truth自动标注相关性 → 填入Excel M列
Step 4: 构建eval_dataset.json → 运行3种策略评估 → 输出对比表
"""
import sys, os, json, math
from typing import List, Dict, Tuple, Optional
from collections import Counter
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 配置
# ============================================================
# 本脚本与数据集同处 eval_data/ 目录，BASE_DIR 已是该目录，不能再拼一层 "eval_data"。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_QUERIES_PATH = os.path.join(BASE_DIR, "eval_queries.json")
XLSX_PATH = os.path.join(BASE_DIR, "eval_annotation_template.xlsx")
EVAL_DATASET_PATH = os.path.join(BASE_DIR, "eval_dataset.json")
TOP_K = 20


# ============================================================
# Step 2: 检索候选文档（混合检索，关闭reranker）
# ============================================================
def retrieve_candidates():
    """对每条查询执行混合检索，获取Top-K候选文档"""
    from services.retriever_service import RetrieverService
    from core.config import settings

    original_reranker = settings.reranker_enabled
    settings.reranker_enabled = False

    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]

    retriever = RetrieverService()
    all_candidates = {}

    print("=" * 60)
    print("  Step 2: 检索候选文档 (Top-20 混合检索)")
    print("=" * 60)

    for q in queries:
        qid = q["query_id"]
        query_text = q["B_nl_zh"]
        jur = q.get("jurisdiction", None)
        # 跨法域查询: 不传jurisdiction，使检索在所有法域中进行
        if jur == "跨法域":
            jur = None

        try:
            results = retriever.hybrid_search(
                query=query_text, top_k=TOP_K, jurisdiction=jur,
            )
            candidates = [(r["law_id"], r["article_number"]) for r in results]
        except Exception as e:
            print(f"  [ERROR] {qid}: {e}")
            candidates = []

        all_candidates[qid] = candidates
        print(f"  {qid}: {len(candidates)} 候选")

    settings.reranker_enabled = original_reranker
    print(f"\n  共检索 {len(all_candidates)} 条查询")
    return all_candidates, retriever  # 复用retriever


def fill_excel_L_column(all_candidates: Dict[str, List[Tuple[str, str]]]):
    """将候选文档写入Excel L列"""
    wb = load_workbook(XLSX_PATH)
    ws = wb["标注总表"]

    updated = 0
    for row in range(2, 82):
        query_id = ws.cell(row=row, column=1).value
        if query_id in all_candidates:
            pairs = all_candidates[query_id]
            formatted = "; ".join(f"{lid} ({art})" for lid, art in pairs)
            ws.cell(row=row, column=12, value=formatted)
            updated += 1

    wb.save(XLSX_PATH)
    print(f"\n  L列已更新 {updated} 条查询的候选文档")
    print(f"  保存至: {XLSX_PATH}")
    return wb


# ============================================================
# Step 3: 自动标注相关性
# ============================================================
def auto_annotate(wb=None):
    """基于K列ground truth自动标注M列"""
    if wb is None:
        wb = load_workbook(XLSX_PATH)
    ws = wb["标注总表"]

    # 从eval_queries.json读取核心查询列表
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    core_ids = {q["query_id"] for q in data["queries"] if q.get("is_core", False)}

    print("\n" + "=" * 60)
    print("  Step 3: 自动标注相关性")
    print("=" * 60)

    annotated = 0
    for row in range(2, 82):
        query_id = ws.cell(row=row, column=1).value
        k_val = ws.cell(row=row, column=11).value
        l_val = ws.cell(row=row, column=12).value

        if not query_id or not k_val or not l_val:
            ws.cell(row=row, column=13, value="")
            continue

        gt_ids = set()
        for part in str(k_val).split(";"):
            part = part.strip()
            if part and "_" in part:
                gt_ids.add(part)

        candidates = str(l_val).split(";")
        relevance_scores = []
        is_core = query_id in core_ids

        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            lid = cand.split("(")[0].strip()
            relevance_scores.append(2 if (is_core and lid in gt_ids) else
                                     1 if lid in gt_ids else 0)

        m_val = ", ".join(str(s) for s in relevance_scores)
        ws.cell(row=row, column=13, value=m_val)
        annotated += 1

        relevant_count = sum(1 for s in relevance_scores if s > 0)
        print(f"  {query_id}: {len(relevance_scores)} 候选, {relevant_count} 相关"
              f" ({'核心' if is_core else '非核心'})")

    wb.save(XLSX_PATH)
    print(f"\n  M列已标注 {annotated} 条查询")
    return wb, core_ids


# ============================================================
# Step 4a: 构建 eval_dataset.json
# ============================================================
def fetch_articles_for_law_ids(gt_ids: set) -> Dict[str, set]:
    """从ES获取每个law_id的所有article_number"""
    from repositories.elasticsearch import ElasticsearchRepository
    repo = ElasticsearchRepository()

    law_articles = {}
    for lid in sorted(gt_ids):
        try:
            law_data = repo.get_law_by_id(lid)
            if law_data and "chunks" in law_data:
                articles = set()
                for chunk in law_data["chunks"]:
                    art = chunk.get("article_number", "")
                    if art:
                        articles.add(art)
                law_articles[lid] = articles
                print(f"    {lid}: {len(articles)} 条款")
            else:
                law_articles[lid] = set()
                print(f"    {lid}: 0 条款(无数据)")
        except Exception as e:
            print(f"    [WARN] {lid}: {e}")
            law_articles[lid] = set()
    return law_articles


def _get_ground_truth_dict() -> Dict[str, List[str]]:
    """ground truth law_id映射"""
    return {
        "Q001": ["US_CFR_T6_P5_2003", "US_CFR_T48_P239_1991"],
        "Q002": ["US_CFR_T45_P164_2000", "US_CFR_T16_P314_2002"],
        "Q003": ["US_CFR_T47_P1_1963", "US_CFR_T47_P4_2004"],
        "Q004": ["US_CFR_T16_P310_2010", "US_CFR_T16_P313_2000"],
        "Q005": ["US_CFR_T6_P5_2003", "US_CFR_T32_P286_2017"],
        "Q006": ["US_CFR_T48_P52_1983", "US_CFR_T48_P1_1983"],
        "Q007": ["US_CFR_T31_P1010_2010", "US_CFR_T31_P1020_2010", "US_CFR_T31_P1022_2010"],
        "Q008": ["US_CFR_T28_P16_1977", "US_CFR_T45_P164_2000", "US_CFR_T16_P313_2000"],
        "Q009": ["US_CFR_T12_P1005_2011"],
        "Q010": ["US_CFR_T47_P64_1963", "US_CFR_T47_P1_1963"],
        "Q011": ["JP_Act_2021_SMP_JP_013", "JP_Act_2002_SMP_JP_007"],
        "Q012": ["JP_Act_2003_SMP_JP_002"],
        "Q013": ["JP_Act_1984_SMP_JP_014"],
        "Q014": ["JP_Act_2009_SMP_JP_027"],
        "Q015": ["JP_Act_1970_SMP_JP_033"],
        "Q016": ["JP_Act_1949_SMP_JP_034", "JP_Act_1952_SMP_JP_042"],
        "Q017": ["JP_Act_2003_SMP_JP_002"],
        "Q018": ["JP_Act_1949_SMP_JP_034", "JP_Constitution_1946_SMP_JP_001"],
        "Q019": ["JP_Act_2000_SMP_JP_038"],
        "Q020": ["JP_Act_2003_SMP_JP_002", "JP_Act_2021_SMP_JP_013"],
        "Q021": ["HK_CAP_486"],
        "Q022": ["HK_CAP_200"],
        "Q023": ["HK_CAP_106"],
        "Q024": ["HK_CAP_528"],
        "Q025": ["HK_CAP_553"],
        "Q026": ["HK_CAP_584"],
        "Q027": ["HK_CAP_584"],
        "Q028": ["HK_CAP_486"],
        "Q029": ["HK_CAP_362"],
        "Q030": ["HK_CAP_593"],
        "Q031": ["MAC_LEI_8_2005"],
        "Q032": ["MAC_LEI_11_2009"],
        "Q033": ["MAC_LEI_14_2001"],
        "Q034": ["MAC_LEI_5_2005"],
        "Q035": ["MAC_LEI_13_2023"],
        "Q036": ["MAC_LEI_8_2023", "MAC_LEI_2_2009"],
        "Q037": ["MAC_LEI_13_2019"],
        "Q038": ["MAC_LEI_2_2006"],
        "Q039": ["TH_ACT_1991"],
        "Q040": ["TH_CONST_2007"],
        "Q041": ["TH_ACT_2002"],
        "Q042": ["TH_ACT_1994"],
        "Q043": ["TH_ACT_1976"],
        "Q044": ["TH_ACT_1979"],
        "Q045": ["TH_ACT_1997"],
        "Q046": ["TH_ACT_1999"],
        "Q047": ["SG_ACT_This_Act_is_the_Pers_2012", "SG_ACT_22_2016"],
        "Q048": ["SG_ACT_9_2018"],
        "Q049": ["SG_ACT_16_2010"],
        "Q050": ["SG_ACT_This_Act_is_the_Fina_2022", "SG_ACT_18_2022"],
        "Q051": ["SG_ACT_43_1999"],
        "Q052": ["SG_ACT_22_2021"],
        "Q053": ["SG_ACT_This_Act_is_the_Spam_2007", "SG_ACT_40_2020"],
        "Q054": ["SG_ACT_This_Act_is_the_Paym_2019", "SG_ACT_1_2021"],
        "Q055": ["VN_QH12SOCIALIST_41_2009", "VN_N_25_2011"],
        "Q056": ["VN_QH11_67_2006"],
        "Q057": ["VN_N_27_2007", "VN_Q_25_2006", "VN_TT_37_2009"],
        "Q058": ["VN_N_75_2003"],
        "Q059": ["VN_Q_252_2003", "VN_N_12_2006"],
        "Q060": ["VN_N_12_2006"],
        "Q061": ["VN_N_87_2010"],
        "Q062": ["VN_N_75_2003"],
        "Q063": ["MY_ACT_709_2010"],
        "Q064": ["MY_ACT_854_2024"],
        "Q065": ["MY_ACT_709_2010"],
        "Q066": ["MY_ACT_322_1985"],
        "Q067": ["MY_ACT_322_1985"],
        "Q068": ["MY_ACT_854_2024"],
        "Q069": ["MY_ACT_709_2010"],
        "Q070": ["MY_ACT_709_2010"],
        "Q071": ["HK_CAP_486", "SG_ACT_This_Act_is_the_Pers_2012", "MY_ACT_709_2010", "MAC_LEI_8_2005", "JP_Act_2003_SMP_JP_002"],
        "Q072": ["JP_Act_2021_SMP_JP_013", "VN_QH11_67_2006", "SG_ACT_22_2021"],
        "Q073": ["JP_Act_2003_SMP_JP_002", "SG_ACT_This_Act_is_the_Pers_2012", "HK_CAP_486", "MY_ACT_709_2010"],
        "Q074": ["HK_CAP_553", "SG_ACT_16_2010", "MAC_LEI_5_2005", "JP_Act_2000_SMP_JP_038"],
        "Q075": ["HK_CAP_106", "SG_ACT_43_1999", "MAC_LEI_14_2001", "VN_QH12SOCIALIST_41_2009"],
        "Q076": ["HK_CAP_584", "SG_ACT_This_Act_is_the_Fina_2022", "MAC_LEI_13_2023"],
        "Q077": ["SG_ACT_9_2018", "MY_ACT_854_2024", "MAC_LEI_13_2019"],
        "Q078": ["HK_CAP_584", "SG_ACT_This_Act_is_the_Paym_2019", "MAC_LEI_2_2006", "TH_ACT_1999"],
        "Q079": ["HK_CAP_362", "SG_ACT_27_2003", "TH_ACT_1979", "MAC_LEI_9_2021"],
        "Q080": ["HK_CAP_486", "SG_ACT_This_Act_is_the_Pers_2012", "MY_ACT_709_2010", "MAC_LEI_8_2005", "JP_Act_2003_SMP_JP_002"],
    }


# ============================================================
# Step 4b: 运行评估（共享retriever，修复跨法域问题）
# ============================================================
def run_evaluation_shared(retriever, dataset_path: str):
    """
    使用共享retriever运行3种策略，避免重复加载模型。
    直接从 scripts.evaluate 复用 EvalQuery、EvalResult 等数据结构。
    """
    from scripts.evaluate import (
        load_dataset, EvalQuery, EvalResult, print_comparison_table,
        STRATEGIES,
    )
    from core.config import settings

    print("\n" + "=" * 60)
    print("  Step 4: 运行检索评估 (共享Retriever)")
    print("=" * 60)

    queries = load_dataset(dataset_path)
    if not queries:
        print("  [ERROR] 数据集为空！")
        return

    all_results = []
    for strategy_key in ["vector", "hybrid", "hybrid+reranker"]:
        strategy = STRATEGIES[strategy_key]
        print(f"\n  ▶ 策略: {strategy['name']}")
        print(f"  {'=' * 40}")

        results: List[EvalResult] = []
        for q in queries:
            # 跨法域查询: jurisdiction设为None
            jur = q.jurisdiction if q.jurisdiction != "跨法域" else None
            original_jur = q.jurisdiction
            q.jurisdiction = jur

            # 配置reranker
            if strategy_key == "vector":
                settings.reranker_enabled = False
            elif strategy_key == "hybrid":
                settings.reranker_enabled = False
            elif strategy_key == "hybrid+reranker":
                settings.reranker_enabled = True

            retrieved = strategy["run"](retriever, q, top_k=TOP_K)
            q.jurisdiction = original_jur  # 恢复

            er = EvalResult(
                query_id=q.id,
                query_text=q.query,
                relevant_count=len(q.get_relevant_articles()),
                retrieved_ids=retrieved,
            )
            er._gold = q.get_relevant_articles()
            er._eval_query = q
            results.append(er)

        # 计算指标
        total = len(results)
        mrr_total = 0.0
        recall_total = {k: 0.0 for k in [5, 10, 20]}
        precision_total = {k: 0.0 for k in [5, 10, 20]}
        ndcg_total = {k: 0.0 for k in [5, 10, 20]}

        for er in results:
            mrr_total += er.mrr
            for k in recall_total:
                recall_total[k] += er.recall_at(k)
            for k in precision_total:
                precision_total[k] += er.precision_at(k)

        for er in results:
            gains = sorted(
                [er._eval_query.get_gain(lid, art) for lid, art in er._gold],
                reverse=True,
            )
            ideal_dcg = sum(
                g / math.log2(i + 2) for i, g in enumerate(gains[:20])
            )
            for k in ndcg_total:
                ndcg_total[k] += er.ndcg_at(k, ideal_dcg)

        summary = {
            "strategy": strategy["name"],
            "total_queries": total,
            "MRR": round(mrr_total / total, 4),
        }
        for k in [5, 10, 20]:
            summary[f"Recall@{k}"] = round(recall_total[k] / total, 4)
            summary[f"Precision@{k}"] = round(precision_total[k] / total, 4)
            summary[f"NDCG@{k}"] = round(ndcg_total[k] / total, 4)

        all_results.append(summary)
        print(f"  {strategy['name']} 完成: MRR={summary['MRR']:.4f}")

    # 恢复reranker设置
    settings.reranker_enabled = True
    print_comparison_table(all_results)
    return all_results


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  检索评估全流程管道")
    print("=" * 60)

    # Step 2: 检索候选文档（同时获得共享retriever）
    all_candidates, retriever = retrieve_candidates()
    wb = fill_excel_L_column(all_candidates)

    # Step 3: 自动标注相关性
    wb, core_ids = auto_annotate(wb)

    # Step 4a: 构建eval_dataset.json
    print("\n" + "-" * 60)
    print("  构建评估数据集...")
    gt_dict = _get_ground_truth_dict()
    all_gt_ids = set()
    for lids in gt_dict.values():
        for lid in lids:
            all_gt_ids.add(lid)
    print(f"  共 {len(all_gt_ids)} 个唯一ground truth law_ids")
    print("  获取ground truth法规的条款列表...")
    law_articles = fetch_articles_for_law_ids(all_gt_ids)

    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        queries_data = json.load(f)["queries"]

    dataset = []
    variant_keys = ["A_keyword_zh", "B_nl_zh", "C_mixed_zhen", "D_pure_en"]
    variant_suffixes = ["A", "B", "C", "D"]

    for q in queries_data:
        qid = q["query_id"]
        jur = q.get("jurisdiction", "")
        is_core = qid in core_ids
        rel_score = 2 if is_core else 1
        gt_lids = gt_dict.get(qid, [])

        relevant = []
        for lid in gt_lids:
            articles = law_articles.get(lid, set())
            if articles:
                for art in articles:
                    relevant.append({"law_id": lid, "article": art, "relevance": rel_score})
            else:
                relevant.append({"law_id": lid, "article": "", "relevance": rel_score})

        for vk, vs in zip(variant_keys, variant_suffixes):
            query_text = q.get(vk, "")
            if not query_text:
                continue
            dataset.append({
                "id": f"{qid}_{vs}",
                "query": query_text,
                "jurisdiction": jur,
                "relevant": relevant,
                "is_core": is_core,
            })

    with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    avg_rel = sum(len(gt_dict.get(q["query_id"], [])) for q in queries_data) / len(queries_data)
    print(f"\n  数据集已保存: {EVAL_DATASET_PATH}")
    print(f"  共 {len(dataset)} 条评估条目 ({len(dataset)//4} 查询 × 4 变体)")

    # Step 4b: 运行评估（共享retriever）
    results = run_evaluation_shared(retriever, EVAL_DATASET_PATH)

    print("\n" + "=" * 60)
    print("  全流程完成！")
    print("=" * 60)
    print(f"\n  输出文件:")
    print(f"    Excel: {XLSX_PATH}  (L列=候选, M列=标注)")
    print(f"    数据集: {EVAL_DATASET_PATH}")

    return results


if __name__ == "__main__":
    main()
