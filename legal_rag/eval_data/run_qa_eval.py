"""
问答评估脚本 (RAG QA Evaluation)
================================
评估指标:
  1. HasAns@5     - 检索结果是否包含答案（与 gold 条款匹配）
  2. Faithfulness - 生成答案是否忠实于检索文档（LLM-as-judge, 1-5分）
  3. Relevance    - 生成答案是否准确回答用户问题（LLM-as-judge, 1-5分）

使用方式:
  cd legal_rag
  python run_qa_eval.py

输出:
  - 终端打印评估汇总表
  - eval_data/eval_qa_results.json  - 每条查询详细结果
  - eval_data/eval_qa_summary.json  - 汇总指标
"""
import sys, os, json
from typing import List, Dict, Optional
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_QUERIES_PATH = os.path.join(BASE_DIR, "eval_data", "eval_queries.json")
EVAL_CLAUSE_PATH = os.path.join(BASE_DIR, "eval_data", "eval_dataset_clause.json")
OUTPUT_RESULTS = os.path.join(BASE_DIR, "eval_data", "eval_qa_results.json")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "eval_data", "eval_qa_summary.json")
TOP_K = 5  # 问答场景取 Top-5

# 只评估 NL 中文变体（80条），兼顾效率与代表性
USE_VARIANT = "B_nl_zh"


# ============================================================
# 1. 加载数据
# ============================================================
def load_queries() -> List[Dict]:
    """加载 eval_queries.json，返回 [{query_id, query, jurisdiction, ...}]"""
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    for item in data.get("queries", []):
        qid = item["query_id"]
        query_text = item.get(USE_VARIANT, "")
        if not query_text:
            continue
        queries.append({
            "query_id": qid,
            "query": query_text,
            "jurisdiction": item.get("jurisdiction", ""),
            "topic": item.get("topic", ""),
            "core_concept": item.get("core_concept", ""),
            "is_core": item.get("is_core", False),
        })
    print(f"  加载 {len(queries)} 条查询 (变体: {USE_VARIANT})")
    return queries


def load_gold() -> Dict[str, List[Dict]]:
    """加载 eval_dataset_clause.json，返回 {query_id: [{law_id, article, relevance, ...}]}"""
    with open(EVAL_CLAUSE_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    gold = {}
    # 注意：clause 数据中 id 格式为 "Q001_A"，需匹配变体
    var_suffix = USE_VARIANT.split("_")[0]  # "B"
    for item in items:
        qid = item.get("id", "")
        # 检查是否匹配当前变体，或者匹配 base query_id
        base_id = qid.split("_")[0] if "_" in qid else qid
        if base_id not in gold:
            gold[base_id] = []
        gold[base_id].extend(item.get("relevant", []))
    return gold


# ============================================================
# 2. 检索 + 问答生成
# ============================================================
def run_qa_pipeline(query: str, jurisdiction: str, gold_articles: List[Dict],
                    retriever, qa_service, settings) -> Dict:
    """执行单条问答评估"""
    from services.retriever_service import RetrieverService

    result = {
        "query": query,
        "has_ans": False,
        "gold_matched": [],
        "retrieved_articles": [],
        "generated_answer": "",
        "faithfulness_score": 0,
        "relevance_score": 0,
        "error": None,
    }

    # ---- Step 1: 检索 ----
    try:
        jur = None if jurisdiction == "跨法域" else jurisdiction
        # 使用混合检索（向量+BM25+RRF，num_workers=0 修复后稳定可用）
        retrieved = retriever.hybrid_search(query=query, top_k=TOP_K, jurisdiction=jur)
        retrieved_articles = [(r["law_id"], r.get("article_number", ""), r.get("content", ""))
                              for r in retrieved]
        result["retrieved_articles"] = [{"law_id": lid, "article": art}
                                        for lid, art, _ in retrieved_articles]
    except Exception as e:
        result["error"] = f"检索失败: {e}"
        return result

    # ---- Step 2: HasAns ----
    gold_set = {(g["law_id"], g.get("article", "")) for g in gold_articles}
    has_ans = False
    matched = []
    for lid, art, _ in retrieved_articles:
        if (lid, art) in gold_set:
            has_ans = True
            matched.append({"law_id": lid, "article": art})
    result["has_ans"] = has_ans
    result["gold_matched"] = matched

    # ---- Step 3: 生成答案 ----
    try:
        # 直接用检索结果构建 prompt，避免 QAService 重新加载 embedding 模型
        ref_text = "\n\n".join([
            f"【来源{i+1}】{jurisdiction}《{r.get('title', '')}》第{r.get('article_number', '')}条\n{r.get('content', '')[:1500]}"
            for i, r in enumerate(retrieved)
        ])
        prompt = f"""你是一位精通境外法律法规的资深法律顾问。请严格基于以下参考资料回答用户问题。

【参考资料】
{ref_text}

【用户问题】
{query}

请用中文回答，要求：
1. 直接回答问题，给出明确结论
2. 引用相关法规条款（标注【来源X】）
3. 如果资料不足，明确说明
4. 回答控制在500字以内"""

        answer = qa_service._call_llm_with_model(
            prompt,
            model=settings.llm_qa_model,
            fallback_model=settings.llm_qa_fallback_model,
        )
        result["generated_answer"] = answer
    except Exception as e:
        result["error"] = f"生成失败: {e}"
        return result

    # ---- Step 4: LLM-as-judge 评估忠实度与相关性 ----
    try:
        faith_score = judge_faithfulness(query, answer, retrieved_articles, qa_service)
        result["faithfulness_score"] = faith_score

        rel_score = judge_relevance(query, answer, qa_service)
        result["relevance_score"] = rel_score
    except Exception as e:
        result["error"] = (result.get("error") or "") + f" | 评估失败: {e}"

    return result


# ============================================================
# 3. LLM-as-Judge
# ============================================================
def judge_faithfulness(query: str, answer: str, retrieved_articles: List[tuple],
                       qa_service) -> int:
    """LLM-as-judge 评估忠实度 (1-5分)"""
    # 构建参考文档摘要
    ref_text = "\n\n".join([f"【来源{i+1}】\n{c[:500]}" for i, (_, _, c) in enumerate(retrieved_articles)])

    prompt = f"""请评估以下生成答案对参考文档的忠实程度。

【用户问题】
{query}

【参考文档】
{ref_text}

【生成答案】
{answer}

请根据以下标准给出1-5分的评价：
1 = 完全编造，与参考文档无关
2 = 大部分内容无法从参考文档推导
3 = 部分忠实，部分编造
4 = 大部分忠实，少量细节无法确认
5 = 完全忠实，所有内容都能从参考文档找到依据

请仅输出一个数字。"""

    try:
        response = qa_service._call_llm(prompt)
        # 提取数字
        for ch in response.strip():
            if ch.isdigit():
                score = int(ch)
                return max(1, min(5, score))
    except Exception:
        pass
    return 0


def judge_relevance(query: str, answer: str, qa_service) -> int:
    """LLM-as-judge 评估相关性 (1-5分)"""
    prompt = f"""请评估以下生成答案与用户问题的相关程度。

【用户问题】
{query}

【生成答案】
{answer}

请根据以下标准给出1-5分的评价：
1 = 完全不相关，答非所问
2 = 部分相关，但偏离主题
3 = 基本相关，但不够精准
4 = 相关且精准
5 = 高度相关，完全覆盖问题要点

请仅输出一个数字。"""

    try:
        response = qa_service._call_llm(prompt)
        for ch in response.strip():
            if ch.isdigit():
                score = int(ch)
                return max(1, min(5, score))
    except Exception:
        pass
    return 0


# ============================================================
# 5. 汇总计算
# ============================================================
def compute_summary(results: List[Dict]) -> Dict:
    """计算汇总指标"""
    n = len(results)
    if n == 0:
        return {}

    has_ans_count = sum(1 for r in results if r["has_ans"])
    faith_scores = [r["faithfulness_score"] for r in results if r["faithfulness_score"] > 0]
    rel_scores = [r["relevance_score"] for r in results if r["relevance_score"] > 0]
    errors = [r for r in results if r.get("error")]

    return {
        "total_queries": n,
        "has_ans_accuracy": round(has_ans_count / max(n, 1), 4),
        "has_ans_count": has_ans_count,
        "avg_faithfulness": round(sum(faith_scores) / max(len(faith_scores), 1), 2) if faith_scores else 0,
        "avg_relevance": round(sum(rel_scores) / max(len(rel_scores), 1), 2) if rel_scores else 0,
        "faithfulness_distribution": {
            "1-2分": sum(1 for r in faith_scores if r <= 2),
            "3分": sum(1 for r in faith_scores if r == 3),
            "4-5分": sum(1 for r in faith_scores if r >= 4),
        },
        "relevance_distribution": {
            "1-2分": sum(1 for r in rel_scores if r <= 2),
            "3分": sum(1 for r in rel_scores if r == 3),
            "4-5分": sum(1 for r in rel_scores if r >= 4),
        },
        "error_count": len(errors),
    }


def print_summary(summary: Dict):
    print(f"\n  {'='*60}")
    print(f"  问答评估结果")
    print(f"  {'='*60}")
    print(f"  总查询数:       {summary['total_queries']}")
    print(f"  错误数:         {summary['error_count']}")
    print(f"  HasAns@5:       {summary['has_ans_accuracy']:.2%} ({summary['has_ans_count']}/{summary['total_queries']})")
    print(f"  Avg Faithfulness: {summary['avg_faithfulness']:.2f} / 5.0")
    print(f"  Avg Relevance:    {summary['avg_relevance']:.2f} / 5.0")
    print(f"  {'-'*60}")
    print(f"  忠实度分布:")
    for k, v in summary["faithfulness_distribution"].items():
        pct = v / max(summary['total_queries'], 1) * 100
        print(f"    {k}: {v} ({pct:.1f}%)")
    print(f"  相关性分布:")
    for k, v in summary["relevance_distribution"].items():
        pct = v / max(summary['total_queries'], 1) * 100
        print(f"    {k}: {v} ({pct:.1f}%)")


# ============================================================
# 6. 主流程
# ============================================================
def main():
    sys.path.insert(0, BASE_DIR)

    print("=" * 60)
    print("  问答评估 (RAG QA Evaluation)")
    print("  变体: " + USE_VARIANT)
    print("  Top-K: " + str(TOP_K))
    print("=" * 60)

    # 加载数据
    queries = load_queries()
    gold_map = load_gold()
    if not queries:
        print("  [ERROR] 无查询数据")
        return

    print(f"  Gold 数据: {len(gold_map)} 条 base query")

    # 延迟导入，避免启动时加载模型
    from services.retriever_service import RetrieverService
    from services.qa_service import QAService
    from core.config import settings

    # 禁用 reranker（混合检索 RRF 融合，不加载重排序模型）
    settings.reranker_enabled = False

    retriever = RetrieverService()
    qa_service = QAService()

    # 逐个评估
    all_results = []
    for i, q in enumerate(queries):
        qid = q["query_id"]
        gold = gold_map.get(qid, [])
        if not gold:
            print(f"  [{i+1}/{len(queries)}] {qid} - 跳过（无 gold）")
            continue

        print(f"  [{i+1}/{len(queries)}] {qid}: {q['query'][:60]}...", end=" ", flush=True)

        result = run_qa_pipeline(
            query=q["query"],
            jurisdiction=q["jurisdiction"],
            gold_articles=gold,
            retriever=retriever,
            qa_service=qa_service,
            settings=settings,
        )
        result["query_id"] = qid
        result["jurisdiction"] = q["jurisdiction"]
        result["topic"] = q["topic"]
        result["is_core"] = q["is_core"]

        if result.get("error"):
            print(f"❌ {result['error'][:50]}")
        else:
            print(f"✓ HasAns={result['has_ans']} Faith={result['faithfulness_score']} Rel={result['relevance_score']}")

        all_results.append(result)

    # 汇总
    summary = compute_summary(all_results)
    print_summary(summary)

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果: {OUTPUT_RESULTS}")
    print(f"  汇总: {OUTPUT_SUMMARY}")

    # 按法域细分
    print(f"\n  {'='*60}")
    print(f"  按法域细分")
    print(f"  {'='*60}")
    jur_groups = defaultdict(list)
    for r in all_results:
        jur_groups[r["jurisdiction"]].append(r)
    for jur in sorted(jur_groups.keys()):
        grp = jur_groups[jur]
        ha = sum(1 for r in grp if r["has_ans"])
        fs = [r["faithfulness_score"] for r in grp if r["faithfulness_score"] > 0]
        rs = [r["relevance_score"] for r in grp if r["relevance_score"] > 0]
        print(f"  {jur:<10} n={len(grp):>2}  HasAns={ha}/{len(grp)}  "
              f"Faith={sum(fs)/max(len(fs),1):.1f}  Rel={sum(rs)/max(len(rs),1):.1f}")

    print("\n" + "=" * 60)
    print("  完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()