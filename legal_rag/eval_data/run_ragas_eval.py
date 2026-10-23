r"""Ragas 标准化评测（仅在宿主隔离 venv .venv-eval 中运行，不进镜像）：输入 run_agent_vs_rag 产物与本地 raw_v2 gold，不触达 ES 与模型。

默认只评 llm_context_precision_without_reference：context_precision 依赖完整上下文与 gold 文本匹配，与 1500 字截断不兼容，需 --metrics 显式启用；NLI 类指标在 DeepSeek 上稳定返回 nan，已停用。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean

# ragas 0.4.3 运行时提示改用 metrics.collections，但 evaluate() 只接受 metrics.Metric 子类，故抑制弃用告警。
warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE = Path(__file__).resolve().parent          # legal_rag/eval_data
ROOT = BASE.parent                              # legal_rag
ENV_PATH = ROOT.parent / ".env"
RAW_DIR = ROOT / "data" / "raw_v2"

QUERY_VARIANT = "B_nl_zh"                       # 与 run_qa_eval.py 的 USE_VARIANT 保持一致
TEXT_FIELDS = ("text_zh", "text_en", "text_pt", "text_original_language")


def _load_llm_config() -> tuple:
    """从 .env 读取 LLM 端点配置，密钥不写入代码。"""
    text = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""

    def get(key: str, default: str = "") -> str:
        m = re.search(rf"^{key}=(\S+)", text, re.MULTILINE)
        return m.group(1) if m else os.environ.get(key, default)

    api_key = get("LEGAL_LLM_API_KEY") or get("LEGAL_DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("未找到 LEGAL_LLM_API_KEY / LEGAL_DEEPSEEK_API_KEY，无法调用 judge 模型")
    base_url = get("LEGAL_LLM_BASE_URL") or get("LEGAL_DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
    model = get("LEGAL_LLM_MODEL") or get("LEGAL_DEEPSEEK_MODEL", "deepseek-flash")
    return api_key, base_url, model


def _load_queries() -> dict:
    """返回 {query_id: 查询文本}，与 run_qa_eval.load_queries 口径一致。"""
    data = json.loads((BASE / "eval_queries.json").read_text(encoding="utf-8"))
    out = {}
    for item in data.get("queries", []):
        text = item.get(QUERY_VARIANT, "")
        if text:
            out[item["query_id"]] = text
    return out


def _load_gold() -> dict:
    """返回 {base_query_id: [relevant]}：clause 数据中 id 形如 "Q001_A"，按 "_" 截断到 base id 后聚合。"""
    items = json.loads((BASE / "eval_dataset_clause.json").read_text(encoding="utf-8"))
    gold = {}
    for item in items:
        qid = item.get("id", "")
        base_id = qid.split("_")[0] if "_" in qid else qid
        gold.setdefault(base_id, []).extend(item.get("relevant", []))
    return gold


def _build_gold_text(needed: set) -> dict:
    """扫描本地语料，为所需 (law_id, article) 构建 gold 条款正文索引（只保留命中 needed 的条目）。"""
    index = {}
    for path in RAW_DIR.rglob("*_cleaned.json"):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(records, list):
            continue
        for rec in records:
            law_id = rec.get("law_id", "")
            for art in (rec.get("article_number"), rec.get("article_number_normalized")):
                key = (law_id, art)
                if art and key in needed:
                    text = ""
                    for field in TEXT_FIELDS:
                        if rec.get(field):
                            text = rec[field]
                            break
                    if text:
                        index.setdefault(key, {})[rec.get("chunk_index", 0)] = text
    return {
        key: "\n".join(chunks[i] for i in sorted(chunks))
        for key, chunks in index.items()
    }


def _load_subset(path: str) -> set:
    """读取固定回归子集文件：JSON 字符串数组（query_id 列表）。"""
    p = Path(path)
    if not p.is_absolute():
        p = BASE / p
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise SystemExit(f"子集文件必须是 query_id 字符串数组：{p}")
    return set(data)


def _load_official_rows(artifact: Path, mode: str, limit: int, subset: set | None = None) -> list:
    data = json.loads(artifact.read_text(encoding="utf-8"))
    rows = []
    for m in (("rag", "agent") if mode == "both" else (mode,)):
        recs = data.get(m, [])
        for rec in (recs[:limit] if limit else recs):
            if subset is not None and rec.get("query_id") not in subset:
                continue
            rec["_mode"] = m
            rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Ragas 标准化评测（宿主隔离 venv）")
    ap.add_argument("--artifact", default="eval_agent_vs_rag_ragas.json", help="待评测的产物文件名")
    ap.add_argument("--mode", choices=["both", "rag", "agent"], default="both")
    ap.add_argument("--limit-records", type=int, default=0, help="每模式取样条数（0=全部）")
    ap.add_argument("--subset", default="", help="固定回归子集文件（query_id 字符串数组），仅评测其中 query_id")
    ap.add_argument("--context-max-chars", type=int, default=1500,
                    help="判分输入（上下文/reference）单条截断字数，0=不截断")
    ap.add_argument("--metrics", default="",
                    help="额外指标（逗号分隔，可选 context_precision）。默认仅 llm_context_precision_without_reference")
    ap.add_argument("--out", default="eval_ragas_report.json", help="报告输出文件名")
    args = ap.parse_args()

    artifact = BASE / args.artifact
    if not artifact.exists():
        raise SystemExit(f"产物不存在：{artifact}")

    api_key, base_url, model = _load_llm_config()
    queries = _load_queries()
    gold = _load_gold()
    subset = _load_subset(args.subset) if args.subset else None
    rows = _load_official_rows(artifact, args.mode, args.limit_records, subset)

    needed = set()
    for rec in rows:
        for g in gold.get(rec["query_id"], []):
            if g.get("article"):
                needed.add((g.get("law_id", ""), g["article"]))
    gold_text = _build_gold_text(needed)

    # 组装样本：reference 缺失的样本无法参与带参考的 context_precision
    # 判分输入（上下文/reference）按 --context-max-chars 统一截断，控制 judge 侧 token 成本
    max_chars = args.context_max_chars

    def _clip(text: str) -> str:
        return text[:max_chars] if max_chars > 0 else text

    samples, meta_rows, skipped = [], [], []
    for rec in rows:
        if not (rec.get("answer") or "").strip() or not rec.get("contexts"):
            skipped.append(f"{rec['query_id']}({rec['_mode']})")
            continue
        refs = [
            gold_text[(g.get("law_id", ""), g["article"])]
            for g in gold.get(rec["query_id"], [])
            if (g.get("law_id", ""), g["article"]) in gold_text
        ]
        context = "\n".join(refs)
        samples.append({
            "user_input": queries.get(rec["query_id"], rec["query_id"]),
            "response": rec.get("answer", ""),
            "retrieved_contexts": [_clip(c) for c in rec.get("contexts", [])],
            "reference": _clip(context),
        })
        meta_rows.append({
            "query_id": rec["query_id"],
            "mode": rec["_mode"],
            "has_ans": rec.get("has_ans"),
            "hand_faithfulness": rec.get("faithfulness_score"),
            "hand_relevance": rec.get("relevance_score"),
            "n_contexts": len(rec.get("contexts", [])),
            "has_reference": bool(context),
        })

    from openai import OpenAI

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import llm_factory
    from ragas.run_config import RunConfig
    from ragas.metrics import LLMContextPrecisionWithoutReference, ContextPrecision

    sys.path.insert(0, str(ROOT))
    from core import llm_usage

    client = llm_usage.wrap_openai_client(OpenAI(api_key=api_key, base_url=base_url))
    llm = llm_factory(model, provider="openai", client=client, max_tokens=4096, temperature=0)

    metrics = [LLMContextPrecisionWithoutReference()]
    requested = {m.strip() for m in args.metrics.split(",") if m.strip()}
    ref_coverage = sum(1 for m in meta_rows if m["has_reference"])
    if "context_precision" in requested:
        # 带参考指标与截断不兼容（结构上依赖完整上下文与 gold 的文本匹配），仅显式请求时启用
        if ref_coverage < len(meta_rows):
            print(f"  [warn] context_precision 需全部样本带 gold 参考（实际 {ref_coverage}/{len(meta_rows)}），已跳过")
        else:
            metrics.append(ContextPrecision())

    print("=" * 78)
    print(f"  Ragas 评测：{len(samples)} 条记录，model={model}")
    print(f"  指标：{[m.name for m in metrics]}")
    print(f"  gold reference 覆盖：{ref_coverage}/{len(samples)}")
    print(f"  判分输入截断：{'不截断' if args.context_max_chars <= 0 else f'{args.context_max_chars} 字/条'}")
    if subset is not None:
        print(f"  固定回归子集：{args.subset}（{len(subset)} 个 query_id）")
    if skipped:
        print(f"  [warn] 跳过 {len(skipped)} 条（无答案或无上下文）：{skipped[:6]}")
    if ref_coverage < len(samples) and "context_precision" in requested:
        print("  [warn] reference 未全覆盖，已跳过 context_precision")
    print("=" * 78)

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(**{k: v for k, v in s.items() if v})
            for s in samples
        ]
    )
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        show_progress=True,
        run_config=RunConfig(max_workers=4),
    )

    frame = result.to_pandas()
    metric_names = [m.name for m in metrics]
    aggregate, coverage = {}, {}
    for name in metric_names:
        column = frame[name].dropna()
        aggregate[name] = None if column.empty else round(float(mean(column)), 4)
        coverage[name] = f"{len(column)}/{len(frame)}"
    # 逐条对齐 Ragas 分数与手写 judge 分数，供 Step 4 对照校验使用
    for row, (_, series) in zip(meta_rows, frame.iterrows()):
        for name in metric_names:
            value = series.get(name)
            row[name] = None if value is None or value != value else round(float(value), 4)
    for name, value in aggregate.items():
        print(f"  {name:44s} = {'nan' if value is None else value}  (有效样本 {coverage[name]})")

    report = {
        "meta": {
            "artifact": args.artifact,
            "mode": args.mode,
            "n_records": len(samples),
            "model": model,
            "context_max_chars": args.context_max_chars,
            "query_variant": QUERY_VARIANT,
            "metrics": metric_names,
            "metric_coverage": coverage,
            "reference_coverage": f"{ref_coverage}/{len(samples)}",
            "subset": args.subset or None,
            "skipped_records": skipped,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "aggregate": aggregate,
        "per_record": meta_rows,
    }
    out_path = BASE / args.out
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  报告已写入：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
