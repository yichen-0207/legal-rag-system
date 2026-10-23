"""批量预抽取法规知识图谱 — 在容器内执行: python run_kg_extraction.py [选项]

把全部（或指定范围）法规的三元组预抽取并落库，填充 legal_kg_triples / legal_kg_entities。
幂等：已缓存且指纹未变的法规自动跳过，不产生 LLM 调用，中断后重跑安全。

示例：--dry-run 只估算成本；--law-ids A,B 指定法规；
     --sort-articles small --limit 3 烟雾测试；--jurisdiction 香港 按法域放量。
"""
import argparse
import json
import os
import sys
from datetime import datetime

# 与 main.py 保持一致：避免 tokenizers 在并行环境下产生告警
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from core.config import settings
from repositories.elasticsearch import ElasticsearchRepository
from services.kg_extraction_service import KGExtractionService, DEFAULT_BATCH_SIZE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量预抽取法规知识图谱（实体关系三元组）"
    )
    parser.add_argument("--jurisdiction", default=None,
                        help="只处理指定法域（如 澳门 / 新加坡）")
    parser.add_argument("--law-ids", default=None,
                        help="只处理显式指定的法规 ID，逗号分隔（优先于 --jurisdiction）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 部法规（试跑用，0 表示全部）")
    parser.add_argument("--sort-articles", choices=["small", "large"], default=None,
                        help="按条款数升/降序排列后再取 --limit，避免 --limit 抽到大法规")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发数，默认 1（串行）；>1 存在实体词典写入竞争风险")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="串行模式下每次调用之间的间隔（秒），用于规避限流")
    parser.add_argument("--force", action="store_true",
                        help="忽略缓存强制重抽（本体/别名表变更后使用）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只估算成本，不调用 LLM")
    parser.add_argument("--rebuild-entities", action="store_true",
                        help="零 LLM 重建实体词典：从已有三元组重新归一并写回（修复历史写入失败）")
    parser.add_argument("--prune-entities", action="store_true",
                        help="（破坏性，需与 --rebuild-entities 同用）清理词典中已不被任何三元组引用的孤立实体")
    return parser.parse_args()


def _write_failure_log(failed_laws: list) -> str:
    """把失败法规明细写入 logs/，返回日志路径；无失败返回空串"""
    if not failed_laws:
        return ""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"kg_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(failed_laws, f, ensure_ascii=False, indent=2)
    return log_path


def main() -> int:
    args = _parse_args()

    repo = ElasticsearchRepository()
    all_laws = repo.get_all_laws(jurisdiction=args.jurisdiction)
    count_map = {law.get("law_id"): law.get("article_count", 0) for law in all_laws}

    if args.law_ids:
        # 显式指定：以用户给定顺序为准；未入库的 ID 会在抽取阶段标记为 not_found
        law_ids = [x.strip() for x in args.law_ids.split(",") if x.strip()]
        scope = "指定法规"
    else:
        law_ids = [law.get("law_id") for law in all_laws if law.get("law_id")]
        scope = args.jurisdiction or "全部法域"

    # 按条款数排序后再截断，避免 --limit 恰好抽到最大的几部法规
    if args.sort_articles:
        law_ids.sort(
            key=lambda lid: count_map.get(lid, 0),
            reverse=(args.sort_articles == "large"),
        )

    if args.limit and args.limit > 0:
        law_ids = law_ids[: args.limit]

    print(f"目标范围：{scope}，共 {len(law_ids)} 部法规")
    if args.sort_articles:
        print(f"（已按条款数{'升' if args.sort_articles == 'small' else '降'}序排列）")
    if not law_ids:
        print("未找到任何法规，退出。")
        return 0

    # ---- 预检：LLM 密钥。--dry-run / --rebuild-entities 均不调用 LLM ----
    if not args.dry_run and not args.rebuild_entities and not settings.llm_api_key:
        print(
            "错误：未配置 LLM 密钥（LEGAL_LLM_API_KEY），无法抽取。"
            "如需仅估算成本请加 --dry-run。",
            file=sys.stderr,
        )
        return 2

    extractor = KGExtractionService()

    # ---- 零 LLM 重建实体词典（历史回溯）----
    if args.rebuild_entities:
        # 未显式指定法规时，自动收敛到「三元组索引中出现过的法规」，不做空查
        scope_ids = law_ids if args.law_ids else None
        scope_desc = f"{len(scope_ids)} 部（指定）" if scope_ids else "全部已抽取法规（自动收敛）"
        print(f"开始重建实体词典（零 LLM，prune={args.prune_entities}），范围：{scope_desc}...")
        summary = extractor.reconcile_entities(scope_ids, prune=args.prune_entities)
        print("\n===== 重建完成 =====")
        print(f"范围 {summary.get('scope', 0)} 部 / 重建 {summary['rebuilt']} / "
              f"无三元组 {summary['empty']} / 失败 {summary['failed']}")
        if args.prune_entities:
            pruned = summary.get("pruned", 0)
            print(f"孤立实体清理: {pruned if pruned >= 0 else '失败（见日志）'}")
        log_path = _write_failure_log(summary.get("failed_laws", []))
        if log_path:
            print(f"失败明细已写入: {log_path}")
        try:
            stats = repo.get_kg_statistics()
            print(
                f"知识层累计: 三元组 {stats.get('triple_count', 0)} 条 / "
                f"已抽取法规 {stats.get('law_count', 0)} 部 / "
                f"实体 {stats.get('entity_count', 0)} 个"
            )
        except Exception as e:
            print(f"（读取知识层统计失败: {type(e).__name__}: {e}）")
        return 0 if summary["failed"] == 0 else 1

    # ---- 成本估算模式 ----
    if args.dry_run:
        summary = extractor.extract_for_laws(law_ids, dry_run=True)
        print("\n===== 成本估算（未调用 LLM）=====")
        print(f"法规总数          : {summary['total']}")
        print(f"缓存命中(可跳过)   : {summary.get('cached_laws', 0)}")
        print(f"待抽取法规        : {summary.get('planned_laws', 0)}")
        print(f"待抽取条款数      : {summary.get('planned_articles', 0)}")
        print(f"预估 LLM 调用次数  : {summary.get('planned_llm_calls', 0)}"
              f"（每批约 {DEFAULT_BATCH_SIZE} 条条款）")
        not_found = [p for p in summary.get("plans", []) if p.get("status") == "not_found"]
        if not_found:
            print(f"数据缺失（会跳过） : {len(not_found)} 部")
        return 0

    # ---- 实际抽取 ----
    print(f"开始抽取（workers={args.workers}, delay={args.delay}s, force={args.force}）...")
    summary = extractor.extract_for_laws(
        law_ids,
        workers=args.workers,
        delay=args.delay,
        force_refresh=args.force,
    )

    print("\n===== 抽取完成 =====")
    print(
        f"成功 {summary['ok']}（其中缓存命中 {summary['cache_hit']}，"
        f"实际 LLM 调用 {summary['llm_calls']}） / 失败 {summary['failed']} / "
        f"未找到 {summary['not_found']} / 空条款 {summary['empty']}"
    )
    print(f"本次三元组总数: {summary['triples_total']}")

    log_path = _write_failure_log(summary.get("failed_laws", []))
    if log_path:
        print(f"失败明细已写入: {log_path}")

    try:
        stats = repo.get_kg_statistics()
        print(
            f"知识层累计: 三元组 {stats.get('triple_count', 0)} 条 / "
            f"已抽取法规 {stats.get('law_count', 0)} 部 / "
            f"实体 {stats.get('entity_count', 0)} 个"
        )
    except Exception as e:
        print(f"（读取知识层统计失败: {type(e).__name__}: {e}）")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断（进度已落库，重跑将自动续作）。")
        sys.exit(130)
