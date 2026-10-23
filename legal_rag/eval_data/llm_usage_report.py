r"""汇总 legal_rag/logs/llm_usage_*.jsonl 账本：按天分文件，每行一条调用记录（ts/model/purpose/tokens/latency/ok），仅用标准库。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def _log_dir(override: Optional[str]) -> Path:
    """账本目录：优先命令行，其次复用 core.llm_usage 的默认解析（含环境变量）。"""
    if override:
        return Path(override)
    sys.path.insert(0, str(ROOT))
    from core import llm_usage

    return Path(llm_usage.log_dir())


def _new_acc() -> dict:
    return {"calls": 0, "ok": 0, "fail": 0, "prompt": 0, "completion": 0, "total": 0, "latency": 0.0}


def _accumulate(acc: dict, rec: dict) -> None:
    acc["calls"] += 1
    acc["ok" if rec.get("ok", True) else "fail"] += 1
    acc["prompt"] += int(rec.get("prompt_tokens", 0) or 0)
    acc["completion"] += int(rec.get("completion_tokens", 0) or 0)
    acc["total"] += int(rec.get("total_tokens", 0) or 0)
    acc["latency"] += float(rec.get("latency_s", 0.0) or 0.0)


def _resolve_dates(args: argparse.Namespace) -> Optional[set[str]]:
    if args.date:
        return set(args.date)
    if args.days:
        today = date.today()
        return {(today - timedelta(days=i)).isoformat() for i in range(args.days)}
    return None


def _iter_records(log_dir: Path, dates: Optional[set[str]]) -> Iterable[dict]:
    for fp in sorted(log_dir.glob("llm_usage_*.jsonl")):
        if dates and fp.stem.replace("llm_usage_", "") not in dates:
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _fmt(acc: dict) -> str:
    avg = acc["latency"] / acc["calls"] if acc["calls"] else 0.0
    rate = acc["fail"] / acc["calls"] * 100 if acc["calls"] else 0.0
    return (
        f"{acc['calls']:>6} 次 | 失败 {acc['fail']:>4} ({rate:4.1f}%) | "
        f"输入 {acc['prompt']:>12,} | 输出 {acc['completion']:>10,} | "
        f"合计 {acc['total']:>12,} | 均耗时 {avg:6.2f}s"
    )


def _cost(acc: dict, price_in: float, price_out: float) -> float:
    return acc["prompt"] / 1e6 * price_in + acc["completion"] / 1e6 * price_out


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 用量账本汇总")
    parser.add_argument("--dir", default=None, help="账本目录（默认取 core.llm_usage 解析结果）")
    parser.add_argument("--date", action="append", help="指定日期 YYYY-MM-DD，可重复")
    parser.add_argument("--days", type=int, default=None, help="汇总最近 N 天")
    parser.add_argument("--json", dest="json_out", default=None, help="另存 JSON 结果路径")
    parser.add_argument("--price-in", type=float, default=None, help="输入单价（元/百万token）")
    parser.add_argument("--price-out", type=float, default=None, help="输出单价（元/百万token）")
    args = parser.parse_args()

    log_dir = _log_dir(args.dir)
    dates = _resolve_dates(args)
    overall = _new_acc()
    by_date: dict[str, dict] = defaultdict(_new_acc)
    by_purpose: dict[str, dict] = defaultdict(_new_acc)
    by_model: dict[str, dict] = defaultdict(_new_acc)

    for rec in _iter_records(log_dir, dates):
        key_date = str(rec.get("ts", ""))[:10]
        for acc in (overall, by_date[key_date], by_purpose[str(rec.get("purpose", "unknown"))], by_model[str(rec.get("model", ""))]):
            _accumulate(acc, rec)

    scope = (
        "全部日期" if dates is None
        else (", ".join(sorted(dates)) if args.date else f"最近 {args.days} 天")
    )
    print("=" * 78)
    print("  LLM 用量账本")
    print(f"  目录：{log_dir}")
    print(f"  区间：{scope}")
    print("=" * 78)
    if not overall["calls"]:
        print("  （无记录）")
        return 0

    print(f"\n总览：{_fmt(overall)}")
    if args.price_in is not None or args.price_out is not None:
        cost = _cost(overall, args.price_in or 0.0, args.price_out or 0.0)
        print(f"  预估成本：{cost:.4f} 元（输入 {args.price_in or 0} / 输出 {args.price_out or 0} 元每百万token）")

    print("\n按日期：")
    for k in sorted(by_date):
        print(f"  {k}  {_fmt(by_date[k])}")

    print("\n按用途（按合计 token 降序）：")
    for k in sorted(by_purpose, key=lambda x: by_purpose[x]["total"], reverse=True):
        print(f"  {k:<36} {_fmt(by_purpose[k])}")

    print("\n按模型：")
    for k in sorted(by_model, key=lambda x: by_model[x]["total"], reverse=True):
        print(f"  {k:<20} {_fmt(by_model[k])}")

    if args.json_out:
        payload = {
            "log_dir": str(log_dir),
            "scope": scope,
            "overall": overall,
            "by_date": by_date,
            "by_purpose": by_purpose,
            "by_model": by_model,
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出：{args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
