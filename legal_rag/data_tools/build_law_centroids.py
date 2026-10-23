# -*- coding: utf-8 -*-
"""预计算法规级质心（median-of-sampled-20）与主题分布，写入 legal_law_centroids 索引，供 /similar 使用。

计算逻辑与 SimilarLawService._compute_representative_vector 严格等价，推荐结果与运行时计算一致。
"""
import argparse
import time

from repositories.elasticsearch import ElasticsearchRepository
from services.similar_law_service import SimilarLawService


def main():
    parser = argparse.ArgumentParser(description="回填法规质心索引 legal_law_centroids")
    parser.add_argument("--force", action="store_true", help="忽略已有质心，全量重算")
    parser.add_argument("--law-id", type=str, default=None, help="仅处理单个法规")
    parser.add_argument("--limit", type=int, default=None, help="最多处理 N 部法规（pilot 用）")
    args = parser.parse_args()

    repo = ElasticsearchRepository()
    service = SimilarLawService()
    repo.ensure_centroid_index()

    if args.law_id:
        laws = [{"law_id": args.law_id, "chunk_count": repo.get_law_chunk_count(args.law_id)}]
    else:
        laws = repo.get_all_laws()
        laws.sort(key=lambda x: x.get("law_id", ""))
        if args.limit:
            laws = laws[: args.limit]

    total = len(laws)
    built = skipped = failed = no_data = 0
    started = time.time()

    for i, law in enumerate(laws, 1):
        law_id = law.get("law_id", "")
        chunk_count = law.get("chunk_count", 0)
        if not law_id:
            continue

        if not args.force:
            existing = repo.get_law_centroid(law_id)
            if existing and existing.get("centroid") and existing.get("chunk_count") == chunk_count:
                skipped += 1
                continue

        t0 = time.time()
        data = service.compute_law_centroid(law_id)
        if not data:
            no_data += 1
            print(f"[{i}/{total}] 跳过 {law_id}：无有效向量数据")
            continue

        if repo.save_law_centroid(law_id, **data):
            built += 1
            print(f"[{i}/{total}] 完成 {law_id} | chunks={data['chunk_count']} | {time.time() - t0:.2f}s")
        else:
            failed += 1
            print(f"[{i}/{total}] 失败 {law_id}：写入质心索引失败")

    elapsed = time.time() - started
    print("=" * 60)
    print(
        f"质心回填完成：total={total} built={built} skipped={skipped} "
        f"no_data={no_data} failed={failed} elapsed={elapsed:.1f}s"
    )
    try:
        count = repo.client.count(index=repo.CENTROID_INDEX).get("count")
        print(f"质心索引当前文档数: {count}")
    except Exception as e:
        print(f"统计质心索引文档数失败: {e}")


if __name__ == "__main__":
    main()
