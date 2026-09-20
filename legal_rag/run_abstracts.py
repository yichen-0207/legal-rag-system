"""独立运行摘要生成脚本 — 在容器内执行: python run_abstracts.py"""
from services.abstract_service import AbstractService
from elasticsearch import Elasticsearch
from core.config import settings

es = Elasticsearch(
    hosts=settings.es_hosts,
    verify_certs=settings.es_verify_certs,
    basic_auth=(settings.es_user, settings.es_password) if settings.es_user else None,
)
resp = es.search(
    index=settings.es_index_name,
    body={"size": 0, "aggs": {"law_ids": {"terms": {"field": "law_id", "size": 500}}}},
)
law_ids = [b["key"] for b in resp["aggregations"]["law_ids"]["buckets"]]
print(f"Found {len(law_ids)} laws to generate abstracts")
print(f"Law IDs: {law_ids[:10]}... (showing first 10)")

svc = AbstractService()
svc.generate_for_laws(law_ids, delay=2.0)
print("Done!")
