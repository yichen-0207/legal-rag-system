"""
图谱多跳推理问答服务 v1.0
------------------------
受限多跳 + 证据约束的 GraphRAG（局部检索）实现：

    问题 → 实体链接（L2 别名命中 + bge-m3 向量 KNN）
         → 以种子实体为中心扩 N 跳子图（复用知识层三元组）
         → 组装「三元组 + 来源条款原文片段」作为证据
         → 1 次 LLM 生成带路径引用的答案

设计取舍：
    - 采用固定跳数（默认 2）而非让 LLM 逐步决策下一跳：成本可控（每次仅 1 次 LLM 调用）、
      结果可复现、幻觉面更小；需要更强推理时可再放开为迭代式 Agent。
    - 证据强约束：prompt 明确要求「只依据给定三元组与条款片段作答」并结构化输出引用，
      使答案可溯源（前端可据此高亮关系路径）。
    - 独立于既有 qa_service，不侵入其 RAG 链路（零回归风险）。
    - 数据范围：仅对已完成知识层抽取的法规有效；知识层无命中时返回 found=False，
      由调用方决定是否回退到普通 RAG 问答。
"""

import json
import logging
import re
from typing import Dict, List, Optional

from openai import OpenAI

from core.config import settings
from repositories.elasticsearch import ElasticsearchRepository
from services.entity_disambiguation_service import (
    EntityDisambiguationService,
    normalize_name,
    _ALIAS_GROUPS,
)

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_EDGES = 60   # 子图规模上限（控制 prompt 长度）
_MAX_EVIDENCE_ARTICLES = 20


class KGReasoningService:
    """基于知识图谱的受限多跳推理问答"""

    def __init__(self, repo: Optional[ElasticsearchRepository] = None):
        self.repo = repo or ElasticsearchRepository()
        self.api_key = settings.llm_api_key or ""
        self.base_url = settings.llm_api_base_url or "https://api.deepseek.com/v1"
        self.model = settings.llm_extraction_model or "deepseek-flash"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    # ================================================================
    #  1. 实体链接：把问题映射到实体词典中的规范实体
    # ================================================================
    def _link_entities(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        两路互补：
          ① L2 别名命中（确定性）：问题里出现已知别名/规范名 → 直接归到规范实体；
          ② bge-m3 向量 KNN（语义）：跨语言/措辞差异的兜底。
        """
        found: Dict[str, Dict] = {}

        # ---- ① L2 别名表命中 ----
        norm_q = normalize_name(question)
        if norm_q:
            for etype, canonical, aliases in _ALIAS_GROUPS:
                hit = False
                for raw in [canonical] + list(aliases):
                    norm_alias = normalize_name(raw)
                    if len(norm_alias) >= 2 and norm_alias in norm_q:
                        hit = True
                        break
                if hit:
                    canonical_id, display = EntityDisambiguationService.resolve(canonical, etype)
                    found.setdefault(canonical_id, {
                        "entity_id": canonical_id, "name": display, "type": etype, "source": "alias",
                    })

        # ---- ② 向量 KNN ----
        try:
            from core.model_loader import get_embedding_model
            vector = get_embedding_model().encode(
                [question], normalize_embeddings=True, show_progress_bar=False
            )[0]
            for hit in self.repo.search_kg_entities_knn(
                [float(x) for x in vector], k=top_k, exclude_types=["Article"]
            ):
                eid = hit.get("entity_id")
                if eid:
                    found.setdefault(eid, {
                        "entity_id": eid, "name": hit.get("name", ""),
                        "type": hit.get("type", ""), "source": "vector",
                        "score": hit.get("score"),
                    })
        except Exception as e:
            logger.warning(f"[KGReasoning] 向量实体链接失败，仅用别名命中: {type(e).__name__}: {e}")

        return list(found.values())[: max(top_k, 3)]

    # ================================================================
    #  2. 子图检索（多跳）
    # ================================================================
    def _collect_subgraph(self, seed_ids: List[str], hops: int = 2) -> List[Dict]:
        """以种子实体为中心扩 hops 跳，返回三元组列表（去重、限量）"""
        seed_set = set(seed_ids)
        collected: Dict[tuple, Dict] = {}

        def _key(t: Dict) -> tuple:
            return (
                t["head"].get("canonical_id") or t["head"].get("id"),
                t.get("relation"),
                t["tail"].get("canonical_id") or t["tail"].get("id"),
                t.get("source_article"),
            )

        def _merge(triples: List[Dict]) -> None:
            for t in triples:
                collected.setdefault(_key(t), t)

        _merge(self.repo.get_kg_triples_by_entities(seed_ids, size=_MAX_EVIDENCE_EDGES * 4))

        if hops >= 2:
            neighbor_ids = set()
            for t in list(collected.values()):
                for side in ("head", "tail"):
                    cid = t[side].get("canonical_id") or t[side].get("id")
                    if cid and cid not in seed_set:
                        neighbor_ids.add(cid)
            if neighbor_ids:
                _merge(self.repo.get_kg_triples_by_entities(
                    list(neighbor_ids)[:200], size=_MAX_EVIDENCE_EDGES * 4
                ))

        triples = sorted(collected.values(), key=lambda t: -float(t.get("confidence", 0)))
        return triples[:_MAX_EVIDENCE_EDGES]

    # ================================================================
    #  3. 证据组装 + LLM 生成
    # ================================================================
    def _build_evidence(self, triples: List[Dict], snippets: Dict[str, str],
                        law_titles: Dict[str, str]) -> str:
        rel_lines = []
        for t in triples:
            head = t["head"].get("name", "")
            tail = t["tail"].get("name", "")
            law = law_titles.get(t.get("law_id", ""), t.get("law_id", ""))
            rel_lines.append(
                f"- ({head}) --{t.get('relation','')}--> ({tail})  ←《{law}》{t.get('source_article','')}"
            )
        evidence = "【知识图谱关系】\n" + "\n".join(rel_lines) if rel_lines else "【知识图谱关系】（无）"

        if snippets:
            art_lines = []
            for ref, content in snippets.items():
                law_id, _, article = ref.partition("::")
                law = law_titles.get(law_id, law_id)
                art_lines.append(f"- 《{law}》{article}：{content}")
            evidence += "\n\n【条款原文摘录】\n" + "\n".join(art_lines)
        return evidence

    def _build_prompt(self, question: str, evidence: str) -> str:
        return (
            "你是跨境法律知识图谱问答助手。请**仅依据下面提供的知识图谱关系与条款原文摘录**回答问题。\n"
            "\n"
            "要求：\n"
            "1. 不得使用给材料之外的知识；材料不足时明确说明「依据现有知识图谱无法确定」。\n"
            "2. answer 用简体中文，条理清晰，可跨法域对比；**answer 内不要使用换行符**（用分号或序号分隔）。\n"
            "3. used_paths 列出你实际依据的图谱关系路径（head/relation/tail/article）。\n"
            "4. cited_articles 列出你引用的条款（格式 law_id::article_number）。\n"
            "\n"
            "请严格输出 JSON（不要 markdown 代码块）：\n"
            "{\n"
            '  "answer": "回答正文",\n'
            '  "used_paths": [{"head":"","relation":"","tail":"","article":""}],\n'
            '  "cited_articles": ["law_id::article_number"]\n'
            "}\n"
            "\n"
            "问题：" + question + "\n\n"
            "材料：\n" + evidence + "\n\n"
            "直接输出 JSON："
        )

    @staticmethod
    def _parse_response(text: str) -> Dict:
        """
        解析 LLM 的 JSON 输出。

        容错要点：模型常在 answer 里直接输出换行，这对 JSON 是非法控制字符，
        标准 json.loads 会失败。故先尝试 strict=True，再尝试 strict=False
        （允许字符串内出现控制字符），最后用正则兜底提取 answer。
        """
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        def _loads(s: str):
            for strict in (True, False):
                try:
                    return json.loads(s, strict=strict)
                except Exception:
                    continue
            return None

        data = _loads(cleaned)
        if not isinstance(data, dict):
            m = re.search(r"\{[\s\S]*\}", cleaned)
            data = _loads(m.group()) if m else None

        if not isinstance(data, dict):
            # 最终兜底：正则提取 answer（保留转义引号的正确处理）
            m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.S)
            answer = text or ""
            if m:
                try:
                    answer = json.loads('"' + m.group(1) + '"')
                except Exception:
                    answer = m.group(1)
            return {"answer": (answer or "").strip(), "used_paths": [], "cited_articles": []}

        return {
            "answer": str(data.get("answer", "")).strip(),
            "used_paths": data.get("used_paths") if isinstance(data.get("used_paths"), list) else [],
            "cited_articles": data.get("cited_articles") if isinstance(data.get("cited_articles"), list) else [],
        }

    def answer(self, question: str, top_k_entities: int = 5, hops: int = 2) -> Dict:
        """
        图谱多跳推理问答主入口。

        Returns:
            {
              "found": bool, "message": str,
              "answer": str, "used_paths": [...], "cited_articles": [...],
              "seed_entities": [...], "subgraph": {"edge_count": n, "articles": [...]},
            }
        """
        question = (question or "").strip()
        if not question:
            return {"found": False, "message": "问题为空", "answer": "", "used_paths": [], "cited_articles": []}

        seeds = self._link_entities(question, top_k_entities)
        if not seeds:
            return {
                "found": False,
                "message": "未能从知识图谱中链接到相关实体（该问题可能不在已抽取的知识范围内）",
                "answer": "", "used_paths": [], "cited_articles": [], "seed_entities": [],
            }

        triples = self._collect_subgraph([s["entity_id"] for s in seeds], hops=hops)
        if not triples:
            return {
                "found": False,
                "message": "已链接到实体，但知识图谱中没有可用关系",
                "answer": "", "used_paths": [], "cited_articles": [], "seed_entities": seeds,
            }

        law_titles = {law.get("law_id"): law.get("title", "") for law in self.repo.get_all_laws()}
        refs = [
            {"law_id": t.get("law_id", ""), "article_number": t.get("source_article", "")}
            for t in triples
        ]
        snippets = self.repo.get_articles_by_refs(refs, max_pairs=_MAX_EVIDENCE_ARTICLES, max_chars=600)

        evidence = self._build_evidence(triples, snippets, law_titles)

        if not self.client:
            return {
                "found": False, "message": "未配置 LLM 密钥，无法生成答案",
                "answer": "", "used_paths": [], "cited_articles": [], "seed_entities": seeds,
            }

        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": self._build_prompt(question, evidence)}],
                "temperature": 0.1,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            }
            extra_body = settings.llm_extra_body()
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = self.client.chat.completions.create(**kwargs)
            parsed = self._parse_response(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"[KGReasoning] LLM 生成失败: {type(e).__name__}: {e}")
            return {
                "found": False, "message": f"生成答案失败: {type(e).__name__}",
                "answer": "", "used_paths": [], "cited_articles": [], "seed_entities": seeds,
            }

        return {
            "found": True,
            "message": "",
            "answer": parsed["answer"],
            "used_paths": parsed["used_paths"],
            "cited_articles": parsed["cited_articles"],
            "seed_entities": seeds,
            "subgraph": {
                "edge_count": len(triples),
                "articles": list(snippets.keys()),
            },
        }
