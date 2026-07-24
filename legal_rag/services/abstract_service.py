import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI
from elasticsearch import Elasticsearch
from core.config import settings

INDEX_NAME = settings.es_index_name


class AbstractService:
    MAX_INPUT_CHARS = 40000

    def __init__(self):
        self.client = self._init_es_client()
        self.llm_client = self._init_llm_client()
        self.model = settings.llm_deepseek_model or "deepseek-v4-flash"

    def _init_es_client(self):
        client_kwargs = {
            "hosts": settings.es_hosts,
            "verify_certs": settings.es_verify_certs,
            "request_timeout": settings.es_request_timeout,
            "headers": {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "compatible-with": "8"
            }
        }

        if settings.es_user and settings.es_password:
            client_kwargs["basic_auth"] = (settings.es_user, settings.es_password)

        return Elasticsearch(**client_kwargs)

    def _init_llm_client(self):
        base_url = settings.llm_deepseek_api_base_url or "https://api.deepseek.com/v1"
        api_key = settings.llm_deepseek_api_key or ""

        if not api_key:
            raise ValueError("DeepSeek API Key 未配置，请设置 settings.llm_deepseek_api_key")

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
            max_retries=2
        )

    def _build_abstract_prompt(self, full_text: str, metadata: Dict) -> str:
        jurisdiction = metadata.get("jurisdiction", "")
        title = metadata.get("title", "")
        passing_date = metadata.get("passing_date", "")

        return f"""请阅读以下法律文本，用中文生成一段150-300字的简明摘要，包含立法目的、核心内容和适用范围。

## 法律信息
- 名称：{title}
- 法域：{jurisdiction}
- 通过日期：{passing_date}

## 要求
1. 用一段话概括，不要分段
2. 字数控制在150-300字之间
3. 语言简洁明了，突出重点
4. 不要输出任何格式标记或标题

## 法律文本：
{full_text}"""

    def _get_law_full_text(self, law_id: str) -> tuple:
        query = {
            "query": {"term": {"law_id": law_id}},
            "sort": [{"chunk_index": "asc"}],
            "size": 1000
        }

        resp = self.client.search(index=INDEX_NAME, body=query)
        hits = resp["hits"]["hits"]

        if not hits:
            raise ValueError(f"法规不存在: {law_id}")

        full_text = ""
        title = ""
        jurisdiction = ""
        passing_date = ""

        for hit in hits:
            source = hit["_source"]
            article_number = source.get("article_number", "")
            content = source.get("content", "")

            if not title:
                title = source.get("title", "")
                jurisdiction = source.get("jurisdiction", "")
                passing_date = source.get("passing_date", "")

            if article_number:
                full_text += f"{article_number} {content}\n\n"
            else:
                full_text += f"{content}\n\n"

        metadata = {
            "jurisdiction": jurisdiction,
            "passing_date": passing_date,
            "title": title
        }

        return full_text.strip(), metadata

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                    stream=False
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"[AbstractService] {self.model} 调用失败 ({attempt+1}/{max_retries}): {e}, 等待 {wait_time}s 重试...")
                    time.sleep(wait_time)
                else:
                    print(f"[AbstractService] {self.model} 调用失败，已重试 {max_retries} 次: {e}")

        if last_error:
            raise Exception(f"LLM调用失败，{self.model} 重试 {max_retries} 次后仍失败: {last_error}")
        raise Exception("LLM调用失败，所有模型均已尝试")

    def _check_abstract_exists(self, law_id: str) -> bool:
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"law_id": law_id}},
                        {"exists": {"field": "abstract.text"}}
                    ]
                }
            },
            "size": 1
        }

        try:
            resp = self.client.search(index=INDEX_NAME, body=query)
            return resp["hits"]["total"]["value"] > 0
        except Exception:
            return False

    def _update_law_abstract(self, law_id: str, abstract_text: str) -> bool:
        update_query = {
            "query": {"term": {"law_id": law_id}},
            "script": {
                "source": "ctx._source.abstract = params.abstract",
                "params": {
                    "abstract": {
                        "text": abstract_text,
                        "generated_at": datetime.now().isoformat(),
                        "model": self.model,
                        "version": "1.0",
                        "language": "zh"
                    }
                }
            }
        }

        try:
            resp = self.client.update_by_query(
                index=INDEX_NAME,
                body=update_query,
                refresh=True
            )
            updated = resp.get("updated", 0)
            print(f"[AbstractService] 已更新 {law_id} 的摘要，更新 {updated} 条文档")
            return updated > 0
        except Exception as e:
            print(f"[AbstractService] 更新 {law_id} 摘要失败: {e}")
            return False

    def generate_for_law(self, law_id: str, force: bool = False) -> bool:
        """为单个法规生成摘要

        Args:
            law_id: 法规ID
            force: 是否强制重新生成（忽略已存在的摘要）

        Returns:
            bool: 是否成功
        """
        try:
            print(f"[AbstractService] 开始生成摘要: {law_id}")

            if not force and self._check_abstract_exists(law_id):
                print(f"[AbstractService] 跳过 {law_id}（已有摘要）")
                return True

            full_text, metadata = self._get_law_full_text(law_id)

            if len(full_text) > self.MAX_INPUT_CHARS:
                full_text = full_text[:self.MAX_INPUT_CHARS]
                print(f"[AbstractService] {law_id} 文本过长，已截断至 {self.MAX_INPUT_CHARS} 字符")

            prompt = self._build_abstract_prompt(full_text, metadata)
            abstract_text = self._call_llm_with_retry(prompt)

            abstract_text = abstract_text.strip()
            if not abstract_text:
                raise ValueError("生成的摘要为空")

            print(f"[AbstractService] {law_id} 生成摘要: {abstract_text[:60]}...")

            if not self._update_law_abstract(law_id, abstract_text):
                raise ValueError("更新摘要失败")

            print(f"[AbstractService] {law_id} 摘要生成成功")
            return True

        except Exception as e:
            print(f"[AbstractService] ❌ {law_id} 摘要生成失败: {e}")
            return False

    def generate_for_laws(self, law_ids: List[str], delay: float = 2.0) -> Dict[str, bool]:
        """批量为多个法规生成摘要

        Args:
            law_ids: 法规ID列表
            delay: 每次调用之间的延迟（秒），用于避免LLM限流

        Returns:
            Dict: 每个法规ID对应的生成结果（True/False）
        """
        results = {}
        success_count = 0
        failed_count = 0
        failed_laws = []

        print(f"[AbstractService] 开始批量生成 {len(law_ids)} 个法规的摘要")

        for i, law_id in enumerate(law_ids, 1):
            print(f"\n[{i}/{len(law_ids)}] 处理: {law_id}")
            try:
                success = self.generate_for_law(law_id)
                results[law_id] = success
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_laws.append(law_id)
            except Exception as e:
                print(f"[AbstractService] ❌ {law_id} 处理异常: {e}")
                results[law_id] = False
                failed_count += 1
                failed_laws.append(law_id)

            if i < len(law_ids):
                time.sleep(delay)

        print(f"\n[AbstractService] 批量生成完成: 成功 {success_count}, 失败 {failed_count}")

        if failed_laws:
            log_path = os.path.join(
                os.path.dirname(__file__),
                f"../logs/abstract_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(failed_laws, f, ensure_ascii=False, indent=2)
            print(f"[AbstractService] 失败日志已保存至: {log_path}")

        return results