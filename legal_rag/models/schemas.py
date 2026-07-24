from pydantic import BaseModel
from typing import Generic, TypeVar, Optional, List, Dict

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    message: str = ""
    code: int = 200


class DocumentMetadata(BaseModel):
    law_id: str
    jurisdiction: str
    title: str
    article_number: Optional[str] = None
    chunk_index: int
    passing_date: Optional[str] = None
    law_number: Optional[str] = None
    publisher: Optional[str] = None
    source_file: Optional[str] = None
    topics: List[str] = []
    topic_labels: List[str] = []


class SearchResult(BaseModel):
    content: str
    metadata: DocumentMetadata
    similarity: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    jurisdiction: Optional[str] = None


class QARequest(BaseModel):
    question: str
    top_k: int = 5
    jurisdictions: Optional[List[str]] = None


class QAAnswer(BaseModel):
    answer: str
    sources: List[SearchResult]
    retrieved_count: int


class DocumentCreate(BaseModel):
    content: str
    metadata: DocumentMetadata


class DocumentUpdate(BaseModel):
    content: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None


class MatchedChunk(BaseModel):
    article_number: Optional[str] = None
    content_preview: str = ""
    similarity: float = 0.0


class SimilarLawItem(BaseModel):
    law_id: str
    title: str
    jurisdiction: str
    similarity: float
    match_chunk_count: int = 0
    top_matched_chunks: List[MatchedChunk] = []


class SimilarLawResponse(BaseModel):
    query_law_id: str
    query_law_title: Optional[str] = ""
    query_jurisdiction: Optional[str] = ""
    query_chunk_count: int = 0
    similar_laws: List[SimilarLawItem] = []
    total_found: int = 0
    returned_count: int = 0
    message: str = ""


class SimilarChunkItem(BaseModel):
    chunk_id: str
    law_id: str
    title: str
    jurisdiction: str
    article_number: Optional[str] = None
    content_preview: str = ""
    similarity: float = 0.0


class SimilarChunkResponse(BaseModel):
    query_chunk_id: str
    query_content: str = ""
    query_article_number: Optional[str] = None
    query_law_id: str = ""
    similar_chunks: List[SimilarChunkItem] = []
    total_found: int = 0
    returned_count: int = 0
    message: str = ""


# ========== 法规对比相关模型 ==========

class CompareClauseItem(BaseModel):
    """对比中的单条条款"""
    law_id: str = ""
    title: str = ""
    article_number: Optional[str] = None
    content: str = ""
    similarity: float = 0.0
    jurisdiction: str = ""


class ComparisonRow(BaseModel):
    """并排对比的一行（法域A vs 法域B）"""
    index: int
    jurisdiction_a: Optional[CompareClauseItem] = None
    jurisdiction_b: Optional[CompareClauseItem] = None


class ScoreDistribution(BaseModel):
    """相似度分布数据点"""
    jurisdiction: str
    similarity: float


class ComparisonTableResult(BaseModel):
    """LLM生成的对比表格结果"""
    table_markdown: str = ""
    jurisdiction_a: str = ""
    jurisdiction_b: str = ""
    topic: str = ""
    timing: Dict[str, str] = {}
    error: Optional[str] = None


class ClausesComparisonResult(BaseModel):
    """条款检索对比结果"""
    comparison_rows: List[ComparisonRow] = []
    scores_distribution: List[ScoreDistribution] = []
    jurisdiction_a: str = ""
    jurisdiction_b: str = ""
    topic: str = ""
    count_a: int = 0
    count_b: int = 0
    timing: Dict[str, str] = {}


class AISummaryResult(BaseModel):
    """AI对比总结结果"""
    summary: str = ""
    sources: List[SearchResult] = []
    retrieved_count: int = 0
    timing: Dict[str, str] = {}
    error: Optional[str] = None


class FullCompareResponse(BaseModel):
    """完整对比分析响应（一步到位）"""
    jurisdiction_a: str
    jurisdiction_b: str
    topic: str
    comparison_table: ComparisonTableResult
    clauses_comparison: ClausesComparisonResult
    ai_summary: AISummaryResult
    timing: Dict[str, str] = {}


# ========== 跨法域专题分析相关模型 ==========

class AggregatedLaw(BaseModel):
    """聚合后的单部法规摘要"""
    law_id: str = ""
    title: str = ""
    article_numbers: List[str] = []
    chunk_count: int = 0
    max_similarity: float = 0.0


class RetrievalResult(BaseModel):
    """阶段一：原始检索结果统计"""
    chunks_a_count: int = 0
    chunks_b_count: int = 0
    top_n_per_jurisdiction: int = 8


class AggregatedResult(BaseModel):
    """阶段二：聚合后的法规级结果"""
    laws_a: List[AggregatedLaw] = []
    laws_b: List[AggregatedLaw] = []
    laws_a_count: int = 0
    laws_b_count: int = 0


class TopicAnalysisResponse(BaseModel):
    """
    跨法域专题分析完整响应

    三阶段流水线：
    阶段一：语义向量检索（分别搜索两个法域，各取 Top-N）
    阶段二：按法规聚合去重（law_id 分组 + chunk_index 排序 + 拼接）
    阶段三：LLM 生成结构化对比分析报告（5个固定维度）
    """
    query: str = ""
    jurisdiction_a: str = ""
    jurisdiction_b: str = ""
    retrieval: RetrievalResult = RetrievalResult()
    aggregated: AggregatedResult = AggregatedResult()
    analysis_report: str = ""
    timing: Dict[str, str] = {}
