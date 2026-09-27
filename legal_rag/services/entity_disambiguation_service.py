"""
法律实体消歧服务 v1.0
----------------------
在 LLM 抽取出的三元组之上做实体归一，解决同一实体在不同法域、不同语言、
不同字形下被当作不同实体的问题（例如「个人数据」/「個人資料」/「personal data」）。

分层设计（成本由低到高，逐层独立交付）：
    L1 规范化   ：NFKC（全角转半角）+ 繁转简 + 小写 + 去空白标点
    L2 别名表   ：登记法律领域已知的同义/跨语言写法，直接映射到规范名
    L3 向量对齐 ：bge-m3 跨语言向量相似度，兜住 L1 字表未覆盖的异体与措辞差异
    L4 类型约束 ：只在相同实体类型内合并，避免「法律主体」被并进「法律概念」

关于 L1 的覆盖范围：
    项目内的繁简字表（repositories.elasticsearch.zh_t2s）是为「法规名称匹配」
    整理的高频字表，未覆盖全部繁体字。此处在其之上叠加法律实体名常用的补充字表，
    残留的字形差异由 L2 别名表显式登记 —— 法律领域的核心实体数量有限
    （数十个），这正是 L2 存在的原因；长尾差异留给 L3 向量对齐。

关于 L3 的定位（基于实测证据，重要）：
    L3 是「变体兜底」，不是「语义聚类」。实测 378 个真实实体、17323 个同类型候选对后
    确认：bge-m3 无法区分「语义相近但法律含义不同」的名称 —— 「最高一年徒刑或一百二十日
    罰金」与「最高二年徒刑或二百四十日罰金」余弦 0.933、「總體規劃」与「詳細規劃」0.929，
    纯阈值方案（0.90）会产生至少 6 对错误合并，这在法律知识图谱里是实质性错误。
    因此 L3 采用「向量相似度 + 三重守卫」，且守卫才是判别主力：
        · 数字一致   ：名称中的数字承载「量」（刑期/金额/期限），不同则必然不同实体
        · 编辑距离   ：归一化后差异必须 <= entity_align_max_edit，区分变体与语义差异
        · 长度上限   ：超长「描述型串」不参与，其表面相似度虚高而含义各异
    实测该组合（阈值 0.95 + 数字一致 + 编辑距离<=1 + 长度<=40）在真实词典 378 个实体上
    仅触发 2 对合并，全部人工核验为纯词缀變体（前缀「處」、助词「的」），可接受；
    相应地，召回面很窄 —— 这是有意的：法律实体名上「宁可漏并，不可误并」。

关于比对范围（两轮）：
    第 1 轮「与词典比对」：把新实体归拢到词典里既有的规范实体；
    第 2 轮「批次内部比对」：一对變体若在**同一批次里首次共同出现**、词典里都还没有
    条目，第 1 轮会因两侧都无处可归而双双落库、形成新的重复实体 —— 这正是历史遗留
    重复的来源。第 2 轮专堵这个复发源，复用同一套守卫，可整体关闭
    （entity_align_intra_batch）。

重要约束：
    Article（条款）是「法规内局部实体」：A 法规的「条款 1」与 B 法规的
    「条款 1」不是同一个东西，因此其规范 ID 必须带 law_id，禁止跨法规合并。
    L3 同样跳过 Article —— 条款号靠 L1 归一即可，向量对齐对它只会有害。
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.config import settings
from repositories.elasticsearch import ElasticsearchRepository, zh_t2s

logger = logging.getLogger(__name__)

# L1：实体名归一化时需要丢弃的字符（空白、分隔符、常见标点）
_IGNORED_CHARS_RE = re.compile(r"[\s\-_·・,，.。;；:：、/\\|()（）\[\]【】{}｛｝\"'“”‘’!！?？]+")

# L1：法律实体名常用的繁体字补充表（检索层字表未覆盖的部分）
_T2S_SUPPLEMENT = {
    '當': '当', '義': '义', '責': '责', '會': '会', '轉': '转', '國': '国',
    '動': '动', '對': '对', '絕': '绝', '議': '议', '滅': '灭', '刪': '删',
    '覽': '览', '詢': '询', '問': '问', '徵': '征', '蒐': '搜', '畫': '画',
    '評': '评', '風': '风', '衝': '冲', '損': '损', '賠': '赔', '償': '偿',
    '帶': '带', '洩': '泄', '遺': '遗', '擴': '扩', '屬': '属', '於': '于',
    '與': '与', '為': '为', '從': '从', '違': '违', '項': '项', '許': '许',
    '課': '课', '補': '补', '檢': '检', '察': '察', '憲': '宪', '環': '环',
    '態': '态', '廢': '废', '棄': '弃', '質': '质', '個': '个', '戶': '户',
    '訓': '训', '釋': '释', '費': '费', '稅': '税', '亞': '亚', '來': '来',
    '東': '东', '門': '门', '臺': '台', '灣': '湾', '馬': '马', '韓': '韩',
    '縣': '县', '詮': '诠', '島': '岛', '陸': '陆', '陽': '阳', '陰': '阴',
    '廣': '广', '諮': '咨', '傾': '倾', '猶': '犹',
}

# L2：别名表。每项为 (实体类型, 规范名, 别名列表)
# 别名可包含繁体、异体与英文写法；命中任一别名即归并到规范名。
# 注意：同一别名不得出现在同类型的两个组中，否则归并结果取决于遍历顺序。
_ALIAS_GROUPS: List[Tuple[str, str, List[str]]] = [
    # ---- 法律主体 ----
    ("LegalSubject", "数据主体", [
        "数据主体", "資料當事人", "资料当事人", "資料主體", "数据当事人",
        "当事人", "信息主体", "个人", "個人", "data subject",
    ]),
    ("LegalSubject", "数据控制者", [
        "数据控制者", "資料控制者", "控制者", "控制人", "数据控制人",
        "控权者", "負責實體", "负责实体", "data controller", "controller",
    ]),
    ("LegalSubject", "数据处理者", [
        "数据处理者", "資料處理者", "处理者", "受託處理者", "受托处理者",
        "data processor", "processor",
    ]),
    ("LegalSubject", "监管机构", [
        "监管机构", "監管機構", "监管机关", "主管机关", "主管機關", "监督机构",
        "个人信息保护机构", "個資保護機關", "公署", "专员", "專員",
        "supervisory authority", "data protection authority",
    ]),
    ("LegalSubject", "接收方", [
        "接收方", "接收者", "受让方", "受讓方", "第三方", "第三者", "recipient",
    ]),

    # ---- 法律义务 ----
    ("Obligation", "通知义务", [
        "通知义务", "通知義務", "告知义务", "告知義務", "通知责任",
        "notification obligation", "通知",
    ]),
    ("Obligation", "取得同意", [
        "取得同意", "获得同意", "獲得同意", "征求同意", "徵求同意",
        "經同意", "obtain consent", "取得当事人同意",
    ]),
    ("Obligation", "安全保障义务", [
        "安全保障义务", "安全維護義務", "安全维护义务", "安全措施义务",
        "安全保護義務", "security obligation",
    ]),
    ("Obligation", "配合义务", [
        "配合义务", "配合義務", "协助义务", "協助義務", "提供协助",
    ]),

    # ---- 法律权利 ----
    ("Right", "访问权", [
        "访问权", "查閱權", "查阅权", "存取權", "查询权", "資訊訪問權",
        "right of access", "access right",
    ]),
    ("Right", "更正权", [
        "更正权", "更正權", "修正权", "補充權", "right to rectification",
    ]),
    ("Right", "删除权", [
        "删除权", "刪除權", "被遗忘权", "被遺忘權", "消除权",
        "right to erasure", "right to be forgotten",
    ]),
    ("Right", "反对权", [
        "反对权", "反對權", "异议权", "異議權", "right to object",
    ]),

    # ---- 处罚措施 ----
    ("Penalty", "罚款", [
        "罚款", "罰款", "罚金", "罰金", "逾期罚款", "行政罚款", "fine", "penalty",
    ]),
    ("Penalty", "刑事责任", [
        "刑事责任", "刑事責任", "刑罰", "刑罚", "监禁", "監禁",
        "criminal liability", "有期徒刑",
    ]),
    ("Penalty", "行政处罚", [
        "行政处罚", "行政處罰", "行政制裁", "限期改正", "administrative sanction",
    ]),

    # ---- 法律概念 ----
    ("Concept", "个人数据", [
        "个人数据", "個人資料", "个人资料", "个人信息", "個人信息",
        "个人资讯", "個人資訊", "個資", "personal data", "personal information",
    ]),
    ("Concept", "跨境传输", [
        "跨境传输", "跨境傳輸", "跨境转移", "跨境轉移", "跨境提供", "跨境流动",
        "数据出境", "資料出境", "转移至境外", "cross-border transfer",
    ]),
    ("Concept", "同意", [
        "同意", "同意書", "同意书", "明示同意", "知情同意", "consent",
    ]),
    ("Concept", "数据泄露", [
        "数据泄露", "資料洩露", "数据外泄", "資料外洩", "个资外泄", "個資外洩",
        "洩漏", "洩露", "data breach",
    ]),
    ("Concept", "国家安全", [
        "国家安全", "國家安全", "national security", "公共安全",
    ]),
    ("Concept", "公共利益", [
        "公共利益", "public interest", "社会公共利益", "社會公共利益",
    ]),
    ("Concept", "自动化决策", [
        "自动化决策", "自動化決策", "自动化决定", "用户画像", "用戶畫像",
        "profiling", "automated decision",
    ]),
    ("Concept", "数据主体权利", [
        "数据主体权利", "資料當事人權利", "当事人权利", "當事人權利", "数据权利",
    ]),

    # ---- 例外情形 ----
    ("Exception", "国家安全例外", [
        "国家安全例外", "國家安全例外", "national security exception",
    ]),
    ("Exception", "公共利益例外", [
        "公共利益例外", "public interest exception",
    ]),
    ("Exception", "法定义务例外", [
        "法定义务例外", "法定義務例外", "遵守法定义务", "法律义务例外",
    ]),
]

# (实体类型, 归一化别名) -> 规范名
_ALIAS_LOOKUP: Dict[Tuple[str, str], str] = {}


def _to_simplified(text: str) -> str:
    """繁转简：先走检索层的法规用字表，再叠加实体名补充字表"""
    return "".join(_T2S_SUPPLEMENT.get(ch, ch) for ch in zh_t2s(text))


def normalize_name(name: str) -> str:
    """
    L1 规范化：NFKC（全角转半角）→ 繁转简 → 小写 → 去空白与标点。

    该函数结果用于「判断两个实体名是否等价」，不用于展示。
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", name)
    text = _to_simplified(text)
    text = text.lower()
    text = _IGNORED_CHARS_RE.sub("", text)
    return text.strip()


# ====================================================================
#  L3 守卫（向量对齐的判别主力，详见模块 docstring）
# ====================================================================

# 数字 token：阿拉伯（含全角）或中文数字串
_NUMERAL_RE = re.compile(r"[0-9０-９]+|[零〇一二三四五六七八九十百千萬万億亿兩两]+")
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
_NUMERAL_T2S = {"兩": "两", "萬": "万", "億": "亿"}


def numeral_signature(name: str) -> str:
    """
    抽取名称中的全部数字 token，归一化后排序拼接，用于「数字一致」守卫。

    法律实体名里的数字承载「量」——刑期（二年/六個月）、金额（二百四十日罰金）、
    期限、条款号。数字不同必然不是同一实体，而向量对这类差异不敏感
    （实测「最高一年徒刑」与「最高二年徒刑」余弦高达 0.933），故用此守卫硬拦。

    返回空串表示名称中无数字，此时该守卫不产生约束（两侧都无数字才相等）。
    """
    tokens = []
    for match in _NUMERAL_RE.finditer(name or ""):
        token = match.group(0).translate(_FULLWIDTH_DIGITS)
        tokens.append("".join(_NUMERAL_T2S.get(ch, ch) for ch in token))
    return "|".join(sorted(tokens))


def edit_distance(a: str, b: str) -> int:
    """
    Levenshtein 距离（滚动数组 DP）。

    用于区分「变体差异」与「语义差异」：变体通常只差 1-2 个字符
    （前缀「處」、助词「的」、繁简表未覆盖的异体字），而语义不同的名称
    差异远大于此（「總體規劃」vs「詳細規劃」）。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,          # 删除
                current[j - 1] + 1,       # 插入
                previous[j - 1] + (ca != cb),  # 替换
            ))
        previous = current
    return previous[-1]


def _build_alias_lookup() -> None:
    """构建别名查表；别名冲突时保留首次登记的组并告警"""
    for etype, canonical, aliases in _ALIAS_GROUPS:
        names = [canonical] + list(aliases)
        for raw in names:
            key = (etype, normalize_name(raw))
            if not key[1]:
                continue
            if key in _ALIAS_LOOKUP and _ALIAS_LOOKUP[key] != canonical:
                logger.warning(
                    f"[消歧] 别名冲突（类型={etype}）: 「{raw}」同时登记为 "
                    f"「{_ALIAS_LOOKUP[key]}」与「{canonical}」，已保留前者"
                )
                continue
            _ALIAS_LOOKUP[key] = canonical


_build_alias_lookup()


class EntityDisambiguationService:
    """基于规则（L1/L2/L4）与向量（L3）的法律实体消歧服务"""

    def __init__(self, repo: Optional[ElasticsearchRepository] = None):
        self.repo = repo or ElasticsearchRepository()
        self._embedding_model = None

    def _get_embedding_model(self):
        """延迟加载共享 embedding 模型（bge-m3，与检索层同一实例，不重复占内存）"""
        if self._embedding_model is None:
            from core.model_loader import get_embedding_model
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @staticmethod
    def resolve(name: str, etype: str, law_id: str = "") -> Tuple[str, str]:
        """
        解析单个实体（L1 规范化 + L2 别名表 + L4 类型约束），返回 (规范 ID, 规范展示名)。

        - Article 类型：规范 ID 带 law_id，跨法规不合并；
        - 命中 L2 别名表：归并到规范名；
        - 未命中：以自身归一化结果作为规范名（等价于「保持独立」），
          是否进一步由 L3 向量对齐归并，由 canonicalize() 决定。

        本方法是纯函数（不读词典、不编码向量），L3 因需读实体词典而放在
        canonicalize() 中作为第二阶段，两者职责分离。
        """
        norm = normalize_name(name)
        if not norm:
            return (f"{etype or 'Concept'}::unknown", name or "unknown")

        if etype == "Article":
            # 条款为法规内局部实体，禁止跨法规合并
            return (f"Article::{law_id}::{norm}", (name or "").strip())

        canonical = _ALIAS_LOOKUP.get((etype, norm))
        if canonical:
            return (f"{etype}::{normalize_name(canonical)}", canonical)
        return (f"{etype}::{norm}", (name or "").strip())

    @staticmethod
    def _guard_reject(name: str, candidate_name: str, similarity: float) -> Optional[str]:
        """
        L3 三重守卫判定：通过返回 None，未通过返回原因（供日志与候选审计）。

        守卫顺序按「代价从低到高、拦截力从强到弱」排列，命中即返回。
        """
        if similarity < settings.entity_align_threshold:
            return "similarity_below_threshold"
        if numeral_signature(name) != numeral_signature(candidate_name):
            return "numeral_mismatch"
        norm_a, norm_b = normalize_name(name), normalize_name(candidate_name)
        if max(len(norm_a), len(norm_b)) > settings.entity_align_max_name_len:
            return "name_too_long"
        if "《" in name or "《" in candidate_name:
            return "title_like_name"
        if edit_distance(norm_a, norm_b) > settings.entity_align_max_edit:
            return "edit_distance_too_large"
        return None

    def _align_keys(
        self,
        keys: Dict[Tuple[str, str], Tuple[str, str]],
    ) -> Dict:
        """
        L3 向量对齐：把 L1/L2 未能归并的实体对齐到实体词典中已有的规范实体。

        只处理「L2 别名表未命中」的实体 —— 已归并的无需再判，且能显著压缩编码量；
        只处理非 Article 类型 —— 条款是法规内局部实体（见模块 docstring）。

        词典快照一次性取回后在内存做同类型矩阵比较，避免逐条 ES 往返。
        任何一步失败都返回空 mapping（等价于不启用 L3），不向调用方抛异常。

        Returns:
            {
              "mapping": {待对齐实体的 canonical_id: 归并后的规范 entity_id
                          （已由 _resolve_alignment 去除互指环与链式映射）},
              "display": {待对齐实体的 canonical_id: 归并后规范实体的展示名},
              "applied": 归并条数,
              "candidates": 相似度较高但被守卫拦下的候选（供人工扩表到 L2）,
              "known_vectors": 词典中已有向量的 entity_id 集合,
              "ready": 词典快照是否成功加载（决定是否需要补算新实体向量）,
            }
        """
        empty = {
            "mapping": {}, "display": {}, "applied": 0,
            "candidates": [], "known_vectors": set(), "ready": False,
        }
        if not settings.entity_align_enabled or not keys:
            return empty

        # 待对齐集合：L2 未命中、非 Article、名称非空
        pending: Dict[str, Dict] = {}
        for (etype, name), (canonical_id, _) in keys.items():
            if etype == "Article" or not name:
                continue
            if _ALIAS_LOOKUP.get((etype, normalize_name(name))):
                continue
            pending[canonical_id] = {"name": name, "type": etype}
        if not pending:
            return empty

        snapshot = self.repo.get_kg_entities_with_vectors(exclude_types=["Article"])
        if snapshot is None:
            logger.warning("[消歧] L3 向量对齐跳过：实体词典读取失败（本次降级为 L1/L2/L4）")
            return empty

        try:
            model = self._get_embedding_model()
        except Exception as e:
            logger.warning(f"[消歧] L3 向量对齐跳过：embedding 模型不可用（{type(e).__name__}）")
            return empty

        known_vectors = {
            d["entity_id"] for d in snapshot
            if d.get("entity_id") and d.get("embedding")
        }

        # 回填：L3 上线前写入的词典条目没有向量，若不补算，它们永远无法被匹配，
        # 向量对齐会「看不到任何候选」而形同虚设。
        missing = [
            d for d in snapshot
            if d.get("entity_id") and d.get("name") and not d.get("embedding")
        ][: settings.entity_align_embed_max]
        if missing:
            try:
                vectors = model.encode(
                    [d["name"] for d in missing],
                    normalize_embeddings=True,
                    batch_size=settings.entity_align_batch_size,
                    show_progress_bar=False,
                )
                # 复用实体词典写入路径完成回写（其内部按 entity_id 合并并保留既有向量）
                backfilled = [
                    {**d, "embedding": [float(x) for x in vec]}
                    for d, vec in zip(missing, vectors)
                ]
                self.repo.merge_kg_entities(backfilled)
                for doc in backfilled:
                    doc.pop("embedding", None)
                for d, vec in zip(missing, vectors):
                    d["embedding"] = [float(x) for x in vec]
                    known_vectors.add(d["entity_id"])
                logger.warning(f"[消歧] L3 已为 {len(missing)} 个历史实体补算向量")
            except Exception as e:
                logger.warning(f"[消歧] L3 词典向量回填失败（本次仍尝试对齐）: {type(e).__name__}")

        # 待对齐实体先编码：批次内部比对不依赖词典，故不能因词典为空而提前返回
        pending_ids = list(pending.keys())
        try:
            query_vectors = model.encode(
                [pending[pid]["name"] for pid in pending_ids],
                normalize_embeddings=True,
                batch_size=settings.entity_align_batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.warning(f"[消歧] L3 待对齐实体编码失败，跳过向量对齐: {type(e).__name__}")
            return {**empty, "known_vectors": known_vectors, "ready": True}

        # 词典按类型分组，只保留有向量的条目。可能为空（首次抽取时词典尚无条目），
        # 此时下面这轮「与词典比对」自然空转，但批次内部比对仍要执行，故不提前返回
        by_type: Dict[str, List[Dict]] = {}
        for d in snapshot:
            if d.get("embedding") and d.get("entity_id"):
                by_type.setdefault(d.get("type", "Concept"), []).append(d)

        # 先收集「两两命中」，最后再统一解析成归并表。
        # 不能边判定边写 mapping：守卫是对称的，若一对變体同时出现在待对齐集合与词典中，
        # 会同时产生 A->B 与 B->A，直接落库等于两个规范 ID 对调（见 _resolve_alignment）。
        accepted: List[Tuple[str, str]] = []
        candidates: List[Dict] = []
        seen_candidates: set = set()

        def record_candidate(
            name: str, etype: str, cand_name: str, cand_id: str,
            score: float, reason: str,
        ) -> None:
            """
            记录「被守卫拦下但相似度不低」的候选，供人工确认后登记进 L2 别名表。

            词典比对与批次内部比对可能对同一对给出同一条记录，故按 (名称, 候选ID) 去重。
            """
            key = (name, cand_id)
            if not cand_id or score < 0.80 or key in seen_candidates:
                return
            seen_candidates.add(key)
            candidates.append({
                "name": name,
                "type": etype,
                "candidate": cand_name,
                "candidate_id": cand_id,
                "similarity": round(score, 4),
                "blocked_by": reason,
            })

        # 代表选取所需的旁证：实体展示名 + 「已积累的别名数」
        names: Dict[str, str] = {}
        alias_counts: Dict[str, int] = {}
        for d in snapshot:
            eid = d.get("entity_id")
            if not eid:
                continue
            names[eid] = d.get("name") or eid
            alias_counts[eid] = len(d.get("aliases") or []) + 1
        dict_ids = set(names)
        for pid, meta in pending.items():
            names.setdefault(pid, meta["name"])

        for etype, docs in by_type.items():
            rows = [i for i, pid in enumerate(pending_ids) if pending[pid]["type"] == etype]
            if not rows:
                continue
            matrix = np.asarray([d["embedding"] for d in docs], dtype=np.float32)
            queries = np.asarray([query_vectors[i] for i in rows], dtype=np.float32)
            if matrix.ndim != 2 or queries.ndim != 2 or matrix.shape[1] != queries.shape[1]:
                logger.warning(f"[消歧] L3 跳过类型 {etype}：向量维度不一致")
                continue
            # 向量均已 L2 归一化，点积即余弦相似度
            similarities = queries @ matrix.T

            for row, pid in zip(range(len(rows)), (pending_ids[i] for i in rows)):
                name = pending[pid]["name"]
                blocked: Optional[Tuple[float, Dict, str]] = None
                for j in np.argsort(-similarities[row])[:5]:
                    cand = docs[int(j)]
                    if cand["entity_id"] == pid:
                        continue  # 命中自身，无需归并
                    score = float(similarities[row, int(j)])
                    reason = self._guard_reject(name, cand.get("name", ""), score)
                    if reason is None:
                        accepted.append((pid, cand["entity_id"]))
                        blocked = None
                        break
                    if blocked is None:
                        blocked = (score, cand, reason)
                # 被守卫拦下但相似度不低的：记录下来供人工确认后登记进 L2 别名表
                if blocked is not None:
                    record_candidate(
                        name, etype, blocked[1].get("name", ""),
                        blocked[1].get("entity_id", ""), blocked[0], blocked[2],
                    )

        # ---- 批次内部两两比对 ----
        # 一对變体若在**同一批次里首次共同出现**、词典里都还没有对应条目，那么上面
        # 那轮「与词典比对」会因两侧都无处可归而双双落库，形成新的重复实体
        #（历史遗留的重复正来源于此）。这里补一轮同类型内部比对，复用同一套守卫，
        # 让两侧都还是新实体时也能归拢到其中更规范的一方（由 _resolve_alignment 定夺）。
        if settings.entity_align_intra_batch:
            all_vectors = np.asarray(query_vectors, dtype=np.float32)
            by_type_pending: Dict[str, List[int]] = {}
            for i, pid in enumerate(pending_ids):
                by_type_pending.setdefault(pending[pid]["type"], []).append(i)
            if all_vectors.ndim == 2:
                for etype, rows in by_type_pending.items():
                    if len(rows) < 2:
                        continue
                    sims = all_vectors[rows] @ all_vectors[rows].T
                    for row, pid in zip(range(len(rows)), (pending_ids[i] for i in rows)):
                        name = pending[pid]["name"]
                        blocked = None
                        for c in np.argsort(-sims[row])[:5]:
                            c = int(c)
                            if c == row:
                                continue  # 自身相似度恒为 1.0
                            other = pending_ids[rows[c]]
                            score = float(sims[row, c])
                            reason = self._guard_reject(name, pending[other]["name"], score)
                            if reason is None:
                                accepted.append((pid, other))
                                blocked = None
                                break
                            if blocked is None:
                                blocked = (score, other, reason)
                        if blocked is not None:
                            record_candidate(
                                name, etype, pending[blocked[1]]["name"],
                                blocked[1], blocked[0], blocked[2],
                            )

        mapping, display = self._resolve_alignment(
            accepted, names, alias_counts, dict_ids, set(pending_ids)
        )

        if mapping:
            logger.warning(
                f"[消歧] L3 向量对齐归并 {len(mapping)} 个实体: "
                + "; ".join(f"「{k}」->「{v}」" for k, v in list(mapping.items())[:10])
            )
        for c in candidates:
            logger.warning(
                f"[消歧] L3 候选被守卫拦下，建议人工确认后登记进 L2 别名表: "
                f"「{c['name']}」~「{c['candidate']}」 相似度={c['similarity']} "
                f"原因={c['blocked_by']}"
            )

        return {
            "mapping": mapping,
            "display": display,
            "applied": len(mapping),
            "candidates": candidates,
            "known_vectors": known_vectors,
            "ready": True,
        }

    @staticmethod
    def _resolve_alignment(
        accepted: List[Tuple[str, str]],
        names: Dict[str, str],
        alias_counts: Dict[str, int],
        dict_ids: set,
        scope: set,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        把「两两命中」解析成无环的最终归并表（并查集），返回 (mapping, display)。

        为什么必须做：守卫是**对称**的（数字/编辑距离/长度判定都与方向无关），
        因此当一对變体（A、B）都已存在于词典、又同时出现在本次待对齐集合中时，
        会同时产出 A->B 与 B->A。若直接落库，`canonicalize` 第 3 步会把两个规范 ID
        对调 —— 实体数不变、归并实际失败。链式命中（A->B、B->C）需要传递闭包才能收敛。

        代表选取规则（**保证确定性**，同一输入两次运行的归并结果必须一致）：
          1. 词典中既有的实体优先 —— 归并目标必须是「已在词典里的」规范实体。
             否则本次批量里的新写法可能反过来吞掉词典里的既有实体，映射退化为空、
             该合并被静默跳过；
          2. 已积累别名多者优先 —— 别名多的那个是被反复归拢的一方，更应作为规范实体；
          3. 名称短者优先 —— 噪声词缀（前缀「處」、助词「的」）只会让名称变长，
             短的那个更接近规范写法；
          4. entity_id 字典序 —— 纯兜底，只为在完全等价时也有确定的唯一解。

        输出阶段遍历**变体簇内的全部节点**，而非只遍历 accepted 的左侧。
        原因：批次内部比对时两侧都是新实体、都不在词典里，规则 1/2 同时打平，
        代表由「名称短者」决定，可能恰好落在左侧。若只遍历左侧，该实体被当作
        「自身即代表」跳过，同伴则永远拿不到映射 —— 合并静默失败、重复照旧产生。
        `scope` 为本次待对齐的规范 ID 集合，用于把输出收敛到本批次涉及的实体上，
        避免把「只在词典里、本批次未出现」的实体也写进 mapping 虚增归并数。
        """
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # 路径压缩
                parent[x], x = root, parent[x]
            return root

        def better(a: str, b: str) -> str:
            known_a, known_b = a in dict_ids, b in dict_ids
            if known_a != known_b:
                return a if known_a else b
            count_a, count_b = alias_counts.get(a, 0), alias_counts.get(b, 0)
            if count_a != count_b:
                return a if count_a > count_b else b
            len_a = len(names.get(a) or a)
            len_b = len(names.get(b) or b)
            if len_a != len_b:
                return a if len_a < len_b else b
            return a if a <= b else b

        for src, dst in accepted:
            root_src, root_dst = find(src), find(dst)
            if root_src == root_dst:
                continue  # 已同属一个变体簇
            winner = better(root_src, root_dst)
            parent[root_dst if winner == root_src else root_src] = winner

        mapping: Dict[str, str] = {}
        display: Dict[str, str] = {}
        # find 会把每个参与判定的节点登记进 parent，这里是簇内全部节点
        for node in list(parent):
            if node not in scope:
                continue
            root = find(node)
            if root != node:  # 自身即代表时无需改判
                mapping[node] = root
                display[node] = names.get(root, root)
        return mapping, display

    def _embed_entities(self, entities: Dict[str, Dict], align: Dict) -> None:
        """
        为非 Article 的规范实体补算 embedding 并挂在实体文档上（由写入路径落库）。

        必要性：向量对齐「匹配已有词典」的前提是词典里有向量。若新入库实体不回写向量，
        词典只能被匹配、不能作为匹配目标，覆盖面无法自增长。
        已有向量的实体不重算（merge_kg_entities 也会优先保留旧向量）。
        """
        if not align.get("ready"):
            return
        known = align.get("known_vectors") or set()
        targets = [
            doc for doc in entities.values()
            if doc["type"] != "Article" and doc["entity_id"] not in known and doc["name"]
        ][: settings.entity_align_embed_max]
        if not targets:
            return
        try:
            vectors = self._get_embedding_model().encode(
                [doc["name"] for doc in targets],
                normalize_embeddings=True,
                batch_size=settings.entity_align_batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.warning(f"[消歧] 新实体向量补算失败（不影响归一结果）: {type(e).__name__}")
            return
        for doc, vec in zip(targets, vectors):
            doc["embedding"] = [float(x) for x in vec]

    def canonicalize(
        self,
        triples: List[Dict],
        law_id: str = "",
        jurisdiction: str = "",
    ) -> Dict:
        """
        对一批三元组做实体归一（L1/L2/L4 + L3），并汇总出需要写入实体词典的条目。

        会就地补写每个实体的 canonical_id（不修改 head/tail 的原始 id/name，
        原始抽取结果始终保留，便于回溯与效果对比）。

        分三步，把「纯函数解析」与「有状态的向量对齐」解耦：
          1. L1/L2/L4 解析出每个 (类型, 名称) 的规范 ID —— 纯函数，无外部依赖；
          2. L3 向量对齐 —— 读实体词典，把 L2 未命中的实体改判到已有规范实体，
             失败静默降级（不影响第 1 步结果）；
          3. 应用结果：写回 canonical_id、汇总实体、为非 Article 规范实体补算向量。

        Returns:
            {
              "triples": 三元组列表（已补 canonical_id）,
              "entities": 实体词典条目列表,
              "stats": {
                  "raw_entities": 归一前实体数,
                  "merged_entities": 归一后实体数,
                  "vector_aligned": L3 归并条数,
                  "vector_candidates": 被守卫拦下、建议人工扩表条数,
              }
            }
        """
        keys: Dict[Tuple[str, str], Tuple[str, str]] = {}
        raw_ids = set()

        # ---- 第 1 步：L1 规范化 + L2 别名表 + L4 类型约束 ----
        for t in triples:
            for side in ("head", "tail"):
                ent = t.get(side)
                if not isinstance(ent, dict):
                    continue
                raw_ids.add(ent.get("id", ""))

                etype = ent.get("type", "Concept")
                key = (etype, ent.get("name", ""))
                if key not in keys:
                    keys[key] = self.resolve(key[1], key[0], law_id)

        # ---- 第 2 步：L3 向量对齐 ----
        # keys 的键是原始 (类型, 名称)，值是可被改写的规范 ID，故对齐结果直接改 keys，
        # 后续第 3 步无需再做二次映射。
        align = {
            "mapping": {}, "display": {}, "applied": 0,
            "candidates": [], "known_vectors": set(), "ready": False,
        }
        if settings.entity_align_enabled:
            try:
                align = self._align_keys(keys)
            except Exception as e:
                logger.warning(f"[消歧] L3 向量对齐异常，本次降级为 L1/L2/L4: {type(e).__name__}: {e}")

        for key, (canonical_id, _) in list(keys.items()):
            hit_id = align["mapping"].get(canonical_id)
            if hit_id:
                keys[key] = (hit_id, align["display"].get(canonical_id) or canonical_id)

        # ---- 第 3 步：写回 canonical_id 并汇总实体词典条目 ----
        entities: Dict[str, Dict] = {}
        for t in triples:
            for side in ("head", "tail"):
                ent = t.get(side)
                if not isinstance(ent, dict):
                    continue
                etype = ent.get("type", "Concept")
                canonical_id, display = keys[(etype, ent.get("name", ""))]
                ent["canonical_id"] = canonical_id

                doc = entities.setdefault(canonical_id, {
                    "entity_id": canonical_id,
                    "name": display,
                    "type": etype,
                    "aliases": [],
                    "jurisdictions": [],
                })
                alias = (ent.get("name") or "").strip()
                if alias and alias not in doc["aliases"]:
                    doc["aliases"].append(alias)
                if jurisdiction and jurisdiction not in doc["jurisdictions"]:
                    doc["jurisdictions"].append(jurisdiction)

        self._embed_entities(entities, align)

        return {
            "triples": triples,
            "entities": list(entities.values()),
            "stats": {
                "raw_entities": len(raw_ids),
                "merged_entities": len(entities),
                "vector_aligned": align["applied"],
                "vector_candidates": len(align["candidates"]),
            },
        }

    def canonicalize_and_index(
        self,
        triples: List[Dict],
        law_id: str = "",
        jurisdiction: str = "",
    ) -> Dict:
        """归一三元组并把得到的实体写入实体词典，返回 canonicalize() 的结果"""
        result = self.canonicalize(triples, law_id, jurisdiction)
        saved = self.repo.merge_kg_entities(result["entities"])
        if saved < 0:
            logger.warning(f"[消歧] 实体词典写入失败，本次仅返回归一结果: law_id={law_id}")
        return result


def get_alias_statistics() -> Dict:
    """消歧规则统计信息（供管理端展示 L1~L4 各层的覆盖范围与生效参数）"""
    return {
        "groups": len(_ALIAS_GROUPS),
        "aliases": len(_ALIAS_LOOKUP),
        "types": sorted({etype for etype, _ in _ALIAS_LOOKUP}),
        "vector_alignment": {
            "enabled": settings.entity_align_enabled,
            "threshold": settings.entity_align_threshold,
            "max_edit_distance": settings.entity_align_max_edit,
            "max_name_length": settings.entity_align_max_name_len,
            "numeral_guard": True,
            "intra_batch": settings.entity_align_intra_batch,
        },
    }


def find_alias_group(name: str, etype: Optional[str] = None) -> Optional[Sequence[str]]:
    """查询某写法所属的别名组（调试/验证用）：返回该组全部写法"""
    norm = normalize_name(name)
    for group_type, canonical, aliases in _ALIAS_GROUPS:
        if etype and group_type != etype:
            continue
        if norm in {normalize_name(a) for a in [canonical] + list(aliases)}:
            return [canonical] + list(aliases)
    return None