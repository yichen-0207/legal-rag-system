"""
法律实体消歧服务 v1.0
----------------------
在 LLM 抽取出的三元组之上做实体归一，解决同一实体在不同法域、不同语言、
不同字形下被当作不同实体的问题（例如「个人数据」/「個人資料」/「personal data」）。

分层设计（成本由低到高，逐层独立交付）：
    L1 规范化   ：NFKC（全角转半角）+ 繁转简 + 小写 + 去空白标点
    L2 别名表   ：登记法律领域已知的同义/跨语言写法，直接映射到规范名
    L3 向量对齐 ：bge-m3 跨语言向量相似度（预留，需先积累实体样本再灰度）
    L4 类型约束 ：只在相同实体类型内合并，避免「法律主体」被并进「法律概念」

关于 L1 的覆盖范围：
    项目内的繁简字表（repositories.elasticsearch.zh_t2s）是为「法规名称匹配」
    整理的高频字表，未覆盖全部繁体字。此处在其之上叠加法律实体名常用的补充字表，
    残留的字形差异由 L2 别名表显式登记 —— 法律领域的核心实体数量有限
    （数十个），这正是 L2 存在的原因；长尾差异留给 L3 向量对齐。

重要约束：
    Article（条款）是「法规内局部实体」：A 法规的「条款 1」与 B 法规的
    「条款 1」不是同一个东西，因此其规范 ID 必须带 law_id，禁止跨法规合并。
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

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
    """基于规则（L1/L2/L4）的法律实体消歧服务"""

    def __init__(self, repo: Optional[ElasticsearchRepository] = None):
        self.repo = repo or ElasticsearchRepository()

    @staticmethod
    def resolve(name: str, etype: str, law_id: str = "") -> Tuple[str, str]:
        """
        解析单个实体，返回 (规范 ID, 规范展示名)。

        - Article 类型：规范 ID 带 law_id，跨法规不合并；
        - 命中 L2 别名表：归并到规范名；
        - 未命中：以自身归一化结果作为规范名（等价于「保持独立」）。
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

    def canonicalize(
        self,
        triples: List[Dict],
        law_id: str = "",
        jurisdiction: str = "",
    ) -> Dict:
        """
        对一批三元组做实体归一，并汇总出需要写入实体词典的条目。

        会就地补写每个实体的 canonical_id（不修改 head/tail 的原始 id/name，
        原始抽取结果始终保留，便于回溯与效果对比）。

        Returns:
            {
              "triples": 三元组列表（已补 canonical_id）,
              "entities": 实体词典条目列表,
              "stats": {"raw_entities": 归一前实体数, "merged_entities": 归一后实体数}
            }
        """
        entities: Dict[str, Dict] = {}
        raw_ids = set()

        for t in triples:
            for side in ("head", "tail"):
                ent = t.get(side)
                if not isinstance(ent, dict):
                    continue
                raw_ids.add(ent.get("id", ""))

                etype = ent.get("type", "Concept")
                canonical_id, display = self.resolve(ent.get("name", ""), etype, law_id)
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

        return {
            "triples": triples,
            "entities": list(entities.values()),
            "stats": {
                "raw_entities": len(raw_ids),
                "merged_entities": len(entities),
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
    """别名表统计信息（供管理端展示消歧规则的覆盖范围）"""
    return {
        "groups": len(_ALIAS_GROUPS),
        "aliases": len(_ALIAS_LOOKUP),
        "types": sorted({etype for etype, _ in _ALIAS_LOOKUP}),
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