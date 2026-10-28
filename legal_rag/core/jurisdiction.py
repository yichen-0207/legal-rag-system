import re
from enum import Enum
from typing import Dict, FrozenSet, List, Sequence, Tuple


class Jurisdiction(str, Enum):
    """法域枚举类"""
    MAINLAND_CHINA = "中国大陆"
    HONGKONG = "中国香港"
    MACAU = "中国澳门"
    SINGAPORE = "新加坡"
    TAIWAN = "台湾"
    MALAYSIA = "马来西亚"
    JAPAN = "日本"
    KOREA = "韩国"
    USA = "美国"
    UK = "英国"
    GERMANY = "德国"
    FRANCE = "法国"


# 旧称/英文名 → 规范中文名；数据与接口入参统一走 normalize_jurisdiction
JURISDICTION_ALIASES: Dict[str, str] = {
    "中国": "中国大陆",
    "china": "中国大陆",
    "mainland china": "中国大陆",
    "香港": "中国香港",
    "hongkong": "中国香港",
    "hong kong": "中国香港",
    "hk": "中国香港",
    "澳门": "中国澳门",
    "macau": "中国澳门",
    "macao": "中国澳门",
    "mo": "中国澳门",
    "澳洲": "澳大利亚",
    "australia": "澳大利亚",
    "korea": "韩国",
    "south korea": "韩国",
    "thailand": "泰国",
    "vietnam": "越南",
    "viet nam": "越南",
    "malaysia": "马来西亚",
    "japan": "日本",
    "singapore": "新加坡",
    "sg": "新加坡",
    "usa": "美国",
    "us": "美国",
    "united states": "美国",
    "mac": "中国澳门",
    "tw": "台湾",
    "taiwan": "台湾",
}


# 法域信息映射表：(英文目录名, ID前缀代码)
# 目录名与 ID 前缀必须与 data/raw 下的实际目录、law_id 前缀一致：
# 管理端新增法规时按此表定位目录并生成 law_id，不一致会把同一法域的数据写散到两处
# （hongkong/ 与 hk/、汉名目录与 australia/ 并存）。
JURISDICTION_INFO: Dict[str, Tuple[str, str]] = {
    "中国大陆": ("china", "CN"),
    "中国香港": ("hk", "HK"),
    "中国澳门": ("macau", "MO"),
    "新加坡": ("singapore", "SG"),
    "台湾": ("taiwan", "TW"),
    "马来西亚": ("malaysia", "MY"),
    "日本": ("japan", "JP"),
    "韩国": ("korea", "KR"),
    "泰国": ("thailand", "TH"),
    "越南": ("vietnam", "VN"),
    "澳大利亚": ("australia", "AU"),
    "美国": ("usa", "US"),
    "英国": ("uk", "UK"),
    "德国": ("germany", "DE"),
    "法国": ("france", "FR"),
}


def normalize_jurisdiction(jurisdiction: str) -> str:
    """把旧称或英文名归一为规范中文名；未知值原样返回。"""
    if not jurisdiction:
        return jurisdiction
    name = jurisdiction.strip()
    if name in JURISDICTION_INFO:
        return name
    return JURISDICTION_ALIASES.get(name.lower(), name)


def normalize_jurisdictions(jurisdictions) -> list:
    """批量归一并保序去重；空入参原样返回。

    与 ES 索引、法规质心里的 `jurisdiction` 字段保持同一口径。调用方常用旧称
    （"香港"）或英文名，而索引存的是规范名（"中国香港"），不归一时字面比较会
    整片失配（点名法域过滤失效、子图被滤空）。
    """
    if not jurisdictions:
        return jurisdictions
    normalized: list = []
    for item in jurisdictions:
        name = normalize_jurisdiction(str(item or "").strip())
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def jurisdiction_aliases(jurisdiction: str) -> list:
    """返回规范名及其旧称，供 ES 过滤兼容改名前后的存量数据。"""
    canonical = normalize_jurisdiction(jurisdiction)
    aliases = [canonical]
    for old, new in JURISDICTION_ALIASES.items():
        if new == canonical and old not in aliases:
            aliases.append(old)
    return aliases


def get_jurisdiction_info(jurisdiction: str) -> Tuple[str, str]:
    """返回法域对应的 (英文目录名, ID 前缀代码)。"""
    canonical = normalize_jurisdiction(jurisdiction)
    if canonical in JURISDICTION_INFO:
        return JURISDICTION_INFO[canonical]

    # 仍未匹配时使用默认值
    dir_name = canonical.lower().replace(" ", "_")
    prefix = canonical[:2].upper()[:2] if len(canonical) >= 2 else canonical.upper()
    return (dir_name, prefix)


def get_all_jurisdictions() -> list:
    """获取所有已定义的法域列表"""
    return list(JURISDICTION_INFO.keys())


# ============================================================
# 查询期法域识别：检索 / 智能体 / 专题分析共同依赖的唯一事实源
# ============================================================

# 触发词表：规范法域 → 查询中可能出现的写法。
# 新增法域入库后必须同步维护，缺项时该法域只能靠语义向量兜底，召回精度更低。
RETRIEVAL_TRIGGERS: Dict[str, List[str]] = {
    "中国香港": ["中国香港", "香港", "hongkong", "hong kong", "hk"],
    "中国澳门": ["中国澳门", "澳门", "澳門", "macau", "macao"],
    "中国大陆": ["中国大陆", "mainland china", "china"],
    "新加坡": ["新加坡", "singapore"],
    "马来西亚": ["马来西亚", "malaysia"],
    "日本": ["日本", "japan"],
    "韩国": ["韩国", "韓国", "south korea", "korea"],
    "泰国": ["泰国", "thailand"],
    "越南": ["越南", "vietnam"],
    "澳大利亚": ["澳大利亚", "澳洲", "australia"],
    "美国": ["美国", "united states", "usa"],
}

# 未被香港/澳门/台湾限定的「中国」→ 中国大陆；「中国」不直接入触发词表，
# 否则「中国香港」会被同时判为中国大陆。
CHINA_STANDALONE_RE = re.compile(r"中国(?!香港|澳门|澳門|台湾|臺灣)")

# 区域 → 法域清单（跨法域扇出）
REGION_MAP: Dict[str, List[str]] = {
    "亚太地区": ["中国香港", "日本", "韩国", "新加坡", "马来西亚", "泰国",
                 "越南", "中国澳门", "中国大陆", "澳大利亚"],
    "亚太": ["中国香港", "日本", "韩国", "新加坡", "马来西亚", "泰国",
             "越南", "中国澳门", "中国大陆", "澳大利亚"],
    "东南亚各国": ["新加坡", "马来西亚", "泰国", "越南"],
    "东南亚": ["新加坡", "马来西亚", "泰国", "越南"],
    "东亚": ["日本", "韩国", "中国香港", "中国澳门", "中国大陆"],
    "亚洲各国": ["日本", "韩国", "新加坡", "马来西亚", "泰国", "越南",
                 "中国香港", "中国澳门", "中国大陆"],
    "亚洲主要国家": ["日本", "韩国", "新加坡", "马来西亚", "泰国", "越南",
                     "中国香港", "中国澳门", "中国大陆"],
    "亚洲各司法管辖区": ["日本", "韩国", "新加坡", "马来西亚", "泰国", "越南",
                         "中国香港", "中国澳门", "中国大陆"],
    "各国": ["中国香港", "日本", "韩国", "新加坡", "马来西亚", "泰国", "越南",
             "中国澳门", "中国大陆", "澳大利亚", "美国"],
    "不同法域": ["中国香港", "日本", "韩国", "新加坡", "马来西亚", "泰国", "越南",
                 "中国澳门", "中国大陆", "澳大利亚", "美国"],
    "各司法管辖区": ["中国香港", "日本", "韩国", "新加坡", "马来西亚", "泰国", "越南",
                     "中国澳门", "中国大陆", "澳大利亚", "美国"],
}

# 查询中直接点名的法域关键词 → 规范名（含英文与旧称）
JURISDICTION_KEYWORDS: Dict[str, str] = {
    "美国": "美国", "日本": "日本", "新加坡": "新加坡",
    "泰国": "泰国", "越南": "越南", "中国香港": "中国香港",
    "中国澳门": "中国澳门", "马来西亚": "马来西亚",
    "韩国": "韩国", "澳大利亚": "澳大利亚",
    "香港": "中国香港", "澳门": "中国澳门",
    "中国大陆": "中国大陆", "china": "中国大陆",
    "macau": "中国澳门", "macao": "中国澳门",
    "hong kong": "中国香港", "singapore": "新加坡",
    "japan": "日本", "thailand": "泰国",
    "vietnam": "越南", "malaysia": "马来西亚",
    "korea": "韩国", "australia": "澳大利亚", "澳洲": "澳大利亚",
    "united states": "美国", "usa": "美国",
}

# 比较型问句（无法确定区域时的扇出信号）
COMPARE_KEYWORDS: List[str] = [
    "有何异同", "有何差异", "比较", "区别", "差异", "异同", "不同模式", "versus", "vs",
]

# 有数据的法域清单：比较型查询兜底扇出用
ALL_DATA_JURISDICTIONS: List[str] = [
    "美国", "日本", "韩国", "新加坡", "中国香港", "中国澳门",
    "中国大陆", "马来西亚", "泰国", "越南", "澳大利亚",
]

# 以英文语料为主的法域：专题分析必须改用英文检索词，
# 否则用中文主题词去检索英文条文，召回显著劣化。新增英文法域后需同步补充。
ENGLISH_JURISDICTIONS: FrozenSet[str] = frozenset({
    "新加坡", "中国香港", "香港", "韩国", "澳大利亚", "美国",
    "Singapore", "Hong Kong", "Korea", "Australia", "USA",
})


def trigger_hit(trigger: str, query: str, lowered: str) -> bool:
    """触发词匹配：ASCII 词加边界，避免 'hk' 命中 'shk' 这类子串误判；中文用子串。"""
    if trigger.isascii():
        return re.search(
            rf"(?<![a-z0-9]){re.escape(trigger)}(?![a-z0-9])", lowered
        ) is not None
    return trigger in query


def detect_single_jurisdiction(query: str) -> Tuple[str, List[str]]:
    """识别查询中的**唯一**法域，返回 (法域或空串, 命中的全部法域)。

    命中多个必须由调用方按「非唯一」处理：单法域会被当成检索硬过滤条件，
    直接丢掉其余法域的召回（「美国与日本的跨境规则」曾被过滤成只搜美国）。
    """
    text = query or ""
    lowered = text.lower()
    matched = {
        jurisdiction
        for jurisdiction, triggers in RETRIEVAL_TRIGGERS.items()
        if any(trigger_hit(t, text, lowered) for t in triggers)
    }
    if CHINA_STANDALONE_RE.search(text):
        matched.add("中国大陆")
    ordered = [j for j in ALL_DATA_JURISDICTIONS if j in matched]
    if len(ordered) != 1:
        return "", ordered
    return ordered[0], ordered
