from enum import Enum
from typing import Dict, Tuple


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
