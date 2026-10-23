from enum import Enum
from typing import Dict, Tuple


class Jurisdiction(str, Enum):
    """法域枚举类"""
    MACAU = "澳门"
    HONGKONG = "香港"
    SINGAPORE = "新加坡"
    CHINA = "中国"
    TAIWAN = "台湾"
    MALAYSIA = "马来西亚"
    JAPAN = "日本"
    KOREA = "韩国"
    USA = "美国"
    UK = "英国"
    GERMANY = "德国"
    FRANCE = "法国"


# 法域信息映射表：(英文目录名, ID前缀代码)
JURISDICTION_INFO: Dict[str, Tuple[str, str]] = {
    "澳门": ("macau", "MO"),
    "香港": ("hongkong", "HK"),
    "新加坡": ("singapore", "SG"),
    "中国": ("china", "CN"),
    "台湾": ("taiwan", "TW"),
    "马来西亚": ("malaysia", "MY"),
    "日本": ("japan", "JP"),
    "韩国": ("korea", "KR"),
    "美国": ("usa", "US"),
    "英国": ("uk", "UK"),
    "德国": ("germany", "DE"),
    "法国": ("france", "FR"),
}


def get_jurisdiction_info(jurisdiction: str) -> Tuple[str, str]:
    """返回法域对应的 (英文目录名, ID 前缀代码)。"""
    if jurisdiction in JURISDICTION_INFO:
        return JURISDICTION_INFO[jurisdiction]
    
    # 大小写不敏感兜底匹配
    jur_lower = jurisdiction.lower().strip()
    for key, value in JURISDICTION_INFO.items():
        if key.lower() == jur_lower:
            return value
    
    # 仍未匹配时使用默认值
    dir_name = jur_lower.replace(" ", "_")
    prefix = jurisdiction[:2].upper()[:2] if len(jurisdiction) >= 2 else jurisdiction.upper()
    return (dir_name, prefix)


def get_all_jurisdictions() -> list:
    """获取所有已定义的法域列表"""
    return list(JURISDICTION_INFO.keys())
