from typing import List, Dict, Optional, Any, Tuple
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, TransportError, ApiError
from core.config import settings
from core.jurisdiction import jurisdiction_aliases, normalize_jurisdiction
from core.model_loader import get_embedding_model
from utils.topic_classifier import TopicClassifier


_ZH_T2S_MAP = {
    '網': '网', '絡': '络', '個': '个', '資': '资', '訊': '讯', '護': '护',
    '產': '产', '電': '电', '腦': '脑', '擊': '击', '竊': '窃', '盜': '盗',
    '僞': '伪', '關': '关', '係': '系', '機': '机', '構': '构', '團': '团',
    '體': '体', '監': '监', '辦': '办', '廳': '厅', '處': '处', '員': '员',
    '職': '职', '區': '区', '域': '域', '內': '内', '務': '务', '險': '险',
    '報': '报', '舉': '举', '復': '复', '製': '制', '傳': '传', '輸': '输',
    '發': '发', '佈': '布', '獲': '获', '數': '数', '據': '据', '隱': '隐',
    '私': '私', '祕': '秘', '密': '密', '條': '条', '約': '约', '規': '规',
    '則': '则', '範': '范', '圍': '围', '標': '标', '準': '准', '審': '审',
    '計': '计', '訴': '诉', '訟': '讼', '罰': '罚', '執': '执', '實': '实',
    '施': '施', '細': '细', '總': '总', '附': '附', '錄': '录', '匯': '汇',
    '兌': '兑', '幣': '币', '貨': '货', '銀': '银', '證': '证', '券': '券',
    '債': '债', '權': '权', '務': '务', '業': '业', '廠': '厂', '號': '号',
    '碼': '码', '圖': '图', '視': '视', '頻': '频', '聲': '声', '軟': '软',
    '頁': '页', '連': '连', '鏈': '链', '結': '结', '載': '载', '郵': '邮',
    '即': '即', '時': '时', '討': '讨', '論': '论', '壇': '坛',
    '購': '购', '簽': '签', '署': '署', '訂': '订', '契': '契',
    '知': '知', '識': '识', '版': '版', '專': '专', '利': '利',
    '著': '著', '營': '营', '經': '经', '照': '照', '許': '许',
    '可': '可', '書': '书', '登': '登', '記': '记', '註': '注', '冊': '册',
    '備': '备', '案': '案', '開': '开', '創': '创', '新': '新', '研': '研',
    '究': '究', '試': '试', '驗': '验', '測': '测', '檢': '检', '查': '查',
    '鑑': '鉴', '定': '定', '評': '评', '估': '估', '核': '核', '准': '准',
    '認': '认', '責': '责', '任': '任', '義': '义', '利': '利', '益': '益',
    '損': '损', '害': '害', '賠': '赔', '償': '偿', '補': '补', '助': '助',
    '獎': '奖', '勵': '励', '懲': '惩', '戒': '戒', '紀': '纪', '律': '律',
    '衛': '卫', '生': '生', '醫': '医', '療': '疗', '藥': '药', '健': '健',
    '康': '康', '環': '环', '境': '境', '汙': '污', '染': '染', '廢': '废',
    '棄': '弃', '勞': '劳', '僱': '雇', '傭': '佣', '就': '就', '薪': '薪',
    '福': '福', '休': '休', '假': '假', '解': '解', '辭': '辞', '退': '退',
    '消': '消', '費': '费', '者': '者', '兒': '儿', '童': '童', '少': '少',
    '青': '青', '老': '老', '殘': '残', '障': '障', '疾': '疾', '婦': '妇',
    '幼': '幼', '家': '家', '庭': '庭', '教': '教', '育': '育', '學': '学',
    '習': '习', '校': '校', '院': '院', '科': '科', '技': '技', '術': '术',
    '工': '工', '農': '农', '漁': '渔', '礦': '矿', '交': '交', '通': '通',
    '運': '运', '物': '物', '流': '流', '能': '能', '源': '源', '力': '力',
    '水': '水', '土': '土', '建': '建', '設': '设', '築': '筑', '屋': '屋',
    '宇': '宇', '公': '公', '共': '共', '秩': '秩', '序': '序', '國': '国',
    '家': '家', '民': '民', '族': '族', '社': '社', '會': '会', '政': '政',
    '治': '治', '軍': '军', '事': '事', '外': '外', '司': '司', '法': '法',
    '立': '立', '行': '行', '政': '政', '察': '察', '警': '警', '獄': '狱',
    '稅': '税', '海': '海', '關': '关', '財': '财', '預': '预', '算': '算',
    '統': '统', '際': '际', '地': '地', '城': '城', '市': '市', '鄉': '乡',
    '鎮': '镇', '村': '村', '莊': '庄', '歐': '欧', '盟': '盟', '聯': '联',
    '合': '合', '世': '世', '界': '界', '貿': '贸', '易': '易', '投': '投',
    '引': '引', '並': '并', '重': '重', '組': '组', '整': '整', '合': '合',
    '打': '打', '犯': '犯', '罪': '罪', '刑': '刑', '民': '民', '仲': '仲',
    '裁': '裁', '調': '调', '節': '节', '約': '约',
}


def zh_t2s(text: str) -> str:
    """繁体转简体（轻量级，覆盖法规名称常用字）"""
    if not text:
        return text
    return ''.join(_ZH_T2S_MAP.get(ch, ch) for ch in text)


def zh_s2t(text: str) -> str:
    """简体转繁体（反向映射，轻量级）"""
    if not text:
        return text
    s2t = {v: k for k, v in _ZH_T2S_MAP.items()}
    return ''.join(s2t.get(ch, ch) for ch in text)


# 完整繁简字符表（OpenCC 字符集），仅用于检索侧查询改写；上方轻量表服务于
# 法规名/实体名规范化，口径已与实体消歧、KG 链路绑定，两者互不影响。
# 必须改写的原因：IK 词典不含繁体字，繁体正文被切成一字一 token，与简体查询的
# 词级 token 空间不相交，繁体语料上简体查询 BM25 召回为 0。做法是把同一查询的
# 繁、简两种写法一并送入 multi_match，任一写法命中即可。
_ZH_CONVERT_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _load_zh_convert() -> Dict[str, Dict[str, str]]:
    """加载完整繁简字符表；失败时降级为空表（等于不改写，不阻断检索）"""
    global _ZH_CONVERT_CACHE
    if _ZH_CONVERT_CACHE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "data", "zh_convert.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _ZH_CONVERT_CACHE = {"t2s": data.get("t2s", {}), "s2t": data.get("s2t", {})}
        except Exception as e:
            logger.warning(f"[繁简改写] 字表加载失败，本次不做改写: {e}")
            _ZH_CONVERT_CACHE = {"t2s": {}, "s2t": {}}
    return _ZH_CONVERT_CACHE


def zh_convert(text: str, direction: str) -> str:
    """按完整字表做繁简转换；direction 为 't2s'（繁→简）或 's2t'（简→繁）"""
    if not text:
        return text
    table = _load_zh_convert().get(direction) or {}
    return ''.join(table.get(ch, ch) for ch in text)


def zh_query_variants(text: str) -> List[str]:
    """返回查询的繁简写法列表（原样 / 繁转简 / 简转繁，去重保序）。

    简→繁存在多候选（如「系」可为「系/係」），只取 OpenCC 首选；错误候选对 BM25
    只是多出若干不命中 token，不会误伤召回。
    """
    if not text:
        return []
    variants = [text]
    for direction in ("t2s", "s2t"):
        converted = zh_convert(text, direction)
        if converted and converted not in variants:
            variants.append(converted)
    return variants


# 法域英文 -> 中文映射
_JURISDICTION_ZH_MAP: Dict[str, str] = {
    "us": "美国",
    "usa": "美国",
    "united states": "美国",
    "cn": "中国大陆",
    "china": "中国大陆",
    "hk": "中国香港",
    "hong kong": "中国香港",
    "mo": "中国澳门",
    "macau": "中国澳门",
    "sg": "新加坡",
    "singapore": "新加坡",
}


def _jurisdiction_to_zh(jur: str) -> str:
    """将法域名称归一为规范简体中文名。"""
    if not jur:
        return jur
    lower_jur = jur.lower().strip()
    zh = jur if any('\u4e00' <= c <= '\u9fff' for c in jur) else _JURISDICTION_ZH_MAP.get(lower_jur, jur)
    return normalize_jurisdiction(zh)


def _expand_jurisdiction_where(where: Optional[Dict]) -> Optional[Dict]:
    """把 where 中的法域值展开为规范名 + 旧称，兼容改名前后的索引数据。"""
    if not where or "jurisdiction" not in where:
        return where
    raw_values = where["jurisdiction"]
    values = raw_values if isinstance(raw_values, list) else [raw_values]
    expanded: List[str] = []
    for value in values:
        for alias in jurisdiction_aliases(value):
            if alias not in expanded:
                expanded.append(alias)
    if expanded == values:
        return where
    return {**where, "jurisdiction": expanded}
_TOPIC_LABEL_ZH_CACHE: Optional[Dict[str, str]] = None
_TOPIC_FALLBACK_ZH: Dict[str, str] = {
    # 美国词表中未在 taxonomy_merged.json 中定义的主题
    "public_administration": "公共治理",
    "government_contracting": "政府采购",
    "intellectual_property_licensing": "知识产权",
    "information_classification": "信息分类",
    "administrative_procedures": "行政程序",
    "records_management": "档案管理",
    "electronic_transactions": "电子交易与电子政务",
}


def _load_topic_label_zh_mapping() -> Dict[str, str]:
    """加载 taxonomy_merged.json，返回 {id: label_zh} 映射"""
    global _TOPIC_LABEL_ZH_CACHE
    if _TOPIC_LABEL_ZH_CACHE is not None:
        return _TOPIC_LABEL_ZH_CACHE

    taxonomy_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "taxonomy_merged.json"
    )
    mapping = {}
    try:
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for t in data.get("topics", []):
            tid = t.get("id", "")
            label = t.get("label_zh", "")
            if tid and label:
                mapping[tid] = label
    except Exception:
        pass

    _TOPIC_LABEL_ZH_CACHE = mapping
    return mapping


def _to_label_zh(topic_id: str) -> str:
    """将英文 topic_id 转为中文标签，找不到则返回原值"""
    mapping = _load_topic_label_zh_mapping()
    if topic_id in mapping:
        return mapping[topic_id]
    return _TOPIC_FALLBACK_ZH.get(topic_id, topic_id)


# 已废弃的粗粒度主题：无中文标签且缺乏区分度，展示层与入库层统一剔除
EXCLUDED_TOPICS = {"legal_definitions", "regulatory_scope", "法律定义"}


def is_excluded_topic(value: str) -> bool:
    """判断主题（英文 ID 或中文标签）是否属于剔除名单"""
    return str(value or "").strip().lower() in EXCLUDED_TOPICS


def _label_zh_to_topic_id(label_zh: str) -> str:
    """将中文标签转为英文 topic_id，找不到则返回原值"""
    mapping = _load_topic_label_zh_mapping()
    # 反向查找：label_zh -> id
    for tid, label in mapping.items():
        if label == label_zh:
            return tid
    # fallback 反查
    for tid, label in _TOPIC_FALLBACK_ZH.items():
        if label == label_zh:
            return tid
    return label_zh


# 同法规条款引用解析（组件D：引用链多跳）
# 条款编号在法规内是局部的，正文「第X條」即同法规引用，无需解析法规名，精度高。
# 跨法规引用（如「第7/2023號法律第三十六條」）需「法规名 -> law_id」映射，且 KG 中
# cross_refers 极少，本阶段不参与扩展。

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000,
}

# 「第X條 / 第X条」，X 为阿拉伯数字（含全角）或中文数字
_ARTICLE_REF_RE = re.compile(r"第\s*([0-9０-９]{1,4}|[零〇一二三四五六七八九十百千兩两]{1,8})\s*[條条]")

# 匹配位置前紧邻「...號法律/法令/行政法規/行政命令/批示」时，该条款号属于另一部法规
_OTHER_LAW_TAIL_RE = re.compile(
    r"號\s*(?:法律|法令|行政法規|行政命令|批示|訓令|決議|規章)[^。；;]{0,60}$"
)

# 跨法规引用（可确定性推导目标 law_id）：
#   澳门「第N/YYYY號法律」→ MAC_LEI_N_YYYY，「第N/YYYY號行政法規」→ MAC_REG_N_YYYY
#   香港「第N章」→ HK_CAP_N
# 这些引用已被 _OTHER_LAW_TAIL_RE 从同法规引用中剔除（剔除本身正确），故跨法规
# 解析独立进行、不复用该守卫。
_MAC_CROSS_LAW_RE = re.compile(r"第\s*(\d{1,4})\s*/\s*(\d{4})\s*號\s*(法律|行政法規)")
_HK_CAP_RE = re.compile(r"第\s*(\d{1,4})\s*章")
_MAC_LAW_TYPE_CODE = {"法律": "LEI", "行政法規": "REG"}
# 跨法规引用后紧随的条款号
_ARTICLE_AFTER_RE = re.compile(
    r"第\s*([0-9０-９]{1,4}|[零〇一二三四五六七八九十百千兩两]{1,8})\s*[條条]"
)

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _cn_numeral_to_int(text: str) -> Optional[int]:
    """中文/阿拉伯数字转整数。支持「十五」「二十一」「一百零三」；无法解析返回 None"""
    if not text:
        return None
    text = text.translate(_FULLWIDTH_DIGITS).strip()
    if text.isdigit():
        return int(text)
    if all(ch in ("零", "〇") for ch in text):
        return 0

    total = 0     # 已结算的更高位
    number = 0    # 当前累计的个位数
    for ch in text:
        if ch in ("零", "〇"):
            continue
        value = _CN_DIGITS.get(ch)
        if value is None:
            return None
        if value >= 10:
            # 「十五」= 1*10+5，「二十」= 2*10，「一百零三」= 100+3
            total += (number or 1) * value
            number = 0
        else:
            number = value
    return total + number


def parse_intra_law_refs(content: str, article_number: str) -> List[str]:
    """从条款正文解析同法规内被引用的条款号，返回数字字符串列表（如 ["2", "15"]）。

    剔除三类跨法规误判：书名号未闭合（条款号在书名号内）、条款号紧跟闭合书名号、
    条款号紧跟「...號法律/法令/行政法規/...」。解析只表示正文出现了该条款号，
    目标不存在由回取阶段自然过滤。
    """
    if not content:
        return []

    self_num = str(article_number or "").strip()
    refs: List[str] = []
    seen = set()

    for match in _ARTICLE_REF_RE.finditer(content):
        prefix = content[: match.start()]
        stripped = prefix.rstrip()
        # 书名号未闭合：当前条款号位于某部法规的标题内部
        if prefix.count("《") > prefix.count("》"):
            continue
        # 条款号紧跟闭合书名号：如「《收費總表》第一百一十五條」，归书名号内的法规
        if stripped.endswith("》"):
            continue
        # 紧邻前文出现其他法规名，条款号归其所有
        if _OTHER_LAW_TAIL_RE.search(stripped[-80:]):
            continue

        number = _cn_numeral_to_int(match.group(1))
        if not number or number <= 0:
            continue

        ref = str(number)
        if ref == self_num or ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)

    return refs


def parse_cross_law_refs(content: str, article_number: str,
                         law_id: str) -> List[Tuple[str, str]]:
    """解析条款正文中指向【其他法规】的引用，返回 [(目标 law_id, 目标条款号), ...]。

    仅覆盖能从文本确定性推导 law_id 的模式：澳门「第N/YYYY號法律/行政法規」、
    香港「第N章」，且要求引用后 8 字符内紧随具体条款号。目标是否存在交给回取阶段
    过滤，因此不依赖法规字典，解析失败返回空、不阻断主链路。
    """
    if not content:
        return []

    self_law = law_id or ""
    self_article = str(article_number or "")
    out: List[Tuple[str, str]] = []
    seen = set()

    def _pick(target_law: str, after_pos: int) -> None:
        if not target_law or target_law == self_law:
            return  # 自引用无扩展价值
        m = _ARTICLE_AFTER_RE.search(content, after_pos, after_pos + 8)
        if not m:
            return
        # 法规名与条款号之间若出现句读分隔（逗号/顿号/分号/句号等），该条款号多半是
        # 本条款自身的条号而非被引法规的条号（如「…第5/2009號行政法規，第7條…」），
        # 直接跳过以降低误连边。合法的直引形式（「…行政法規第十七條」）间隔为空或「的」。
        gap = content[after_pos:m.start()]
        if any(ch in "，,、。；;：:！？!?…" for ch in gap):
            return
        number = _cn_numeral_to_int(m.group(1))
        if not number or number <= 0 or str(number) == self_article:
            return
        key = (target_law, str(number))
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    for m in _MAC_CROSS_LAW_RE.finditer(content):
        code = _MAC_LAW_TYPE_CODE.get(m.group(3))
        if code:
            _pick(f"MAC_{code}_{int(m.group(1))}_{m.group(2)}", m.end())
    for m in _HK_CAP_RE.finditer(content):
        _pick(f"HK_CAP_{int(m.group(1))}", m.end())

    return out


def _join_chunks_without_overlap(chunks: List[str]) -> str:
    """按顺序拼接同一 article 的子块，并裁掉相邻子块间的切分重叠。

    对每对相邻块检测「上块尾部 == 下块头部」，扫描上限 min(chunk_overlap + 20, 两侧块长)，
    下限 30（低于 30 字符的重复多为语料巧合，不裁）；无重叠的旧数据输出与直接拼接一致。
    """
    parts: List[str] = []
    prev = ""
    for raw in chunks:
        if not raw:
            continue
        limit = min(settings.chunk_overlap + 20, len(prev), len(raw))
        cut = 0
        for k in range(limit, 29, -1):
            if prev.endswith(raw[:k]):
                cut = k
                break
        tail = raw[cut:] if cut else raw
        if tail:
            parts.append(tail)
        prev = raw
    return "\n\n".join(parts)


class ElasticsearchRepository:
    def __init__(self):
        self._init_client()
        self._model = None  # 懒加载，避免启动时加载大模型
        self._non_active_cache = None  # 失效法规 law_id 缓存：(mtime, ids)
        self._law_title_index_cache = None  # 法规标题索引缓存：{简体归一标题: law_id}
        self._init_classifier()
        self._create_index_if_not_exists()
    
    def _init_classifier(self):
        self.classifier = TopicClassifier()
    
    def _init_client(self):
        client_kwargs: Dict[str, Any] = {
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
        
        self.client = Elasticsearch(**client_kwargs)
        
        if not self.client.ping():
            raise ConnectionError("无法连接到 Elasticsearch")
    
    def _get_model(self):
        """懒加载 embedding 模型（首次使用时加载）"""
        if self._model is None:
            self._model = get_embedding_model()
        return self._model
    
    def _create_index_if_not_exists(self):
        if not self.client.indices.exists(index=settings.es_index_name):
            index_settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "text_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop"]
                            },
                            "chinese_analyzer": {
                                "type": "custom",
                                "tokenizer": "ik_max_word",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "chinese_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            }
                        },
                        # 简体改写字段：入库时 content 繁→简，BM25 与 content 并行打分
                        "content_s": {
                            "type": "text",
                            "analyzer": "chinese_analyzer"
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {"type": "object"},
                        "law_id": {"type": "keyword"},
                        "jurisdiction": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "chinese_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            }
                        },
                        "article_number": {"type": "keyword"},
                        "chunk_type": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "passing_date": {"type": "keyword"},
                        "effective_date": {"type": "keyword"},
                        "publication_date": {"type": "keyword"},
                        "topic": {"type": "keyword"},
                        "topics": {"type": "keyword"},
                        "topic_labels": {"type": "keyword"},
                        "topics_details": {
                            "type": "nested",
                            "properties": {
                                "id": {"type": "keyword"},
                                "label_zh": {"type": "keyword"},
                                "score": {"type": "float"},
                                "matched_keywords": {"type": "keyword"}
                            }
                        },
                        "taxonomy_version": {"type": "keyword"}
                    }
                }
            }
            self.client.indices.create(index=settings.es_index_name, body=index_settings)
        else:
            # 索引已存在：补挂后加字段（定义一致时 ES 幂等接受，失败仅告警）
            try:
                self.client.indices.put_mapping(
                    index=settings.es_index_name,
                    body={"properties": {
                        "content_s": {"type": "text", "analyzer": "chinese_analyzer"},
                        # 存量索引补挂 chunk_type（keyword）：不回填历史文档，仅让新写入生效
                        "chunk_type": {"type": "keyword"}
                    }}
                )
            except Exception as e:
                logger.warning(f"[索引映射] 补充字段补挂失败（不影响已有数据检索）: {e}")
    
    def prune_law_docs(self, law_id: str, keep_ids: List[str]) -> int:
        """删除某法规下不在 keep_ids 中的陈旧文档，保证重复入库结果幂等。

        入库用显式 _id upsert，但条款减少/重排后旧 doc_id 会残留为孤儿文档。
        """
        if not law_id:
            return 0
        must_not = [{"ids": {"values": keep_ids}}] if keep_ids else []
        try:
            resp = self.client.delete_by_query(
                index=settings.es_index_name,
                body={"query": {"bool": {
                    "filter": [{"term": {"law_id": law_id}}],
                    "must_not": must_not
                }}},
                refresh=True,
                conflicts="proceed"
            )
            return int(resp.get("deleted", 0))
        except Exception as e:
            logger.warning(f"[入库清理] law_id={law_id} 陈旧文档清理失败: {e}")
            return 0
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        import torch
        import os
        with torch.no_grad():
            # 禁用 tqdm 进度条，避免与 uvicorn 日志系统冲突导致进程退出
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            embeddings = self._get_model().encode(
                texts,
                batch_size=settings.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings.tolist()
    
    def _extract_topic_ids(self, raw_topics: Any) -> List[str]:
        """将 metadata 中的 topics（对象数组或字符串数组）统一转换为字符串数组。"""
        if not isinstance(raw_topics, list):
            return []
        result = []
        for t in raw_topics:
            if isinstance(t, dict):
                tid = t.get("id") or t.get("topic_id") or t.get("label_en")
                if tid:
                    result.append(str(tid))
            elif isinstance(t, str) and t:
                result.append(t)
        return result

    def _normalize_topic_fields(self, content: str, metadata: Dict) -> Dict:
        """统一生成主题字段：优先用 metadata 已有 topics，否则用分类器自动标注。"""
        clean_metadata = {k: v for k, v in metadata.items()
                          if k not in ('topic', 'topics', 'topic_labels', 'topics_details', 'taxonomy_version',
                                       'passing_date', 'effective_date', 'publication_date')}

        existing_ids = self._extract_topic_ids(metadata.get("topics"))
        existing_details = metadata.get("topics_details")

        if existing_ids:
            topic_ids = existing_ids
            topic_labels = []
            details = []
            for tid in topic_ids:
                label = _to_label_zh(tid)
                topic_labels.append(label)
                detail = {"id": tid, "label_zh": label, "score": 0.0, "matched_keywords": []}
                # 已有 details 时沿用其 score 与 matched_keywords
                if isinstance(existing_details, list):
                    for d in existing_details:
                        if isinstance(d, dict) and (d.get("id") == tid or d.get("topic_id") == tid):
                            detail["score"] = d.get("score", 0.0)
                            detail["matched_keywords"] = d.get("matched_keywords", [])
                            break
                details.append(detail)
            return {
                "metadata": clean_metadata,
                "topics": topic_ids,
                "topic_labels": topic_labels,
                "topics_details": details,
                "taxonomy_version": metadata.get("taxonomy_version", self.classifier.taxonomy_version),
            }

        auto = self.classifier.classify_document(content)
        return {
            "metadata": clean_metadata,
            "topics": auto["topics"],
            "topic_labels": auto["topic_labels"],
            "topics_details": auto["topics_details"],
            "taxonomy_version": auto["taxonomy_version"],
        }

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str],
                      embed_texts: Optional[List[str]] = None):
        """批量写入文档（显式 _id，重复写入即 upsert）。

        embed_texts 为与 documents 等长的 embedding 输入（带标题前缀）；为空或长度
        不符时回退为直接用 documents 向量化，content 字段始终写入原文。
        """
        if embed_texts and len(embed_texts) == len(documents):
            embed_inputs = [et if et else d for et, d in zip(embed_texts, documents)]
        else:
            embed_inputs = documents
        embeddings = self._embed_texts(embed_inputs)
        operations = []
        failed_docs = []

        for doc_id, content, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            topic_fields = self._normalize_topic_fields(content, metadata)
            operations.append({"index": {"_index": settings.es_index_name, "_id": doc_id}})
            doc = {
                "content": content,
                "content_s": zh_convert(content, "t2s"),
                "embedding": embedding,
                "metadata": topic_fields["metadata"],
                "law_id": metadata.get("law_id", ""),
                "jurisdiction": metadata.get("jurisdiction", ""),
                "title": metadata.get("title", ""),
                "article_number": metadata.get("article_number", "") or "",
                "chunk_type": metadata.get("chunk_type", "") or "",
                "chunk_index": metadata.get("chunk_index", 0),
                "passing_date": metadata.get("passing_date", "") or None,
                "effective_date": metadata.get("effective_date", "") or None,
                "publication_date": metadata.get("publication_date", "") or None,
                # 主题字段提升到顶层，供筛选/聚合/统计使用
                "topic": metadata.get("topic", ""),
                "topics": topic_fields["topics"],
                "topic_labels": topic_fields["topic_labels"],
                "topics_details": topic_fields["topics_details"],
                "taxonomy_version": topic_fields["taxonomy_version"],
            }
            operations.append(doc)

        response = self.client.bulk(body=operations, refresh=True)
        
        if response.get("errors"):
            error_items = []
            for item in response.get("items", []):
                if "error" in item.get("index", {}):
                    error_info = item["index"]
                    error_items.append({
                        "doc_id": error_info.get("_id"),
                        "error_type": error_info.get("error", {}).get("type", ""),
                        "error_reason": error_info.get("error", {}).get("reason", "")
                    })
                    failed_docs.append(error_info.get("_id"))
            
            if error_items:
                # 重试失败的文档
                retry_operations = []
                for i, doc_id in enumerate(ids):
                    if doc_id in failed_docs:
                        idx = ids.index(doc_id)
                        retry_operations.append({"index": {"_index": settings.es_index_name, "_id": doc_id}})
                        retry_operations.append(operations[idx * 2 + 1])  # 文档内容在偶数索引+1位置
                
                if retry_operations:
                    retry_response = self.client.bulk(body=retry_operations, refresh=True)
                    if retry_response.get("errors"):
                        for item in retry_response.get("items", []):
                            if "error" in item.get("index", {}):
                                error_info = item["index"]
                                error_items.append({
                                    "doc_id": error_info.get("_id"),
                                    "error_type": error_info.get("error", {}).get("type", ""),
                                    "error_reason": error_info.get("error", {}).get("reason", "")
                                })
                
                raise ValueError(f"部分文档添加失败: {error_items}")
    
    def query(self, query_texts: List[str], n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        
        for text in query_texts:
            query_embedding = self._embed_texts([text])[0]
            
            knn_base = {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": n_results,
                "num_candidates": n_results * 10
            }
            
            if where:
                where = _expand_jurisdiction_where(where)
                filter_clauses = []
                for k, v in where.items():
                    if isinstance(v, list) and len(v) > 1:
                        filter_clauses.append({
                            "bool": {
                                "should": [{"term": {k: item}} for item in v],
                                "minimum_should_match": 1
                            }
                        })
                    else:
                        actual_value = v[0] if isinstance(v, list) else v
                        filter_clauses.append({"term": {k: actual_value}})

                # filter 嵌入 KNN 内部，保证「向量相似 AND 过滤条件」语义
                search_body = {
                    "knn": {
                        **knn_base,
                        "filter": [{"bool": {"filter": filter_clauses}}]
                    }
                }
            else:
                search_body = {"knn": knn_base}
            
            response = self.client.search(index=settings.es_index_name, body=search_body)
            
            hits = response.get("hits", {}).get("hits", [])
            query_ids, query_docs, query_metas, query_distances = [], [], [], []
            
            for hit in hits:
                source = hit.get("_source", {})
                query_ids.append(hit.get("_id"))
                query_docs.append(source.get("content", ""))
                metadata = source.get("metadata", {}).copy()
                metadata["law_id"] = source.get("law_id", "")
                metadata["title"] = source.get("title", "")
                metadata["jurisdiction"] = source.get("jurisdiction", "")
                metadata["article_number"] = source.get("article_number", "")
                metadata["chunk_index"] = source.get("chunk_index", 0)
                metadata["topic"] = source.get("topic", "")
                metadata["topics"] = source.get("topics", [])
                metadata["topic_labels"] = source.get("topic_labels", [])
                query_metas.append(metadata)
                score = hit.get("_score", 0)
                query_distances.append(1 - score if score > 0 else 1.0)
            
            results["ids"].append(query_ids)
            results["documents"].append(query_docs)
            results["metadatas"].append(query_metas)
            results["distances"].append(query_distances)
        
        return results

    def hybrid_search(self, query_text: str, n_results: int = 5,
                      where: Optional[Dict] = None,
                      knn_weight: float = 0.5,
                      bm25_weight: float = 0.5,
                      rrf_k: int = 60) -> Dict:
        """混合检索：向量 KNN + BM25，应用层 RRF 融合。

        knn_weight / bm25_weight 在 RRF 模式下未使用，仅为兼容保留；rrf_k 为 RRF 常数。
        返回格式与 query() 相同，另含 _scores（RRF 分）与 _similarities（真实余弦）。
        """
        if not query_text.strip():
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]],
                    "_scores": [[]], "_similarities": [[]]}

        query_embedding = self._embed_texts([query_text])[0]
        return self._hybrid_search_with_vector(query_text, query_embedding, n_results, where, rrf_k)

    def hybrid_search_batch(self, query_texts: List[str], n_results: int = 5,
                           where: Optional[Dict] = None, rrf_k: int = 60) -> List[Dict]:
        """批量混合检索：一次批量编码所有查询，再并发执行 ES 搜索（IO 密集）。

        返回 List[Dict]，每项格式与 hybrid_search() 相同。
        """
        if not query_texts:
            return []
        # 一次性批量编码，避免 N 次独立推理
        embeddings = self._embed_texts(query_texts)
        with ThreadPoolExecutor(max_workers=min(len(query_texts), 8)) as executor:
            futures = [
                executor.submit(self._hybrid_search_with_vector, query_text, embedding, n_results, where, rrf_k)
                for query_text, embedding in zip(query_texts, embeddings)
            ]
            results = [f.result() for f in futures]
        return results

    def _hybrid_search_with_vector(self, query_text: str, query_embedding: List[float],
                                   n_results: int = 5, where: Optional[Dict] = None,
                                   rrf_k: int = 60) -> Dict:
        """使用预计算向量执行混合搜索（跳过 embedding 计算）。

        ES 许可证可能不支持内置 RRF，故分别执行 KNN 与 BM25，在应用层手动计算 RRF 合并。
        """
        # 两次查询互不依赖，并行只承担较大者而非两者之和
        with ThreadPoolExecutor(max_workers=2) as executor:
            knn_future = executor.submit(self._knn_search, query_embedding, n_results * 3, where)
            bm25_future = executor.submit(self._bm25_search, query_text, n_results * 3, where)
            knn_results = knn_future.result()
            bm25_results = bm25_future.result()

        merged = self._merge_rrf_results(knn_results, bm25_results, rrf_k=rrf_k)
        merged = merged[:n_results]

        query_ids, query_docs, query_metas, query_distances, query_scores = [], [], [], [], []
        query_similarities = []  # 真实余弦相似度（仅 KNN 命中；BM25-only 为 None），供展示用
        for item in merged:
            source = item["source"]
            query_ids.append(item["_id"])
            query_docs.append(source.get("content", ""))

            metadata = source.get("metadata", {}).copy()
            metadata["law_id"] = source.get("law_id", "")
            metadata["title"] = source.get("title", "")
            metadata["jurisdiction"] = source.get("jurisdiction", "")
            metadata["article_number"] = source.get("article_number", "")
            metadata["chunk_index"] = source.get("chunk_index", 0)
            metadata["topic"] = source.get("topic", "")
            metadata["topics"] = source.get("topics", [])
            metadata["topic_labels"] = source.get("topic_labels", [])
            query_metas.append(metadata)

            rrf_score = item["rrf_score"]
            query_scores.append(rrf_score)
            # distances 仅为兼容保留：RRF 分范围极小，反推的相似度无区分度（曾致前端
            # 全部显示 50%）；展示分改用 _similarities 中的真实余弦。
            query_distances.append(1.0 / (1.0 + abs(rrf_score)) if rrf_score != 0 else 1.0)
            query_similarities.append(item.get("vector_similarity"))

        return {
            "ids": [query_ids],
            "documents": [query_docs],
            "metadatas": [query_metas],
            "distances": [query_distances],
            "_scores": [query_scores],
            "_similarities": [query_similarities],
        }

    def _knn_search(self, query_embedding: List[float], n_results: int,
                    where: Optional[Dict] = None) -> List[Dict]:
        """纯向量 KNN 搜索，返回原始 hits 列表。"""
        knn_base = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": n_results,
            "num_candidates": min(n_results * 10, 10000)
        }

        filter_clauses = []
        if where:
            where = _expand_jurisdiction_where(where)
            for k, v in where.items():
                if isinstance(v, list) and len(v) > 1:
                    filter_clauses.append({
                        "bool": {
                            "should": [{"term": {k: item}} for item in v],
                            "minimum_should_match": 1
                        }
                    })
                else:
                    actual_value = v[0] if isinstance(v, list) else v
                    filter_clauses.append({"term": {k: actual_value}})

        # 时效过滤：剔除已废止/已删除法规（组件C）
        excluded = self._excluded_law_filter()
        if excluded:
            filter_clauses.append({"bool": {"must_not": [excluded]}})

        # 必须显式设置顶层 size：ES 默认 size=10，仅设 knn.k 不会让返回条数超过 10。
        # 否则向量召回被截断在 10 条，RRF 中向量通道近乎失效、退化为 BM25 主导。
        if filter_clauses:
            search_body = {
                "size": n_results,
                "knn": {
                    **knn_base,
                    "filter": [{"bool": {"filter": filter_clauses}}]
                }
            }
        else:
            search_body = {"size": n_results, "knn": knn_base}

        response = self.client.search(index=settings.es_index_name, body=search_body)
        return response.get("hits", {}).get("hits", [])

    def _bm25_search(self, query_text: str, n_results: int,
                     where: Optional[Dict] = None) -> List[Dict]:
        """BM25 全文搜索，返回原始 hits 列表。

        查询串同时携带繁、简两种写法：IK 词典不含繁体，只按输入写法检索时简体查询
        在繁体语料上会零召回。写法由 zh_query_variants 生成。
        """
        variants = zh_query_variants(query_text)
        analysis_query = " ".join(variants) if len(variants) > 1 else query_text
        bm25_query = {
            "multi_match": {
                "query": analysis_query,
                # content_s 为入库时繁→简改写字段，为繁体语料提供第二条命中通道
                "fields": ["content^3", "content_s^3", "title^5", "article_number^2"],
                "type": "best_fields",
                "analyzer": "chinese_analyzer"
            }
        }

        filter_clauses = []
        if where:
            where = _expand_jurisdiction_where(where)
            for k, v in where.items():
                if isinstance(v, list) and len(v) > 1:
                    filter_clauses.append({
                        "bool": {
                            "should": [{"term": {k: item}} for item in v],
                            "minimum_should_match": 1
                        }
                    })
                else:
                    actual_value = v[0] if isinstance(v, list) else v
                    filter_clauses.append({"term": {k: actual_value}})

        # 时效过滤：剔除已废止/已删除法规（组件C）
        excluded = self._excluded_law_filter()
        if excluded:
            filter_clauses.append({"bool": {"must_not": [excluded]}})

        if filter_clauses:
            query_body = {
                "bool": {
                    "must": bm25_query,
                    "filter": filter_clauses
                }
            }
        else:
            query_body = bm25_query

        search_body = {
            "query": query_body,
            "size": n_results,
            "_source": True
        }

        response = self.client.search(index=settings.es_index_name, body=search_body)
        return response.get("hits", {}).get("hits", [])

    def _merge_rrf_results(self, knn_hits: List[Dict], bm25_hits: List[Dict],
                           rrf_k: int = 60) -> List[Dict]:
        """手动 RRF 融合：对 KNN 与 BM25 结果按 RRF 分合并去重排序。

        额外附带 KNN 命中的真实余弦：ES dense_vector(cosine) 的 _score 为 (1+cos)/2
        （正交=0.5），故还原 cos = 2*_score - 1，仅供上层展示；BM25-only 命中记为 None。
        """
        scores: Dict[str, float] = {}
        sources: Dict[str, Dict] = {}
        cosines: Dict[str, float] = {}

        for rank, hit in enumerate(knn_hits, start=1):
            doc_id = hit.get("_id")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            sources.setdefault(doc_id, hit.get("_source", {}))
            # KNN 已按相似度降序，首次出现即其最高分
            es_score = hit.get("_score")
            if es_score is not None:
                cosines.setdefault(doc_id, 2.0 * float(es_score) - 1.0)

        for rank, hit in enumerate(bm25_hits, start=1):
            doc_id = hit.get("_id")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            sources.setdefault(doc_id, hit.get("_source", {}))

        merged = []
        for doc_id, rrf_score in scores.items():
            merged.append({
                "_id": doc_id,
                "rrf_score": rrf_score,
                "vector_similarity": cosines.get(doc_id),
                "source": sources.get(doc_id, {})
            })

        merged.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged

    # 法规时效过滤（组件C）
    # law_catalog.json 的 status 标记时效；检索层默认剔除已废止/已删除法规，否则失效
    # 条款会被 LLM 当作现行法引用。不改 ES 文档，仅按 law_id 排除；缓存依 mtime 失效。

    # 不可检索的状态：repealed 已废止 / deleted 已删除 / missing 数据缺失
    # 注意 amended（已修订）表示法规仍然现行有效，只是内容被修改过，不排除
    _NON_RETRIEVABLE_STATUS = ("repealed", "deleted", "missing")

    def _non_active_law_ids(self) -> List[str]:
        """读取当前不可检索的 law_id 列表（按文件 mtime 缓存）。

        目录不可读时返回空列表（降级为不过滤），保证检索不被阻断。
        """
        import os
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "law_catalog.json",
        )
        try:
            mtime = os.path.getmtime(catalog_path)
        except OSError:
            return []

        if self._non_active_cache and self._non_active_cache[0] == mtime:
            return self._non_active_cache[1]

        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"[时效过滤] 读取法规目录失败，本次不做时效过滤: {e}")
            return []

        laws = data.get("laws", []) if isinstance(data, dict) else data
        if isinstance(laws, dict):
            laws = list(laws.values())

        ids = [
            x.get("law_id")
            for x in laws
            if isinstance(x, dict) and x.get("law_id")
            and x.get("status") in self._NON_RETRIEVABLE_STATUS
        ]
        if ids:
            logger.info(f"[时效过滤] 本次排除 {len(ids)} 部已废止/已删除法规")
        self._non_active_cache = (mtime, ids)
        return ids

    def _excluded_law_filter(self) -> Optional[Dict]:
        """构造排除失效法规的 ES 过滤子句，无失效法规时返回 None"""
        ids = self._non_active_law_ids()
        return {"terms": {"law_id": ids}} if ids else None

    def _law_title_index(self) -> Dict[str, str]:
        """{法规标题简体归一形式: law_id}，进程内缓存（法规仅数百部，聚合一次即可）。

        归一固定为繁→简：反向存在多候选，无法由简体标题反推存储的繁体标题。
        """
        if self._law_title_index_cache is None:
            index: Dict[str, str] = {}
            try:
                for law in self.get_all_laws():
                    law_id = law.get("law_id")
                    law_title = law.get("title") or ""
                    if law_id and law_title:
                        index.setdefault(zh_convert(law_title, "t2s"), law_id)
            except Exception as e:
                # 构建失败不写入缓存，避免一次偶发失败把该能力永久关掉
                logger.warning(f"[法规名定位] 标题索引构建失败: {e}")
                return index
            self._law_title_index_cache = index
        return self._law_title_index_cache

    def _resolve_law_id(self, title: str) -> Optional[str]:
        """把法规名解析为 law_id：先简体归一后精确匹配，再退化为双向包含（取最长者，保证确定性）"""
        normalized = zh_convert(title, "t2s")
        if not normalized:
            return None
        index = self._law_title_index()
        if normalized in index:
            return index[normalized]
        candidates = [
            (len(key), law_id) for key, law_id in index.items()
            if key and (key in normalized or normalized in key)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]

    def find_article(self, title: str, article_number: str) -> Optional[Dict]:
        """按「法规名 + 条款号」精确定位一条条款，繁简写法均可。

        先把法规名简体归一解析为 law_id，再按 (law_id, article_number) 精确查询。
        不用 title.keyword/match 兜底：简→繁为一对多拼不出真实繁体标题，且 IK 下
        match title 是 OR 语义，返回结果不确定。
        """
        try:
            law_id = self._resolve_law_id(title)
            if not law_id:
                return None
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"law_id": law_id}},
                            {"term": {"article_number": article_number}}
                        ]
                    }
                },
                "size": 1,
                "_source": True
            }
            response = self.client.search(index=settings.es_index_name, body=query_body)
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                return None
            hit = hits[0]
            source = hit.get("_source", {})
            metadata = source.get("metadata", {}).copy()
            metadata["law_id"] = source.get("law_id", "")
            metadata["title"] = source.get("title", "")
            metadata["jurisdiction"] = source.get("jurisdiction", "")
            metadata["article_number"] = source.get("article_number", "")
            metadata["chunk_index"] = source.get("chunk_index", 0)
            metadata["topics"] = source.get("topics", [])
            metadata["topic_labels"] = source.get("topic_labels", [])
            return {
                "content": source.get("content", ""),
                "title": metadata["title"],
                "article_number": metadata["article_number"],
                "jurisdiction": _jurisdiction_to_zh(metadata.get("jurisdiction", "")),
                "score": 1.0,
                "law_id": metadata["law_id"],
                "source_file": source.get("source_file", ""),
                "chunk_index": metadata["chunk_index"],
                "topics": metadata["topics"],
                "topic_labels": metadata["topic_labels"],
            }
        except Exception as e:
            return None

    def get(self, doc_id: str) -> Optional[Dict]:
        try:
            response = self.client.get(index=settings.es_index_name, id=doc_id)
            return response.get("_source", {})
        except NotFoundError:
            return None

    def get_article_full_text(self, law_id: str, article_number: str) -> Optional[str]:
        """按 law_id + article_number 聚合该 article 的全部 chunk，按序拼成完整文本（供回取上下文）。"""
        if not law_id or not article_number:
            return None
        try:
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"law_id": law_id}},
                            {"term": {"article_number": article_number}}
                        ]
                    }
                },
                "sort": [{"chunk_index": "asc"}],
                "size": 1000,
                "_source": ["content", "chunk_index"]
            }
            response = self.client.search(index=settings.es_index_name, body=query_body)
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                return None
            chunks = [(hit["_source"].get("chunk_index", 0), hit["_source"].get("content", "")) for hit in hits]
            chunks.sort(key=lambda x: x[0])
            return _join_chunks_without_overlap([content for _, content in chunks])
        except Exception:
            return None

    def get_articles_full_text_batch(self, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
        """批量回取多个 article 的完整文本（单次 msearch，把 N 次往返压缩为 1 次）。

        返回 {(law_id, article_number): 完整文本}，未命中的键不出现。
        """
        keys = [(lid, an) for lid, an in pairs if lid and an]
        if not keys:
            return {}

        body: List[Dict] = []
        for law_id, article_number in keys:
            body.append({"index": settings.es_index_name})
            body.append({
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"law_id": law_id}},
                            {"term": {"article_number": article_number}}
                        ]
                    }
                },
                "sort": [{"chunk_index": "asc"}],
                "size": 1000,
                "_source": ["content", "chunk_index"]
            })

        try:
            response = self.client.msearch(body=body)
        except Exception:
            return {}

        result: Dict[Tuple[str, str], str] = {}
        for key, item in zip(keys, response.get("responses", [])):
            hits = item.get("hits", {}).get("hits", [])
            if not hits:
                continue
            chunks = [(h["_source"].get("chunk_index", 0), h["_source"].get("content", "")) for h in hits]
            chunks.sort(key=lambda x: x[0])
            text = _join_chunks_without_overlap([content for _, content in chunks])
            if text:
                result[key] = text
        return result

    def get_articles_docs_batch(self, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Dict]:
        """批量回取多个 article 的完整文档（正文 + 元信息，单次 msearch）。

        比 get_articles_full_text_batch 多返回 title/jurisdiction/topics 等，供引用链扩展
        直接构造检索结果项；未命中的键不出现。
        """
        keys = [(lid, an) for lid, an in pairs if lid and an]
        if not keys:
            return {}

        body: List[Dict] = []
        for law_id, article_number in keys:
            body.append({"index": settings.es_index_name})
            body.append({
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"law_id": law_id}},
                            {"term": {"article_number": article_number}}
                        ]
                    }
                },
                "sort": [{"chunk_index": "asc"}],
                "size": 1000,
                "_source": ["content", "chunk_index", "title", "jurisdiction",
                            "law_id", "article_number", "source_file",
                            "topics", "topic_labels"]
            })

        try:
            response = self.client.msearch(body=body)
        except Exception:
            return {}

        result: Dict[Tuple[str, str], Dict] = {}
        for key, item in zip(keys, response.get("responses", [])):
            hits = item.get("hits", {}).get("hits", [])
            if not hits:
                continue
            source = hits[0].get("_source", {})
            chunks = [(h["_source"].get("chunk_index", 0), h["_source"].get("content", "")) for h in hits]
            chunks.sort(key=lambda x: x[0])
            content = _join_chunks_without_overlap([text for _, text in chunks])
            if not content:
                continue
            result[key] = {
                "content": content,
                "title": source.get("title", ""),
                "article_number": source.get("article_number", ""),
                "jurisdiction": source.get("jurisdiction", ""),
                "law_id": source.get("law_id", ""),
                "source_file": source.get("source_file", ""),
                "chunk_index": source.get("chunk_index", 0),
                "topics": source.get("topics", []),
                "topic_labels": source.get("topic_labels", []),
            }
        return result

    def get_referenced_articles(self, seeds: List[Tuple[str, str, str]],
                                max_refs: int = 5) -> List[Dict]:
        """引用链单跳扩展（组件D）：回取种子条款引用到的其他条款，不额外调用 LLM。

        引用来源：同法规内引用（parse_intra_law_refs，优先级更高）与跨法规引用
        （parse_cross_law_refs，受 settings.reference_expand_cross_law_enabled 控制）。
        seeds 为 [(law_id, article_number, content), ...]，max_refs 控制上下文膨胀。

        返回 get_articles_docs_batch 形状的文档，每项额外带 retrieval_hop=2、
        referenced_by（引用方条款号）、referenced_law_id；跨法规项另带 cross_law=True。
        已在 seeds 中的条款不重复返回，回取不到的目标静默跳过。
        """
        if not seeds or max_refs <= 0:
            return []

        seed_keys = {(law_id, str(article_number)) for law_id, article_number, _ in seeds if law_id}

        # 按「同法规内 → 跨法规」收集候选后轮转分配配额：每种子每轮最多取 1 条，
        # 避免排名第一的种子独占配额、挤掉后续种子的第二跳目标。
        per_seed: List[List[Tuple[Tuple[str, str], Tuple[str, str], bool]]] = []
        for law_id, article_number, content in seeds:
            if not law_id:
                continue
            items: List[Tuple[Tuple[str, str], Tuple[str, str], bool]] = []
            for ref in parse_intra_law_refs(content, article_number):
                items.append(((law_id, ref), (law_id, str(article_number)), False))
            if settings.reference_expand_cross_law_enabled:
                for target_law, target_art in parse_cross_law_refs(content, article_number, law_id):
                    items.append(((target_law, target_art), (law_id, str(article_number)), True))
            if items:
                per_seed.append(items)

        targets: List[Tuple[str, str]] = []
        origin: Dict[Tuple[str, str], Tuple[str, str]] = {}
        cross_flag: Dict[Tuple[str, str], bool] = {}
        seen = set()
        cursors = [0] * len(per_seed)
        while len(targets) < max_refs and per_seed:
            advanced = False
            for i, items in enumerate(per_seed):
                while cursors[i] < len(items):
                    key, src, is_cross = items[cursors[i]]
                    cursors[i] += 1
                    if key in seed_keys or key in seen:
                        continue
                    seen.add(key)
                    targets.append(key)
                    origin[key] = src
                    cross_flag[key] = is_cross
                    advanced = True
                    break
                if len(targets) >= max_refs:
                    break
            if not advanced:
                break

        docs = self.get_articles_docs_batch(targets)
        results = []
        for key in targets:
            doc = docs.get(key)
            if not doc:
                continue
            source_law_id, source_article = origin[key]
            doc["retrieval_hop"] = 2
            doc["referenced_by"] = source_article
            doc["referenced_law_id"] = source_law_id
            if cross_flag.get(key):
                doc["cross_law"] = True
            results.append(doc)
        return results

    def update(self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None):
        update_body = {}
        if content:
            update_body["doc"] = {"content": content}
            if content:
                embedding = self._embed_texts([content])[0]
                update_body["doc"]["embedding"] = embedding
        if metadata:
            if "doc" not in update_body:
                update_body["doc"] = {}
            update_body["doc"]["metadata"] = metadata
        
        if update_body:
            self.client.update(index=settings.es_index_name, id=doc_id, body=update_body)
    
    def delete(self, doc_id: str):
        self.client.delete(index=settings.es_index_name, id=doc_id)
    
    def count(self) -> int:
        response = self.client.count(index=settings.es_index_name)
        return response.get("count", 0)
    
    def clear(self):
        index_name = settings.es_index_name
        # 先判断名称是否为 alias：是则删除其指向的真实索引，再删除 alias 本身
        is_alias = False
        concrete_indices = []
        try:
            alias_info = self.client.indices.get_alias(name=index_name)
            is_alias = True
            concrete_indices = list(alias_info.keys())
        except NotFoundError:
            is_alias = False

        if is_alias and concrete_indices:
            self.client.indices.delete(index=",".join(concrete_indices))
            self.client.indices.delete_alias(index=concrete_indices, name=index_name)
        elif self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

        self._create_index_if_not_exists()
    
    def get_all_laws(self, jurisdiction: Optional[str] = None, topics: Optional[str] = None) -> List[Dict]:
        """获取所有法规列表（按law_id聚合）"""
        query = {
            "size": 0,
            "aggs": {
                "laws": {
                    "terms": {
                        "field": "law_id",
                        "size": 10000
                    },
                    "aggs": {
                        "title": {"terms": {"field": "title.keyword", "size": 1}},
                        "jurisdiction": {"terms": {"field": "jurisdiction", "size": 1}},
                        "chunk_count": {"value_count": {"field": "chunk_index"}},
                        # 条款数按 article_number 去重
                        "article_count": {
                            "cardinality": {
                                "field": "article_number"
                            }
                        }
                    }
                }
            }
        }
        
        # 构建查询条件（AND 逻辑）
        must_clauses = []
        if jurisdiction:
            aliases = jurisdiction_aliases(jurisdiction)
            must_clauses.append({"bool": {"should": [{"term": {"jurisdiction": alias}} for alias in aliases],
                                          "minimum_should_match": 1}})
        if topics:
            for topic_label in [t.strip() for t in topics.split(",") if t.strip()]:
                must_clauses.append({"term": {"topic_labels": topic_label}})
        
        if must_clauses:
            query["query"] = {"bool": {"must": must_clauses}}
        
        response = self.client.search(index=settings.es_index_name, body=query)
        
        laws = []
        for bucket in response.get("aggregations", {}).get("laws", {}).get("buckets", []):
            law_id = bucket.get("key")
            title_buckets = bucket.get("title", {}).get("buckets", [])
            title = title_buckets[0].get("key") if title_buckets else ""
            jur_buckets = bucket.get("jurisdiction", {}).get("buckets", [])
            jur = jur_buckets[0].get("key") if jur_buckets else ""
            chunk_count = bucket.get("chunk_count", {}).get("value", 0)
            article_count = bucket.get("article_count", {}).get("value", 0)
            
            laws.append({
                "law_id": law_id,
                "title": title,
                "jurisdiction": _jurisdiction_to_zh(jur),
                "chunk_count": chunk_count,
                "article_count": article_count if article_count > 0 else chunk_count
            })
        
        return laws
    
    def get_law_meta(self, law_id: str) -> Optional[Dict]:
        """取法规级元数据（轻量）：单条文档的字段投影，不拉正文。

        供 Agent 判断法规是否现行有效、法域与通过/生效日期；法规不存在时返回 None。
        """
        if not law_id:
            return None
        query = {
            "size": 1,
            "query": {"term": {"law_id": law_id}},
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["title", "jurisdiction", "status", "passing_date",
                        "effective_date", "publication_date", "topics", "metadata.title"],
        }
        try:
            response = self.client.search(index=settings.es_index_name, body=query)
        except Exception as e:
            logger.warning(f"[法规元数据] 获取失败 {law_id}: {e}")
            return None

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return None
        source = hits[0].get("_source", {}) or {}
        meta = source.get("metadata") or {}
        return {
            "law_id": law_id,
            "title": source.get("title") or meta.get("title") or "",
            "jurisdiction": _jurisdiction_to_zh(source.get("jurisdiction", "")),
            "status": source.get("status", "") or "",
            "passing_date": source.get("passing_date") or "",
            "effective_date": source.get("effective_date") or "",
            "publication_date": source.get("publication_date") or "",
            "topics": source.get("topics") or [],
        }

    def get_law_outline(self, law_id: str) -> List[Dict]:
        """获取法规条款目录（轻量）：每个 article_number 取最小 chunk_index 的一条。

        ES 侧用 collapse 去重，避免为目录拉取整部法规正文。
        """
        query = {
            "size": 5000,
            "query": {"term": {"law_id": law_id}},
            "collapse": {"field": "article_number"},
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["article_number", "chunk_index", "chunk_type", "metadata.article_title", "metadata.part_title"],
        }

        try:
            response = self.client.search(index=settings.es_index_name, body=query)
        except Exception as e:
            logger.warning(f"[目录] 获取法规条款目录失败 {law_id}: {e}")
            return []

        outline = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            meta = source.get("metadata", {}) or {}
            outline.append({
                "article_number": source.get("article_number", ""),
                "article_title": meta.get("article_title", ""),
                # 所属编/章/节：供前端按章节聚合目录
                "part_title": meta.get("part_title", ""),
                "chunk_index": source.get("chunk_index", 0),
                # 条目类型：substantive 正文条款 / schedule 附表 / fragment 解析遗留片段
                "chunk_type": source.get("chunk_type", "") or "",
            })
        return outline

    def get_law_by_id(
        self,
        law_id: str,
        page: int = 1,
        page_size: int = 1000,
        include_outline: bool = False
    ) -> Optional[Dict]:
        """根据 law_id 分页获取法规详情（chunk_index 升序）。

        一次性取前 1000 块会让超长法规被静默截断，故改为分页；page_size 默认 1000 与原
        上限一致，保证调用方行为不变。include_outline 附带轻量条款目录。
        """
        try:
            page = max(1, int(page or 1))
        except (TypeError, ValueError):
            page = 1
        try:
            page_size = min(max(1, int(page_size or 1000)), 2000)
        except (TypeError, ValueError):
            page_size = 1000

        offset = (page - 1) * page_size

        # ES 默认 max_result_window=10000，from+size 超出会直接抛错；
        # 此处显式拦截并给出可读提示，避免长法规翻页到边界时返回 500。
        if offset + page_size > 10000:
            raise ValueError(
                f"页码超出可访问范围（每页 {page_size} 块，最多可翻到第 {10000 // page_size} 页）"
            )

        query = {
            "query": {"term": {"law_id": law_id}},
            "from": offset,
            "size": page_size,
            "sort": [{"chunk_index": {"order": "asc"}}],
            "track_total_hits": True,
            "_source": [
                "content", "article_number", "chunk_index", "chunk_type", "title", "jurisdiction",
                "passing_date", "effective_date", "publication_date", "topics", "metadata",
            ],
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])
        total_chunks = response.get("hits", {}).get("total", {}).get("value", 0)

        if not hits:
            if total_chunks == 0:
                return None
            # 页码超出该法规实际块数：额外取首块补齐元数据，正文返回空列表
            fallback = self.client.search(
                index=settings.es_index_name,
                body={
                    "query": {"term": {"law_id": law_id}},
                    "size": 1,
                    "sort": [{"chunk_index": {"order": "asc"}}],
                    "_source": ["title", "jurisdiction", "passing_date", "effective_date",
                                "publication_date", "topics", "metadata"],
                },
            )
            meta_hits = fallback.get("hits", {}).get("hits", [])
            if not meta_hits:
                return None
        else:
            meta_hits = hits

        chunks = []
        for hit in hits:
            source = hit.get("_source", {})
            meta = source.get("metadata", {}) or {}
            chunks.append({
                "content": source.get("content", ""),
                "article_number": source.get("article_number", ""),
                # 条款标题/所属篇章位于 metadata 子对象中（顶层无该字段）
                "article_title": meta.get("article_title", ""),
                "part_title": meta.get("part_title", ""),
                "chunk_index": source.get("chunk_index", 0),
                # 条目类型：供详情页把附表/解析碎片与正文条款分区渲染
                "chunk_type": source.get("chunk_type", "") or "",
            })

        first_source = meta_hits[0].get("_source", {})
        first_meta = first_source.get("metadata", {}) or {}

        result = {
            "law_id": law_id,
            "title": first_source.get("title", ""),
            "title_original": first_meta.get("title", ""),
            "jurisdiction": _jurisdiction_to_zh(first_source.get("jurisdiction", "")),
            "passing_date": first_source.get("passing_date", ""),
            "effective_date": first_source.get("effective_date", ""),
            "publication_date": first_source.get("publication_date", ""),
            "topics": first_source.get("topics", []) or [],
            "page": page,
            "page_size": page_size,
            "total_chunks": total_chunks,
            "has_more": offset + len(chunks) < total_chunks,
            "chunks": chunks,
        }

        if include_outline:
            result["outline"] = self.get_law_outline(law_id)
            result["total_articles"] = len(result["outline"])

        return result
    
    def delete_by_law_id(self, law_id: str) -> int:
        """根据law_id批量删除所有相关文档"""
        query = {
            "query": {"term": {"law_id": law_id}}
        }

        response = self.client.delete_by_query(index=settings.es_index_name, body=query, refresh=True)
        return response.get("deleted", 0)

    def exists_law_id(self, law_id: str) -> bool:
        """检查数据库中是否已存在指定law_id的文档"""
        query = {
            "size": 1,
            "query": {"term": {"law_id": law_id}}
        }

        try:
            response = self.client.search(index=settings.es_index_name, body=query)
            return response.get("hits", {}).get("total", {}).get("value", 0) > 0
        except Exception:
            return False

    def get_law_chunk_count(self, law_id: str) -> int:
        """获取指定law_id的文档块数量"""
        query = {
            "size": 0,
            "query": {"term": {"law_id": law_id}}
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        return response.get("hits", {}).get("total", {}).get("value", 0)
    
    def query_with_filters(self, n_results: int = 10, where: Optional[Dict] = None) -> Dict:
        """仅按筛选条件检索，不进行语义搜索"""
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        
        search_body = {}

        if where:
            filter_clauses = []
            for k, v in where.items():
                if isinstance(v, list) and len(v) > 1:
                    filter_clauses.append({
                        "bool": {
                            "should": [{"term": {k: item}} for item in v],
                            "minimum_should_match": 1
                        }
                    })
                else:
                    actual_value = v[0] if isinstance(v, list) else v
                    filter_clauses.append({"term": {k: actual_value}})
            search_body["query"] = {"bool": {"filter": filter_clauses}}
        else:
            search_body["query"] = {"match_all": {}}
        
        search_body["size"] = n_results
        search_body["sort"] = [{"law_id": {"order": "asc"}}, {"chunk_index": {"order": "asc"}}]
        
        response = self.client.search(index=settings.es_index_name, body=search_body)
        
        hits = response.get("hits", {}).get("hits", [])
        query_ids, query_docs, query_metas, query_distances = [], [], [], []
        
        for hit in hits:
            source = hit.get("_source", {})
            query_ids.append(hit.get("_id"))
            query_docs.append(source.get("content", ""))
            metadata = source.get("metadata", {}).copy()
            metadata["law_id"] = source.get("law_id", "")
            metadata["title"] = source.get("title", "")
            metadata["jurisdiction"] = source.get("jurisdiction", "")
            metadata["article_number"] = source.get("article_number", "")
            metadata["chunk_index"] = source.get("chunk_index", 0)
            metadata["topic"] = source.get("topic", "")
            metadata["topics"] = source.get("topics", [])
            metadata["topic_labels"] = source.get("topic_labels", [])
            query_metas.append(metadata)
            query_distances.append(0.0)
        
        results["ids"].append(query_ids)
        results["documents"].append(query_docs)
        results["metadatas"].append(query_metas)
        results["distances"].append(query_distances)
        
        return results
    
    def get_all_jurisdictions(self) -> List[str]:
        """获取所有法域列表（返回简体中文）"""
        query = {
            "size": 0,
            "aggs": {
                "jurisdictions": {
                    "terms": {
                        "field": "jurisdiction",
                        "size": 100
                    }
                }
            }
        }
        
        response = self.client.search(index=settings.es_index_name, body=query)
        buckets = response.get("aggregations", {}).get("jurisdictions", {}).get("buckets", [])
        
        zh_jurisdictions = set()
        result = []
        for bucket in buckets:
            jur = bucket.get("key")
            zh_jur = _jurisdiction_to_zh(jur)
            if zh_jur not in zh_jurisdictions:
                zh_jurisdictions.add(zh_jur)
                result.append(zh_jur)
        
        return result

    def sample_documents(self, jurisdiction: str, size: int = 200,
                         fields: Optional[List[str]] = None) -> List[Dict]:
        """按法域随机采样文档，用于构建法域语义向量。

        fields 默认 ['title', 'content']；返回 [{'title': ..., 'content': ...}, ...]。
        """
        if fields is None:
            fields = ['title', 'content']

        query = {
            "size": size,
            "query": {
                "function_score": {
                    "query": {"bool": {"should": [{"term": {"jurisdiction": alias}}
                                                  for alias in jurisdiction_aliases(jurisdiction)],
                                       "minimum_should_match": 1}},
                    "random_score": {"seed": int(time.time()), "field": "_seq_no"},
                    "boost_mode": "replace"
                }
            },
            "_source": fields
        }

        try:
            response = self.client.search(index=settings.es_index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])
            return [hit["_source"] for hit in hits]
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"采样法域 {jurisdiction} 文档失败: {e}")
            return []

    def get_all_topics(self) -> List[Dict]:
        """获取所有主题列表（含 doc_count 与各法域分布），按 count 降序。

        返回 [{"id", "label_zh", "count", "by_jurisdiction"}, ...]。
        """
        query = {
            "size": 0,
            "aggs": {
                "topics": {
                    "terms": {
                        "field": "topic_labels",
                        "size": 100
                    },
                    "aggs": {
                        "jurisdiction_breakdown": {
                            "terms": {"field": "jurisdiction", "size": 10}
                        }
                    }
                }
            }
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        buckets = response.get("aggregations", {}).get("topics", {}).get("buckets", [])

        result = []
        for bucket in buckets:
            raw_label = bucket.get("key", "")
            count = bucket.get("doc_count", 0)

            # 英文 ID 转中文标签
            label_zh = _to_label_zh(raw_label) if raw_label and raw_label[0].islower() and '_' in raw_label else raw_label

            if is_excluded_topic(raw_label) or is_excluded_topic(label_zh):
                continue

            by_jur = {}
            for jur_bucket in bucket.get("jurisdiction_breakdown", {}).get("buckets", []):
                by_jur[jur_bucket["key"]] = jur_bucket["doc_count"]

            result.append({
                "id": _label_zh_to_topic_id(label_zh),
                "label_zh": label_zh,
                "count": count,
                "by_jurisdiction": by_jur
            })

        # 按 count 降序排列
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def get_topic_filter_values(self) -> List[str]:
        """获取所有可用的主题标签值（用于下拉筛选的简单列表，返回中文）"""
        query = {
            "size": 0,
            "aggs": {
                "topics": {
                    "terms": {
                        "field": "topic_labels",
                        "size": 100
                    }
                }
            }
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        buckets = response.get("aggregations", {}).get("topics", {}).get("buckets", [])

        result = []
        for bucket in buckets:
            raw_label = bucket.get("key", "")
            if not raw_label:
                continue
            # 英文 ID 转中文标签
            label_zh = _to_label_zh(raw_label) if raw_label[0].islower() and '_' in raw_label else raw_label
            if is_excluded_topic(raw_label) or is_excluded_topic(label_zh):
                continue
            result.append(label_zh)
        return result

    def get_topic_frequencies(self) -> Dict[str, int]:
        """获取所有主题标签的出现频次（用于 IDF 加权），如 {"数据保护": 120, ...}"""
        query = {
            "size": 0,
            "aggs": {
                "topics": {
                    "terms": {
                        "field": "topic_labels",
                        "size": 200
                    }
                }
            }
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        buckets = response.get("aggregations", {}).get("topics", {}).get("buckets", [])

        return {bucket.get("key"): bucket.get("doc_count", 0) for bucket in buckets if bucket.get("key")}

    def batch_get_law_topic_distributions(self, law_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """批量获取多个法规的主题分布（单次聚合，不拉取向量/正文）。

        返回 {law_id: {"数据保护": 12, ...}}，替代逐法规拉取整部 embedding 的 N+1 查询。
        """
        if not law_ids:
            return {}

        query = {
            "size": 0,
            "query": {"terms": {"law_id": law_ids}},
            "aggs": {
                "laws": {
                    "terms": {"field": "law_id", "size": max(len(law_ids), 10)},
                    "aggs": {
                        "topics": {"terms": {"field": "topic_labels", "size": 50}}
                    }
                }
            }
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        result: Dict[str, Dict[str, int]] = {}
        for law_bucket in response.get("aggregations", {}).get("laws", {}).get("buckets", []):
            law_id = law_bucket.get("key")
            topic_buckets = law_bucket.get("topics", {}).get("buckets", [])
            result[law_id] = {
                t.get("key"): t.get("doc_count", 0)
                for t in topic_buckets if t.get("key")
            }
        return result

    def get_law_chunk_vectors(self, law_id: str) -> List[Dict]:
        """获取指定法规所有 chunk 的向量及元数据（不返回 content，仅必要字段）"""
        query = {
            "query": {"term": {"law_id": law_id}},
            "size": 1000,
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["embedding", "law_id", "title", "jurisdiction", "chunk_index", "topic_labels"]
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            return []

        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "embedding": source.get("embedding", []),
                "law_id": source.get("law_id", ""),
                "title": source.get("title", ""),
                "jurisdiction": source.get("jurisdiction", ""),
                "chunk_index": source.get("chunk_index", 0),
                "topic_labels": source.get("topic_labels", []),
            })

        return results

    # 法规质心（预计算，供相似法规推荐使用，避免运行时拉取整部法规向量）
    CENTROID_INDEX = "legal_law_centroids"

    def ensure_centroid_index(self) -> bool:
        """确保质心索引存在（幂等，进程内缓存就绪标记）"""
        if getattr(self, "_centroid_index_ready", False):
            return True
        try:
            if not self.client.indices.exists(index=self.CENTROID_INDEX):
                self.client.indices.create(
                    index=self.CENTROID_INDEX,
                    body={
                        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                        "mappings": {
                            "properties": {
                                "law_id": {"type": "keyword"},
                                # 质心仅作查询向量下发，无需被检索，故 index=false 省开销
                                "centroid": {"type": "dense_vector", "dims": 1024, "index": False},
                                "jurisdiction": {"type": "keyword"},
                                "title": {"type": "keyword"},
                                "chunk_count": {"type": "integer"},
                                "topic_distribution": {"type": "object", "enabled": False},
                                "updated_at": {"type": "date"},
                            }
                        },
                    },
                )
            self._centroid_index_ready = True
            return True
        except Exception as e:
            logger.warning(f"[Centroid] 索引创建/检查失败: {e}")
            return False

    def save_law_centroid(
        self,
        law_id: str,
        centroid: List[float],
        jurisdiction: str = "",
        title: str = "",
        chunk_count: int = 0,
        topic_distribution: Optional[Dict] = None,
    ) -> bool:
        """写入/覆盖指定法规的质心文档（doc_id = law_id）"""
        if not self.ensure_centroid_index():
            return False
        from datetime import datetime, timezone
        try:
            self.client.index(
                index=self.CENTROID_INDEX,
                id=law_id,
                body={
                    "law_id": law_id,
                    "centroid": centroid,
                    "jurisdiction": jurisdiction,
                    "title": title,
                    "chunk_count": chunk_count,
                    "topic_distribution": topic_distribution or {},
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                refresh=True,
            )
            return True
        except Exception as e:
            logger.warning(f"[Centroid] 写入失败 {law_id}: {e}")
            return False

    def get_law_centroid(self, law_id: str) -> Optional[Dict]:
        """读取指定法规的质心文档，不存在则返回 None"""
        if not self.ensure_centroid_index():
            return None
        try:
            resp = self.client.get(index=self.CENTROID_INDEX, id=law_id)
            return resp.get("_source", {})
        except NotFoundError:
            return None
        except Exception as e:
            logger.warning(f"[Centroid] 读取失败 {law_id}: {e}")
            return None

    def delete_law_centroid(self, law_id: str) -> bool:
        """删除指定法规的质心文档（不存在时静默成功）"""
        if not self.ensure_centroid_index():
            return False
        try:
            self.client.delete(index=self.CENTROID_INDEX, id=law_id, refresh=True)
            return True
        except NotFoundError:
            return True
        except Exception as e:
            logger.warning(f"[Centroid] 删除失败 {law_id}: {e}")
            return False

    def get_law_chunk_indices(self, law_id: str) -> List[int]:
        """获取指定法规所有 chunk 的 chunk_index（升序，仅投影该字段，开销极低）"""
        query = {
            "query": {"term": {"law_id": law_id}},
            "size": 10000,
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["chunk_index"],
        }
        response = self.client.search(index=settings.es_index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])
        return [h.get("_source", {}).get("chunk_index", 0) for h in hits]

    def get_law_chunks_by_indices(self, law_id: str, indices: List[int]) -> List[Dict]:
        """按 chunk_index 精确回取指定法规的少量 chunk（含 embedding），用于质心采样"""
        if not indices:
            return []
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"law_id": law_id}},
                        {"terms": {"chunk_index": indices}},
                    ]
                }
            },
            "size": len(indices),
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["embedding", "chunk_index", "title", "jurisdiction"],
        }
        response = self.client.search(index=settings.es_index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "embedding": source.get("embedding", []),
                "chunk_index": source.get("chunk_index", 0),
                "title": source.get("title", ""),
                "jurisdiction": source.get("jurisdiction", ""),
            })
        return results

    def knn_search_similar(
        self,
        query_vector: List[float],
        top_k: int = 20,
        exclude_law_id: Optional[str] = None,
        jurisdiction_filter: Optional[str] = None,
        min_score: float = 0.3
    ) -> List[Dict]:
        """用给定向量执行 KNN 相似搜索，返回去重前的原始 chunk 结果。

        法域过滤与自身排除都放入 knn.filter 做预过滤：post_filter 是「先取 k 条再排除」，
        大法规自身即可占满候选名额导致结果为空；放入 filter 则不占名额。
        """
        # 构建 KNN 预过滤器（在向量搜索前收窄候选集）
        filter_clauses = []
        if jurisdiction_filter:
            aliases = jurisdiction_aliases(jurisdiction_filter)
            filter_clauses.append({"bool": {"should": [{"term": {"jurisdiction": alias}} for alias in aliases],
                                            "minimum_should_match": 1}})

        must_not_clauses = []
        if exclude_law_id:
            must_not_clauses.append({"term": {"law_id": exclude_law_id}})

        knn_body = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k * 5,
            # HNSW 搜索开销与 num_candidates 近似成正比，故取 max(10*k, 100) 即可
            "num_candidates": max(top_k * 10, 100)
        }
        if filter_clauses or must_not_clauses:
            knn_body["filter"] = {"bool": {
                "filter": filter_clauses,
                "must_not": must_not_clauses
            }}

        search_body = {"knn": knn_body, "size": top_k * 5}

        response = self.client.search(index=settings.es_index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])

        results = []
        for hit in hits:
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            similarity = float(score)

            if similarity < min_score:
                continue

            results.append({
                "id": hit.get("_id"),
                "law_id": source.get("law_id", ""),
                "title": source.get("title", ""),
                "jurisdiction": source.get("jurisdiction", ""),
                "article_number": source.get("article_number", ""),
                "content_preview": source.get("content", "")[:150] if source.get("content") else "",
                "similarity": similarity,
            })

        return results

    def get_topic_statistics(self, limit: int = 50) -> List[Dict]:
        """获取主题统计（含各法域分布与示例法规），用于看板的词云、热力图。"""
        query = {
            "size": 0,
            "aggs": {
                "topics": {
                    "terms": {
                        "field": "topics",
                        "size": limit
                    },
                    "aggs": {
                        "by_jurisdiction": {
                            "terms": {
                                "field": "jurisdiction",
                                "size": 10
                            }
                        }
                    }
                }
            }
        }

        try:
            response = self.client.search(index=settings.es_index_name, body=query)
            topic_buckets = response.get("aggregations", {}).get("topics", {}).get("buckets", [])

            result = []
            for bucket in topic_buckets:
                topic_name = bucket.get("key", "")
                total_count = bucket.get("doc_count", 0)

                if is_excluded_topic(topic_name) or is_excluded_topic(_to_label_zh(topic_name)):
                    continue

                jur_buckets = bucket.get("by_jurisdiction", {}).get("buckets", [])
                by_jurisdiction = {b.get("key"): b.get("doc_count") for b in jur_buckets}

                # 每个主题取前 3 条示例法规
                sample_query = {
                    "size": 3,
                    "query": {"term": {"topic_labels": topic_name}},
                    "_source": ["law_id", "title", "jurisdiction"]
                }
                sample_response = self.client.search(index=settings.es_index_name, body=sample_query)
                samples = []
                for hit in sample_response.get("hits", {}).get("hits", []):
                    source = hit.get("_source", {})
                    samples.append({
                        "law_id": source.get("law_id"),
                        "law_name_zh": source.get("title"),
                        "jurisdiction": source.get("jurisdiction")
                    })

                result.append({
                    "topic": topic_name,
                    "label_zh": _to_label_zh(topic_name),
                    "count": total_count,
                    "by_jurisdiction": by_jurisdiction,
                    "laws": samples[:5]
                })

            return result

        except Exception as e:
            print(f"⚠️ 获取主题统计失败: {e}")
            return []

    # 跨法域专题分析缓存（analysis_reports_cache）

    CACHE_INDEX = "analysis_reports_cache"

    def _ensure_cache_index(self):
        """确保缓存索引存在，不存在则自动创建（静默失败）"""
        try:
            if self.client.indices.exists(index=self.CACHE_INDEX):
                return
            self.client.indices.create(
                index=self.CACHE_INDEX,
                body={
                    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                    "mappings": {
                        "properties": {
                            "cache_type": {"type": "keyword"},  # "analysis" | "compare"
                            "topic_id": {"type": "keyword"},
                            "jurisdiction_a": {"type": "keyword"},
                            "jurisdiction_b": {"type": "keyword"},
                            "fingerprint": {"type": "keyword"},
                            "report_data": {"type": "object", "enabled": False},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "cache_version": {"type": "integer"},
                        }
                    },
                }
            )
        except Exception as e:
            print(f"[Cache] 索引创建/检查失败（缓存功能将不可用）: {e}")

    def _cache_doc_id(self, cache_type: str, topic_id: str, jurisdiction_a: str, jurisdiction_b: str) -> str:
        """生成缓存文档 ID"""
        return f"{cache_type}::{topic_id}::{jurisdiction_a}::{jurisdiction_b}"

    def save_analysis_cache(
        self,
        topic_id: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        fingerprint: str,
        report_data: Dict,
        cache_type: str = "analysis",
    ) -> bool:
        """保存/更新缓存（upsert），支持 analysis 与 compare 两种类型。

        写入前先清除旧文档，确保读取到的始终是最新结果。
        """
        import datetime, copy
        self._ensure_cache_index()
        doc_id = self._cache_doc_id(cache_type, topic_id, jurisdiction_a, jurisdiction_b)

        try:
            old_doc = self.client.get(index=self.CACHE_INDEX, id=doc_id, ignore=[404])
            if old_doc.get("found"):
                self.client.delete(index=self.CACHE_INDEX, id=doc_id)
                self.client.indices.refresh(index=self.CACHE_INDEX)
                logger.info(f"[Cache] 已清除旧缓存: {doc_id} (准备写入新数据)")
        except Exception as e:
            logger.warning(f"[Cache] 清除旧缓存失败（不影响后续写入）: {e}")

        # 深拷贝 + 剥离运行时内部字段（不存入缓存）
        cache_data = copy.deepcopy(report_data)
        for key in ("_raw_response_a", "_raw_response_b", "_cache_hit", "_cache_fingerprint", "timing"):
            cache_data.pop(key, None)
        # extraction 中的 _raw_response 也剥离
        ext = cache_data.get("extraction", {})
        if isinstance(ext, dict):
            for side_key in list(ext.keys()):
                if side_key.startswith("_"):
                    ext.pop(side_key, None)
                elif isinstance(ext[side_key], dict):
                    ext[side_key].pop("_raw_response", None)

        # ES 中 topic 字段为 text，不支持对象：compare 传纯字符串，analysis 传入字典
        import json
        topic_data = cache_data.get("topic")
        if isinstance(topic_data, dict):
            cache_data["topic"] = json.dumps(topic_data, ensure_ascii=False)
            logger.info(f"[Cache] topic对象已转换为JSON字符串: {cache_data['topic'][:80]}...")

        try:
            self.client.index(
                index=self.CACHE_INDEX,
                id=doc_id,
                body={
                    "cache_type": cache_type,
                    "topic_id": topic_id,
                    "jurisdiction_a": jurisdiction_a,
                    "jurisdiction_b": jurisdiction_b,
                    "fingerprint": fingerprint,
                    "report_data": cache_data,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "cache_version": 1,
                },
            )
            self.client.indices.refresh(index=self.CACHE_INDEX)

            # 验证写入：立即读回确认
            verify = self.client.get(index=self.CACHE_INDEX, id=doc_id)
            if verify["found"]:
                logger.info(f"[Cache] 已保存并验证[{cache_type}]: {doc_id}, fp={fingerprint[:16]}...")
                return True
            else:
                logger.error(f"[Cache] 写入后验证失败！doc_id={doc_id}")
                return False
        except Exception as e:
            logger.error(f"[Cache] 保存失败: {type(e).__name__}: {e}")
            return False

    def get_analysis_cache(
        self,
        topic_id: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        cache_type: str = "analysis",
    ) -> Optional[Dict]:
        """
        查询缓存。返回 {fingerprint, report_data, updated_at} 或 None。
        topic_id="__all__" 时返回该类型的全部记录列表。
        """
        self._ensure_cache_index()

        # 特殊模式：返回全部记录
        if topic_id == "__all__":
            try:
                resp = self.client.search(
                    index=self.CACHE_INDEX,
                    body={"query": {"term": {"cache_type": cache_type}}, "size": 50},
                )
                return [hit["_source"] for hit in resp["hits"]["hits"]]
            except Exception as e:
                logger.error(f"[Cache] 批量查询失败: {e}")
                return []

        doc_id = self._cache_doc_id(cache_type, topic_id, jurisdiction_a, jurisdiction_b)
        try:
            resp = self.client.get(index=self.CACHE_INDEX, id=doc_id)
            src = resp.get("_source", {})
            return {
                "fingerprint": src.get("fingerprint"),
                "report_data": src.get("report_data"),
                "updated_at": src.get("updated_at"),
            }
        except NotFoundError:
            return None
        except Exception as e:
            print(f"[Cache] 查询失败: {e}")
            return None

    def delete_analysis_cache(
        self,
        topic_id: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        cache_type: str = "analysis",
    ) -> bool:
        """删除指定缓存"""
        doc_id = self._cache_doc_id(cache_type, topic_id, jurisdiction_a, jurisdiction_b)
        try:
            self.client.delete(index=self.CACHE_INDEX, id=doc_id)
            return True
        except NotFoundError:
            return True
        except Exception as e:
            print(f"[Cache] 删除失败: {e}")
            return False

    # 法规摘要缓存层（summary）

    def _summary_cache_doc_id(self, law_id: str) -> str:
        """生成摘要缓存文档 ID"""
        return f"summary::{law_id}::_::_"

    def compute_law_fingerprint(self, law_id: str) -> str:
        """计算法规指纹（标题 + 条款数 + 通过日期），用于检测数据变化。

        公开方法：缓存层与 KG 都用它判断抽取结果是否失效，指纹相同即可复用。
        """
        import hashlib
        try:
            law = self.get_law_by_id(law_id)
            if law:
                chunk_count = len(law.get("chunks", []))
                title = law.get("title", "")
                passing_date = law.get("passing_date", "")
                raw = f"{title}_{chunk_count}_{passing_date}"
                return hashlib.md5(raw.encode("utf-8")).hexdigest()
        except Exception:
            pass
        return ""

    def save_summary_cache(self, law_id: str, summary_data: Dict) -> bool:
        """保存法规摘要缓存（无过期时间，仅在数据变化时失效）"""
        import datetime, copy
        self._ensure_cache_index()
        doc_id = self._summary_cache_doc_id(law_id)
        fingerprint = self.compute_law_fingerprint(law_id)

        try:
            old_doc = self.client.get(index=self.CACHE_INDEX, id=doc_id, ignore=[404])
            if old_doc.get("found"):
                self.client.delete(index=self.CACHE_INDEX, id=doc_id)

            cache_data = copy.deepcopy(summary_data)

            self.client.index(
                index=self.CACHE_INDEX,
                id=doc_id,
                body={
                    "cache_type": "summary",
                    "law_id": law_id,
                    "topic_id": law_id,
                    "jurisdiction_a": "_",
                    "jurisdiction_b": "_",
                    "fingerprint": fingerprint,
                    "report_data": cache_data,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "cache_version": 1,
                },
            )
            self.client.indices.refresh(index=self.CACHE_INDEX)

            verify = self.client.get(index=self.CACHE_INDEX, id=doc_id)
            if verify["found"]:
                logger.info(f"[Cache] 已保存摘要缓存: {doc_id}, fp={fingerprint[:16]}...")
                return True
            else:
                logger.error(f"[Cache] 摘要缓存写入后验证失败！doc_id={doc_id}")
                return False
        except Exception as e:
            logger.error(f"[Cache] 保存摘要缓存失败: {type(e).__name__}: {e}")
            return False

    def get_summary_cache(self, law_id: str) -> Optional[Dict]:
        """
        获取法规摘要缓存：存在且指纹未变时返回摘要数据，否则返回 None（并删除失效缓存）。
        """
        self._ensure_cache_index()
        doc_id = self._summary_cache_doc_id(law_id)
        
        try:
            resp = self.client.get(index=self.CACHE_INDEX, id=doc_id)
            src = resp.get("_source", {})
            cached_fingerprint = src.get("fingerprint", "")
            report_data = src.get("report_data")
            
            if not cached_fingerprint or not report_data:
                return None
            
            current_fingerprint = self.compute_law_fingerprint(law_id)
            if current_fingerprint and cached_fingerprint != current_fingerprint:
                logger.info(f"[Cache] 摘要缓存指纹不匹配，失效。law_id={law_id}")
                self.client.delete(index=self.CACHE_INDEX, id=doc_id)
                return None
            
            logger.info(f"[Cache] 命中摘要缓存: {law_id}")
            return report_data
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"[Cache] 查询摘要缓存失败: {e}")
            return None

    def delete_summary_cache(self, law_id: str) -> bool:
        """删除指定法规的摘要缓存"""
        doc_id = self._summary_cache_doc_id(law_id)
        try:
            self.client.delete(index=self.CACHE_INDEX, id=doc_id)
            return True
        except NotFoundError:
            return True
        except Exception as e:
            logger.error(f"[Cache] 删除摘要缓存失败: {e}")
            return False

    def delete_all_summary_cache(self) -> bool:
        """删除所有摘要缓存"""
        try:
            query = {"query": {"term": {"cache_type": "summary"}}}
            response = self.client.delete_by_query(index=self.CACHE_INDEX, body=query, refresh=True)
            logger.info(f"[Cache] 已删除 {response.get('deleted', 0)} 条摘要缓存")
            return True
        except Exception as e:
            logger.error(f"[Cache] 删除所有摘要缓存失败: {e}")
            return False

    # 法规知识层（KG）持久化：legal_kg_triples / legal_kg_entities
    # 按 law_id 组织数据，新增法规只抽新增部分；三元组 ID 由内容哈希决定（幂等 upsert）；
    # 用 law 指纹判断缓存失效，指纹未变则跳过 LLM 抽取；entities 的 1024 维向量一次性
    # 建好（dims 建索引后不可改），供实体消歧的跨语言向量对齐使用。

    KG_TRIPLE_INDEX = "legal_kg_triples"
    KG_ENTITY_INDEX = "legal_kg_entities"
    # 实体合并候选（被 L3 守卫拦下的高相似实体对），供人工复核；仅辅助展示，不自动归并
    KG_REVIEW_INDEX = "legal_kg_review"

    def _ensure_kg_indices(self) -> bool:
        """确保知识层索引存在，不存在则创建（失败时返回 False，调用方降级为实时抽取）"""
        try:
            if not self.client.indices.exists(index=self.KG_TRIPLE_INDEX):
                self.client.indices.create(
                    index=self.KG_TRIPLE_INDEX,
                    body={
                        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                        "mappings": {
                            "properties": {
                                "law_id": {"type": "keyword"},
                                "jurisdiction": {"type": "keyword"},
                                "fingerprint": {"type": "keyword"},
                                "head_id": {"type": "keyword"},
                                "head_type": {"type": "keyword"},
                                "head_name": {"type": "keyword"},
                                # canonical_* 由实体消歧回填；未消歧时等于 head_id / tail_id
                                "head_canonical_id": {"type": "keyword"},
                                "tail_id": {"type": "keyword"},
                                "tail_type": {"type": "keyword"},
                                "tail_name": {"type": "keyword"},
                                "tail_canonical_id": {"type": "keyword"},
                                "relation": {"type": "keyword"},
                                "source_article": {"type": "keyword"},
                                "confidence": {"type": "float"},
                                "extracted_at": {"type": "date"},
                            }
                        },
                    },
                )
            if not self.client.indices.exists(index=self.KG_ENTITY_INDEX):
                self.client.indices.create(
                    index=self.KG_ENTITY_INDEX,
                    body={
                        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                        "mappings": {
                            "properties": {
                                "entity_id": {"type": "keyword"},
                                "name": {"type": "keyword"},
                                "type": {"type": "keyword"},
                                "aliases": {"type": "keyword"},
                                "jurisdictions": {"type": "keyword"},
                                "embedding": {
                                    "type": "dense_vector",
                                    "dims": 1024,
                                    "index": True,
                                    "similarity": "cosine",
                                },
                                "updated_at": {"type": "date"},
                            }
                        },
                    },
                )
            return True
        except Exception as e:
            logger.warning(f"[KG] 知识层索引创建/检查失败（本次将降级为实时抽取）: {e}")
            return False

    @staticmethod
    def _kg_triple_doc_id(law_id: str, triple: Dict) -> str:
        """三元组文档 ID：由「法规 + 头实体 + 关系 + 尾实体 + 来源条款」哈希得到，保证幂等写入"""
        import hashlib
        head = triple.get("head", {}) or {}
        tail = triple.get("tail", {}) or {}
        raw = (
            f"{law_id}|{head.get('id', '')}|{triple.get('relation', '')}"
            f"|{tail.get('id', '')}|{triple.get('source_article', '')}"
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _kg_entity_doc_id(entity_id: str) -> str:
        """
        实体文档 ID：ES 的 _id 上限 512 字节，超长实体规范名直接用 entity_id 会触发
        「id is too long」使整批写入失败；故仅超长时改用定长哈希，_source.entity_id
        仍保留完整 ID，词典查询与 canonical_id 关联不受影响。
        """
        raw = entity_id.encode("utf-8")
        if len(raw) <= 512:
            return entity_id
        import hashlib
        return "h_" + hashlib.md5(raw).hexdigest()

    def save_kg_triples(
        self,
        law_id: str,
        triples: List[Dict],
        jurisdiction: str = "",
        fingerprint: str = "",
    ) -> int:
        """
        覆盖式保存某法规的三元组：先删旧再批量写入，法规修订后旧关系必须整体失效。
        成功返回写入条数，失败返回 -1（调用方据此降级）。
        """
        if not self._ensure_kg_indices():
            return -1

        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 即使本次抽取为空也要清除，否则修订为「无可提取关系」后旧三元组会残留，
        # 继续参与引用链与实体统计。
        self.delete_kg_triples(law_id)

        if not triples:
            return 0

        actions = []
        for t in triples:
            head = t.get("head", {}) or {}
            tail = t.get("tail", {}) or {}
            actions.append({
                "_index": self.KG_TRIPLE_INDEX,
                "_id": self._kg_triple_doc_id(law_id, t),
                "_source": {
                    "law_id": law_id,
                    "jurisdiction": jurisdiction or t.get("source_jurisdiction", ""),
                    "fingerprint": fingerprint,
                    "head_id": head.get("id", ""),
                    "head_type": head.get("type", ""),
                    "head_name": head.get("name", ""),
                    "head_canonical_id": head.get("canonical_id") or head.get("id", ""),
                    "tail_id": tail.get("id", ""),
                    "tail_type": tail.get("type", ""),
                    "tail_name": tail.get("name", ""),
                    "tail_canonical_id": tail.get("canonical_id") or tail.get("id", ""),
                    "relation": t.get("relation", ""),
                    "source_article": t.get("source_article", ""),
                    "confidence": float(t.get("confidence", 0.8)),
                    "extracted_at": now,
                },
            })

        try:
            from elasticsearch.helpers import bulk
            success, failed = bulk(
                self.client, actions, refresh=True,
                raise_on_error=False, stats_only=True,
            )
            logger.info(
                f"[KG] 三元组已落库: law_id={law_id}, 成功={success}, 失败={failed}, "
                f"fp={(fingerprint or '')[:16]}"
            )
            return int(success)
        except Exception as e:
            logger.error(f"[KG] 三元组落库失败: {type(e).__name__}: {e}")
            return -1

    @staticmethod
    def _kg_triple_from_source(s: Dict) -> Dict:
        """把 legal_kg_triples 的 _source 还原为 kg_extraction_service 的三元组格式"""
        return {
            "head": {
                "id": s.get("head_id", ""),
                "type": s.get("head_type", ""),
                "name": s.get("head_name", ""),
                "canonical_id": s.get("head_canonical_id", ""),
            },
            "relation": s.get("relation", ""),
            "tail": {
                "id": s.get("tail_id", ""),
                "type": s.get("tail_type", ""),
                "name": s.get("tail_name", ""),
                "canonical_id": s.get("tail_canonical_id", ""),
            },
            "law_id": s.get("law_id", ""),
            "source_article": s.get("source_article", ""),
            "source_jurisdiction": s.get("jurisdiction", ""),
            "confidence": s.get("confidence", 0.8),
        }

    def get_all_kg_triples(self, size: int = 20000) -> List[Dict]:
        """
        读取全部三元组，供跨法域聚合等分析场景使用。

        ES 默认 max_result_window=10000，故用 PIT + search_after 分页（PIT 是 _shard_doc
        排序的前置条件）；PIT 不可用时降级为单次读取（上限 10000 条）。
        """
        results: List[Dict] = []
        page_size = 2000

        pit_id = None
        try:
            pit = self.client.open_point_in_time(index=self.KG_TRIPLE_INDEX, keep_alive="1m")
            pit_id = pit.get("id")
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] PIT 打开失败，降级为单次读取: {e}")

        try:
            search_after = None
            while len(results) < size:
                body = {
                    "query": {"match_all": {}},
                    "size": min(page_size, size - len(results)),
                    "track_total_hits": False,
                }
                if pit_id:
                    body["sort"] = [{"_shard_doc": "asc"}]
                    body["pit"] = {"id": pit_id, "keep_alive": "1m"}
                else:
                    # 降级：单次读取，受 max_result_window(10000) 限制
                    body["size"] = min(size, 10000)

                if search_after:
                    body["search_after"] = search_after
                try:
                    resp = self.client.search(body=body)
                except Exception as e:
                    logger.warning(f"[KG] 读取全部三元组失败: {e}")
                    break

                pit_id = resp.get("pit_id") or pit_id
                hits = resp.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for hit in hits:
                    results.append(self._kg_triple_from_source(hit.get("_source", {})))
                search_after = hits[-1].get("sort")
                if not search_after or not pit_id:
                    break  # 降级模式无 sort，读一轮即返回
        finally:
            if pit_id:
                try:
                    self.client.close_point_in_time(body={"id": pit_id})
                except Exception:
                    pass

        return results

    def get_kg_triples(self, law_id: str, size: int = 2000) -> List[Dict]:
        """
        读取某法规的全部三元组，还原为 kg_extraction_service 的输出格式，
        使「读缓存」与「实时抽取」的结果形态一致。
        """
        try:
            resp = self.client.search(
                index=self.KG_TRIPLE_INDEX,
                body={
                    "query": {"term": {"law_id": law_id}},
                    "size": size,
                    "sort": [{"source_article": "asc"}, {"confidence": "desc"}],
                },
            )
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] 读取三元组失败: {e}")
            return []

        return [
            self._kg_triple_from_source(hit.get("_source", {}))
            for hit in resp.get("hits", {}).get("hits", [])
        ]

    def get_kg_triples_by_source_articles(
        self, refs: List[Tuple[str, str]], size: int = 200
    ) -> List[Dict]:
        """回取「由指定条款正文抽取出的」三元组（按 law_id + source_article 精确匹配）。

        以条款为入口，用于把检索命中条款直接接到图谱上：条款节点 canonical_id 由抽取时
        命名决定，不能按条号反推，只能按 source_article 反查。refs 为
        [(law_id, article_number), ...]，size 为返回上限（按置信度降序）。
        """
        pairs = [(str(law).strip(), str(art).strip()) for law, art in refs if law and art]
        if not pairs:
            return []
        should = [
            {"bool": {"filter": [
                {"term": {"law_id": law_id}},
                {"term": {"source_article": article}},
            ]}}
            for law_id, article in pairs
        ]
        try:
            resp = self.client.search(
                index=self.KG_TRIPLE_INDEX,
                body={
                    "query": {"bool": {"should": should, "minimum_should_match": 1}},
                    "size": size,
                    "sort": [{"confidence": "desc"}],
                },
            )
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] 读取条款三元组失败: {e}")
            return []

        return [
            self._kg_triple_from_source(hit.get("_source", {}))
            for hit in resp.get("hits", {}).get("hits", [])
        ]

    def get_kg_triples_by_entities(
        self, canonical_ids: List[str], size: int = 2000
    ) -> List[Dict]:
        """
        读取「头或尾规范实体命中给定集合」的全部三元组，用于实体邻域图构建。

        只按 canonical_id 匹配，故同一实体跨法规/法域的关系会被一并取回，
        这是邻域图能跨法规连通的前提。
        """
        ids = [c for c in canonical_ids if c]
        if not ids:
            return []
        try:
            resp = self.client.search(
                index=self.KG_TRIPLE_INDEX,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"terms": {"head_canonical_id": ids}},
                                {"terms": {"tail_canonical_id": ids}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                    "size": size,
                    "sort": [{"confidence": "desc"}],
                },
            )
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] 读取实体三元组失败: {e}")
            return []

        return [
            self._kg_triple_from_source(hit.get("_source", {}))
            for hit in resp.get("hits", {}).get("hits", [])
        ]

    def get_articles_by_refs(
        self, refs: List[Dict], max_pairs: int = 20, max_chars: int = 600
    ) -> Dict[str, str]:
        """
        按 (law_id, article_number) 批量取条款原文片段（供图谱问答组装证据）。

        refs 为 [{"law_id":..., "article_number":...}]，去重后最多 max_pairs 个，
        单次 msearch 返回 {f"{law_id}::{article_number}": 片段}（截断至 max_chars）。
        """
        pairs: List[tuple] = []
        seen = set()
        for ref in refs or []:
            law_id = (ref.get("law_id") or "").strip()
            article = (ref.get("article_number") or "").strip()
            if not law_id or not article:
                continue
            key = (law_id, article)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) >= max_pairs:
                break
        if not pairs:
            return {}

        body: List[Dict] = []
        for law_id, article in pairs:
            body.append({"index": settings.es_index_name})
            body.append({
                "query": {"bool": {"filter": [
                    {"term": {"law_id": law_id}},
                    {"term": {"article_number": article}},
                ]}},
                "size": 5,
                "_source": ["content", "chunk_index"],
            })

        try:
            resp = self.client.msearch(body=body)
        except NotFoundError:
            return {}
        except Exception as e:
            logger.warning(f"[KG] 批量读取条款原文失败: {e}")
            return {}

        result: Dict[str, str] = {}
        for (law_id, article), item in zip(pairs, resp.get("responses", [])):
            hits = item.get("hits", {}).get("hits", [])
            if not hits:
                continue
            hits.sort(key=lambda h: h.get("_source", {}).get("chunk_index", 0))
            content = "\n".join(h.get("_source", {}).get("content", "") for h in hits)
            if content.strip():
                result[f"{law_id}::{article}"] = content[:max_chars]
        return result

    def get_kg_cache_meta(self, law_id: str) -> Optional[Dict]:
        """
        读取某法规知识层缓存元信息 {fingerprint, triple_count, extracted_at}，未抽取过返回 None。
        用于判断能否直接复用缓存、跳过 LLM 调用。
        """
        try:
            resp = self.client.search(
                index=self.KG_TRIPLE_INDEX,
                body={
                    "query": {"term": {"law_id": law_id}},
                    "size": 1,
                    "sort": [{"extracted_at": "desc"}],
                    "_source": ["fingerprint", "extracted_at"],
                    "track_total_hits": True,
                },
            )
        except NotFoundError:
            return None
        except Exception as e:
            logger.warning(f"[KG] 读取缓存元信息失败: {e}")
            return None

        hits = resp.get("hits", {})
        total = hits.get("total", {}).get("value", 0)
        if not total:
            return None
        src = (hits.get("hits") or [{}])[0].get("_source", {})
        return {
            "fingerprint": src.get("fingerprint", ""),
            "triple_count": total,
            "extracted_at": src.get("extracted_at", ""),
        }

    def delete_kg_triples(self, law_id: str) -> int:
        """删除某法规的全部三元组（法规被删除或重新抽取前调用）"""
        try:
            resp = self.client.delete_by_query(
                index=self.KG_TRIPLE_INDEX,
                body={"query": {"term": {"law_id": law_id}}},
                refresh=True,
                conflicts="proceed",
            )
            return resp.get("deleted", 0)
        except NotFoundError:
            return 0
        except Exception as e:
            logger.warning(f"[KG] 删除三元组失败: {e}")
            return 0

    def get_kg_statistics(self) -> Dict:
        """知识层统计：三元组总数、已抽取法规数、实体总数、各法域覆盖情况。

        覆盖情况用于把「图谱页/图谱问答对某法域为空」明确呈现为"尚未抽取"，
        而不是让调用方误以为该法域没有实体关系。全部为 ES 聚合，零 LLM。
        """
        result = {"triple_count": 0, "law_count": 0, "entity_count": 0,
                  "triples_by_jurisdiction": {}, "covered_laws_by_jurisdiction": {},
                  "total_laws_by_jurisdiction": {}, "coverage": 0.0}
        try:
            resp = self.client.search(
                index=self.KG_TRIPLE_INDEX,
                body={
                    "size": 0,
                    "track_total_hits": True,
                    "aggs": {
                        "laws": {"cardinality": {"field": "law_id"}},
                        "by_jur": {
                            "terms": {"field": "jurisdiction", "size": 40},
                            "aggs": {"laws": {"cardinality": {"field": "law_id"}}},
                        },
                    },
                },
            )
            result["triple_count"] = resp.get("hits", {}).get("total", {}).get("value", 0)
            result["law_count"] = resp.get("aggregations", {}).get("laws", {}).get("value", 0)
            for bucket in resp.get("aggregations", {}).get("by_jur", {}).get("buckets", []):
                jur = bucket.get("key")
                result["triples_by_jurisdiction"][jur] = bucket.get("doc_count", 0)
                result["covered_laws_by_jurisdiction"][jur] = (
                    bucket.get("laws", {}).get("value", 0)
                )
        except Exception as e:
            logger.warning(f"[KG] 三元组统计失败: {e}")
        try:
            result["entity_count"] = int(
                self.client.count(index=self.KG_ENTITY_INDEX).get("count", 0)
            )
        except Exception:
            pass
        # 各法域的法规总数（来自法规目录），用于计算覆盖率
        try:
            for law in self.get_all_laws():
                jur = _jurisdiction_to_zh(str(law.get("jurisdiction") or ""))
                if jur:
                    result["total_laws_by_jurisdiction"][jur] = (
                        result["total_laws_by_jurisdiction"].get(jur, 0) + 1
                    )
        except Exception as e:
            logger.warning(f"[KG] 法规目录统计失败: {e}")
        total_laws = sum(result["total_laws_by_jurisdiction"].values())
        if total_laws:
            result["coverage"] = round(result["law_count"] / total_laws, 4)
        return result

    def get_kg_law_ids(self) -> List[str]:
        """
        返回三元组索引中出现过的全部 law_id（terms 聚合）。

        用于把「重建实体词典」等操作收敛到真正有知识层的法规，避免逐条空查。
        """
        try:
            resp = self.client.search(
                index=self.KG_TRIPLE_INDEX,
                body={
                    "size": 0,
                    "aggs": {"laws": {"terms": {"field": "law_id", "size": 10000}}},
                },
            )
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] 读取已抽取法规列表失败: {e}")
            return []
        buckets = resp.get("aggregations", {}).get("laws", {}).get("buckets", [])
        return [b["key"] for b in buckets if b.get("key")]

    # ---- 实体词典（legal_kg_entities）-------------------------------
    # 实体词典是全局的：记录规范实体名、别名集合、类型与出现法域。刻意不记录 law_ids
    # （重新抽取后易失真），「实体出现在哪些法规」可随时由三元组索引查出。

    def get_kg_entities_batch(self, entity_ids: List[str]) -> Dict[str, Dict]:
        """按 entity_id 批量读取实体词典条目（单次 msearch），返回 {entity_id: 文档}"""
        keys = [e for e in entity_ids if e]
        if not keys:
            return {}

        body: List[Dict] = []
        for entity_id in keys:
            body.append({"index": self.KG_ENTITY_INDEX})
            body.append({"query": {"term": {"entity_id": entity_id}}, "size": 1})

        try:
            response = self.client.msearch(body=body)
        except NotFoundError:
            return {}
        except Exception as e:
            logger.warning(f"[KG] 批量读取实体失败: {e}")
            return {}

        result: Dict[str, Dict] = {}
        for entity_id, item in zip(keys, response.get("responses", [])):
            hits = item.get("hits", {}).get("hits", [])
            if hits:
                result[entity_id] = hits[0].get("_source", {})
        return result

    def merge_kg_entities(self, entities: List[Dict]) -> int:
        """
        合并写入实体词典：别名/法域取并集，已有条目累加、新条目直接写入。

        采用「批量读 + 批量写」两次往返，避免逐条 read-modify-write；重复抽取不会产生重复项
        也不丢历史别名。返回写入条数，失败返回 -1。
        """
        if not entities:
            return 0
        if not self._ensure_kg_indices():
            return -1

        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        existing = self.get_kg_entities_batch([e.get("entity_id", "") for e in entities])

        actions = []
        for ent in entities:
            entity_id = ent.get("entity_id", "")
            if not entity_id:
                continue
            old = existing.get(entity_id, {})
            merged = {
                "entity_id": entity_id,
                "name": ent.get("name", "") or old.get("name", ""),
                "type": ent.get("type", "") or old.get("type", ""),
                "aliases": sorted(set(old.get("aliases", [])) | set(ent.get("aliases", []))),
                "jurisdictions": sorted(
                    set(old.get("jurisdictions", [])) | set(ent.get("jurisdictions", []))
                ),
                "updated_at": now,
            }
            # 保留既有向量（L3 向量对齐写入后不应被覆盖为空）
            if old.get("embedding"):
                merged["embedding"] = old["embedding"]
            elif ent.get("embedding"):
                merged["embedding"] = ent["embedding"]
            actions.append({
                "_index": self.KG_ENTITY_INDEX,
                "_id": self._kg_entity_doc_id(entity_id),
                "_source": merged,
            })

        if not actions:
            return 0

        try:
            from elasticsearch.helpers import bulk
            success, failed = bulk(
                self.client, actions, refresh=True,
                raise_on_error=False, stats_only=True,
            )
            logger.info(f"[KG] 实体词典已合并: 成功={success}, 失败={failed}")
            return int(success)
        except Exception as e:
            logger.error(f"[KG] 实体词典合并失败: {type(e).__name__}: {e}")
            return -1

    def prune_kg_entities(self, referenced_ids: set) -> int:
        """
        删除实体词典中「已不再被任何三元组引用」的条目（重建后的残留清理）。

        实体合并会改变三元组 canonical_id，旧条目不会自动移除而成为孤立实体；本方法把
        词典收敛到当前三元组实际引用的规范实体集合。

        **破坏性操作**：referenced_ids 必须是全部三元组引用的 canonical_id 全集，否则会
        误删仍在使用的实体。返回删除条数；失败返回 -1。
        """
        ref = {r for r in (referenced_ids or set()) if r}
        try:
            resp = self.client.search(
                index=self.KG_ENTITY_INDEX,
                body={"query": {"match_all": {}}, "size": 10000, "_source": ["entity_id"]},
            )
        except NotFoundError:
            return 0
        except Exception as e:
            logger.warning(f"[KG] 读取实体词典失败（跳过清理）: {e}")
            return -1

        hits = resp.get("hits", {}).get("hits", [])
        stale = [
            h.get("_source", {}).get("entity_id")
            for h in hits
            if h.get("_source", {}).get("entity_id")
            and h["_source"]["entity_id"] not in ref
        ]
        if not stale:
            return 0

        actions = [
            {"_op_type": "delete", "_index": self.KG_ENTITY_INDEX,
             "_id": self._kg_entity_doc_id(eid)}
            for eid in stale
        ]
        try:
            from elasticsearch.helpers import bulk
            success, failed = bulk(
                self.client, actions, refresh=True,
                raise_on_error=False, stats_only=True,
            )
            logger.info(f"[KG] 实体词典清理：删除 {success} 个孤立实体（失败 {failed}）")
            return int(success)
        except Exception as e:
            logger.error(f"[KG] 实体词典清理失败: {type(e).__name__}: {e}")
            return -1

    # ---- 实体合并候选复核（legal_kg_review，仅辅助人工）-----------------

    def _ensure_kg_review_index(self) -> bool:
        """确保实体合并候选索引存在，不存在则创建"""
        try:
            if not self.client.indices.exists(index=self.KG_REVIEW_INDEX):
                self.client.indices.create(
                    index=self.KG_REVIEW_INDEX,
                    body={
                        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                        "mappings": {
                            "properties": {
                                "name": {"type": "keyword"},
                                "type": {"type": "keyword"},
                                "candidate": {"type": "keyword"},
                                "candidate_id": {"type": "keyword"},
                                "similarity": {"type": "float"},
                                "blocked_by": {"type": "keyword"},
                                "actionable": {"type": "boolean"},
                                "first_seen": {"type": "date"},
                                "last_seen": {"type": "date"},
                                "seen_count": {"type": "integer"},
                            }
                        },
                    },
                )
            return True
        except Exception as e:
            logger.warning(f"[KG] 复核候选索引创建/检查失败: {e}")
            return False

    @staticmethod
    def _kg_review_doc_id(name: str, candidate_id: str) -> str:
        import hashlib
        return hashlib.md5(f"{name}|{candidate_id}".encode("utf-8")).hexdigest()

    def save_kg_review_candidates(self, candidates: List[Dict]) -> int:
        """
        持久化「实体合并候选」（被 L3 守卫拦下的高相似实体对），供人工复核。

        文档 ID 由 (name, candidate_id) 哈希得到：同一候选反复被拦只累加 seen_count 并刷新
        last_seen，不产生重复行；first_seen 首次写入时确定。失败返回 -1（调用方可降级）。
        """
        import datetime
        rows = [c for c in (candidates or []) if c.get("name") and c.get("candidate_id")]
        if not rows:
            return 0
        if not self._ensure_kg_review_index():
            return -1

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ids = [self._kg_review_doc_id(c["name"], c["candidate_id"]) for c in rows]

        # 读取既有条目：保留 first_seen 并累加 seen_count（读-并-写，与实体词典一致）
        existing: Dict[str, Dict] = {}
        try:
            resp = self.client.search(
                index=self.KG_REVIEW_INDEX,
                body={"query": {"ids": {"values": ids}}, "size": len(ids)},
            )
            for hit in resp.get("hits", {}).get("hits", []):
                existing[hit["_id"]] = hit.get("_source", {})
        except NotFoundError:
            pass
        except Exception as e:
            logger.warning(f"[KG] 读取既有复核候选失败（本次仅新增）: {e}")

        actions = []
        for c, doc_id in zip(rows, ids):
            old = existing.get(doc_id, {})
            actions.append({
                "_index": self.KG_REVIEW_INDEX,
                "_id": doc_id,
                "_source": {
                    "name": c.get("name", ""),
                    "type": c.get("type", ""),
                    "candidate": c.get("candidate", ""),
                    "candidate_id": c.get("candidate_id", ""),
                    "similarity": float(c.get("similarity", 0.0)),
                    "blocked_by": c.get("blocked_by", ""),
                    "actionable": bool(c.get("actionable", False)),
                    "first_seen": old.get("first_seen") or now,
                    "last_seen": now,
                    "seen_count": int(old.get("seen_count", 0)) + 1,
                },
            })
        try:
            from elasticsearch.helpers import bulk
            success, failed = bulk(
                self.client, actions, refresh=True,
                raise_on_error=False, stats_only=True,
            )
            logger.info(f"[KG] 复核候选已落库: 成功={success}, 失败={failed}")
            return int(success)
        except Exception as e:
            logger.error(f"[KG] 复核候选落库失败: {type(e).__name__}: {e}")
            return -1

    def list_kg_review_candidates(
        self,
        actionable_only: bool = True,
        blocked_by: Optional[str] = None,
        min_similarity: float = 0.0,
        page_size: int = 50,
    ) -> List[Dict]:
        """
        列出待人工确认的实体合并候选（按相似度、出现次数降序）。

        actionable_only=True 只返回「放宽阈值即可归并」的候选（已通过名称长度/数字/
        书名号守卫），用于滤除英文整句描述串等噪声。
        """
        clauses: List[Dict] = [{"range": {"similarity": {"gte": min_similarity}}}]
        if actionable_only:
            clauses.append({"term": {"actionable": True}})
        if blocked_by:
            clauses.append({"term": {"blocked_by": blocked_by}})
        try:
            resp = self.client.search(
                index=self.KG_REVIEW_INDEX,
                body={
                    "query": {"bool": {"filter": clauses}},
                    "size": page_size,
                    "sort": [{"similarity": "desc"}, {"seen_count": "desc"}],
                },
            )
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] 读取复核候选失败: {e}")
            return []

        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            doc = dict(hit.get("_source", {}))
            doc["id"] = hit.get("_id", "")
            results.append(doc)
        return results

    def dismiss_kg_review_candidates(self, ids: List[str]) -> int:
        """按文档 ID 删除已确认/已处理的复核候选。返回删除条数；失败返回 -1。"""
        keys = [i for i in (ids or []) if i]
        if not keys:
            return 0
        try:
            resp = self.client.delete_by_query(
                index=self.KG_REVIEW_INDEX,
                body={"query": {"ids": {"values": keys}}},
                refresh=True,
                conflicts="proceed",
            )
            return resp.get("deleted", 0)
        except NotFoundError:
            return 0
        except Exception as e:
            logger.warning(f"[KG] 删除复核候选失败: {e}")
            return -1

    def list_kg_entities(
        self,
        q: Optional[str] = None,
        entity_type: Optional[str] = None,
        page_size: int = 50,
    ) -> List[Dict]:
        """列出实体词典条目，可按规范名/别名关键词与实体类型过滤"""
        clauses: List[Dict] = []
        if q:
            # name 与 aliases 均为 keyword，用通配匹配实现子串检索
            clauses.append({
                "bool": {
                    "should": [
                        {"wildcard": {"name": f"*{q}*"}},
                        {"wildcard": {"aliases": f"*{q}*"}},
                    ],
                    "minimum_should_match": 1,
                }
            })
        if entity_type:
            clauses.append({"term": {"type": entity_type}})

        query = {"bool": {"filter": clauses}} if clauses else {"match_all": {}}
        try:
            resp = self.client.search(
                index=self.KG_ENTITY_INDEX,
                body={"query": query, "size": page_size, "sort": [{"entity_id": "asc"}]},
            )
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] 查询实体词典失败: {e}")
            return []

        return [hit.get("_source", {}) for hit in resp.get("hits", {}).get("hits", [])]

    def search_kg_entities_knn(
        self,
        vector: List[float],
        k: int = 5,
        exclude_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        实体词典 KNN 检索：在 embedding 字段上取近邻，用于图谱问答的实体链接。

        embedding 由 bge-m3 生成（与检索层同模型），故跨语言也能匹配。
        """
        if not vector:
            return []
        knn = {
            "field": "embedding",
            "query_vector": vector,
            "k": k,
            "num_candidates": max(50, k * 20),
        }
        if exclude_types:
            knn["filter"] = {"bool": {"must_not": [{"terms": {"type": exclude_types}}]}}
        try:
            resp = self.client.search(
                index=self.KG_ENTITY_INDEX,
                body={
                    "knn": knn,
                    "size": k,
                    "_source": ["entity_id", "name", "type", "jurisdictions"],
                },
            )
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] 实体 KNN 检索失败: {e}")
            return []

        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            src["score"] = hit.get("_score")
            results.append(src)
        return results

    def get_kg_entities_with_vectors(
        self,
        exclude_types: Optional[List[str]] = None,
        size: int = 5000,
    ) -> Optional[List[Dict]]:
        """
        读取实体词典（含 embedding），供 L3 向量对齐在内存中做同类型比对。

        词典规模在数千以内，一次搜索即可覆盖，后续用矩阵运算比较，避免逐条 KNN。
        失败返回 None（调用方据此跳过向量对齐，而非当成「词典为空」）。
        """
        query: Dict = {"match_all": {}}
        if exclude_types:
            query = {"bool": {"must_not": [{"terms": {"type": exclude_types}}]}}
        try:
            resp = self.client.search(
                index=self.KG_ENTITY_INDEX,
                body={
                    "query": query,
                    "size": size,
                    "_source": ["entity_id", "name", "type", "aliases",
                                "jurisdictions", "embedding"],
                },
            )
        except NotFoundError:
            return []
        except Exception as e:
            logger.warning(f"[KG] 读取实体向量失败: {type(e).__name__}: {e}")
            return None

        return [hit.get("_source", {}) for hit in resp.get("hits", {}).get("hits", [])]

    def delete_kg_entities(self) -> int:
        """清空实体词典（别名表或本体调整后重建用）"""
        try:
            resp = self.client.delete_by_query(
                index=self.KG_ENTITY_INDEX,
                body={"query": {"match_all": {}}},
                refresh=True,
                conflicts="proceed",
            )
            return resp.get("deleted", 0)
        except NotFoundError:
            return 0
        except Exception as e:
            logger.warning(f"[KG] 清空实体词典失败: {e}")
            return 0


# 日志别名
import logging
logger = logging.getLogger(__name__)
