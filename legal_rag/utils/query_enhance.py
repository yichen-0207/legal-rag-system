# -*- coding: utf-8 -*-
"""查询增强：中→英关键词扩展 + 「第X条」条款号 / 法规名提取。

RetrieverService 与 QAService 曾各维护一份实现，问答侧词表更弱且「命中即返回」，
实测会漏英文法规、甚至因追加泛化词反而 0 命中；故合并为唯一实现（24 组、累积命中）。
"""
from __future__ import annotations

import re

from core.jurisdiction import ALL_TRIGGERS

# 中→英关键词映射：(中文触发词元组, 英文扩展串)，弥补境外英文法规的召回
ZH_EN_MAP = [
    # 数据保护与隐私
    (('数据保护', '个人数据', '隐私', '个人信息', '个人资料'), 'personal data protection privacy GDPR'),
    # 网络安全
    (('网络安全', '网络', '信息安全', '网络犯罪', '黑客'), 'cybersecurity network security information security cybercrime'),
    # 电子交易与签名
    (('电子', '数字化', '数字签名', '电子签名', '电子交易', '电子政务'), 'electronic digital signature e-commerce e-government'),
    # 电信与通讯
    (('电信', '通讯', '通信', '电话营销', '电话'), 'telecommunications communication telemarketing telephone'),
    # 金融监管
    (('金融', '银行', '保险', '证券', '金融科技'), 'financial banking insurance securities fintech'),
    # 反洗钱
    (('反洗钱', '反恐融资', '反恐'), 'anti-money laundering AML counter-terrorism financing CTF'),
    # 消费者保护
    (('消费者', '消费者保护', '消费', '消费者权益'), 'consumer protection consumer rights'),
    # 人工智能
    (('人工智能', 'AI', '机器学习'), 'artificial intelligence machine learning AI'),
    # 跨境数据流动
    (('跨境', '数据流动', '数据传输', '数据跨境'), 'cross-border data flow data transfer cross-border data'),
    # 牌照与监管
    (('牌照', '许可', '监管', '许可证', '牌照发放'), 'license licensing regulation permit'),
    # 数据泄露与通知
    (('通知', '通报', '披露', '数据泄露', '泄露'), 'notification disclosure breach notification data breach'),
    # 版权与知识产权
    (('版权', '著作权', '知识产权', '专利', '商标'), 'copyright intellectual property patent trademark'),
    # 公司与企业
    (('公司', '企业', '经营', '商业'), 'company corporate business enterprise commercial'),
    # 劳动与雇佣
    (('劳动', '雇佣', '劳工', '员工', '就业'), 'labor employment employee worker'),
    # 环境与环保
    (('环境', '环保', '排放', '污染'), 'environmental protection emission pollution'),
    # 医疗与药品
    (('药品', '医疗', '健康', '药物'), 'medical pharmaceutical healthcare health drug'),
    # 行政与政府
    (('行政', '政府', '官员', '公共'), 'administrative government official public'),
    # 刑事与处罚
    (('刑事', '犯罪', '处罚', '刑罚', '罚款'), 'criminal penalty offense punishment fine'),
    # 法律与法规
    (('法律', '法规', '条例', '法令', '法例'), 'law regulation act ordinance legislation'),
    # 电子认证
    (('电子签名', '电子认证', '认证'), 'electronic signature e-signature authentication'),
    # 金融科技（补充）
    (('金融科技', 'fintech', 'FinTech', 'FINTECH'), 'financial technology fintech'),
    # 版权（补充英文）
    (('版权', 'copyright'), 'copyright'),
    # 垃圾信息
    (('垃圾', '骚扰', '垃圾信息'), 'spam junk unsolicited'),
    # 广播与电视
    (('广播', '电视', '无线电', '频段'), 'broadcast television radio spectrum frequency'),
    # 数据保留
    (('数据保留', '数据保存', '留存'), 'data retention data preservation'),
]

# 「第X条 / 第X條」
_ARTICLE_RE = re.compile(r"第(\d+)[条條]")

# 法域触发词：用于从「法域 + 第X条」结构中截出法规名。
# 长词必须排在短词前（"中国澳门" 先于 "澳门"），否则会截出错误的法规名。
_JURIS_TRIGGERS = tuple(dict.fromkeys(
    ["us"] + [kw.lower() for aliases in ALL_TRIGGERS.values() for kw in aliases]
))


def _first_jurisdiction_span(q: str):
    """返回首个命中法域词的 (起始位置, 长度)；ASCII 词加边界，避免 us 命中 business。"""
    lowered = q.lower()
    for trigger in _JURIS_TRIGGERS:
        if trigger.isascii():
            m = re.search(rf"(?<![a-z0-9]){re.escape(trigger)}(?![a-z0-9])", lowered)
            if m:
                return m.start(), len(trigger)
        else:
            pos = q.find(trigger)
            if pos != -1:
                return pos, len(trigger)
    return None


def enhance_query(query: str) -> str:
    """返回增强后的检索串：原查询 + 累积命中的英文术语 + 条款号 / 法规名。"""
    q = query or ""
    extra_terms = []

    for cn_keywords, en_keywords in ZH_EN_MAP:
        for cn_kw in cn_keywords:
            if cn_kw in q:
                extra_terms.append(en_keywords)
                break

    # 定向查询：抽出条款号数字与法规名，帮助 BM25 精确命中
    m = _ARTICLE_RE.search(q)
    if m:
        extra_terms.append(m.group(1))
        span = _first_jurisdiction_span(q)
        if span:
            pos, length = span
            law_name = q[pos + length:m.start()].strip('的 \t')
            if law_name:
                extra_terms.append(law_name)

    if extra_terms:
        return q + ' ' + ' '.join(extra_terms)
    return q


# ============================================================
# 检索文本归一：只作用于召回，不作用于精排
# ============================================================

# 法域触发词按长度倒序，避免「中国香港」先被「香港」截断
_JURIS_STRIP_TERMS = tuple(sorted(
    {t for aliases in ALL_TRIGGERS.values() for t in aliases},
    key=len, reverse=True,
))

# 只删问句脚手架；不删「法律 / 法规 / 规定 / 条文 / 条款」等可能承载语义的词。
# 依据：2026-11-15 的 Q4 对照（R@20 0.2568→0.2896，MRR 0.2386→0.2498）。
_SCAFFOLD_ZH = ("核心法律规定", "相关法律规定", "核心规定", "相关规定",
                "有哪些", "是什么", "请问")
_SCAFFOLD_EN = ("what are the key legal provisions on",
                "what are the legal requirements for",
                "what are the key legal provisions",
                "what are", "what is")


def _strip_jurisdiction_terms(text: str) -> str:
    """删法域名；ASCII 词加边界，避免 us 命中 business。"""
    for term in _JURIS_STRIP_TERMS:
        if term.isascii():
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                          " ", text, flags=re.I)
        else:
            text = text.replace(term, " ")
    return text


def _strip_query_scaffold(text: str) -> str:
    """删问句脚手架与标点，保留主题词。"""
    for term in _SCAFFOLD_ZH:
        text = text.replace(term, " ")
    for term in _SCAFFOLD_EN:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                      " ", text, flags=re.I)
    text = re.sub(r"[？?。，,、；;：:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_query_for_retrieval(query: str) -> str:
    """返回用于 BM25 / 向量召回的查询串；精排必须继续使用原查询。

    删除法域名与高置信问句脚手架，保留「法律 / 法规 / 规定」等可能承载语义的词。
    只能在调用方确定会施加单法域过滤时使用——跨法域/全局检索里法域名是唯一地理信号。
    归一后为空时回退原查询，避免空串检索。
    """
    q = query or ""
    text = _strip_query_scaffold(_strip_jurisdiction_terms(q))
    if not text:
        return q
    return enhance_query(text)
