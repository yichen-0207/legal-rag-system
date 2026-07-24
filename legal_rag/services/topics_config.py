"""
跨法域专题对比 — 预设专题配置 v2

设计原则：
- 每个专题严格对应澳门+新加坡双方均有实质性法规的领域
- search_queries 取自 taxonomy_merged.json 的关键词，确保 ES 能命中
- dimensions 设计贴合各法实际条文结构，让 LLM 提取有据可依
"""

TOPICS: list[dict] = [
    # ==========================================================================
    #  1. 个人资料保护与隐私权（澳门：散见于多部法律 / 新加坡：PDPA）
    # ==========================================================================
    {
        "id": "personal_data_protection",
        "name": "个人资料保护与隐私权",
        "description": "比较两个法域在个人信息收集、处理、存储、使用全流程中的保护机制差异",
        "search_queries": [
            "个人资料", "个人信息", "隐私", "私隱", "数据保护", "data protection",
            "personal data", "privacy", "sensitive data", "数据处理", "資料處理",
            "資料當事人", "数据当事人", "数据控制者", "数据主体"
        ],
        "dimensions": [
            {"key": "scope_of_application", "label": "适用范围（涵盖哪些主体和数据类型）"},
            {"key": "lawful_basis", "label": "数据处理合法性基础（同意/合同/法定义务等）"},
            {"key": "consent_requirements", "label": "有效同意的要件与撤回机制"},
            {"key": "data_subject_rights", "label": "数据主体的具体权利（查阅/更正/删除等）"},
            {"key": "processor_obligations", "label": "数据处理者的安全保障义务"},
            {"key": "cross_border_transfer", "label": "数据出境/跨境转移的条件限制"},
            {"key": "breach_notification", "label": "数据泄露时的通报义务与时限"},
            {"key": "penalties", "label": "违规处罚措施与力度"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  2. 网络安全与关键基础设施保护（澳门：網絡安全法 / 新加坡：Cybersecurity Act）
    # ==========================================================================
    {
        "id": "cybersecurity_infrastructure",
        "name": "网络安全与关键基础设施保护",
        "description": "比较两个法域在关键基础设施运营者网络安全义务、事件报告、监督检查等方面的框架",
        "search_queries": [
            "网络安全", "關鍵基礎設施", "基础设施", "cybersecurity", "critical infrastructure",
            "网络攻击", "安全事件", "incident response", "安全评估", "等级保护",
            "网络运营者", "信息系统安全", "網絡安全"
        ],
        "dimensions": [
            {"key": "scope_of_applicability", "label": "适用范围（哪些系统/运营者被纳入）"},
            {"key": "classification_system", "label": "分级分类/等级保护制度"},
            {"key": "operator_obligations", "label": "关键基础设施运营者的安全义务"},
            {"key": "incident_reporting", "label": "安全事件报告要求（时限、对象、内容）"},
            {"key": "supervision_enforcement", "label": "监管机构的检查与执法权力"},
            {"key": "technical_standards", "label": "技术标准与认证要求"},
            {"key": "penalties_sanctions", "label": "违规处罚措施"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  3. 电子交易与数字签名（澳门：電子文件及電子簽名 / 新加坡：ETA）
    # ==========================================================================
    {
        "id": "electronic_transactions_signatures",
        "name": "电子交易与数字签名",
        "description": "比较两个法域在电子文件效力、数字签名认证、电子合同等方面的法律框架",
        "search_queries": [
            "电子文件", "电子签名", "数字签名", "電子簽名", "electronic signature",
            "digital signature", "electronic document", "electronic transaction",
            "电子合同", "电子记录", "时间戳", "电子认证", "e-signature"
        ],
        "dimensions": [
            {"key": "legal_effect", "label": "电子文件/签名的法律效力认定"},
            {"key": "signature_requirements", "label": "有效数字签名的技术要件"},
            {"key": "certification_authority", "label": "电子认证机构的管理与责任"},
            {"key": "electronic_contract", "label": "电子合同的成立与有效性"},
            {"key": "evidence_admissibility", "label": "电子证据的可采性与证明力"},
            {"key": "liability_framework", "label": "责任归属与赔偿机制"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  4. 消费者权益保护（澳门：消費者權益保護法 / 新加坡：CPFTA）
    # ==========================================================================
    {
        "id": "consumer_rights_protection",
        "name": "消费者权益保护",
        "description": "比较两个法域在消费者知情权、公平交易、商品服务质量、争议解决等方面的保护制度",
        "search_queries": [
            "消费者", "消費者權益", "公平交易", "consumer protection", "consumer rights",
            "unfair practice", "商品安全", "服务质量", "退货", "赔偿",
            "不正當營商行為", "经营行为", "广告", "价格"
        ],
        "dimensions": [
            {"key": "consumer_basic_rights", "label": "消费者基本权利范围"},
            {"key": "unfair_trading_practices", "label": "禁止的不正当营商行为类型"},
            {"key": "product_safety_quality", "label": "商品与服务质量安全标准"},
            {"key": "information_disclosure", "label": "经营者信息披露义务"},
            {"key": "contract_terms_regulation", "label": "不公平合同条款的规制"},
            {"key": "dispute_resolution", "label": "消费者争议解决途径"},
            {"key": "enforcement_penalties", "label": "执法机制与处罚措施"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  5. 电脑犯罪与网络违法（澳门：打擊電腦犯罪法 / 新加坡：CMA）
    # ==========================================================================
    {
        "id": "cybercrime_offenses",
        "name": "电脑犯罪与网络违法行为",
        "description": "比较两个法域对非法入侵、恶意程序、电脑诈骗、电子伪造等网络犯罪的定罪与量刑标准",
        "search_queries": [
            "电脑犯罪", "計算機犯罪", "黑客", "非法入侵", "hacking", "computer misuse",
            "computer crime", "malware", "恶意软件", "电脑诈骗", "伪造",
            "电脑伪造", "非法截取", "打擊電腦犯罪"
        ],
        "dimensions": [
            {"key": "offense_categories", "label": "罪名分类与构成要件"},
            {"key": "unauthorized_access", "label": "非法侵入计算机系统的定义与刑罚"},
            {"key": "interception_modification", "label": "非法截取/篡改数据的定罪标准"},
            {"key": "malicious_programs", "label": "传播恶意程序的刑事责任"},
            {"key": "computer_fraud_forgery", "label": "电脑诈骗与电子伪造的处罚"},
            {"key": "jurisdiction_scope", "label": "管辖权与域外效力"},
            {"key": "investigation_powers", "label": "侦查机关的取证权限"},
            {"key": "sentencing_guidelines", "label": "量刑幅度与加重情节"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  6. 电信监管与通讯服务（澳门：電信綱要法 / 新加坡：Telecommunications Act）
    # ==========================================================================
    {
        "id": "telecommunications_regulation",
        "name": "电信监管与通讯服务",
        "description": "比较两个法域在电信牌照管理、频谱分配、互联互通、用户权益等方面的监管框架",
        "search_queries": [
            "电信", "電信", "通讯", "通訊", "telecommunications", "telecom",
            "运营商", "频谱", "频率", "牌照", "license", "interconnection",
            "公共电信", "无线电", "radio spectrum", "特许"
        ],
        "dimensions": [
            {"key": "licensing_regime", "label": "电信业务许可/牌照制度"},
            {"key": "spectrum_management", "label": "无线电频谱管理与分配"},
            {"key": "universal_service", "label": "普遍服务义务与覆盖要求"},
            {"key": "interconnection_rules", "label": "运营商互联互通规则"},
            {"key": "user_protection", "label": "终端用户权益保护"},
            {"key": "regulatory_authority", "label": "监管机构及其职权"},
            {"key": "penalties_violations", "label": "违规处罚措施"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  7. 金融体系监管（澳门：金融體系法律制度 / 新加坡：FSMA）
    # ==========================================================================
    {
        "id": "financial_system_supervision",
        "name": "金融体系监管合规",
        "description": "比较两个法域在金融机构准入、审慎监管、资本充足、风险管理等方面的监管框架",
        "search_queries": [
            "金融", "银行", "金融體系", "financial system", "banking",
            "金融机构", "capital adequacy", "审慎监管", "prudential",
            "风险管理", "金融稳定", "信用机构", "持牌", "license"
        ],
        "dimensions": [
            {"key": "regulated_entities", "label": "受监管的金融机构类型"},
            {"key": "licensing_conditions", "label": "金融业务准入条件与审批"},
            {"key": "prudential_requirements", "label": "审慎监管要求（资本/流动性）"},
            {"key": "risk_management", "label": "风险管理框架与内部控制"},
            {"key": "corporate_governance", "label": "公司治理与董事责任"},
            {"key": "supervisory_powers", "label": "监管机构的检查与干预权限"},
            {"key": "enforcement_sanctions", "label": "违规处罚与纪律措施"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  8. 反洗钱与金融犯罪防控（澳门：預防清洗黑錢 / 新加坡：PSA+FSMA）
    # ==========================================================================
    {
        "id": "anti_money_laundering",
        "name": "反洗钱与金融犯罪防控",
        "description": "比较两个法域在客户尽职调查、可疑交易报告、资产冻结、反恐融资等方面的制度设计",
        "search_queries": [
            "洗钱", "清洗黑錢", "黑钱", "money laundering", "AML",
            "恐怖融资", "terrorist financing", "可疑交易", "suspicious transaction",
            "客户尽职", "CDD", "实益拥有", "beneficial owner", "冻结", "没收",
            "政治公众人物", "PEP", "上游犯罪"
        ],
        "dimensions": [
            {"key": "customer_due_diligence", "label": "客户尽职调查(CDD)要求"},
            {"key": "beneficial_owner", "label": "实益拥有人识别与登记"},
            {"key": "suspicious_transaction_report", "label": "可疑交易报告义务"},
            {"key": "record_keeping", "label": "交易记录保存期限与要求"},
            {"key": "freezing_confiscation", "label": "资产冻结与没收程序"},
            {"key": "internal_control", "label": "机构内控与合规官任命"},
            {"key": "criminal_liability", "label": "洗钱罪的构成与刑罚"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  9. 数据跨境转移规则（双方均有相关规定）
    # ==========================================================================
    {
        "id": "cross_border_data_transfer",
        "name": "数据跨境转移规则",
        "description": "比较两个法域在个人/重要数据向境外传输的条件、机制和监管要求",
        "search_queries": [
            "跨境转移", "数据出境", "transfer abroad", "cross-border transfer",
            "出境", "境外", "overseas", "国际传输", "白名单", "adequacy",
            "standard contractual clauses", "binding corporate rules"
        ],
        "dimensions": [
            {"key": "transfer_conditions", "label": "允许跨境转移的前提条件"},
            {"key": "assessment_mechanism", "label": "安全评估/充分性认定机制"},
            {"key": "consent_role", "label": "数据主体同意在跨境中的作用"},
            {"key": "recipient_requirements", "label": "接收方的保护水平要求"},
            {"key": "regulatory_approval", "label": "是否需要监管机构事先批准"},
            {"key": "penalties_breach", "label": "违反跨境规则的处罚"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },

    # ==========================================================================
    #  10. 通讯监控与执法权力（澳门：通訊截取法 / 新加坡：相关刑事程序法）
    # ==========================================================================
    {
        "id": "surveillance_lawful_intercept",
        "name": "通讯监控与执法权力",
        "description": "比较两个法域授权执法部门进行通讯截取、数据获取的法律条件与程序保障",
        "search_queries": [
            "通讯截取", "监听", "监控", "截聽", "surveillance", "interception",
            "wiretapping", "实时监控", "通讯数据", "communications data",
            "metadata", "执法", "law enforcement", "取证"
        ],
        "dimensions": [
            {"key": "authorization_requirements", "label": "授权截取/监控的条件与审批层级"},
            {"key": "scope_of_surveillance", "label": "可监控的通讯类型与范围"},
            {"key": "duration_limits", "label": "监控令的有效期限与续期"},
            {"key": "data_retention_use", "label": "截获数据的保存与使用限制"},
            {"key": "oversight_accountability", "label": "监督机制与问责安排"},
            {"key": "subject_notification", "label": "是否/何时通知被监控对象"},
            {"key": "judicial_remedy", "label": "被监控者的救济途径"},
            {"key": "source_articles", "label": "引用法条编号（仅记录）"}
        ]
    },
]


def get_topic_by_id(topic_id: str) -> dict | None:
    """根据ID获取专题配置"""
    for t in TOPICS:
        if t["id"] == topic_id:
            return t
    return None


def get_all_topics() -> list[dict]:
    """获取所有专题列表（精简版，不含完整维度用于列表展示）"""
    return [{"id": t["id"], "name": t["name"], "description": t["description"]} for t in TOPICS]
