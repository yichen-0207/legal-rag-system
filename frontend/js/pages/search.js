/**
 * 法规全文检索页面
 */
registerPage('search', function(container) {
    container.innerHTML = `
        <h1 class="page-title">法规全文检索</h1>
        <p class="page-desc">输入自然语言问题或关键词，系统将返回语义最相关的法规条款</p>
        <div class="alert alert-info">提示：不输入关键词，仅选择法域或主题即可浏览相关法规</div>

        <!-- 搜索栏 -->
        <div class="search-filter-bar">
            <input type="text" id="search-query" class="input search-filter-input" placeholder="例如：数据保护 跨境传输">
            <div class="search-filter-item">
                <label>法域</label>
                <select id="search-jurisdiction" class="select search-filter-select">
                    <option value="">全部法域</option>
                </select>
            </div>
            <div class="search-filter-item search-topic-wrapper">
                <label>主题</label>
                <div id="search-topic-trigger" class="search-topic-trigger">
                    <span id="search-topic-label">全部主题</span>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
                </div>
                <div id="search-topic-dropdown" class="search-topic-dropdown" style="display:none;"></div>
                <select id="search-topic" class="select search-filter-select" multiple style="display:none;">
                </select>
            </div>
            <button id="search-btn" class="btn btn-primary search-filter-btn">检索</button>
        </div>

        <!-- 结果区 -->
        <div id="search-results"></div>
    `;

    // 加载法域和主题选项
    Promise.all([
        api.get('/search/jurisdictions'),
        api.get('/search/topics')
    ]).then(function(results) {
        const jurData = results[0];
        const topicData = results[1];

        const jurSelect = document.getElementById('search-jurisdiction');
        if (jurData && jurData.data) {
            jurData.data.forEach(function(j) {
                const opt = document.createElement('option');
                opt.value = j;
                opt.textContent = j;
                jurSelect.appendChild(opt);
            });
        }

        const topicSelect = document.getElementById('search-topic');
        const topicDropdown = document.getElementById('search-topic-dropdown');
        const topicTrigger = document.getElementById('search-topic-trigger');
        const topicLabel = document.getElementById('search-topic-label');

        function updateTopicLabel() {
            const selected = Array.from(topicSelect.selectedOptions).map(function(o) { return o.value; });
            if (selected.length === 0) {
                topicLabel.textContent = '全部主题';
            } else if (selected.length === 1) {
                topicLabel.textContent = selected[0];
            } else {
                topicLabel.textContent = '已选 ' + selected.length + ' 个主题';
            }
        }

        function toggleTopicDropdown(show) {
            topicDropdown.style.display = show ? 'block' : 'none';
            topicTrigger.classList.toggle('active', show);
        }

        if (topicData && topicData.data) {
            topicData.data.forEach(function(t) {
                const value = t.label_zh || t.topic || '';
                // 原生 select option
                const opt = document.createElement('option');
                opt.value = value;
                opt.textContent = value + ' (' + (t.count || 0) + ')';
                topicSelect.appendChild(opt);

                // 自定义下拉项
                const item = document.createElement('div');
                item.className = 'search-topic-item';
                item.dataset.value = value;
                item.innerHTML = '<span class="search-topic-check"></span><span>' + escapeHtml(value) + ' <span style="color:var(--color-text-muted);font-size:0.75rem;">(' + (t.count || 0) + ')</span></span>';
                item.addEventListener('click', function(e) {
                    e.stopPropagation();
                    const optToToggle = Array.from(topicSelect.options).find(function(o) { return o.value === value; });
                    if (optToToggle) {
                        optToToggle.selected = !optToToggle.selected;
                        item.classList.toggle('selected', optToToggle.selected);
                        updateTopicLabel();
                    }
                });
                topicDropdown.appendChild(item);
            });
        }

        topicTrigger.addEventListener('click', function(e) {
            e.stopPropagation();
            const isHidden = topicDropdown.style.display === 'none';
            toggleTopicDropdown(isHidden);
        });

        document.addEventListener('click', function() {
            toggleTopicDropdown(false);
        });

        topicDropdown.addEventListener('click', function(e) {
            e.stopPropagation();
        });

        updateTopicLabel();

        // 恢复之前保存的搜索状态（从法规详情返回时）
        restoreSearchState();
    }).catch(function() { /* 静默失败 */ });

    // 搜索按钮
    document.getElementById('search-btn').addEventListener('click', doSearch);
    document.getElementById('search-query').addEventListener('keydown', function(e) {
        if (e.key === 'Enter') doSearch();
    });
});

function doSearch() {
    const query = document.getElementById('search-query').value.trim();
    const jurisdiction = document.getElementById('search-jurisdiction').value;
    const topicSelect = document.getElementById('search-topic');
    const topics = Array.from(topicSelect.selectedOptions).map(function(o) { return o.value; });

    const resultsEl = document.getElementById('search-results');
    showLoading(resultsEl);

    if (!query) {
        // 浏览模式：获取法规列表
        const params = {};
        if (jurisdiction) params.jurisdiction = jurisdiction;
        if (topics.length > 0) params.topics = topics.join(',');

        api.get('/search/laws', params).then(function(data) {
            if (data && data.data) {
                renderLawList(resultsEl, data.data);
            } else {
                resultsEl.innerHTML = '<div class="alert alert-warning">未找到相关法规</div>';
            }
        }).catch(function(err) {
            showError(resultsEl, '检索失败: ' + err.message);
        });
    } else {
        // 语义检索模式
        const params = { query: query, top_k: 50 };
        if (jurisdiction) params.jurisdiction = jurisdiction;
        if (topics.length > 0) params.topics = topics.join(',');

        api.get('/search', params).then(function(data) {
            if (data && data.data) {
                renderSearchResults(resultsEl, data.data, query);
            } else {
                resultsEl.innerHTML = '<div class="alert alert-warning">未找到相关结果</div>';
            }
        }).catch(function(err) {
            showError(resultsEl, '检索失败: ' + err.message);
        });
    }
}

function renderLawList(container, laws) {
    if (!laws || laws.length === 0) {
        container.innerHTML = '<div class="alert alert-warning">未找到相关法规</div>';
        return;
    }

    let html = '<div class="alert alert-success">找到 ' + laws.length + ' 部相关法规</div>';
    html += '<div class="grid-2">';

    laws.forEach(function(law) {
        const lawId = law.law_id || '';
        const title = law.title || '未知法规';
        const jurisdiction = law.jurisdiction || '未知';
        const articleCount = law.article_count || 0;

        html += '<div class="card">';
        html += '<h3 style="font-family:var(--font-serif);font-size:1.0625rem;font-weight:600;margin-bottom:0.5rem;">' + escapeHtml(title) + '</h3>';
        html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">';
        html += '<span class="badge badge-info">' + escapeHtml(jurisdiction) + '</span>';
        html += '<span style="font-size:0.75rem;color:var(--color-text-muted);">' + articleCount + '条</span>';
        html += '</div>';
        html += '<div style="display:flex;gap:0.5rem;margin-top:0.75rem;">';
        html += '<button class="btn btn-secondary btn-sm" onclick="viewLawDetail(\'' + lawId + '\')">查看全文</button>';
        html += '<button class="btn btn-secondary btn-sm" onclick="fetchSummary(\'' + lawId + '\', this)">展开摘要</button>';
        html += '<button class="btn btn-secondary btn-sm" onclick="findSimilar(\'' + lawId + '\', this)">找相似</button>';
        html += '</div>';
        html += '<div id="summary-' + lawId + '" style="margin-top:0.75rem;"></div>';
        html += '<div id="similar-' + lawId + '" style="margin-top:0.75rem;"></div>';
        html += '</div>';
    });

    html += '</div>';
    container.innerHTML = html;
}

/**
 * 繁简字符映射表（覆盖法规名称常用字）
 * 用于标题匹配时的繁简兼容，避免"网络安全法" vs "網絡安全法" 匹配失败
 */
var _ZH_T2S_MAP = {
    '網': '网', '絡': '络', '網絡': '网络', '絡網': '络网',
    '個': '个', '資': '资', '訊': '讯', '料': '料', '保護': '保护',
    '護': '护', '產': '产', '電': '电', '腦': '脑', '擊': '击',
    '竊': '窃', '盜': '盗', '竊盜': '窃盗', '僞': '伪', '造': '造',
    '關': '关', '係': '系', '關係': '关系', '機': '机', '構': '构',
    '機構': '机构', '團': '团', '體': '体', '團體': '团体',
    '監': '监', '管': '管', '監管': '监管', '辦': '办', '公': '公',
    '辦公': '办公', '廳': '厅', '處': '处', '員': '员', '職': '职',
    '職員': '职员', '區': '区', '域': '域', '區域': '区域',
    '內': '内', '務': '务', '內務': '内务', '安': '安', '全': '全',
    '安全': '安全', '險': '险', '風險': '风险', '險情': '险情',
    '報': '报', '舉報': '举报', '復': '复', '製': '制', '複製': '复制',
    '傳': '传', '輸': '输', '傳輸': '传输', '播': '播', '傳播': '传播',
    '發': '发', '發布': '发布', '發送': '发送', '佈': '布',
    '獲': '获', '獲得': '获得', '取': '取', '獲取': '获取',
    '數': '数', '據': '据', '數據': '数据', '隱': '隐', '私': '私',
    '隱私': '隐私', '祕': '秘', '密': '密', '祕密': '秘密',
    '條': '条', '約': '约', '條約': '条约', '規': '规', '則': '则',
    '規則': '规则', '範': '范', '範圍': '范围', '圍': '围',
    '標': '标', '準': '准', '標準': '标准', '審': '审', '計': '计',
    '審計': '审计', '監察': '监察', '察': '察',
    '訴': '诉', '訟': '讼', '訴訟': '诉讼', '罰': '罚', '處罰': '处罚',
    '罰則': '罚则', '款': '款', '罰款': '罚款',
    '執': '执', '行': '行', '執行': '执行', '實': '实', '施': '施',
    '實施': '实施', '細': '细', '則': '则', '細則': '细则',
    '總': '总', '則': '则', '總則': '总则', '附': '附',
    '附則': '附则', '錄': '录', '附錄': '附录',
    '匯': '汇', '兌': '兑', '匯兌': '汇兑', '幣': '币', '貨': '货',
    '貨幣': '货币', '銀': '银', '行': '行', '銀行': '银行',
    '險': '险', '保險': '保险', '證': '证', '券': '券', '證券': '证券',
    '債': '债', '權': '权', '債權': '债权', '務': '务', '債務': '债务',
    '產': '产', '業': '业', '產業': '产业', '廠': '厂', '商': '商',
    '廠商': '厂商', '號': '号', '碼': '码', '號碼': '号码',
    '圖': '图', '片': '片', '圖片': '图片', '視': '视', '頻': '频',
    '視頻': '视频', '音': '音', '聲': '声', '聲音': '声音',
    '軟': '软', '硬': '硬', '件': '件', '軟件': '软件', '硬件': '硬件',
    '網': '网', '站': '站', '網站': '网站', '頁': '页', '網頁': '网页',
    '連': '连', '接': '接', '連接': '连接', '鏈': '链', '結': '结',
    '鏈結': '链接', '下載': '下载', '載': '载', '上傳': '上传',
    '郵': '邮', '件': '件', '郵件': '邮件', '即時': '即时', '通訊': '通讯',
    '討論': '讨论', '論壇': '论坛', '區': '区', '討論區': '讨论区',
    '交': '交', '易': '易', '交易': '交易', '購': '购', '物': '物',
    '購物': '购物', '電子商務': '电子商务', '務': '务',
    '簽': '签', '署': '署', '簽署': '签署', '訂': '订', '立': '立',
    '訂立': '订立', '約': '约', '契約': '契约',
    '知': '知', '識': '识', '知識': '知识', '產': '产', '權': '权',
    '產權': '产权', '版': '版', '權': '权', '版權': '版权',
    '專': '专', '利': '利', '專利': '专利', '商': '商', '標': '标',
    '商標': '商标', '著': '著', '作': '作', '著作': '著作',
    '營': '营', '業': '业', '營業': '营业', '經': '经', '營': '营',
    '經營': '经营', '執照': '执照', '許': '许', '可': '可',
    '許可': '许可', '證': '证', '證書': '证书', '執': '执',
    '登記': '登记', '記': '记', '註冊': '注册', '冊': '册',
    '備': '备', '案': '案', '備案': '备案',
    '開': '开', '發': '发', '開發': '开发', '創': '创', '新': '新',
    '創新': '创新', '研': '研', '究': '究', '研究': '研究',
    '試': '试', '驗': '验', '試驗': '试验', '測': '测', '試': '试',
    '測試': '测试', '檢': '检', '查': '查', '檢查': '检查',
    '驗證': '验证', '證': '证', '明': '明', '證明': '证明',
    '鑑': '鉴', '定': '定', '鑑定': '鉴定', '評': '评', '估': '估',
    '評估': '评估', '審': '审', '查': '查', '審查': '审查',
    '核': '核', '准': '准', '核准': '核准', '認': '认', '證': '证',
    '認證': '认证', '可': '可', '認可': '认可',
    '責': '责', '任': '任', '責任': '责任', '義': '义', '義務': '义务',
    '權利': '权利', '利': '利', '益': '益', '利益': '利益',
    '損': '损', '害': '害', '損害': '损害', '賠': '赔', '償': '偿',
    '賠償': '赔偿', '補': '补', '助': '助', '補助': '补助',
    '獎': '奖', '勵': '励', '獎勵': '奖励', '懲': '惩', '戒': '戒',
    '懲戒': '惩戒', '紀': '纪', '律': '律', '紀律': '纪律',
    '衛': '卫', '生': '生', '衛生': '卫生', '醫': '医', '療': '疗',
    '醫療': '医疗', '藥': '药', '品': '品', '藥品': '药品',
    '健': '健', '康': '康', '健康': '健康',
    '環': '环', '境': '境', '環境': '环境', '保護': '保护',
    '污染': '污染', '汙': '污', '染': '染', '排放': '排放',
    '废': '废', '棄': '弃', '廢棄': '废弃',
    '勞': '劳', '工': '工', '勞工': '劳工', '僱': '雇', '傭': '佣',
    '僱傭': '雇佣', '就業': '就业', '業': '业', '職業': '职业',
    '薪': '薪', '資': '资', '薪資': '薪资', '福': '福', '利': '利',
    '福利': '福利', '休': '休', '假': '假', '休假': '休假',
    '解': '解', '僱': '雇', '解僱': '解雇', '辭': '辞', '退': '退',
    '辭退': '辞退',
    '消': '消', '費': '费', '消費': '消费', '者': '者',
    '消費者': '消费者', '權益': '权益',
    '兒': '儿', '童': '童', '兒童': '儿童', '少': '少', '年': '年',
    '少年': '少年', '青': '青', '青少年': '青少年',
    '老': '老', '人': '人', '老人': '老人', '殘': '残', '障': '障',
    '殘障': '残障', '疾': '疾', '殘疾': '残疾',
    '婦': '妇', '女': '女', '婦女': '妇女', '幼': '幼', '兒': '儿',
    '幼兒': '幼儿',
    '家': '家', '庭': '庭', '家庭': '家庭', '教': '教', '育': '育',
    '教育': '教育', '學': '学', '習': '习', '學習': '学习',
    '學校': '学校', '校': '校', '學院': '学院', '院': '院',
    '大學': '大学', '學': '学',
    '科': '科', '學': '学', '科學': '科学', '技': '技', '術': '术',
    '技術': '技术', '工': '工', '業': '业', '工業': '工业',
    '農': '农', '業': '业', '農業': '农业', '漁': '渔', '業': '业',
    '漁業': '渔业', '礦': '矿', '業': '业', '礦業': '矿业',
    '交': '交', '通': '通', '交通': '交通', '運': '运', '輸': '输',
    '運輸': '运输', '物': '物', '流': '流', '物流': '物流',
    '能': '能', '源': '源', '能源': '能源', '電': '电', '力': '力',
    '電力': '电力', '水': '水', '資源': '资源', '水資源': '水资源',
    '土': '土', '地': '地', '土地': '土地', '建': '建', '設': '设',
    '建設': '建设', '築': '筑', '建築': '建筑', '屋': '屋', '宇': '宇',
    '屋宇': '屋宇',
    '公': '公', '共': '共', '公共': '公共', '安': '安', '全': '全',
    '安全': '安全', '秩': '秩', '序': '序', '秩序': '秩序',
    '國': '国', '家': '家', '國家': '国家', '民': '民', '族': '族',
    '民族': '民族', '社': '社', '會': '会', '社會': '社会',
    '政': '政', '治': '治', '政治': '政治', '軍': '军', '事': '事',
    '軍事': '军事', '外': '外', '交': '交', '外交': '外交',
    '司': '司', '法': '法', '司法': '司法', '立': '立', '法': '法',
    '立法': '立法', '行': '行', '政': '政', '行政': '行政',
    '檢': '检', '察': '察', '檢察': '检察', '警': '警', '察': '察',
    '警察': '警察', '監': '监', '獄': '狱', '監獄': '监狱',
    '稅': '税', '務': '务', '稅務': '税务', '關': '关', '稅': '税',
    '關稅': '关税', '海': '海', '關': '关', '海關': '海关',
    '財': '财', '政': '政', '財政': '财政', '預': '预', '算': '算',
    '預算': '预算', '審': '审', '計': '计', '審計': '审计',
    '統': '统', '計': '计', '統計': '统计',
    '國': '国', '際': '际', '國際': '国际', '區': '区', '域': '域',
    '區域': '区域', '地': '地', '區': '区', '地區': '地区',
    '城': '城', '市': '市', '城市': '城市', '鄉': '乡', '鎮': '镇',
    '鄉鎮': '乡镇', '村': '村', '莊': '庄', '村莊': '村庄',
    '歐': '欧', '盟': '盟', '歐盟': '欧盟', '聯': '联', '合': '合',
    '國': '国', '聯合國': '联合国', '世': '世', '界': '界',
    '世界': '世界', '貿': '贸', '易': '易', '貿易': '贸易',
    '投': '投', '資': '资', '投資': '投资', '引': '引', '資': '资',
    '引資': '引资', '並': '并', '購': '购', '並購': '并购',
    '重': '重', '組': '组', '重組': '重组', '整': '整', '合': '合',
    '整合': '整合',
    '打擊': '打击', '擊': '击', '犯罪': '犯罪', '犯': '犯',
    '罪': '罪', '刑事': '刑事', '刑': '刑', '罰': '罚', '刑罰': '刑罚',
    '民事': '民事', '訴訟': '诉讼', '仲裁': '仲裁', '仲': '仲',
    '裁': '裁', '調解': '调解', '調': '调', '解': '解',
};

/**
 * 将繁体字符串转换为简体（基于映射表，覆盖法规名称常用字）
 */
function _zhT2S(str) {
    if (!str) return str;
    var result = '';
    for (var i = 0; i < str.length; i++) {
        var ch = str.charAt(i);
        if (_ZH_T2S_MAP[ch]) {
            result += _ZH_T2S_MAP[ch];
        } else {
            result += ch;
        }
    }
    return result;
}

/**
 * 繁简兼容的字符串包含判断
 * 返回 true 表示 a 包含 b 或 b 包含 a（双向判断，繁简互转后比较）
 */
function _zhIncludes(a, b) {
    if (!a || !b) return false;
    var aS = _zhT2S(a);
    var bS = _zhT2S(b);
    return aS.indexOf(bS) !== -1 || bS.indexOf(aS) !== -1;
}
function extractTargetLawName(query) {
    var triggers = ['澳门', '澳門', 'macau', 'Macau', 'MACAU',
                    '香港', 'Hong Kong', 'hongkong', 'Hongkong', 'HONG KONG',
                    '新加坡', 'Singapore', 'SINGAPORE'];

    // Try 1: 含 "第X条" 模式
    var match = query.match(/第(\d+)[条條]/);
    if (match) {
        for (var i = 0; i < triggers.length; i++) {
            var pos = query.indexOf(triggers[i]);
            if (pos !== -1) {
                var name = query.substring(pos + triggers[i].length, match.index).replace(/[的\s]/g, '');
                return name || null;
            }
        }
    }

    // Try 2: 提取法域关键词后的法规名称（不限语言）
    for (var i = 0; i < triggers.length; i++) {
        var pos = query.indexOf(triggers[i]);
        if (pos !== -1) {
            var afterJuris = query.substring(pos + triggers[i].length);

            // 中文法规名称
            var cnMatch = afterJuris.match(/^[的\s]*([\u4e00-\u9fff]{2,20}(?:法|條例|条例|法规|規則|规则|法令|令|守则|守則|制度|指引))(?=[的关于和与,，。．\s]|$)/);
            if (cnMatch) {
                return cnMatch[1];
            }

            // 英文法规名称（以 Act/Ordinance/Rule 等结尾）
            var enMatch = afterJuris.match(/^\s*((?:[A-Za-z][-'A-Za-z]+[\s-]){0,5}(?:Act|Ordinance|Rule|Regulation|Program|Directive|Policy|Standard|Order|Code)(?:\s\d{4})?)(?=[\s\u4e00-\u9fff,，。．;(]|$)/);
            if (enMatch) {
                return enMatch[1].trim();
            }

            // 日韩等其他语言：捕获法域后的连续非空格内容作为法规名
            // e.g. "日本個人情報保護法" → "個人情報保護法"
            var otherMatch = afterJuris.match(/^[的\s]*([^\s,，。．;;(（]+)/);
            if (otherMatch) {
                return otherMatch[1].trim();
            }
        }
    }

    return null;
}

function renderSearchResults(container, results, query) {
    // 只渲染前 50 条避免 DOM 过大（后端 top_k=500 用于 RRF 融合质量）
    var displayResults = results.slice(0, 50);
    var remainingCount = results.length - 50;

    // 从查询中提取目标法规名称（与后端 _extract_law_name 逻辑一致）
    var targetLawName = extractTargetLawName(query);
    
    // 按是否为目标法规分组
    var targetResults = [];
    var otherResults = [];
    
    // 先尝试按标题精确匹配（支持繁简转换）
    var hasTitleMatch = false;
    displayResults.forEach(function(item) {
        var title = (item.metadata && item.metadata.title) || item.title || '';
        var isTarget = targetLawName && _zhIncludes(title, targetLawName);
        if (isTarget) {
            targetResults.push(item);
            hasTitleMatch = true;
        } else {
            otherResults.push(item);
        }
    });

    // 如果没有标题匹配（跨语言情况：如中文名"个人资料保护法" vs ES标题"Personal Data Protection Act"），
    // 用语言无关的"频次+分数加权"法确定目标法规：
    // 1. 对每个 law_id 计算加权分数 = ∑score × (1 + count/10)，同时考虑匹配质量和数量
    // 2. 如果头部法规显著领先其他法规（加权分 >= 2倍第二名），则认定为目标法规
    // 3. 仅当用户查询中明确提取到法规名称时才启用此回退（targetLawName !== null）
    if (targetLawName && !hasTitleMatch && otherResults.length > 0) {
        var lawScores = {};  // { law_id: { totalScore, count } }
        otherResults.forEach(function(item) {
            var lawId = (item.metadata && item.metadata.law_id) || item.law_id;
            var score = item.similarity || item.score || 0;
            if (lawId) {
                if (!lawScores[lawId]) {
                    lawScores[lawId] = { totalScore: 0, count: 0 };
                }
                lawScores[lawId].totalScore += score;
                lawScores[lawId].count += 1;
            }
        });

        // 按加权分排序：加权分 = 总分 × (1 + 命中数/10)，同时衡量质量和覆盖率
        var ranked = Object.keys(lawScores).map(function(lid) {
            var s = lawScores[lid];
            return { lawId: lid, weightedScore: s.totalScore * (1 + s.count / 10), count: s.count };
        }).sort(function(a, b) { return b.weightedScore - a.weightedScore; });

        // 只有当头部法规显著领先时才认定（加权分 >= 2倍第二名 或 第二名不存在）
        if (ranked.length > 0) {
            var top = ranked[0];
            var isDominant = ranked.length === 1 || top.weightedScore >= ranked[1].weightedScore * 2;
            if (isDominant && top.count >= 2) {
                var newTargets = [];
                var remaining = [];
                otherResults.forEach(function(item) {
                    var lawId = (item.metadata && item.metadata.law_id) || item.law_id;
                    if (lawId === top.lawId) {
                        newTargets.push(item);
                    } else {
                        remaining.push(item);
                    }
                });
                targetResults = newTargets;
                otherResults = remaining;
            }
        }
    }

    // 按 law_id 聚合（辅助函数）
    function groupByLaw(items) {
        const lawDict = {};
        items.forEach(function(item) {
            const metadata = item.metadata || {};
            const lawId = metadata.law_id || item.law_id;
            if (!lawId) return;

            if (!lawDict[lawId]) {
                lawDict[lawId] = {
                    law_id: lawId,
                    title: metadata.title || item.title || '未知法规',
                    jurisdiction: metadata.jurisdiction || item.jurisdiction || '未知',
                    max_score: item.similarity || item.score || 0,
                    clauses: []
                };
            }

            const article = metadata.article_number || item.article_number || '条款';
            const content = item.content || '';
            const score = item.similarity || item.score || 0;

            const existing = lawDict[lawId].clauses.find(function(c) { return c.article_number === article; });
            if (existing) {
                if (score > existing.score) { existing.content = content; existing.score = score; }
            } else {
                lawDict[lawId].clauses.push({ article_number: article, content: content, score: score });
            }

            if (score > lawDict[lawId].max_score) lawDict[lawId].max_score = score;
        });
        return Object.values(lawDict);
    }

    const targetLaws = groupByLaw(targetResults);
    const otherLaws = groupByLaw(otherResults);
    const totalClauses = targetLaws.concat(otherLaws).reduce(function(sum, l) { return sum + l.clauses.length; }, 0);

    let html = '<div class="alert alert-success">检索成功</div>';

    // 渲染目标法规
    if (targetLaws.length > 0) {
        html += '<div class="section-divider"><span>目标法规</span></div>';
        html += '<div class="grid-2">';
        targetLaws.forEach(function(law) { html += renderLawCard(law); });
        html += '</div>';
    }

    // 渲染其他相关法规
    if (otherLaws.length > 0) {
        html += '<div class="section-divider section-divider-other"><span>其他相关法规</span></div>';
        html += '<div class="grid-2">';
        otherLaws.forEach(function(law) { html += renderLawCard(law); });
        html += '</div>';
    }

    container.innerHTML = html;
}

function renderLawCard(law) {
    const lawId = law.law_id;
    const maxScorePercent = Math.round(law.max_score * 100);
    let scoreColor = '#d32f2f';
    if (maxScorePercent >= 70) scoreColor = '#2e7d32';
    else if (maxScorePercent >= 50) scoreColor = '#ed6c02';

    let html = '<div class="card">';
    html += '<h3 style="font-family:var(--font-serif);font-size:1.0625rem;font-weight:600;margin-bottom:0.5rem;">' + escapeHtml(law.title) + '</h3>';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">';
    html += '<span class="badge badge-info">' + escapeHtml(law.jurisdiction) + '</span>';
    html += '<span style="font-size:0.75rem;color:' + scoreColor + ';font-weight:600;">最高匹配度 ' + maxScorePercent + '%</span>';
    html += '</div>';

    // 条款列表
    law.clauses.sort(function(a, b) { return b.score - a.score; });
    law.clauses.forEach(function(clause) {
        const cPercent = Math.round(clause.score * 100);
        html += '<div style="background:var(--color-bg);border-left:3px solid var(--color-primary);padding:0.5rem 0.75rem;margin-bottom:0.5rem;font-size:0.8125rem;">';
        html += '<div style="font-weight:600;margin-bottom:0.25rem;">' + escapeHtml(clause.article_number) + ' <span style="color:var(--color-text-muted);font-weight:400;">(' + cPercent + '%)</span></div>';
        html += '<div style="color:var(--color-text-secondary);line-height:1.5;">' + escapeHtml(clause.content.substring(0, 200)) + (clause.content.length > 200 ? '...' : '') + '</div>';
        html += '</div>';
    });

    html += '<div style="display:flex;gap:0.5rem;margin-top:0.75rem;">';
    html += '<button class="btn btn-secondary btn-sm" onclick="viewLawDetail(\'' + lawId + '\')">查看全文</button>';
    html += '<button class="btn btn-secondary btn-sm" onclick="findSimilar(\'' + lawId + '\', this)">找相似</button>';
    html += '</div>';
    html += '<div id="similar-' + lawId + '" style="margin-top:0.75rem;"></div>';
    html += '</div>';
    return html;
}

// 在进入法规详情前保存搜索状态
function saveSearchState() {
    var state = {
        query: document.getElementById('search-query').value,
        jurisdiction: document.getElementById('search-jurisdiction').value,
        topics: Array.from(document.getElementById('search-topic').selectedOptions).map(function(o) { return o.value; }),
        resultsHtml: document.getElementById('search-results').innerHTML
    };
    try { sessionStorage.setItem('search_state', JSON.stringify(state)); } catch(e) {}
}

// 从法规详情返回时恢复搜索状态
function restoreSearchState() {
    var raw;
    try { raw = sessionStorage.getItem('search_state'); } catch(e) {}
    if (!raw) return;

    var state;
    try { state = JSON.parse(raw); } catch(e) {}
    if (!state) return;

    // 清除已恢复的状态，避免重复恢复
    try { sessionStorage.removeItem('search_state'); } catch(e) {}

    // 恢复查询词
    if (state.query) {
        document.getElementById('search-query').value = state.query;
    }

    // 恢复法域
    if (state.jurisdiction) {
        document.getElementById('search-jurisdiction').value = state.jurisdiction;
    }

    // 恢复主题选择
    if (state.topics && state.topics.length > 0) {
        var topicSelect = document.getElementById('search-topic');
        var dropdownItems = document.querySelectorAll('.search-topic-item');
        Array.from(topicSelect.options).forEach(function(opt) {
            if (state.topics.indexOf(opt.value) !== -1) {
                opt.selected = true;
                dropdownItems.forEach(function(item) {
                    if (item.dataset.value === opt.value) item.classList.add('selected');
                });
            }
        });
        var topicLabel = document.getElementById('search-topic-label');
        if (state.topics.length === 1) {
            topicLabel.textContent = state.topics[0];
        } else {
            topicLabel.textContent = '已选 ' + state.topics.length + ' 个主题';
        }
    }

    // 恢复搜索结果HTML（若无查询词则不触发重新搜索）
    var resultsEl = document.getElementById('search-results');
    if (state.resultsHtml) {
        resultsEl.innerHTML = state.resultsHtml;
    }
}

function viewLawDetail(lawId) {
    saveSearchState();
    window.location.hash = '#/law-detail/' + lawId;
}

function fetchSummary(lawId, btn) {
    const container = document.getElementById('summary-' + lawId);
    if (!container) return;

    if (container.innerHTML.trim()) {
        container.innerHTML = '';
        btn.textContent = '展开摘要';
        return;
    }

    btn.textContent = '生成中...';
    btn.disabled = true;

    api.get('/search/law/' + lawId + '/summary').then(function(data) {
        if (data && data.data) {
            renderSummaryCard(container, data.data);
            btn.textContent = '收起摘要';
        } else {
            container.innerHTML = '<div class="alert alert-warning">摘要生成失败</div>';
            btn.textContent = '展开摘要';
        }
        btn.disabled = false;
    }).catch(function(err) {
        container.innerHTML = '<div class="alert alert-error">摘要获取失败: ' + err.message + '</div>';
        btn.textContent = '展开摘要';
        btn.disabled = false;
    });
}

function renderSummaryCard(container, summary) {
    const basicInfo = summary.basic_info || {};
    const lawName = basicInfo.law_name || summary.law_name || '';
    const jurisdiction = basicInfo.jurisdiction || '';
    const purpose = summary.purpose || '';
    const scope = summary.scope || '';
    const chapters = summary.chapters || [];
    const keywords = summary.keywords || {};
    const compliance = summary.compliance_points || {};
    const legalEffect = summary.legal_effect || {};

    let html = '<div class="card" style="border-color:var(--color-border-deep);">';
    
    // 标题行（包含状态、类型、日期等）
    const status = basicInfo.status || '';
    const statusEmoji = (status.includes('有效') || !status) ? 'OK' : '!';
    const lawType = basicInfo.law_type || '';
    const passingDate = basicInfo.passing_date || '';
    const articleCount = basicInfo.article_count || '';
    
    const titleParts = [escapeHtml(lawName)];
    if (lawType) titleParts.push(escapeHtml(lawType));
    if (status) titleParts.push('[' + statusEmoji + '] ' + escapeHtml(status));
    if (passingDate && passingDate.length >= 10) titleParts.push(passingDate.substring(0, 10) + '通过');
    else if (passingDate) titleParts.push(passingDate + '通过');
    if (articleCount) titleParts.push(articleCount + '条');
    
    html += '<h4 style="font-family:var(--font-serif);font-size:1rem;font-weight:600;margin-bottom:0.5rem;">' + titleParts.join(' | ') + '</h4>';
    if (jurisdiction) html += '<div style="font-size:0.75rem;color:var(--color-text-muted);margin-bottom:0.75rem;">' + escapeHtml(jurisdiction) + '</div>';

    // 立法目的和适用范围
    html += '<div class="grid-2" style="gap:0.75rem;margin-bottom:0.75rem;">';
    html += '<div><strong>立法目的</strong><p style="font-size:0.8125rem;color:var(--color-text-secondary);margin-top:0.25rem;">' + escapeHtml(purpose.substring(0, 200)) + (purpose.length > 200 ? '...' : '') + '</p></div>';
    html += '<div><strong>适用范围</strong><p style="font-size:0.8125rem;color:var(--color-text-secondary);margin-top:0.25rem;">' + escapeHtml(scope.substring(0, 200)) + (scope.length > 200 ? '...' : '') + '</p></div>';
    html += '</div>';

    // 企业合规要点（新增）
    if (compliance && compliance.core_obligations) {
        html += '<details style="margin-top:0.5rem;"><summary style="cursor:pointer;font-size:0.8125rem;font-weight:600;">企业合规要点</summary>';
        html += '<div style="margin-top:0.5rem;">';
        const obligations = compliance.core_obligations || [];
        const obligationCount = compliance.obligation_count || obligations.length;
        html += '<div style="font-size:0.8125rem;margin-bottom:0.375rem;"><strong>核心义务: 制度、人员、报告 (' + obligationCount + '项)</strong></div>';
        obligations.slice(0, 5).forEach(function(obl, idx) {
            html += '<div style="font-size:0.8125rem;margin-bottom:0.25rem;padding-left:1rem;">' + (idx + 1) + '. ' + escapeHtml(obl) + '</div>';
        });
        if (compliance.penalties) {
            html += '<div style="font-size:0.8125rem;margin-top:0.375rem;"><strong>违规后果:</strong> ' + escapeHtml(compliance.penalties.substring(0, 150)) + (compliance.penalties.length > 150 ? '...' : '') + '</div>';
        }
        if (compliance.key_deadlines && compliance.key_deadlines.length > 0) {
            html += '<div style="font-size:0.8125rem;margin-top:0.375rem;"><strong>关键时限:</strong></div>';
            compliance.key_deadlines.slice(0, 3).forEach(function(dl) {
                html += '<div style="font-size:0.8125rem;padding-left:1rem;">• ' + escapeHtml(dl) + '</div>';
            });
        }
        html += '</div></details>';
    }

    // 法律效力（新增）
    if (legalEffect && Object.keys(legalEffect).length > 0) {
        html += '<details style="margin-top:0.5rem;"><summary style="cursor:pointer;font-size:0.8125rem;font-weight:600;">法律效力</summary>';
        html += '<div style="margin-top:0.5rem;font-size:0.8125rem;">';
        const effectItems = [];
        if (legalEffect.law_type_detail) effectItems.push('类型: ' + escapeHtml(legalEffect.law_type_detail));
        if (legalEffect.current_status) effectItems.push('状态: ' + escapeHtml(legalEffect.current_status));
        if (legalEffect.years_in_effect) effectItems.push('生效时长: ' + escapeHtml(legalEffect.years_in_effect));
        if (legalEffect.hierarchy_position) effectItems.push('层级位置: ' + escapeHtml(legalEffect.hierarchy_position));
        if (effectItems.length > 0) {
            html += '<div>' + effectItems.join(' | ') + '</div>';
        }
        html += '</div></details>';
    }

    // 主要章节
    if (chapters.length > 0) {
        html += '<details style="margin-top:0.5rem;"><summary style="cursor:pointer;font-size:0.8125rem;font-weight:600;">主要章节 (' + chapters.length + '章)</summary>';
        html += '<div style="margin-top:0.5rem;">';
        chapters.slice(0, 8).forEach(function(ch) {
            html += '<div style="font-size:0.8125rem;margin-bottom:0.375rem;"><strong>▶ ' + escapeHtml(ch.title || '') + '</strong> (' + escapeHtml(ch.articles || '') + ')';
            if (ch.summary) html += ' - ' + escapeHtml(ch.summary);
            html += '</div>';
            if (ch.key_articles && ch.key_articles.length > 0) {
                ch.key_articles.slice(0, 2).forEach(function(ka) {
                    html += '<div style="font-size:0.75rem;padding-left:1rem;color:var(--color-text-muted);">• ' + escapeHtml(ka) + '</div>';
                });
            }
        });
        html += '</div></details>';
    }

    // 关键词（支持新旧两种格式）
    if (keywords && typeof keywords === 'object' && !Array.isArray(keywords)) {
        html += '<details style="margin-top:0.5rem;"><summary style="cursor:pointer;font-size:0.8125rem;font-weight:600;">关键词</summary>';
        html += '<div style="margin-top:0.5rem;">';
        
        // 新版格式：分类字典
        const categoryIcons = { core: '核心', subjects: '主体', responsibilities: '责任' };
        Object.keys(categoryIcons).forEach(function(key) {
            const iconLabel = categoryIcons[key];
            const kwList = keywords[key] || [];
            if (kwList.length > 0) {
                html += '<div style="margin-bottom:0.375rem;"><strong>' + iconLabel + '</strong>: ';
                kwList.slice(0, 5).forEach(function(kw, idx) {
                    html += '<code style="font-size:0.75rem;background:var(--color-bg);padding:2px 6px;border-radius:3px;">' + escapeHtml(kw) + '</code>';
                    if (idx < Math.min(kwList.length, 5) - 1) html += ' ';
                });
                html += '</div>';
            }
        });
        
        // 旧版格式：简单列表
        if (Array.isArray(keywords) && keywords.length > 0) {
            html += '<div style="display:flex;flex-wrap:wrap;gap:4px;">';
            keywords.slice(0, 8).forEach(function(kw) {
                html += '<code style="font-size:0.75rem;background:var(--color-bg);padding:2px 6px;border-radius:3px;">' + escapeHtml(kw) + '</code>';
            });
            html += '</div>';
        }
        
        html += '</div></details>';
    }

    html += '</div>';
    container.innerHTML = html;
}

function findSimilar(lawId, btn) {
    const container = document.getElementById('similar-' + lawId);
    if (!container) return;

    if (container.innerHTML.trim()) {
        container.innerHTML = '';
        btn.textContent = '找相似';
        return;
    }

    btn.textContent = '查找中...';
    btn.disabled = true;

    api.get('/search/law/' + lawId + '/similar', { top_k: 5 }).then(function(data) {
        if (data && data.data && data.data.similar_laws) {
            renderSimilarLaws(container, data.data, lawId);
            btn.textContent = '收起相似';
        } else {
            container.innerHTML = '<div class="alert alert-info">暂未找到相似法规</div>';
            btn.textContent = '找相似';
        }
        btn.disabled = false;
    }).catch(function(err) {
        container.innerHTML = '<div class="alert alert-error">查找失败: ' + err.message + '</div>';
        btn.textContent = '找相似';
        btn.disabled = false;
    });
}

function renderSimilarLaws(container, simData, currentLawId) {
    const similarLaws = simData.similar_laws || [];
    if (similarLaws.length === 0) {
        container.innerHTML = '<div class="alert alert-info">暂未找到相似法规</div>';
        return;
    }

    let html = '<div class="card" style="border-color:var(--color-border-deep);">';
    html += '<div class="card-title">相关法规推荐</div>';
    html += '<div style="font-size:0.75rem;color:var(--color-text-muted);margin-bottom:0.75rem;">基于《' + escapeHtml(simData.query_law_title || currentLawId) + '》的语义分析，从 ' + (simData.total_found || 0) + ' 部相关法规中推荐：</div>';

    similarLaws.forEach(function(item, idx) {
        const simPercent = Math.round((item.similarity || 0) * 100);
        let barColor = '#d32f2f';
        if (simPercent >= 75) barColor = '#2e7d32';
        else if (simPercent >= 55) barColor = '#ed6c02';

        html += '<div style="margin-bottom:0.75rem;">';
        html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">';
        html += '<span style="font-size:0.75rem;color:var(--color-text-muted);min-width:42px;">相似度</span>';
        html += '<div style="flex:1;background:var(--color-border);height:8px;">';
        html += '<div style="width:' + simPercent + '%;background:' + barColor + ';height:100%;"></div>';
        html += '</div>';
        html += '<span style="font-size:0.75rem;font-weight:600;color:' + barColor + ';min-width:38px;text-align:right;">' + simPercent + '%</span>';
        html += '</div>';

        html += '<div style="background:var(--color-bg);border:1px solid var(--color-border);padding:0.75rem 1rem;">';
        html += '<div style="display:flex;justify-content:space-between;align-items:flex-start;">';
        html += '<div style="flex:1;">';
        html += '<div style="font-weight:600;font-size:0.875rem;color:var(--color-text);margin-bottom:0.25rem;">' + escapeHtml(item.title || '未知法规') + '</div>';
        html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:0.375rem;">';
        html += '<span class="badge badge-info">' + escapeHtml(item.jurisdiction || '') + '</span>';
        html += '<span class="badge badge-warning">' + (item.match_chunk_count || 0) + '个匹配片段</span>';
        html += '<span class="badge badge-muted">ID: ' + escapeHtml(item.law_id || '') + '</span>';
        html += '</div>';
        html += '</div>';
        html += '<button class="btn btn-secondary btn-sm" onclick="viewLawDetail(\'' + (item.law_id || '') + '\')">查看</button>';
        html += '</div>';

        // 匹配条款预览
        const topChunks = item.top_matched_chunks || [];
        if (topChunks.length > 0) {
            html += '<details style="margin-top:0.375rem;"><summary style="cursor:pointer;font-size:0.75rem;color:var(--color-text-muted);">匹配条款预览（' + topChunks.length + '条）</summary>';
            html += '<div style="margin-top:0.375rem;">';
            topChunks.slice(0, 3).forEach(function(chunk) {
                const chunkSim = Math.round((chunk.similarity || 0) * 100);
                html += '<div style="background:var(--color-bg);border-left:3px solid var(--color-primary);padding:0.5rem 0.75rem;margin:0.375rem 0;font-size:0.8125rem;">';
                html += '<div style="font-weight:600;color:var(--color-text);margin-bottom:0.25rem;">' + escapeHtml(chunk.article_number || '条款') + ' <span style="color:var(--color-text-muted);font-weight:400;font-size:0.72rem;">(相似度 ' + chunkSim + '%)</span></div>';
                html += '<div style="color:var(--color-text-secondary);">' + escapeHtml((chunk.content_preview || '').substring(0, 150)) + '</div>';
                html += '</div>';
            });
            html += '</div></details>';
        }

        html += '</div>';
    });

    html += '</div>';
    container.innerHTML = html;
}
