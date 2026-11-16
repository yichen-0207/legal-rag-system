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
                const opt = document.createElement('option');
                opt.value = value;
                opt.textContent = value + ' (' + (t.count || 0) + ')';
                topicSelect.appendChild(opt);

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

        // 恢复从法规详情返回前的搜索条件
        restoreSearchState();
    }).catch(function() { /* 静默失败 */ });

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

function renderSearchResults(container, results, query) {
    // 只渲染前 50 条避免 DOM 过大；引用链扩展项（retrieval_hop=2）被追加在末尾且不参与排名，
    // 若截断将永远看不到，其数量有后端上界，故一并取回渲染。
    var topResults = results.slice(0, 50);
    var hopResults = results.filter(function(item) { return item.retrieval_hop === 2; });
    var displayResults = topResults.concat(hopResults);

    function groupByLaw(items) {
        const lawDict = {};
        items.forEach(function(item) {
            const metadata = item.metadata || {};
            const lawId = metadata.law_id || item.law_id;
            if (!lawId) return;

            // 展示分只认后端 match_score（真实余弦 0~1）；旧 similarity/score 是 RRF 反推的伪相似度，不可用。
            const mscore = (item.match_score === undefined || item.match_score === null) ? null : item.match_score;

            if (!lawDict[lawId]) {
                lawDict[lawId] = {
                    law_id: lawId,
                    title: metadata.title || item.title || '未知法规',
                    jurisdiction: metadata.jurisdiction || item.jurisdiction || '未知',
                    max_score: mscore,
                    clauses: []
                };
            }

            const article = metadata.article_number || item.article_number || '条款';
            const content = item.content || '';

            const existing = lawDict[lawId].clauses.find(function(c) { return c.article_number === article; });
            if (existing) {
                // 同一条款可能既被语义命中、又被引用链补入（组件D）：保留得分高者，
                // 标记跟随胜出项，避免直接命中的条款被误标成「引用补入」
                const exScore = (existing.match_score === null || existing.match_score === undefined) ? -1 : existing.match_score;
                const newScore = (mscore === null) ? -1 : mscore;
                if (newScore > exScore) {
                    existing.content = content;
                    existing.match_score = mscore;
                    existing.retrieval_hop = item.retrieval_hop;
                    existing.referenced_by = item.referenced_by;
                    existing.cross_law = item.cross_law;
                    existing.referenced_law_id = item.referenced_law_id;
                    existing.referenced_law_name = item.referenced_law_name;
                }
            } else {
                lawDict[lawId].clauses.push({
                    article_number: article,
                    content: content,
                    match_score: mscore,
                    retrieval_hop: item.retrieval_hop,
                    referenced_by: item.referenced_by,
                    cross_law: item.cross_law,
                    referenced_law_id: item.referenced_law_id,
                    referenced_law_name: item.referenced_law_name
                });
            }

            const curMax = lawDict[lawId].max_score;
            if (mscore !== null && (curMax === null || curMax === undefined || mscore > curMax)) {
                lawDict[lawId].max_score = mscore;
            }
        });
        return Object.values(lawDict);
    }

    // 不分档：全部法规按「该法下最高条款匹配度」降序；缺失分数（引用链补入项）排在最后。
    var laws = groupByLaw(displayResults).sort(function(a, b) {
        var aScore = (a.max_score === null || a.max_score === undefined) ? -1 : a.max_score;
        var bScore = (b.max_score === null || b.max_score === undefined) ? -1 : b.max_score;
        return bScore - aScore;
    });

    var html = '<div class="alert alert-success">检索成功</div>';
    if (laws.length === 0) {
        html += '<div class="alert alert-warning">未找到相关法规</div>';
        container.innerHTML = html;
        return;
    }
    html += '<div class="law-result-list">';
    laws.forEach(function(law) {
        // 每部法只保留匹配度最高的前 3 条，条款内同样按匹配度降序
        law.clauses = law.clauses.slice().sort(function(a, b) {
            var aScore = (a.match_score === null || a.match_score === undefined) ? -1 : a.match_score;
            var bScore = (b.match_score === null || b.match_score === undefined) ? -1 : b.match_score;
            return bScore - aScore;
        }).slice(0, 3);
        html += renderLawCard(law);
    });
    html += '</div>';
    container.innerHTML = html;
}

function renderLawCard(law) {
    const lawId = law.law_id;
    // 最高匹配度取该法规下条款 match_score 的最大值；全部缺失则不显示（旧实现用伪相似度，
    // 恒为 50%，会产生「所有法规都是 50%」的误导）。
    const hasMax = (law.max_score !== null && law.max_score !== undefined);
    const maxScorePercent = hasMax ? Math.round(law.max_score * 100) : null;
    let scoreColor = '#d32f2f';
    if (hasMax) {
        if (maxScorePercent >= 70) scoreColor = '#2e7d32';
        else if (maxScorePercent >= 50) scoreColor = '#ed6c02';
    }

    let html = '<div class="card law-card">';
    html += '<div class="law-card-head">';
    html += '<div>';
    html += '<h3 class="law-card-title">' + escapeHtml(law.title) + '</h3>';
    html += '<span class="badge badge-info">' + escapeHtml(law.jurisdiction) + '</span>';
    html += '</div>';
    html += '<span class="law-card-score" style="color:' + scoreColor + ';">' + (hasMax ? '最高匹配度 ' + maxScorePercent + '%' : '') + '</span>';
    html += '</div>';

    // 条款已在上层按匹配度降序截取前 3 条，这里只负责渲染
    law.clauses.forEach(function(clause) {
        const hasC = (clause.match_score !== null && clause.match_score !== undefined);
        const cPercent = hasC ? Math.round(clause.match_score * 100) : null;
        // retrieval_hop=2 为引用链补入项，无余弦分，故标注引用来源而非匹配度。
        const isRef = clause.retrieval_hop === 2 && clause.referenced_by;
        // 跨法规引用的条号属于另一部法规，必须带法规名，避免误读为本法规条号。
        let refLabel = '引用自第 ' + escapeHtml(clause.referenced_by) + ' 条';
        if (clause.cross_law) {
            const srcLaw = clause.referenced_law_name || clause.referenced_law_id || '其他法规';
            refLabel = '引用自《' + escapeHtml(srcLaw) + '》第 ' + escapeHtml(clause.referenced_by) + ' 条';
        }
        const meta = isRef
            ? '<span class="badge badge-ref">' + refLabel + '</span>'
            : (hasC ? '<span style="color:var(--color-text-muted);font-weight:400;">(' + cPercent + '%)</span>' : '');
        html += '<div style="background:var(--color-bg);border-left:3px solid ' + (isRef ? 'var(--color-text-muted)' : 'var(--color-primary)') + ';padding:0.5rem 0.75rem;margin-bottom:0.5rem;font-size:0.8125rem;">';
        html += '<div style="font-weight:600;margin-bottom:0.25rem;">' + escapeHtml(clause.article_number) + ' ' + meta + '</div>';
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

// 进详情前保存搜索条件：不存结果 HTML（旧 DOM 快照会与最新代码/后端数据脱节），
// 返回时按条件重新检索。
function saveSearchState() {
    var state = {
        query: document.getElementById('search-query').value,
        jurisdiction: document.getElementById('search-jurisdiction').value,
        topics: Array.from(document.getElementById('search-topic').selectedOptions).map(function(o) { return o.value; })
    };
    try { sessionStorage.setItem('search_state', JSON.stringify(state)); } catch(e) {}
}

// 返回时先还原筛选控件再重新检索；调用点在主题下拉构建完成后，可安全回填选中态。
function restoreSearchState() {
    var raw;
    try { raw = sessionStorage.getItem('search_state'); } catch(e) {}
    if (!raw) return;

    var state;
    try { state = JSON.parse(raw); } catch(e) {}
    // 无论解析成功与否都先清除：避免损坏的快照被反复读取、每次都白跑一次恢复
    try { sessionStorage.removeItem('search_state'); } catch(e) {}
    if (!state) return;

    var hasFilter = false;

    if (state.query) {
        document.getElementById('search-query').value = state.query;
        hasFilter = true;
    }

    if (state.jurisdiction) {
        document.getElementById('search-jurisdiction').value = state.jurisdiction;
        hasFilter = true;
    }

    if (state.topics && state.topics.length > 0) {
        hasFilter = true;
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

    // 有条件才重新检索，避免每次进入本页都请求全量法规列表。
    if (hasFilter) doSearch();
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

    html += '<div class="grid-2" style="gap:0.75rem;margin-bottom:0.75rem;">';
    html += '<div><strong>立法目的</strong><p style="font-size:0.8125rem;color:var(--color-text-secondary);margin-top:0.25rem;">' + escapeHtml(purpose.substring(0, 200)) + (purpose.length > 200 ? '...' : '') + '</p></div>';
    html += '<div><strong>适用范围</strong><p style="font-size:0.8125rem;color:var(--color-text-secondary);margin-top:0.25rem;">' + escapeHtml(scope.substring(0, 200)) + (scope.length > 200 ? '...' : '') + '</p></div>';
    html += '</div>';

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
