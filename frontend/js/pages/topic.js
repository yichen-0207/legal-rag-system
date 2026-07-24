/**
 * 跨法域专题分析页面
 */
registerPage('topic', function(container) {
    container.innerHTML = `
        <h1 class="page-title">跨法域合规地图</h1>
        <p class="page-desc">选择专题和法域，系统构建结构化对比表与合规仪表盘，支持 AI 深度解读。</p>
        <hr class="divider">

        <!-- 选择面板 -->
        <div style="margin-bottom:1.5rem;">
            <div style="margin-bottom:0.75rem;">
                <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">选择对比专题</label>
                <select id="topic-select" class="select">
                    <option value="">加载中...</option>
                </select>
                <div id="topic-desc" style="font-size:0.75rem;color:var(--color-text-muted);margin-top:0.25rem;"></div>
            </div>
            <div class="grid-2" style="margin-bottom:0.75rem;">
                <div>
                    <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">法域 A</label>
                    <select id="topic-jur-a" class="select"></select>
                </div>
                <div>
                    <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">法域 B</label>
                    <select id="topic-jur-b" class="select"></select>
                </div>
            </div>
            <div class="grid-2" style="margin-bottom:0.75rem;">
                <div>
                    <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">每法域检索条款数: <span id="topic-topn-val">8</span></label>
                    <input type="range" id="topic-topn" min="5" max="20" value="8" style="width:100%;">
                </div>
                <div style="display:flex;align-items:flex-end;">
                    <label style="font-size:0.8125rem;color:var(--color-text-secondary);display:flex;align-items:center;gap:0.375rem;">
                        <input type="checkbox" id="topic-include-summary" checked> 生成AI差异总结
                    </label>
                </div>
            </div>
            <button id="topic-run" class="btn btn-primary">生成分析报告</button>
        </div>

        <!-- 状态提示 -->
        <div id="topic-status"></div>

        <!-- 结果区 -->
        <div id="topic-result"></div>
    `;

    // 加载专题列表和法域
    Promise.all([
        api.get('/structured-analysis/topics'),
        api.get('/search/jurisdictions')
    ]).then(function(results) {
        const topicData = results[0];
        const jurData = results[1];

        const topicSelect = document.getElementById('topic-select');
        topicSelect.innerHTML = '';
        if (topicData && topicData.data) {
            topicData.data.forEach(function(t) {
                const opt = document.createElement('option');
                opt.value = t.id;
                opt.textContent = t.name;
                opt.dataset.desc = t.description || '';
                topicSelect.appendChild(opt);
            });
            topicSelect.addEventListener('change', function() {
                const desc = this.options[this.selectedIndex].dataset.desc || '';
                document.getElementById('topic-desc').textContent = desc;
            });
            // 触发初始描述
            if (topicSelect.options.length > 0) {
                document.getElementById('topic-desc').textContent = topicSelect.options[0].dataset.desc || '';
            }
        }

        const jurA = document.getElementById('topic-jur-a');
        const jurB = document.getElementById('topic-jur-b');
        if (jurData && jurData.data) {
            jurData.data.forEach(function(j, idx) {
                const optA = document.createElement('option');
                optA.value = j; optA.textContent = j;
                jurA.appendChild(optA);
                const optB = document.createElement('option');
                optB.value = j; optB.textContent = j;
                jurB.appendChild(optB);
            });
            if (jurData.data.length > 1) jurB.selectedIndex = 1;
        }
    }).catch(function() {});

    // TopN 滑块
    document.getElementById('topic-topn').addEventListener('input', function() {
        document.getElementById('topic-topn-val').textContent = this.value;
    });

    // 运行分析
    document.getElementById('topic-run').addEventListener('click', runTopicAnalysis);
});

function runTopicAnalysis() {
    const topicId = document.getElementById('topic-select').value;
    const jurA = document.getElementById('topic-jur-a').value;
    const jurB = document.getElementById('topic-jur-b').value;
    const topN = parseInt(document.getElementById('topic-topn').value);
    const includeSummary = document.getElementById('topic-include-summary').checked;

    if (!topicId) { alert('请选择专题'); return; }
    if (jurA === jurB) { alert('请选择不同的法域'); return; }

    const statusEl = document.getElementById('topic-status');
    const resultEl = document.getElementById('topic-result');

    statusEl.innerHTML = '<div class="alert alert-info">正在生成合规地图，请稍候...</div>';
    resultEl.innerHTML = '';

    renderSkeleton();

    const params = {
        topic_id: topicId,
        jurisdiction_a: jurA,
        jurisdiction_b: jurB,
        top_n: topN,
        include_summary: includeSummary
    };

    const url = new URL(API_BASE + '/structured-analysis/stream');
    Object.entries(params).forEach(function(kv) { url.searchParams.set(kv[0], kv[1]); });

    const eventSource = new EventSource(url.toString());

    const collectedData = {
        topic: {},
        extraction: {},
        dashboard: null,
        summary: '',
        dimensions: [],
        jurisdictions: [jurA, jurB],
        timing: {}
    };

    let isFirstRender = true;

    function updateTimingDisplay() {
        const timingEl = document.getElementById('topic-timing');
        if (!timingEl) return;

        const timing = collectedData.timing;
        let html = '<div style="display:flex;flex-wrap:wrap;gap:0.75rem;margin-top:0.5rem;">';
        
        const timingLabels = {
            search: '检索',
            llm_extraction: 'LLM提取',
            llm_dashboard: '仪表盘',
            total: '总计'
        };

        for (const [key, label] of Object.entries(timingLabels)) {
            if (timing[key]) {
                html += '<span class="badge badge-muted" style="padding:0.25rem 0.5rem;font-size:0.75rem;">';
                html += '<span style="color:var(--color-text-secondary);">' + label + ':</span> ';
                html += '<span style="font-weight:600;color:var(--color-primary);">' + timing[key] + '</span>';
                html += '</span>';
            }
        }
        
        
        
        html += '</div>';
        timingEl.innerHTML = html;
    }

    function switchTab(tabName) {
        const tab = document.querySelector('.topic-tab-btn[data-tab="' + tabName + '"]');
        if (!tab) return;
        const target = document.getElementById('topic-tab-' + tabName);
        if (!target || target.style.display === 'block') return; // 已在此tab
        // 模拟点击切换
        tab.click();
    }

    function renderSkeleton() {
        let html = '';
        html += '<div style="margin-bottom:1rem;display:flex;gap:0.5rem;">';
        html += '<button class="btn btn-primary btn-sm topic-tab-btn" data-tab="dashboard">合规仪表盘</button>';
        html += '<button class="btn btn-secondary btn-sm topic-tab-btn" data-tab="table">结构化对比表</button>';
        html += '</div>';

        html += '<div id="topic-tab-dashboard">';
        html += '<div class="skeleton-card" style="margin-bottom:1rem;"><div class="skeleton-title"></div><div class="skeleton-text"></div></div>';
        html += '<div class="grid-3">';
        html += '<div class="skeleton-card"><div class="skeleton-circle"></div><div class="skeleton-text"></div></div>';
        html += '<div class="skeleton-card"><div class="skeleton-circle"></div><div class="skeleton-text"></div></div>';
        html += '<div class="skeleton-card"><div class="skeleton-bar"></div><div class="skeleton-text"></div></div>';
        html += '</div>';
        html += '<div class="skeleton-card" style="margin-top:1rem;height:350px;"><div class="skeleton-chart"></div></div>';
        html += '<div class="skeleton-card" style="margin-top:1rem;"><div class="skeleton-title"></div><div class="skeleton-text"></div><div class="skeleton-text"></div><div class="skeleton-text"></div></div>';
        html += '</div>';

        html += '<div id="topic-tab-table" style="display:none;">';
        html += '<div id="topic-table-loading" class="alert alert-info" style="margin-bottom:1rem;">正在生成对比表...</div>';
        html += '<div id="topic-table-content"></div>';
        html += '</div>';

        html += '<div id="topic-timing"></div>';
        html += '<hr class="divider" style="margin-top:2rem;">';
        html += '<h2 class="section-title">继续追问 AI</h2>';
        html += '<div class="card" style="margin-bottom:1rem;">';
        html += '<div style="display:flex;gap:0.5rem;margin-bottom:0.75rem;">';
        html += '<input type="text" id="topic-follow-up-input" class="input" placeholder="例如：这两个法域在处罚力度上还有什么不同？" style="flex:1;">';
        html += '<button id="topic-follow-up-btn" class="btn btn-primary">追问</button>';
        html += '</div>';
        html += '<div id="topic-follow-up-history"></div>';
        html += '</div>';

        resultEl.innerHTML = html;

        resultEl.querySelectorAll('.topic-tab-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                resultEl.querySelectorAll('.topic-tab-btn').forEach(function(b) {
                    b.className = 'btn btn-secondary btn-sm topic-tab-btn';
                });
                this.className = 'btn btn-primary btn-sm topic-tab-btn';
                resultEl.querySelectorAll('[id^="topic-tab-"]').forEach(function(el) { el.style.display = 'none'; });
                document.getElementById('topic-tab-' + this.dataset.tab).style.display = 'block';

                if (this.dataset.tab === 'table') {
                    renderTableRows();
                }
            });
        });
    }

    function renderDashboard(dashboard) {
        const tabEl = document.getElementById('topic-tab-dashboard');
        if (tabEl) {
            tabEl.innerHTML = renderTopicDashboard(dashboard, jurA, jurB);
            renderTopicRadar(dashboard, jurA, jurB);
            bindAIInterpretButtons();
        }
        statusEl.innerHTML = '<div class="alert alert-success">合规仪表盘已就绪！</div>';
    }

    let tableRenderTimer = null;
    function renderStreamTable() {
        const contentEl = document.getElementById('topic-table-content');
        const loadingEl = document.getElementById('topic-table-loading');
        const dimensions = collectedData.dimensions || [];
        const dataA = collectedData.extraction[jurA] || null;
        const dataB = collectedData.extraction[jurB] || null;

        if (!contentEl) return;

        // 如果没有 dimensions 或两个法域数据都没有，暂时不渲染
        if (dimensions.length === 0 || (!dataA && !dataB)) {
            if (loadingEl) loadingEl.style.display = 'block';
            return;
        }

        if (loadingEl) loadingEl.style.display = 'none';

        let fullHtml = '<h2 class="section-title">结构化对比表</h2>';

        function renderSourceArticlesList(items, idPrefix) {
            let articles = [];
            if (Array.isArray(items)) {
                articles = items;
            } else if (typeof items === 'string') {
                try {
                    articles = JSON.parse(items.replace(/'/g, '"'));
                    if (!Array.isArray(articles)) {
                        articles = [items];
                    }
                } catch (e) {
                    articles = [items];
                }
            } else {
                articles = ['—'];
            }
            if (articles.length === 0) {
                return '<div style="font-size:0.875rem;color:var(--color-text-muted);">无</div>';
            }
            const maxVisible = 3;
            const isLong = articles.length > maxVisible;
            const visibleArticles = articles.slice(0, maxVisible);
            let html = '<div class="source-articles-container">';
            visibleArticles.forEach(function(article, i) {
                html += '<div style="font-size:0.875rem;color:var(--color-text);line-height:1.6;padding:0.25rem 0;border-bottom:1px solid var(--color-border);">' + escapeHtml(article) + '</div>';
            });
            if (isLong) {
                html += '<div id="' + idPrefix + '-full" style="display:none;">';
                articles.slice(maxVisible).forEach(function(article) {
                    html += '<div style="font-size:0.875rem;color:var(--color-text);line-height:1.6;padding:0.25rem 0;border-bottom:1px solid var(--color-border);">' + escapeHtml(article) + '</div>';
                });
                html += '</div>';
                html += '<button onclick="toggleSourceArticles(\'' + idPrefix + '\')" id="' + idPrefix + '-toggle" style="margin-top:0.5rem;padding:0.25rem 0.5rem;font-size:0.75rem;color:var(--color-primary);background:none;border:none;cursor:pointer;text-decoration:underline;">展开全部 (' + articles.length + '条)</button>';
            }
            html += '</div>';
            return html;
        }

        // source_articles 展开/收起全局函数
        if (!window.toggleSourceArticles) {
            window.toggleSourceArticles = function(idPrefix) {
                const fullEl = document.getElementById(idPrefix + '-full');
                const toggleBtn = document.getElementById(idPrefix + '-toggle');
                if (fullEl && toggleBtn) {
                    if (fullEl.style.display === 'none') {
                        fullEl.style.display = 'block';
                        toggleBtn.textContent = '收起';
                    } else {
                        fullEl.style.display = 'none';
                        toggleBtn.textContent = '展开全部 (' + fullEl.children.length + '条)';
                    }
                }
            };
        }

        dimensions.forEach(function(dim, idx) {
            const key = dim.key;
            const label = dim.label;
            const valA = dataA ? dataA[key] : null;
            const valB = dataB ? dataB[key] : null;
            const hasA = valA !== null && valA !== undefined && valA !== '—';
            const hasB = valB !== null && valB !== undefined && valB !== '—';

            const payload = JSON.stringify({
                dimension_label: label,
                detail_a: hasA ? (typeof valA === 'string' ? valA : JSON.stringify(valA)) : '暂无数据',
                detail_b: hasB ? (typeof valB === 'string' ? valB : JSON.stringify(valB)) : '暂无数据'
            }).replace(/"/g, '&quot;');

            const rowHtml = '<div class="card" style="margin-bottom:0.75rem;' + (hasA && hasB ? '' : 'opacity:0.7;') + '">';
            const rowContent = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">';
            const rowTitle = '<h3 style="font-family:var(--font-serif);font-size:1rem;font-weight:600;margin:0;">' + escapeHtml(label) + '</h3>';
            const rowBtn = hasA && hasB ? '<button class="btn btn-secondary btn-sm ai-interpret-btn" data-target="ai-interpret-table-' + key + '" data-payload="' + payload + '">AI 解读</button>' : '';
            const rowEnd = '</div><div id="ai-interpret-table-' + key + '" class="ai-interpret-result" style="display:none;margin-bottom:0.75rem;"></div><div class="grid-2">';
            const colA = '<div style="padding:0.75rem;background:var(--color-bg);border-radius:6px;">';
            const colAHeader = '<div style="font-size:0.75rem;font-weight:600;color:var(--color-primary);margin-bottom:0.375rem;">' + escapeHtml(jurA) + (hasA ? '' : ' <span style="color:var(--color-text-muted);font-weight:400;">（等待中...）</span>') + '</div>';

            let colAValue, colBValue;
            if (key === 'source_articles') {
                colAValue = hasA
                    ? '<div class="source-articles-list">' + renderSourceArticlesList(valA, 'table-source-a-' + key + '-' + idx) + '</div></div>'
                    : '<div style="font-size:0.875rem;color:var(--color-text-muted);">等待数据...</div></div>';
                colBValue = hasB
                    ? '<div class="source-articles-list">' + renderSourceArticlesList(valB, 'table-source-b-' + key + '-' + idx) + '</div></div>'
                    : '<div style="font-size:0.875rem;color:var(--color-text-muted);">等待数据...</div></div>';
            } else {
                colAValue = hasA
                    ? '<div style="font-size:0.875rem;color:var(--color-text);line-height:1.6;">' + escapeHtml(typeof valA === 'string' ? valA : JSON.stringify(valA)) + '</div></div>'
                    : '<div style="font-size:0.875rem;color:var(--color-text-muted);">等待数据...</div></div>';
                colBValue = hasB
                    ? '<div style="font-size:0.875rem;color:var(--color-text);line-height:1.6;">' + escapeHtml(typeof valB === 'string' ? valB : JSON.stringify(valB)) + '</div></div>'
                    : '<div style="font-size:0.875rem;color:var(--color-text-muted);">等待数据...</div></div>';
            }

            const colB = '<div style="padding:0.75rem;background:var(--color-bg);border-radius:6px;">';
            const colBHeader = '<div style="font-size:0.75rem;font-weight:600;color:var(--color-secondary);margin-bottom:0.375rem;">' + escapeHtml(jurB) + (hasB ? '' : ' <span style="color:var(--color-text-muted);font-weight:400;">（等待中...）</span>') + '</div>';
            const end = '</div></div>';

            fullHtml += rowHtml + rowContent + rowTitle + rowBtn + rowEnd + colA + colAHeader + colAValue + colB + colBHeader + colBValue + end;
        });

        if (collectedData.summary) {
            fullHtml += '<h2 class="section-title">AI 差异要点</h2>';
            fullHtml += '<div class="alert alert-info">' + escapeHtml(collectedData.summary) + '</div>';
        }

        contentEl.innerHTML = fullHtml;

        // 如果两个法域数据都全了，绑定AI解读按钮
        if (dataA && dataB) {
            bindAIInterpretButtons();
        }
    }

    function renderTableRows() {
        // 改为委托给 renderStreamTable，保持向后兼容（tab点击仍然可用）
        renderStreamTable();
    }

    function bindAIInterpretButtons() {
        resultEl.querySelectorAll('.ai-interpret-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                const payload = JSON.parse(btn.dataset.payload || '{}');
                const targetId = btn.dataset.target;
                const topicName = collectedData.topic.name || '';
                requestAIInterpret(targetId, payload, jurA, jurB, topicName);
            });
        });

        const followUpBtn = document.getElementById('topic-follow-up-btn');
        const followUpInput = document.getElementById('topic-follow-up-input');
        if (followUpBtn && followUpInput) {
            followUpBtn.addEventListener('click', function() {
                const question = followUpInput.value.trim();
                if (!question) return;
                requestFollowUp(question, collectedData, jurA, jurB, collectedData.topic.name || '');
                followUpInput.value = '';
            });
            followUpInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') followUpBtn.click();
            });
        }
    }

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            const type = data.type;

            if (type === 'cache_hit') {
                eventSource.close();
                statusEl.innerHTML = '<div class="alert alert-success">合规地图生成完成！</div>';
                renderTopicResult(resultEl, data.data, jurA, jurB);
                return;
            }

            if (type === 'search_done') {
                if (isFirstRender) {
                    renderSkeleton();
                    isFirstRender = false;
                }
                statusEl.innerHTML = '<div class="alert alert-info">检索完成，正在提取结构化数据...</div>';
                return;
            }

            // 逐维度流式事件：每个维度生成完成时立即更新表格
            if (type === 'extraction_dimension') {
                const jur = data.jurisdiction;
                const key = data.key;
                const value = data.value;
                if (!collectedData.extraction[jur]) {
                    collectedData.extraction[jur] = {};
                }
                collectedData.extraction[jur][key] = value;
                // 首次有维度数据到达时存入 dimensions 并切换到对比表
                if (data.dimensions && data.dimensions.length > 0 && collectedData.dimensions.length === 0) {
                    collectedData.dimensions = data.dimensions;
                }
                switchTab('table');
                renderStreamTable();
                return;
            }

            if (type === 'extracted') {
                const jur = data.jurisdiction;
                collectedData.extraction[jur] = data.data;
                // extracted 事件中附带了 dimensions，提前存入
                if (data.dimensions && data.dimensions.length > 0) {
                    collectedData.dimensions = data.dimensions;
                }
                const keys = Object.keys(collectedData.extraction);
                statusEl.innerHTML = '<div class="alert alert-info">已获取 ' + jur + ' 数据（' + keys.length + '/2）...</div>';
                // 切换到对比表tab（谁先生成就优先展示谁）
                switchTab('table');
                // 逐条渲染对比表
                renderStreamTable();
                return;
            }

            if (type === 'extraction_done') {
                collectedData.dimensions = data.data.dimensions || [];
                collectedData.jurisdictions = data.data.jurisdictions || [jurA, jurB];
                statusEl.innerHTML = '<div class="alert alert-info">两个法域数据已就绪，正在生成仪表盘...</div>';
                // 两个法域数据都到了，重新渲染完整表格
                renderStreamTable();
                return;
            }

            if (type === 'timing_update') {
                collectedData.timing = { ...collectedData.timing, ...data.data };
                updateTimingDisplay();
                return;
            }

            if (type === 'dashboard_done') {
                collectedData.dashboard = data.data;
                collectedData.topic.name = data.data.topic_name || '';
                // 渲染仪表盘数据但不自动切换tab，保持用户当前查看的tab
                renderDashboard(data.data);
                return;
            }

            if (type === 'summary_chunk') {
                collectedData.summary += data.data;
                // 实时更新对比表中的总结区域（如果已渲染）
                const contentEl = document.getElementById('topic-table-content');
                if (contentEl && collectedData.summary) {
                    // 查找或创建 summary 区
                    let summaryEl = document.getElementById('topic-stream-summary');
                    if (!summaryEl) {
                        const existing = contentEl.querySelector('.topic-summary-section');
                        if (existing) {
                            existing.remove();
                        }
                        const section = document.createElement('div');
                        section.className = 'topic-summary-section';
                        section.id = 'topic-stream-summary-container';
                        section.innerHTML = '<h2 class="section-title">AI 差异要点</h2><div id="topic-stream-summary" class="alert alert-info"></div>';
                        contentEl.appendChild(section);
                    }
                    summaryEl = document.getElementById('topic-stream-summary');
                    if (summaryEl) {
                        summaryEl.textContent = collectedData.summary;
                    }
                }
                return;
            }

            if (type === 'done') {
                eventSource.close();
                collectedData.topic = data.data.topic || {};
                collectedData.matched_laws = data.data.matched_laws || {};

                if (!collectedData.dashboard && data.data.dashboard) {
                    collectedData.dashboard = data.data.dashboard;
                    renderDashboard(data.data.dashboard);
                }

                if (!collectedData.extraction || Object.keys(collectedData.extraction).length === 0) {
                    collectedData.extraction = data.data.extraction || {};
                }
                if (!collectedData.dimensions || collectedData.dimensions.length === 0) {
                    const topicData = data.data.topic || {};
                    collectedData.dimensions = topicData.dimensions || [];
                }
                if (!collectedData.summary) {
                    collectedData.summary = data.data.summary || '';
                }

                if (data.data.timing) {
                    collectedData.timing = { ...collectedData.timing, ...data.data.timing };
                    updateTimingDisplay();
                }
                // 不管前面流式渲染有没有成功，最终确保对比表被渲染
                renderStreamTable();
                statusEl.innerHTML = '<div class="alert alert-success">合规地图生成完成！</div>';
                return;
            }

            if (type === 'error') {
                eventSource.close();
                statusEl.innerHTML = '<div class="alert alert-error">分析失败：' + data.data + '</div>';
            }
        } catch (e) {
            console.error('解析 SSE 消息失败:', e);
        }
    };

    eventSource.onerror = function(error) {
        eventSource.close();
        statusEl.innerHTML = '<div class="alert alert-error">连接异常，请重试</div>';
        console.error('SSE 连接错误:', error);
    };
}

function renderTopicResult(container, result, jurA, jurB) {
    const dashboard = result.dashboard || {};
    const extraction = result.extraction || {};
    const summary = result.summary || '';
    const topic = result.topic || {};
    const dimensions = (topic.dimensions || []).filter(function(d) { return d.key !== 'source_articles'; });
    const dataA = extraction[jurA] || {};
    const dataB = extraction[jurB] || {};
    const topicName = dashboard.topic_name || topic.name || '';

    let html = '';

    // 视图切换
    html += '<div style="margin-bottom:1rem;display:flex;gap:0.5rem;">';
    html += '<button class="btn btn-primary btn-sm topic-tab-btn" data-tab="dashboard">合规仪表盘</button>';
    html += '<button class="btn btn-secondary btn-sm topic-tab-btn" data-tab="table">结构化对比表</button>';
    html += '</div>';

    // 仪表盘视图
    html += '<div id="topic-tab-dashboard">';
    html += renderTopicDashboard(dashboard, jurA, jurB);
    html += '</div>';

    // 对比表视图
    html += '<div id="topic-tab-table" style="display:none;">';
    html += renderTopicTable(dimensions, dataA, dataB, jurA, jurB, summary);
    html += '</div>';

    // 全局 AI 追问区
    html += '<hr class="divider" style="margin-top:2rem;">';
    html += '<h2 class="section-title">继续追问 AI</h2>';
    html += '<div class="card" style="margin-bottom:1rem;">';
    html += '<div style="display:flex;gap:0.5rem;margin-bottom:0.75rem;">';
    html += '<input type="text" id="topic-follow-up-input" class="input" placeholder="例如：这两个法域在处罚力度上还有什么不同？" style="flex:1;">';
    html += '<button id="topic-follow-up-btn" class="btn btn-primary">追问</button>';
    html += '</div>';
    html += '<div id="topic-follow-up-history"></div>';
    html += '</div>';

    container.innerHTML = html;

    // 绑定 Tab 切换
    container.querySelectorAll('.topic-tab-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            container.querySelectorAll('.topic-tab-btn').forEach(function(b) {
                b.className = 'btn btn-secondary btn-sm topic-tab-btn';
            });
            this.className = 'btn btn-primary btn-sm topic-tab-btn';
            container.querySelectorAll('[id^="topic-tab-"]').forEach(function(el) { el.style.display = 'none'; });
            document.getElementById('topic-tab-' + this.dataset.tab).style.display = 'block';
        });
    });

    // 渲染雷达图
    renderTopicRadar(dashboard, jurA, jurB);

    // 绑定 AI 解读按钮
    container.querySelectorAll('.ai-interpret-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            const payload = JSON.parse(btn.dataset.payload || '{}');
            const targetId = btn.dataset.target;
            requestAIInterpret(targetId, payload, jurA, jurB, topicName);
        });
    });

    // 绑定追问按钮
    document.getElementById('topic-follow-up-btn').addEventListener('click', function() {
        const input = document.getElementById('topic-follow-up-input');
        const question = input.value.trim();
        if (!question) return;
        requestFollowUp(question, result, jurA, jurB, topicName);
        input.value = '';
    });
    document.getElementById('topic-follow-up-input').addEventListener('keydown', function(e) {
        if (e.key === 'Enter') document.getElementById('topic-follow-up-btn').click();
    });
}

function renderTopicRadar(dashboard, jurA, jurB) {
    const radar = dashboard.radar || {};
    const radarLabels = dashboard.radar_labels || [];
    if (radarLabels.length === 0 || !radar[jurA] || !radar[jurB]) return;

    const canvas = document.getElementById('topic-radar');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();

    const colorA = '#3B82F6';
    const colorB = '#EF4444';

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: radarLabels,
            datasets: [
                {
                    label: jurA,
                    data: radar[jurA],
                    backgroundColor: colorA + '33',
                    borderColor: colorA,
                    pointBackgroundColor: colorA,
                    borderWidth: 2
                },
                {
                    label: jurB,
                    data: radar[jurB],
                    backgroundColor: colorB + '33',
                    borderColor: colorB,
                    pointBackgroundColor: colorB,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 5,
                    ticks: { stepSize: 1, font: { size: 11 } },
                    pointLabels: { font: { size: 12 } }
                }
            },
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

function renderTopicDashboard(dashboard, jurA, jurB) {
    const scores = dashboard.scores || {};
    const scoreReasons = dashboard.score_reasons || {};
    const differences = dashboard.key_differences || [];
    const radar = dashboard.radar || {};
    const radarLabels = dashboard.radar_labels || [];
    const topicName = dashboard.topic_name || '';

    let html = '<h2 class="section-title">「' + topicName + '」合规仪表盘</h2>';
    html += '<div style="font-size:0.8125rem;color:var(--color-text-muted);margin-bottom:1rem;">对比法域：' + jurA + ' vs ' + jurB + '</div>';

    // 评分卡片
    html += '<div class="grid-3">';
    html += renderScoreCard(jurA, scores[jurA] || 3.0, scoreReasons[jurA] || '');
    html += renderScoreCard(jurB, scores[jurB] || 3.0, scoreReasons[jurB] || '');

    const nMajor = differences.filter(function(d) { return d.severity === 'major'; }).length;
    const nModerate = differences.filter(function(d) { return d.severity === 'moderate'; }).length;
    const nMinor = differences.filter(function(d) { return d.severity === 'minor'; }).length;
    html += '<div class="card">';
    html += '<div class="card-title">关键差异点</div>';
    html += '<div style="font-family:var(--font-serif);font-size:1.5rem;font-weight:700;color:var(--color-primary);">' + differences.length + ' 个</div>';
    if (nMajor) html += '<div style="font-size:0.8125rem;color:#DC2626;margin-top:0.25rem;">● 重大 ' + nMajor + '</div>';
    if (nModerate) html += '<div style="font-size:0.8125rem;color:#F59E0B;">● 实质 ' + nModerate + '</div>';
    if (nMinor) html += '<div style="font-size:0.8125rem;color:#10B981;">● 轻微 ' + nMinor + '</div>';
    html += '</div>';
    html += '</div>';

    // 雷达图
    if (radarLabels.length > 0 && radar[jurA] && radar[jurB]) {
        html += '<h2 class="section-title">多维度雷达对比</h2>';
        html += '<div class="card" style="margin-bottom:1rem;">';
        html += '<div style="height:350px;position:relative;"><canvas id="topic-radar"></canvas></div>';
        html += '</div>';
    }

    // 关键差异点详情
    if (differences.length > 0) {
        html += '<h2 class="section-title">关键差异点详情</h2>';
        differences.forEach(function(diff, i) {
            const severityConfig = {
                major: ['重大差异', '#FEE2E2', '#DC2626'],
                moderate: ['实质差异', '#FEF3C7', '#F59E0B'],
                minor: ['轻微差异', '#ECFDF5', '#10B981']
            };
            const cfg = severityConfig[diff.severity] || ['差异', '#F9FAFB', '#666'];
            const payload = JSON.stringify({
                dimension_label: diff.dimension || '',
                detail_a: diff[jurA + '_detail'] || '',
                detail_b: diff[jurB + '_detail'] || ''
            }).replace(/"/g, '&quot;');
            html += '<div class="card" style="margin-bottom:0.75rem;border-left:3px solid ' + cfg[2] + ';">';
            html += '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">';
            html += '<div style="flex:1;">';
            html += '<span class="badge" style="background:' + cfg[1] + ';color:' + cfg[2] + ';">' + cfg[0] + '</span>';
            html += '<strong style="margin-left:0.5rem;">' + escapeHtml(diff.dimension || '') + '</strong>';
            html += '<p style="font-size:0.875rem;color:var(--color-text-secondary);margin-top:0.375rem;">' + escapeHtml(diff.summary || '') + '</p>';
            html += '<div id="ai-interpret-dashboard-' + i + '" class="ai-interpret-result" style="display:none;margin-top:0.75rem;"></div>';
            html += '</div>';
            html += '<div style="flex-shrink:0;">';
            html += '<button class="btn btn-secondary btn-sm ai-interpret-btn" data-target="ai-interpret-dashboard-' + i + '" data-payload="' + payload + '">AI 解读</button>';
            html += '</div>';
            html += '</div>';
            html += '</div>';
        });
    }

    return html;
}

function renderScoreCard(jurisdiction, score, reason) {
    const filled = Math.round(score);
    let stars = '';
    for (let i = 0; i < 5; i++) stars += i < filled ? '&#9679;' : '&#9675;';
    const color = jurisdiction.includes('澳门') || jurisdiction.includes('Macau') ? '#3B82F6' : '#EF4444';

    let html = '<div class="card" style="text-align:center;">';
    html += '<div style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);margin-bottom:0.5rem;">' + escapeHtml(jurisdiction) + ' 保护力度</div>';
    html += '<div style="font-size:1.5rem;color:' + color + ';margin-bottom:0.25rem;">' + stars + '</div>';
    html += '<div style="font-family:var(--font-serif);font-size:1.5rem;font-weight:700;color:' + color + ';">' + score.toFixed(1) + ' / 5.0</div>';
    if (reason) html += '<div style="font-size:0.75rem;color:var(--color-text-muted);margin-top:0.375rem;">' + escapeHtml(reason) + '</div>';
    html += '</div>';
    return html;
}

function renderTopicTable(dimensions, dataA, dataB, jurA, jurB, summary) {
    let html = '<h2 class="section-title">结构化对比表</h2>';

    dimensions.forEach(function(dim) {
        const key = dim.key;
        const label = dim.label;
        const valA = dataA[key] || '—';
        const valB = dataB[key] || '—';
        const payload = JSON.stringify({
            dimension_label: label,
            detail_a: typeof valA === 'string' ? valA : JSON.stringify(valA),
            detail_b: typeof valB === 'string' ? valB : JSON.stringify(valB)
        }).replace(/"/g, '&quot;');

        html += '<div class="card" style="margin-bottom:0.75rem;">';
        html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">';
        html += '<h3 style="font-family:var(--font-serif);font-size:1rem;font-weight:600;margin:0;">' + escapeHtml(label) + '</h3>';
        html += '<button class="btn btn-secondary btn-sm ai-interpret-btn" data-target="ai-interpret-table-' + key + '" data-payload="' + payload + '">AI 解读</button>';
        html += '</div>';
        html += '<div id="ai-interpret-table-' + key + '" class="ai-interpret-result" style="display:none;margin-bottom:0.75rem;"></div>';
        html += '<div class="grid-2">';
        html += '<div style="padding:0.75rem;background:var(--color-bg);border-radius:6px;">';
        html += '<div style="font-size:0.75rem;font-weight:600;color:var(--color-primary);margin-bottom:0.375rem;">' + escapeHtml(jurA) + '</div>';
        html += '<div style="font-size:0.875rem;color:var(--color-text);line-height:1.6;">' + escapeHtml(typeof valA === 'string' ? valA : JSON.stringify(valA)) + '</div>';
        html += '</div>';
        html += '<div style="padding:0.75rem;background:var(--color-bg);border-radius:6px;">';
        html += '<div style="font-size:0.75rem;font-weight:600;color:var(--color-secondary);margin-bottom:0.375rem;">' + escapeHtml(jurB) + '</div>';
        html += '<div style="font-size:0.875rem;color:var(--color-text);line-height:1.6;">' + escapeHtml(typeof valB === 'string' ? valB : JSON.stringify(valB)) + '</div>';
        html += '</div>';
        html += '</div>';
        html += '</div>';
    });

    if (summary) {
        html += '<h2 class="section-title">AI 差异要点</h2>';
        html += '<div class="alert alert-info">' + escapeHtml(summary) + '</div>';
    }

    return html;
}


// AI 解读请求

// AI 解读请求
function requestAIInterpret(targetId, payload, jurA, jurB, topicName) {
    const targetEl = document.getElementById(targetId);
    if (!targetEl) return;

    targetEl.style.display = 'block';
    targetEl.innerHTML = '<div class="loading" style="padding:1rem;"><div class="spinner"></div>AI 正在深度解读...</div>';

    const url = new URL(API_BASE + '/structured-analysis/ai-interpret');
    url.searchParams.set('dimension_label', payload.dimension_label || '');
    url.searchParams.set('detail_a', payload.detail_a || '');
    url.searchParams.set('detail_b', payload.detail_b || '');
    url.searchParams.set('jurisdiction_a', jurA);
    url.searchParams.set('jurisdiction_b', jurB);
    url.searchParams.set('topic_name', topicName);

    fetch(url.toString(), { method: 'POST' }).then(function(resp) { return resp.json(); }).then(function(data) {
        if (data && data.success && data.data) {
            targetEl.innerHTML = '<div class="markdown-body" style="background:var(--color-info-bg);padding:0.75rem 1rem;border-radius:6px;border-left:3px solid var(--color-info);">' + renderMarkdown(data.data.interpretation || '') + '</div>';
        } else {
            targetEl.innerHTML = '<div class="alert alert-error">AI 解读失败：' + (data ? data.message : '未知错误') + '</div>';
        }
    }).catch(function(err) {
        targetEl.innerHTML = '<div class="alert alert-error">AI 解读请求失败：' + err.message + '</div>';
    });
}

// 追问请求
function requestFollowUp(question, result, jurA, jurB, topicName) {
    const historyEl = document.getElementById('topic-follow-up-history');
    if (!historyEl) return;

    const qDiv = document.createElement('div');
    qDiv.className = 'card';
    qDiv.style.marginBottom = '0.75rem';
    qDiv.innerHTML = '<div style="font-weight:600;font-size:0.8125rem;color:var(--color-primary);margin-bottom:0.375rem;">Q: ' + escapeHtml(question) + '</div><div class="follow-up-answer" style="font-size:0.875rem;color:var(--color-text-secondary);"><div class="loading" style="padding:0;"><div class="spinner"></div>思考中...</div></div>';
    historyEl.appendChild(qDiv);

    const answerEl = qDiv.querySelector('.follow-up-answer');

    const url = new URL(API_BASE + '/topic-analysis/follow-up');
    url.searchParams.set('follow_up_question', question);

    fetch(url.toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ original_analysis: result })
    }).then(function(resp) { return resp.json(); }).then(function(data) {
        if (data && data.success && data.data) {
            answerEl.innerHTML = '<div class="markdown-body">' + renderMarkdown(data.data.answer || '') + '</div>';
        } else {
            answerEl.innerHTML = '<div class="alert alert-error" style="margin:0;">追问失败：' + (data ? data.message : '未知错误') + '</div>';
        }
    }).catch(function(err) {
        answerEl.innerHTML = '<div class="alert alert-error" style="margin:0;">追问请求失败：' + err.message + '</div>';
    });
}

// ===== 网络图谱渲染代码已移至独立页面 kg.js =====
