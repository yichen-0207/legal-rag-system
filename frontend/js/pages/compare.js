/**
 * 法规对比分析页面
 */
registerPage('compare', function(container) {
    container.innerHTML = `
        <h1 class="page-title">法规对比分析</h1>
        <p class="page-desc">选择法域和主题，快速查看法规差异</p>
        <hr class="divider">

        <!-- 选择面板 -->
        <div style="margin-bottom:1.5rem;">
            <div class="grid-2" style="margin-bottom:0.75rem;">
                <div>
                    <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">法域 A</label>
                    <select id="cmp-jur-a" class="select"></select>
                </div>
                <div>
                    <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">法域 B</label>
                    <select id="cmp-jur-b" class="select"></select>
                </div>
            </div>
            <div class="grid-2" style="margin-bottom:0.75rem;">
                <div>
                    <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">对比主题</label>
                    <select id="cmp-topic" class="select">
                        <option value="数据保护">数据保护</option>
                        <option value="网络安全">网络安全</option>
                        <option value="人工智能">人工智能</option>
                        <option value="金融监管">金融监管</option>
                        <option value="跨境贸易">跨境贸易</option>
                        <option value="知识产权">知识产权</option>
                        <option value="custom">其他（自定义）</option>
                    </select>
                </div>
                <div>
                    <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">每个法域检索条款数: <span id="cmp-topk-val">5</span></label>
                    <input type="range" id="cmp-topk" min="2" max="15" value="5" style="width:100%;">
                </div>
            </div>
            <div id="cmp-custom-topic-wrap" style="display:none;margin-bottom:0.75rem;">
                <input type="text" id="cmp-custom-topic" class="input" placeholder="例如：跨境数据传输">
            </div>
            <button id="cmp-run" class="btn btn-primary">开始对比</button>
        </div>

        <!-- 状态提示 -->
        <div id="cmp-status"></div>

        <!-- 结果区 -->
        <div id="cmp-result"></div>
    `;

    // 加载法域
    api.get('/search/jurisdictions').then(function(data) {
        if (data && data.data) {
            const jurA = document.getElementById('cmp-jur-a');
            const jurB = document.getElementById('cmp-jur-b');
            data.data.forEach(function(j, idx) {
                const optA = document.createElement('option');
                optA.value = j; optA.textContent = j;
                jurA.appendChild(optA);
                const optB = document.createElement('option');
                optB.value = j; optB.textContent = j;
                jurB.appendChild(optB);
            });
            if (data.data.length > 1) jurB.selectedIndex = 1;
        }
    }).catch(function() {});

    // 主题切换
    document.getElementById('cmp-topic').addEventListener('change', function() {
        document.getElementById('cmp-custom-topic-wrap').style.display = this.value === 'custom' ? 'block' : 'none';
    });

    // TopK 滑块
    document.getElementById('cmp-topk').addEventListener('input', function() {
        document.getElementById('cmp-topk-val').textContent = this.value;
    });

    // 运行对比
    document.getElementById('cmp-run').addEventListener('click', runCompare);
});

function runCompare() {
    const jurA = document.getElementById('cmp-jur-a').value;
    const jurB = document.getElementById('cmp-jur-b').value;
    const topicSelect = document.getElementById('cmp-topic');
    const topic = topicSelect.value === 'custom' ? document.getElementById('cmp-custom-topic').value.trim() : topicSelect.value;
    const topK = parseInt(document.getElementById('cmp-topk').value);

    if (!topic) { alert('请输入或选择对比主题'); return; }
    if (jurA === jurB) { alert('请选择不同的法域'); return; }

    const statusEl = document.getElementById('cmp-status');
    const resultEl = document.getElementById('cmp-result');
    statusEl.innerHTML = '<div class="alert alert-info">正在检索 ' + jurA + ' 和 ' + jurB + ' 的相关条款...</div>';
    resultEl.innerHTML = '';

    const params = {
        jurisdiction_a: jurA,
        jurisdiction_b: jurB,
        topic: topic,
        top_k: topK
    };

    const tStart = Date.now();

    api.get('/compare', params).then(function(data) {
        if (!data || !data.success) {
            statusEl.innerHTML = '<div class="alert alert-error">对比失败：' + (data ? data.message : '未知错误') + '</div>';
            return;
        }

        const result = data.data;
        const isCacheHit = result._cache_hit || false;
        const tAfterSearch = Date.now();
        const searchTime = ((tAfterSearch - tStart) / 1000).toFixed(2) + 's';

        // 尝试获取 AI 深度分析
        const lawsA = result.laws_a || result.results_a || [];
        const lawsB = result.laws_b || result.results_b || [];

        // 先渲染基础结果
        statusEl.innerHTML = '<div class="alert alert-success">对比分析完成</div>';

        // 如果缓存命中且已有 ai_analysis，直接使用
        let aiAnalysis = result.ai_analysis || '';
        if (isCacheHit && aiAnalysis) {
            renderCompareResult(resultEl, result, jurA, jurB, aiAnalysis, searchTime, isCacheHit);
        } else if (lawsA.length > 0 && lawsB.length > 0) {
            renderCompareResult(resultEl, result, jurA, jurB, '', searchTime, isCacheHit);
            let fullAnalysis = '';
            fetchAIDeepAnalysis(lawsA, lawsB, jurA, jurB, topic,
                function(chunk) {
                    fullAnalysis += chunk;
                    const analysisEl = document.getElementById('cmp-ai-analysis-content');
                    if (analysisEl) {
                        analysisEl.innerHTML = renderMarkdown(fullAnalysis) + '<span class="qa-cursor">▌</span>';
                    }
                },
                function() {
                    const analysisEl = document.getElementById('cmp-ai-analysis-content');
                    if (analysisEl) {
                        analysisEl.innerHTML = renderMarkdown(fullAnalysis);
                    }
                    const loadingEl = document.getElementById('cmp-ai-analysis-loading');
                    if (loadingEl) loadingEl.style.display = 'none';

                    api.post('/compare/cache-ai', {
                        topic: topic,
                        jurisdiction_a: jurA,
                        jurisdiction_b: jurB,
                        ai_analysis: fullAnalysis
                    }).catch(function() {});
                },
                function(err) {
                    const analysisEl = document.getElementById('cmp-ai-analysis-content');
                    if (analysisEl) {
                        analysisEl.innerHTML = '<div class="alert alert-error">AI 分析失败: ' + escapeHtml(err) + '</div>';
                    }
                }
            );
        } else {
            renderCompareResult(resultEl, result, jurA, jurB, '', searchTime, isCacheHit);
        }
    }).catch(function(err) {
        statusEl.innerHTML = '<div class="alert alert-error">请求失败：' + err.message + '</div>';
    });
}

function fetchAIDeepAnalysis(lawsA, lawsB, jurA, jurB, topic, onChunk, onDone, onError) {
    function buildContext(label, laws) {
        const parts = ['### ' + label];
        laws.slice(0, 3).forEach(function(l) {
            const arts = (l.article_numbers || []).slice(0, 8).join(', ');
            const kw = (l.keywords || []).slice(0, 6).join(', ');
            const preview = (l.chunks_preview || []).slice(0, 3).map(function(c) {
                return c.article_number + ' ' + c.content.substring(0, 150);
            }).join('\n');
            parts.push('- 《' + l.title + '》(' + (l.article_count || 0) + '条)\n  涉及条款: ' + arts + '\n  关键词: ' + kw + '\n  内容预览:\n' + preview);
        });
        return parts.join('\n');
    }

    const payload = {
        jurisdiction_a: jurA,
        jurisdiction_b: jurB,
        topic: topic,
        context_a: buildContext(jurA, lawsA),
        context_b: buildContext(jurB, lawsB)
    };

    api.streamPost('/compare/deep-analysis', payload, function(chunk) {
        if (chunk.type === 'chunk' && onChunk) {
            onChunk(chunk.data);
        } else if (chunk.type === 'done' && onDone) {
            onDone();
        } else if (chunk.type === 'error' && onError) {
            onError(chunk.data);
        }
    }, function(err) {
        if (onError) onError(err);
    });
}

function renderCompareResult(container, result, jurA, jurB, aiAnalysis, searchTime, isCacheHit) {
    const stats = result.stats || {};
    const lawsA = result.laws_a || result.results_a || [];
    const lawsB = result.laws_b || result.results_b || [];
    const autoSummary = result.auto_summary || '';

    let html = '';

    // 缓存状态 + 耗时
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">';
    html += '<div style="font-size:0.75rem;color:var(--color-text-muted);">检索耗时: ' + (searchTime || '?') + '</div>';
    if (isCacheHit) {
        html += '<div style="font-size:0.75rem;font-weight:bold;color:var(--color-primary);">CACHE HIT</div>';
    } else {
        html += '<div style="font-size:0.75rem;color:var(--color-secondary);">首次生成</div>';
    }
    html += '</div>';

    // 对比概览
    if (autoSummary) {
        html += '<h2 class="section-title">对比概览</h2>';
        html += '<div class="card" style="margin-bottom:1.5rem;">';
        html += '<div class="markdown-body">' + renderMarkdown(autoSummary) + '</div>';
        html += '</div>';
    }

    // 指标总览（水平条形图）
    html += renderCompareStats(stats, jurA, jurB);

    // AI 深度分析
    if (aiAnalysis) {
        html += '<h2 class="section-title">AI 深度对比分析报告</h2>';
        html += '<div class="card" style="margin-bottom:1.5rem;">';
        html += '<div style="font-size:0.75rem;color:var(--color-text-muted);margin-bottom:0.5rem;">由大语言模型基于两法域检索结果自动生成的专业对比分析</div>';
        html += '<div class="markdown-body" id="cmp-ai-analysis-content">' + renderMarkdown(aiAnalysis) + '</div>';
        html += '</div>';
    } else if (lawsA.length > 0 && lawsB.length > 0) {
        html += '<h2 class="section-title">AI 深度对比分析报告</h2>';
        html += '<div class="card" style="margin-bottom:1.5rem;">';
        html += '<div style="font-size:0.75rem;color:var(--color-text-muted);margin-bottom:0.5rem;">由大语言模型基于两法域检索结果自动生成的专业对比分析</div>';
        html += '<div id="cmp-ai-analysis-loading" class="loading"><div class="spinner"></div>AI 正在生成深度对比分析报告...</div>';
        html += '<div class="markdown-body" id="cmp-ai-analysis-content"></div>';
        html += '</div>';
    }

    // 条款对照表（改进版）
    html += '<h2 class="section-title">条款对照表（Top 5 差异条款）</h2>';
    html += renderClauseTable(lawsA, lawsB, jurA, jurB);

    // 关键词对比
    html += renderKeywordsComparison(stats, jurA, jurB);

    container.innerHTML = html;
}

function renderCompareStats(stats, jurA, jurB) {
    const articlesA = (stats.total_articles || {})[jurA] || 0;
    const articlesB = (stats.total_articles || {})[jurB] || 0;
    const countA = (stats.law_counts || {})[jurA] || 0;
    const countB = (stats.law_counts || {})[jurB] || 0;
    const bestA = ((stats.best_similarity || {})[jurA] || 0) * 100;
    const bestB = ((stats.best_similarity || {})[jurB] || 0) * 100;
    const avgA = ((stats.avg_similarity || {})[jurA] || 0) * 100;
    const avgB = ((stats.avg_similarity || {})[jurB] || 0) * 100;

    const metrics = [
        { label: '命中条款数', valA: articlesA, valB: articlesB, unit: '', max: Math.max(articlesA, articlesB, 1), deltaFn: function(a, b) { return a > b ? jurA + '领先' : (b > a ? jurB + '领先' : '持平'); } },
        { label: '涉及法规数', valA: countA, valB: countB, unit: '', max: Math.max(countA, countB, 1), deltaLabel: '共' + (countA + countB) + '部' },
        { label: '最高匹配度', valA: bestA, valB: bestB, unit: '%', max: 100, deltaFn: function(a, b) { var d = Math.abs(a - b); return d > 3 ? '差距' + Math.round(d) + '%' : (d > 1 ? '差距' + Math.round(d) + '%' : '接近'); } },
        { label: '平均匹配度', valA: avgA, valB: avgB, unit: '%', max: 100, deltaFn: function(a, b) { var d = Math.abs(a - b); return d > 3 ? '差距' + Math.round(d) + '%' : (d > 1 ? '差距' + Math.round(d) + '%' : '接近'); } }
    ];

    let html = '<h2 class="section-title">对比指标总览</h2>';
    html += '<div class="card" style="margin-bottom:1.5rem;">';

    metrics.forEach(function(m) {
        const pctA = m.max > 0 ? (m.valA / m.max) * 100 : 0;
        const pctB = m.max > 0 ? (m.valB / m.max) * 100 : 0;
        const displayA = m.unit === '%' ? m.valA.toFixed(0) : Math.round(m.valA);
        const displayB = m.unit === '%' ? m.valB.toFixed(0) : Math.round(m.valB);

        var deltaLabel = m.deltaLabel || '';
        if (!deltaLabel && m.deltaFn) {
            deltaLabel = m.deltaFn(m.valA, m.valB);
        }
        var deltaColor = '#E65100';
        if (deltaLabel === '接近' || deltaLabel === '持平') deltaColor = '#2E7D32';

        html += '<div style="margin-bottom:1rem;padding-bottom:1rem;border-bottom:1px solid var(--color-border);">';
        html += '<div style="display:flex;align-items:center;gap:0.75rem;">';
        html += '<div style="width:90px;flex-shrink:0;font-size:0.8125rem;font-weight:600;color:var(--color-text);">' + m.label + '</div>';
        html += '<div style="flex:1;min-width:0;">';

        // Bar A
        html += '<div style="display:flex;align-items:center;font-size:0.8125rem;margin-bottom:0.375rem;">';
        html += '<span style="width:60px;flex-shrink:0;text-align:right;margin-right:0.5rem;color:var(--color-text-secondary);font-size:0.75rem;">' + jurA + '</span>';
        html += '<div style="flex:1;background:var(--color-border);height:20px;border-radius:2px;overflow:hidden;">';
        html += '<div style="width:' + Math.min(pctA, 100) + '%;background:var(--color-primary);height:100%;transition:width 0.3s;"></div>';
        html += '</div>';
        html += '<span style="width:60px;flex-shrink:0;margin-left:0.5rem;font-weight:600;font-size:0.8125rem;">' + displayA + m.unit + '</span>';
        html += '</div>';

        // Bar B
        html += '<div style="display:flex;align-items:center;font-size:0.8125rem;">';
        html += '<span style="width:60px;flex-shrink:0;text-align:right;margin-right:0.5rem;color:var(--color-text-secondary);font-size:0.75rem;">' + jurB + '</span>';
        html += '<div style="flex:1;background:var(--color-border);height:20px;border-radius:2px;overflow:hidden;">';
        html += '<div style="width:' + Math.min(pctB, 100) + '%;background:var(--color-secondary);height:100%;transition:width 0.3s;"></div>';
        html += '</div>';
        html += '<span style="width:60px;flex-shrink:0;margin-left:0.5rem;font-weight:600;font-size:0.8125rem;">' + displayB + m.unit + '</span>';
        html += '</div>';

        html += '</div>'; // flex:1

        // Delta label
        if (deltaLabel) {
            html += '<div style="width:80px;flex-shrink:0;text-align:center;">';
            html += '<span class="badge" style="background:' + (deltaColor === '#2E7D32' ? 'var(--color-success-bg)' : 'var(--color-warning-bg)') + ';color:' + deltaColor + ';font-size:0.75rem;padding:4px 8px;">' + deltaLabel + '</span>';
            html += '</div>';
        }

        html += '</div>'; // flex row
        html += '</div>'; // margin-bottom
    });

    html += '</div>';
    return html;
}

// ===== 文本清洗函数 =====
function cleanTextForDisplay(text, maxLen) {
    maxLen = maxLen || 280;
    if (!text) return '';
    text = text.trim();

    var hasChinese = /[\u4e00-\u9fa5]/.test(text);

    var cleaned;
    if (hasChinese) {
        // 中文文本：只提取中文部分
        var lines = text.split('\n');
        var chineseLines = [];
        for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            if (!line) continue;
            var chineseChars = (line.match(/[\u4e00-\u9fa5]/g) || []).length;
            var totalPrintable = (line.match(/[\u4e00-\u9fa5a-zA-Z0-9]/g) || []).length;
            if (totalPrintable > 0 && chineseChars / totalPrintable > 0.3) {
                chineseLines.push(line);
            }
        }
        cleaned = chineseLines.length > 0 ? chineseLines.join(' ') : text;
        // 去掉开头的非中文字符
        cleaned = cleaned.replace(/^[^\u4e00-\u9fa5]{0,20}/, '').trim();
    } else {
        // 英文文本：保留完整首句
        var sentences = text.split(/(?<=[.!?])\s+/);
        cleaned = '';
        for (var j = 0; j < sentences.length; j++) {
            var s = sentences[j].trim();
            if (s.length > 30) {
                cleaned = s;
                break;
            }
        }
        if (!cleaned) cleaned = text.substring(0, maxLen);
        // 去掉开头的小写字母残留
        cleaned = cleaned.replace(/^[a-z]\s+/, '').trim();
    }

    // 截断
    if (cleaned.length > maxLen) {
        var cutPoint = cleaned.lastIndexOf(' ', maxLen);
        if (cutPoint > maxLen * 0.7) {
            cleaned = cleaned.substring(0, cutPoint) + ' ...';
        } else {
            cleaned = cleaned.substring(0, maxLen) + '...';
        }
    }

    return cleaned;
}

// ===== 关键概念高亮 =====
function highlightKeyConcepts(text) {
    if (!text) return '';
    var escaped = escapeHtml(text);

    // 中文概念高亮
    var zhConcepts = [
        { pattern: /(個人資料|數據保護|網絡安全|關鍵基礎設施|預警中心|監管實體|營運者)/g, color: '#C62828' },
        { pattern: /(未經授權|正常運作|完整性|保密性|可用性|電腦系統)/g, color: '#1565C0' },
        { pattern: /(處罰|罰款|監禁|刑事|違規|違反)/g, color: '#E65100' },
        { pattern: /(機構|委員會|專員|部門|當局)/g, color: '#6A1B9A' }
    ];

    zhConcepts.forEach(function(c) {
        escaped = escaped.replace(c.pattern, '<b style="color:' + c.color + '">$1</b>');
    });

    // 英文概念高亮
    var enConcepts = [
        { pattern: /\b(cybersecurity|unauthorised? access|critical information infrastructure)\b/gi, color: '#C62828' },
        { pattern: /\b(computer\s*system|CII|incident|licence|commissioner)\b/gi, color: '#1565C0' },
        { pattern: /\b(penalty|fine|imprisonment|offence|liability)\b/gi, color: '#E65100' }
    ];

    enConcepts.forEach(function(c) {
        escaped = escaped.replace(c.pattern, '<b style="color:' + c.color + '">$&</b>');
    });

    return escaped;
}

// ===== 判断是否为实质性条款 =====
function isSubstantiveChunk(content) {
    if (!content) return false;
    var text = content.trim();
    if (text.length < 50) return false;
    // 排除纯标题句型
    var titlePatterns = [
        /^This\s+Act\s+is\s+the\s+.+Act\s+\d{4}\s*\.$/i,
        /^An\s+Act\s+to\s+/i,
        /^本法[是为].{0,30}$/
    ];
    for (var i = 0; i < titlePatterns.length; i++) {
        if (titlePatterns[i].test(text)) return false;
    }
    return true;
}

function renderClauseTable(lawsA, lawsB, jurA, jurB) {
    const allChunksA = [];
    lawsA.forEach(function(law) {
        (law.all_chunks || []).forEach(function(c) {
            if (isSubstantiveChunk(c.content)) {
                allChunksA.push(Object.assign({}, c, { law_title: law.title, law_id: law.law_id }));
            }
        });
    });
    const allChunksB = [];
    lawsB.forEach(function(law) {
        (law.all_chunks || []).forEach(function(c) {
            if (isSubstantiveChunk(c.content)) {
                allChunksB.push(Object.assign({}, c, { law_title: law.title, law_id: law.law_id }));
            }
        });
    });

    if (allChunksA.length === 0 && allChunksB.length === 0) {
        return '<div class="alert alert-warning">暂无实质性条款内容</div>';
    }

    const pairedCount = Math.min(allChunksA.length, allChunksB.length);
    const diffPairs = [];
    for (let i = 0; i < pairedCount; i++) {
        const ca = allChunksA[i];
        const cb = allChunksB[i];
        const simA = ca.similarity || 0;
        const simB = cb.similarity || 0;
        const diffScore = Math.abs(simA - simB) + (1 - Math.min(simA, simB)) * 0.3;
        diffPairs.push({ ca: ca, cb: cb, diffScore: diffScore });
    }
    diffPairs.sort(function(a, b) { return b.diffScore - a.diffScore; });
    const topPairs = diffPairs.slice(0, 5);

    // 表头
    let html = '<div style="display:grid;grid-template-columns:1fr 1fr 80px;gap:1rem;margin-bottom:0.5rem;padding:0.5rem;background:var(--color-bg);border-radius:6px;">';
    html += '<div style="font-size:0.875rem;font-weight:600;">' + jurA + '（共 ' + allChunksA.length + ' 条）</div>';
    html += '<div style="font-size:0.875rem;font-weight:600;">' + jurB + '（共 ' + allChunksB.length + ' 条）</div>';
    html += '<div style="font-size:0.875rem;font-weight:600;text-align:center;">差异</div>';
    html += '</div>';

    topPairs.forEach(function(pair) {
        const ca = pair.ca;
        const cb = pair.cb;
        const simA = ca.similarity || 0;
        const simB = cb.similarity || 0;
        const diffVal = Math.abs(simA - simB);
        const artA = ca.article_number || '条款';
        const artB = cb.article_number || '条款';
        const pctA = Math.round(simA * 100);
        const pctB = Math.round(simB * 100);
        const displayA = highlightKeyConcepts(cleanTextForDisplay(ca.content || ''));
        const displayB = highlightKeyConcepts(cleanTextForDisplay(cb.content || ''));

        let diffLabel, diffColor;
        if (diffVal > 0.15) { diffLabel = '较大'; diffColor = '#d32f2f'; }
        else if (diffVal > 0.05) { diffLabel = '中等'; diffColor = '#ed6c02'; }
        else { diffLabel = '接近'; diffColor = '#2e7d32'; }

        html += '<div style="display:grid;grid-template-columns:1fr 1fr 80px;gap:1rem;margin-bottom:1rem;padding:0.75rem;border:1px solid var(--color-border);border-radius:6px;">';

        // Column A
        html += '<div style="min-width:0;">';
        html += '<div style="font-size:0.8125rem;margin-bottom:0.375rem;"><strong>' + escapeHtml(artA) + '</strong> · 《' + escapeHtml(ca.law_title || '') + '》 <span class="badge badge-muted">匹配 ' + pctA + '%</span></div>';
        html += '<div style="font-size:0.8125rem;line-height:1.6;word-break:break-word;">' + displayA + '</div>';
        html += '</div>';

        // Column B
        html += '<div style="min-width:0;">';
        html += '<div style="font-size:0.8125rem;margin-bottom:0.375rem;"><strong>' + escapeHtml(artB) + '</strong> · 《' + escapeHtml(cb.law_title || '') + '》 <span class="badge badge-muted">匹配 ' + pctB + '%</span></div>';
        html += '<div style="font-size:0.8125rem;line-height:1.6;word-break:break-word;">' + displayB + '</div>';
        html += '</div>';

        // Diff column
        html += '<div style="display:flex;align-items:center;justify-content:center;">';
        html += '<span class="badge" style="background:' + (diffColor === '#2e7d32' ? 'var(--color-success-bg)' : diffColor === '#ed6c02' ? 'var(--color-warning-bg)' : 'var(--color-error-bg)') + ';color:' + diffColor + ';font-size:0.75rem;padding:4px 8px;">' + diffLabel + '</span>';
        html += '</div>';

        html += '</div>';
    });

    return html;
}

// ===== 关键词对比 =====
function renderKeywordsComparison(stats, jurA, jurB) {
    const kwsA = (stats.top_keywords || {})[jurA] || [];
    const kwsB = (stats.top_keywords || {})[jurB] || [];

    if (kwsA.length === 0 && kwsB.length === 0) return '';

    const setA = new Set(kwsA);
    const setB = new Set(kwsB);
    const common = [];
    const onlyA = [];
    const onlyB = [];

    setA.forEach(function(kw) {
        if (setB.has(kw)) common.push(kw);
        else onlyA.push(kw);
    });
    setB.forEach(function(kw) {
        if (!setA.has(kw)) onlyB.push(kw);
    });

    let html = '<h2 class="section-title">关键词对比</h2>';
    html += '<div class="grid-2">';

    // JurA keywords
    html += '<div class="card">';
    html += '<div class="card-title">' + escapeHtml(jurA) + ' (' + kwsA.length + '个)</div>';
    if (kwsA.length > 0) {
        html += '<div style="display:flex;flex-wrap:wrap;gap:4px;">';
        kwsA.forEach(function(kw) {
            html += '<span style="background:var(--color-info-bg);color:var(--color-info);padding:3px 10px;font-size:0.8125rem;">' + escapeHtml(kw) + '</span>';
        });
        html += '</div>';
    } else {
        html += '<div style="font-size:0.75rem;color:var(--color-text-muted);">未检测到关键词</div>';
    }
    html += '</div>';

    // JurB keywords
    html += '<div class="card">';
    html += '<div class="card-title">' + escapeHtml(jurB) + ' (' + kwsB.length + '个)</div>';
    if (kwsB.length > 0) {
        html += '<div style="display:flex;flex-wrap:wrap;gap:4px;">';
        kwsB.forEach(function(kw) {
            html += '<span style="background:var(--color-warning-bg);color:var(--color-warning);padding:3px 10px;font-size:0.8125rem;">' + escapeHtml(kw) + '</span>';
        });
        html += '</div>';
    } else {
        html += '<div style="font-size:0.75rem;color:var(--color-text-muted);">未检测到关键词</div>';
    }
    html += '</div>';

    html += '</div>';

    // 共同/独有分析
    if (common.length > 0 || onlyA.length > 0 || onlyB.length > 0) {
        const parts = [];
        if (common.length > 0) parts.push('<strong>共同</strong>: ' + common.join(', '));
        if (onlyA.length > 0) parts.push('<strong>仅' + jurA + '</strong>: ' + onlyA.slice(0, 5).join(', '));
        if (onlyB.length > 0) parts.push('<strong>仅' + jurB + '</strong>: ' + onlyB.slice(0, 5).join(', '));
        html += '<div style="font-size:0.8125rem;color:var(--color-text-secondary);margin-top:0.75rem;">' + parts.join(' | ') + '</div>';
    }

    return html;
}
