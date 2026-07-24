/**
 * 法规详情页面
 * 路由: #/law-detail/{law_id}
 */
registerPage('law-detail', function(container) {
    // 从 hash 中提取 law_id
    const hash = window.location.hash;
    const lawId = hash.replace('#/law-detail/', '');

    if (!lawId) {
        container.innerHTML = '<div class="alert alert-warning">未选择法规，请先从检索页面选择</div>';
        container.innerHTML += '<button class="btn btn-secondary" onclick="goBackToSearch()">返回检索</button>';
        return;
    }

    container.innerHTML = `
        <button class="btn btn-ghost btn-sm" onclick="goBackToSearch()" style="margin-bottom:1rem;">← 返回检索</button>
        <div id="law-detail-content"></div>
    `;

    const contentEl = document.getElementById('law-detail-content');
    showLoading(contentEl);

    // 并行获取法规详情和相似法规
    Promise.all([
        api.get('/search/law/' + lawId),
        api.get('/search/law/' + lawId + '/similar', { top_k: 5 })
    ]).then(function(results) {
        const lawData = results[0];
        const similarData = results[1];

        if (!lawData || !lawData.success) {
            showError(contentEl, '获取法规详情失败：' + (lawData ? lawData.message : '未知错误'));
            return;
        }

        renderLawDetail(contentEl, lawData.data, similarData && similarData.data ? similarData.data : null);
    }).catch(function(err) {
        showError(contentEl, '请求失败：' + err.message);
    });
});

/**
 * 清理法规正文的文本排版
 * - 规范化换行符
 * - 段落内的单 \n 合并为空格
 * - 连续的 \n\n 作为段落分隔
 */
function cleanLawText(text) {
    if (!text) return '';
    // 1. 规范化换行
    var t = text.replace(/\r\n/g, '\n');
    // 2. 行首行尾空白修剪
    t = t.replace(/^[ \t]+/gm, '').replace(/[ \t]+$/gm, '');
    // 3. 将连续的空白行（\n\n+）替换为段落分隔标记
    t = t.replace(/\n{2,}/g, '\n¶\n');
    // 4. 将段落内剩余的单 \n 替换为空格（修复文本提取造成的断行）
    t = t.replace(/\n(?!¶)/g, ' ');
    // 5. 恢复段落标记为 \n
    t = t.replace(/¶/g, '');
    // 6. 合并多余空格
    t = t.replace(/[ \t]{2,}/g, ' ');
    return t.trim();
}

/**
 * 检测文本开头的章节标题（如 "1.2 Declaratory rulings"、"Section 5" 等）
 * 返回 { title, rest } 或 null
 */
function extractSectionHeading(text) {
    // 匹配数字+小数的标题模式：1.2 xxx, 3.1.4 xxx
    var m = text.match(/^(\d+(?:\.\d+)+)\s+(.+?)(?=[\.\n]|$)/);
    if (m) {
        var title = m[1] + ' ' + m[2].replace(/\s+/g, ' ').trim();
        var rest = text.substring(m[0].length).trim();
        return { title: title, rest: rest };
    }
    // 匹配 "Section X.Y" 或 "Sec. X" 模式
    m = text.match(/^(Section|Sec\.|Article|Art\.|Part|Chapter|Title|Rule)\s+(\d+(?:\.\d+)?)\b(.+?)?(?=[\n]|$)/i);
    if (m) {
        var title = m[0].trim();
        var rest = text.substring(m[0].length).trim();
        return { title: title, rest: rest };
    }
    // 匹配全大写短标题（如 "PRELIMINARY"、"DEFINITIONS"）
    m = text.match(/^([A-Z][A-Z\s]{2,30}?)(?=[\n\s]|$)/);
    if (m && m[1].trim().length > 2) {
        title = m[1].trim();
        rest = text.substring(m[0].length).trim();
        return { title: title, rest: rest };
    }
    return null;
}

/**
 * 将法规正文格式化为 HTML（段落 + 标题）
 */
function formatLawContent(content) {
    if (!content) return '<p style="color:var(--color-text-muted);font-size:0.875rem;">（空）</p>';

    var text = cleanLawText(content);
    if (!text) return '<p style="color:var(--color-text-muted);font-size:0.875rem;">（空）</p>';

    // 按段落分割
    var paragraphs = text.split('\n').filter(function(p) { return p.trim(); });
    if (paragraphs.length === 0) return '<p style="color:var(--color-text-muted);font-size:0.875rem;">（空）</p>';

    var html = '';
    var firstProcessed = false;

    paragraphs.forEach(function(para, idx) {
        para = para.trim();
        if (!para) return;

        // 第一段检测章节标题
        if (idx === 0) {
            var heading = extractSectionHeading(para);
            if (heading) {
                html += '<h4 style="font-size:0.9rem;font-weight:600;margin:0.75rem 0 0.375rem;color:var(--color-text);">' +
                    escapeHtml(heading.title) + '</h4>';
                if (heading.rest) {
                    html += '<p style="margin:0.375rem 0;line-height:1.75;font-size:0.875rem;color:var(--color-text-secondary);text-align:justify;">' +
                        escapeHtml(heading.rest) + '</p>';
                }
                firstProcessed = true;
                return;
            }
        }

        // 再次检测是否为标题
        var heading2 = extractSectionHeading(para);
        if (heading2) {
            html += '<h4 style="font-size:0.9rem;font-weight:600;margin:0.75rem 0 0.375rem;color:var(--color-text);">' +
                escapeHtml(heading2.title) + '</h4>';
            if (heading2.rest) {
                html += '<p style="margin:0.375rem 0;line-height:1.75;font-size:0.875rem;color:var(--color-text-secondary);text-align:justify;">' +
                    escapeHtml(heading2.rest) + '</p>';
            }
            return;
        }

        html += '<p style="margin:0.375rem 0;line-height:1.75;font-size:0.875rem;color:var(--color-text-secondary);text-align:justify;">' +
            escapeHtml(para) + '</p>';
    });

    return html;
}

function renderLawDetail(container, lawData, similarData) {
    const title = lawData.title || '未知法规';
    const jurisdiction = lawData.jurisdiction || '';
    const passingDate = lawData.passing_date || '';
    const lawType = lawData.law_type || '';
    const chunks = lawData.chunks || [];

    let html = '';

    // 标题
    html += '<h1 class="page-title">' + escapeHtml(title) + '</h1>';

    // 元信息
    const metaParts = [];
    if (jurisdiction) metaParts.push('法域: ' + jurisdiction);
    if (lawType) metaParts.push('类型: ' + lawType);
    if (passingDate) metaParts.push('通过日期: ' + passingDate);
    if (metaParts.length > 0) {
        html += '<p style="font-size:0.875rem;color:var(--color-text-secondary);margin-bottom:1rem;">' + metaParts.join(' | ') + '</p>';
    }

    html += '<hr class="divider">';

    // 法规全文：按 article_number 聚合 sub_chunk，按 chunk_index 排序拼接
    const articleMap = {};
    chunks.forEach(function(chunk) {
        const key = chunk.article_number || '条款';
        if (!articleMap[key]) {
            articleMap[key] = [];
        }
        articleMap[key].push(chunk);
    });

    const articles = Object.keys(articleMap).map(function(articleNum) {
        const parts = articleMap[articleNum].slice().sort(function(a, b) {
            return (a.chunk_index || 0) - (b.chunk_index || 0);
        });
        // 取第一个 chunk 的 article_title 作为条款名
        const articleTitle = parts[0] ? (parts[0].article_title || '') : '';
        // 拼接内容后清理文本
        const rawContent = parts.map(function(p) { return p.content || ''; }).join('\n');
        const content = rawContent;
        const firstIndex = parts[0] ? (parts[0].chunk_index || 0) : 0;
        return { articleNum: articleNum, articleTitle: articleTitle, content: content, firstIndex: firstIndex };
    }).sort(function(a, b) { return a.firstIndex - b.firstIndex; });

    html += '<h2 class="section-title">法规全文</h2>';
    html += '<p style="font-size:0.8125rem;color:var(--color-text-muted);margin-bottom:1rem;">共 ' + articles.length + ' 条条款</p>';

    if (articles.length === 0) {
        html += '<div class="alert alert-info">该法规暂无内容</div>';
    } else {
        articles.forEach(function(article) {
            html += '<div style="margin-bottom:1rem;padding-bottom:0.75rem;border-bottom:1px solid var(--color-border);">';
            // 条款编号 + 标题
            let headingText = '第 ' + article.articleNum + ' 条';
            if (article.articleTitle) {
                headingText += ' ' + escapeHtml(article.articleTitle);
            }
            html += '<div style="font-weight:600;margin-bottom:0.5rem;font-family:var(--font-serif);font-size:0.9375rem;color:var(--color-text);">' + headingText + '</div>';
            // 格式化后的正文
            html += formatLawContent(article.content);
            html += '</div>';
        });
    }

    html += '<hr class="divider">';

    // 相似法规推荐
    html += '<h2 class="section-title">相关法规推荐</h2>';

    if (similarData && similarData.similar_laws && similarData.similar_laws.length > 0) {
        const queryTitle = similarData.query_law_title || title;
        const total = similarData.total_found || 0;
        html += '<p style="font-size:0.8125rem;color:var(--color-text-muted);margin-bottom:1rem;">基于《' + escapeHtml(queryTitle) + '》的语义分析，从 ' + total + ' 部相关法规中推荐以下结果：</p>';

        similarData.similar_laws.forEach(function(item) {
            const simLawId = item.law_id || '';
            const simTitle = item.title || '未知法规';
            const simJurisdiction = item.jurisdiction || '';
            const similarity = item.similarity || 0;
            const matchCount = item.match_chunk_count || 0;
            const simPercent = Math.round(similarity * 100);

            let barColor = '#d32f2f';
            if (simPercent >= 75) barColor = '#2e7d32';
            else if (simPercent >= 55) barColor = '#ed6c02';

            html += '<div style="margin-bottom:1rem;">';
            html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.375rem;">';
            html += '<span style="font-size:0.75rem;color:var(--color-text-muted);min-width:42px;">相似度</span>';
            html += '<div style="flex:1;background:var(--color-border);height:8px;border-radius:4px;">';
            html += '<div style="width:' + simPercent + '%;background:' + barColor + ';height:100%;border-radius:4px;"></div>';
            html += '</div>';
            html += '<span style="font-size:0.75rem;font-weight:600;color:' + barColor + ';min-width:38px;text-align:right;">' + simPercent + '%</span>';
            html += '</div>';

            html += '<div style="background:var(--color-bg);border:1px solid var(--color-border);padding:0.75rem 1rem;border-radius:6px;">';
            html += '<div style="display:flex;justify-content:space-between;align-items:flex-start;">';
            html += '<div style="flex:1;">';
            html += '<div style="font-weight:600;font-size:0.875rem;color:var(--color-text);margin-bottom:0.25rem;">' + escapeHtml(simTitle) + '</div>';
            html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:0.25rem;">';
            html += '<span class="badge badge-info">' + escapeHtml(simJurisdiction) + '</span>';
            html += '<span class="badge badge-warning">' + matchCount + '个匹配片段</span>';
            html += '<span class="badge badge-muted">ID: ' + escapeHtml(simLawId) + '</span>';
            html += '</div>';
            html += '</div>';
            html += '<button class="btn btn-secondary btn-sm" onclick="navigate(\'law-detail/' + simLawId + '\')">查看</button>';
            html += '</div>';
            html += '</div>';
            html += '</div>';
        });
    } else {
        html += '<div class="alert alert-info">暂未找到相似法规</div>';
    }

    container.innerHTML = html;
}

// 返回检索页，保留之前搜索状态
function goBackToSearch() {
    navigate('search');
}