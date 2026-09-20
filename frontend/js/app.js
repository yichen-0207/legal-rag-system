/**
 * 境外法规检索平台 - 共享工具函数
 */

// Docker 环境：使用相对路径，通过 Nginx 反向代理到后端
// 本地开发调试时改为 'http://127.0.0.1:8001/api/v1'
const API_BASE = '/api/v1';

// ===== API 请求封装 =====
const api = {
    async get(path, params = {}) {
        const url = new URL(API_BASE + path, window.location.origin);
        Object.entries(params).forEach(([k, v]) => {
            if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
        });
        const resp = await fetch(url.toString());
        if (!resp.ok) throw new Error(`API错误: ${resp.status}`);
        return resp.json();
    },

    async post(path, body) {
        const resp = await fetch(API_BASE + path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!resp.ok) throw new Error(`API错误: ${resp.status}`);
        return resp.json();
    },

    async streamPost(path, body, onChunk, onDoneOrError, onError) {
        const onDone = onError ? onDoneOrError : null;
        const onErr = onError || onDoneOrError;

        try {
            const resp = await fetch(API_BASE + path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (!resp.ok) throw new Error(`API错误: ${resp.status}`);

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(trimmed.slice(6));
                            onChunk(data);
                        } catch (e) { /* skip invalid JSON */ }
                    }
                }
            }

            if (onDone) onDone();
        } catch (e) {
            if (onErr) onErr(e.message);
        }
    }
};

// ===== 导航 =====
function navigate(page) {
    window.location.hash = '#/' + page;
}

function getCurrentPage() {
    const hash = window.location.hash.slice(2) || 'dashboard';
    // 提取页面名称（去掉参数部分，如 law-detail/MO_xxx -> law-detail）
    const slashIndex = hash.indexOf('/');
    return slashIndex > 0 ? hash.substring(0, slashIndex) : hash;
}

// ===== 侧边栏激活状态 =====
function updateSidebarActive(page) {
    // law-detail 是 search 的子页面，高亮 search
    const activePage = page === 'law-detail' ? 'search' : page;
    document.querySelectorAll('.sidebar-nav a').forEach(a => {
        a.classList.toggle('active', a.dataset.page === activePage);
    });
}

// ===== 工具函数 =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateStr) {
    if (!dateStr) return '-';
    const d = new Date(dateStr);
    return d.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' });
}

function showLoading(container) {
    container.innerHTML = '<div class="loading"><div class="spinner"></div>加载中...</div>';
}

function showError(container, msg) {
    container.innerHTML = `<div class="alert alert-error">${escapeHtml(msg)}</div>`;
}

// ===== 简单的 Markdown 渲染 =====
function renderMarkdown(text) {
    if (!text) return '';

    // 1. 提前提取 Markdown 表格，避免被全局转义/换行处理破坏
    var tablePlaceholders = [];
    var placeholderPrefix = '##MARKDOWN_TABLE_';
    text = text.replace(/(^\s*\|[^\n]*\|\n?)+/gm, function(match) {
        var lines = match.trim().split('\n');
        var hasSeparator = lines.some(function(l) {
            return /^\s*\|(?:[-:|\s]+\|)+$/.test(l);
        });
        if (hasSeparator && lines.length >= 2) {
            var idx = tablePlaceholders.length;
            tablePlaceholders.push(renderMarkdownTable(lines));
            return placeholderPrefix + idx + '##';
        }
        return match;
    });

    // 2. 对其余文本进行 Markdown 渲染
    var html = escapeHtml(text);

    // 代码块
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // 行内代码
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // 粗体
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // 斜体
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // 标题
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    // 无序列表
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    // 引用
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
    // 换行
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // 3. 恢复表格占位符
    tablePlaceholders.forEach(function(tableHtml, idx) {
        html = html.replace(placeholderPrefix + idx + '##', tableHtml);
    });

    return '<p>' + html + '</p>';
}

// 将 Markdown 表格行数组渲染为 HTML 表格
function renderMarkdownTable(lines) {
    // 过滤分隔行，保留表头和数据行
    var rows = lines.filter(function(l) {
        return !/^\s*\|(?:[-:|\s]+\|)+$/.test(l);
    });
    if (rows.length === 0) return '';

    // 按首行确定期望列数
    var firstCells = parseMarkdownTableRow(rows[0]);
    var expectedCols = firstCells.length;
    if (expectedCols === 0) return '';

    var html = '<table class="markdown-table" style="border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.875rem;">';
    rows.forEach(function(row, idx) {
        var cells = parseMarkdownTableRow(row, expectedCols);
        var tag = idx === 0 ? 'th' : 'td';
        var cellStyle = 'border:1px solid var(--color-border);padding:8px 12px;text-align:left;vertical-align:top;';
        if (idx === 0) {
            cellStyle += 'background:var(--color-bg);font-weight:600;';
        }
        html += '<tr>';
        cells.forEach(function(cell) {
            html += '<' + tag + ' style="' + cellStyle + '">' + escapeHtml(cell.trim()) + '</' + tag + '>';
        });
        html += '</tr>';
    });
    html += '</table>';
    return html;
}

// 解析单行 Markdown 表格，按期望列数合并多余单元格（处理单元格内容含 | 的情况）
function parseMarkdownTableRow(row, expectedCols) {
    // 去掉首尾 '|'，按 '|' 分割
    var raw = row.replace(/^\s*\|/, '').replace(/\|\s*$/, '');
    // 保护转义的 \|
    var escapedPipePlaceholder = '\x00PIPE\x00';
    raw = raw.replace(/\\\|/g, escapedPipePlaceholder);
    var cells = raw.split('|').map(function(c) { return c.replace(new RegExp(escapedPipePlaceholder, 'g'), '|'); });

    if (!expectedCols || cells.length <= expectedCols) {
        return cells;
    }
    // 单元格数超过期望列数：将多余部分合并到最后一列
    var merged = cells.slice(0, expectedCols - 1);
    merged.push(cells.slice(expectedCols - 1).join('|'));
    return merged;
}

// ===== 页面路由注册 =====
const pages = {};

function registerPage(name, renderFn) {
    pages[name] = renderFn;
}

function renderCurrentPage() {
    const page = getCurrentPage();
    updateSidebarActive(page);
    const container = document.getElementById('page-content');
    if (!container) return;

    if (pages[page]) {
        pages[page](container);
    } else {
        container.innerHTML = '<div class="alert alert-warning">页面未找到</div>';
    }
}

// 监听路由变化
window.addEventListener('hashchange', renderCurrentPage);
window.addEventListener('DOMContentLoaded', renderCurrentPage);
