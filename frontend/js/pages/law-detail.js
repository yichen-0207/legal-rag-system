// 法规详情页，路由 #/law-detail/{law_id}；数据来源 GET /api/v1/search/law/{law_id}
// DOM 结构（与 style.css 的 .law-detail 设计块对齐）：.law-header / .law-toolbar / .law-body 双栏

var LD_PAGE_SIZE = 2000;      // 后端上限，最大法规（4202 块）也仅需 3 页
var LD_FS_KEY = 'ld_font_size';

// 法条分项标记：1. / 1) / (1) / （一） / 一、 / (a) / (i) / ① 等
var LAW_ITEM_RE = /^(\([0-9]{1,3}\)|（[0-9]{1,3}）|\([a-zA-Z]{1,3}\)|（[a-zA-Z]{1,3}）|\([ivxlcdmIVXLCDM]{1,6}\)|（[ivxlcdmIVXLCDM]{1,6}）|（[一二三四五六七八九十百]{1,3}）|[一二三四五六七八九十百]{1,3}、|[0-9]{1,3}[.、)](?![0-9])|[a-zA-Z][.)](?=[\s\u4e00-\u9fff])|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*/;

/* ===== 数据质量兜底 =====
 * raw_v2 的 metadata.article_title 大量为空/碎片且正文混入 PDF 页眉；仅在展示层保守兜底，不改索引数据。
 */
var LD_NOISE_RE = /^(第[IVXLC]+ ?部\s*Part\s*[IVXLC]+|[-—–]?\s*\d{1,3}\s*[-—–]?$|\d{1,4}\s*\/\s*\d{1,4}$|第\s*\d{1,4}\s*[頁页]$)/;
var LD_TITLE_FRAG_RE = /^[（(\[]|^(Amended|Repealed|Added|Substituted|Inserted|Deleted)\b|^號|^附表|^Sch|s\.\s*\d+\)?\s*$/i;
var LD_NOTE_PAREN_RE = /[\s]*[（(]\s*(?:(?:由|经|經|依據|依据|按照|編輯修訂|編輯|修正)[^)）]{0,60}|(?:第)?\s*\d{1,4}\s*[A-Z]{0,3}\s*[條条][^)）]{0,40})[)）]/g;
// 未闭合注记（PDF 换行导致右括号丢失）：「( 第118C 條」「( 編輯修訂——2019 年第2 號」
var LD_NOTE_OPEN_RE = /[\s]*[（(]\s*(?:(?:由|经|經|依據|依据|按照|編輯修訂|編輯|修正)[^)）]{0,60}|(?:第)?\s*\d{1,4}\s*[A-Z]{0,3}\s*[條条][^)）]{0,40})$/;
var LD_NOTE_TAIL_RE = /[\s]*\d{1,4}\s*[A-Z]{0,3}\s*[條条]\s*[)）]/g;

/** 剥离括号注记：「( 第119 條)」「( 由2022 年第16 號第20 條增補)」「( 編輯修訂——2019 年第2 號」 */
function ldStripNotes(s) {
    return String(s).replace(LD_NOTE_PAREN_RE, ' ').replace(LD_NOTE_TAIL_RE, ' ').replace(LD_NOTE_OPEN_RE, '');
}

function ldTrimEdges(s) {
    return s.replace(/\s+/g, ' ').trim().replace(/^[、,，.。;；:：\-—]+|[、,，.。;；:：\-—]+$/g, '').trim();
}

/** 按顿逗或空格边界截断，避免截出半个词 */
function ldCutAtWord(s, max) {
    if (s.length <= max) return s;
    var seg = s.slice(0, max);
    var idx = Math.max(seg.lastIndexOf('，'), seg.lastIndexOf('、'), seg.lastIndexOf(','), seg.lastIndexOf(' '));
    if (idx >= max * 0.5) seg = seg.slice(0, idx);
    return seg.replace(/[\s，,、。；;：:\-—]+$/, '') + '…';
}

function ldCleanTitle(raw) {
    if (!raw) return '';
    var t = ldTrimEdges(ldStripNotes(raw));
    if (!t || t.length < 2 || LD_TITLE_FRAG_RE.test(t)) return '';
    return t.length > 60 ? ldCutAtWord(t, 60) : t;
}

/**
 * 从正文首行派生标题：仅在具备高置信结构特征（Section/Article/第N条/编号标题或「( 第N 條)」注记）时采用；
 * 小写拉丁字母、括号开头、部级标题、分项正文一律放弃。
 */
function ldDeriveTitle(firstLine) {
    if (!firstLine) return '';
    var noted = /[（(]\s*(?:由\s*\d{4}\s*年|(?:第)?\s*\d{1,4}\s*[A-Z]{0,3}\s*[條条])/.test(firstLine);
    var t = ldTrimEdges(ldStripNotes(firstLine));
    if (!t || t.length < 4) return '';
    if (/^[a-z]/.test(t) || LD_TITLE_FRAG_RE.test(t)) return '';
    if (/^第[IVXLC]+ ?部/.test(t)) return '';
    if (/^(一|二|三|四|五|六|七|八|九|十)、/.test(t)) return '';

    var numPre = t.match(/^\d{1,4}[A-Z]{0,3}[.、]?\s+/);
    if (numPre) {
        var rest = t.slice(numPre[0].length);
        if (/^[（(\[]/.test(rest) || /^由\s*\d{4}/.test(rest)) return '';
    }

    var structural = /^(Section|Article|Sec\.|Art\.|Rule|Regulation|Chapter)\s*\d/i.test(t) ||
        /^第[一二三四五六七八九十百千]+[条條]/.test(t) || !!numPre;
    if (!structural && !noted) return '';
    if (/[。；;]$/.test(t) && !numPre) return '';

    return ldCutAtWord(t, 30);
}

function firstContentLine(content) {
    var lines = String(content || '').replace(/\r\n?/g, '\n').split('\n');
    for (var i = 0; i < lines.length; i++) {
        var t = lines[i].trim();
        if (t && !LD_NOISE_RE.test(t)) return t;
    }
    return '';
}

function metaTitle(group) {
    return ldCleanTitle(group.title);
}

/** 目录标题：元数据优先，其次正文首行派生 */
function resolveTitle(group) {
    if (group.__title !== undefined) return group.__title;
    var t = metaTitle(group);
    for (var i = 0; !t && i < group.chunks.length; i++) {
        t = ldDeriveTitle(firstContentLine(group.chunks[i].content));
    }
    group.__title = t;
    return t;
}

var ldState = null;

// 离开页面时移除滚动监听，避免跨页面残留
window.addEventListener('hashchange', detachScrollSpy);

registerPage('law-detail', function(container) {
    var lawId = decodeURIComponent(window.location.hash.replace('#/law-detail/', '').split('?')[0]);

    detachScrollSpy();

    if (!lawId) {
        container.innerHTML = '<div class="alert alert-warning">未选择法规，请先从检索页面选择</div>' +
            '<button class="btn btn-secondary" onclick="goBackToSearch()">返回检索</button>';
        return;
    }

    var storedFs = 'm';
    try { storedFs = localStorage.getItem(LD_FS_KEY) || 'm'; } catch (e) { /* 隐私模式忽略 */ }

    ldState = {
        lawId: lawId,
        page: 0,
        totalChunks: 0,
        totalArticles: 0,
        hasMore: false,
        fs: storedFs,
        chunks: [],
        // 分区分增量渲染状态：正文 / 附表 / 解析遗留片段各自独立维护
        rendered: { substantive: [], schedule: [], fragment: [] },
        numToIdx: {},       // article_number -> 目录序号
        scrollHandler: null,
        scrollRaf: 0
    };

    container.innerHTML =
        '<div class="law-detail">' +
            '<div class="law-detail__crumb">' +
                '<button class="btn btn-ghost btn-sm" onclick="goBackToSearch()">← 返回检索</button>' +
            '</div>' +
            '<div id="law-detail-content"><div class="loading"><div class="spinner"></div>加载中...</div></div>' +
        '</div>';

    loadLawPage(1).then(function(data) {
        if (!data) return;
        renderLawDetail(data);
        loadSimilar();   // 相似推荐异步二级加载，不阻塞全文展示
    });
});

/* ===== 数据加载 ===== */

function loadLawPage(page) {
    var params = { page: page, page_size: LD_PAGE_SIZE };
    if (page === 1) params.outline = 'true';

    return api.get('/search/law/' + ldState.lawId, params).then(function(resp) {
        if (!resp || !resp.success) {
            showError(document.getElementById('law-detail-content'),
                '获取法规详情失败：' + ((resp && resp.message) || '未知错误'));
            return null;
        }
        return resp.data;
    }).catch(function(err) {
        showError(document.getElementById('law-detail-content'), '请求失败：' + err.message);
        return null;
    });
}

function loadSimilar() {
    api.get('/search/law/' + ldState.lawId + '/similar', { top_k: 5 }).then(function(resp) {
        renderSimilarSection(resp && resp.data ? resp.data : null);
    }).catch(function() {
        // 相似推荐为增强功能，失败时静默，不干扰正文阅读
        renderSimilarSection(null);
    });
}

/* ===== 页面渲染 ===== */

function renderLawDetail(data) {
    // 条款数只计正文条款，排除附表与解析遗留片段
    ldState.totalArticles = data.outline ? countArticles(data.outline) : (data.total_articles || 0);

    var name = data.law_name_zh || data.title || '未知法规';

    var html = '';
    html += '<header class="law-header">';
    html += '<h1 class="law-header__title">' + escapeHtml(name) + '</h1>';
    html += '<div class="law-meta">' + metaHtml(data) + '</div>';
    html += '<div class="law-stats" id="law-stats">' + statsHtml() + '</div>';
    html += '</header>';

    html += '<div class="law-toolbar">' + toolbarHtml() + '</div>';

    html += '<div class="law-body">';
    html += '<nav class="law-toc" id="law-toc">' + tocHtml(data) + '</nav>';
    html += '<div class="law-content" id="law-content" data-fs="' + ldState.fs + '">';
    html += '<div id="law-article-list"></div>';
    // 附表：源文档中位于正文之后，与法条序列分区展示
    html += '<section class="law-extra" id="law-sched-section" hidden>' +
        '<h2 class="law-extra__head">附表</h2>' +
        '<div id="law-schedule-list"></div>' +
    '</section>';
    // 解析遗留片段：不进目录，仅如实保留内容，避免静默丢失
    html += '<section class="law-extra law-extra--frag" id="law-frag-section" hidden>' +
        '<h2 class="law-extra__head">未归类片段</h2>' +
        '<p class="law-extra__hint">以下内容为源文档解析遗留片段（附表表格碎片／页眉），未纳入条款目录，仅供参考。</p>' +
        '<div id="law-fragment-list"></div>' +
    '</section>';
    html += '<div class="law-more" id="law-more"></div>';
    html += '</div>';
    html += '</div>';

    html += '<section class="law-similar" id="law-similar" hidden></section>';

    document.getElementById('law-detail-content').innerHTML = html;

    applyPageData(data);
    attachScrollSpy();
}

function metaHtml(data) {
    var parts = [];

    if (data.jurisdiction) {
        parts.push('<span class="badge badge-info">' + escapeHtml(data.jurisdiction) + '</span>');
    }

    var statusMap = { active: '现行有效', repealed: '已废止', amended: '已修订' };
    if (data.status) {
        var cls = data.status === 'active' ? 'badge-success' : 'badge-warning';
        parts.push('<span class="badge ' + cls + '">' + escapeHtml(statusMap[data.status] || data.status) + '</span>');
    }

    if (data.passing_date) {
        parts.push('<span class="law-meta__date"><b>通过</b>' + escapeHtml(data.passing_date) + '</span>');
    }
    if (data.effective_date) {
        parts.push('<span class="law-meta__date"><b>生效</b>' + escapeHtml(data.effective_date) + '</span>');
    }
    if (data.publication_date) {
        parts.push('<span class="law-meta__date"><b>公布</b>' + escapeHtml(data.publication_date) + '</span>');
    }

    (data.topics || []).forEach(function(t) {
        parts.push('<span class="badge badge-muted">' + escapeHtml(t) + '</span>');
    });

    return parts.join('');
}

function statsHtml() {
    var chunkCount = ldState.totalChunks || ldState.chunks.length;
    var html = '<span><b>' + articleCount() + '</b>条款</span>';
    html += '<span><b>' + chunkCount + '</b>索引块</span>';

    // 字数仅在全部索引块加载完成后展示，避免展示不完整的统计值
    if (!ldState.hasMore && ldState.chunks.length > 0) {
        var chars = 0;
        ldState.chunks.forEach(function(c) { chars += (c.content || '').length; });
        html += '<span><b>' + chars.toLocaleString() + '</b>字数</span>';
    }
    return html;
}

/** 章节归类：substantive 正文条款 / schedule 附表 / fragment 解析遗留片段 */
function streamOf(num, chunkType) {
    var t = String(chunkType || '').toLowerCase();
    if (t === 'schedule' || t === 'fragment') return t;
    // 历史数据无 chunk_type 时，按编号前缀兜底（Sch1 / 附表1）
    if (/^(Sch|附表)/i.test(String(num || ''))) return 'schedule';
    return 'substantive';
}

function displayNum(num) {
    var m = /^Sch\s*([0-9IVXLC]+)$/i.exec(String(num || ''));
    return m ? ('附表' + m[1].toUpperCase()) : (num || '');
}

/** 目录与计数只统计正文条款 */
function countArticles(outline) {
    var n = 0;
    outline.forEach(function(it) {
        if (streamOf(it.article_number, it.chunk_type) === 'substantive') n++;
    });
    return n;
}

function articleCount() {
    if (ldState.totalArticles) return ldState.totalArticles;
    return groupArticles(ldState.chunks).filter(function(g) {
        return streamOf(g.num, g.chunkType) === 'substantive';
    }).length;
}

function toolbarHtml() {
    var html = '<span class="law-toolbar__label">字号</span>';
    [['s', 'A−'], ['m', 'A'], ['l', 'A+']].forEach(function(pair) {
        var active = ldState.fs === pair[0] ? ' is-active' : '';
        html += '<button type="button" class="law-toolbar__btn' + active + '" data-fs="' + pair[0] + '" ' +
            'onclick="ldSetFs(\'' + pair[0] + '\')">' + pair[1] + '</button>';
    });
    html += '<span class="law-toolbar__sep"></span>';
    html += '<button type="button" class="law-toolbar__btn law-toolbar__right" onclick="ldPrint()">打印</button>';
    return html;
}

function tocHtml(data) {
    var outline = data.outline || [];
    if (!outline.length) {
        // 无目录时退化为提示，正文仍可正常阅读
        return '<div class="law-toc__head">条款目录</div><ul class="law-toc__list">' +
            '<li class="law-toc__item"><a href="javascript:void(0)">目录不可用</a></li></ul>';
    }

    var html = '<div class="law-toc__head">条款目录</div><ul class="law-toc__list">';
    var schedHtml = '';

    outline.forEach(function(item, idx) {
        ldState.numToIdx[item.article_number] = idx;

        // 解析遗留片段不进入目录；附表单独成组
        var stream = streamOf(item.article_number, item.chunk_type);
        if (stream === 'fragment') return;

        var t = ldCleanTitle(item.article_title);
        // 元数据标题不可信时先占位，待正文加载后由 backfillTocTitles 补全
        var full = t ? escapeHtml(truncateTitle(t, 60)).replace(/"/g, '&quot;') : '';
        var li = '<li class="law-toc__item" data-target="a' + idx + '"' + (t ? '' : ' data-pending="1"') + '>' +
            '<a href="javascript:void(0)" onclick="ldGoArticle(' + idx + ')">' +
            '<span class="law-toc__num">' + escapeHtml(displayNum(item.article_number)) + '</span>' +
            '<span class="law-toc__text" title="' + full + '">' +
                escapeHtml(truncateTitle(t, 26) || '—') +
            '</span>' +
            '</a></li>';

        if (stream === 'schedule') schedHtml += li;
        else html += li;
    });

    html += '</ul>';

    if (schedHtml) {
        html += '<div class="law-toc__head law-toc__head--sub">附表</div>' +
            '<ul class="law-toc__list">' + schedHtml + '</ul>';
    }

    return html;
}

/* ===== 正文：累积索引块 → 条款分组 → 增量渲染 ===== */

function applyPageData(data) {
    ldState.totalChunks = data.total_chunks || ldState.totalChunks;
    ldState.hasMore = !!data.has_more;
    ldState.page = data.page || ldState.page;

    if (data.chunks && data.chunks.length) {
        ldState.chunks = ldState.chunks.concat(data.chunks);
    }

    var groups = groupArticles(ldState.chunks);
    syncArticles(groups);
    backfillTocTitles(groups);
    renderMoreBar();
    refreshStats();
}

function groupArticles(chunks) {
    var groups = [];
    var cur = null;

    chunks.forEach(function(c) {
        var num = c.article_number || '';
        // 索引块按 chunk_index 升序，article_number 变化即为条款边界
        if (!cur || cur.num !== num) {
            cur = {
                num: num,
                title: c.article_title || '',
                partTitle: c.part_title || '',
                chunkType: c.chunk_type || '',
                firstIndex: c.chunk_index || 0,
                chunks: []
            };
            groups.push(cur);
        }
        if (!cur.title && c.article_title) cur.title = c.article_title;
        if (!cur.partTitle && c.part_title) cur.partTitle = c.part_title;
        if (!cur.chunkType && c.chunk_type) cur.chunkType = c.chunk_type;
        cur.chunks.push(c);
    });

    return groups;
}

/**
 * 增量同步条款 DOM：按 chunk_type 分流为正文/附表/片段三区，各分区与已渲染节点逐项比对
 * （首块序号 + 块数），从首个不一致处重渲染，避免「加载更多」时整篇重排。
 */
function syncArticles(groups) {
    var streams = { substantive: [], schedule: [], fragment: [] };
    groups.forEach(function(g) {
        streams[streamOf(g.num, g.chunkType)].push(g);
    });

    syncStream('substantive', streams.substantive, 'law-article-list', true);
    syncStream('schedule', streams.schedule, 'law-schedule-list', false);
    syncStream('fragment', streams.fragment, 'law-fragment-list', false);

    toggleSection('law-sched-section', streams.schedule.length);
    toggleSection('law-frag-section', streams.fragment.length);
}

function toggleSection(id, count) {
    var el = document.getElementById(id);
    if (!el) return;
    if (count) el.removeAttribute('hidden');
    else el.setAttribute('hidden', '');
}

function syncStream(key, groups, containerId, withParts) {
    var list = document.getElementById(containerId);
    if (!list) return;

    var state = ldState.rendered[key];
    var i = 0;
    while (i < state.length && i < groups.length &&
           state[i].firstIndex === groups[i].firstIndex &&
           state[i].count === groups[i].chunks.length) {
        i++;
    }

    for (var k = i; k < state.length; k++) {
        state[k].nodes.forEach(function(node) {
            if (node.parentNode) node.parentNode.removeChild(node);
        });
    }
    state.length = i;

    for (var j = i; j < groups.length; j++) {
        var nodes = [];
        // 编/章/节分隔标题：仅当与上一条款的所属篇章不同时输出
        if (withParts && groups[j].partTitle && (j === 0 || groups[j - 1].partTitle !== groups[j].partTitle)) {
            var part = document.createElement('div');
            part.className = 'law-part';
            part.textContent = groups[j].partTitle;
            list.appendChild(part);
            nodes.push(part);
        }
        var article = articleEl(groups[j]);
        list.appendChild(article);
        nodes.push(article);
        state.push({ firstIndex: groups[j].firstIndex, count: groups[j].chunks.length, nodes: nodes });
    }
}

/** 目录标题回填：outline 只含元数据标题，正文加载后再按需补全占位项 */
function backfillTocTitles(groups) {
    groups.forEach(function(g) {
        var idx = ldState.numToIdx[g.num];
        if (idx === undefined) return;
        var li = document.querySelector('.law-toc__item[data-target="a' + idx + '"]');
        if (!li || li.getAttribute('data-pending') !== '1') return;

        var t = resolveTitle(g);
        if (!t) return;

        var textEl = li.querySelector('.law-toc__text');
        textEl.textContent = truncateTitle(t, 26);
        textEl.setAttribute('title', truncateTitle(t, 60).replace(/"/g, '&quot;'));
        li.removeAttribute('data-pending');
    });
}

function articleEl(group) {
    var idx = ldState.numToIdx[group.num];

    var el = document.createElement('article');
    el.className = 'law-article';
    el.id = (idx === undefined) ? ('ax' + group.firstIndex) : ('a' + idx);

    var numEl = document.createElement('div');
    numEl.className = 'law-article__num';
    numEl.textContent = articleLabel(group.num);
    el.appendChild(numEl);

    var main = document.createElement('div');
    main.className = 'law-article__main';

    // 标题仅取自元数据：正文派生的标题是首行摘要，正文本身已含该行，重复展示无意义
    var title = metaTitle(group);
    if (title) {
        var titleEl = document.createElement('div');
        titleEl.className = 'law-article__title';
        titleEl.textContent = title;
        main.appendChild(titleEl);
    }

    var body = document.createElement('div');
    body.className = 'law-article__body';
    body.innerHTML = formatArticleBody(group.chunks);
    main.appendChild(body);

    el.appendChild(main);
    return el;
}

function articleLabel(num) {
    var d = displayNum(num);
    if (!d) return '条款';
    if (/^(附表|Sch)/.test(d)) return d;
    if (/[條条]|Article|Sec|§|Rule|Art\./i.test(d)) return d;
    if (/^[0-9]/.test(d)) return '第 ' + d + ' 条';
    return d;
}

/** 条款正文排版：分项（(1)/(a)/一、/① 等）悬挂缩进，其余行合并为段落 */
function formatArticleBody(chunks) {
    var lines = [];
    chunks.forEach(function(c, i) {
        var prev = i > 0 ? (chunks[i - 1].content || '') : '';
        var text = stripChunkOverlap(c.content || '', prev);
        text.replace(/\r\n?/g, '\n').split('\n').forEach(function(line) {
            var trimmed = line.replace(/[ \t]+$/g, '').trim();
            if (trimmed && !LD_NOISE_RE.test(trimmed)) lines.push(trimmed);
        });
    });

    if (!lines.length) return '';

    var html = '';
    var buffer = [];

    function flush() {
        if (buffer.length) {
            html += '<p>' + escapeHtml(buffer.join(' ')) + '</p>';
            buffer = [];
        }
    }

    lines.forEach(function(line) {
        var m = line.match(LAW_ITEM_RE);
        if (m) {
            flush();
            html += '<div class="law-item">' +
                '<span class="law-item__marker">' + escapeHtml(m[0].trim()) + '</span>' +
                '<span>' + escapeHtml(line.slice(m[0].length)) + '</span>' +
            '</div>';
        } else {
            buffer.push(line);
        }
    });
    flush();

    return html;
}

/**
 * 去除相邻索引块的重叠前缀（切块时保留约 150 字重叠，直接拼接会出现重复内容）
 */
function stripChunkOverlap(text, prevText) {
    if (!prevText || !text) return text;
    var max = Math.min(200, text.length, prevText.length);
    for (var len = max; len >= 20; len--) {
        if (prevText.slice(prevText.length - len) === text.slice(0, len)) {
            return text.slice(len);
        }
    }
    return text;
}

function truncateTitle(text, max) {
    if (!text) return '';
    var t = String(text).replace(/\s+/g, ' ').trim().replace(/[\s:：,，;；.。、\-—]+$/g, '');
    if (t.length > max) t = t.slice(0, max).trim() + '…';
    return t;
}

/* ===== 工具条与分页交互 ===== */

function ldSetFs(fs) {
    ldState.fs = fs;
    try { localStorage.setItem(LD_FS_KEY, fs); } catch (e) { /* 忽略 */ }

    var content = document.getElementById('law-content');
    if (content) content.setAttribute('data-fs', fs);

    document.querySelectorAll('.law-toolbar__btn[data-fs]').forEach(function(btn) {
        btn.classList.toggle('is-active', btn.getAttribute('data-fs') === fs);
    });
}

function ldPrint() {
    window.print();
}

function ldLoadMore() {
    setMoreButtonLoading();
    loadLawPage(ldState.page + 1).then(function(data) {
        if (data) applyPageData(data);
        else renderMoreBar();
    });
}

/** 目录跳转：目标条款尚未加载时按页加载直至出现 */
function ldGoArticle(idx, tries) {
    tries = tries || 0;
    var el = document.getElementById('a' + idx);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
    }
    if (!ldState.hasMore || tries >= 5) return;

    setMoreButtonLoading();
    loadLawPage(ldState.page + 1).then(function(data) {
        if (!data) return;          // 加载失败时停止递进，避免连续无效请求
        applyPageData(data);
        ldGoArticle(idx, tries + 1);
    });
}

function setMoreButtonLoading() {
    var btn = document.getElementById('law-more-btn');
    if (btn) { btn.disabled = true; btn.textContent = '加载中…'; }
}

function renderMoreBar() {
    var el = document.getElementById('law-more');
    if (!el) return;

    if (ldState.hasMore) {
        el.innerHTML = '<button type="button" class="law-toolbar__btn" id="law-more-btn" ' +
            'onclick="ldLoadMore()">加载更多</button>' +
            '<span class="law-more__hint">已显示 ' + ldState.chunks.length + ' / ' + ldState.totalChunks + ' 个索引块</span>';
    } else if (ldState.totalChunks > LD_PAGE_SIZE) {
        el.innerHTML = '<span class="law-more__hint">已显示全部 ' + ldState.chunks.length + ' 个索引块</span>';
    } else {
        el.innerHTML = '';
    }
}

function refreshStats() {
    var el = document.getElementById('law-stats');
    if (el) el.innerHTML = statsHtml();
}

/* ===== 目录联动（滚动高亮当前条款） ===== */

function attachScrollSpy() {
    detachScrollSpy();
    ldState.scrollHandler = function() {
        if (ldState.scrollRaf) return;
        ldState.scrollRaf = window.requestAnimationFrame(updateActiveToc);
    };
    window.addEventListener('scroll', ldState.scrollHandler, { passive: true });
    updateActiveToc();
}

function detachScrollSpy() {
    if (!ldState || !ldState.scrollHandler) return;
    window.removeEventListener('scroll', ldState.scrollHandler);
    ldState.scrollHandler = null;
    if (ldState.scrollRaf) {
        window.cancelAnimationFrame(ldState.scrollRaf);
        ldState.scrollRaf = 0;
    }
}

/**
 * 条款自上而下排列，用二分查找定位「最后一个已越过视口顶部」的条款，
 * 避免每次滚动都对全部条款取 rect（大法规可达数千条）。
 */
function updateActiveToc() {
    ldState.scrollRaf = 0;

    // 正文分区在前、附表/片段分区在后，DOM 顺序即纵向顺序，二分查找仍成立
    var arts = document.querySelectorAll('#law-content .law-article');
    if (!arts.length) return;

    var lo = 0, hi = arts.length - 1, found = -1;
    while (lo <= hi) {
        var mid = (lo + hi) >> 1;
        if (arts[mid].getBoundingClientRect().top <= 140) {
            found = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    var activeId = (found < 0 ? arts[0] : arts[found]).id;
    var toc = document.getElementById('law-toc');
    var items = document.querySelectorAll('.law-toc__item');

    for (var i = 0; i < items.length; i++) {
        var isActive = items[i].getAttribute('data-target') === activeId;
        items[i].classList.toggle('is-active', isActive);

        if (isActive && toc) {
            var top = items[i].offsetTop;
            var height = toc.clientHeight;
            var scrollTop = toc.scrollTop;
            if (top < scrollTop) {
                toc.scrollTop = Math.max(0, top - 12);
            } else if (top + items[i].offsetHeight > scrollTop + height) {
                toc.scrollTop = top + items[i].offsetHeight - height + 12;
            }
        }
    }
}

/* ===== 相关法规推荐 ===== */

function renderSimilarSection(data) {
    var el = document.getElementById('law-similar');
    if (!el) return;

    var list = (data && data.similar_laws) ? data.similar_laws : [];
    if (!list.length) return;

    el.removeAttribute('hidden');

    var html = '<div class="law-similar__lead">基于《' +
        escapeHtml(data.query_law_title || '当前法规') +
        '》的语义相似度，推荐以下 ' + list.length + ' 部相关法规：</div>';
    html += '<div class="law-similar__list">';

    list.forEach(function(item) {
        var lawId = String(item.law_id || '').replace(/[^A-Za-z0-9_\-.]/g, '');
        var pct = Math.round((item.similarity || 0) * 100);
        var fillCls = pct >= 75 ? ' is-high' : (pct >= 55 ? ' is-mid' : '');

        html += '<div class="law-similar__card">';
        html += '<div>';
        html += '<div class="law-similar__title">' + escapeHtml(item.title || '未知法规') + '</div>';
        html += '<div class="law-meta">';
        if (item.jurisdiction) {
            html += '<span class="badge badge-info">' + escapeHtml(item.jurisdiction) + '</span>';
        }
        html += '<span class="badge badge-muted">' + (item.match_chunk_count || 0) + ' 个匹配片段</span>';
        html += '</div>';
        html += '<div class="law-similar__meter">';
        html += '<div class="law-similar__track"><div class="law-similar__fill' + fillCls +
            '" style="width:' + pct + '%"></div></div>';
        html += '<span class="law-similar__pct">' + pct + '%</span>';
        html += '</div>';
        html += '</div>';
        html += '<button type="button" class="btn btn-secondary btn-sm" ' +
            'onclick="ldOpenSimilar(\'' + lawId + '\')">查看</button>';
        html += '</div>';
    });

    html += '</div>';
    el.innerHTML = html;
}

function ldOpenSimilar(lawId) {
    if (lawId) navigate('law-detail/' + lawId);
}

// 返回检索页，保留之前搜索状态
function goBackToSearch() {
    navigate('search');
}
