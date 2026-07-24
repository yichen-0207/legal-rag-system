// ===== 知识图谱页面 v2.0 =====
// 实体类型配置（形状区分，颜色改用治法域区分）
const ENTITY_DISPLAY = {
    LegalSubject: { label: '法律主体', shape: 'roundRect' },
    Obligation:   { label: '法律义务', shape: 'diamond' },
    Right:        { label: '法律权利', shape: 'roundRect' },
    Penalty:      { label: '处罚措施', shape: 'triangle' },
    Concept:      { label: '法律概念', shape: 'ellipse' },
    Article:      { label: '条款',     shape: 'ellipse' },
};

// 法域配色方案 — HSL 均匀分布，支持任意数量法域
// 使用黄金角确保相邻色相视觉差异最大化
var JUR_COLOR_CACHE = {};
function getJurColor(jurisdiction) {
    if (!jurisdiction) return '#6B7280';
    if (JUR_COLOR_CACHE[jurisdiction]) return JUR_COLOR_CACHE[jurisdiction];

    // 预设高频法域固定颜色（保持品牌一致性）
    var PREDEFINED = {
        '澳门': '#2563EB', '中国澳门': '#2563EB', 'Macau': '#2563EB',
        '香港': '#DC2626', '中国香港': '#DC2626', 'Hong Kong': '#DC2626',
        '新加坡': '#D97706', 'Singapore': '#D97706',
        '美国': '#7C3AED', 'USA': '#7C3AED', 'US': '#7C3AED',
        '欧盟': '#0D9488', 'EU': '#0D9488',
        '英国': '#E11D48', 'UK': '#E11D48', 'United Kingdom': '#E11D48',
        '日本': '#BE123C', 'Japan': '#BE123C',
        '韩国': '#1D4ED8', 'South Korea': '#1D4ED8', 'Korea': '#1D4ED8',
        '德国': '#F59E0B', 'Germany': '#F59E0B',
        '法国': '#6366F1', 'France': '#6366F1',
        '印度': '#EA580C', 'India': '#EA580C',
        '澳大利亚': '#0891B2', 'Australia': '#0891B2',
        '加拿大': '#059669', 'Canada': '#059669',
        '巴西': '#0E7490', 'Brazil': '#0E7490',
        '俄罗斯': '#4F46E5', 'Russia': '#4F46E5',
        '中国': '#8B2500', 'China': '#8B2500', '中国大陆': '#8B2500',
    };

    // 先尝试精确匹配
    if (PREDEFINED[jurisdiction]) {
        JUR_COLOR_CACHE[jurisdiction] = PREDEFINED[jurisdiction];
        return PREDEFINED[jurisdiction];
    }

    // 尝试前缀匹配（处理 "中国香港" 等复合名称）
    for (var key in PREDEFINED) {
        if (jurisdiction.indexOf(key) !== -1) {
            JUR_COLOR_CACHE[jurisdiction] = PREDEFINED[key];
            return PREDEFINED[key];
        }
    }

    // 未知法域：使用 hash 在 HSL 色环上均匀采样
    // 固定饱和度和明度，仅变化色相，确保所有颜色视觉协调
    var hash = 0;
    for (var i = 0; i < jurisdiction.length; i++) {
        hash = jurisdiction.charCodeAt(i) + ((hash << 6) - hash);
    }
    // 黄金角 ~137.5°，在 0-360 范围内分布最均匀
    var GOLDEN_ANGLE = 137.508;
    // 用 hash 决定起始偏移，然后按黄金角步进
    var hue = ((Math.abs(hash) * GOLDEN_ANGLE) % 360 + 360) % 360;
    // 饱和度 65-75%，明度 50-60%，颜色鲜艳又不刺眼
    var sat = 55 + (Math.abs(hash) % 20);
    var lig = 45 + (Math.abs(hash) % 15);
    var color = 'hsl(' + Math.round(hue) + ', ' + sat + '%, ' + lig + '%)';

    JUR_COLOR_CACHE[jurisdiction] = color;
    return color;
}

// 动态更新法域图例
function updateKGLegend(nodes) {
    var legend = document.getElementById('kg-legend');
    if (!legend) return;

    // 收集当前图谱中的法域列表，去重并保留顺序
    var jurMap = {};
    nodes.forEach(function(n) {
        var j = n.jurisdiction || '未知';
        if (!jurMap[j]) jurMap[j] = 0;
        jurMap[j]++;
    });

    var jurList = Object.keys(jurMap).sort(function(a, b) { return jurMap[b] - jurMap[a]; });
    if (jurList.length === 0) return;

    var jurHtml = jurList.map(function(jur) {
        var color = getJurColor(jur);
        return '<span><span class="kg-legend-icon" style="display:inline-block;width:8px;height:8px;background:' + color + ';border-radius:50%;"></span> ' + escapeHtml(jur) + '</span>';
    }).join('');

    // 保留其他图例行（类型、连线、大小），只替换法域行
    var children = legend.children;
    if (children.length > 0) {
        // 替换第一行（法域行）
        children[0].innerHTML = '<span style="font-weight:600;min-width:3em;">法域：</span>' + jurHtml;
    }
}

// 颜色提亮工具（同时支持 hex 和 hsl）
function lightenColor(color, amount) {
    if (!color) return '#9CA3AF';
    // 如果是 HSL 格式，直接降低饱和度来提高亮度感
    if (color.startsWith('hsl')) {
        var parts = color.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
        if (parts) {
            var h = parseInt(parts[1]), s = parseInt(parts[2]), l = parseInt(parts[3]);
            l = Math.min(100, l + amount * 0.6);
            s = Math.max(0, s - amount * 0.3);
            return 'hsl(' + h + ', ' + Math.round(s) + '%, ' + Math.round(l) + '%)';
        }
        return color;
    }
    // hex 格式
    color = color.replace('#', '');
    var r = parseInt(color.substring(0,2), 16);
    var g = parseInt(color.substring(2,4), 16);
    var b = parseInt(color.substring(4,6), 16);
    r = Math.min(255, r + amount);
    g = Math.min(255, g + amount);
    b = Math.min(255, b + amount);
    return '#' + [r,g,b].map(function(c) { return c.toString(16).padStart(2,'0'); }).join('');
}

registerPage('kg', function(container) {
    container.innerHTML = `
        <style>
        .kg-mode-btn {
            padding:0.375rem 0.875rem;
            border:1px solid var(--color-border);
            background:var(--color-card-bg);
            color:var(--color-text-secondary);
            border-radius:6px;
            cursor:pointer;
            font-size:0.8125rem;
            transition:all 0.15s;
        }
        .kg-mode-btn:hover { background:var(--color-bg); }
        .kg-mode-btn.active {
            background:var(--color-primary);
            color:#fff;
            border-color:var(--color-primary);
        }
        </style>
        <h1 class="page-title">法规知识图谱</h1>
        <p class="page-desc">探索法规内部条款关联、搜索法律实体、跨法域对比——可视化知识图谱助您理解法律结构。</p>
        <hr class="divider">

        <!-- 模式切换 -->
        <div style="margin-bottom:1rem;">
            <div class="kg-mode-tabs" style="display:flex;gap:0.5rem;">
                <button class="kg-mode-btn active" data-mode="law" onclick="switchKGMode('law')">法规视图</button>
                <button class="kg-mode-btn" data-mode="search" onclick="switchKGMode('search')">实体搜索</button>
                <button class="kg-mode-btn" data-mode="cross" onclick="switchKGMode('cross')">跨法域对比</button>
            </div>
        </div>

        <!-- 模式1：法规视图 -->
        <div id="kg-law-panel" class="kg-panel">
            <div class="card" style="padding:1rem;">
                <div class="grid-2" style="margin-bottom:0.75rem;">
                    <div>
                        <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">选择法域</label>
                        <select id="kg-law-jur" class="select" onchange="filterLawsByJur()"><option value="">请选择法域</option></select>
                    </div>
                    <div>
                        <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">选择法规</label>
                        <select id="kg-law-select" class="select"><option value="">请先选择法域</option></select>
                    </div>
                </div>
                <div style="display:flex;gap:0.5rem;">
                    <button id="kg-law-btn" class="btn btn-primary" onclick="loadLawGraph()">查看图谱</button>
                </div>
                <div style="font-size:0.75rem;color:var(--color-text-muted);margin-top:0.375rem;">选择法域后自动筛选该法域下的法规，图谱以圆形布局展示</div>
            </div>
        </div>

        <!-- 模式2：实体搜索 -->
        <div id="kg-search-panel" class="kg-panel" style="display:none;">
            <div class="card" style="padding:1rem;">
                <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">搜索法律实体</label>
                <div style="display:flex;gap:0.5rem;">
                    <input type="text" id="kg-search-input" class="input" placeholder="例如：数据控制者、跨境传输、罚款、个人数据" style="flex:1;" onkeydown="if(event.key==='Enter')searchKGEntities()">
                    <button class="btn btn-primary" onclick="searchKGEntities()">搜索</button>
                </div>
                <div style="font-size:0.75rem;color:var(--color-text-muted);margin-top:0.375rem;">输入法律实体关键词，系统检索相关条款并构建邻域图谱</div>
                <div id="kg-search-results" class="kg-search-results" style="margin-top:0.5rem;"></div>
            </div>
        </div>

        <!-- 模式3：跨法域对比 -->
        <div id="kg-cross-panel" class="kg-panel" style="display:none;">
            <div class="card" style="padding:1rem;">
                <div class="grid-2" style="margin-bottom:0.75rem;">
                    <div>
                        <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">法域 A</label>
                        <select id="kg-cross-jur-a" class="select"></select>
                    </div>
                    <div>
                        <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">法域 B</label>
                        <select id="kg-cross-jur-b" class="select"></select>
                    </div>
                </div>
                <div class="grid-2" style="margin-bottom:0.75rem;">
                    <div>
                        <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">对比专题</label>
                        <select id="kg-cross-topic" class="select"></select>
                    </div>
                    <div style="display:flex;align-items:flex-end;">
                        <button class="btn btn-primary" onclick="loadCrossGraph()">开始对比</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- 图谱容器 -->
        <div class="card" style="margin-top:0.75rem;padding:0;">
            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.625rem 1rem;border-bottom:1px solid var(--color-border);">
                <div id="kg-graph-title" class="kg-graph-title" style="font-weight:600;font-size:0.8125rem;color:var(--color-text);"></div>
                <div style="display:flex;gap:0.375rem;">
                    <button class="btn btn-sm" onclick="toggleKGLayout()" id="kg-layout-btn" style="font-size:0.75rem;">布局：力导向</button>
                    <button class="btn btn-sm" onclick="resetKGView()" style="font-size:0.75rem;">重置视图</button>
                    <button class="btn btn-sm" onclick="fitKGView()" style="font-size:0.75rem;">适应布局</button>
                </div>
            </div>
            <div id="kg-canvas-container" style="position:relative;height:500px;background:var(--color-bg);overflow:hidden;cursor:grab;">
                <canvas id="kg-canvas" style="width:100%;height:100%;"></canvas>
                <div id="kg-tooltip" style="position:absolute;display:none;background:rgba(255,255,255,0.96);border:1px solid var(--color-border);border-radius:6px;padding:0.5rem 0.75rem;font-size:0.75rem;max-width:280px;box-shadow:0 4px 12px rgba(0,0,0,0.08);pointer-events:none;z-index:10;"></div>
                <!-- 图例 -->
                <div id="kg-legend" style="position:absolute;bottom:6px;left:6px;background:rgba(255,255,255,0.94);border:1px solid var(--color-border);border-radius:6px;padding:5px 9px;font-size:0.65rem;color:var(--color-text-secondary);z-index:5;line-height:1.7;pointer-events:none;">
                    <div style="display:flex;flex-wrap:wrap;gap:0.2rem 0.6rem;">
                        <span style="font-weight:600;min-width:3em;">法域：</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:8px;height:8px;background:#2563EB;border-radius:50%;"></span> 澳门</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:8px;height:8px;background:#DC2626;border-radius:50%;"></span> 香港</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:8px;height:8px;background:#D97706;border-radius:50%;"></span> 新加坡</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:8px;height:8px;background:#7C3AED;border-radius:50%;"></span> 美国</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:8px;height:8px;background:#0D9488;border-radius:50%;"></span> 欧盟</span>
                    </div>
                    <div style="display:flex;flex-wrap:wrap;gap:0.2rem 0.6rem;margin-top:1px;">
                        <span style="font-weight:600;min-width:3em;">类型：</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:9px;height:5px;background:#6B7280;border-radius:2px;"></span> 主体</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:7px;height:7px;background:#6B7280;transform:rotate(45deg);"></span> 义务</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:9px;height:5px;background:#6B7280;border-radius:2px;"></span> 权利</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-bottom:9px solid #6B7280;"></span> 处罚</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:7px;height:7px;background:#6B7280;border-radius:50%;"></span> 概念</span>
                        <span><span class="kg-legend-icon" style="display:inline-block;width:7px;height:7px;background:#6B7280;border-radius:50%;"></span> 条款</span>
                    </div>
                    <div style="display:flex;flex-wrap:wrap;gap:0.2rem 0.6rem;margin-top:1px;">
                        <span style="font-weight:600;min-width:3em;">连线：</span>
                        <span>|<span style="display:inline-block;width:16px;height:0;border-top:1.5px dashed #CBD5E1;margin:0 2px;"></span>同法域</span>
                        <span>|<span style="display:inline-block;width:16px;height:2.5px;background:#A78BFA;margin:0 2px;border-radius:2px;"></span>跨法域</span>
                    </div>
                    <div style="display:flex;flex-wrap:wrap;gap:0.2rem 0.6rem;margin-top:1px;">
                        <span style="font-weight:600;min-width:3em;">节点大小：</span>
                        <span><span style="display:inline-block;width:13px;height:13px;background:#6B7280;border-radius:50%;vertical-align:middle;"></span> 高连接</span>
                        <span><span style="display:inline-block;width:9px;height:9px;background:#6B7280;border-radius:50%;vertical-align:middle;"></span> 中连接</span>
                        <span><span style="display:inline-block;width:6px;height:6px;background:#6B7280;border-radius:50%;vertical-align:middle;"></span> 低连接</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- 控制栏 -->
        <div class="card" style="margin-top:0.5rem;padding:0.625rem 1rem;">
            <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center;font-size:0.8125rem;">
                <label style="display:flex;align-items:center;gap:0.25rem;color:var(--color-text-secondary);">
                    <input type="checkbox" id="kg-filter-same-law" checked> 同法规关联
                </label>
                <label style="display:flex;align-items:center;gap:0.25rem;color:var(--color-text-secondary);">
                    <input type="checkbox" id="kg-filter-cross" checked> 跨域关联
                </label>
                <label style="display:flex;align-items:center;gap:0.375rem;color:var(--color-text-secondary);">
                    跨域阈值
                    <input type="range" id="kg-edge-threshold" min="0" max="0.9" step="0.05" value="0.35" style="width:80px;">
                    <span id="kg-threshold-val" style="font-weight:600;min-width:2rem;">0.35</span>
                </label>
                <label style="display:flex;align-items:center;gap:0.25rem;color:var(--color-text-secondary);">
                    标签
                    <select id="kg-label-mode" class="select" style="width:auto;font-size:0.75rem;">
                        <option value="smart">智能</option>
                        <option value="core">仅核心</option>
                        <option value="all">全部</option>
                        <option value="none">无</option>
                    </select>
                </label>
                <span style="color:var(--color-text-muted);font-size:0.75rem;" id="kg-stats-badge"></span>
            </div>
            <div style="font-size:0.7rem;color:var(--color-text-muted);margin-top:0.25rem;">提示：力导向布局更易识别核心条款，圆形布局适合查看整体结构</div>
        </div>

        <!-- 详情面板 -->
        <div id="kg-detail-panel" style="display:none;margin-top:0.5rem;"></div>

        <!-- 智能洞察 -->
        <div id="kg-insight-panel" class="card" style="margin-top:0.5rem;padding:0.625rem 1rem;display:none;">
            <div style="font-weight:600;font-size:0.8125rem;color:var(--color-text);margin-bottom:0.25rem;">图谱洞察</div>
            <div id="kg-insight-text" style="font-size:0.8125rem;color:var(--color-text-secondary);line-height:1.7;"></div>
        </div>
    `;

    // 加载下拉列表
    Promise.all([
        api.get('/search/jurisdictions'),
        api.get('/structured-analysis/topics'),
        api.get('/search/laws', { page_size: 200 })
    ]).then(function(results) {
        var jurData = results[0];
        var topics = results[1];
        var lawsResp = results[2];

        // 存储所有法规数据用于按法域筛选
        var allLaws = [];
        if (lawsResp && lawsResp.data) {
            allLaws = lawsResp.data.laws || lawsResp.data.data || lawsResp.data || [];
        }
        window._kgAllLaws = allLaws;

        // 法域下拉（法规视图）
        var jurSet = {};
        allLaws.forEach(function(l) {
            var j = l.jurisdiction || '';
            if (j) jurSet[j] = true;
        });
        var jurList = Object.keys(jurSet).sort();
        var lawJurSel = document.getElementById('kg-law-jur');
        jurList.forEach(function(j) {
            var opt = document.createElement('option');
            opt.value = j;
            opt.textContent = j;
            lawJurSel.appendChild(opt);
        });

        // 法域选择（跨法域对比）
        var allJurList = (jurData && jurData.data) || jurData || [];
        ['kg-cross-jur-a', 'kg-cross-jur-b'].forEach(function(id) {
            var sel = document.getElementById(id);
            (allJurList.forEach || Array.prototype.forEach).call(allJurList, function(j) {
                var opt = document.createElement('option');
                opt.value = typeof j === 'string' ? j : (j.code || j.name || j);
                opt.textContent = typeof j === 'string' ? j : (j.name || j.code || j);
                sel.appendChild(opt);
            });
        });

        // 专题选择
        var topicList = (topics && topics.data) || topics || [];
        var topicSel = document.getElementById('kg-cross-topic');
        (topicList.forEach || Array.prototype.forEach).call(topicList, function(t) {
            var opt = document.createElement('option');
            opt.value = t.id || t;
            opt.textContent = t.name || t;
            topicSel.appendChild(opt);
        });
    });
});

window.filterLawsByJur = function() {
    var jur = document.getElementById('kg-law-jur').value;
    var lawSelect = document.getElementById('kg-law-select');
    lawSelect.innerHTML = '';
    if (!jur) {
        lawSelect.innerHTML = '<option value="">请先选择法域</option>';
        return;
    }
    var allLaws = window._kgAllLaws || [];
    var filtered = allLaws.filter(function(l) { return (l.jurisdiction || '') === jur; });
    if (filtered.length === 0) {
        lawSelect.innerHTML = '<option value="">该法域暂无法规</option>';
        return;
    }
    filtered.forEach(function(l) {
        var opt = document.createElement('option');
        opt.value = l.law_id || l.id || '';
        opt.textContent = l.title || l.name || '';
        lawSelect.appendChild(opt);
    });
};

// ===== 模式切换 =====
window.switchKGMode = function(mode) {
    document.querySelectorAll('.kg-mode-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.mode === mode);
    });
    document.querySelectorAll('.kg-panel').forEach(function(p) { p.style.display = 'none'; });
    document.getElementById('kg-' + mode + '-panel').style.display = 'block';
};

// ===== 清除图谱 =====
function clearGraph() {
    if (window._kgState) window._kgState.running = false;
    window._kgState = null;
    document.getElementById('kg-insight-panel').style.display = 'none';
    var canvas = document.getElementById('kg-canvas');
    if (canvas) {
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    document.getElementById('kg-graph-title').textContent = '';
    document.getElementById('kg-detail-panel').style.display = 'none';
    document.getElementById('kg-stats-badge').textContent = '';
}

// ===== 渲染图谱 =====
function renderKG(data, title, layoutMode) {
    clearGraph();
    var nodes = data.nodes || [];
    var edges = data.edges || [];
    if (nodes.length === 0) { document.getElementById('kg-graph-title').textContent = '暂无数据'; return; }

    var showAll = data._showAll;
    var MAX_VISIBLE = 40;
    // 抗杂乱：后端未过滤时前端补一刀（用户可点击展开全部）
    var truncated = false;
    var totalOriginal = data.raw_node_count || (data.nodes ? data.nodes.length : nodes.length);
    if (totalOriginal > MAX_VISIBLE && !showAll) {
        nodes.sort(function(a,b) { return (b.importance_score || 0) - (a.importance_score || 0); });
        var kept = {};
        nodes.slice(0, MAX_VISIBLE).forEach(function(n) { kept[n.id] = true; });
        nodes = nodes.slice(0, MAX_VISIBLE);
        edges = edges.filter(function(e) { return kept[e.from] && kept[e.to]; });
        truncated = true;
    }

    document.getElementById('kg-graph-title').textContent = title || '知识图谱';
    var badgeText = nodes.length + ' 节点 | ' + edges.length + ' 边';

    // 动态更新法域图例
    updateKGLegend(nodes);

    if (truncated) {
        badgeText += ' | <span id="kg-expand-all-btn" style="color:#D97706;font-weight:600;cursor:pointer;text-decoration:underline;">仅显示前40个重要节点（共' + totalOriginal + '个），点击展开全部</span>';
    } else if (showAll) {
        badgeText += ' | <span id="kg-collapse-btn" style="color:#059669;font-weight:600;cursor:pointer;text-decoration:underline;">已展开全部节点，点击收缩</span>';
    }
    document.getElementById('kg-stats-badge').innerHTML = badgeText;

    // 注册"展开全部"点击事件
    var expandBtn = document.getElementById('kg-expand-all-btn');
    if (expandBtn) {
        expandBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            data._showAll = true;
            renderKG(data, title, layoutMode);
        });
    }

    var collapseBtn = document.getElementById('kg-collapse-btn');
    if (collapseBtn) {
        collapseBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            data._showAll = false;
            renderKG(data, title, layoutMode);
        });
    }

    var canvas = document.getElementById('kg-canvas');
    var container = document.getElementById('kg-canvas-container');
    if (!canvas || !container) return;

    var dpr = window.devicePixelRatio || 1;
    var width = container.clientWidth;
    var height = container.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    var isCircular = layoutMode === 'circular';

    // 初始化位置
    var positions = {}, velocities = {};
    var cx = width / 2, cy = height / 2;

    if (isCircular && nodes.length > 0) {
        // 圆形展开布局：度高的靠内、度低的靠外，整体松散自然
        var baseRadius = Math.min(width, height) * 0.38;
        var maxDeg = 0, totalDeg = 0;
        nodes.forEach(function(n) { var d = n.degree || 0; maxDeg = Math.max(maxDeg, d); totalDeg += d; });
        var avgDeg = totalDeg / nodes.length;

        // 按度数排序，同度数节点聚在一起
        var sortedNodes = nodes.slice().sort(function(a,b) { return (b.degree || 0) - (a.degree || 0); });
        sortedNodes.forEach(function(n, i) {
            var angle = (i / nodes.length) * Math.PI * 2 + (Math.random() - 0.5) * 0.3;
            var degRatio = maxDeg > 0 ? ((n.degree || avgDeg) / Math.max(maxDeg, 1)) : 0.5;
            // 度数高 → 半径小（靠近中心），度数低 → 半径大（靠外）
            var minR = 0.35, maxR = 1.0;
            var r = baseRadius * (minR + (1 - degRatio * 0.6) * (maxR - minR));
            // 加随机偏移，让圆形不那么规整
            r += (Math.random() - 0.5) * baseRadius * 0.2;
            positions[n.id] = {
                x: cx + Math.cos(angle) * r,
                y: cy + Math.sin(angle) * r
            };
            velocities[n.id] = { x: 0, y: 0 };
        });
    } else {
        // 法域聚簇分布（原有逻辑）
        var jurGroups = {};
        nodes.forEach(function(n) {
            var jur = n.jurisdiction || '';
            if (!jurGroups[jur]) jurGroups[jur] = [];
            jurGroups[jur].push(n);
        });
        var jurList = Object.keys(jurGroups);
        jurList.forEach(function(jur, idx) {
            var group = jurGroups[jur];
            var clusterCx, clusterCy;
            if (jurList.length === 2) {
                clusterCx = idx === 0 ? width * 0.28 : width * 0.72;
                clusterCy = cy + (idx === 0 ? -height * 0.06 : height * 0.06);
            } else {
                var angle = (idx / jurList.length) * Math.PI * 2;
                clusterCx = cx + Math.cos(angle) * Math.min(width, height) * 0.30;
                clusterCy = cy + Math.sin(angle) * Math.min(width, height) * 0.26;
            }
            group.forEach(function(node, i) {
                var ga = Math.PI * (3 - Math.sqrt(5));
                var a = i * ga + idx * 2.0;
                var r = 24 + Math.sqrt(i) * 20;
                positions[node.id] = { x: clusterCx + Math.cos(a) * r, y: clusterCy + Math.sin(a) * r };
                velocities[node.id] = { x: 0, y: 0 };
            });
        });
    }

    var state = {
        nodes: nodes, edges: edges, positions: positions, velocities: velocities,
        zoom: 1, panX: 0, panY: 0,
        hoveredNode: null, selectedNode: null, draggedNode: null,
        hoveredEdge: null, selectedEdge: null,
        filterSameLaw: true, filterCross: true,
        edgeThreshold: parseFloat(document.getElementById('kg-edge-threshold').value),
        labelMode: document.getElementById('kg-label-mode').value,
        layoutTick: 0, maxLayoutTicks: isCircular ? 3 : 300, running: true,
        isCircular: isCircular,
        _time: 0  // 用于呼吸动画
    };
    window._kgState = state;
    bindKGControls(state);
    generateKGInsight(data);

    // 力导向（圆形布局少量迭代让节点自然放松）
    function tick() {
        if (!state.running) return;
        if (state.layoutTick < state.maxLayoutTicks) {
            applyKGForces(state, width, height);
            state.layoutTick++;
        }
        state._time += 0.02;
        drawKG(ctx, state, width, height);
        requestAnimationFrame(tick);
    }
    tick();
    bindKGCanvas(canvas, container, state);
}

// ===== 力导向布局 =====
function applyKGForces(state, w, h) {
    var nodes = state.nodes, pos = state.positions, vel = state.velocities;
    var n = nodes.length, tick = state.layoutTick;
    var progress = tick / state.maxLayoutTicks;
    var temp = Math.max(0.05, 1 - progress * 0.96);
    var k = Math.sqrt(w * h / n) * 0.7;
    var rep = k * k * 1.2, spr = 0.018 * temp;

    var jurGroups = {};
    nodes.forEach(function(n) { var j = n.jurisdiction || ''; if (!jurGroups[j]) jurGroups[j] = []; jurGroups[j].push(n); });

    for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
            var ni = nodes[i], nj = nodes[j];
            var pi = pos[ni.id], pj = pos[nj.id];
            var dx = pi.x - pj.x, dy = pi.y - pj.y;
            var dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
            var sameJur = (ni.jurisdiction || '') === (nj.jurisdiction || '');
            var force = rep * (sameJur ? 1.0 : 1.8) / (dist * dist);
            vel[ni.id].x += (dx / dist) * force;
            vel[ni.id].y += (dy / dist) * force;
            vel[nj.id].x -= (dx / dist) * force;
            vel[nj.id].y -= (dy / dist) * force;
        }
    }

    state.edges.forEach(function(e) {
        var u = pos[e.from], v = pos[e.to];
        if (!u || !v) return;
        var dx = v.x - u.x, dy = v.y - u.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        var ideal = k * 1.4;
        var force = (dist - ideal) * spr;
        u.x += (dx / dist) * force; u.y += (dy / dist) * force;
        v.x -= (dx / dist) * force; v.y -= (dy / dist) * force;
    });

    Object.keys(jurGroups).forEach(function(jur) {
        var group = jurGroups[jur];
        if (group.length < 2) return;
        var gx = 0, gy = 0;
        group.forEach(function(n) { gx += pos[n.id].x; gy += pos[n.id].y; });
        gx /= group.length; gy /= group.length;
        group.forEach(function(n) {
            var p = pos[n.id];
            p.x += (gx - p.x) * 0.03 * temp;
            p.y += (gy - p.y) * 0.03 * temp;
        });
    });

    nodes.forEach(function(n) {
        var p = pos[n.id], v = vel[n.id];
        p.x += v.x * temp; p.y += v.y * temp;
        v.x *= 0.85; v.y *= 0.85;
        p.x = Math.max(20, Math.min(w - 20, p.x));
        p.y = Math.max(20, Math.min(h - 20, p.y));
    });
}

// ===== 绘制图谱 =====
function drawKG(ctx, state, w, h) {
    ctx.clearRect(0, 0, w, h);

    // 背景微网格
    ctx.save();
    ctx.strokeStyle = 'rgba(0,0,0,0.035)';
    ctx.lineWidth = 0.5;
    var gridStep = 40;
    for (var gx = 0; gx <= w; gx += gridStep) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
    }
    for (var gy = 0; gy <= h; gy += gridStep) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
    ctx.restore();

    ctx.save();
    ctx.translate(state.panX + w / 2, state.panY + h / 2);
    ctx.scale(state.zoom, state.zoom);
    ctx.translate(-w / 2, -h / 2);

    var si = 1 / Math.max(state.zoom, 0.1);
    var density = state.edges.length / Math.max(state.nodes.length, 1);
    var baseAlpha = density > 4 ? 0.12 : (density > 2.5 ? 0.2 : 0.35);

    var neighborSet = {};
    if (state.hoveredNode && state._adj) {
        (state._adj[state.hoveredNode.id] || []).forEach(function(item) { neighborSet[item] = true; });
    }
    // 边悬停时，只高亮两个端点
    if (state.hoveredEdge && !state.hoveredNode) {
        neighborSet[state.hoveredEdge.from] = true;
        neighborSet[state.hoveredEdge.to] = true;
    }

    var adj = {};
    state.nodes.forEach(function(n) { adj[n.id] = []; });
    state.edges.forEach(function(e) {
        if (adj[e.from]) adj[e.from].push(e.to);
        if (adj[e.to]) adj[e.to].push(e.from);
    });
    state._adj = adj;

    // 绘制边
    state.edges.forEach(function(e) {
        var u = state.positions[e.from], v = state.positions[e.to];
        if (!u || !v) return;
        var isCross = e.type === 'cross_jurisdiction' || e.type === 'semantic_similar';
        if (!state.filterCross && isCross) return;
        if (!state.filterSameLaw && !isCross) return;
        if (isCross && e.similarity != null && e.similarity < state.edgeThreshold) return;

        var midX = (u.x + v.x) / 2, midY = (u.y + v.y) / 2;
        var dx = v.x - u.x, dy = v.y - u.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        var curv = isCross ? 0.12 : 0.04;
        var ctrlX = midX - (dy / dist) * dist * curv;
        var ctrlY = midY + (dx / dist) * dist * curv;

        var isNodeHovered = state.hoveredNode && (state.hoveredNode.id === e.from || state.hoveredNode.id === e.to);
        var isEdgeHovered = state.hoveredEdge && state.hoveredEdge.from === e.from && state.hoveredEdge.to === e.to && state.hoveredEdge.type === e.type;
        var isEdgeSelected = state.selectedEdge && state.selectedEdge.from === e.from && state.selectedEdge.to === e.to && state.selectedEdge.type === e.type;
        var isHovered = isNodeHovered || isEdgeHovered || isEdgeSelected;

        // 跨域边发光效果
        if (isCross && isHovered) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(u.x, u.y);
            ctx.quadraticCurveTo(ctrlX, ctrlY, v.x, v.y);
            ctx.strokeStyle = 'rgba(167,139,250,0.15)';
            ctx.lineWidth = 8 * si;
            ctx.stroke();
            ctx.restore();
        }

        ctx.beginPath();
        ctx.moveTo(u.x, u.y);
        ctx.quadraticCurveTo(ctrlX, ctrlY, v.x, v.y);

        if (isHovered) {
            ctx.strokeStyle = isCross ? '#8B5CF6' : '#F59E0B';
            ctx.lineWidth = 2.5 * si;
            ctx.globalAlpha = 0.9;
        } else if (state.hoveredNode || state.hoveredEdge) {
            ctx.strokeStyle = '#CBD5E1';
            ctx.lineWidth = 0.7 * si;
            ctx.globalAlpha = 0.08;
        } else if (isCross) {
            ctx.strokeStyle = '#A78BFA';
            ctx.lineWidth = 1.8 * si;
            ctx.globalAlpha = Math.min(0.6, baseAlpha * 2.0);
            ctx.setLineDash([]);
        } else {
            ctx.strokeStyle = '#64748B';
            ctx.lineWidth = 1.2 * si;
            ctx.globalAlpha = Math.min(0.65, baseAlpha * 2.0);
            ctx.setLineDash([5 * si, 4 * si]);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;

        // 跨域边：在端点处加渐变收束小圆
        if (isCross && !isHovered && !state.hoveredNode) {
            ctx.save();
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.globalAlpha = 0.25;
            ctx.fillStyle = '#A78BFA';
            ctx.beginPath(); ctx.arc(u.x, u.y, 1.8 * si, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(v.x, v.y, 1.8 * si, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
        }

        // 端点圆点（提高边与节点的视觉连接感）
        ctx.save();
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.fillStyle = isHovered ? (isCross ? '#8B5CF6' : '#F59E0B') : (isCross ? '#A78BFA' : '#64748B');
        ctx.globalAlpha = isHovered ? 0.7 : 0.3;
        ctx.beginPath(); ctx.arc(u.x, u.y, 2.5 * si, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(v.x, v.y, 2.5 * si, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    });

    // 度数阈值
    var degThreshold = 0;
    if (state.nodes.length > 12) {
        var degs = state.nodes.map(function(n) { return n.degree || 0; }).sort(function(a,b) { return a - b; });
        degThreshold = degs[Math.floor(degs.length * 0.7)] || 1;
    }

    // 全局节点阴影
    ctx.shadowColor = 'rgba(0,0,0,0.10)';
    ctx.shadowBlur = 6 * si;
    ctx.shadowOffsetY = 3 * si;

    // 绘制节点
    state.nodes.forEach(function(n) {
        var p = state.positions[n.id];
        if (!p) return;
        var et = ENTITY_DISPLAY[n.entity_type] || ENTITY_DISPLAY.Article;
        // 按连接度分档：0=极小灰色, 1-2=小, 3-7=中, 8+=大
        var deg = n.degree || 0;
        var baseR;
        if (deg >= 8) {
            baseR = 13;          // 大节点（核心枢纽）
        } else if (deg >= 3) {
            baseR = 9;           // 中节点
        } else if (deg >= 1) {
            baseR = 6;           // 小节点
        } else {
            baseR = 4;           // 极小节点
        }
        var radius = (n.is_core ? Math.max(baseR, 12) : baseR) * si;
        var isHovered = state.hoveredNode && state.hoveredNode.id === n.id;
        var isSelected = state.selectedNode && state.selectedNode.id === n.id;
        var isDimmed = (state.hoveredNode && !isHovered && !neighborSet[n.id]) ||
                       (state.hoveredEdge && !state.hoveredNode && !neighborSet[n.id]);

        // 核心节点微呼吸
        var pulse = 0;
        if (n.is_core && !isHovered && !isSelected) {
            pulse = Math.sin(state._time * 2 + n.id.length) * 0.12 + 0.12;
            radius += pulse * 3 * si;
        }

        if (isDimmed) ctx.globalAlpha = 0.18;

        // 光晕
        if (isHovered || isSelected) {
            ctx.save();
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius + 10 * si, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(245,158,11,0.15)';
            ctx.fill();
            ctx.restore();
        }

        // 绘制形状
        ctx.save();
        // 径向渐变填充
        var jurColor = getJurColor(n.jurisdiction);
        var grad = ctx.createRadialGradient(p.x - radius * 0.3, p.y - radius * 0.3, radius * 0.1, p.x, p.y, radius);
        grad.addColorStop(0, lightenColor(jurColor, 40));
        grad.addColorStop(1, jurColor);

        ctx.beginPath();
        if (et.shape === 'diamond') {
            ctx.moveTo(p.x, p.y - radius);
            ctx.lineTo(p.x + radius * 1.15, p.y);
            ctx.lineTo(p.x, p.y + radius);
            ctx.lineTo(p.x - radius * 1.15, p.y);
        } else if (et.shape === 'triangle') {
            var triH = radius * 1.15;
            ctx.moveTo(p.x, p.y - triH);
            ctx.lineTo(p.x + triH * 0.866, p.y + triH * 0.5);
            ctx.lineTo(p.x - triH * 0.866, p.y + triH * 0.5);
        } else if (et.shape === 'roundRect') {
            var rw = radius * 1.6, rh = radius * 1.1, rr = radius * 0.35;
            var hw = rw / 2, hh = rh / 2;
            ctx.moveTo(p.x + hw - rr, p.y - hh);
            ctx.lineTo(p.x - hw + rr, p.y - hh);
            ctx.arcTo(p.x - hw, p.y - hh, p.x - hw, p.y - hh + rr, rr);
            ctx.lineTo(p.x - hw, p.y + hh - rr);
            ctx.arcTo(p.x - hw, p.y + hh, p.x - hw + rr, p.y + hh, rr);
            ctx.lineTo(p.x + hw - rr, p.y + hh);
            ctx.arcTo(p.x + hw, p.y + hh, p.x + hw, p.y + hh - rr, rr);
            ctx.lineTo(p.x + hw, p.y - hh + rr);
            ctx.arcTo(p.x + hw, p.y - hh, p.x + hw - rr, p.y - hh, rr);
        } else {
            ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        }
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // 描边
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.lineWidth = (isSelected ? 3 : 1.8) * si;
        ctx.strokeStyle = isSelected ? '#F59E0B' : (isHovered ? '#FFFFFF' : 'rgba(255,255,255,0.85)');
        ctx.stroke();
        ctx.restore();

        // 选中高亮外圈
        if (isSelected) {
            ctx.save();
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius + 4 * si, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(245,158,11,0.4)';
            ctx.lineWidth = 2 * si;
            ctx.setLineDash([4 * si, 4 * si]);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        if (isDimmed) ctx.globalAlpha = 1;

        // 标签
        var showLabel = (state.labelMode === 'all') ||
            (state.labelMode === 'core' && (isHovered || isSelected || n.is_core)) ||
            (state.labelMode === 'smart' && (isHovered || isSelected || n.is_core || (n.degree || 0) >= degThreshold));
        if (state.labelMode === 'none') showLabel = isHovered || isSelected;

        if (showLabel) {
            ctx.save();
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.font = Math.max(8, 11 * si) + 'px "Inter", "Noto Sans SC", sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            var label = buildKGLabels(n);
            var maxLen = state.zoom >= 1.5 ? 24 : 12;
            var displayText = label.length > maxLen ? label.substring(0, maxLen) + '…' : label;
            var labelY = p.y + radius + 4 * si;

            // 标签阴影（提高可读性）
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.fillText(displayText, p.x + 0.8 * si, labelY + 0.8 * si);
            ctx.fillStyle = '#1F2937';
            ctx.fillText(displayText, p.x, labelY);
            ctx.restore();
        }
    });

    ctx.restore();
}

function buildKGLabels(n) {
    // 显示法规简称 + 第X条
    var law = '';
    if (n.law_name) {
        // 取法规名称中"法"之前的部分做简称，如"澳门个人资料保护法" → "个资保护法"
        var name = n.law_name.replace(/^[澳门香港新加坡美国欧盟]+\s*/g, '');
        law = name.length > 8 ? name.substring(0, 6) + '…' : name;
    }
    if (n.article_number) {
        if (law) return law + ' 第' + n.article_number + '条';
        return '第' + n.article_number + '条';
    }
    return law || (n.label || n.title || n.id || '').substring(0, 12);
}

// 构建关联线标签（简短显示）
function buildKGEdgeLabel(e, state) {
    var fromNode = state.nodes.find(function(n) { return n.id === e.from; });
    var toNode = state.nodes.find(function(n) { return n.id === e.to; });
    var fromLabel = fromNode ? (buildKGLabels(fromNode) || fromNode.id) : e.from;
    var toLabel = toNode ? (buildKGLabels(toNode) || toNode.id) : e.to;
    var isCross = e.type === 'cross_jurisdiction' || e.type === 'semantic_similar';

    if (isCross) {
        var sim = e.similarity != null ? e.similarity.toFixed(2) : '';
        return '语义关联' + (sim ? ' (' + sim + ')' : '');
    }
    return fromLabel + ' → ' + toLabel;
}

// 构建关联线详细信息
function buildKGEdgeDetail(e, state) {
    var fromNode = state.nodes.find(function(n) { return n.id === e.from; });
    var toNode = state.nodes.find(function(n) { return n.id === e.to; });
    var fromLabel = fromNode ? (buildKGLabels(fromNode) || fromNode.id) : e.from;
    var toLabel = toNode ? (buildKGLabels(toNode) || toNode.id) : e.to;
    var isCross = e.type === 'cross_jurisdiction' || e.type === 'semantic_similar';
    var sim = e.similarity != null ? e.similarity.toFixed(2) : '-';
    var fromJur = fromNode ? (fromNode.jurisdiction || '-') : '-';
    var toJur = toNode ? (toNode.jurisdiction || '-') : '-';
    var fromLaw = fromNode ? (fromNode.law_name || '-') : '-';
    var toLaw = toNode ? (toNode.law_name || '-') : '-';

    return {
        label: isCross ? '语义关联' : '法规内关联',
        fromNode: fromLabel,
        toNode: toLabel,
        fromJur: fromJur,
        toJur: toJur,
        fromLaw: fromLaw,
        toLaw: toLaw,
        type: isCross ? '跨法域语义关联' : '同法域法规内关联',
        similarity: sim,
        reason: e.reason || e.label || ''
    };
}

// ===== Canvas 交互 =====
function bindKGCanvas(canvas, container, state) {
    function getPos(evt) {
        var rect = canvas.getBoundingClientRect();
        return {
            x: (evt.clientX - rect.left - state.panX - rect.width / 2) / state.zoom + rect.width / 2,
            y: (evt.clientY - rect.top - state.panY - rect.height / 2) / state.zoom + rect.height / 2,
            clientX: evt.clientX, clientY: evt.clientY
        };
    }
    function findNode(pos) {
        for (var i = state.nodes.length - 1; i >= 0; i--) {
            var n = state.nodes[i], p = state.positions[n.id];
            if (!p) continue;
            var r = (8 + Math.min(6, (n.degree || 0) * 0.5)) / Math.max(state.zoom, 0.1);
            var dx = pos.x - p.x, dy = pos.y - p.y;
            if (dx * dx + dy * dy <= r * r) return n;
        }
        return null;
    }
    // 检测鼠标是否靠近某条关联线（沿贝塞尔曲线采样检测）
    function findEdge(pos) {
        var hitThreshold = 8 / Math.max(state.zoom, 0.1);
        var bestEdge = null, bestDist = hitThreshold;

        state.edges.forEach(function(e) {
            var u = state.positions[e.from], v = state.positions[e.to];
            if (!u || !v) return;
            var isCross = e.type === 'cross_jurisdiction' || e.type === 'semantic_similar';
            if (!state.filterCross && isCross) return;
            if (!state.filterSameLaw && !isCross) return;
            if (isCross && e.similarity != null && e.similarity < state.edgeThreshold) return;

            var midX = (u.x + v.x) / 2, midY = (u.y + v.y) / 2;
            var dx = v.x - u.x, dy = v.y - u.y;
            var dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
            var curv = isCross ? 0.12 : 0.04;
            var ctrlX = midX - (dy / dist) * dist * curv;
            var ctrlY = midY + (dx / dist) * dist * curv;

            // 采样 12 个点检测距离
            for (var t = 0; t <= 1; t += 0.083) {
                var mt = 1 - t;
                var px = mt * mt * u.x + 2 * mt * t * ctrlX + t * t * v.x;
                var py = mt * mt * u.y + 2 * mt * t * ctrlY + t * t * v.y;
                var ddx = pos.x - px, ddy = pos.y - py;
                var d = Math.sqrt(ddx * ddx + ddy * ddy);
                if (d < bestDist) {
                    bestDist = d;
                    bestEdge = e;
                }
            }
        });

        return bestEdge;
    }

    canvas.addEventListener('mousemove', function(e) {
        var pos = getPos(e);
        // 优先检测节点（节点在上层）
        var node = findNode(pos);
        var edge = null;
        if (!node) {
            edge = findEdge(pos);
        }
        state.hoveredNode = node;
        state.hoveredEdge = edge;
        canvas.style.cursor = (node || edge) ? 'pointer' : 'grab';

        var tooltip = document.getElementById('kg-tooltip');
        if (node && tooltip) {
            var et = ENTITY_DISPLAY[node.entity_type] || ENTITY_DISPLAY.Article;
            tooltip.style.display = 'block';
            tooltip.style.left = (pos.clientX - container.getBoundingClientRect().left + 12) + 'px';
            tooltip.style.top = (pos.clientY - container.getBoundingClientRect().top + 12) + 'px';
            tooltip.innerHTML = '<div style="font-weight:600;color:var(--color-text);margin-bottom:0.25rem;">' + escapeHtml(buildKGLabels(node)) + '</div>' +
                '<div style="color:var(--color-text-muted);margin-bottom:0.25rem;">' + escapeHtml(node.jurisdiction || '') + ' · ' + escapeHtml(node.law_name || '') + '</div>' +
                '<div style="color:var(--color-text-secondary);font-size:0.7rem;margin-bottom:0.25rem;">' +
                escapeHtml(et.label) + ' | 连接 ' + (node.degree || 0) +
                (node.is_core ? ' | 核心' : '') +
                (node.primary_theme ? ' | ' + node.primary_theme : '') +
                '</div>' +
                '<div style="color:var(--color-text-secondary);line-height:1.5;">' +
                escapeHtml(String(node.content_preview || '').substring(0, 120)) + '</div>';
        } else if (edge && tooltip) {
            var info = buildKGEdgeDetail(edge, state);
            tooltip.style.display = 'block';
            tooltip.style.left = (pos.clientX - container.getBoundingClientRect().left + 12) + 'px';
            tooltip.style.top = (pos.clientY - container.getBoundingClientRect().top + 12) + 'px';
            tooltip.innerHTML = '<div style="font-weight:600;color:var(--color-text);margin-bottom:0.25rem;">' + escapeHtml(info.label) + '</div>' +
                '<div style="color:var(--color-text-secondary);font-size:0.8125rem;margin-bottom:0.25rem;">' +
                escapeHtml(info.fromNode) + ' <span style="color:var(--color-text-muted);">→</span> ' + escapeHtml(info.toNode) +
                '</div>' +
                '<div style="color:var(--color-text-muted);font-size:0.75rem;">' +
                escapeHtml(info.fromJur) + ' · ' + escapeHtml(info.toJur) +
                ' | 相似度 ' + info.similarity +
                '</div>' +
                (info.reason ? '<div style="color:var(--color-text-secondary);font-size:0.75rem;margin-top:0.125rem;">' + escapeHtml(String(info.reason).substring(0, 100)) + '</div>' : '');
        } else if (tooltip) {
            tooltip.style.display = 'none';
        }
    });

    canvas.addEventListener('mouseleave', function() {
        state.hoveredNode = null;
        state.hoveredEdge = null;
        var tooltip = document.getElementById('kg-tooltip');
        if (tooltip) tooltip.style.display = 'none';
    });

    canvas.addEventListener('mousedown', function(e) {
        var pos = getPos(e);
        var node = findNode(pos);
        if (node) {
            state.selectedNode = node;
            state.selectedEdge = null;
            state.draggedNode = node;
            var p = state.positions[node.id];
            state._dragOffX = p.x - pos.x;
            state._dragOffY = p.y - pos.y;
            showKGDetail(node);
            return;
        }
        // 没点到节点，检测边
        var edge = findEdge(pos);
        if (edge) {
            state.selectedEdge = edge;
            state.selectedNode = null;
            state.draggedNode = null;
            showKGEdgeDetail(edge, state);
            return;
        }
        state.selectedNode = null;
        state.selectedEdge = null;
        document.getElementById('kg-detail-panel').style.display = 'none';
    });

    window.addEventListener('mousemove', function(e) {
        if (!state.draggedNode) return;
        var rect = canvas.getBoundingClientRect();
        var pos = {
            x: (e.clientX - rect.left - state.panX - rect.width / 2) / state.zoom + rect.width / 2,
            y: (e.clientY - rect.top - state.panY - rect.height / 2) / state.zoom + rect.height / 2
        };
        state.positions[state.draggedNode.id].x = pos.x + state._dragOffX || 0;
        state.positions[state.draggedNode.id].y = pos.y + state._dragOffY || 0;
    });
    window.addEventListener('mouseup', function() { state.draggedNode = null; });

    canvas.addEventListener('wheel', function(e) {
        e.preventDefault();
        state.zoom = Math.max(0.3, Math.min(3, state.zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
    }, { passive: false });
}

// ===== 详情面板（节点） =====
function showKGDetail(node) {
    var panel = document.getElementById('kg-detail-panel');
    var et = ENTITY_DISPLAY[node.entity_type] || ENTITY_DISPLAY.Article;
    var html = '<div class="card" style="padding:0.75rem 1rem;">';
    html += '<div style="margin-bottom:0.5rem;">';
    html += '<span class="badge" style="background:#3B82F6;color:#fff;">' + escapeHtml(node.jurisdiction || '') + '</span>';
    html += '<span class="badge" style="margin-left:0.375rem;background:#6B7280;color:#fff;">' + escapeHtml(et.label) + '</span>';
    if (node.is_core) html += '<span class="badge badge-warning" style="margin-left:0.375rem;">核心</span>';
    html += '</div>';
    html += '<h3 style="font-size:1rem;font-weight:600;margin:0 0 0.5rem;color:var(--color-text);">' + escapeHtml(buildKGLabels(node)) + '</h3>';
    if (node.law_name) html += '<div style="margin-bottom:0.25rem;font-size:0.8125rem;"><strong>法规：</strong>' + escapeHtml(node.law_name) + '</div>';
    if (node.article_number) html += '<div style="margin-bottom:0.25rem;font-size:0.8125rem;"><strong>条款号：</strong>' + escapeHtml(node.article_number) + '</div>';
    html += '<div style="margin-bottom:0.25rem;font-size:0.8125rem;"><strong>连接度：</strong>' + (node.degree || 0) + '</div>';
    if (node.primary_theme) html += '<div style="margin-bottom:0.25rem;font-size:0.8125rem;"><strong>主题：</strong>' + escapeHtml(node.primary_theme) + '</div>';
    html += '<div style="margin-top:0.5rem;padding:0.5rem;background:var(--color-bg);border-radius:4px;line-height:1.6;white-space:pre-wrap;font-size:0.8125rem;max-height:200px;overflow-y:auto;">' + escapeHtml(String(node.content_preview || '').substring(0, 500)) + '</div>';
    html += '</div>';
    panel.innerHTML = html;
    panel.style.display = 'block';
}

// ===== 详情面板（关联线） =====
function showKGEdgeDetail(edge, state) {
    var panel = document.getElementById('kg-detail-panel');
    var info = buildKGEdgeDetail(edge, state);
    var html = '<div class="card" style="padding:0.75rem 1rem;">';
    html += '<div style="margin-bottom:0.5rem;">';
    html += '<span class="badge badge-info" style="background:#3B82F6;color:#fff;">' + escapeHtml(info.type) + '</span>';
    if (info.similarity !== '-') {
        html += '<span class="badge badge-warning" style="margin-left:0.375rem;background:#D97706;color:#fff;">相似度 ' + info.similarity + '</span>';
    }
    html += '</div>';
    html += '<h3 style="font-size:1rem;font-weight:600;margin:0 0 0.75rem;color:var(--color-text);">' + escapeHtml(info.label) + '</h3>';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.8125rem;">';

    // 起点节点
    html += '<div style="padding:0.5rem;background:var(--color-bg);border-radius:6px;">';
    html += '<div style="font-weight:600;color:var(--color-text-muted);font-size:0.7rem;margin-bottom:0.25rem;">起点节点</div>';
    html += '<div style="color:var(--color-text);font-weight:500;">' + escapeHtml(info.fromNode) + '</div>';
    html += '<div style="color:var(--color-text-muted);font-size:0.75rem;">' + escapeHtml(info.fromJur) + '</div>';
    html += '<div style="color:var(--color-text-muted);font-size:0.75rem;word-break:break-all;">' + escapeHtml(String(info.fromLaw).substring(0, 40)) + '</div>';
    html += '</div>';

    // 终点节点
    html += '<div style="padding:0.5rem;background:var(--color-bg);border-radius:6px;">';
    html += '<div style="font-weight:600;color:var(--color-text-muted);font-size:0.7rem;margin-bottom:0.25rem;">终点节点</div>';
    html += '<div style="color:var(--color-text);font-weight:500;">' + escapeHtml(info.toNode) + '</div>';
    html += '<div style="color:var(--color-text-muted);font-size:0.75rem;">' + escapeHtml(info.toJur) + '</div>';
    html += '<div style="color:var(--color-text-muted);font-size:0.75rem;word-break:break-all;">' + escapeHtml(String(info.toLaw).substring(0, 40)) + '</div>';
    html += '</div>';

    html += '</div>';

    // 关联理由
    if (info.reason) {
        html += '<div style="margin-top:0.5rem;padding:0.5rem;background:var(--color-warning-bg);border-radius:6px;">';
        html += '<div style="font-weight:600;color:var(--color-warning);font-size:0.75rem;margin-bottom:0.25rem;">关联说明</div>';
        html += '<div style="color:var(--color-text-secondary);font-size:0.8125rem;line-height:1.6;">' + escapeHtml(info.reason) + '</div>';
        html += '</div>';
    }

    html += '</div>';
    panel.innerHTML = html;
    panel.style.display = 'block';
}

// ===== 控制绑定 =====
function bindKGControls(state) {
    function sync() {
        state.filterSameLaw = document.getElementById('kg-filter-same-law').checked;
        state.filterCross = document.getElementById('kg-filter-cross').checked;
        state.edgeThreshold = parseFloat(document.getElementById('kg-edge-threshold').value);
        state.labelMode = document.getElementById('kg-label-mode').value;
        document.getElementById('kg-threshold-val').textContent = state.edgeThreshold.toFixed(2);
    }
    document.getElementById('kg-filter-same-law').addEventListener('change', sync);
    document.getElementById('kg-filter-cross').addEventListener('change', sync);
    document.getElementById('kg-edge-threshold').addEventListener('input', sync);
    document.getElementById('kg-label-mode').addEventListener('change', sync);
    sync();
}

window.resetKGView = function() {
    var state = window._kgState;
    if (!state) return;
    state.zoom = 1; state.panX = 0; state.panY = 0;
};

window.fitKGView = function() {
    var state = window._kgState;
    if (!state || state.nodes.length === 0) return;
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    state.nodes.forEach(function(n) { var p = state.positions[n.id]; if (p) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); } });
    if (!isFinite(minX)) return;
    var w = document.getElementById('kg-canvas-container').clientWidth;
    var h = document.getElementById('kg-canvas-container').clientHeight;
    var gw = maxX - minX + 60, gh = maxY - minY + 60;
    state.zoom = Math.min(w / gw, h / gh, 2);
    state.panX = (w / 2 - (minX + maxX) / 2) * state.zoom;
    state.panY = (h / 2 - (minY + maxY) / 2) * state.zoom;
};

// ===== 智能洞察 =====
function generateKGInsight(data) {
    var nodes = data.nodes || [];
    var edges = data.edges || [];
    if (nodes.length === 0) { document.getElementById('kg-insight-panel').style.display = 'none'; return; }

    // 统计高连接节点
    var highDeg = nodes.filter(function(n) { return (n.degree || 0) >= 8; });
    var midDeg = nodes.filter(function(n) { var d = n.degree || 0; return d >= 3 && d < 8; });
    var lowDeg = nodes.filter(function(n) { var d = n.degree || 0; return d >= 1 && d < 3; });
    var zeroDeg = nodes.filter(function(n) { return (n.degree || 0) === 0; });

    // 统计法域
    var jurMap = {};
    nodes.forEach(function(n) {
        var j = n.jurisdiction || '未知';
        if (!jurMap[j]) jurMap[j] = 0;
        jurMap[j]++;
    });
    var jurList = Object.keys(jurMap).sort(function(a,b) { return jurMap[b] - jurMap[a]; });

    // 统计边类型
    var crossEdges = edges.filter(function(e) { return e.type === 'cross_jurisdiction' || e.type === 'semantic_similar'; });

    // 核心节点名称
    var coreNodes = nodes.filter(function(n) { return n.is_core || (n.degree || 0) >= 8; });

    // 生成洞察文本
    var parts = [];

    // 规模概览
    parts.push('该图谱包含 ' + nodes.length + ' 个节点、' + edges.length + ' 条边');

    // 连接度分布
    if (highDeg.length > 0) {
        parts.push('高连接节点 ' + highDeg.length + ' 个（核心枢纽）');
    }
    if (midDeg.length > 0) {
        parts.push('中连接节点 ' + midDeg.length + ' 个');
    }
    if (lowDeg.length > 0) {
        parts.push('低连接节点 ' + lowDeg.length + ' 个');
    }
    if (zeroDeg.length > 0 && zeroDeg.length < nodes.length) {
        parts.push('孤立节点 ' + zeroDeg.length + ' 个');
    }

    // 法域覆盖
    if (jurList.length === 1) {
        parts.push('涉及 ' + jurList[0] + ' 单一法域');
    } else {
        parts.push('涉及 ' + jurList.length + ' 个法域（' + jurList.join('、') + '）');
    }

    // 边分布
    if (crossEdges.length > 0) {
        parts.push('其中跨法域关联 ' + crossEdges.length + ' 条');
    }

    // 核心条款
    if (coreNodes.length > 0) {
        var coreLabels = coreNodes.slice(0, 5).map(function(n) { return buildKGLabels(n); });
        var text = '核心条款：' + coreLabels.join('、');
        if (coreNodes.length > 5) text += ' 等';
        parts.push(text);
    }

    document.getElementById('kg-insight-text').textContent = parts.join('，') + '。';
    document.getElementById('kg-insight-panel').style.display = 'block';
}

// ===== 加载法规视图 =====
window.loadLawGraph = function() {
    var lawId = document.getElementById('kg-law-select').value;
    if (!lawId) return;
    document.getElementById('kg-graph-title').textContent = '加载中...';
    api.get('/kg/law/' + encodeURIComponent(lawId)).then(function(resp) {
        if (resp.success && resp.data) {
            var name = (resp.data.nodes || []).find(function(n) { return n.law_id === lawId; });
            renderKG(resp.data, (name && name.law_name) || lawId, window._kgLayoutMode);
        } else {
            document.getElementById('kg-graph-title').textContent = '数据加载失败';
        }
    }).catch(function() {
        document.getElementById('kg-graph-title').textContent = '请求失败';
    });
};

// ===== 搜索实体 =====
window.searchKGEntities = function() {
    var q = document.getElementById('kg-search-input').value.trim();
    if (!q) return;
    document.getElementById('kg-graph-title').textContent = '搜索中...';
    api.get('/kg/search-entity', { q: q, page_size: 20 }).then(function(resp) {
        if (resp.success && resp.data) {
            renderKG(resp.data, '实体搜索: ' + q);
        } else {
            document.getElementById('kg-graph-title').textContent = '未找到相关结果';
            document.getElementById('kg-search-results').innerHTML = '<div class="alert alert-info">未找到相关结果，请尝试其他关键词</div>';
        }
    }).catch(function() {
        document.getElementById('kg-graph-title').textContent = '搜索请求失败';
    });
};

// ===== 跨法域对比 =====
window.loadCrossGraph = function() {
    var jurA = document.getElementById('kg-cross-jur-a').value;
    var jurB = document.getElementById('kg-cross-jur-b').value;
    var topic = document.getElementById('kg-cross-topic').value;
    if (!jurA || !jurB || !topic) return;
    document.getElementById('kg-graph-title').textContent = '加载中...';
    api.get('/structured-analysis/network-graph', {
        topic_id: topic, jurisdiction_a: jurA, jurisdiction_b: jurB, top_n: 10
    }).then(function(resp) {
        if (resp.success && resp.data) {
            // 跨法域对比：强制两个法域使用对比色（一暖一冷），确保视觉可分辨
            var CROSS_COLORS = ['#DC2626', '#2563EB', '#D97706', '#059669', '#7C3AED', '#0891B2', '#EA580C', '#6366F1'];
            var jurSet = {};
            (resp.data.nodes || []).forEach(function(n) {
                var j = n.jurisdiction || '';
                if (j) jurSet[j] = true;
            });
            var jurList = Object.keys(jurSet).sort();
            // 清除缓存，按法域数量分配差异色
            jurList.forEach(function(jur, idx) {
                delete JUR_COLOR_CACHE[jur];
                JUR_COLOR_CACHE[jur] = CROSS_COLORS[idx % CROSS_COLORS.length];
            });
            renderKG(resp.data, jurA + ' ↔ ' + jurB + ' 跨法域对齐');
        } else {
            document.getElementById('kg-graph-title').textContent = '暂无跨法域对齐数据';
        }
    }).catch(function() {
        document.getElementById('kg-graph-title').textContent = '对比请求失败';
    });
};

// ===== 布局切换 =====
// 默认力导向布局（undefined），可切换为圆形布局（'circular'）
window._kgLayoutMode = undefined;

window.toggleKGLayout = function() {
    window._kgLayoutMode = window._kgLayoutMode === 'circular' ? undefined : 'circular';
    var btn = document.getElementById('kg-layout-btn');
    if (btn) {
        btn.textContent = window._kgLayoutMode === 'circular' ? '布局：圆形' : '布局：力导向';
    }
    // 如果有已渲染的图谱，重新布局
    var state = window._kgState;
    if (state) {
        var title = document.getElementById('kg-graph-title').textContent;
        // 用当前 data 重新渲染，data 存在 _kgData 中
        var data = window._kgLastData;
        if (data) renderKG(data, title, window._kgLayoutMode);
    }
};

// 保存最近一次渲染的 data，方便布局切换时复用
var _origRenderKG = renderKG;
renderKG = function(data, title, layoutMode) {
    window._kgLastData = data;
    _origRenderKG(data, title, layoutMode);
};
