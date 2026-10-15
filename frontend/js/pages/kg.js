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

    // 预设高频法域固定颜色（统一为中等饱和度的柔和色，避免纯红纯蓝造成的视觉疲劳）
    var PREDEFINED = {
        '澳门': '#4C7DF0', '中国澳门': '#4C7DF0', 'Macau': '#4C7DF0',
        '香港': '#E0605E', '中国香港': '#E0605E', 'Hong Kong': '#E0605E',
        '新加坡': '#E0A03A', 'Singapore': '#E0A03A',
        '美国': '#8B7BE8', 'USA': '#8B7BE8', 'US': '#8B7BE8',
        '欧盟': '#3EAF9E', 'EU': '#3EAF9E',
        '英国': '#E2668A', 'UK': '#E2668A', 'United Kingdom': '#E2668A',
        '日本': '#D46A85', 'Japan': '#D46A85',
        '韩国': '#5B8DEF', 'South Korea': '#5B8DEF', 'Korea': '#5B8DEF',
        '德国': '#E8B04B', 'Germany': '#E8B04B',
        '法国': '#8E93E8', 'France': '#8E93E8',
        '印度': '#E58A4A', 'India': '#E58A4A',
        '澳大利亚': '#4AA8C4', 'Australia': '#4AA8C4',
        '加拿大': '#48B584', 'Canada': '#48B584',
        '巴西': '#3E9AB0', 'Brazil': '#3E9AB0',
        '俄罗斯': '#7C83E0', 'Russia': '#7C83E0',
        '中国': '#B5705A', 'China': '#B5705A', '中国大陆': '#B5705A',
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
    // 饱和度 45-58%、明度 56-66%：中等饱和、偏明亮，观感柔和且互相可区分
    var sat = 45 + (Math.abs(hash) % 14);
    var lig = 56 + (Math.abs(hash) % 11);
    var color = 'hsl(' + Math.round(hue) + ', ' + sat + '%, ' + lig + '%)';

    JUR_COLOR_CACHE[jurisdiction] = color;
    return color;
}

// 动态更新图例（按当前着色依据：法域 / 社区）
function updateKGLegend(nodes, communities, mode) {
    var jurRow = document.getElementById('kg-legend-jur');
    if (!jurRow) return;

    // ---- 按社区着色 ----
    if (mode === 'community') {
        var commMap = {};
        nodes.forEach(function(n) {
            var cid = (n.community_id === undefined || n.community_id === null) ? -1 : n.community_id;
            if (cid < 0) return;
            if (!commMap[cid]) commMap[cid] = 0;
            commMap[cid]++;
        });
        var cids = Object.keys(commMap).map(Number).sort(function(a, b) { return commMap[b] - commMap[a]; });
        if (cids.length === 0) return;
        var commHtml = cids.slice(0, 12).map(function(cid) {
            var color = getCommunityColor(cid);
            return '<span><span class="kg-legend-icon" style="display:inline-block;width:8px;height:8px;background:' + color + ';border-radius:50%;"></span> ' + escapeHtml(getCommunityLabel(cid, communities)) + '</span>';
        }).join('');
        jurRow.innerHTML = '<span style="font-weight:600;min-width:3em;">社区：</span>' + commHtml;
        return;
    }

    // ---- 默认：按法域着色 ----
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
    jurRow.innerHTML = '<span style="font-weight:600;min-width:3em;">法域：</span>' + jurHtml;
}

// 图例折叠/展开
window.toggleKGLegend = function() {
    var body = document.getElementById('kg-legend-body');
    var toggle = document.getElementById('kg-legend-toggle');
    if (!body) return;
    var collapsed = body.style.display === 'none';
    body.style.display = collapsed ? 'block' : 'none';
    if (toggle) toggle.textContent = collapsed ? '▾' : '▸';
};

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

// 社区配色：Louvain 簇按固定调色板循环取色，保证簇间差异明显且可复现
var COMMUNITY_PALETTE = ['#2563EB', '#DC2626', '#D97706', '#059669', '#7C3AED',
    '#0891B2', '#EA580C', '#BE123C', '#4F46E5', '#0D9488', '#65A30D', '#DB2777'];
function getCommunityColor(cid) {
    if (cid === undefined || cid === null || cid < 0) return '#9CA3AF';
    return COMMUNITY_PALETTE[cid % COMMUNITY_PALETTE.length];
}
function getCommunityLabel(cid, communities) {
    if (communities) {
        for (var i = 0; i < communities.length; i++) {
            if (communities[i].community_id === cid) return communities[i].label || ('簇 ' + cid);
        }
    }
    return '簇 ' + cid;
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
                <button class="kg-mode-btn" data-mode="concept" onclick="switchKGMode('concept')">概念对比</button>
                <button class="kg-mode-btn" data-mode="reasoning" onclick="switchKGMode('reasoning')">图谱问答</button>
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
                    <div style="display:flex;align-items:flex-end;gap:0.75rem;">
                        <div>
                            <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">视图</label>
                            <select id="kg-cross-view" class="select" style="width:auto;font-size:0.8125rem;">
                                <option value="matrix" selected>对齐矩阵（法规级，推荐）</option>
                                <option value="graph">关系图谱（条款级）</option>
                            </select>
                        </div>
                        <button class="btn btn-primary" onclick="loadCrossGraph()">开始对比</button>
                    </div>
                </div>
                <div style="font-size:0.75rem;color:var(--color-text-muted);margin-top:0.375rem;">对齐矩阵：行/列=两法域法规，格值=语义对口的条款对数量，点击格子查看具体条款对</div>
            </div>
            <!-- 法规级对齐矩阵 -->
            <div class="card" id="kg-cross-matrix-card" style="margin-top:0.75rem;padding:0.5rem;">
                <div id="kg-cross-matrix" style="width:100%;height:460px;"></div>
                <div id="kg-cross-matrix-fallback"></div>
            </div>
            <div id="kg-cross-detail" style="margin-top:0.5rem;"></div>
        </div>

        <!-- 模式4：概念对比（跨法域概念对齐矩阵） -->
        <div id="kg-concept-panel" class="kg-panel" style="display:none;">
            <div class="card" style="padding:1rem;">
                <div style="display:flex;gap:0.75rem;align-items:flex-end;flex-wrap:wrap;">
                    <div>
                        <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">最少法域数</label>
                        <select id="kg-concept-minjur" class="select" style="width:auto;font-size:0.8125rem;">
                            <option value="2">≥2 个法域</option>
                            <option value="3">≥3 个法域</option>
                            <option value="1">≥1 个法域</option>
                        </select>
                    </div>
                    <div>
                        <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">概念数量</label>
                        <select id="kg-concept-topn" class="select" style="width:auto;font-size:0.8125rem;">
                            <option value="20">Top 20</option>
                            <option value="30" selected>Top 30</option>
                            <option value="50">Top 50</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" onclick="loadConceptMatrix()">生成矩阵</button>
                </div>
                <div style="font-size:0.75rem;color:var(--color-text-muted);margin-top:0.375rem;">热力图：行=法律概念，列=法域，色深=相关条文数；点击任意概念查看各法域的法规与条款明细</div>
            </div>
            <div class="card" style="margin-top:0.75rem;padding:0.5rem;">
                <div id="kg-concept-heatmap" style="width:100%;height:480px;"></div>
                <div id="kg-concept-fallback"></div>
            </div>
            <div id="kg-concept-detail" style="margin-top:0.5rem;"></div>
        </div>

        <!-- 模式5：图谱问答（多跳推理 + 路径证据） -->
        <div id="kg-reasoning-panel" class="kg-panel" style="display:none;">
            <div class="card" style="padding:1rem;">
                <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.375rem;">图谱问答（基于实体关系多跳推理，答案附路径证据）</label>
                <div style="display:flex;gap:0.5rem;">
                    <input type="text" id="kg-reasoning-input" class="input" placeholder="例如：个人数据的跨境传输在各法域如何规定？" style="flex:1;" onkeydown="if(event.key==='Enter')runKGReasoning()">
                    <button class="btn btn-primary" onclick="runKGReasoning()">提问</button>
                </div>
                <div style="font-size:0.75rem;color:var(--color-text-muted);margin-top:0.375rem;">仅对已完成知识层抽取的法规有效；无命中时会提示（可改用普通问答）。答案可溯源到关系路径。</div>
                <div id="kg-reasoning-result" style="margin-top:0.75rem;"></div>
            </div>
        </div>

        <!-- 图谱容器 -->
        <div class="card" id="kg-graph-card" style="margin-top:0.75rem;padding:0;">
            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.625rem 1rem;border-bottom:1px solid var(--color-border);">
                <div id="kg-graph-title" class="kg-graph-title" style="font-weight:600;font-size:0.8125rem;color:var(--color-text);"></div>
                <div id="kg-view-controls" style="display:flex;gap:0.375rem;">
                    <button class="btn btn-sm" onclick="toggleKGLayout()" id="kg-layout-btn" style="font-size:0.75rem;">布局：力导向</button>
                    <button class="btn btn-sm" onclick="resetKGView()" style="font-size:0.75rem;">重置视图</button>
                    <button class="btn btn-sm" onclick="fitKGView()" style="font-size:0.75rem;">适应布局</button>
                </div>
            </div>
            <div id="kg-canvas-container" style="position:relative;height:500px;background:var(--color-bg);overflow:hidden;cursor:grab;">
                <canvas id="kg-canvas" style="width:100%;height:100%;"></canvas>
                <div id="kg-echarts" style="width:100%;height:100%;display:none;"></div>
                <div id="kg-tooltip" style="position:absolute;display:none;background:rgba(255,255,255,0.96);border:1px solid var(--color-border);border-radius:6px;padding:0.5rem 0.75rem;font-size:0.75rem;max-width:280px;box-shadow:0 4px 12px rgba(0,0,0,0.08);pointer-events:none;z-index:10;"></div>
                <!-- 图例（右上角，可折叠，避免遮挡节点） -->
                <div id="kg-legend" style="position:absolute;top:6px;right:6px;background:rgba(255,255,255,0.95);border:1px solid var(--color-border);border-radius:6px;font-size:0.65rem;color:var(--color-text-secondary);z-index:6;line-height:1.7;max-width:270px;max-height:calc(100% - 20px);overflow:auto;">
                    <div id="kg-legend-header" onclick="toggleKGLegend()" style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;padding:4px 9px;cursor:pointer;border-bottom:1px solid var(--color-border);font-weight:600;user-select:none;">
                        <span>图例</span><span id="kg-legend-toggle">▾</span>
                    </div>
                    <div id="kg-legend-body" style="padding:5px 9px;">
                    <div id="kg-legend-jur" style="display:flex;flex-wrap:wrap;gap:0.2rem 0.6rem;">
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
                    引擎
                    <select id="kg-render-engine" class="select" style="width:auto;font-size:0.75rem;">
                        <option value="echarts" selected>ECharts</option>
                        <option value="canvas">Canvas（自绘）</option>
                    </select>
                </label>
                <label style="display:flex;align-items:center;gap:0.25rem;color:var(--color-text-secondary);">
                    着色
                    <select id="kg-color-mode" class="select" style="width:auto;font-size:0.75rem;">
                        <option value="jur">按法域</option>
                        <option value="community">按社区</option>
                    </select>
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

    // 渲染引擎切换：用最近一次数据重绘（ECharts ↔ Canvas）
    var engineSel = document.getElementById('kg-render-engine');
    if (engineSel) {
        engineSel.addEventListener('change', function() {
            if (window._kgLastData) {
                renderKG(
                    window._kgLastData,
                    document.getElementById('kg-graph-title').textContent,
                    window._kgLayoutMode
                );
            }
        });
    }

    // 控制栏（同法规/跨域/阈值/标签）统一绑定一次（两种引擎共用）
    bindKGControlsOnce();

    // 着色依据切换：Canvas 模式刷新图例；ECharts 模式重绘以重新配色
    var colorSel = document.getElementById('kg-color-mode');
    if (colorSel) {
        colorSel.addEventListener('change', function() {
            var mode = colorSel.value;
            if (window._kgState) {
                window._kgState.colorMode = mode;
                updateKGLegend(window._kgState.nodes, window._kgState.communities, mode);
            }
            if (getKGRenderEngine() !== 'canvas' && window._kgLastData) {
                renderKG(
                    window._kgLastData,
                    document.getElementById('kg-graph-title').textContent,
                    window._kgLayoutMode
                );
            }
        });
    }
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
    // 跨法域模式可能处于"对齐矩阵"视图（此时隐藏图谱卡片）：切换模式时恢复图谱卡片
    var graphCard = document.getElementById('kg-graph-card');
    var matrixCard = document.getElementById('kg-cross-matrix-card');
    var crossView = document.getElementById('kg-cross-view');
    var inMatrix = (mode === 'cross' && crossView && crossView.value === 'matrix');
    if (graphCard) graphCard.style.display = inMatrix ? 'none' : 'block';
    if (matrixCard) matrixCard.style.display = inMatrix ? 'block' : 'none';
    // 首次进入概念对比时自动加载矩阵
    if (mode === 'concept' && !window._kgConceptLoaded) {
        window._kgConceptLoaded = true;
        loadConceptMatrix();
    }
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

    // 着色依据（法域 / 社区）与社区图例
    var colorMode = document.getElementById('kg-color-mode') ? document.getElementById('kg-color-mode').value : 'jur';
    var communities = data.communities || [];
    updateKGLegend(nodes, communities, colorMode);

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

    // 渲染引擎分流：ECharts 走独立渲染路径；Canvas 为降级备选
    var engine = getKGRenderEngine();
    var legendEl = document.getElementById('kg-legend');
    var viewControls = document.getElementById('kg-view-controls');
    if (engine !== 'canvas') {
        // ECharts 自带图例/配色与漫游，隐藏 HTML 图例与 Canvas 专有视图按钮
        if (legendEl) legendEl.style.display = 'none';
        if (viewControls) viewControls.style.display = 'none';
        generateKGInsight(data);
        renderKGWithECharts(data, title);
        return;
    }
    if (legendEl) legendEl.style.display = 'block';
    if (viewControls) viewControls.style.display = 'flex';
    var canvasEl = document.getElementById('kg-canvas');
    var echartEl = document.getElementById('kg-echarts');
    if (canvasEl) canvasEl.style.display = 'block';
    if (echartEl) echartEl.style.display = 'none';
    if (window._kgEchartsInstance) {
        window._kgEchartsInstance.dispose();
        window._kgEchartsInstance = null;
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

    var prMax = 0;
    nodes.forEach(function(n) { if (n.pagerank && n.pagerank > prMax) prMax = n.pagerank; });

    var state = {
        nodes: nodes, edges: edges, positions: positions, velocities: velocities,
        zoom: 1, panX: 0, panY: 0,
        hoveredNode: null, selectedNode: null, draggedNode: null,
        hoveredEdge: null, selectedEdge: null,
        filterSameLaw: true, filterCross: true,
        edgeThreshold: parseFloat(document.getElementById('kg-edge-threshold').value),
        labelMode: document.getElementById('kg-label-mode').value,
        colorMode: colorMode, communities: communities, prMax: prMax,
        layoutTick: 0, maxLayoutTicks: isCircular ? 3 : 300, running: true,
        isCircular: isCircular,
        _time: 0  // 用于呼吸动画
    };
    window._kgState = state;
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
        if (n.pagerank != null && state.prMax > 0) {
            // 有 PageRank：按相对影响力定大小（开方压缩，避免极端值撑爆）
            var prT = Math.min(1, n.pagerank / state.prMax);
            baseR = 4.5 + Math.sqrt(prT) * 7.5;   // 4.5 ~ 12
        } else if (deg >= 8) {
            baseR = 11.5;        // 大节点（核心枢纽）
        } else if (deg >= 3) {
            baseR = 8.5;         // 中节点
        } else if (deg >= 1) {
            baseR = 5.5;         // 小节点
        } else {
            baseR = 3.8;         // 极小节点
        }
        var radius = (n.is_core ? Math.max(baseR, 11) : baseR) * si;
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

        var jurColor = (state.colorMode === 'community')
            ? getCommunityColor(n.community_id)
            : getJurColor(n.jurisdiction);

        // 柔和光晕：为每个节点铺一层低透明度外圈，增加层次感（被打断高亮时不画）
        if (!isDimmed) {
            ctx.save();
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius + 4.5 * si, 0, Math.PI * 2);
            ctx.fillStyle = jurColor;
            ctx.globalAlpha = 0.16;
            ctx.fill();
            ctx.restore();
        }

        // 绘制形状
        ctx.save();
        // 径向渐变填充
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
            var maxLen = state.zoom >= 1.5 ? 44 : 28;
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

// 法规名智能缩写：既要短，又要让人一眼认出是哪部法规
// - 去法域前缀与英文赘述（"Act on the…"）
// - 英文按整词截断、中文按字数截断，尽量保留可辨识的主体词
function shortenLawName(name, maxLen) {
    if (!name) return '';
    maxLen = maxLen || 16;
    var s = String(name).trim();

    // 去法域前缀（用多字词交替，避免字符类误删单字，如"法人登记"的"法"）
    s = s.replace(
        /^[\s【\[]*(?:中国大陆|澳大利亚|马来西亚|新加坡|港澳|澳门|香港|美国|欧盟|日本|泰国|越南|臺灣|台湾|韩国|韓國|英国|德国|法国|印度|俄罗斯|加拿大|巴西|中国)?[\s】\]]*/,
        ''
    );
    // 去英文常见赘述前缀
    s = s.replace(/^(the\s+)?act\s+on\s+(the\s+)?/i, '');
    s = s.replace(/^(the|a|an)\s+/i, '');
    s = s.trim();
    if (!s) return '';

    // 中文（含中英混排）按字符截断
    if (/[\u4e00-\u9fff]/.test(s)) {
        return s.length > maxLen ? s.substring(0, maxLen) + '…' : s;
    }

    // 纯英文：按整词截断，避免把单词切一半
    var words = s.split(/\s+/);
    var out = '';
    for (var i = 0; i < words.length; i++) {
        var w = words[i];
        if (out && (out.length + 1 + w.length) > maxLen) break;
        out = out ? out + ' ' + w : w;
        if (out.length >= maxLen) break;
    }
    if (!out) out = s.substring(0, maxLen);
    return out.length < s.length ? out + '…' : out;
}

function buildKGLabels(n) {
    // 实体节点（LLM 知识层）：直接显示实体名
    if (n.node_kind === 'entity') {
        return n.label || n.title || n.id || '';
    }
    // 显示法规简称 + 第X条（法规简称保留可辨识的主体词；完整名称见 tooltip/详情）
    var law = shortenLawName(n.law_name || '', 24);
    if (n.article_number) {
        if (law) return law + ' 第' + n.article_number + '条';
        return '第' + n.article_number + '条';
    }
    return law || shortenLawName(n.label || n.title || n.id || '', 24);
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
    var isEntityRel = e.edge_kind === 'entity_relation';
    var sim = e.similarity != null ? e.similarity.toFixed(2) : '-';
    var fromJur = fromNode ? (fromNode.jurisdiction || '-') : '-';
    var toJur = toNode ? (toNode.jurisdiction || '-') : '-';
    var fromLaw = fromNode ? (fromNode.law_name || '-') : '-';
    var toLaw = toNode ? (toNode.law_name || '-') : '-';

    return {
        label: isEntityRel ? (e.relation_label || '实体关系') : (isCross ? '语义关联' : '法规内关联'),
        fromNode: fromLabel,
        toNode: toLabel,
        fromJur: fromJur,
        toJur: toJur,
        fromLaw: fromLaw,
        toLaw: toLaw,
        type: isEntityRel ? '实体关系' : (isCross ? '跨法域语义关联' : '同法域法规内关联'),
        similarity: sim,
        similarityLabel: isEntityRel ? '置信度' : '相似度',
        reason: e.reason || (isEntityRel
            ? (e.source_article ? '来源条款：' + e.source_article : '')
            : (e.label || ''))
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
                ' | ' + escapeHtml(info.similarityLabel || '相似度') + ' ' + info.similarity +
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
        html += '<span class="badge badge-warning" style="margin-left:0.375rem;background:#D97706;color:#fff;">' + escapeHtml(info.similarityLabel || '相似度') + ' ' + info.similarity + '</span>';
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
// 控制栏统一绑定（页面初始化时只绑一次，避免每次重渲染重复挂载监听）
// Canvas 模式：更新窗口状态（drawKG 每帧读取）；ECharts 模式：重渲染图谱（带去抖）
var _kgControlTimer = null;
function onKGControlChange() {
    var thr = parseFloat(document.getElementById('kg-edge-threshold').value);
    var same = document.getElementById('kg-filter-same-law').checked;
    var cross = document.getElementById('kg-filter-cross').checked;
    var label = document.getElementById('kg-label-mode').value;
    document.getElementById('kg-threshold-val').textContent = isNaN(thr) ? '0.00' : thr.toFixed(2);

    var state = window._kgState;
    if (state) {  // Canvas 模式：就地更新，无需重绘调用
        state.filterSameLaw = same;
        state.filterCross = cross;
        state.edgeThreshold = thr;
        state.labelMode = label;
        return;
    }
    // ECharts 模式：边过滤/标签需重渲染，做 120ms 去抖避免拖动滑块时反复重建力导向
    if (window._kgLastData) {
        if (_kgControlTimer) clearTimeout(_kgControlTimer);
        _kgControlTimer = setTimeout(function() {
            renderKG(
                window._kgLastData,
                document.getElementById('kg-graph-title').textContent,
                window._kgLayoutMode
            );
        }, 120);
    }
}

function bindKGControlsOnce() {
    var ids = ['kg-filter-same-law', 'kg-filter-cross', 'kg-edge-threshold', 'kg-label-mode'];
    ids.forEach(function(id) {
        var el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('change', onKGControlChange);
        el.addEventListener('input', onKGControlChange);
    });
    // 初始化阈值显示
    var thr = document.getElementById('kg-edge-threshold');
    if (thr) document.getElementById('kg-threshold-val').textContent = parseFloat(thr.value).toFixed(2);
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

    // 社区结构（Louvain）
    var communities = data.communities || [];
    if (communities.length > 1) {
        var topComms = communities.slice(0, 3).map(function(c) { return (c.label || '簇') + '(' + c.size + ')'; });
        parts.push('识别出 ' + communities.length + ' 个主题社区，如 ' + topComms.join('、'));
    }

    // 核心枢纽（PageRank）
    var hubs = data.top_hubs || [];
    if (hubs.length > 0) {
        var hubLabels = hubs.slice(0, 5).map(function(h) { return h.label; });
        parts.push('核心枢纽（PageRank）：' + hubLabels.join('、'));
    }

    document.getElementById('kg-insight-text').textContent = parts.join('，') + '。';
    document.getElementById('kg-insight-panel').style.display = 'block';
}

// ===== 加载法规视图 =====
window.loadLawGraph = function() {
    var lawId = document.getElementById('kg-law-select').value;
    if (!lawId) return;
    // 知识层首次构建需调用 LLM，可能耗时数十秒，此处给出明确提示
    document.getElementById('kg-graph-title').textContent = '加载中（首次构建知识层可能需要数十秒）...';
    api.get('/kg/law/' + encodeURIComponent(lawId)).then(function(resp) {
        if (resp.success && resp.data) {
            var isLLM = resp.data.graph_source === 'llm';
            var sourceTag = isLLM ? '知识层（实体关系）' : '条款关联（已降级）';
            if (isLLM && resp.data.cache_hit) sourceTag += ' · 缓存命中';
            var law = (resp.data.nodes || []).find(function(n) { return n.law_id === lawId; });
            var title = ((law && law.law_name) || lawId) + ' · ' + sourceTag;
            renderKG(resp.data, title, window._kgLayoutMode);
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
    document.getElementById('kg-search-results').innerHTML = '';
    // 优先：实体词典 → 实体邻域知识图谱
    api.get('/kg/entity-graph', { q: q, limit: 50 }).then(function(resp) {
        var nodes = (resp.success && resp.data && resp.data.nodes) || [];
        if (nodes.length > 0) {
            renderKG(resp.data, '实体图谱: ' + q);
            return null;
        }
        // 降级：知识层尚无该实体，回退条款检索建图
        return api.get('/kg/search-entity', { q: q, page_size: 20 }).then(function(r2) {
            if (r2.success && r2.data) {
                renderKG(r2.data, '条款关联（已降级）: ' + q);
            } else {
                document.getElementById('kg-graph-title').textContent = '未找到相关结果';
                document.getElementById('kg-search-results').innerHTML = '<div class="alert alert-info">未找到相关结果，请尝试其他关键词</div>';
            }
        });
    }).catch(function() {
        document.getElementById('kg-graph-title').textContent = '搜索请求失败';
    });
};

// ===== 跨法域对比 =====
window.loadCrossGraph = function() {
    var jurA = document.getElementById('kg-cross-jur-a').value;
    var jurB = document.getElementById('kg-cross-jur-b').value;
    var topic = document.getElementById('kg-cross-topic').value;
    var view = document.getElementById('kg-cross-view') ? document.getElementById('kg-cross-view').value : 'matrix';
    if (!jurA || !jurB || !topic) return;

    var graphCard = document.getElementById('kg-graph-card');
    var matrixCard = document.getElementById('kg-cross-matrix-card');
    document.getElementById('kg-cross-detail').innerHTML = '';

    // ---- 视图二：法规级对齐矩阵（默认，总览）----
    if (view === 'matrix') {
        if (graphCard) graphCard.style.display = 'none';
        if (matrixCard) matrixCard.style.display = 'block';
        renderCrossMatrix(jurA, jurB, topic);
        return;
    }

    // ---- 视图一：条款级关系图谱 ----
    if (matrixCard) matrixCard.style.display = 'none';
    if (graphCard) graphCard.style.display = 'block';
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

// ===== 法规级对齐矩阵 =====
var _kgCrossMatrixChart = null;

function renderCrossMatrix(jurA, jurB, topic) {
    var el = document.getElementById('kg-cross-matrix');
    var fallback = document.getElementById('kg-cross-matrix-fallback');
    fallback.innerHTML = '<div style="padding:1rem;color:var(--color-text-muted);">正在计算语义对齐（首次需编码条款，可能数秒）...</div>';
    api.get('/structured-analysis/alignment-matrix', {
        topic_id: topic, jurisdiction_a: jurA, jurisdiction_b: jurB, top_n: 10
    }).then(function(resp) {
        if (!resp.success || !resp.data || !(resp.data.cells || []).length) {
            fallback.innerHTML = '<div class="alert alert-info" style="margin:0.5rem;">未找到可对齐的法规数据（可尝试提高每法域检索条款数或更换专题）</div>';
            return;
        }
        fallback.innerHTML = '';
        drawCrossMatrix(resp.data, jurA, jurB, topic);
    }).catch(function() {
        fallback.innerHTML = '<div class="alert alert-info" style="margin:0.5rem;">对齐矩阵加载失败</div>';
    });
}

function drawCrossMatrix(data, jurA, jurB, topic) {
    var lawsA = data.laws_a || [];
    var lawsB = data.laws_b || [];
    var yLabels = lawsA.map(function(l) { return l.law_name || l.law_id; });
    var xLabels = lawsB.map(function(l) { return l.law_name || l.law_id; });
    var el = document.getElementById('kg-cross-matrix');
    el.innerHTML = '';

    if (!window.echarts) {
        renderCrossMatrixTable(data, jurA, jurB, topic);
        el.style.display = 'none';
        return;
    }
    el.style.display = 'block';
    var rowH = 30;
    el.style.height = Math.max(320, rowH * yLabels.length + 140) + 'px';
    if (_kgCrossMatrixChart) { _kgCrossMatrixChart.dispose(); }
    _kgCrossMatrixChart = echarts.init(el);

    _kgCrossMatrixChart.setOption({
        tooltip: {
            position: 'top',
            formatter: function(p) {
                if (!p.value || !p.value[2]) return '';
                return '<b>' + escapeHtml(yLabels[p.value[1]]) + '</b><br/>↕<br/><b>' +
                    escapeHtml(xLabels[p.value[0]]) + '</b><br/>语义对口条款对：' + p.value[2];
            }
        },
        grid: { left: 190, right: 30, top: 14, bottom: 90 },
        xAxis: {
            type: 'category', data: xLabels, splitArea: { show: true },
            axisLabel: { interval: 0, rotate: 42, fontSize: 10, width: 120, overflow: 'truncate' }
        },
        yAxis: {
            type: 'category', data: yLabels, splitArea: { show: true },
            axisLabel: { fontSize: 10, width: 180, overflow: 'truncate' }
        },
        visualMap: {
            min: 0, max: Math.max(1, data.max_count || 1), calculable: true, orient: 'horizontal',
            left: 'center', bottom: 6,
            inRange: { color: ['#F1F5F9', '#BFDBFE', '#60A5FA', '#2563EB', '#1E3A8A'] }
        },
        series: [{
            name: '条款对数', type: 'heatmap', data: data.cells,
            label: { show: true, fontSize: 10, formatter: function(p) { return p.value[2] || ''; } },
            emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } }
        }]
    });
    _kgCrossMatrixChart.off('click');
    _kgCrossMatrixChart.on('click', function(p) {
        if (!p.value || !p.value[2]) return;
        var la = lawsA[p.value[1]], lb = lawsB[p.value[0]];
        if (la && lb) showCrossDetail(jurA, jurB, topic, la, lb);
    });
}

// ECharts 不可用时的 HTML 表格降级
function renderCrossMatrixTable(data, jurA, jurB, topic) {
    var lawsA = data.laws_a || [];
    var lawsB = data.laws_b || [];
    var countMap = {};
    (data.cells || []).forEach(function(c) { countMap[c[1] + '_' + c[0]] = c[2]; });
    var html = '<div style="overflow:auto;"><table style="border-collapse:collapse;font-size:0.72rem;">';
    html += '<tr><th style="border:1px solid var(--color-border);padding:3px;"></th>';
    lawsB.forEach(function(l) {
        html += '<th style="border:1px solid var(--color-border);padding:3px;writing-mode:vertical-rl;max-height:110px;">' + escapeHtml(l.law_name || l.law_id) + '</th>';
    });
    html += '</tr>';
    lawsA.forEach(function(la, i) {
        html += '<tr><td style="border:1px solid var(--color-border);padding:3px;">' + escapeHtml(la.law_name || la.law_id) + '</td>';
        lawsB.forEach(function(lb, j) {
            var v = countMap[i + '_' + j] || 0;
            var bg = v ? 'rgba(37,99,235,' + Math.min(0.85, 0.15 + v * 0.06) + ')' : 'transparent';
            html += '<td data-i="' + i + '" data-j="' + j + '" style="border:1px solid var(--color-border);padding:3px;text-align:center;cursor:pointer;background:' + bg + ';">' + (v || '') + '</td>';
        });
        html += '</tr>';
    });
    html += '</table></div>';
    var box = document.getElementById('kg-cross-matrix-fallback');
    box.innerHTML = html;
    box.querySelectorAll('td[data-i]').forEach(function(td) {
        td.addEventListener('click', function() {
            var i = parseInt(td.dataset.i, 10), j = parseInt(td.dataset.j, 10);
            if ((countMap[i + '_' + j] || 0) > 0) showCrossDetail(jurA, jurB, topic, lawsA[i], lawsB[j]);
        });
    });
}

window.showCrossDetail = function(jurA, jurB, topic, lawA, lawB) {
    var panel = document.getElementById('kg-cross-detail');
    panel.innerHTML = '<div class="card" style="padding:0.75rem 1rem;color:var(--color-text-muted);">明细加载中...</div>';
    api.get('/structured-analysis/alignment-detail', {
        topic_id: topic, jurisdiction_a: jurA, jurisdiction_b: jurB,
        law_a: lawA.law_id, law_b: lawB.law_id, top_n: 10
    }).then(function(resp) {
        if (!resp.success || !resp.data) {
            panel.innerHTML = '<div class="alert alert-info">未找到该法规对的对齐明细</div>';
            return;
        }
        var d = resp.data;
        var html = '<div class="card" style="padding:0.75rem 1rem;">';
        html += '<div style="font-weight:600;margin-bottom:0.5rem;">' + escapeHtml(d.law_a.law_name) +
                ' <span style="color:var(--color-text-muted);">↔</span> ' + escapeHtml(d.law_b.law_name) +
                ' <span class="badge" style="background:#2563EB;color:#fff;margin-left:0.375rem;">' + d.pair_count + ' 对</span></div>';
        if (!(d.pairs || []).length) {
            html += '<div style="color:var(--color-text-muted);font-size:0.8125rem;">暂无达到阈值的条款对</div>';
        }
        (d.pairs || []).forEach(function(pr) {
            html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:0.5rem;padding-bottom:0.5rem;border-bottom:1px solid var(--color-border);">';
            html += '<div style="font-size:0.75rem;line-height:1.5;"><span class="badge" style="background:#EFF6FF;color:#1E40AF;">第' + escapeHtml(pr.a_article || '?') + '条</span><div style="color:var(--color-text-secondary);margin-top:0.2rem;">' + escapeHtml((pr.a_preview || '').substring(0, 120)) + '…</div></div>';
            html += '<div style="font-size:0.75rem;line-height:1.5;"><span class="badge" style="background:#FEF2F2;color:#B91C1C;">第' + escapeHtml(pr.b_article || '?') + '条</span><div style="color:var(--color-text-secondary);margin-top:0.2rem;">' + escapeHtml((pr.b_preview || '').substring(0, 120)) + '…</div></div>';
            html += '</div>';
        });
        html += '</div>';
        panel.innerHTML = html;
    }).catch(function() {
        panel.innerHTML = '<div class="alert alert-info">明细加载失败</div>';
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

// ===== 概念对比（跨法域概念对齐矩阵）=====
var _kgConceptChart = null;

window.loadConceptMatrix = function() {
    var minJur = parseInt(document.getElementById('kg-concept-minjur').value || '2', 10);
    var topN = parseInt(document.getElementById('kg-concept-topn').value || '30', 10);
    var fallback = document.getElementById('kg-concept-fallback');
    document.getElementById('kg-concept-detail').innerHTML = '';
    fallback.innerHTML = '<div style="padding:1rem;color:var(--color-text-muted);">加载中...</div>';
    api.get('/kg/concept-matrix', { top_n: topN, min_jurisdictions: minJur }).then(function(resp) {
        if (!resp.success || !resp.data || !(resp.data.concepts || []).length) {
            fallback.innerHTML = '<div class="alert alert-info" style="margin:0.5rem;">暂无跨法域概念数据（知识层可能尚未构建，或没有概念跨法域出现）</div>';
            return;
        }
        fallback.innerHTML = '';
        renderConceptHeatmap(resp.data);
    }).catch(function() {
        fallback.innerHTML = '<div class="alert alert-info" style="margin:0.5rem;">矩阵加载失败</div>';
    });
};

function renderConceptHeatmap(data) {
    var jurs = data.jurisdictions || [];
    var concepts = data.concepts || [];
    var yLabels = concepts.map(function(c) { return c.name; });
    var maxV = data.max_count || 1;

    var cells = [];
    concepts.forEach(function(c, yi) {
        jurs.forEach(function(j, xi) {
            var v = (c.counts && c.counts[j]) ? c.counts[j] : 0;
            cells.push([xi, yi, v]);
        });
    });

    var el = document.getElementById('kg-concept-heatmap');

    // ECharts 不可用（CDN 未加载）时降级为 HTML 表格
    if (!window.echarts) {
        renderConceptTable(concepts, jurs);
        el.style.display = 'none';
        return;
    }
    el.style.display = 'block';
    el.style.height = Math.max(320, 26 * yLabels.length + 120) + 'px';
    if (!_kgConceptChart) {
        _kgConceptChart = echarts.init(el);
    } else {
        _kgConceptChart.dispose();
        _kgConceptChart = echarts.init(el);
    }
    _kgConceptChart.setOption({
        tooltip: {
            position: 'top',
            formatter: function(p) {
                return '<b>' + escapeHtml(yLabels[p.value[1]]) + '</b><br/>' +
                    escapeHtml(jurs[p.value[0]]) + '：' + p.value[2] + ' 条';
            }
        },
        grid: { left: 200, right: 40, top: 16, bottom: 70 },
        xAxis: { type: 'category', data: jurs, splitArea: { show: true }, axisLabel: { interval: 0 } },
        yAxis: {
            type: 'category', data: yLabels, splitArea: { show: true },
            axisLabel: {
                fontSize: 11,
                formatter: function(v) { return v.length > 16 ? v.substring(0, 16) + '…' : v; }
            }
        },
        visualMap: {
            min: 0, max: maxV, calculable: true, orient: 'horizontal',
            left: 'center', bottom: 8,
            inRange: { color: ['#EFF6FF', '#93C5FD', '#2563EB', '#1E3A8A'] }
        },
        series: [{
            name: '条文数', type: 'heatmap', data: cells,
            label: { show: true, fontSize: 10, formatter: function(p) { return p.value[2] || ''; } },
            emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } }
        }]
    });
    _kgConceptChart.off('click');
    _kgConceptChart.on('click', function(p) {
        var c = concepts[p.value[1]];
        if (c) showConceptDetail(c);
    });
}

function renderConceptTable(concepts, jurs) {
    var html = '<div style="overflow:auto;"><table style="border-collapse:collapse;font-size:0.75rem;width:100%;">';
    html += '<tr><th style="text-align:left;padding:4px 6px;border:1px solid var(--color-border);">概念</th>';
    jurs.forEach(function(j) { html += '<th style="padding:4px 6px;border:1px solid var(--color-border);">' + escapeHtml(j) + '</th>'; });
    html += '</tr>';
    concepts.forEach(function(c, yi) {
        html += '<tr style="cursor:pointer;" data-yi="' + yi + '"><td style="padding:4px 6px;border:1px solid var(--color-border);">' + escapeHtml(c.name) + '</td>';
        jurs.forEach(function(j) {
            var v = c.counts && c.counts[j] ? c.counts[j] : 0;
            var bg = v ? 'rgba(37,99,235,' + Math.min(0.8, 0.15 + v * 0.08) + ')' : 'transparent';
            html += '<td style="padding:4px 6px;border:1px solid var(--color-border);text-align:center;background:' + bg + ';">' + (v || '') + '</td>';
        });
        html += '</tr>';
    });
    html += '</table></div>';
    var box = document.getElementById('kg-concept-fallback');
    box.innerHTML = html;
    box.querySelectorAll('tr[data-yi]').forEach(function(tr) {
        tr.addEventListener('click', function() { showConceptDetail(concepts[parseInt(tr.dataset.yi, 10)]); });
    });
}

window.showConceptDetail = function(concept) {
    var panel = document.getElementById('kg-concept-detail');
    panel.innerHTML = '<div class="card" style="padding:0.75rem 1rem;color:var(--color-text-muted);">明细加载中...</div>';
    api.get('/kg/concept-detail', { entity_id: concept.entity_id }).then(function(resp) {
        if (!resp.success || !resp.data) {
            panel.innerHTML = '<div class="alert alert-info">未找到该概念的条款明细</div>';
            return;
        }
        var d = resp.data;
        var html = '<div class="card" style="padding:0.75rem 1rem;">';
        html += '<div style="font-weight:600;margin-bottom:0.5rem;">概念：' + escapeHtml(d.name) +
                ' <span class="badge" style="background:#6B7280;color:#fff;margin-left:0.375rem;">' + escapeHtml(d.type) + '</span></div>';
        (d.jurisdictions || []).forEach(function(g) {
            html += '<div style="margin-bottom:0.625rem;">';
            html += '<div style="font-weight:600;color:var(--color-text-secondary);font-size:0.8125rem;margin-bottom:0.25rem;">' +
                    escapeHtml(g.jurisdiction) + '（' + g.article_count + ' 条）</div>';
            html += '<div style="display:flex;flex-wrap:wrap;gap:0.375rem;">';
            (g.articles || []).slice(0, 40).forEach(function(a) {
                var rel = (a.relations && a.relations.length) ? ' · ' + escapeHtml(a.relations.join('/')) : '';
                var label = (a.law_name || a.law_id || '').substring(0, 18);
                html += '<span class="badge" style="background:var(--color-bg);color:var(--color-text-secondary);border:1px solid var(--color-border);">' +
                        escapeHtml(label) + ' ' + escapeHtml(a.article_number || '') + rel + '</span>';
            });
            html += '</div></div>';
        });
        html += '</div>';
        panel.innerHTML = html;
    }).catch(function() {
        panel.innerHTML = '<div class="alert alert-info">明细加载失败</div>';
    });
};

// 窗口缩放时同步 ECharts 实例尺寸（热力图 + 图谱 + 跨域矩阵，只注册一次）
window.addEventListener('resize', function() {
    if (_kgConceptChart) _kgConceptChart.resize();
    if (_kgCrossMatrixChart) _kgCrossMatrixChart.resize();
    if (window._kgEchartsInstance) window._kgEchartsInstance.resize();
});

// ===== 渲染引擎：ECharts 2D =====
function getKGRenderEngine() {
    var sel = document.getElementById('kg-render-engine');
    return sel ? sel.value : 'echarts';
}
function getKGColorMode() {
    var sel = document.getElementById('kg-color-mode');
    return sel ? sel.value : 'jur';
}
function getKGLabelMode() {
    var sel = document.getElementById('kg-label-mode');
    return sel ? sel.value : 'smart';
}

/** 读取底部控制栏的边过滤条件（ECharts 模式下从 DOM 取值，与 Canvas 共用同一组控件） */
function getKGEdgeFilters() {
    var same = document.getElementById('kg-filter-same-law');
    var cross = document.getElementById('kg-filter-cross');
    var thr = document.getElementById('kg-edge-threshold');
    return {
        filterSameLaw: same ? same.checked : true,
        filterCross: cross ? cross.checked : true,
        edgeThreshold: thr ? parseFloat(thr.value) : 0,
    };
}

/**
 * 按控制栏条件过滤边（与 Canvas 的 drawKG 判定保持一致）。
 *
 * 例外：LLM 实体关系边（edge_kind === 'entity_relation'）的 type 也是 semantic_similar，
 * 若套用"跨域"复选框，取消勾选会把实体图的边一次性清空。故实体边只受阈值（置信度）约束，
 * 不受两个复选框约束 —— 这两个开关语义上只针对"条款关联图"。
 */
function filterKGEdges(edges, f) {
    return (edges || []).filter(function(e) {
        var isEntity = e.edge_kind === 'entity_relation';
        var isCross = e.type === 'cross_jurisdiction' || e.type === 'semantic_similar';
        if (isEntity) {
            if (e.similarity != null && e.similarity < f.edgeThreshold) return false;
            return true;
        }
        if (!f.filterCross && isCross) return false;
        if (!f.filterSameLaw && !isCross) return false;
        if (isCross && e.similarity != null && e.similarity < f.edgeThreshold) return false;
        return true;
    });
}

/**
 * 分簇初始布局：按（法域 / 社区）把节点分组，各组中心均匀分布在圆环上，
 * 组内用黄金角（叶序）螺旋散开，作为力导向的初始坐标。
 *
 * 为什么必须做：ECharts 力导向默认随机初始化，跨法域的两个簇会被挤成一团，
 * 从起点就"糊"在中间；给定分簇初值后，力导向只做微调，两簇一眼可分。
 */
function computeKGClusterPositions(nodes, colorMode, width, height) {
    var groups = {};
    nodes.forEach(function(n, i) {
        var key = (colorMode === 'community')
            ? String(n.community_id != null ? n.community_id : -1)
            : (n.jurisdiction || '未知');
        (groups[key] || (groups[key] = [])).push(i);
    });
    var keys = Object.keys(groups);
    var nG = keys.length;
    var cx = width / 2, cy = height / 2;
    var ringR = nG <= 1 ? 0 : Math.min(width, height) * 0.30;
    var pos = new Array(nodes.length);

    keys.forEach(function(k, gi) {
        var members = groups[k];
        var angle = nG <= 1 ? 0 : (gi / nG) * Math.PI * 2 - Math.PI / 2;
        var gx = cx + (nG <= 1 ? 0 : Math.cos(angle) * ringR);
        var gy = cy + (nG <= 1 ? 0 : Math.sin(angle) * ringR);
        // 簇内螺旋半径随簇规模增长，但设上限避免溢出容器
        var ga = Math.PI * (3 - Math.sqrt(5));   // 黄金角
        var step = Math.max(11, Math.min(20, Math.min(width, height) * 0.05));
        members.forEach(function(idx, j) {
            var a = j * ga + gi * 1.7;
            var r = step * Math.sqrt(j);
            pos[idx] = [
                Math.max(14, Math.min(width - 14, gx + Math.cos(a) * r)),
                Math.max(14, Math.min(height - 14, gy + Math.sin(a) * r)),
            ];
        });
    });
    return pos;
}

/** 计算需要显示标签的节点集合（依据"标签"下拉：智能 / 仅核心 / 全部 / 无） */
function computeKGLabelSet(nodes, labelMode) {
    var show = {};
    if (labelMode === 'all') {
        nodes.forEach(function(_, i) { show[i] = true; });
        return show;
    }
    if (labelMode === 'none') {
        return show;
    }
    // 重要性：优先 PageRank，其次度数
    var maxPR = 0, maxDeg = 0;
    nodes.forEach(function(n) {
        maxPR = Math.max(maxPR, n.pagerank || 0);
        maxDeg = Math.max(maxDeg, n.degree || 0);
    });
    function importance(n) {
        var pr = maxPR > 0 ? (n.pagerank || 0) / maxPR : 0;
        var dg = maxDeg > 0 ? (n.degree || 0) / maxDeg : 0;
        return pr * 0.7 + dg * 0.3;
    }
    var order = nodes.map(function(_, i) { return i; })
        .sort(function(a, b) { return importance(nodes[b]) - importance(nodes[a]); });
    // 智能：约前 22%；仅核心：约前 8% 且必含 is_core 节点
    var ratio = labelMode === 'core' ? 0.08 : 0.22;
    var count = Math.max(5, Math.min(24, Math.round(nodes.length * ratio)));
    order.slice(0, count).forEach(function(i) { show[i] = true; });
    if (labelMode === 'core') {
        nodes.forEach(function(n, i) { if (n.is_core) show[i] = true; });
    }
    return show;
}

function renderKGWithECharts(data, title) {
    var nodes = data.nodes || [];
    var edges = data.edges || [];
    var el = document.getElementById('kg-echarts');
    var canvasEl = document.getElementById('kg-canvas');
    if (!el) return;
    if (canvasEl) canvasEl.style.display = 'none';
    el.style.display = 'block';

    if (!window.echarts) {
        // 库未加载（理论上本地内置后不会发生）：静默回退到 Canvas，不打断用户
        var engineSel = document.getElementById('kg-render-engine');
        if (engineSel) engineSel.value = 'canvas';
        console.warn('[KG] ECharts 未加载，已自动回退到 Canvas 引擎');
        renderKG(data, title, window._kgLayoutMode);
        return;
    }

    if (window._kgEchartsInstance) { window._kgEchartsInstance.dispose(); }
    var chart = echarts.init(el);
    window._kgEchartsInstance = chart;

    var width = el.clientWidth || 800;
    var height = el.clientHeight || 500;
    var colorMode = getKGColorMode();
    var labelMode = getKGLabelMode();
    var communities = data.communities || [];
    var prMax = 0;
    nodes.forEach(function(n) { if (n.pagerank && n.pagerank > prMax) prMax = n.pagerank; });

    function colorOf(n) {
        return (colorMode === 'community')
            ? getCommunityColor(n.community_id)
            : getJurColor(n.jurisdiction);
    }
    function sizeOf(n) {
        if (n.pagerank != null && prMax > 0) {
            return 8 + Math.sqrt(Math.min(1, n.pagerank / prMax)) * 16;   // 8 ~ 24
        }
        var d = n.degree || 0;
        return d >= 8 ? 18 : (d >= 3 ? 12 : 8);
    }

    // 分类（ECharts 图例 + 自动配色）
    var catIndex = {};
    var categories = [];
    function categoryOf(n) {
        var key = (colorMode === 'community')
            ? ('社区 ' + (n.community_id != null ? getCommunityLabel(n.community_id, communities) : '其他'))
            : (n.jurisdiction || '未知');
        if (catIndex[key] === undefined) {
            catIndex[key] = categories.length;
            categories.push({ name: key, itemStyle: { color: colorOf(n) } });
        }
        return catIndex[key];
    }

    var labelSet = computeKGLabelSet(nodes, labelMode);
    var idToIndex = {};
    var nodeData = nodes.map(function(n, i) {
        idToIndex[n.id] = i;
        var fullName = buildKGLabels(n);
        return {
            id: n.id,
            name: fullName,
            symbolSize: sizeOf(n),
            category: categoryOf(n),
            value: n.content_preview || '',
            itemStyle: { color: colorOf(n) },
            // 逐节点控制标签显示：避免几十个长标签全开导致文字互相压叠
            label: { show: !!labelSet[i] },
        };
    });

    // 边过滤（与控制栏的复选框/阈值一致），再转换为 ECharts 边数据
    var edgeFilters = getKGEdgeFilters();
    var linkData = filterKGEdges(edges, edgeFilters)
        .filter(function(e) { return idToIndex[e.from] !== undefined && idToIndex[e.to] !== undefined; })
        .map(function(e) {
            return { source: e.from, target: e.to, relation: (e.relation_label || e.label || '关联') };
        });

    // 边透明度按密度自适应：边越多越淡，避免"毛线团"
    var density = linkData.length / Math.max(nodes.length, 1);
    var edgeAlpha = density > 4 ? 0.16 : (density > 2 ? 0.28 : 0.45);

    var initPositions = computeKGClusterPositions(nodes, colorMode, width, height);

    var baseFormatter = function(p) {
        if (p.dataType === 'edge') return escapeHtml(p.data.relation || '关联');
        var n = nodes[p.dataIndex] || {};
        return '<b>' + escapeHtml(buildKGLabels(n)) + '</b><br/>' +
            escapeHtml(n.jurisdiction || '') + ' · ' + escapeHtml(n.law_name || '') +
            (n.community_id != null ? '<br/>社区：' + escapeHtml(getCommunityLabel(n.community_id, communities)) : '');
    };

    // 图例：类别不多时显示（跨法域通常 2 类）；类别过多则省略，避免图例占满画面
    var legendOption = [];
    if (categories.length > 0 && categories.length <= 12) {
        legendOption = [{
            data: categories.map(function(c) { return c.name; }),
            top: 4, right: 8, type: 'scroll',
            textStyle: { fontSize: 10 }, itemWidth: 10, itemHeight: 10,
        }];
    }

    chart.setOption({
        tooltip: { formatter: baseFormatter },
        legend: legendOption,
        animation: false,
        series: [{
            type: 'graph',
            layout: 'force',
            roam: true,
            draggable: true,
            data: nodeData,
            links: linkData,
            categories: categories,
            // 分簇初值 + 较大斥力 + 较小重力：两簇分开且不塌缩到中心
            force: {
                initLayout: function(nodeIndex) {
                    return initPositions[nodeIndex] || [width / 2, height / 2];
                },
                repulsion: 320,
                edgeLength: [70, 150],
                gravity: 0.04,
                friction: 0.6,
                layoutAnimation: true,
            },
            label: {
                show: false, position: 'right', fontSize: 10, color: '#334155',
                formatter: function(p) {
                    var s = p.name || '';
                    return s.length > 22 ? s.substring(0, 22) + '…' : s;
                },
            },
            labelLayout: { hideOverlap: true },   // 标签冲突时自动隐藏，进一步抗重叠
            lineStyle: { color: 'source', curveness: 0.12, opacity: edgeAlpha, width: 1 },
            emphasis: {
                focus: 'adjacency',
                label: { show: true },
                lineStyle: { opacity: 0.85, width: 2 },
            },
            blur: { itemStyle: { opacity: 0.25 }, lineStyle: { opacity: 0.05 } },
        }]
    });
    bindEChartsGraphEvents(chart, nodes);
}

// 节点点击 → 复用现有详情面板（图例高亮由 ECharts 自身处理）
function bindEChartsGraphEvents(chart, nodes) {
    chart.off('click');
    chart.on('click', function(p) {
        if (p.dataType !== 'node') return;
        var n = null;
        if (p.data && p.data.id) {
            n = nodes.filter(function(x) { return x.id === p.data.id; })[0];
        }
        if (!n && p.dataIndex != null) n = nodes[p.dataIndex];
        if (n) showKGDetail(n);
    });
}

// ===== 图谱问答（多跳推理 + 路径证据）=====
window.runKGReasoning = function() {
    var q = (document.getElementById('kg-reasoning-input').value || '').trim();
    if (!q) return;
    var box = document.getElementById('kg-reasoning-result');
    box.innerHTML = '<div style="color:var(--color-text-muted);">推理中（实体链接 → 子图检索 → 生成）...</div>';
    api.post('/kg/reasoning-qa', { question: q }).then(function(resp) {
        var d = resp && resp.data;
        if (!d) { box.innerHTML = '<div class="alert alert-info">请求失败</div>'; return; }
        if (!d.found) {
            box.innerHTML = '<div class="alert alert-info">' + escapeHtml(d.message || '未找到图谱依据') + '</div>';
            return;
        }
        var html = '<div class="card" style="padding:0.75rem 1rem;">';
        html += '<div style="font-weight:600;margin-bottom:0.375rem;">回答</div>';
        html += '<div style="white-space:pre-wrap;line-height:1.7;font-size:0.875rem;">' + escapeHtml(d.answer || '') + '</div>';

        if (d.seed_entities && d.seed_entities.length) {
            var names = d.seed_entities.map(function(s) { return escapeHtml(s.name || s.entity_id); }).join('、');
            html += '<div style="margin-top:0.5rem;font-size:0.75rem;color:var(--color-text-muted);">命中实体：' + names + '</div>';
        }
        if (d.used_paths && d.used_paths.length) {
            html += '<div style="margin-top:0.625rem;font-weight:600;font-size:0.8125rem;">推理路径证据</div>';
            html += '<div style="font-size:0.8125rem;line-height:1.8;">';
            d.used_paths.slice(0, 20).forEach(function(p) {
                html += '<div>(' + escapeHtml(p.head || '') +
                    ') <span style="color:var(--color-primary);">--' + escapeHtml(p.relation || '') + '--&gt;</span> (' +
                    escapeHtml(p.tail || '') + ')' +
                    (p.article ? ' <span style="color:var(--color-text-muted);">[' + escapeHtml(String(p.article)) + ']</span>' : '') +
                    '</div>';
            });
            html += '</div>';
        }
        if (d.cited_articles && d.cited_articles.length) {
            var cited = d.cited_articles.slice(0, 20).map(function(a) { return escapeHtml(String(a)); }).join('、');
            html += '<div style="margin-top:0.5rem;font-size:0.75rem;color:var(--color-text-muted);">引用条款：' + cited + '</div>';
        }
        html += '</div>';
        box.innerHTML = html;
    }).catch(function() {
        box.innerHTML = '<div class="alert alert-info">请求失败（请稍后再试）</div>';
    });
};
