/**
 * 数据总览看板页面
 */
registerPage('dashboard', function(container) {
    container.innerHTML = `
        <h1 class="page-title">数据总览看板</h1>
        <p class="page-desc">展示境外法规数据库的整体数据概览和统计分析</p>
        <div id="dashboard-content"></div>
    `;

    const content = document.getElementById('dashboard-content');
    showLoading(content);

    api.get('/stats').then(function(data) {
        if (!data || !data.data) {
            showError(content, '暂无统计数据');
            return;
        }
        renderDashboard(content, data.data);
    }).catch(function(err) {
        showError(content, '获取统计数据失败: ' + err.message);
    });
});

function renderDashboard(container, stats) {
    let html = '';

    // 维度一：核心统计指标
    const overview = stats.overview || {};
    html += '<h2 class="section-title">核心数据概览</h2>';
    html += '<div class="stat-cards">';
    html += statCard('法规总数', overview.total_laws || 0);
    html += statCard('条款总数', overview.total_clauses || 0);
    html += statCard('覆盖法域', overview.jurisdiction_count || 0);
    html += statCard('主题类别', overview.topic_count || 0);
    html += '</div>';

    // 维度二：法域分布
    const jurisdiction = stats.jurisdiction || {};
    html += '<h2 class="section-title">法域分布</h2>';
    html += '<div class="grid-2">';
    html += '<div class="card chart-card">';
    html += '<div class="card-title">法规数量分布</div>';
    html += '<div style="height:250px;position:relative;"><canvas id="law-pie-chart"></canvas></div>';
    html += '</div>';
    html += '<div class="card chart-card">';
    html += '<div class="card-title">条款数量分布</div>';
    html += '<div class="chart-scroll-wrap" style="height:250px;"><canvas id="clause-bar-chart"></canvas></div>';
    html += '</div>';
    html += '</div>';

    // 维度三：主题分类分布
    const topics = stats.topics || [];
    html += '<h2 class="section-title">主题分类分布</h2>';
    html += '<div class="grid-2">';
    html += '<div class="card chart-card">';
    html += '<div class="card-title">主题词云</div>';
    html += '<div id="topic-wordcloud" style="height: 350px; position: relative; overflow: hidden;"></div>';
    html += '</div>';
    html += '<div class="card chart-card">';
    html += '<div class="card-title">主题-法域热力图</div>';
    html += '<div id="topic-heatmap-wrap" class="chart-scroll-wrap"><canvas id="topic-heatmap"></canvas></div>';
    html += '</div>';
    html += '</div>';

    // 维度四：时间变化趋势
    const timeTrend = stats.time_trend || {};
    html += '<h2 class="section-title">立法时间趋势</h2>';
    html += '<div class="grid-2">';
    html += '<div class="card chart-card">';
    html += '<div class="card-title">立法数量趋势</div>';
    html += '<div class="chart-scroll-wrap" style="height:300px;"><canvas id="time-line-chart"></canvas></div>';
    html += '</div>';
    html += '<div class="card chart-card">';
    html += '<div class="card-title">各时段法规总量</div>';
    html += '<div class="chart-scroll-wrap" style="height:300px;"><canvas id="time-bar-chart"></canvas></div>';
    html += '</div>';
    html += '</div>';

    container.innerHTML = html;

    // 渲染图表
    renderJurisdictionCharts(jurisdiction);
    renderTopicCharts(topics);
    renderTimeCharts(timeTrend);
}

function statCard(label, value) {
    return '<div class="stat-card"><div class="stat-value">' + value + '</div><div class="stat-label">' + label + '</div></div>';
}

function renderJurisdictionCharts(jurisdiction) {
    const lawCounts = jurisdiction.law_counts || [];
    const clauseCounts = jurisdiction.clause_counts || [];

    if (lawCounts.length > 0) {
        const ctx = document.getElementById('law-pie-chart').getContext('2d');
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: lawCounts.map(function(item) { return item.jurisdiction; }),
                datasets: [{
                    data: lawCounts.map(function(item) { return item.count; }),
                    backgroundColor: ['#8B2500', '#A0522D', '#3D5A6E', '#2E6B3E', '#8B6914']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    if (clauseCounts.length > 0) {
        const ctx = document.getElementById('clause-bar-chart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: clauseCounts.map(function(item) { return item.jurisdiction; }),
                datasets: [{
                    label: '条款数量',
                    data: clauseCounts.map(function(item) { return item.count; }),
                    backgroundColor: '#3D5A6E'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }
}

// 计算两个中文字符串的字符级 Jaccard 相似度
function topicSimilarity(a, b) {
    var setA = new Set(a.replace(/\s+/g, ''));
    var setB = new Set(b.replace(/\s+/g, ''));
    var intersect = 0, union = 0;
    var allChars = new Set([...setA, ...setB]);
    allChars.forEach(function(ch) {
        if (setA.has(ch) && setB.has(ch)) intersect++;
        union++;
    });
    return union > 0 ? intersect / union : 0;
}

// 从主题列表中选取最具代表性的主题（去相似、按数量排序）
function selectRepresentativeTopics(topics, maxCount) {
    // 按 count 降序排序
    var sorted = topics.slice().sort(function(a, b) { return (b.count || 0) - (a.count || 0); });
    var selected = [];
    var SIMILARITY_THRESHOLD = 0.5;

    sorted.forEach(function(item) {
        var name = item.label_zh || item.topic || '';
        if (!name) return;
        // 检查是否与已选主题相似
        var isSimilar = selected.some(function(s) {
            return topicSimilarity(name, s.label_zh || s.topic || '') >= SIMILARITY_THRESHOLD;
        });
        if (!isSimilar) {
            selected.push(item);
        }
    });

    // 取前 maxCount 个
    return selected.slice(0, maxCount);
}

function renderTopicCharts(topics) {
    // 真实词云实现
    var wordcloudEl = document.getElementById('topic-wordcloud');
    if (topics.length > 0) {
        renderWordCloud(wordcloudEl, topics);
    } else {
        wordcloudEl.innerHTML = '暂无主题数据';
    }

    // 真实热力图实现 — 选取最具代表性的主题（最多 10 个）
    var selectedTopics = selectRepresentativeTopics(topics, 10);

    var heatmapData = [];
    var topicsList = [];
    var jurisdictionsSet = new Set();

    selectedTopics.forEach(function(item) {
        var topicName = item.label_zh || item.topic || '';
        var byJur = item.by_jurisdiction || {};
        topicsList.push(topicName);
        Object.keys(byJur).forEach(function(jur) {
            jurisdictionsSet.add(jur);
            heatmapData.push({ topic: topicName, jurisdiction: jur, count: byJur[jur] });
        });
    });

    if (heatmapData.length > 0) {
        const canvas = document.getElementById('topic-heatmap');
        const wrap = document.getElementById('topic-heatmap-wrap');
        const jurisdictions = Array.from(jurisdictionsSet);
        if (canvas && wrap) {
            setupHeatmapCanvas(canvas, wrap, topicsList.length, jurisdictions.length);
            const ctx = canvas.getContext('2d');
            const width = parseInt(canvas.style.width || wrap.clientWidth, 10);
            const height = parseInt(canvas.style.height || wrap.clientHeight, 10);
            renderHeatmap(ctx, width, height, topicsList, jurisdictions, heatmapData);
        }
    }
}

// 词云渲染函数（3D 视差版）
function renderWordCloud(container, topics) {
    container.innerHTML = '';

    const words = topics.slice(0, 30).map(function(item) {
        return {
            text: item.label_zh || item.topic || '',
            weight: item.count || 1
        };
    });
    if (words.length === 0) {
        container.innerHTML = '暂无主题数据';
        return;
    }

    // 为容器设置 3D 透视
    container.style.perspective = '800px';
    container.style.overflow = 'hidden';
    container.style.cursor = 'grab';
    container.style.position = 'relative';

    const maxWeight = Math.max.apply(null, words.map(function(w) { return w.weight; }));
    const minWeight = Math.min.apply(null, words.map(function(w) { return w.weight; }));
    const maxFontSize = 48;
    const minFontSize = 14;

    // 在容器内创建一个 tilt 层，应用 3D 变换
    const tiltLayer = document.createElement('div');
    tiltLayer.style.cssText = 'position:relative;width:100%;height:100%;transform-style:preserve-3d;transition:transform 0.1s ease-out;';
    container.appendChild(tiltLayer);

    const cw = container.clientWidth || 600;
    const ch = 350;
    tiltLayer.style.height = ch + 'px';

    const colors = ['#8B2500', '#A0522D', '#3D5A6E', '#2E6B3E', '#8B6914', '#5B3A8C', '#7A4C2C'];
    const centerX = cw / 2;
    const centerY = ch / 2;

    words.forEach(function(word, index) {
        const fontSize = minFontSize + (maxFontSize - minFontSize) * (word.weight - minWeight) / (maxWeight - minWeight || 1);

        const span = document.createElement('span');
        span.textContent = word.text;
        span.style.cssText = [
            'position:absolute',
            'left:0',
            'top:0',
            'font-size:' + fontSize + 'px',
            'font-weight:' + (word.weight > (maxWeight + minWeight) / 2 ? '700' : '500'),
            'font-family:"Noto Serif SC", "SimSun", serif',
            'white-space:nowrap',
            'color:' + colors[index % colors.length],
            'user-select:none',
            'pointer-events:none',
            // Z 轴深度：大字更靠前（z 正），小字靠后（z 负）
            'transform:translateZ(' + Math.round((fontSize - minFontSize) * 2) + 'px)',
            // 文字阴影增强立体感
            'text-shadow:' + [
                '0 1px 2px rgba(0,0,0,0.15)',
                '0 2px 6px rgba(0,0,0,0.06)'
            ].join(','),
            // 大字略微旋转增加自然感
            'transform:rotate(' + (Math.random() * 4 - 2).toFixed(1) + 'deg) translateZ(' + Math.round((fontSize - minFontSize) * 2) + 'px)',
        ].join(';');

        // 测量文字宽度
        var temp = document.createElement('span');
        temp.style.cssText = 'visibility:hidden;position:absolute;font-size:' + fontSize + 'px;font-weight:700;font-family:"Noto Serif SC", "SimSun", serif;white-space:nowrap;';
        temp.textContent = word.text;
        document.body.appendChild(temp);
        var tw = temp.offsetWidth || (word.text.length * fontSize * 0.7);
        document.body.removeChild(temp);

        var th = fontSize;

        // 螺旋布局
        var placed = false;
        var existing = tiltLayer.querySelectorAll('span');
        var existingRects = [];
        existing.forEach(function(el) {
            var l = parseFloat(el.style.left) || 0;
            var t = parseFloat(el.style.top) || 0;
            var fs = parseFloat(el.style.fontSize) || 14;
            existingRects.push({ x: l, y: t, w: el.offsetWidth || (el.textContent.length * fs * 0.7), h: fs });
        });

        for (var r = 0; r < 250 && !placed; r += 6) {
            for (var angle = 0; angle < Math.PI * 2 && !placed; angle += Math.PI / 10) {
                var x = centerX + r * Math.cos(angle) - tw / 2;
                var y = centerY + r * Math.sin(angle) - th / 2;

                if (x < 8 || x + tw > cw - 8 || y < 8 || y + th > ch - 8) continue;

                var collision = false;
                for (var i = 0; i < existingRects.length; i++) {
                    var p = existingRects[i];
                    if (x < p.x + p.w + 4 && x + tw + 4 > p.x && y < p.y + p.h + 4 && y + th + 4 > p.y) {
                        collision = true;
                        break;
                    }
                }

                if (!collision) {
                    span.style.left = x + 'px';
                    span.style.top = y + 'px';
                    // Z 轴深度：越靠近中心 Z 值越大（突出显示核心主题）
                    var distFromCenter = Math.sqrt(Math.pow(x + tw / 2 - centerX, 2) + Math.pow(y + th / 2 - centerY, 2));
                    var zBoost = Math.max(0, 1 - distFromCenter / (cw / 2)) * 20;
                    var zValue = Math.round((fontSize - minFontSize) * 2 + zBoost);
                    span.style.transform = 'rotate(' + (Math.random() * 4 - 2).toFixed(1) + 'deg) translateZ(' + zValue + 'px)';

                    tiltLayer.appendChild(span);
                    existingRects.push({ x: x, y: y, w: tw, h: th });
                    placed = true;
                }
            }
        }

        // 如果螺旋没放下，简单的随机放置
        if (!placed) {
            span.style.left = Math.random() * (cw - tw - 20) + 10 + 'px';
            span.style.top = Math.random() * (ch - th - 20) + 10 + 'px';
            tiltLayer.appendChild(span);
        }
    });

    // 鼠标移动 3D 视差效果
    container.addEventListener('mousemove', function(e) {
        var rect = container.getBoundingClientRect();
        var x = (e.clientX - rect.left) / rect.width;
        var y = (e.clientY - rect.top) / rect.height;
        // 鼠标位置映射到旋转角度：中心为 0，边缘最大 ±12deg
        var tiltX = (y - 0.5) * -20;
        var tiltY = (x - 0.5) * 20;
        tiltLayer.style.transform = [
            'rotateX(' + tiltX.toFixed(1) + 'deg)',
            'rotateY(' + tiltY.toFixed(1) + 'deg)'
        ].join(' ');
    });

    container.addEventListener('mouseleave', function() {
        tiltLayer.style.transform = 'rotateX(0deg) rotateY(0deg)';
    });
}

function setupHeatmapCanvas(canvas, wrap, topicCount, jurCount) {
    const dpr = window.devicePixelRatio || 1;
    const wrapWidth = Math.max(wrap.clientWidth, 300);
    // 保证每个单元格有足够可视空间；主题过多时增加画布高度并启用容器滚动
    const minCellHeight = 36;
    const minCellWidth = 60;
    // 限制画布最大宽度，超出部分由容器滚动显示
    const maxPlotWidth = Math.max(wrapWidth - 180, 500);
    const plotWidth = Math.min(Math.max(wrapWidth - 180, minCellWidth * jurCount), maxPlotWidth);
    // 限制画布最大高度，超出部分由容器滚动显示
    const maxPlotHeight = 600;
    const plotHeight = Math.min(Math.max(minCellHeight * topicCount, 200), maxPlotHeight);
    const width = plotWidth + 180;
    const height = plotHeight + 150;

    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// 热力图颜色阶：法律文书暖棕色系（羊皮纸 -> 赭石 -> 深褐）
// 阈值与背景色、文字色均经过对比度校验，确保可读性
const HEATMAP_COLOR_STOPS = [
    { threshold: 0.0, bg: '#faf8f3', text: '#a8a29e' },
    { threshold: 0.2, bg: '#f3e9d8', text: '#92400e' },
    { threshold: 0.4, bg: '#e6c9a0', text: '#78350f' },
    { threshold: 0.6, bg: '#c58940', text: '#ffffff' },
    { threshold: 0.8, bg: '#92400e', text: '#ffffff' },
    { threshold: 1.0, bg: '#451a03', text: '#ffffff' }
];

function getHeatmapColor(ratio) {
    for (let i = HEATMAP_COLOR_STOPS.length - 1; i >= 0; i--) {
        if (ratio >= HEATMAP_COLOR_STOPS[i].threshold) {
            return HEATMAP_COLOR_STOPS[i];
        }
    }
    return HEATMAP_COLOR_STOPS[0];
}

function drawRoundRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

// 热力图渲染函数
function renderHeatmap(ctx, width, height, topics, jurisdictions, data) {
    ctx.clearRect(0, 0, width, height);

    if (topics.length === 0 || jurisdictions.length === 0) return;

    // 计算边距
    const marginLeft = 150;
    const marginTop = 24;
    const marginRight = 24;
    const marginBottom = 120;
    const plotWidth = width - marginLeft - marginRight;
    const plotHeight = height - marginTop - marginBottom;

    const gap = 4;
    const cellWidth = plotWidth / jurisdictions.length;
    const cellHeight = plotHeight / topics.length;

    // 找到最大值用于颜色映射
    const maxCount = Math.max.apply(null, data.map(function(d) { return d.count; }));

    // 绘制单元格
    topics.forEach(function(topic, i) {
        jurisdictions.forEach(function(jur, j) {
            const item = data.find(function(d) { return d.topic === topic && d.jurisdiction === jur; });
            const count = item ? item.count : 0;

            const x = marginLeft + j * cellWidth + gap / 2;
            const y = marginTop + i * cellHeight + gap / 2;
            const w = Math.max(cellWidth - gap, 1);
            const h = Math.max(cellHeight - gap, 1);

            const ratio = maxCount > 0 ? count / maxCount : 0;
            const style = getHeatmapColor(ratio);

            // 绘制圆角单元格
            drawRoundRect(ctx, x, y, w, h, 8);
            ctx.fillStyle = style.bg;
            ctx.fill();

            // 绘制数值
            if (cellWidth > 28 && cellHeight > 22) {
                ctx.fillStyle = style.text;
                ctx.font = 'bold 13px "Inter", "Noto Sans SC", sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const text = count > 0 ? count.toString() : '-';
                ctx.fillText(text, x + w / 2, y + h / 2);
            }
        });
    });

    // 绘制 Y 轴标签（主题）
    ctx.fillStyle = '#475569';
    ctx.font = '13px "Inter", "Noto Sans SC", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    topics.forEach(function(topic, i) {
        const y = marginTop + i * cellHeight + cellHeight / 2;
        const maxChars = Math.max(8, Math.floor(marginLeft / 13));
        const label = topic.length > maxChars ? topic.substring(0, maxChars) + '...' : topic;
        ctx.fillText(label, marginLeft - 12, y);
    });

    // 绘制 X 轴标签（法域）
    ctx.fillStyle = '#475569';
    ctx.font = '13px "Inter", "Noto Sans SC", sans-serif';
    ctx.textBaseline = 'top';
    const rotateLabels = jurisdictions.length > 4;
    jurisdictions.forEach(function(jur, j) {
        const x = marginLeft + j * cellWidth + cellWidth / 2;
        const y = marginTop + topics.length * cellHeight + 16;
        ctx.save();
        ctx.translate(x, y);
        if (rotateLabels) {
            ctx.textAlign = 'right';
            ctx.rotate(-Math.PI / 5);
        } else {
            ctx.textAlign = 'center';
        }
        ctx.fillText(jur, 0, 0);
        ctx.restore();
    });

    // 绘制颜色图例
    const legendX = marginLeft;
    const legendY = height - 34;
    const legendW = 160;
    const legendH = 10;
    const segmentW = legendW / (HEATMAP_COLOR_STOPS.length - 1);

    HEATMAP_COLOR_STOPS.slice(1).forEach(function(stop, idx) {
        const sx = legendX + idx * segmentW;
        drawRoundRect(ctx, sx, legendY, segmentW - 1, legendH, 2);
        ctx.fillStyle = stop.bg;
        ctx.fill();
    });

    ctx.fillStyle = '#64748b';
    ctx.font = '11px "Inter", "Noto Sans SC", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('低', legendX - 2, legendY + 14);
    ctx.textAlign = 'right';
    ctx.fillText('高', legendX + legendW + 2, legendY + 14);
}

function renderTimeCharts(timeTrend) {
    const timeSeries = timeTrend.data || [];

    if (timeSeries.length === 0) return;

    const periods = timeSeries.map(function(item) { return item.period; });

    // 折线图（不分法域，整体趋势）
    const lineCtx = document.getElementById('time-line-chart').getContext('2d');
    const totalValues = timeSeries.map(function(item) { return item.total || 0; });

    new Chart(lineCtx, {
        type: 'line',
        data: {
            labels: periods,
            datasets: [{
                label: '立法数量',
                data: totalValues,
                borderColor: '#3D5A6E',
                backgroundColor: '#3D5A6E',
                tension: 0.3,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // 柱状图
    const totalCounts = timeSeries.map(function(item) { return item.total || 0; });
    const barCtx = document.getElementById('time-bar-chart').getContext('2d');
    new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: periods,
            datasets: [{
                label: '法规总量',
                data: totalCounts,
                backgroundColor: '#2E6B3E'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}
