const QA_STORAGE_KEY = 'legal_rag_qa_state';

function loadQAState() {
    try {
        const saved = localStorage.getItem(QA_STORAGE_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            return {
                sessionId: parsed.sessionId || generateUUID(),
                messages: (parsed.messages || []).map(function(m) {
                    return {
                        role: m.role,
                        content: m.content,
                        sources: m.sources,
                        jurisdictions: m.jurisdictions,
                        route: m.route,
                        trace: m.trace
                    };
                }),
                pendingImages: [],
                jurisdictions: parsed.jurisdictions || [],
                topK: parsed.topK || 5,
                // 深度模式（Agent 多步探索）：随会话持久化，刷新后保持用户选择
                deepMode: parsed.deepMode === true
            };
        }
    } catch (e) {
        console.warn('加载对话历史失败:', e);
    }
    return {
        sessionId: generateUUID(),
        messages: [],
        pendingImages: [],
        jurisdictions: [],
        topK: 5,
        deepMode: false
    };
}

function saveQAState() {
    try {
        const toSave = {
            sessionId: window.qaState.sessionId,
            messages: window.qaState.messages.map(function(m) {
                return { role: m.role, content: m.content, sources: m.sources, jurisdictions: m.jurisdictions, route: m.route, trace: m.trace };
            }),
            jurisdictions: window.qaState.jurisdictions,
            topK: window.qaState.topK,
            deepMode: window.qaState.deepMode === true
        };
        localStorage.setItem(QA_STORAGE_KEY, JSON.stringify(toSave));
    } catch (e) {
        console.warn('保存对话历史失败:', e);
    }
}

registerPage('qa', function(container) {
    if (!window.qaState) {
        window.qaState = loadQAState();
    }

    container.innerHTML = `
        <h1 class="page-title">法规智能问答</h1>
        <p class="page-desc">基于境外法规知识库的AI智能问答系统，支持精准检索、深度解读、多轮对话、跨法域对比、图片理解</p>
        <hr class="divider">

        <!-- 设置栏 -->
        <div style="display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;align-items:flex-end;">
            <div style="flex:1;min-width:200px;">
                <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">限定法域（可选）</label>
                <div class="jur-selector">
                    <div id="qa-jur-tags" class="jur-tags">
                        <span class="jur-placeholder">点击选择法域...</span>
                    </div>
                    <div id="qa-jur-trigger" class="jur-trigger">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
                    </div>
                    <div id="qa-jur-dropdown" class="jur-dropdown" style="display:none;">
                    </div>
                    <select id="qa-jurisdictions" class="select" multiple style="display:none;">
                    </select>
                </div>
            </div>
            <div style="width:150px;">
                <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">检索数量: <span id="qa-topk-val">5</span></label>
                <input type="range" id="qa-topk" min="3" max="10" value="5" style="width:100%;">
            </div>
            <div style="min-width:190px;">
                <label style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);display:block;margin-bottom:0.25rem;">深度模式（多步推理）</label>
                <button id="qa-deep-toggle" type="button" class="btn btn-secondary btn-sm" style="width:100%;"
                    title="开启后由智能体自主多步检索、追引用、查图谱后再作答；复杂问题更全面，简单问题仍走快速通道">
                    <span id="qa-deep-state">已关闭</span>
                </button>
            </div>
            <button id="qa-new-session" type="button" class="btn btn-secondary btn-sm">新建会话</button>
        </div>

        <!-- 聊天区域 -->
        <div id="qa-chat" style="min-height:400px;max-height:calc(100vh - 320px);overflow-y:auto;border:1px solid var(--color-border);border-radius:8px;padding:1.25rem;margin-bottom:1rem;background:var(--color-bg);">
            <div class="loading" id="qa-welcome">开始对话吧，输入您的问题或上传法规图片</div>
        </div>

        <!-- 图片预览区 -->
        <div id="qa-image-preview" style="display:none;margin-bottom:0.75rem;">
            <div style="font-size:0.8125rem;font-weight:600;color:var(--color-text-secondary);margin-bottom:0.5rem;">已添加的图片：</div>
            <div id="qa-image-list" style="display:flex;gap:0.5rem;flex-wrap:wrap;"></div>
        </div>

        <!-- 输入区域（DeepSeek风格） -->
        <div style="position:relative;border:1px solid var(--color-border-deep);border-radius:8px;background:var(--color-surface);padding:0.5rem;">
            <textarea id="qa-input" class="textarea" placeholder="请输入您的问题，或粘贴/上传图片..." style="border:none;resize:none;min-height:50px;outline:none;padding:0.5rem;" rows="2"></textarea>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;padding-top:0.5rem;border-top:1px solid var(--color-border);">
                <div style="display:flex;gap:0.5rem;align-items:center;">
                    <label class="btn btn-ghost btn-sm" style="cursor:pointer;" title="上传图片">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
                        <input type="file" id="qa-file-input" accept="image/jpeg,image/png,image/webp,image/gif" multiple style="display:none;">
                    </label>
                    <span id="qa-image-count" style="font-size:0.75rem;color:var(--color-text-muted);display:none;"></span>
                </div>
                <button id="qa-send" class="btn btn-primary btn-sm">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>
                    发送
                </button>
            </div>
        </div>
    `;

    api.get('/search/jurisdictions').then(function(data) {
        if (data && data.data) {
            const allJurs = data.data;
            const savedJurs = window.qaState.jurisdictions || [];
            const select = document.getElementById('qa-jurisdictions');
            const dropdown = document.getElementById('qa-jur-dropdown');
            allJurs.forEach(function(j) {
                const opt = document.createElement('option');
                opt.value = j;
                opt.textContent = j;
                opt.selected = savedJurs.indexOf(j) !== -1;
                select.appendChild(opt);

                const item = document.createElement('div');
                item.className = 'jur-dropdown-item';
                item.setAttribute('data-value', j);
                item.innerHTML = '<span>' + j + '</span>';
                dropdown.appendChild(item);
            });
            setupJurSelector();
        }
    }).catch(function() {});

    setupQAEvents(container);
    renderQAMessages();
});

function setupJurSelector() {
    const trigger = document.getElementById('qa-jur-trigger');
    const dropdown = document.getElementById('qa-jur-dropdown');
    const tags = document.getElementById('qa-jur-tags');
    const select = document.getElementById('qa-jurisdictions');

    trigger.addEventListener('click', function(e) {
        e.stopPropagation();
        dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
        updateDropdownSelection();
    });

    document.addEventListener('click', function(e) {
        if (!dropdown.contains(e.target) && !trigger.contains(e.target)) {
            dropdown.style.display = 'none';
        }
    });

    dropdown.addEventListener('click', function(e) {
        const item = e.target.closest('.jur-dropdown-item');
        if (!item) return;

        const value = item.getAttribute('data-value');
        const option = select.querySelector('option[value="' + value + '"]');
        option.selected = !option.selected;
        renderJurTags();
        updateDropdownSelection();
        window.qaState.jurisdictions = Array.from(select.selectedOptions).map(function(o) { return o.value; });
        saveQAState();
    });

    function renderJurTags() {
        const selected = Array.from(select.selectedOptions).map(function(o) { return o.value; });
        tags.innerHTML = '';

        if (selected.length === 0) {
            tags.innerHTML = '<span class="jur-placeholder">点击选择法域...</span>';
            return;
        }

        selected.forEach(function(jur) {
            const tag = document.createElement('span');
            tag.className = 'jur-tag';
            tag.innerHTML = '<span>' + jur + '</span><span class="jur-tag-remove" data-value="' + jur + '">x</span>';
            tags.appendChild(tag);
        });

        tags.querySelectorAll('.jur-tag-remove').forEach(function(btn) {
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                const value = this.getAttribute('data-value');
                const option = select.querySelector('option[value="' + value + '"]');
                option.selected = false;
                renderJurTags();
                updateDropdownSelection();
                window.qaState.jurisdictions = Array.from(select.selectedOptions).map(function(o) { return o.value; });
                saveQAState();
            });
        });
    }

    function updateDropdownSelection() {
        const selected = Array.from(select.selectedOptions).map(function(o) { return o.value; });
        dropdown.querySelectorAll('.jur-dropdown-item').forEach(function(item) {
            const value = item.getAttribute('data-value');
            item.classList.toggle('selected', selected.includes(value));
        });
    }

    renderJurTags();
    updateDropdownSelection();
}

function setupQAEvents(container) {
    const input = document.getElementById('qa-input');
    const sendBtn = document.getElementById('qa-send');
    const fileInput = document.getElementById('qa-file-input');
    const topkSlider = document.getElementById('qa-topk');
    const topkVal = document.getElementById('qa-topk-val');

    sendBtn.addEventListener('click', sendQAMessage);

    input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQAMessage();
        }
    });

    fileInput.addEventListener('change', function(e) {
        handleImageFiles(e.target.files);
        fileInput.value = '';
    });

    input.addEventListener('paste', function(e) {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const file = items[i].getAsFile();
                if (file) handleImageFiles([file]);
            }
        }
    });

    topkSlider.addEventListener('input', function() {
        topkVal.textContent = this.value;
        window.qaState.topK = parseInt(this.value);
        saveQAState();
    });

    const deepToggle = document.getElementById('qa-deep-toggle');
    updateDeepToggleUI();
    if (deepToggle) {
        deepToggle.addEventListener('click', function() {
            window.qaState.deepMode = !window.qaState.deepMode;
            updateDeepToggleUI();
            saveQAState();
        });
    }

    // 新建会话 - 使用事件委托，确保 SPA 路由切换后仍能响应
    container.addEventListener('click', function(e) {
        const target = e.target.closest('#qa-new-session');
        if (!target) return;

        e.preventDefault();
        window.qaState.sessionId = generateUUID();
        window.qaState.messages = [];
        window.qaState.pendingImages = [];
        window.qaState.jurisdictions = [];
        if (input) {
            input.value = '';
            input.style.height = 'auto';
        }
        const jurSelect = document.getElementById('qa-jurisdictions');
        if (jurSelect) {
            Array.from(jurSelect.options).forEach(function(o) { o.selected = false; });
            const tagsEl = document.getElementById('qa-jur-tags');
            if (tagsEl) tagsEl.innerHTML = '<span class="jur-placeholder">点击选择法域...</span>';
        }
        saveQAState();
        renderQAMessages();
        updateImagePreview();
        const chatEl = document.getElementById('qa-chat');
        if (chatEl) chatEl.scrollTop = 0;
    });
}

function updateDeepToggleUI() {
    const btn = document.getElementById('qa-deep-toggle');
    const label = document.getElementById('qa-deep-state');
    if (!btn || !label) return;
    const on = window.qaState.deepMode === true;
    label.textContent = on ? '已开启' : '已关闭';
    btn.classList.toggle('btn-primary', on);
    btn.classList.toggle('btn-secondary', !on);
}

function handleImageFiles(files) {
    for (let i = 0; i < files.length; i++) {
        if (window.qaState.pendingImages.length >= 4) break;
        const file = files[i];
        if (!['image/jpeg', 'image/png', 'image/webp', 'image/gif'].includes(file.type)) continue;

        const reader = new FileReader();
        reader.onload = function(e) {
            const dataUrl = e.target.result;
            const exists = window.qaState.pendingImages.some(function(img) { return img.data === dataUrl; });
            if (!exists) {
                window.qaState.pendingImages.push({ data: dataUrl, mime_type: file.type });
                updateImagePreview();
            }
        };
        reader.readAsDataURL(file);
    }
}

function updateImagePreview() {
    const previewEl = document.getElementById('qa-image-preview');
    const listEl = document.getElementById('qa-image-list');
    const countEl = document.getElementById('qa-image-count');

    if (window.qaState.pendingImages.length === 0) {
        previewEl.style.display = 'none';
        countEl.style.display = 'none';
        return;
    }

    previewEl.style.display = 'block';
    countEl.style.display = 'inline';
    countEl.textContent = window.qaState.pendingImages.length + ' 张图片';

    listEl.innerHTML = '';
    window.qaState.pendingImages.forEach(function(img, idx) {
        const div = document.createElement('div');
        div.style.cssText = 'position:relative;display:inline-block;';
        div.innerHTML = '<img src="' + img.data + '" style="width:80px;height:80px;object-fit:cover;border-radius:6px;border:1px solid var(--color-border);">' +
            '<button onclick="removeQAImage(' + idx + ')" style="position:absolute;top:-6px;right:-6px;width:20px;height:20px;border-radius:50%;background:var(--color-error);color:white;border:none;cursor:pointer;font-size:12px;line-height:1;display:flex;align-items:center;justify-content:center;">x</button>';
        listEl.appendChild(div);
    });
}

function removeQAImage(idx) {
    window.qaState.pendingImages.splice(idx, 1);
    updateImagePreview();
}

function sendQAMessage() {
    const input = document.getElementById('qa-input');
    const question = input.value.trim();
    const images = window.qaState.pendingImages;

    if (!question && images.length === 0) return;

    const finalQuestion = question || '请分析这张图片中的法律条款内容';

    const selectedJurs = Array.from(document.getElementById('qa-jurisdictions').selectedOptions).map(function(o) { return o.value; });
    window.qaState.messages.push({
        role: 'user',
        content: finalQuestion,
        images: images.map(function(img) { return img.data; }),
        jurisdictions: selectedJurs
    });

    input.value = '';
    window.qaState.pendingImages = [];
    updateImagePreview();

    renderQAMessages();

    const assistantMsg = { role: 'assistant', content: '', streaming: true, status: '正在检索相关法规…', trace: [] };
    window.qaState.messages.push(assistantMsg);
    renderQAMessages();

    const payload = {
        question: finalQuestion,
        top_k: window.qaState.topK
    };
    if (selectedJurs.length > 0) payload.jurisdictions = selectedJurs;
    if (window.qaState.messages.length > 2) {
        payload.history = window.qaState.messages.slice(0, -2).map(function(m) {
            return { role: m.role, content: m.content, jurisdictions: m.jurisdictions };
        });
    }
    if (images.length > 0) {
        payload.images = images.map(function(img) {
            const parts = img.data.split(',');
            return { data: parts[1], mime_type: img.mime_type };
        });
    }

    // 流式请求统一走 POST 把参数放入请求体，避免多轮历史拼进 URL 触发 414
    const isMultimodal = images.length > 0;
    // 深度模式走 Agent 端点（mode=auto 由后端复杂度路由决定是否多步），图片问答保持原链路
    const useAgent = window.qaState.deepMode === true && !isMultimodal;
    const streamPath = useAgent ? '/agent/stream' : '/qa/stream';
    if (useAgent) payload.mode = 'auto';

    api.streamPost(streamPath, payload,
        function(chunk) { handleQAChunk(chunk, assistantMsg); },
        function(err) { handleQAError(err, assistantMsg); }
    );
}

function handleQAChunk(chunk, assistantMsg) {
    if (chunk.type === 'content') {
        assistantMsg.content += chunk.data || '';
        assistantMsg.status = '';
        renderQAMessages();
    } else if (chunk.type === 'status') {
        assistantMsg.status = chunk.data || '';
        renderQAMessages();
    } else if (chunk.type === 'sources') {
        assistantMsg.sources = chunk.sources || [];
        assistantMsg.jurisdictions = chunk.jurisdictions || [];
    } else if (chunk.type === 'route') {
        // Agent 复杂度路由结果：fast=走快速通道，agent=多步探索。用于「推理过程」展示。
        assistantMsg.route = { mode: chunk.mode, reason: chunk.reason };
        if (chunk.mode === 'agent' && !assistantMsg.content) {
            assistantMsg.status = '正在规划检索步骤…';
        }
        renderQAMessages();
    } else if (chunk.type === 'rewrite') {
        // 追问指代消解：把依赖上文的追问改写为自足检索式（仅作检索种子）
        assistantMsg.rewrite = { original: chunk.original || '', rewritten: chunk.rewritten || '' };
        renderQAMessages();
    } else if (chunk.type === 'plan') {
        // 显式规划：探索前产出的检索计划（子问题/目标法域/重点方面）
        assistantMsg.plan = {
            sub_questions: chunk.sub_questions || [],
            jurisdictions: chunk.jurisdictions || [],
            aspects: chunk.aspects || []
        };
        renderQAMessages();
    } else if (chunk.type === 'step') {
        // 工具调用开始：先占位，结果由 observation 事件补齐摘要
        assistantMsg.trace = assistantMsg.trace || [];
        assistantMsg.trace.push({ step: chunk.step, tool: chunk.tool, args: chunk.args, summary: '' });
        renderQAMessages();
    } else if (chunk.type === 'observation') {
        // 工具执行结果摘要：按 step+tool 回填到最后一条未完成的轨迹上
        const tr = (assistantMsg.trace || []).slice().reverse()
            .find(function(t) { return t.step === chunk.step && t.tool === chunk.tool && !t.summary; });
        if (tr) tr.summary = chunk.summary || '';
        renderQAMessages();
    } else if (chunk.type === 'end') {
        // end.trace 为权威轨迹（含每步摘要），覆盖增量累积的近似结果，保证展示与后端一致
        if (Array.isArray(chunk.trace) && chunk.trace.length > 0) {
            assistantMsg.trace = chunk.trace;
        }
        // E：探索收口原因；A：引用溯源校验结果。均由后端在 end 事件一并下发
        assistantMsg.stopReason = chunk.stop_reason || '';
        assistantMsg.citationCheck = chunk.citation_check || null;
        assistantMsg.streaming = false;
        assistantMsg.status = '';
        saveQAState();
        renderQAMessages();
    } else if (chunk.type === 'error') {
        assistantMsg.content = '问答失败：' + (chunk.data || '');
        assistantMsg.streaming = false;
        assistantMsg.status = '';
        saveQAState();
        renderQAMessages();
    }
}

function handleQAError(err, assistantMsg) {
    assistantMsg.content = '请求失败：' + err;
    assistantMsg.streaming = false;
    saveQAState();
    renderQAMessages();
}

function qaToolLabel(tool) {
    const map = { search_laws: '检索条文', follow_references: '追溯引用', query_graph: '图谱多跳', get_evidence_status: '核对证据覆盖' };
    return map[tool] || tool || '工具';
}

// E：探索收口原因的中文映射（与后端 stop_reason 取值一一对应）
function stopReasonLabel(stop) {
    const map = {
        model_stop: '证据自评通过，主动完成',
        step_budget: '步数预算耗尽后收口',
        tool_budget: '工具调用预算耗尽后收口',
        timeout: '达到时长上限，用已收集证据作答',
        no_progress: '连续检索无新增证据，提前收口',
        llm_error: '探索中断，用已收集证据作答',
        no_evidence_degraded: '探索无有效证据，已回退普通问答'
    };
    return map[stop] || stop || '';
}

function renderQaTrace(msg) {
    // trace 的 t.step 是模型决策轮次：同轮并行工具共享 step，按 step 分组并标注「并行 K 项」
    const trace = msg.trace || [];
    const isAgent = msg.route && msg.route.mode === 'agent';
    const plan = msg.plan || null;
    const rewrite = msg.rewrite || null;
    const planItems = plan ? (plan.sub_questions || []).concat(plan.jurisdictions || [], plan.aspects || []) : [];
    const hasPlan = planItems.length > 0;
    const hasRewrite = !!(rewrite && rewrite.rewritten && rewrite.rewritten !== rewrite.original);
    if (!isAgent && trace.length === 0 && !hasPlan && !hasRewrite) return '';

    const stepOrder = [];
    const stepCount = {};
    trace.forEach(function(t) {
        const key = (t.step === undefined || t.step === null) ? '?' : t.step;
        if (stepCount[key] === undefined) { stepCount[key] = 0; stepOrder.push(key); }
        stepCount[key] += 1;
    });

    let html = '<details style="margin-bottom:0.75rem;background:var(--color-bg);border:1px dashed var(--color-border);border-radius:6px;"' +
        (msg.streaming ? ' open' : '') + '>';
    const nSteps = stepOrder.length;
    html += '<summary style="cursor:pointer;font-size:0.8125rem;color:var(--color-text-secondary);user-select:none;padding:0.5rem 0.75rem;font-weight:600;">' +
        '推理过程' + (nSteps > 0 ? '（' + nSteps + ' 步 / ' + trace.length + ' 次调用）' : '') + '</summary>';
    html += '<div style="padding:0 0.75rem 0.625rem;">';
    if (hasRewrite) {
        html += '<div style="padding:0.5rem 0;border-bottom:1px dashed var(--color-border);">';
        html += '<div style="font-size:0.75rem;font-weight:600;color:var(--color-primary);margin-bottom:0.25rem;">追问指代消解</div>';
        html += '<div style="font-size:0.75rem;color:var(--color-text-secondary);line-height:1.6;">' +
            escapeHtml(rewrite.original) + ' <span style="color:var(--color-text-muted);">→</span> ' +
            escapeHtml(rewrite.rewritten) + '</div>';
        html += '</div>';
    }
    if (hasPlan) {
        html += '<div style="padding:0.5rem 0 0.5rem;border-bottom:1px dashed var(--color-border);margin-bottom:0.25rem;">';
        html += '<div style="font-size:0.75rem;font-weight:600;color:var(--color-primary);margin-bottom:0.25rem;">检索计划</div>';
        if ((plan.sub_questions || []).length) {
            html += '<div style="font-size:0.75rem;color:var(--color-text-secondary);line-height:1.6;">子问题：' + escapeHtml(plan.sub_questions.join('；')) + '</div>';
        }
        if ((plan.jurisdictions || []).length) {
            html += '<div style="font-size:0.75rem;color:var(--color-text-secondary);line-height:1.6;">目标法域：' + escapeHtml(plan.jurisdictions.join('、')) + '</div>';
        }
        if ((plan.aspects || []).length) {
            html += '<div style="font-size:0.75rem;color:var(--color-text-secondary);line-height:1.6;">重点方面：' + escapeHtml(plan.aspects.join('、')) + '</div>';
        }
        html += '</div>';
    }
    if (trace.length === 0) {
        html += '<div style="font-size:0.75rem;color:var(--color-text-muted);">正在规划检索步骤…</div>';
    } else {
        let seq = 0;
        let lastKey = null;
        trace.forEach(function(t) {
            const key = (t.step === undefined || t.step === null) ? '?' : t.step;
            const isFirstOfStep = (key !== lastKey);
            if (isFirstOfStep) { seq += 1; lastKey = key; }
            const k = stepCount[key];
            html += '<div style="display:flex;gap:0.5rem;padding:0.3125rem 0;border-top:1px solid var(--color-border);">';
            html += '<span style="flex-shrink:0;font-size:0.75rem;font-weight:600;color:var(--color-primary);min-width:5.5rem;">' +
                (isFirstOfStep ? '第 ' + seq + ' 步' + (k > 1 ? '（并行 ' + k + ' 项）' : '') : '') + '</span>';
            html += '<span style="flex-shrink:0;font-size:0.75rem;color:var(--color-text);font-weight:600;">' + escapeHtml(qaToolLabel(t.tool)) + '</span>';
            html += '<span style="font-size:0.75rem;color:var(--color-text-secondary);line-height:1.5;word-break:break-word;">' + escapeHtml(t.summary || '执行中…') + '</span>';
            html += '</div>';
        });
    }
    // A + E：面板底部展示收口原因与引用校验（仅在 end 事件已到、即流结束后出现）
    const stopLabel = msg.stopReason ? stopReasonLabel(msg.stopReason) : '';
    const cc = msg.citationCheck;
    if (stopLabel || cc) {
        html += '<div style="margin-top:0.5rem;padding-top:0.5rem;border-top:1px dashed var(--color-border);display:flex;flex-wrap:wrap;gap:0.5rem;font-size:0.75rem;">';
        if (stopLabel) {
            html += '<span style="color:var(--color-text-muted);">收口原因：<span style="color:var(--color-text-secondary);">' + escapeHtml(stopLabel) + '</span></span>';
        }
        if (cc) {
            const bad = cc.invalid_count || 0;
            html += '<span style="color:' + (bad > 0 ? 'var(--color-error)' : 'var(--color-text-muted)') + ';">引用校验：' +
                '共 ' + (cc.ref_count || 0) + ' 处引用' +
                (bad > 0 ? '，<b>' + bad + ' 处越界（' + escapeHtml((cc.invalid_refs || []).join(', ')) + '）</b>，越界来源无对应条文，请留意' : '，全部有对应来源') +
                '</span>';
        }
        html += '</div>';
    }
    html += '</div></details>';
    return html;
}

function renderQAMessages() {
    const chatEl = document.getElementById('qa-chat');
    if (!chatEl) return;

    const messages = window.qaState.messages;
    if (messages.length === 0) {
        chatEl.innerHTML = '<div class="loading">开始对话吧，输入您的问题或上传法规图片</div>';
        return;
    }

    let html = '';
    messages.forEach(function(msg, idx) {
        if (msg.role === 'user') {
            html += '<div style="margin-bottom:1.25rem;display:flex;justify-content:flex-end;">';
            html += '<div style="max-width:85%;display:flex;flex-direction:column;align-items:flex-end;gap:0.375rem;">';
            html += '<div style="background:var(--color-primary);color:white;padding:0.75rem 1rem;border-radius:12px 12px 4px 12px;font-size:0.9375rem;line-height:1.6;word-break:break-word;">' + escapeHtml(msg.content) + '</div>';
            if (msg.images && msg.images.length > 0) {
                html += '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;justify-content:flex-end;">';
                msg.images.forEach(function(imgData) {
                    html += '<img src="' + imgData + '" style="width:120px;height:120px;object-fit:cover;border-radius:8px;border:1px solid var(--color-border);">';
                });
                html += '</div>';
            }
            html += '</div></div>';
        } else {
            html += '<div style="margin-bottom:1.5rem;display:flex;align-items:flex-start;gap:0.75rem;">';
            html += '<div style="width:32px;height:32px;border-radius:50%;background:var(--color-info);color:white;display:flex;align-items:center;justify-content:center;font-size:0.875rem;font-weight:600;flex-shrink:0;">AI</div>';
            html += '<div style="flex:1;min-width:0;background:var(--color-surface);border:1px solid var(--color-border);border-radius:8px 8px 8px 4px;padding:1rem;box-shadow:0 1px 2px rgba(0,0,0,0.03);">';
            html += renderQaTrace(msg);
            if (msg.streaming && !msg.content && msg.status) {
                // 检索/生成阶段尚无正文，先展示阶段提示，避免十几秒空窗
                html += '<div class="qa-status"><span class="qa-status-spinner"></span>' + escapeHtml(msg.status) + '</div>';
            } else if (msg.streaming) {
                html += '<div class="markdown-body qa-answer-body">' + renderMarkdown(msg.content) + '<span class="qa-cursor">▌</span></div>';
            } else {
                html += '<div class="markdown-body qa-answer-body">' + renderMarkdown(msg.content) + '</div>';
            }
            if (msg.sources && msg.sources.length > 0) {
                html += '<details style="margin-top:0.75rem;"><summary style="cursor:pointer;font-size:0.8125rem;color:var(--color-text-muted);user-select:none;">引用法规来源 (' + msg.sources.length + ' 条)</summary>';
                html += '<div style="margin-top:0.5rem;display:grid;gap:0.5rem;">';
                msg.sources.forEach(function(s, i) {
                    html += '<div style="background:var(--color-bg);border:1px solid var(--color-border);padding:0.625rem 0.75rem;border-radius:6px;">';
                    html += '<div style="font-size:0.8125rem;font-weight:600;color:var(--color-text);margin-bottom:0.125rem;">来源 ' + (i + 1) + ' · ' + escapeHtml(s.jurisdiction || '') + '《' + escapeHtml(s.title || '') + '》' + escapeHtml(s.article_number || '') + '</div>';
                    html += '<div style="font-size:0.75rem;color:var(--color-text-secondary);line-height:1.5;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">' + escapeHtml((s.content || '').substring(0, 200)) + '</div>';
                    html += '</div>';
                });
                html += '</div></details>';
            }
            html += '</div></div>';
        }
    });

    chatEl.innerHTML = html;
    chatEl.scrollTop = chatEl.scrollHeight;
}

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

const style = document.createElement('style');
style.textContent = '@keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }';
document.head.appendChild(style);
