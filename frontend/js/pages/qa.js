/**
 * 法规智能问答页面
 */
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
                        jurisdictions: m.jurisdictions
                    };
                }),
                pendingImages: [],
                jurisdictions: parsed.jurisdictions || [],
                topK: parsed.topK || 5
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
        topK: 5
    };
}

function saveQAState() {
    try {
        const toSave = {
            sessionId: window.qaState.sessionId,
            messages: window.qaState.messages.map(function(m) {
                return { role: m.role, content: m.content, sources: m.sources, jurisdictions: m.jurisdictions };
            }),
            jurisdictions: window.qaState.jurisdictions,
            topK: window.qaState.topK
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

    // 加载法域选项
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

    // 绑定事件
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

    // 初始化时根据已保存的选择渲染标签
    renderJurTags();
    updateDropdownSelection();
}

function setupQAEvents(container) {
    const input = document.getElementById('qa-input');
    const sendBtn = document.getElementById('qa-send');
    const fileInput = document.getElementById('qa-file-input');
    const topkSlider = document.getElementById('qa-topk');
    const topkVal = document.getElementById('qa-topk-val');

    // 发送按钮
    sendBtn.addEventListener('click', sendQAMessage);

    // Enter 发送，Shift+Enter 换行
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQAMessage();
        }
    });

    // 文件选择
    fileInput.addEventListener('change', function(e) {
        handleImageFiles(e.target.files);
        fileInput.value = '';
    });

    // 粘贴图片
    input.addEventListener('paste', function(e) {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const file = items[i].getAsFile();
                if (file) handleImageFiles([file]);
            }
        }
    });

    // TopK 滑块
    topkSlider.addEventListener('input', function() {
        topkVal.textContent = this.value;
        window.qaState.topK = parseInt(this.value);
        saveQAState();
    });

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
        // 清空法域选择
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

    // 添加用户消息
    const selectedJurs = Array.from(document.getElementById('qa-jurisdictions').selectedOptions).map(function(o) { return o.value; });
    window.qaState.messages.push({
        role: 'user',
        content: finalQuestion,
        images: images.map(function(img) { return img.data; }),
        jurisdictions: selectedJurs
    });

    // 清空输入
    input.value = '';
    window.qaState.pendingImages = [];
    updateImagePreview();

    renderQAMessages();

    // 添加助手占位消息
    const assistantMsg = { role: 'assistant', content: '', streaming: true };
    window.qaState.messages.push(assistantMsg);
    renderQAMessages();

    // 构建请求
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

    // 流式请求
    const chatEl = document.getElementById('qa-chat');
    const isMultimodal = images.length > 0;
    const url = isMultimodal ? '/qa/stream' : '/qa/stream';
    const method = isMultimodal ? 'streamPost' : 'streamGet';

    if (isMultimodal) {
        api.streamPost(url, payload,
            function(chunk) { handleQAChunk(chunk, assistantMsg); },
            function(err) { handleQAError(err, assistantMsg); }
        );
    } else {
        // GET 流式请求
        const params = new URLSearchParams();
        params.set('question', finalQuestion);
        params.set('top_k', window.qaState.topK);
        if (selectedJurs.length > 0) params.set('jurisdictions', selectedJurs.join(','));
        if (payload.history) params.set('history', JSON.stringify(payload.history));

        const streamUrl = new URL(API_BASE + '/qa/stream?' + params.toString(), window.location.origin);
        fetch(streamUrl.toString())
            .then(function(resp) {
                if (!resp.ok) throw new Error('API错误: ' + resp.status);
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                function read() {
                    reader.read().then(function(result) {
                        if (result.done) {
                            assistantMsg.streaming = false;
                            renderQAMessages();
                            return;
                        }
                        buffer += decoder.decode(result.value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();
                        lines.forEach(function(line) {
                            const trimmed = line.trim();
                            if (trimmed.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(trimmed.slice(6));
                                    handleQAChunk(data, assistantMsg);
                                } catch (e) {}
                            }
                        });
                        read();
                    }).catch(function(err) {
                        handleQAError(err.message, assistantMsg);
                    });
                }
                read();
            })
            .catch(function(err) {
                handleQAError(err.message, assistantMsg);
            });
    }
}

function handleQAChunk(chunk, assistantMsg) {
    if (chunk.type === 'content') {
        assistantMsg.content += chunk.data || '';
        renderQAMessages();
    } else if (chunk.type === 'sources') {
        assistantMsg.sources = chunk.sources || [];
        assistantMsg.jurisdictions = chunk.jurisdictions || [];
    } else if (chunk.type === 'end') {
        assistantMsg.streaming = false;
        saveQAState();
        renderQAMessages();
    } else if (chunk.type === 'error') {
        assistantMsg.content = '问答失败：' + (chunk.data || '');
        assistantMsg.streaming = false;
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
            if (msg.streaming) {
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

// 添加闪烁动画
const style = document.createElement('style');
style.textContent = '@keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }';
document.head.appendChild(style);
